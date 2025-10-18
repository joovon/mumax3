package cuda

import (
	// "github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Stores the necessary state to perform FFT-accelerated convolution
// with magnetostatic kernel (or other kernel of same symmetry).
type StrayFieldConvolution struct {
	inputSize        [3]int            // 3D size of the input/output data
	realKernSize     [3]int            // Size of kernel and logical FFT size.
	fftKernLogicSize [3]int            // logic size FFTed kernel, real parts only, we store less
	fftRBuf          [3]*data.Slice    // FFT input buf; 2D: Z shares storage with X.
	fftCBuf          [3]*data.Slice    // FFT output buf; 2D: Z shares storage with X.
	kern             [3][3]*data.Slice // FFT kernel on device
	fwPlan           fft3DR2CPlan      // Forward FFT (1 component)
	bwPlan           fft3DC2RPlan      // Backward FFT (1 component)
}

// Initializes a convolution to evaluate the stray field for the given mesh geometry.
// Sanity-checked if test == true (slow-ish for large meshes).
func NewStrayField(inputSize, PBC [3]int, kernel [3][3]*data.Slice, test bool) *StrayFieldConvolution {
	c := new(StrayFieldConvolution)
	c.inputSize = inputSize
	c.realKernSize = kernel[X][X].Size()
	c.init(kernel)
	if test {
		testStrayFieldConvolution(c, PBC, kernel)
	}
	return c
}

// Calculate the stray field of m * vol * Bsat, store result in B.
//
//	m:    magnetization normalized to unit length
//	vol:  unitless mask used to scale m's length, may be nil
//	Bsat: saturation magnetization in Tesla
//	B:    resulting stray field field, in Tesla
func (c *StrayFieldConvolution) Exec(B, m, vol *data.Slice, Msat MSlice) {
	util.Argument(B.Size() == c.inputSize && m.Size() == c.inputSize)
	if c.is2D() {
		c.exec2D(B, m, vol, Msat)
	} else {
		c.exec3D(B, m, vol, Msat)
	}
}

func (c *StrayFieldConvolution) exec3D(outp, inp, vol *data.Slice, Msat MSlice) {
	for i := 0; i < 3; i++ { // FW FFT
		c.fwFFT(i, inp, vol, Msat)
	}

	// kern mul
	kernMulRSymm3D_async(c.fftCBuf,
		c.kern[X][X], c.kern[Y][Y], c.kern[Z][Z],
		c.kern[Y][Z], c.kern[X][Z], c.kern[X][Y],
		c.fftKernLogicSize[X], c.fftKernLogicSize[Y], c.fftKernLogicSize[Z])

	for i := 0; i < 3; i++ { // BW FFT
		c.bwFFT(i, outp)
	}
}

func (c *StrayFieldConvolution) exec2D(outp, inp, vol *data.Slice, Msat MSlice) {
	// Convolution is separated into
	// a 1D convolution for z and a 2D convolution for xy.
	// So only 2 FFT buffers are needed at the same time.
	Nx, Ny := c.fftKernLogicSize[X], c.fftKernLogicSize[Y]

	// Z
	c.fwFFT(Z, inp, vol, Msat)
	kernMulRSymm2Dz_async(c.fftCBuf[Z], c.kern[Z][Z], Nx, Ny)
	c.bwFFT(Z, outp)

	// XY
	c.fwFFT(X, inp, vol, Msat)
	c.fwFFT(Y, inp, vol, Msat)
	kernMulRSymm2Dxy_async(c.fftCBuf[X], c.fftCBuf[Y],
		c.kern[X][X], c.kern[Y][Y], c.kern[X][Y], Nx, Ny)
	c.bwFFT(X, outp)
	c.bwFFT(Y, outp)
}

func (c *StrayFieldConvolution) is2D() bool {
	return c.inputSize[Z] == 1
}

// forward FFT component i
func (c *StrayFieldConvolution) fwFFT(i int, inp, vol *data.Slice, Msat MSlice) {
	zero1_async(c.fftRBuf[i])
	in := inp.Comp(i)
	copyPadMul(c.fftRBuf[i], in, vol, c.realKernSize, c.inputSize, Msat)
	c.fwPlan.ExecAsync(c.fftRBuf[i], c.fftCBuf[i])
}

// backward FFT component i
func (c *StrayFieldConvolution) bwFFT(i int, outp *data.Slice) {
	c.bwPlan.ExecAsync(c.fftCBuf[i], c.fftRBuf[i])
	out := outp.Comp(i)
	copyUnPad(out, c.fftRBuf[i], c.inputSize, c.realKernSize)
}

func (c *StrayFieldConvolution) init(realKern [3][3]*data.Slice) {
	// init device buffers
	// 2D re-uses fftBuf[X] as fftBuf[Z], 3D needs all 3 fftBufs.
	nc := fftR2COutputSizeFloats(c.realKernSize)
	c.fftCBuf[X] = NewSlice(1, nc)
	c.fftCBuf[Y] = NewSlice(1, nc)
	if c.is2D() {
		c.fftCBuf[Z] = c.fftCBuf[X]
	} else {
		c.fftCBuf[Z] = NewSlice(1, nc)
	}

	c.fftRBuf[X] = NewSlice(1, c.realKernSize)
	c.fftRBuf[Y] = NewSlice(1, c.realKernSize)
	if c.is2D() {
		c.fftRBuf[Z] = c.fftRBuf[X]
	} else {
		c.fftRBuf[Z] = NewSlice(1, c.realKernSize)
	}

	// init FFT plans
	c.fwPlan = newFFT3DR2C(c.realKernSize[X], c.realKernSize[Y], c.realKernSize[Z])
	c.bwPlan = newFFT3DC2R(c.realKernSize[X], c.realKernSize[Y], c.realKernSize[Z])

	// init FFT kernel

	// logic size of FFT(kernel): store real parts only
	c.fftKernLogicSize = fftR2COutputSizeFloats(c.realKernSize)
	util.Assert(c.fftKernLogicSize[X]%2 == 0)
	c.fftKernLogicSize[X] /= 2

	// physical size of FFT(kernel): store only non-redundant part exploiting Y, Z mirror symmetry
	// X mirror symmetry already exploited: FFT(kernel) is purely real.
	physKSize := [3]int{c.fftKernLogicSize[X], c.fftKernLogicSize[Y]/2 + 1, c.fftKernLogicSize[Z]/2 + 1}

	output := c.fftCBuf[0]
	input := c.fftRBuf[0]
	fftKern := data.NewSlice(1, physKSize)
	kfull := data.NewSlice(1, output.Size()) // not yet exploiting symmetry
	kfulls := kfull.Scalars()
	kCSize := physKSize
	kCSize[X] *= 2                     // size of kernel after removing Y,Z redundant parts, but still complex
	kCmplx := data.NewSlice(1, kCSize) // not yet exploiting X symmetry
	kc := kCmplx.Scalars()

	for i := 0; i < 3; i++ {
		for j := i; j < 3; j++ { // upper triangular part
			if realKern[i][j] != nil { // ignore 0's
				// FW FFT
				data.Copy(input, realKern[i][j])
				c.fwPlan.ExecAsync(input, output)
				data.Copy(kfull, output)

				// extract non-redundant part (Y,Z symmetry)
				for iz := 0; iz < kCSize[Z]; iz++ {
					for iy := 0; iy < kCSize[Y]; iy++ {
						for ix := 0; ix < kCSize[X]; ix++ {
							kc[iz][iy][ix] = kfulls[iz][iy][ix]
						}
					}
				}

				// extract real parts (X symmetry)
				scaleRealParts(fftKern, kCmplx, 1/float32(c.fwPlan.InputLen()))
				c.kern[i][j] = GPUCopy(fftKern)
			}
		}
	}
}

func (c *StrayFieldConvolution) Free() {
	if c == nil {
		return
	}
	c.inputSize = [3]int{}
	c.realKernSize = [3]int{}
	for i := 0; i < 3; i++ {
		c.fftCBuf[i].Free()
		c.fftRBuf[i].Free()
		c.fftCBuf[i] = nil
		c.fftRBuf[i] = nil

		for j := 0; j < 3; j++ {
			c.kern[i][j].Free()
			c.kern[i][j] = nil
		}
		c.fwPlan.Free()
		c.bwPlan.Free()

		cudaCtx.SetCurrent()
	}
}

// Compares FFT-accelerated convolution against brute-force on sparse data.
// This is not really needed but very quickly uncovers newly introduced bugs.
// Copy of testConvolution
func testStrayFieldConvolution(c *StrayFieldConvolution, PBC [3]int, realKern [3][3]*data.Slice) {
	if PBC != [3]int{0, 0, 0} {
		// the brute-force method does not work for pbc.
		util.Log("skipping convolution self-test for PBC")
		return
	}
	util.Log("//convolution self-test...")
	inhost := data.NewSlice(3, c.inputSize)
	initConvTestInput(inhost.Vectors())
	gpu := NewSlice(3, c.inputSize)
	defer gpu.Free()
	data.Copy(gpu, inhost)

	Msat := NewSlice(1, [3]int{1, 1, 256})
	defer Msat.Free()
	Memset(Msat, 1)

	vol := data.NilSlice(1, c.inputSize)
	c.Exec(gpu, gpu, vol, ToMSlice(Msat))

	output := gpu.HostCopy()

	brute := data.NewSlice(3, c.inputSize)
	bruteConv(inhost.Vectors(), brute.Vectors(), realKern)

	a, b := output.Host(), brute.Host()
	err := float32(0)
	for c := range a {
		for i := range a[c] {
			if fabs(a[c][i]-b[c][i]) > err {
				err = fabs(a[c][i] - b[c][i])
			}
		}
	}
	if err > CONV_TOLERANCE {
		util.Fatal("convolution self-test tolerance: ", err, " FAIL")
	}
}
