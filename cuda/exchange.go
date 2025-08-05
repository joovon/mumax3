package cuda

import (
	"unsafe"

	"github.com/mumax/3/data"
)

// Add exchange field to Beff.
//
//	m: normalized magnetization
//	B: effective field in Tesla
//	Aex: exchange stiffness
//
// see exchange.cu
func AddExchange(B, m *data.Slice, Aex SymmLUT, Msat MSlice, regions *Bytes, mesh *data.Mesh) {
	c := mesh.CellSize()
	wx := float32(2 / (c[X] * c[X]))
	wy := float32(2 / (c[Y] * c[Y]))
	wz := float32(2 / (c[Z] * c[Z]))
	N := mesh.Size()
	pbc := mesh.PBC_code()
	cfg := make3DConf(N)
	k_addexchange_async(B.DevPtr(X), B.DevPtr(Y), B.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		Msat.DevPtr(0), Msat.Mul(0),
		unsafe.Pointer(Aex), regions.Ptr,
		wx, wy, wz, N[X], N[Y], N[Z], pbc, cfg)
}

// Finds the average exchange strength around each cell, as used by ExchCoupling and DindCoupling.
func ExchangeDecode(dst *data.Slice, Aex SymmLUT, regions *Bytes, mesh *data.Mesh) {
	c := mesh.CellSize()
	wx := float32(2 / (c[X] * c[X]))
	wy := float32(2 / (c[Y] * c[Y]))
	wz := float32(2 / (c[Z] * c[Z]))
	N := mesh.Size()
	pbc := mesh.PBC_code()
	cfg := make3DConf(N)
	k_exchangedecode_async(dst.DevPtr(0), unsafe.Pointer(Aex), regions.Ptr, wx, wy, wz, N[X], N[Y], N[Z], pbc, cfg)
}
