package engine

// ****************************************
// Author(s): Joo-Von Kim, C2N, CNRS/Univ. Paris-Saclay
//
// This module extends the RunningAverage() routine provided by MuMax3 to facilitate
// two related use cases:
//
// 1) StrobeAverage() functions like RunningAverage(), but instead of performing a
//    time average using the state at each time step, the state at each user-defined
//    strobe period is used. This can simulate, for example, pump-probe experiments.
//
// 2) FourierCosAverage() and FourierSinAverage() function like StrobeAverage(),
//    but the state sampled at the user-supplied sampling period is weighted by
//    cos(omega t) or sin(omega t). This essentially allows one to compute a Fourier
//    coefficient of a quantity, at frequency omega, on the fly. The sampling period
//    here is arbitrary and does not need to be related to omega, but it should be
//    chosen to obtain a sufficient number of samples within a simulation run.
//
// USAGE:
//     StrobeAverage(quantity, period)
//     FourierCosAverage(quantity, frequency, period)
//     FourierSinAverage(quantity, frequency, period)
//
// Acknowledgements:
// This work was supported by Horizon Europe Research and Innovation Programme of the
// European Commission under grant agreement No. 101070290 (NIMFEIA).
//
// ****************************************

import (
	"math"

	"github.com/mumax/3/cuda"
)

func init() {
	DeclFunc("StrobeAverage", StrobeAverage, "Records the time-average of a quantity at a strobe frequency from the moment this function is called.")
	DeclFunc("FourierCosAverage", FourierCosAverage, "Records the cosine-weighted time-average of a quantity at a sampling frequency from the moment this function is called.")
	DeclFunc("FourierSinAverage", FourierSinAverage, "Records the sine-weighted time-average of a quantity at a sampling frequency from the moment this function is called.")
}

// StrobeAverage returns the running average of a quantity recorded at a strobe frequency,
// starting at the moment StrobeAverage() is called.
// User supplied strobe period, tstrobe
func StrobeAverage(q Quantity, tstrobe float64) Quantity {
	ra := runningAverage{q, nil, Time, 0}
	ra.avg = cuda.Buffer(q.NComp(), SizeOf(q))
	cuda.Zero(ra.avg)
	PostStep(func() {
		dt := Time - ra.prev_t
		if dt < tstrobe { // Don't update the time average if we have not reached the strobe period
			return
		}
		ra.prev_t = Time
		ra.total_t += dt
		val := ValueOf(q)
		defer cuda.Recycle(val)
		cuda.Madd2(ra.avg, ra.avg, val, float32((ra.total_t-dt)/ra.total_t), float32(dt/ra.total_t))
	})
	return &ra
}

// FourierCosAverage returns the running average of a quantity recorded at a sampling frequency,
// weighted by cos(omega t), starting at the moment FourierCosAverage() is called.
// User supplied Fourier frequency (in Hz), f0, and sampling period, tsample
func FourierCosAverage(q Quantity, f0 float64, tsample float64) Quantity {
	ra := runningAverage{q, nil, Time, 0}
	ra.avg = cuda.Buffer(q.NComp(), SizeOf(q))
	cuda.Zero(ra.avg)
	PostStep(func() {
		dt := Time - ra.prev_t
		if dt < tsample { // Don't update the time average if we have not reached the sampling period
			return
		}
		ra.prev_t = Time
		ra.total_t += dt
		val := ValueOf(q)
		defer cuda.Recycle(val)
		trigfactor := math.Cos(2 * math.Pi * f0 * Time)
		cuda.Madd2(ra.avg, ra.avg, val, float32((ra.total_t-dt)/ra.total_t), float32(trigfactor*dt/ra.total_t))
	})
	return &ra
}

// FourierSinAverage returns the running average of a quantity recorded at a sampling frequency,
// weighted by sin(omega t), starting at the moment FourierSinAverage() is called.
// User supplied Fourier frequency (in Hz), f0, and sampling period, tsample
func FourierSinAverage(q Quantity, f0 float64, tsample float64) Quantity {
	ra := runningAverage{q, nil, Time, 0}
	ra.avg = cuda.Buffer(q.NComp(), SizeOf(q))
	cuda.Zero(ra.avg)
	PostStep(func() {
		dt := Time - ra.prev_t
		if dt < tsample { // Don't update the time average if we have not reached the sampling period
			return
		}
		ra.prev_t = Time
		ra.total_t += dt
		val := ValueOf(q)
		defer cuda.Recycle(val)
		trigfactor := math.Sin(2 * math.Pi * f0 * Time)
		cuda.Madd2(ra.avg, ra.avg, val, float32((ra.total_t-dt)/ra.total_t), float32(trigfactor*dt/ra.total_t))
	})
	return &ra
}
