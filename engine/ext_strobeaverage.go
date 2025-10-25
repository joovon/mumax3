package engine

//

import (
	"github.com/mumax/3/cuda"
)

func init() {
	DeclFunc("StrobeAverage", StrobeAverage, "Records the time-average of a quantity at a strobe frequency from the moment this function is called.")
}

// StrobeAverage returns the running average of a quantity recorded at a strobe frequency,
// starting at the moment StrobeAverage() is called.
// User supplied strobe period, ts
func StrobeAverage(q Quantity, ts float64) Quantity {
	ra := runningAverage{q, nil, Time, 0}
	ra.avg = cuda.Buffer(q.NComp(), SizeOf(q))
	cuda.Zero(ra.avg)
	PostStep(func() {
		dt := Time - ra.prev_t
		if dt < ts { // Don't update the time average if we have not reached the strobe period
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
