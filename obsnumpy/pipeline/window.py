"""A processing window pinned in absolute time.

Observed data are cut at absolute times: the cut, the detrend, the taper and
the filters act on the samples of a fixed window, wherever the source is. A
synthetic that moves with a source parameter (a time shift ``tau``) must be
processed on that same window to be compared with the data. Cut and taper do
not commute with a shift, so the synthetic is first interpolated onto the
fixed grid, ``S(tau)``, and then processed, ``P``:

    s(tau) = P . S(tau) . v,      ds/dtau = P . S'(tau) . v,

and the adjoint of ``x -> s`` is ``S(tau).T . P.T``.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from .steps import Shift


def round_away(x: float) -> int:
    """obspy's ``compatibility.round_away``: the nearest integer, halves away
    from zero."""
    lo, hi = math.floor(x), math.ceil(x)
    if lo != hi and abs(x - lo) == abs(hi - x):
        return int(int(x) + int(np.sign(x)))
    return int(np.round(x))


@dataclasses.dataclass(frozen=True)
class FixedWindow:
    """``n`` samples ``dt`` apart, the first at ``first_time``.

    ``first_time`` is absolute: an ``obspy.UTCDateTime``, or seconds in the
    caller's own reference (a float). The window's arithmetic follows that
    type, so UTCDateTime windows round to nanoseconds as obspy does.
    """

    first_time: object
    dt: float
    n: int

    @classmethod
    def from_trim(cls, trace_start, dt, t_cut_start, t_cut_end, pad_samples=0, trace_npts=None):
        """The native-rate samples that obspy keeps when it cuts a trace.

        The rule is obsproclib's ``flex_cut_trace(trace, t_cut_start,
        t_cut_end, dynamic_npts=pad_samples)``: the cut times move out by
        ``pad_samples * dt``, then ``Trace.trim(nearest_sample=True)``.

        - The first sample is ``round_away((start - trace_start) / dt)``, so
          an exact half rounds outward.
        - The count is measured from that new first sample, which UTCDateTime
          has rounded to the nanosecond.

        Floor/ceil differs from this whenever a cut time lies within half a
        sample of the grid. For the cmt3d settings it keeps 826 samples where
        obspy keeps 827.

        Raises when the window would begin before the trace, or (given
        ``trace_npts``) end after it: obspy would then keep fewer samples.
        """
        sr = 1.0 / dt                         # obspy multiplies by stats.sampling_rate
        start = t_cut_start - pad_samples * dt
        end = t_cut_end + pad_samples * dt
        i0 = round_away((start - trace_start) * sr)
        if i0 < 0:
            raise ValueError(f"the window starts {-i0} samples before the trace")
        first = trace_start + i0 * dt
        n = round_away((end - first) * sr) + 1
        if trace_npts is not None and i0 + n > trace_npts:
            raise ValueError(f"the window ends {i0 + n - trace_npts} samples after the trace")
        return cls(first, float(dt), int(n))

    @property
    def last_time(self):
        return self.first_time + (self.n - 1) * self.dt

    def offset_from(self, t) -> float:
        """Seconds from ``t`` to the window's first sample."""
        return float(self.first_time - t)

    def indices_in(self, trace_start, dt=None, tol=1e-6) -> slice:
        """The window as a slice of a trace on the same grid (e.g. the observed
        data it was cut from). Raises if the trace is not on the grid to
        ``tol`` samples, or is sampled at another ``dt``."""
        dt = self.dt if dt is None else dt
        if abs(dt - self.dt) > tol * self.dt:
            raise ValueError(f"the trace is sampled at {dt}, the window at {self.dt}")
        x = self.offset_from(trace_start) / self.dt
        i0 = int(round(x))
        if abs(x - i0) > tol:
            raise ValueError(f"the trace is {x - i0:+.3g} samples off the window's grid")
        if i0 < 0:
            raise ValueError("the window starts before the trace")
        return slice(i0, i0 + self.n)

    def shift_step(self, trace_start, tau=0.0, a=12) -> Shift:
        """``S(tau)``: Lanczos from a trace whose first sample, before the
        shift, is at ``trace_start``, onto this window. The trace may be
        sampled at another ``dt`` (the step reads it from its input). A
        positive ``tau`` delays the trace, so its first sample sits at
        ``trace_start + tau``. ``.d_dtau()`` gives ``S'(tau)``."""
        return Shift(tau=float(tau), new_dt=self.dt, npts=self.n,
                     offset_s=self.offset_from(trace_start), a=a)

    def shift_operator(self, trace_start, n_in, dt_in, tau=0.0, a=12, xp=np,
                       dtype=np.float64):
        """``S(tau)`` and ``dS/dtau`` as sparse ``(n, n_in)`` matrices, for a
        trace of ``n_in`` samples at ``dt_in``."""
        step = self.shift_step(trace_start, tau, a)
        return (step.operator(n_in, dt_in, xp, dtype),
                step.d_dtau().operator(n_in, dt_in, xp, dtype))
