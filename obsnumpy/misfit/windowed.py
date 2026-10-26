"""The windowed, weighted, data-normalised L2 misfit of Sawade et al. (2022), eq. (5).

Per trace (station, component) and per group k of windows (e.g. one per wave
type), with windows w, window tapers tap_kw, a weight and the data d:

    C   = sum_k sum_traces  weight_k * 1/2 sum_w int tap_kw (s - d)^2 dt  /  N_k
    N_k = sum_w int tap_kw d^2 dt                    (not divided where N_k = 0)

Everything except the synthetic s is fixed by the data. So it collects, once,
into one effective weight trace per (station, component):

    omega(t) = sum_k weight_k * sum_w tap_kw(t) / N_k
    C(s)     = 1/2 sum int omega (s - d)^2 dt
    dC/ds    = omega (s - d) dt       (the residual weight: the adjoint source)

Overlapping windows add, and groups that share one processing collapse into
one omega, so an evaluation costs one weighted sum whatever the number of
windows and groups.
"""
from __future__ import annotations

import dataclasses

import numpy as np
from scipy.signal import windows as _windows

from ..backend import xp_of
from .base import Misfit


def windows_to_tapers(windows, npts, alpha=0.25, kind="tukey"):
    """Per trace, the sum over its windows of each window's taper, placed on
    the window: an array ``(nsta, ncomp, npts)``.

    ``windows[i][c]`` is a list of ``(left, right)`` sample pairs, ``right``
    exclusive, for station ``i`` and component ``c``; an empty list means no
    window. Each window's taper is ``scipy.signal.windows.tukey(right - left,
    alpha)`` (obswinlib's ``add_tapers``), or ones for ``kind="boxcar"``.
    """
    if kind not in ("tukey", "boxcar"):
        raise ValueError(f"unknown taper kind {kind!r}")
    nsta = len(windows)
    ncomp = max((len(w) for w in windows), default=0)
    out = np.zeros((nsta, ncomp, npts))
    for i, per_sta in enumerate(windows):
        for c, wins in enumerate(per_sta):
            for left, right in wins:
                left, right = int(left), int(right)
                if not 0 <= left < right <= npts:
                    raise ValueError(f"window ({left}, {right}) of trace ({i}, {c}) is not "
                                     f"inside [0, {npts})")
                n = right - left
                out[i, c, left:right] += _windows.tukey(n, alpha) if kind == "tukey" else 1.0
    return out


@dataclasses.dataclass
class WindowedMisfit(Misfit):
    """``C(s) = 1/2 sum int omega (s - d)^2 dt``.

    ``d`` and ``omega`` are float64 numpy arrays ``(nsta, ncomp, npts)``; build
    ``omega`` from windows with :meth:`build` or :meth:`from_groups`.
    """

    d: np.ndarray
    omega: np.ndarray
    dt: float

    @classmethod
    def build(cls, d, dt, tapers, weights=1.0, normalize=True):
        """One group of windows: ``tapers`` is the per-trace sum of window tapers
        (:func:`windows_to_tapers`), ``weights`` a scalar or one weight per trace
        ``(nsta, ncomp)``. ``normalize=False`` leaves out ``N``."""
        return cls.from_groups(d, dt, [(tapers, weights)], normalize)

    @classmethod
    def from_groups(cls, d, dt, groups, normalize=True):
        """Several groups of windows (e.g. one per wave type), ``[(tapers,
        weights), ...]``, each normalised by its own ``N`` and summed into one
        ``omega``."""
        d = np.ascontiguousarray(d, dtype=np.float64)
        omega = np.zeros_like(d)
        for tapers, weights in groups:
            tapers = np.asarray(tapers, dtype=np.float64)
            scale = np.broadcast_to(np.asarray(weights, dtype=np.float64), d.shape[:-1])[..., None]
            if normalize:
                N = np.sum(tapers * d * d, axis=-1, keepdims=True) * dt
                scale = scale / np.where(N != 0.0, N, 1.0)
            omega += scale * tapers
        return cls(d=d, omega=omega, dt=float(dt))

    @property
    def mask(self) -> np.ndarray:
        """Where ``omega > 0``: the samples any window covers."""
        return self.omega > 0

    def take(self, cols) -> "WindowedMisfit":
        return WindowedMisfit(d=self.d[..., cols], omega=self.omega[..., cols], dt=self.dt)

    def trace_cost(self, s):
        """Per trace, ``1/2 int omega (s - d)^2 dt``: ``(..., nsta, ncomp)``."""
        xp = xp_of(s)
        d, omega = self._on(xp, "d"), self._on(xp, "omega")
        if xp is np:
            r = s - d
            sq = omega * r * r
        else:
            sq = _kernels(xp)["sq"](s, d, omega)
        return 0.5 * self.dt * sq.sum(axis=-1)

    def _residual_weight(self, s):
        """``omega (s - d) dt``."""
        xp = xp_of(s)
        d, omega = self._on(xp, "d"), self._on(xp, "omega")
        if xp is np:
            return omega * (s - d) * self.dt
        return _kernels(xp)["rw"](s, d, omega, xp.float64(self.dt))

    def gauss_newton(self, J, trace_weights=None):
        """``J^T omega J dt``, ``(..., nparam, nparam)``: the Gauss-Newton Hessian
        of ``C`` for ``J`` ``(..., nparam, nsta, ncomp, npts)``, with ``omega``
        times ``trace_weights`` ``(..., nsta, ncomp)`` if given."""
        xp = xp_of(J)
        omega = self._on(xp, "omega")
        Jf = J.reshape(J.shape[:-3] + (-1,))
        if trace_weights is None:
            wf = omega.reshape(-1)
        else:
            tw = xp.asarray(trace_weights, dtype=np.float64)
            wf = (omega * tw[..., None]).reshape(tw.shape[:-2] + (1, -1))
        return (Jf * wf) @ xp.swapaxes(Jf, -1, -2) * self.dt


_KERNELS = {}


def _kernels(cp):
    """One-pass CuPy kernels for ``omega (s - d)^2`` and ``omega (s - d) dt``."""
    if not _KERNELS:
        _KERNELS["sq"] = cp.ElementwiseKernel(
            "T s, T d, T w", "T c", "c = w * (s - d) * (s - d)", "obsnumpy_weighted_sq")
        _KERNELS["rw"] = cp.ElementwiseKernel(
            "T s, T d, T w, T dt", "T r", "r = w * (s - d) * dt", "obsnumpy_residual_weight")
    return _KERNELS
