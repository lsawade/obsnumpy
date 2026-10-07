"""The windowed, weighted, data-normalised L2 misfit of Sawade et al. (2022), eq. (5).

Per trace (station, component) and per group k of windows (e.g. one per wave
type), with windows w, window tapers tap_kw, a weight and the data d:

    C = sum_k sum_traces  weight_k * 1/2 sum_w int tap_kw (s - d)^2 dt  /  N_k
    N_k = sum_w int tap_kw d^2 dt                    (not divided where N_k = 0)

This is cmt3d's ``CostGradHess.cost``, summed over its wave types.
Everything except the synthetic s is fixed by the data. So it collects, once,
into one effective weight trace per (station, component):

    Omega(t) = sum_k weight_k * sum_w tap_kw(t) / N_k
    C(s)     = 1/2 sum int Omega (s - d)^2 dt
    dC/ds    = Omega (s - d) dt       (the residual weight: the adjoint source)

Overlapping windows add, and groups that share one processing collapse into
one Omega, so an evaluation costs one weighted sum whatever the number of
windows and groups.
"""
from __future__ import annotations

import dataclasses

import numpy as np
from scipy.signal import windows as _windows

from .pipeline.backend import xp_of


def windows_to_tapers(windows, npts, alpha=0.25, kind="tukey", power=1):
    """Per trace, the sum over its windows of each window's taper, placed on
    the window: an array ``(nsta, ncomp, npts)``.

    ``windows[i][c]`` is a list of ``(left, right)`` sample pairs, ``right``
    exclusive, for station ``i`` and component ``c``; an empty list means no
    window. Each window's taper is ``scipy.signal.windows.tukey(right - left,
    alpha)`` (obswinlib's ``add_tapers``), or ones for ``kind="boxcar"``.

    ``power=2`` sums the squared tapers instead. cmt3d's ``hess`` uses those;
    see :meth:`WindowedMisfit.gauss_newton`.
    """
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
                if kind == "tukey":
                    tap = _windows.tukey(right - left, alpha=alpha)
                elif kind == "boxcar":
                    tap = np.ones(right - left)
                else:
                    raise ValueError(f"unknown taper kind {kind!r}")
                out[i, c, left:right] += tap ** power
    return out


def _scale(d, dt, tapers, weights, normalize):
    """Per trace ``weight / N``, ``N = int tapers d^2 dt``; N = 0 is not
    divided (cmt3d), and neither is anything with ``normalize=False``.
    Shape ``(nsta, ncomp, 1)``."""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim == d.ndim - 1:                       # one weight per trace
        weights = weights[..., None]
    weights = np.broadcast_to(weights, d.shape[:-1] + (1,))
    if not normalize:
        return weights
    N = np.sum(tapers * d * d, axis=-1, keepdims=True) * dt
    return weights / np.where(N != 0.0, N, 1.0)


_FUSED = {}


def _fused(cp):
    """One-pass CuPy kernels for the cost's summand and the residual weight."""
    if "sq" not in _FUSED:
        _FUSED["sq"] = cp.ElementwiseKernel(
            "T s, T d, T w", "T c", "c = w * (s - d) * (s - d)", "obsnumpy_weighted_sq")
        _FUSED["rw"] = cp.ElementwiseKernel(
            "T s, T d, T w, T dt", "T r", "r = w * (s - d) * dt", "obsnumpy_residual_weight")
    return _FUSED


def weighted_cost(s, d, omega, dt):
    """``1/2 sum omega (s - d)^2 dt`` over the last three axes."""
    xp = xp_of(s)
    if xp is np:
        r = s - d
        return 0.5 * dt * (omega * r * r).sum(axis=(-3, -2, -1))
    sq = _fused(xp)["sq"](s, d, omega)          # one pass, then CuPy's reduction
    return 0.5 * dt * sq.reshape(sq.shape[:-3] + (-1,)).sum(axis=-1)


def weighted_residual(s, d, omega, dt):
    """``omega (s - d) dt``, the shape of ``s``."""
    xp = xp_of(s)
    if xp is np:
        return omega * (s - d) * dt
    return _fused(xp)["rw"](s, d, omega, xp.float64(dt))


@dataclasses.dataclass
class WindowedMisfit:
    """The misfit, with all its data-side weights collected into ``omega``.

    ``d`` and ``omega`` are float64 numpy arrays ``(nsta, ncomp, npts)``. A
    synthetic ``s`` may be numpy or CuPy, of shape ``(..., nsta, ncomp,
    npts)``: leading axes are a batch. ``d`` and ``omega`` move to a device
    once, on first use there, and the result follows ``s``.
    """

    d: np.ndarray
    omega: np.ndarray
    dt: float
    omega_cmt3d: np.ndarray | None = None

    def __post_init__(self):
        self._dev = {}

    @classmethod
    def build(cls, d, dt, tapers=None, weights=1.0, normalize=True, groups=None,
              tapers_sq=None):
        """Collect windows, weights and normalisation into ``omega``.

        ``tapers`` is the per-trace sum of window tapers (:func:`windows_to_tapers`),
        and ``weights`` a scalar or one weight per trace ``(nsta, ncomp)``.
        ``groups`` is instead a list of ``(tapers, weights)`` pairs, one per
        group of windows (e.g. wave type), each normalised by its own ``N``
        and summed into one ``omega``. ``normalize=False`` leaves out ``N``.

        ``tapers_sq`` (or a third element per group) is the sum of the
        *squared* window tapers. It is needed only for
        ``gauss_newton(..., like_cmt3d=True)``.
        """
        d = np.ascontiguousarray(d, dtype=np.float64)
        if groups is None:
            if tapers is None:
                raise ValueError("give tapers, or groups")
            groups = [(tapers, weights) + (() if tapers_sq is None else (tapers_sq,))]
        elif tapers is not None or tapers_sq is not None:
            raise ValueError("give tapers or groups, not both")
        omega = np.zeros_like(d)
        omega_sq = np.zeros_like(d)
        have_sq = True
        for g in groups:
            tap = np.asarray(g[0], dtype=np.float64)
            scale = _scale(d, dt, tap, g[1], normalize)
            omega += scale * tap
            if len(g) > 2 and g[2] is not None:
                # cmt3d's hess: tap^2 in the numerator, the same N (built from tap)
                omega_sq += scale * np.asarray(g[2], dtype=np.float64)
            else:
                have_sq = False
        return cls(d=d, omega=omega, dt=float(dt), omega_cmt3d=omega_sq if have_sq else None)

    def _on(self, xp, name):
        if xp is np:
            return getattr(self, name)
        key = (name, xp.cuda.Device().id)
        if key not in self._dev:
            self._dev[key] = xp.asarray(getattr(self, name))
        return self._dev[key]

    @property
    def mask(self) -> np.ndarray:
        """Where ``omega > 0``: the samples any window covers."""
        return self.omega > 0

    def cost(self, s):
        """``C(s)``: a float, or an array over the leading (batch) axes."""
        xp = xp_of(s)
        c = weighted_cost(s, self._on(xp, "d"), self._on(xp, "omega"), self.dt)
        return float(c) if c.ndim == 0 else c

    def residual_weight(self, s):
        """``dC/ds = omega (s - d) dt``, the shape of ``s``."""
        xp = xp_of(s)
        return weighted_residual(s, self._on(xp, "d"), self._on(xp, "omega"), self.dt)

    def gradient(self, s, J):
        """``dC/dm = <dC/ds, J_m>`` for ``J`` of shape ``(..., nparam, nsta, ncomp,
        npts)``, the processed derivatives of ``s``. Returns ``(..., nparam)``.
        This is cmt3d's ``grad``."""
        rw = self.residual_weight(s)
        return (J * rw[..., None, :, :, :]).sum(axis=(-3, -2, -1))

    def gauss_newton(self, J, like_cmt3d=False):
        """``J^T omega J dt``, ``(..., nparam, nparam)``: the Gauss-Newton
        Hessian of ``C`` for ``J`` of shape ``(..., nparam, nsta, ncomp, npts)``.

        cmt3d's ``hess`` instead weights each window by its taper *squared*
        (``(J_i tap) @ (J_j tap)``), against the taper in its cost. The two
        differ on the tapers' ramps. ``like_cmt3d=True`` reproduces it. It
        needs ``tapers_sq`` at build time.
        """
        xp = xp_of(J)
        if like_cmt3d:
            if self.omega_cmt3d is None:
                raise ValueError("like_cmt3d needs tapers_sq (windows_to_tapers(power=2)) "
                                 "at build time")
            w = self._on(xp, "omega_cmt3d")
        else:
            w = self._on(xp, "omega")
        lead = J.shape[:-4]
        npar = J.shape[-4]
        Jf = J.reshape(lead + (npar, -1))
        wf = w.reshape(-1)
        return (Jf * wf) @ xp.swapaxes(Jf, -1, -2) * self.dt
