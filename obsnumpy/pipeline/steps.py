"""Processing steps, each applied to a whole array at once.

A step acts on the last axis (time) or on axis -2 (components, with stations
on axis -3). All other leading axes are free: stations, components, partial
derivatives, chains, or the traces of a Green-function basis.

Each step reproduces one obspy operation (obsproclib's where obspy has none),
and ``tests/test_pipeline.py`` checks it against that operation. A step:

- returns its result on the input's backend (numpy in, numpy out; CuPy in,
  CuPy out) and never moves data between devices;
- computes in the input's floating-point precision (a :class:`Pipeline` converts an
  integer input to float64 first);
- states whether it is linear, ``y(a x + b z) = a y(x) + b y(z)``;
- if linear, has an adjoint, ``<step(x), g> = <x, step.adjoint(g, dt, npts)>``.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from . import tables
from .backend import cached_to_xp, signal_of, xp_of


class Step:
    """The step contract.

    ``linear``
        whether ``y(a x + b z) = a y(x) + b y(z)``.
    ``acts_on``
        ``"time"`` (the last axis) or ``"component"`` (axis -2, stations on -3).
    """

    linear: bool = True
    acts_on: str = "time"

    def is_linear(self, dt: float, npts: int) -> bool:
        """Whether the step is linear for an input of ``npts`` samples at ``dt``.
        For most steps that is ``linear``; for :class:`PreFilt` it depends on
        the input."""
        return self.linear

    def adjoint(self, g, dt: float, npts: int):
        """The adjoint: maps a cotangent ``g`` on the output back onto the input
        of ``npts`` samples at ``dt``, so that ``<step(x), g> = <x, adjoint(g)>``."""
        raise NotImplementedError(f"{type(self).__name__} has no adjoint")

    def out_dt(self, dt: float) -> float:
        """The output's sampling interval for an input sampled at ``dt``."""
        return dt

    def out_npts(self, npts: int) -> int:
        """The output's number of samples for an input of ``npts``."""
        return npts

    def out_start(self, dt: float) -> float:
        """Seconds from the input's first sample to the output's first."""
        return 0.0

    def out_components(self, components: list) -> list:
        """The output's component names for the input's ``components``."""
        return components

    def __call__(self, x, dt: float):
        raise NotImplementedError


def _real_dtype(x):
    return x.dtype if x.dtype.kind == "f" else np.float64


@dataclasses.dataclass
class Detrend(Step):
    """``Trace.detrend("linear" | "demean")``: remove the least-squares line
    (or the mean) along the last axis."""

    type: str = "linear"

    def __post_init__(self):
        if self.type not in ("linear", "demean"):
            raise ValueError(f"Detrend type must be 'linear' or 'demean', not {self.type!r}")

    def __call__(self, x, dt):
        xp = xp_of(x)
        x = x - x.mean(axis=-1, keepdims=True)
        if self.type == "linear":
            n = x.shape[-1]
            t = xp.arange(n, dtype=x.dtype) - (n - 1) / 2.0
            x = x - ((x @ t) / (t @ t))[..., None] * t
        return x

    def adjoint(self, g, dt, npts):
        """Itself: removing a least-squares fit is a symmetric projection."""
        return self(g, dt)


@dataclasses.dataclass
class Taper(Step):
    """``Trace.taper(max_percentage, type, max_length, side)``, with obspy's
    window: for ``type="cosine"`` the p = 1 convention, in which the ramp
    reaches 1 at sample ``wlen - 1``."""

    max_percentage: float | None = 0.05
    type: str = "cosine"
    max_length: float | None = None
    side: str = "both"

    def __call__(self, x, dt):
        n = x.shape[-1]
        sr = None if self.max_length is None else 1.0 / dt
        key = ("taper", n, self.max_percentage, self.max_length, sr, self.type, self.side)
        w = cached_to_xp(xp_of(x), key,
                         lambda: tables.taper_window(n, self.max_percentage, self.max_length,
                                                     sr, self.type, self.side),
                         dtype=_real_dtype(x))
        return x * w

    def adjoint(self, g, dt, npts):
        """Itself: a multiplication."""
        return self(g, dt)


class _Butterworth(Step):
    """Butterworth sections, as obspy designs and applies them. Subclasses give
    ``btype``, ``freqs``, ``corners`` and ``zerophase``."""

    def _sos(self, dt, xp, dtype):
        df = 1.0 / dt
        key = ("sos", self.btype, self.freqs, df, self.corners)
        return cached_to_xp(xp, key,
                            lambda: tables.butterworth_sos(self.btype, self.freqs, df,
                                                           self.corners),
                            dtype=dtype)

    def __call__(self, x, dt):
        xp = xp_of(x)
        sos = self._sos(dt, xp, _real_dtype(x))
        sosfilt = signal_of(xp).sosfilt
        y = sosfilt(sos, x, axis=-1)
        if self.zerophase:
            y = sosfilt(sos, y[..., ::-1], axis=-1)[..., ::-1]
        return y

    def adjoint(self, g, dt, npts):
        """A causal filter F (zero initial state, a finite record) is a lower
        triangular Toeplitz matrix, so its transpose is R F R, with R the time
        reversal. The zero-phase filter R F R F is therefore symmetric: its
        adjoint is itself. The causal one's is the filter run backwards."""
        if self.zerophase:
            return self(g, dt)
        xp = xp_of(g)
        sos = self._sos(dt, xp, _real_dtype(g))
        return signal_of(xp).sosfilt(sos, g[..., ::-1], axis=-1)[..., ::-1]


@dataclasses.dataclass
class Bandpass(_Butterworth):
    """``Trace.filter("bandpass", freqmin=, freqmax=, corners=, zerophase=)``:
    Butterworth sections, run forwards, and with ``zerophase`` once more over
    the reversed result. A ``freqmax`` at or above Nyquist gives obspy's
    highpass, with its warning."""

    freqmin: float
    freqmax: float
    corners: int = 4
    zerophase: bool = False
    btype = "bandpass"

    @property
    def freqs(self):
        return (float(self.freqmin), float(self.freqmax))


@dataclasses.dataclass
class Highpass(_Butterworth):
    """``Trace.filter("highpass", freq=, corners=, zerophase=)``."""

    freq: float
    corners: int = 4
    zerophase: bool = False
    btype = "highpass"

    @property
    def freqs(self):
        return (float(self.freq),)


@dataclasses.dataclass
class Lowpass(_Butterworth):
    """``Trace.filter("lowpass", freq=, corners=, zerophase=)``."""

    freq: float
    corners: int = 4
    zerophase: bool = False
    btype = "lowpass"

    @property
    def freqs(self):
        return (float(self.freq),)


@dataclasses.dataclass
class PreFilt(Step):
    """The frequency-domain cosine SAC taper of ``Trace.remove_response(pre_filt=)``,
    without a response: obsproclib's ``filter_trace``.

    It runs an rfft at obspy's ``_npts2nfft(npts)`` and multiplies by
    ``cosine_sac_taper(freqs, flimit)``. The Nyquist bin is made real, and the
    irfft is truncated to ``npts``.

    ``nyquist`` is how the Nyquist bin is made real:

    ``"abs"``
        obspy's and obsproclib's ``X[-1] = abs(X[-1])``. This flips the sign
        of a negative bin, so the step is nonlinear when the taper does not
        vanish at Nyquist. It does vanish when ``flimit[3]`` is at or below
        Nyquist, the usual case, and then the step is linear.
    ``"real"``
        its real part (it is real already, for a real input), so the step is
        always linear. It differs from ``"abs"`` only where that is nonlinear.

    Where linear, the step is symmetric (zero-padding, a real even frequency
    response, truncation), and its adjoint is itself.
    """

    flimit: tuple
    nyquist: str = "abs"

    def __post_init__(self):
        self.flimit = tuple(float(f) for f in self.flimit)
        if len(self.flimit) != 4 or list(self.flimit) != sorted(self.flimit):
            raise ValueError(f"flimit must be four ascending frequencies, not {self.flimit}")
        if self.nyquist not in ("abs", "real"):
            raise ValueError(f"nyquist must be 'abs' or 'real', not {self.nyquist!r}")

    @property
    def linear(self):
        """Linear whatever the input only with ``nyquist="real"``; see :meth:`is_linear`."""
        return self.nyquist == "real"

    def is_linear(self, dt, npts):
        """Linear for ``nyquist="real"``, or where the taper vanishes at Nyquist."""
        if self.nyquist == "real":
            return True
        return tables.sac_taper(tables.npts2nfft(npts), float(dt), self.flimit)[-1] == 0.0

    def adjoint(self, g, dt, npts):
        """Itself, where linear."""
        if not self.is_linear(dt, npts):
            raise ValueError("PreFilt(nyquist='abs') is nonlinear here: the taper does not "
                             "vanish at Nyquist. Use nyquist='real' for an adjoint.")
        return self(g, dt)

    def __call__(self, x, dt):
        xp = xp_of(x)
        n = x.shape[-1]
        nfft = tables.npts2nfft(n)
        X = xp.fft.rfft(x, n=nfft, axis=-1)
        key = ("sac_taper", nfft, float(dt), self.flimit)
        X *= cached_to_xp(xp, key, lambda: tables.sac_taper(nfft, float(dt), self.flimit),
                          dtype=_real_dtype(x))
        if self.nyquist == "abs":
            X[..., -1] = xp.abs(X[..., -1])
        else:
            X[..., -1] = X[..., -1].real
        return xp.fft.irfft(X, n=nfft, axis=-1)[..., :n]


@dataclasses.dataclass
class Cut(Step):
    """Keep the samples ``[start, stop)`` of the last axis, Python slice
    semantics. To cut at absolute times, see O2's ``FixedWindow``."""

    start: int | None = None
    stop: int | None = None

    def _slice(self, npts):
        return slice(self.start, self.stop).indices(npts)[:2]

    def out_npts(self, npts):
        i0, i1 = self._slice(npts)
        return max(i1 - i0, 0)

    def out_start(self, dt):
        if self.start is not None and self.start < 0:
            raise ValueError("out_start needs a non-negative Cut.start")
        return (self.start or 0) * dt

    def __call__(self, x, dt):
        return x[..., self.start:self.stop]

    def adjoint(self, g, dt, npts):
        """Zero-fill back to ``npts`` samples."""
        xp = xp_of(g)
        out = xp.zeros(g.shape[:-1] + (npts,), dtype=g.dtype)
        out[..., self.start:self.stop] = g
        return out


@dataclasses.dataclass
class Interpolate(Step):
    """``Trace.interpolate(1/new_dt, method="lanczos", a=a, starttime=, npts=)``
    as a sparse linear operator, with the window ``"lanczos"``.

    ``offset_s`` is the output's first sample relative to the input's first, in
    seconds. It is the parameter a time shift moves. To reproduce obspy's
    output to rounding, pass obspy's own value,
    ``starttime.timestamp - trace.stats.starttime.timestamp``. That
    difference of epoch seconds is rounded to ~1e-7 s.

    Unlike obspy, this step does not refuse extrapolation: taps outside the
    record contribute zero.

    With ``deriv=1`` the step applies the operator's derivative in
    ``offset_s`` instead, ``dL/d offset_s``, which has the same sparsity.
    """

    new_dt: float
    npts: int
    offset_s: float = 0.0
    a: int = 12
    deriv: int = 0

    @classmethod
    def from_utc(cls, old_start, new_start, new_dt, npts, a=12, like_obspy=False):
        """The step from a trace starting at ``old_start`` to one starting at
        ``new_start`` (both ``UTCDateTime``).

        By default ``offset_s`` is their exact difference (UTCDateTime keeps
        nanoseconds). ``like_obspy=True`` takes obspy's
        ``new_start.timestamp - old_start.timestamp`` instead: a difference of
        two float64 epoch seconds, each rounded to ~1e-7 s today. It reproduces
        ``Trace.interpolate`` to rounding, and differs from the exact offset by
        that much.
        """
        if like_obspy:
            offset_s = new_start.timestamp - old_start.timestamp
        else:
            offset_s = (new_start.ns - old_start.ns) * 1e-9
        return cls(new_dt=new_dt, npts=npts, offset_s=offset_s, a=a)

    def out_dt(self, dt):
        return self.new_dt

    def out_npts(self, npts):
        return self.npts

    def out_start(self, dt):
        return self.offset_s

    def operator(self, n_in, dt, xp=np, dtype=np.float64, transpose=False):
        """The sparse ``(npts, n_in)`` operator ``L`` (``dL/d offset_s`` with
        ``deriv=1``), so that ``y = L @ x``; with ``transpose=True``, ``L.T``."""
        args = (int(n_in), float(dt), float(self.offset_s), float(self.new_dt),
                int(self.npts), int(self.a), int(self.deriv))
        build = tables.lanczos_matrix_t if transpose else tables.lanczos_matrix
        return cached_to_xp(xp, ("lanczos", transpose) + args, lambda: build(*args),
                            dtype=dtype)

    def __call__(self, x, dt):
        xp = xp_of(x)
        L = self.operator(x.shape[-1], dt, xp, _real_dtype(x))
        flat = x.reshape(-1, x.shape[-1])
        return (L @ flat.T).T.reshape(x.shape[:-1] + (self.npts,))

    def adjoint(self, g, dt, npts):
        """``L.T`` applied to the last axis."""
        xp = xp_of(g)
        Lt = self.operator(npts, dt, xp, _real_dtype(g), transpose=True)
        flat = g.reshape(-1, g.shape[-1])
        return (Lt @ flat.T).T.reshape(g.shape[:-1] + (npts,))


@dataclasses.dataclass
class Shift(Step):
    """A time shift by ``tau`` seconds onto a fixed output grid: Lanczos
    interpolation whose offset moves with ``tau``.

    A positive ``tau`` delays the signal: the output at time ``t`` is the
    input at ``t - tau``. ``offset_s`` is the output's first sample relative
    to the input's first at ``tau = 0``, so the interpolation's offset is
    ``offset_s - tau``. :meth:`d_dtau` is the step that gives the output's
    derivative in ``tau``.
    """

    tau: float
    new_dt: float
    npts: int
    offset_s: float = 0.0
    a: int = 12
    deriv: int = 0

    def interpolate(self) -> Interpolate:
        """The Lanczos step at this ``tau`` (in ``offset_s``)."""
        return Interpolate(self.new_dt, self.npts, self.offset_s - self.tau, self.a, self.deriv)

    def d_dtau(self) -> "Shift":
        """The derivative in ``tau``: ``-dL/d offset_s``, on the same taps."""
        return dataclasses.replace(self, deriv=1)

    @property
    def _sign(self):
        return -1.0 if self.deriv else 1.0

    def out_dt(self, dt):
        return self.new_dt

    def out_npts(self, npts):
        return self.npts

    def out_start(self, dt):
        return self.offset_s - self.tau

    def operator(self, n_in, dt, xp=np, dtype=np.float64):
        """The sparse ``(npts, n_in)`` operator ``S(tau)`` (``dS/dtau`` for ``deriv=1``)."""
        L = self.interpolate().operator(n_in, dt, xp, dtype)
        return -L if self.deriv else L

    def __call__(self, x, dt):
        y = self.interpolate()(x, dt)
        return -y if self.deriv else y

    def adjoint(self, g, dt, npts):
        y = self.interpolate().adjoint(g, dt, npts)
        return -y if self.deriv else y


@dataclasses.dataclass
class RotateNE2RT(Step):
    """``obspy.signal.rotate.rotate_ne_rt``, on the component axis -2.

    The components at indices ``n`` and ``e`` become R and T, in place of N
    and E. ``baz`` is the back azimuth in degrees: one per station (axis -3),
    or a scalar.
    """

    baz: np.ndarray
    n: int = 0
    e: int = 1
    acts_on = "component"

    def out_components(self, components):
        components = list(components)
        if components[self.n] != "N" or components[self.e] != "E":
            raise ValueError(f"RotateNE2RT needs N at {self.n} and E at {self.e}, "
                             f"not {components}")
        components[self.n], components[self.e] = "R", "T"
        return components

    def __call__(self, x, dt):
        xp = xp_of(x)
        b = xp.asarray(np.radians(np.asarray(self.baz, dtype=np.float64)), dtype=_real_dtype(x))
        if b.ndim == 1:
            b = b[:, None]
        s, c = xp.sin(b), xp.cos(b)
        n, e = x[..., self.n, :], x[..., self.e, :]
        out = x.copy()
        out[..., self.n, :] = -e * s - n * c
        out[..., self.e, :] = -e * c + n * s
        return out

    def matrices(self, nsta, ncomp=3):
        """The rotation as one ``(ncomp, ncomp)`` matrix per station,
        ``(nsta, ncomp, ncomp)``: ``out[i] = M[i] @ in[i]`` over components."""
        b = np.radians(np.broadcast_to(np.asarray(self.baz, dtype=np.float64), (nsta,)))
        M = np.broadcast_to(np.eye(ncomp), (nsta, ncomp, ncomp)).copy()
        n, e = self.n, self.e
        M[:, n, n], M[:, n, e] = -np.cos(b), -np.sin(b)
        M[:, e, n], M[:, e, e] = np.sin(b), -np.cos(b)
        return M

    def adjoint(self, g, dt, npts):
        """The inverse rotation (the rotation is orthogonal)."""
        xp = xp_of(g)
        b = xp.asarray(np.radians(np.asarray(self.baz, dtype=np.float64)), dtype=_real_dtype(g))
        if b.ndim == 1:
            b = b[:, None]
        s, c = xp.sin(b), xp.cos(b)
        r, t = g[..., self.n, :], g[..., self.e, :]
        out = g.copy()
        out[..., self.n, :] = -r * c + t * s
        out[..., self.e, :] = -r * s - t * c
        return out


def _window_sum(x, w):
    """``y[i] = sum_j w[j + K] x[i - j]`` for ``j = -K..K``, with zero outside.

    This is ``np.convolve(x, w)[K:K + n]`` along the last axis, one shifted
    multiply-add per tap.
    """
    xp = xp_of(x)
    K = (len(w) - 1) // 2
    n = x.shape[-1]
    y = xp.zeros_like(x)
    for j in range(-K, K + 1):
        wj = float(w[j + K])
        if wj == 0.0 or abs(j) >= n:
            continue
        if j >= 0:
            y[..., j:] += wj * x[..., :n - j]
        else:
            y[..., :n + j] += wj * x[..., -j:]
    return y


def _check_kernel(kernel):
    k = np.array(kernel, dtype=np.float64).ravel()
    if len(k) % 2 != 1:
        raise ValueError("the kernel must have an odd length 2K + 1 (taps -K..K)")
    k.flags.writeable = False
    return k


@dataclasses.dataclass(eq=False)
class Convolve(Step):
    """Convolution with a centred kernel ``w(-K..K)``, zero outside the record.

    ``y[i] = sum_j w(j) x[i - j]``. gf3d's Gaussian source-time-function
    conversion (``GF_STF_GAUSS``) is this step with ``gf3d.stf_kernel``'s taps.
    """

    kernel: np.ndarray

    def __post_init__(self):
        self.kernel = _check_kernel(self.kernel)

    def __call__(self, x, dt):
        return _window_sum(x, self.kernel)

    def adjoint(self, g, dt, npts):
        """The correlation: the window sum with the kernel reversed."""
        return _window_sum(g, self.kernel[::-1])


@dataclasses.dataclass(eq=False)
class StepConvolve(Step):
    """Convolution with a smoothed unit step: a running sum plus a centred kernel.

    ``y[i] = dt * (P[i - K - 1] + sum_j w(j) x[i - j])``, where ``P`` is the
    cumulative sum of ``x`` and ``P[<0] = 0``. The response to a unit impulse
    is ``dt * w`` over ``-K..K``, then ``dt`` from ``K + 1`` on. This is the
    Heaviside source time function of a Green-function database: gf3d's
    ``GF_STF_HEAVI`` conversion with ``gf3d.stf_kernel``'s taps.
    """

    kernel: np.ndarray

    def __post_init__(self):
        self.kernel = _check_kernel(self.kernel)

    def __call__(self, x, dt):
        xp = xp_of(x)
        K = (len(self.kernel) - 1) // 2
        n = x.shape[-1]
        y = _window_sum(x, self.kernel)
        if n > K + 1:
            y[..., K + 1:] += xp.cumsum(x, axis=-1)[..., :n - K - 1]
        return dt * y

    def adjoint(self, g, dt, npts):
        """``dt * (the correlation with the kernel + sum_{i >= m + K + 1} g[i])``:
        the window sum with the kernel reversed, plus a suffix sum."""
        xp = xp_of(g)
        K = (len(self.kernel) - 1) // 2
        n = g.shape[-1]
        y = _window_sum(g, self.kernel[::-1])
        if n > K + 1:
            suffix = xp.cumsum(g[..., ::-1], axis=-1)[..., ::-1]
            y[..., :n - K - 1] += suffix[..., K + 1:]
        return dt * y
