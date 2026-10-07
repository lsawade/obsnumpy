"""Host-side tables for the pipeline steps.

Each table is built once on the host in float64 and cached by its length and
parameters. Arrays are returned read-only, since the cache shares them. The
steps move them to the device on use (:func:`.backend.cached_to_xp`).

Each table has one definition, shared with the rest of obsnumpy or with obspy:

- the taper window is :func:`obsnumpy.process.taper.taper_window`;
- the filter sections are :mod:`obsnumpy.process.filter`'s;
- the SAC frequency taper is obspy's :func:`~obspy.signal.invsim.cosine_sac_taper`.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import scipy.sparse as ssp

from ..process import filter as _filter
from ..process import taper as _taper
from .backend import xp_of


def _frozen(a):
    a = np.ascontiguousarray(a, dtype=np.float64)
    a.flags.writeable = False
    return a


@lru_cache(maxsize=128)
def taper_window(npts: int, max_percentage, max_length=None, sampling_rate=None,
                 type: str = "cosine", side: str = "both") -> np.ndarray:
    """obspy's ``Trace.taper`` window for ``npts`` samples (p = 1 for cosine)."""
    return _frozen(_taper.taper_window(npts, max_percentage, sampling_rate, type=type,
                                       max_length=max_length, side=side))


@lru_cache(maxsize=128)
def butterworth_sos(btype: str, freqs: tuple, df: float, corners: int) -> np.ndarray:
    """Butterworth second-order sections, designed as obspy designs them."""
    if btype == "bandpass":
        sos = _filter.bandpass_sos(freqs[0], freqs[1], df, corners)
    elif btype == "highpass":
        sos = _filter.highpass_sos(freqs[0], df, corners)
    elif btype == "lowpass":
        sos = _filter.lowpass_sos(freqs[0], df, corners)
    else:
        raise ValueError(f"unknown filter type {btype!r}")
    return _frozen(sos)


def _factor_max(n: int) -> int:
    """The largest prime factor of ``n``."""
    m, p = 1, 2
    while p * p <= n:
        while n % p == 0:
            m, n = max(m, p), n // p
        p += 1
    return max(m, n)


def npts2nfft(npts: int) -> int:
    """obspy's ``obspy.signal.util._npts2nfft(npts, smart=True)``.

    Twice ``npts`` rounded up to even, nudged upwards by up to 20 samples past
    a large prime factor (>= 500) when that is longer than 5000. Failing that,
    the next power of two.
    """
    nfft = 2 * (npts + 1) if npts & 1 else 2 * npts
    if nfft > 5000 and _factor_max(nfft) >= 500:
        for i in range(1, 11):
            if _factor_max(nfft + 2 * i) < 500:
                return nfft + 2 * i
        return 1 << (nfft - 1).bit_length()
    return nfft


@lru_cache(maxsize=64)
def sac_taper(nfft: int, dt: float, flimit: tuple) -> np.ndarray:
    """obspy's ``cosine_sac_taper(freqs, flimit)`` on the frequency axis of an
    rfft of length ``nfft``: ``linspace(0, 1/(2 dt), nfft//2 + 1)``, as obspy's
    response removal and obsproclib's ``filter_trace`` build it."""
    from obspy.signal.invsim import cosine_sac_taper

    freqs = np.linspace(0, 1.0 / (2.0 * dt), nfft // 2 + 1)
    return _frozen(cosine_sac_taper(freqs, flimit=flimit))


def lanczos_positions(n_out: int, old_dt: float, offset_s: float, new_dt: float):
    """The output samples' positions in input samples, in obspy's arithmetic.

    Output sample ``j`` sits at ``offset + j * dt_factor`` input samples, where
    ``offset = offset_s / old_dt`` and ``dt_factor = new_dt / old_dt``.
    """
    return offset_s / old_dt + np.arange(n_out) * (new_dt / old_dt)


# d/du sinc(u) = sum_{k>=1} (-1)^k 2k pi^(2k) u^(2k-1) / (2k+1)!, as pi^2 u times a
# polynomial in z = (pi u)^2; used for |u| < 0.1, where the closed form cancels.
_DSINC_SERIES = [(-1) ** k * 2 * k / float(np.prod(np.arange(1, 2 * k + 2)))
                 for k in range(1, 8)]


def dsinc(u):
    """The derivative of ``sinc``: ``(cos(pi u) - sinc(u)) / u``, and its
    Taylor series near 0, where that form loses digits. numpy or CuPy."""
    xp = xp_of(u)
    u = xp.asarray(u, dtype=xp.float64)
    small = xp.abs(u) < 0.1
    z = (np.pi * u) ** 2
    acc = xp.zeros_like(u)
    for c in reversed(_DSINC_SERIES):
        acc = acc * z + c
    safe = xp.where(small, 1.0, u)
    return xp.where(small, np.pi ** 2 * u * acc, (xp.cos(np.pi * u) - xp.sinc(u)) / safe)


def lanczos_kernel(d, a: int, deriv: int = 0):
    """obspy's Lanczos kernel ``K(d) = sinc(d) * sinc(d / a)`` at offsets ``d``
    (in samples), or with ``deriv=1`` its derivative ``K'(d)``. numpy or CuPy."""
    xp = xp_of(d)
    if deriv == 0:
        return xp.sinc(d) * xp.sinc(d / a)
    if deriv == 1:
        return dsinc(d) * xp.sinc(d / a) + xp.sinc(d) * dsinc(d / a) / a
    raise ValueError("deriv must be 0 or 1")


@lru_cache(maxsize=32)
def lanczos_matrix(n_in: int, old_dt: float, offset_s: float, new_dt: float, n_out: int,
                   a: int, deriv: int = 0) -> ssp.csr_matrix:
    """Sparse ``(n_out, n_in)`` operator of obspy's ``lanczos_interpolation``
    (``window="lanczos"``), or with ``deriv=1`` its derivative in ``offset_s``.

    Output ``j`` at position ``x`` (see :func:`lanczos_positions`) takes the
    taps ``floor(x) - a + 1 .. floor(x) + a``, each with weight
    ``K(d) = sinc(d) * sinc(d / a)``, where ``d = x - tap``. Taps outside the
    record contribute zero. ``K`` and ``K'`` both vanish at ``|d| = a``, so the
    operator and its derivative are continuous as ``x`` crosses a sample; the
    second derivative is not.

    The derivative's weights are ``K'(d) / old_dt``, on the same taps.
    """
    if a < 1:
        raise ValueError("a must be at least 1.")
    x = lanczos_positions(n_out, old_dt, offset_s, new_dt)
    i0 = np.floor(x).astype(np.int64)
    taps = i0[:, None] + np.arange(-a + 1, a + 1)[None, :]
    d = x[:, None] - taps
    w = lanczos_kernel(d, a, deriv)
    if deriv == 1:
        w = w / old_dt
    ok = (taps >= 0) & (taps < n_in)
    rows = np.broadcast_to(np.arange(n_out)[:, None], taps.shape)
    L = ssp.csr_matrix((w[ok], (rows[ok], taps[ok])), shape=(n_out, n_in))
    L.sum_duplicates()
    return L


@lru_cache(maxsize=32)
def lanczos_matrix_t(*args) -> ssp.csr_matrix:
    """The transpose of :func:`lanczos_matrix`, as CSR (for adjoints)."""
    return lanczos_matrix(*args).T.tocsr()
