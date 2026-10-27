"""A forward model for synthetics that are linear in a basis, with its
adjoint gradient and its Jacobian, batched over chains.

    v_r(t) = scale_r * sum_m U[r, c, t, m] W_m     contraction (GEMV; GEMM over a batch)
    y      = S(tau) . v                            shift onto a fixed absolute window
    s      = Rot . P . y                           the data's processing, compiled
    C      = misfit.cost(s)

The model knows nothing about where the basis comes from. W can be a moment
tensor contracted with strain Green functions, a force vector, or weights on
precomputed elementary seismograms. The caller maps its parameters to W, and
maps the gradient back through its own dW/dtheta.

A batch of K chains is evaluated at once: ``W`` is ``(K, nbasis)`` and
``tau`` ``(K,)``. The contraction is one GEMM, ``S(tau_k)`` a gather of
``2a`` Lanczos taps per output sample with per-chain weights, and the
processing one GEMM over all traces.
"""
from __future__ import annotations

import dataclasses
from collections import OrderedDict

import numpy as np

from .pipeline import tables
from .backend import device_key, xp_of


# ----------------------------------------------------------------------------- the basis

@dataclasses.dataclass
class Basis:
    """A basis for one source region.

    ``U``
        ``(nrec, ncomp, nt, nbasis)``, basis index last; float32 by default.
    ``t0``
        absolute time of ``U``'s first sample, in the window's reference (a
        ``UTCDateTime`` or float seconds, as ``FixedWindow.first_time``).
    ``dt``
        ``U``'s sampling interval.
    ``scale``
        ``(nrec,)`` per-receiver factor applied after the contraction.
    """

    U: object
    t0: object
    dt: float
    scale: object

    def __post_init__(self):
        # the contraction reshapes U to (nrec * ncomp * nt, nbasis): a view
        # only when U is C-contiguous; otherwise every call would copy it
        self.U = xp_of(self.U).ascontiguousarray(self.U)
        if self.U.ndim != 4:
            raise ValueError(f"U must be (nrec, ncomp, nt, nbasis), got shape {self.U.shape}")
        if self.scale.shape != (self.U.shape[0],):
            raise ValueError(f"scale must be ({self.U.shape[0]},), got {self.scale.shape}")

    @property
    def nbytes(self) -> int:
        return int(self.U.nbytes)

    def rotated(self, rotation) -> "Basis":
        """This basis with a component step (e.g. ``RotateNE2RT``) applied to
        ``U`` once, in float64, cast back to ``U``'s dtype.

        The rotation acts per receiver on the components. It commutes with the
        contraction, the shift and the processing, so a model with this basis
        and ``rotation=None`` gives what the original basis gives with the
        rotation, without rotating on every evaluation."""
        xp = xp_of(self.U)
        Rm = xp.asarray(rotation.matrices(self.U.shape[0], self.U.shape[1]))
        U = xp.einsum("rij,rjtm->ritm", Rm, self.U.astype(xp.float64))
        return dataclasses.replace(self, U=U.astype(self.U.dtype))

    def to(self, xp) -> "Basis":
        """This basis with ``U`` and ``scale`` on ``xp``'s current device."""
        if xp_of(self.U) is xp:
            return self
        if xp is np:
            return dataclasses.replace(self, U=self.U.get(), scale=self.scale.get())
        return dataclasses.replace(self, U=xp.asarray(self.U), scale=xp.asarray(self.scale))


class BasisStore:
    """A keyed LRU of bases on one device, filled on demand.

    ``fill(key)`` builds the :class:`Basis` for ``key`` on a miss. The store
    moves it to ``xp`` (numpy, or CuPy for the current device) and keeps at
    most ``max_items`` bases and ``max_bytes`` bytes of ``U``, evicting the
    least recently used. A single basis larger than ``max_bytes`` is kept
    alone. ``stats`` counts hits, misses and evictions.
    """

    def __init__(self, fill, max_items=None, max_bytes=None, xp=np):
        self.fill = fill
        self.max_items = max_items
        self.max_bytes = max_bytes
        self.xp = xp
        self._items: OrderedDict = OrderedDict()
        self._bytes = 0
        self._stats = dict(hits=0, misses=0, evictions=0)

    def __contains__(self, key):
        return key in self._items

    def __len__(self):
        return len(self._items)

    @property
    def nbytes(self) -> int:
        return self._bytes

    @property
    def stats(self) -> dict:
        return dict(self._stats, items=len(self._items), bytes=self._bytes)

    def get(self, key) -> Basis:
        if key in self._items:
            self._items.move_to_end(key)
            self._stats["hits"] += 1
            return self._items[key]
        self._stats["misses"] += 1
        basis = self.fill(key).to(self.xp)
        self._make_room(basis.nbytes)
        self._items[key] = basis
        self._bytes += basis.nbytes
        return basis

    def _make_room(self, incoming):
        def over():
            n, b = len(self._items) + 1, self._bytes + incoming
            return ((self.max_items is not None and n > self.max_items)
                    or (self.max_bytes is not None and b > self.max_bytes))
        while self._items and over():
            _, old = self._items.popitem(last=False)
            self._bytes -= old.nbytes
            self._stats["evictions"] += 1

    def clear(self):
        self._items.clear()
        self._bytes = 0


# ----------------------------------------------------------------------------- the model

class LinearBasisModel:
    """``s = Rot . P . S(tau) . (scale * U W)`` and its misfit.

    ``window``
        the :class:`~obsnumpy.pipeline.FixedWindow` the data were processed
        on. ``S(tau)`` interpolates the contracted basis onto it.
    ``processing``
        a :class:`~obsnumpy.pipeline.Pipeline` of linear time steps on that
        window (its last step may interpolate onto the data's grid). It is
        compiled once per device into a matrix ``P``.
    ``rotation``
        an optional component step after ``P``, e.g. ``RotateNE2RT(baz)``.
    ``misfit``
        a :class:`~obsnumpy.misfit.WindowedMisfit` on the processed grid;
        needed for the cost and the gradient.
    ``a``
        the Lanczos half-width of ``S(tau)``.
    ``prune``
        compute the cost and the gradient only at output samples some window
        covers (``misfit.mask``), dropping the other columns of ``P``. It
        changes no result.

    ``W`` is cast to ``U``'s dtype for the contraction (float32 by default);
    everything after the contraction is float64.
    """

    def __init__(self, window, processing, rotation=None, misfit=None, a=12, prune=False):
        self.window = window
        self.processing = processing
        self.rotation = rotation
        self.misfit = misfit
        self.a = int(a)
        self.prune = bool(prune)
        self._P = {}

    # -- compiled processing, per device ------------------------------------------------

    def _kept_columns(self):
        """The output samples some window covers (host indices)."""
        if self.misfit is None:
            raise ValueError("prune needs a misfit")
        mask = self.misfit.mask
        return np.flatnonzero(mask.reshape(-1, mask.shape[-1]).any(axis=0))

    def _proc(self, xp, pruned):
        """``(P, keep)``: the compiled processing, and with ``pruned`` the
        output columns kept (``None`` for all)."""
        key = (device_key(xp), pruned)
        if key not in self._P:
            P = self.processing.matrix(self.window.n, self.window.dt, xp=xp, dtype=np.float64)
            keep = None
            if pruned:
                keep = xp.asarray(self._kept_columns())
                P = xp.ascontiguousarray(P[:, keep])
            self._P[key] = (P, keep)
        return self._P[key]

    # -- the pieces ---------------------------------------------------------------------

    @staticmethod
    def _batch(W, tau, nbasis):
        xp = xp_of(W)
        W = xp.asarray(W)
        if W.ndim == 1:
            W = W[None]
        if W.shape[-1] != nbasis:
            raise ValueError(f"W has {W.shape[-1]} weights, the basis {nbasis}")
        tau = np.atleast_1d(np.asarray(tau.get() if hasattr(tau, "get") else tau,
                                       dtype=np.float64))
        if tau.shape != (W.shape[0],):
            raise ValueError(f"tau must be ({W.shape[0]},), got {tau.shape}")
        return W, tau

    def contract(self, basis, W):
        """``scale * U W``: ``(..., nrec, ncomp, nt)`` float64 for ``W`` of
        shape ``(..., nbasis)``."""
        R, C, nt, m = basis.U.shape
        lead = W.shape[:-1]
        U2 = basis.U.reshape(-1, m)
        v = _scaled_gemm(W.reshape(-1, m), U2.T, basis.U.dtype).reshape(lead + (R, C, nt))
        return v * basis.scale[:, None, None]

    def _taps(self, basis, tau, xp):
        """Per chain, where the window's samples fall in the basis, in the
        arithmetic of ``Shift``. Raises when a tap falls outside the basis."""
        dt_b, n, a = float(basis.dt), self.window.n, self.a
        nt = basis.U.shape[2]
        off = self.window.offset_from(basis.t0)
        step = self.window.dt / dt_b
        x0 = (off - tau) / dt_b                                  # host, (K,)
        lo = np.floor(x0) - a + 1
        hi = np.floor(x0 + (n - 1) * step) + a
        if lo.min() < 0 or hi.max() > nt - 1:
            k = int(np.argmin(lo)) if lo.min() < 0 else int(np.argmax(hi))
            raise ValueError(
                f"tau = {tau[k]:g} s needs basis samples {int(lo[k])}..{int(hi[k])}, "
                f"the basis has 0..{nt - 1}: it does not cover the window at this shift")
        if float(step).is_integer():
            # every output sits at the same fraction past its sample: the tap
            # weights depend on the chain only
            f0 = np.floor(x0)
            first = xp.asarray((f0 - a + 1).astype(np.int64))
            base = first[:, None] + xp.arange(n, dtype=xp.int64) * int(step)
            return _Taps(base=base, frac=None, frac0=x0 - f0, step=step)
        x = xp.asarray(x0)[:, None] + xp.arange(n, dtype=xp.float64) * step
        fl = xp.floor(x)
        return _Taps(base=fl.astype(xp.int64) - (a - 1), frac=x - fl, frac0=None, step=step)

    def _weights(self, taps, deriv, dt_b, xp):
        """The tap weights, ``(K, 1, 2a)`` when they depend on the chain only
        (computed on the host), else ``(K, n, 2a)``."""
        q = np.arange(-self.a + 1, self.a + 1, dtype=np.float64)
        if taps.frac is None:
            w = xp.asarray(tables.lanczos_kernel(taps.frac0[:, None, None] - q, self.a, deriv))
        else:
            w = tables.lanczos_kernel(taps.frac[..., None] - xp.asarray(q), self.a, deriv)
        if deriv:
            w = w * (-1.0 / dt_b)                 # dx/dtau = -1/dt
        return xp.ascontiguousarray(w)

    def shift(self, basis, v, tau, deriv=0, taps=None):
        """``S(tau) v`` (``deriv=1``: ``S'(tau) v``), per chain: ``v`` is
        ``(K, ..., nt)``, the result ``(K, ..., n)`` on the window."""
        xp = xp_of(v)
        taps = taps if taps is not None else self._taps(basis, tau, xp)
        w = self._weights(taps, deriv, float(basis.dt), xp)
        K, mid = v.shape[0], v.shape[1:-1]
        vf = xp.ascontiguousarray(v.reshape(K, -1, v.shape[-1]), dtype=xp.float64)
        if xp is np:
            y = _gather_np(vf, taps, w, self.window.n)
        else:
            y = _gather_cuda(xp, vf, taps, w, self.window.n)
        return y.reshape((K,) + mid + (self.window.n,))

    def shift_adjoint(self, basis, g, tau, taps=None):
        """``S(tau)^T g``: ``g`` is ``(K, ..., n)``, the result ``(K, ..., nt)``."""
        xp = xp_of(g)
        taps = taps if taps is not None else self._taps(basis, tau, xp)
        w = self._weights(taps, 0, float(basis.dt), xp)
        K, mid = g.shape[0], g.shape[1:-1]
        nt = basis.U.shape[2]
        gf = xp.ascontiguousarray(g.reshape(K, -1, g.shape[-1]), dtype=xp.float64)
        if xp is np:
            out = _scatter_np(gf, taps, w, nt)
        else:
            out = _scatter_cuda(xp, gf, taps, w, nt)
        return out.reshape((K,) + mid + (nt,))

    def _process(self, y, xp, pruned):
        """``Rot . P . y`` for ``y`` of shape ``(..., nrec, ncomp, n)``."""
        P, keep = self._proc(xp, pruned)
        lead = y.shape[:-1]
        s = (y.reshape(-1, y.shape[-1]) @ P).reshape(lead + (P.shape[1],))
        if self.rotation is not None:
            s = self.rotation(s, self.window.dt)
        return s, keep

    def _process_adjoint(self, g, xp, pruned):
        P, _ = self._proc(xp, pruned)
        if self.rotation is not None:
            g = self.rotation.adjoint(g, self.window.dt, g.shape[-1])
        lead = g.shape[:-1]
        return (g.reshape(-1, g.shape[-1]) @ P.T).reshape(lead + (P.shape[0],))

    def _cost_and_rw(self, s, keep, need_rw, trace_weights=None, per_trace=False):
        """The cost ``(K,)`` (``per_trace``: the per-trace costs ``(K, nrec,
        ncomp)``, without ``trace_weights``) and the residual weight, with
        ``omega`` times ``trace_weights`` ``(K, nrec, ncomp)`` if given. With
        ``keep``, on the pruned samples."""
        m = self.misfit
        if m is None:
            raise ValueError("the cost needs a misfit")
        if keep is not None:
            key = ("pruned", id(m))
            if key not in self._P:
                self._P[key] = m.take(self._kept_columns())
            m = self._P[key]
        C = m.trace_cost(s) if per_trace else m.cost(s, trace_weights)
        return C, (m.residual_weight(s, trace_weights) if need_rw else None)

    # -- the evaluations ---------------------------------------------------------------

    def forward(self, basis, W, tau, return_s=True, trace_weights=None, per_trace=False):
        """``s`` ``(K, nrec, ncomp, npts)`` and ``C`` ``(K,)`` (``None`` without a
        misfit). With ``return_s=False`` and ``prune``, only ``C``, computed on the
        windows' samples; ``s`` is then ``None``.

        ``trace_weights`` ``(K, nrec, ncomp)`` multiply the misfit's ``omega``
        per chain and trace: ``C_k = sum_i trace_weights[k, i] c_ki``.
        ``per_trace`` returns the per-trace costs ``c`` ``(K, nrec, ncomp)``
        instead of ``C`` (these do not include ``trace_weights``)."""
        W, tau = self._batch(W, tau, basis.U.shape[-1])
        xp = xp_of(basis.U)
        y = self.shift(basis, self.contract(basis, W), tau)
        pruned = self.prune and not return_s
        s, keep = self._process(y, xp, pruned)
        C = None if self.misfit is None else self._cost_and_rw(
            s, keep, False, trace_weights, per_trace)[0]
        return (s if return_s else None), C

    def gradient(self, basis, W, tau, trace_weights=None, per_trace=False):
        """``C`` ``(K,)``, ``dC/dW`` ``(K, nbasis)`` and ``dC/dtau`` ``(K,)``, by the
        adjoint: ``g = S^T P^T Rot^T dC/ds``, ``dC/dW = (scale U)^T g`` and
        ``dC/dtau = <dC/ds, Rot P S'(tau) v>``. ``trace_weights`` and
        ``per_trace`` as in :meth:`forward`; the derivatives are those of the
        weighted ``C``."""
        W, tau = self._batch(W, tau, basis.U.shape[-1])
        xp = xp_of(basis.U)
        taps = self._taps(basis, tau, xp)
        v = self.contract(basis, W)
        y = self.shift(basis, v, tau, taps=taps)
        s, keep = self._process(y, xp, self.prune)
        C, rw = self._cost_and_rw(s, keep, True, trace_weights, per_trace)
        gy = self._process_adjoint(rw, xp, self.prune)                 # on the window
        dtau = (gy * self.shift(basis, v, tau, deriv=1, taps=taps)).sum(axis=(-3, -2, -1))
        gv = self.shift_adjoint(basis, gy, tau, taps=taps) * basis.scale[:, None, None]
        K, m = W.shape
        U2 = basis.U.reshape(-1, m)
        dW = _scaled_gemm(gv.reshape(K, -1), U2, basis.U.dtype)
        return C, dW, dtau

    def jacobian(self, basis, W, tau, dW):
        """``s`` ``(K, nrec, ncomp, npts)`` and ``J`` ``(K, nd + 1, nrec, ncomp,
        npts)``: the processed derivatives along the caller's derivative weights
        ``dW`` ``(K, nd, nbasis)``, and in the last row along ``tau``. One GEMM
        contracts ``[W, dW]``."""
        W, tau = self._batch(W, tau, basis.U.shape[-1])
        xp = xp_of(basis.U)
        dW = xp.asarray(dW)
        if dW.ndim == 2:
            dW = dW[None]
        if dW.shape[0] != W.shape[0] or dW.shape[-1] != W.shape[-1]:
            raise ValueError(f"dW must be ({W.shape[0]}, nd, {W.shape[-1]}), got {dW.shape}")
        taps = self._taps(basis, tau, xp)
        v = self.contract(basis, xp.concatenate([W[:, None], dW], axis=1))  # (K, nd+1, R, C, nt)
        y = self.shift(basis, v, tau, taps=taps)
        dy_tau = self.shift(basis, v[:, 0], tau, deriv=1, taps=taps)
        rows = xp.concatenate([y, dy_tau[:, None]], axis=1)               # (K, nd+2, R, C, n)
        out, _ = self._process(rows, xp, False)
        return out[:, 0], out[:, 1:]


# ----------------------------------------------------------------------------- S(tau), applied

@dataclasses.dataclass
class _Taps:
    """Where the window's samples fall in the basis, per chain ``k``: output
    ``j`` reads basis samples ``base[k, j] + t`` for ``t = 0 .. 2a - 1``.
    ``frac`` is ``(K, n)``, or ``None`` when the grids' ratio is an integer.
    Then every output of chain ``k`` sits ``frac0[k]`` past its sample, and
    the weights depend on the chain only."""

    base: object
    frac: object
    frac0: object
    step: float


def _gather_np(v, taps, w, n):
    K, RC, _ = v.shape
    ntaps = w.shape[-1]
    y = np.zeros((K, RC, n))
    if taps.step == 1.0:
        # one gather of each chain's slab, then contiguous multiply-adds
        idx = taps.base[:, :1] + np.arange(n + ntaps - 1)
        slab = np.take_along_axis(v, idx[:, None, :], axis=-1)
        for t in range(ntaps):
            y += w[:, 0, t][:, None, None] * slab[..., t:t + n]
        return y
    kk, rr = np.arange(K)[:, None, None], np.arange(RC)[None, :, None]
    for t in range(ntaps):
        wt = w[:, :, t][:, None, :]
        y += wt * v[kk, rr, (taps.base + t)[:, None, :]]
    return y


def _scatter_np(g, taps, w, nt):
    K, RC, n = g.shape
    ntaps = w.shape[-1]
    out = np.zeros((K, RC, nt))
    kk, rr = np.arange(K)[:, None, None], np.arange(RC)[None, :, None]
    if taps.step == 1.0:
        slab = np.zeros((K, RC, n + ntaps - 1))
        for t in range(ntaps):
            slab[..., t:t + n] += w[:, 0, t][:, None, None] * g
        idx = taps.base[:, :1] + np.arange(n + ntaps - 1)
        out[kk, rr, idx[:, None, :]] = slab                       # distinct indices
        return out
    for t in range(ntaps):
        contrib = w[:, :, t][:, None, :] * g
        idx = (taps.base + t)[:, None, :]
        if taps.step >= 1.0:
            out[kk, rr, idx] += contrib            # positions advance >= 1: no index repeats
        else:
            np.add.at(out, (kk, rr, idx), contrib)
    return out


_CUDA_SRC = r"""
extern "C" {

// y[k, r, j] = sum_t w[k, j or 0, t] * v[k, r, base[k, j] + t]
__global__ void shift_gather(const double* __restrict__ v, const long long* __restrict__ base,
                             const double* __restrict__ w, double* __restrict__ y,
                             long long K, long long RC, long long nt, long long n,
                             int ntaps, int wvar)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= K * RC * n) return;
    long long j = i % n, kr = i / n, k = kr / RC;
    const double* vv = v + kr * nt + base[k * n + j];
    const double* ww = w + (wvar ? (k * n + j) : k) * ntaps;
    double acc = 0.0;
    for (int t = 0; t < ntaps; ++t) acc += ww[t] * vv[t];
    y[i] = acc;
}

// the adjoint where the grids share dt (base[k, j] = base[k, 0] + j): each
// basis sample pulls its 2a contributions, without atomics
__global__ void shift_pull(const double* __restrict__ g, const long long* __restrict__ base,
                           const double* __restrict__ w, double* __restrict__ out,
                           long long K, long long RC, long long nt, long long n, int ntaps)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= K * RC * nt) return;
    long long s = i % nt, kr = i / nt, k = kr / RC;
    long long j0 = s - base[k * n];
    const double* gg = g + kr * n;
    const double* ww = w + k * ntaps;
    double acc = 0.0;
    for (int t = 0; t < ntaps; ++t) {
        long long j = j0 - t;
        if (j >= 0 && j < n) acc += ww[t] * gg[j];
    }
    out[i] = acc;
}

// the general adjoint: scatter with atomics (indices may repeat)
__global__ void shift_scatter(const double* __restrict__ g, const long long* __restrict__ base,
                              const double* __restrict__ w, double* __restrict__ out,
                              long long K, long long RC, long long nt, long long n,
                              int ntaps, int wvar)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= K * RC * n) return;
    long long j = i % n, kr = i / n, k = kr / RC;
    double* oo = out + kr * nt + base[k * n + j];
    const double* ww = w + (wvar ? (k * n + j) : k) * ntaps;
    double gv = g[i];
    for (int t = 0; t < ntaps; ++t) atomicAdd(oo + t, ww[t] * gv);
}

}
"""
_MODULES = {}


def _kernel(cp, name):
    dev = device_key(cp)
    if dev not in _MODULES:
        _MODULES[dev] = cp.RawModule(code=_CUDA_SRC)
    return _MODULES[dev].get_function(name)


def _launch(cp, name, total, *args):
    threads = 256
    _kernel(cp, name)(((total + threads - 1) // threads,), (threads,), args)


def _gather_cuda(cp, v, taps, w, n):
    K, RC, nt = v.shape
    y = cp.empty((K, RC, n), dtype=cp.float64)
    base = cp.ascontiguousarray(taps.base)
    _launch(cp, "shift_gather", K * RC * n, v, base, w, y, cp.int64(K), cp.int64(RC),
            cp.int64(nt), cp.int64(n), cp.int32(w.shape[-1]), cp.int32(w.shape[1] != 1))
    return y


def _scatter_cuda(cp, g, taps, w, nt):
    K, RC, n = g.shape
    base = cp.ascontiguousarray(taps.base)
    if taps.step == 1.0:
        out = cp.empty((K, RC, nt), dtype=cp.float64)
        _launch(cp, "shift_pull", K * RC * nt, g, base, w, out, cp.int64(K), cp.int64(RC),
                cp.int64(nt), cp.int64(n), cp.int32(w.shape[-1]))
        return out
    out = cp.zeros((K, RC, nt), dtype=cp.float64)
    _launch(cp, "shift_scatter", K * RC * n, g, base, w, out, cp.int64(K), cp.int64(RC),
            cp.int64(nt), cp.int64(n), cp.int32(w.shape[-1]), cp.int32(w.shape[1] != 1))
    return out


def _scaled_gemm(A, B, dtype):
    """``A @ B`` with ``A`` cast to ``dtype`` (``B``'s), returned in float64.

    Each row of ``A`` is divided by its largest magnitude before the cast and
    multiplied back after. In float32 the product then keeps its relative
    precision whatever the rows' magnitudes: a moment-tensor derivative
    weight (per dyne-cm, ~1e-37) would otherwise underflow.
    """
    xp = xp_of(A)
    A = xp.asarray(A, dtype=xp.float64)
    s = xp.abs(A).max(axis=1, keepdims=True)
    s = xp.where(s > 0, s, 1.0)
    return ((A / s).astype(dtype) @ B).astype(xp.float64) * s
