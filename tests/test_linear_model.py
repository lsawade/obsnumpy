"""Tests of obsnumpy.linear_model on a synthetic, self-contained problem.

A random band-limited basis (3 receivers, 3 components, 20 basis vectors) on
a 1 s grid. It is shifted onto a FixedWindow, processed by a cmt3d-like chain
that ends in Lanczos to 2 s, rotated, and compared through a WindowedMisfit
with random windows.

The oracle for the forward model is the explicit route, step by step with
O1-O3's own steps: contract, Shift(tau), the pipeline uncompiled, the
rotation, the misfit. The derivatives are checked by central differences,
and against each other: the adjoint gradient against the Jacobian's.
"""
import types

import numpy as np
import pytest

from obsnumpy.linear_model import Basis, BasisStore, LinearBasisModel
from obsnumpy.misfit import WindowedMisfit, windows_to_tapers
from obsnumpy.pipeline import (Detrend, FixedWindow, Interpolate, Pipeline, PreFilt,
                               RotateNE2RT, Shift, Taper)

R, C, NT, M = 3, 3, 400, 20
DT_B = 1.0
RNG = np.random.default_rng(0)


def smooth(rng, shape, width=4.0):
    x = rng.standard_normal(shape)
    k = np.exp(-0.5 * (np.arange(-12, 13) / width) ** 2)
    return np.apply_along_axis(lambda t: np.convolve(t, k, mode="same"), -1, x)


def make_case(window_dt=1.0, n=250, out_dt=2.0, out_n=110, seed=0):
    rng = np.random.default_rng(seed)
    U = np.moveaxis(smooth(rng, (R, C, M, NT)), 2, 3).copy()           # (R, C, NT, M)
    basis = Basis(U=U, t0=0.0, dt=DT_B, scale=rng.uniform(0.5, 2.0, R))
    window = FixedWindow(first_time=60.0, dt=window_dt, n=n)
    processing = Pipeline([Detrend("linear"), Detrend("demean"), Taper(0.05),
                           PreFilt((0.01, 0.02, 0.15, 0.25)),
                           Detrend("linear"), Detrend("demean"), Taper(0.05),
                           Interpolate(out_dt, out_n, 10.3)])
    rotation = RotateNE2RT(rng.uniform(0, 360, R))
    model = LinearBasisModel(window, processing, rotation)
    W0, tau0 = rng.standard_normal(M), 1.3
    s0, _ = model.forward(basis, W0, tau0)
    d = s0[0] + 0.05 * np.abs(s0).max() * smooth(rng, s0.shape[1:], 2.0)
    windows = [[[(int(a), int(a) + int(rng.integers(15, 50)))
                 for a in rng.integers(0, out_n - 50, rng.integers(1, 3))]
                for _ in range(C)] for _ in range(R)]
    misfit = WindowedMisfit.build(d, out_dt, windows_to_tapers(windows, out_n),
                                  weights=rng.uniform(0.5, 2.0, (R, C)))
    model.misfit = misfit
    return types.SimpleNamespace(basis=basis, window=window, processing=processing,
                                 rotation=rotation, misfit=misfit, model=model, rng=rng)


@pytest.fixture(scope="module")
def case():
    return make_case()


def explicit(case, W, tau):
    """contract -> Shift(tau) -> pipeline step by step -> rotation -> cost."""
    b = case.basis
    v = np.einsum("rctm,m->rct", b.U.astype(np.float64), W) * b.scale[:, None, None]
    shift = Shift(tau, case.window.dt, case.window.n, case.window.offset_from(b.t0))
    y = shift(v, b.dt)
    s, dt = case.processing(y, case.window.dt)
    s = case.rotation(s, dt)
    return s, case.misfit.cost(s)


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a - b).max() / np.abs(b).max())


def assert_second_order(errs):
    assert errs[0] < 1e-2 and 3.5 < errs[0] / errs[1] < 4.5, errs


# ----------------------------------------------------------------------------- forward

def test_forward_is_the_explicit_route(case):
    W = case.rng.standard_normal(M)
    for tau in (0.0, 1.3, -7.77, 15.2):
        s, c = case.model.forward(case.basis, W, tau)
        s_ref, c_ref = explicit(case, W, tau)
        assert s.shape == (1,) + s_ref.shape
        assert rel(s[0], s_ref) <= 1e-12
        assert abs(c[0] - c_ref) <= 1e-12 * abs(c_ref)


def test_processing_after_the_shift_matters(case):
    """The absolute frame: shifting and then processing on the fixed window
    (the model) is not processing on the tau = 0 window and then shifting the
    processed trace. Cut, detrend and taper do not commute with the shift."""
    W, tau = case.rng.standard_normal(M), 4.6
    s, _ = case.model.forward(case.basis, W, tau)
    s0, _ = case.model.forward(case.basis, W, 0.0)
    out = case.processing.steps[-1]
    shifted_after = Shift(tau, out.new_dt, out.npts, 0.0)(s0[0], out.new_dt)
    assert rel(shifted_after, s[0]) > 1e-3


def test_batch_equals_single_calls(case):
    W = case.rng.standard_normal((5, M))
    tau = np.array([0.0, 1.3, -7.77, 15.2, 3.0])
    s, c = case.model.forward(case.basis, W, tau)
    Cg, dW, dt = case.model.gradient(case.basis, W, tau)
    dWd = case.rng.standard_normal((5, 2, M))
    sj, J = case.model.jacobian(case.basis, W, tau, dWd)
    for k in range(5):
        s1, c1 = case.model.forward(case.basis, W[k], tau[k])
        assert rel(s[k], s1[0]) <= 1e-13 and abs(c[k] - c1[0]) <= 1e-13 * abs(c1[0])
        g1 = case.model.gradient(case.basis, W[k], tau[k])
        assert rel(dW[k], g1[1][0]) <= 1e-13 and abs(dt[k] - g1[2][0]) <= 1e-13 * abs(g1[2][0])
        j1 = case.model.jacobian(case.basis, W[k], tau[k], dWd[k])
        assert rel(J[k], j1[1][0]) <= 1e-13 and rel(sj[k], j1[0][0]) <= 1e-13


def test_float32_basis(case):
    """The default: a float32 basis and contraction, float64 after."""
    b32 = Basis(U=case.basis.U.astype(np.float32), t0=0.0, dt=DT_B, scale=case.basis.scale)
    W = case.rng.standard_normal(M)
    s32, c32 = case.model.forward(b32, W, 1.3)
    s64, c64 = case.model.forward(case.basis, W, 1.3)
    assert s32.dtype == np.float64
    assert 0 < rel(s32, s64) <= 1e-5


def test_float32_whatever_the_magnitude_of_W(case):
    """Weights of any magnitude keep float32's relative precision: a weight
    vector scaled to 1e-37 (a moment-tensor derivative per dyne-cm) would
    underflow if cast as it is."""
    b32 = Basis(U=case.basis.U.astype(np.float32), t0=0.0, dt=DT_B, scale=case.basis.scale)
    W = case.rng.standard_normal(M)
    tiny, huge = 1e-37, 1e30
    s1, _ = case.model.forward(b32, W, 1.3)
    st, _ = case.model.forward(b32, tiny * W, 1.3)
    sh, _ = case.model.forward(b32, huge * W, 1.3)
    assert rel(st / tiny, s1) <= 1e-12 and rel(sh / huge, s1) <= 1e-12
    _, J = case.model.jacobian(b32, W, 1.3, tiny * case.rng.standard_normal((1, M)))
    assert np.abs(J[0, 0]).max() > 0


def test_tau_outside_the_basis_raises(case):
    with pytest.raises(ValueError, match="does not cover the window"):
        case.model.forward(case.basis, np.ones(M), 50.0)
    with pytest.raises(ValueError, match="does not cover the window"):
        case.model.forward(case.basis, np.ones(M), -80.0)


# ----------------------------------------------------------------------------- the shift and its adjoint

# window dt / basis dt: the same grid; finer (a scatter with repeated
# indices); coarser; an integer ratio (weights per chain only)
WINDOW_DTS = [1.0, 0.6, 1.7, 2.0]


@pytest.mark.parametrize("window_dt", WINDOW_DTS)
def test_shift_adjoint_dot(window_dt):
    """<S v, g> = <v, S^T g> per chain, and S, S' are O2's Shift step."""
    cs = make_case(window_dt=window_dt, n=int(250 / window_dt))
    rng = np.random.default_rng(3)
    v = rng.standard_normal((4, R, C, NT))
    tau = np.array([0.0, 1.3, -7.77, 15.2])
    y = cs.model.shift(cs.basis, v, tau)
    g = rng.standard_normal(y.shape)
    lhs = (y * g).sum(axis=(1, 2, 3))
    rhs = (v * cs.model.shift_adjoint(cs.basis, g, tau)).sum(axis=(1, 2, 3))
    assert np.abs(lhs - rhs).max() <= 1e-12 * np.linalg.norm(y) * np.linalg.norm(g)
    # and the shift is O2's Shift step
    for k in range(4):
        ref = Shift(tau[k], cs.window.dt, cs.window.n, cs.window.offset_from(0.0))(v[k], DT_B)
        assert rel(y[k], ref) <= 1e-13
        dref = Shift(tau[k], cs.window.dt, cs.window.n, cs.window.offset_from(0.0)).d_dtau()(
            v[k], DT_B)
        assert rel(cs.model.shift(cs.basis, v[k:k + 1], tau[k:k + 1], deriv=1)[0], dref) <= 1e-13


@pytest.mark.parametrize("window_dt", [0.6, 1.7])
def test_forward_other_window_grids(window_dt):
    cs = make_case(window_dt=window_dt, n=int(250 / window_dt))
    W = cs.rng.standard_normal(M)
    s, c = cs.model.forward(cs.basis, W, 2.1)
    s_ref, c_ref = explicit(cs, W, 2.1)
    assert rel(s[0], s_ref) <= 1e-12 and abs(c[0] - c_ref) <= 1e-12 * abs(c_ref)


# ----------------------------------------------------------------------------- gradient and Jacobian

def test_gradient_in_W_by_central_differences(case):
    """C is quadratic in W, so central differences are exact up to rounding."""
    W, tau = case.rng.standard_normal(M), 1.3
    _, dW, _ = case.model.gradient(case.basis, W, tau)
    for m in (0, 7, 19):
        e = np.zeros(M)
        e[m] = 1e-3
        cd = (case.model.forward(case.basis, W + e, tau)[1][0]
              - case.model.forward(case.basis, W - e, tau)[1][0]) / 2e-3
        assert abs(dW[0, m] - cd) <= 1e-8 * np.abs(dW).max()


def test_gradient_in_tau_by_central_differences(case):
    """Second order. The window's samples sit 0.3 past the basis's at tau =
    1.3 (offset 60 - 1.3), so +-h never crosses a sample."""
    W, tau = case.rng.standard_normal(M), 1.3
    _, _, dtau = case.model.gradient(case.basis, W, tau)
    errs = []
    for h in (0.05, 0.025):
        cd = (case.model.forward(case.basis, W, tau + h)[1][0]
              - case.model.forward(case.basis, W, tau - h)[1][0]) / (2 * h)
        errs.append(abs(cd - dtau[0]) / abs(dtau[0]))
    assert_second_order(errs)


def test_jacobian_by_central_differences(case):
    W, tau = case.rng.standard_normal(M), 1.3
    dWd = case.rng.standard_normal((2, M))
    s, J = case.model.jacobian(case.basis, W, tau, dWd)
    assert J.shape == (1, 3) + s.shape[1:]
    assert rel(s[0], case.model.forward(case.basis, W, tau)[0][0]) <= 1e-13
    for i in range(2):                                    # linear in W: exact
        h = 1e-3
        cd = (case.model.forward(case.basis, W + h * dWd[i], tau)[0]
              - case.model.forward(case.basis, W - h * dWd[i], tau)[0]) / (2 * h)
        assert rel(J[0, i], cd[0]) <= 1e-9
    errs = []
    for h in (0.05, 0.025):
        cd = (case.model.forward(case.basis, W, tau + h)[0]
              - case.model.forward(case.basis, W, tau - h)[0]) / (2 * h)
        errs.append(rel(J[0, 2], cd[0]))
    assert_second_order(errs)


def test_adjoint_and_jacobian_routes_agree(case):
    """dC/dtheta = J^T dC/ds: the adjoint gradient along each direction of
    dW, and along tau, equals the Jacobian's."""
    W, tau = case.rng.standard_normal((2, M)), np.array([1.3, -4.2])
    dWd = case.rng.standard_normal((2, 4, M))
    _, dW, dtau = case.model.gradient(case.basis, W, tau)
    s, J = case.model.jacobian(case.basis, W, tau, dWd)
    via_J = case.misfit.gradient(s, J)                    # (K, 5)
    assert rel(via_J[:, :4], np.einsum("kdm,km->kd", dWd, dW)) <= 1e-10
    assert rel(via_J[:, 4], dtau) <= 1e-10


# ----------------------------------------------------------------------------- a rotated basis

def test_rotated_basis_replaces_the_rotation(case):
    """Rotating U once and dropping the per-evaluation rotation gives the same
    forward, gradient and Jacobian (float64 basis)."""
    M_ = case.rotation.matrices(R)
    x = case.rng.standard_normal((R, C, 7))
    assert rel(np.einsum("rij,rjt->rit", M_, x), case.rotation(x, 1.0)) <= 1e-15
    rb = case.basis.rotated(case.rotation)
    assert rb.U.flags.c_contiguous                        # else every contraction copies U
    plain = LinearBasisModel(case.window, case.processing, None, case.misfit)
    W, tau = case.rng.standard_normal((2, M)), np.array([1.3, -4.2])
    dWd = case.rng.standard_normal((2, 3, M))
    s0, c0 = case.model.forward(case.basis, W, tau)
    s1, c1 = plain.forward(rb, W, tau)
    assert rel(s1, s0) <= 1e-13 and rel(c1, c0) <= 1e-13
    for a, b in zip(plain.gradient(rb, W, tau), case.model.gradient(case.basis, W, tau)):
        assert rel(a, b) <= 1e-12
    assert rel(plain.jacobian(rb, W, tau, dWd)[1],
               case.model.jacobian(case.basis, W, tau, dWd)[1]) <= 1e-13


# ----------------------------------------------------------------------------- pruning

def test_pruning_changes_nothing(case):
    pruned = LinearBasisModel(case.window, case.processing, case.rotation, case.misfit,
                              prune=True)
    W, tau = case.rng.standard_normal((3, M)), np.array([0.0, 1.3, -7.7])
    assert pruned._proc(np, True)[0].shape[1] < case.model._proc(np, False)[0].shape[1]
    _, c0 = case.model.forward(case.basis, W, tau)
    s1, c1 = pruned.forward(case.basis, W, tau, return_s=False)
    assert s1 is None and rel(c1, c0) <= 1e-13
    g0, g1 = case.model.gradient(case.basis, W, tau), pruned.gradient(case.basis, W, tau)
    for a, b in zip(g1, g0):
        assert rel(a, b) <= 1e-13
    s2, c2 = pruned.forward(case.basis, W, tau)            # with s: the full grid
    assert rel(c2, c0) <= 1e-13 and s2.shape[-1] == case.misfit.d.shape[-1]


# ----------------------------------------------------------------------------- the store

def basis_of(key, nbytes_scale=1):
    U = np.zeros((1, 1, 10 * nbytes_scale, 4), dtype=np.float32)
    return Basis(U=U, t0=0.0, dt=1.0, scale=np.ones(1))


def test_store_lru():
    store = BasisStore(basis_of, max_items=2)
    for key in "ABACA":
        store.get(key)
    assert store.stats == dict(hits=2, misses=3, evictions=1, items=2, bytes=2 * 160)
    assert "A" in store and "C" in store and "B" not in store


def test_store_bytes():
    store = BasisStore(lambda k: basis_of(k, nbytes_scale=k), max_bytes=1000)  # 160 B per scale
    for key in (1, 2, 3):                                  # 160 + 320 + 480 = 960
        store.get(key)
    assert len(store) == 3 and store.nbytes == 960
    store.get(4)                                           # 640: evicts 1, 2 and 3
    assert list(store._items) == [4] and store.stats["evictions"] == 3
    store.get(10)                                          # 1600 > budget: kept alone
    assert list(store._items) == [10] and store.nbytes == 1600


# ----------------------------------------------------------------------------- CuPy

@pytest.mark.gpu
def test_cupy(case, cp):
    gb = case.basis.to(cp)
    assert isinstance(gb.U, cp.ndarray)
    W = case.rng.standard_normal((3, M))
    tau = np.array([0.0, 1.3, -7.7])
    s, c = case.model.forward(gb, cp.asarray(W), tau)
    s0, c0 = case.model.forward(case.basis, W, tau)
    assert isinstance(s, cp.ndarray) and rel(s.get(), s0) <= 1e-12 and rel(c.get(), c0) <= 1e-12
    g, g0 = case.model.gradient(gb, cp.asarray(W), tau), case.model.gradient(case.basis, W, tau)
    for a, b in zip(g, g0):
        assert rel(a.get(), b) <= 1e-12
    dWd = case.rng.standard_normal((3, 2, M))
    sj, J = case.model.jacobian(gb, cp.asarray(W), tau, cp.asarray(dWd))
    assert rel(J.get(), case.model.jacobian(case.basis, W, tau, dWd)[1]) <= 1e-12
    store = BasisStore(lambda k: case.basis, xp=cp)
    assert isinstance(store.get("x").U, cp.ndarray)
    rgb = gb.rotated(case.rotation)
    assert rgb.U.flags.c_contiguous                       # CuPy's einsum returns it strided
    assert rel(rgb.U.get(), case.basis.rotated(case.rotation).U) <= 1e-15


@pytest.mark.gpu
@pytest.mark.parametrize("window_dt", WINDOW_DTS)
def test_cupy_shift(cp, window_dt):
    """The CUDA gather, pull and scatter kernels against numpy."""
    cs = make_case(window_dt=window_dt, n=int(250 / window_dt))
    rng = np.random.default_rng(4)
    v = rng.standard_normal((4, R, C, NT))
    tau = np.array([0.0, 1.3, -7.77, 15.2])
    gb = cs.basis.to(cp)
    for deriv in (0, 1):
        y = cs.model.shift(gb, cp.asarray(v), tau, deriv=deriv)
        assert rel(y.get(), cs.model.shift(cs.basis, v, tau, deriv=deriv)) <= 1e-13
    g = rng.standard_normal(y.shape)
    a = cs.model.shift_adjoint(gb, cp.asarray(g), tau)
    assert rel(a.get(), cs.model.shift_adjoint(cs.basis, g, tau)) <= 1e-13


@pytest.mark.gpu
def test_cupy_upsampling_scatter(cp):
    cs = make_case(window_dt=0.6, n=int(250 / 0.6))
    gb = cs.basis.to(cp)
    W = cs.rng.standard_normal((2, M))
    tau = np.array([0.4, -3.1])
    g, g0 = cs.model.gradient(gb, cp.asarray(W), tau), cs.model.gradient(cs.basis, W, tau)
    for a, b in zip(g, g0):
        assert rel(a.get(), b) <= 1e-12
