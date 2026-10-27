"""Tests of obsnumpy.misfit against cmt3d's CostGradHess formula, written
out here.

The setup has five stations, R/T/Z and random windows: overlapping windows,
traces without windows, a trace whose data vanish in its windows (N = 0, not
divided), and per-trace weights.
"""
import types

import numpy as np
import pytest
from scipy.signal import windows as sw

from obsnumpy.misfit import WindowedMisfit, windows_to_tapers

NSTA, NCOMP, NPTS, DT, NPAR = 5, 3, 400, 0.5, 4
TOL = 1e-13


def smooth(rng, shape):
    """Band-limited random traces."""
    x = rng.standard_normal(shape)
    k = np.exp(-0.5 * (np.arange(-15, 16) / 5.0) ** 2)
    return np.apply_along_axis(lambda t: np.convolve(t, k, mode="same"), -1, x)


def random_windows(rng):
    """Per trace 0-3 windows, some overlapping."""
    out = []
    for i in range(NSTA):
        per = []
        for c in range(NCOMP):
            wins = []
            for _ in range(rng.integers(0, 4)):
                left = int(rng.integers(0, NPTS - 30))
                right = int(min(NPTS, left + rng.integers(20, 150)))
                wins.append((left, right))
            per.append(wins)
        out.append(per)
    out[0][0] = [(30, 130), (100, 220)]          # overlapping, for sure
    out[1][2] = []                               # no window
    out[3][1] = [(50, 120)]                      # data zero there: N = 0
    return out


@pytest.fixture(scope="module")
def case():
    rng = np.random.default_rng(0)
    d = smooth(rng, (NSTA, NCOMP, NPTS))
    d[3, 1] = 0.0
    s = 1.1 * d + 0.3 * smooth(rng, d.shape)
    J = smooth(rng, (NPAR, NSTA, NCOMP, NPTS))
    windows = random_windows(rng)
    weights = rng.uniform(0.5, 2.0, (NSTA, NCOMP))
    return types.SimpleNamespace(d=d, s=s, J=J, windows=windows, weights=weights)


def build(case, **kw):
    return WindowedMisfit.build(case.d, DT, windows_to_tapers(case.windows, NPTS),
                                weights=case.weights, **kw)


# ----------------------------------------------------------------------------- the formula, inline

def inline(case, s, J=None, normalize=True):
    """cmt3d costgradhess.py:124-178 (cost) and :180-256 (grad), on arrays: per
    trace, per window, Tukey(alpha=0.25) on [left, right). The Hessian is the
    Gauss-Newton Hessian of that cost (cmt3d's hess squares the taper)."""
    cost, grad = 0.0, np.zeros(NPAR)
    hess = np.zeros((NPAR, NPAR))
    for i in range(NSTA):
        for c in range(NCOMP):
            ct, fnorm, gt, ht = 0.0, 0.0, np.zeros(NPAR), np.zeros((NPAR, NPAR))
            for left, right in case.windows[i][c]:
                tap = sw.tukey(right - left, alpha=0.25)
                ws, wo = s[i, c, left:right], case.d[i, c, left:right]
                ct += 0.5 * np.sum(tap * (ws - wo) ** 2) * DT
                fnorm += np.sum(tap * wo ** 2) * DT
                if J is not None:
                    wj = J[:, i, c, left:right]
                    gt += np.sum((ws - wo) * tap * wj, axis=-1) * DT
                    ht += (wj * tap) @ wj.T * DT       # the Gauss-Newton Hessian of the cost
            w = case.weights[i, c]
            ct, gt, ht = ct * w, gt * w, ht * w
            if normalize and fnorm != 0:
                ct, gt, ht = ct / fnorm, gt / fnorm, ht / fnorm
            cost, grad, hess = cost + ct, grad + gt, hess + ht
    return cost, grad, hess


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a - b).max() / np.abs(b).max())


# ----------------------------------------------------------------------------- tapers

def test_windows_to_tapers():
    w = [[[(10, 30), (20, 50)], []], [[(0, 60)], [(55, 60)]]]
    t = windows_to_tapers(w, 60)
    ref = np.zeros((2, 2, 60))
    ref[0, 0, 10:30] += sw.tukey(20, 0.25)
    ref[0, 0, 20:50] += sw.tukey(30, 0.25)
    ref[1, 0, 0:60] += sw.tukey(60, 0.25)
    ref[1, 1, 55:60] += sw.tukey(5, 0.25)
    assert np.array_equal(t, ref)
    assert np.array_equal(windows_to_tapers(w, 60, kind="boxcar")[0, 0, 20:30], [2.0] * 10)
    with pytest.raises(ValueError, match="not inside"):
        windows_to_tapers([[[(50, 61)]]], 60)


# ----------------------------------------------------------------------------- against the formula

def test_cost_gradient_hessian_inline(case):
    m = build(case)
    c, g, h = inline(case, case.s, case.J)
    assert abs(m.cost(case.s) - c) <= TOL * abs(c)
    assert rel(m.gradient(case.s, case.J), g) <= TOL
    assert rel(m.gauss_newton(case.J), h) <= TOL


def test_normalize_false(case):
    m = build(case, normalize=False)
    c, g, _ = inline(case, case.s, case.J, normalize=False)
    assert abs(m.cost(case.s) - c) <= TOL * abs(c)
    assert rel(m.gradient(case.s, case.J), g) <= TOL


def test_zero_norm_trace_is_not_divided(case):
    """Trace (3, 1) has zero data in its window: N = 0, so production keeps
    its weighted, undivided cost."""
    m = build(case)
    tap = windows_to_tapers(case.windows, NPTS)
    assert np.array_equal(m.omega[3, 1], case.weights[3, 1] * tap[3, 1])
    assert m.omega[3, 1].max() > 0


def test_mask(case):
    m = build(case)
    assert np.array_equal(m.mask, windows_to_tapers(case.windows, NPTS) > 0)
    assert not m.mask[1, 2].any()



# ----------------------------------------------------------------------------- derivatives, groups, batches

def test_residual_weight_is_the_derivative(case):
    """C is quadratic in s, so a central difference is exact up to rounding."""
    m = build(case)
    v = smooth(np.random.default_rng(5), case.s.shape)
    h = 1e-3
    cd = (m.cost(case.s + h * v) - m.cost(case.s - h * v)) / (2 * h)
    dd = float(np.sum(m.residual_weight(case.s) * v))
    assert abs(cd - dd) <= 1e-9 * abs(dd)


def test_groups_sum(case):
    """Three groups (wave types), each normalised by its own N, collapse into
    one omega whose cost is the sum of the three."""
    rng = np.random.default_rng(7)
    groups = [(windows_to_tapers(random_windows(rng), NPTS), rng.uniform(0.5, 2, (NSTA, NCOMP)))
              for _ in range(3)]
    one = WindowedMisfit.from_groups(case.d, DT, groups)
    parts = [WindowedMisfit.build(case.d, DT, t, w) for t, w in groups]
    total = sum(p.cost(case.s) for p in parts)
    assert abs(one.cost(case.s) - total) <= TOL * abs(total)


def test_batch(case):
    m = build(case)
    rng = np.random.default_rng(9)
    S = case.s + 0.1 * smooth(rng, (5,) + case.s.shape)
    JJ = case.J + 0.1 * smooth(rng, (5,) + case.J.shape)
    c = m.cost(S)
    assert c.shape == (5,)
    for k in range(5):
        assert abs(c[k] - m.cost(S[k])) <= TOL * abs(c[k])
        assert rel(m.gradient(S, JJ)[k], m.gradient(S[k], JJ[k])) <= TOL
        assert rel(m.gauss_newton(JJ)[k], m.gauss_newton(JJ[k])) <= TOL
    assert m.residual_weight(S).shape == S.shape


@pytest.mark.gpu
def test_cupy(case, cp):
    m = build(case)
    s, J = cp.asarray(case.s), cp.asarray(case.J)
    assert abs(m.cost(s) - m.cost(case.s)) <= TOL * abs(m.cost(case.s))
    assert isinstance(m.residual_weight(s), cp.ndarray)
    assert rel(m.residual_weight(s).get(), m.residual_weight(case.s)) <= TOL
    assert rel(m.gradient(s, J).get(), m.gradient(case.s, case.J)) <= TOL
    assert rel(m.gauss_newton(J).get(), m.gauss_newton(case.J)) <= TOL
    S = cp.stack([s, 2 * s])
    assert m.cost(S).shape == (2,)


# ----------------------------------------------------------------------------- probe

def test_probe_normalising_without_the_taper_fails(case):
    """Normalising by the untapered energy in the windows (boxcar) instead of
    the tapered one must fail the comparison with the formula."""
    c, _, _ = inline(case, case.s)
    tap = windows_to_tapers(case.windows, NPTS)
    box = windows_to_tapers(case.windows, NPTS, kind="boxcar")
    N_box = np.sum(box * case.d ** 2, axis=-1, keepdims=True) * DT
    wrong = WindowedMisfit(d=case.d, omega=case.weights[..., None] * tap /
                           np.where(N_box != 0, N_box, 1.0), dt=DT)
    assert abs(wrong.cost(case.s) - c) > 1e3 * TOL * abs(c)


def test_trace_cost_weights_and_take(case):
    m = build(case)
    tc = m.trace_cost(case.s)
    assert tc.shape == case.d.shape[:-1] and abs(tc.sum() / m.cost(case.s) - 1) <= 1e-13
    tw = np.random.default_rng(5).uniform(0.1, 3.0, tc.shape)
    scaled = WindowedMisfit(d=m.d, omega=m.omega * tw[..., None], dt=m.dt)
    assert abs(m.cost(case.s, trace_weights=tw) / scaled.cost(case.s) - 1) <= 1e-13
    assert rel(m.residual_weight(case.s, trace_weights=tw), scaled.residual_weight(case.s)) <= 1e-15
    cols = np.flatnonzero(m.mask.reshape(-1, m.mask.shape[-1]).any(axis=0))
    assert len(cols) < m.d.shape[-1]
    cut = m.take(cols)
    assert abs(cut.cost(case.s[..., cols]) / m.cost(case.s) - 1) <= 1e-13
    assert rel(cut.residual_weight(case.s[..., cols]), m.residual_weight(case.s)[..., cols]) == 0
