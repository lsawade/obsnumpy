"""Tests of the pipeline's adjoints, the tau-derivative and the absolute window.

- Dot products: <A x, g> = <x, A^T g> for every step, for two chains step by
  step and compiled, on numpy and CuPy.
- Derivatives: Interpolate's and Shift's against central differences, with
  second-order convergence (error ratio ~4 as h halves).
- FixedWindow.from_trim against obspy's Trace.trim, after obsproclib's padding.
- The absolute frame: P . S(tau) at tau = 0 reproduces obspy's processing of
  the trimmed trace, and is smooth in tau, where production's cut is not.
"""
import os

import numpy as np
import obspy
import pytest
from obspy import Trace, UTCDateTime

from obsnumpy.pipeline import (Bandpass, Convolve, Cut, Detrend, FixedWindow, Highpass,
                               Interpolate, Lowpass, Pipeline, PreFilt, RotateNE2RT, Shift,
                               StepConvolve, Taper)
from obsnumpy.pipeline.window import round_away

TESTDATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TOL_DOT = 1e-12                  # |<Ax,g> - <x,A^T g>| / (|Ax| |g|)

DT, N = 1.0, 600
RNG = np.random.default_rng(0)
BAZ = RNG.uniform(0, 360, 5)
GAUSS = np.exp(-0.5 * (np.arange(-8, 9) / 3.0) ** 2)
GAUSS /= GAUSS.sum()


def get(a):
    return a.get() if hasattr(a, "get") else a


def dot_error(apply, adjoint, x, g):
    y = apply(x)
    xp_sum = (y * g).sum()
    lhs, rhs = float(get(xp_sum)), float(get((x * adjoint(g)).sum()))
    scale = float(get((y * y).sum())) ** 0.5 * float(get((g * g).sum())) ** 0.5
    return abs(lhs - rhs) / scale


def steps():
    return [Detrend("linear"), Detrend("demean"), Taper(0.05), Taper(0.2, side="left"),
            Bandpass(0.05, 0.2, 4, True), Bandpass(0.05, 0.2, 4, False),
            Highpass(0.02, 4, True), Lowpass(0.2, 6, False),
            PreFilt((0.01, 0.02, 0.2, 0.3)), PreFilt((0.01, 0.02, 0.4, 0.7), nyquist="real"),
            Cut(50, 500), Cut(-100),
            Interpolate(0.37, 900, 3.3), Interpolate(0.37, 900, 3.3, deriv=1),
            Shift(1.7, 0.5, 1000, 3.3), Shift(1.7, 0.5, 1000, 3.3).d_dtau(),
            RotateNE2RT(BAZ), Convolve(RNG.standard_normal(7)), StepConvolve(GAUSS)]


def ids(s):
    return type(s).__name__


def chains():
    """A cmtinversion-like and a cmt3d-like chain on 1 s samples."""
    return [Pipeline([Detrend("linear"), Taper(0.05), Bandpass(0.02, 0.1, 4, True),
                      Interpolate(2.0, 250, 10.3), RotateNE2RT(BAZ)]),
            Pipeline([Cut(20, 580), Detrend("linear"), Detrend("demean"), Taper(0.05),
                      PreFilt((0.005, 0.01, 0.1, 0.15)),
                      Detrend("linear"), Detrend("demean"), Taper(0.05),
                      Interpolate(2.0, 250, 10.3), RotateNE2RT(BAZ)])]


# ----------------------------------------------------------------------------- dot products

@pytest.mark.parametrize("step", steps(), ids=ids)
def test_dot_step(step):
    x = RNG.standard_normal((5, 3, N))
    g = RNG.standard_normal(step(x, DT).shape)
    assert dot_error(lambda v: step(v, DT), lambda v: step.adjoint(v, DT, N), x, g) <= TOL_DOT


@pytest.mark.parametrize("p", chains())
def test_dot_chain(p):
    x = RNG.standard_normal((5, 3, N))
    g = RNG.standard_normal(p(x, DT)[0].shape)
    assert dot_error(lambda v: p(v, DT)[0], lambda v: p.adjoint(v, DT, N), x, g) <= TOL_DOT


@pytest.mark.parametrize("p", chains())
def test_dot_compiled(p):
    timepart, _ = p.split(len(p) - 1)
    run = timepart.compiled(N, DT)
    x = RNG.standard_normal((5, 3, N))
    g = RNG.standard_normal(run(x).shape)
    assert dot_error(run, run.adjoint, x, g) <= TOL_DOT
    assert np.allclose(run.adjoint(g), timepart.adjoint(g, DT, N), rtol=0, atol=1e-12 *
                       np.abs(run.adjoint(g)).max())


@pytest.mark.gpu
@pytest.mark.parametrize("step", steps(), ids=ids)
def test_dot_step_cupy(cp, step):
    x = cp.asarray(RNG.standard_normal((5, 3, N)))
    g = cp.asarray(RNG.standard_normal(step(x, DT).shape))
    assert isinstance(step.adjoint(g, DT, N), cp.ndarray)
    assert dot_error(lambda v: step(v, DT), lambda v: step.adjoint(v, DT, N), x, g) <= TOL_DOT


@pytest.mark.gpu
@pytest.mark.parametrize("p", chains())
def test_dot_chain_cupy(cp, p):
    x = cp.asarray(RNG.standard_normal((5, 3, N)))
    g = cp.asarray(RNG.standard_normal(p(x, DT)[0].shape))
    assert dot_error(lambda v: p(v, DT)[0], lambda v: p.adjoint(v, DT, N), x, g) <= TOL_DOT
    timepart, _ = p.split(len(p) - 1)
    run = timepart.compiled(N, DT, xp=cp)
    g = cp.asarray(RNG.standard_normal(run(x).shape))
    assert dot_error(run, run.adjoint, x, g) <= TOL_DOT


# ----------------------------------------------------------------------------- PreFilt's Nyquist

def test_prefilt_abs_is_nonlinear_past_nyquist():
    """obspy's abs() of the Nyquist bin is nonlinear once the taper reaches
    Nyquist (here 0.5 Hz): matrix() and adjoint() must refuse it there."""
    x, z = RNG.standard_normal((2, 3, N))
    past, below = PreFilt((0.01, 0.02, 0.3, 0.7)), PreFilt((0.01, 0.02, 0.3, 0.45))
    assert not past.is_linear(DT, N) and below.is_linear(DT, N)
    err = np.abs(past(x - z, DT) - past(x, DT) + past(z, DT)).max() / np.abs(past(x, DT)).max()
    assert err > 1e-6                                        # really nonlinear
    with pytest.raises(ValueError, match="PreFilt"):
        Pipeline([Detrend(), past]).matrix(N, DT)
    with pytest.raises(ValueError, match="PreFilt"):
        Pipeline([Detrend(), past]).adjoint(x, DT, N)
    real = PreFilt(past.flimit, nyquist="real")
    assert real.is_linear(DT, N) and real.linear and not past.linear
    # where "abs" is linear the two agree
    assert np.array_equal(below(x, DT), PreFilt(below.flimit, nyquist="real")(x, DT))


# ----------------------------------------------------------------------------- derivatives

def smooth(shape, dt=DT, fmax=0.2, seed=1):
    x = np.random.default_rng(seed).standard_normal(shape)
    return Pipeline([Lowpass(fmax, 6, True)])(x, dt)[0]


def cd_errors(f, d, h):
    """Central-difference errors at h and h/2, relative to the derivative's peak."""
    e = []
    for hh in (h, h / 2):
        cd = (f(hh) - f(-hh)) / (2 * hh)
        e.append(np.abs(cd - d).max() / np.abs(d).max())
    return e


def assert_second_order(e):
    assert e[0] < 1e-2 and 3.5 < e[0] / e[1] < 4.5, e


# The derivative tests sample at 0.8 s, not 1 s, so that a missing 1/dt shows.
DTD = 0.8


def test_interpolate_derivative():
    """Output positions offset/dt + 0.5 j sit 0.3 and 0.8 samples past the
    input grid, so +-h never crosses a sample and the error is O(h^2)."""
    x = smooth((4, 3, N), dt=DTD, fmax=0.2)
    off = 3.3 * DTD
    d = Interpolate(0.5 * DTD, 900, off, deriv=1)(x, DTD)
    e = cd_errors(lambda h: Interpolate(0.5 * DTD, 900, off + h)(x, DTD), d, 0.05 * DTD)
    assert_second_order(e)


def test_interpolate_derivative_is_continuous_across_samples():
    """K and K' vanish at |d| = a, so L and dL/doffset do not jump when an
    output position crosses an input sample (here at offset 3.0)."""
    x = smooth((4, 3, N))
    for deriv in (0, 1):
        lo = Interpolate(0.5, 900, 3.0 - 1e-9, deriv=deriv)(x, DT)
        hi = Interpolate(0.5, 900, 3.0 + 1e-9, deriv=deriv)(x, DT)
        assert np.abs(hi - lo).max() <= 1e-7 * np.abs(lo).max()


def test_shift_derivative_and_sign():
    x = smooth((4, 3, N), dt=DTD, fmax=0.2)
    off = 3.3 * DTD + 1.7                 # positions 3.3 + 0.5 j samples, as above
    s = Shift(tau=1.7, new_dt=0.5 * DTD, npts=900, offset_s=off)
    d = s.d_dtau()(x, DTD)
    e = cd_errors(lambda h: Shift(1.7 + h, 0.5 * DTD, 900, off)(x, DTD), d, 0.05)
    assert_second_order(e)
    # a positive tau delays: shifting by a whole sample is the input one sample later
    y0 = Shift(0.0, DT, 500, 20.0)(x, DT)
    y1 = Shift(DT, DT, 500, 20.0)(x, DT)
    assert np.allclose(y1[..., 1:], y0[..., :-1], rtol=0, atol=1e-13 * np.abs(y0).max())
    assert s.out_start(DT) == off - 1.7


def test_window_shift_operator_derivative():
    """The window's grid (1 s) differs from the trace's (0.8 s)."""
    x = smooth((4, 3, N), dt=DTD, fmax=0.2)
    w = FixedWindow(first_time=50.0, dt=1.0, n=400)
    S, dS = w.shift_operator(trace_start=7.0, n_in=N, dt_in=DTD, tau=2.3)
    flat = x.reshape(-1, N).T
    d = (dS @ flat).T
    e = cd_errors(lambda h: (w.shift_operator(7.0, N, DTD, tau=2.3 + h)[0] @ flat).T, d, 0.05)
    assert_second_order(e)
    assert np.array_equal((S @ flat).T, w.shift_step(7.0, 2.3)(x, DTD).reshape(-1, 400))


# ----------------------------------------------------------------------------- FixedWindow

T_REF = UTCDateTime(2023, 4, 14, 9, 55, 51.48)


def obspy_trim(dt, start, end, pad):
    """obsproclib's flex_cut_trace: pad the cut times, then Trace.trim."""
    tr = Trace(np.zeros(3000), header=dict(delta=dt, starttime=T_REF))
    tr.trim(start - pad * dt, end + pad * dt)
    return tr.stats.starttime, tr.stats.npts


# cut times landing just below, on and just above half a sample, and off it
EDGES = [0.5 - 1e-6, 0.5, 0.5 + 1e-6, 0.3, 0.0]


@pytest.mark.parametrize("dt", [3.92, 0.5, 0.05])
@pytest.mark.parametrize("fs", EDGES)
@pytest.mark.parametrize("fe", EDGES)
def test_from_trim_is_obspy(dt, fs, fe):
    start, end = T_REF + (37 + fs) * dt, T_REF + (811 + fe) * dt
    w = FixedWindow.from_trim(T_REF, dt, start, end, pad_samples=5)
    assert (w.first_time, w.n) == obspy_trim(dt, start, end, 5)


def test_from_trim_cmt3d_window():
    """cmt3d's cut, [centroid + 100, + 3300] s padded by 5 samples, on gf3d's
    grid starting 200 s before the centroid: obspy keeps 827 samples from
    82.24 s; floor/ceil would keep 826."""
    w = FixedWindow.from_trim(T_REF - 200.0, 3.92, T_REF + 100, T_REF + 3300, pad_samples=5)
    assert w.n == 827 and w.offset_from(T_REF) == pytest.approx(82.24, abs=1e-9)
    assert (w.first_time, w.n) == (lambda t: (t.stats.starttime, t.stats.npts))(
        Trace(np.zeros(1000), header=dict(delta=3.92, starttime=T_REF - 200.0)).trim(
            T_REF + 100 - 5 * 3.92, T_REF + 3300 + 5 * 3.92))


def test_from_trim_floats_and_errors():
    w = FixedWindow.from_trim(0.0, 0.5, 10.25, 20.0)         # exactly half: away from zero
    assert (w.first_time, w.n) == (10.5, 20)
    assert round_away(2.5) == 3 and round_away(-2.5) == -3 and round_away(2.4) == 2
    with pytest.raises(ValueError, match="before the trace"):
        FixedWindow.from_trim(0.0, 0.5, 1.0, 20.0, pad_samples=5)
    with pytest.raises(ValueError, match="after the trace"):
        FixedWindow.from_trim(0.0, 0.5, 10.0, 20.0, trace_npts=30)


def test_indices_in():
    w = FixedWindow.from_trim(T_REF, 0.5, T_REF + 10.2, T_REF + 20.0)
    assert w.indices_in(T_REF) == slice(20, 20 + w.n)
    assert w.indices_in(T_REF + 1.0) == slice(18, 18 + w.n)
    with pytest.raises(ValueError, match="off the window's grid"):
        w.indices_in(T_REF + 0.1)
    with pytest.raises(ValueError, match="sampled at"):
        w.indices_in(T_REF, dt=0.25)


# ----------------------------------------------------------------------------- the absolute frame

STATIONS = ["G.ECH", "II.BFO", "IU.GRFO", "IU.PAB"]


@pytest.fixture(scope="module")
def raw():
    """The four NEZ raw test stations, 20 Hz, from their own (common) start."""
    st = obspy.read(os.path.join(TESTDATA, "raw", "*.mseed"))
    st.merge()
    x = np.empty((len(STATIONS), 3, 6000))
    for i, code in enumerate(STATIONS):
        net, sta = code.split(".")
        for c, comp in enumerate("NEZ"):
            x[i, c] = st.select(network=net, station=sta, component=comp)[0].data[:6000]
    return x, st.select(network="G", station="ECH")[0].stats.starttime


def window_chain(start_offset, n_out=300, new_dt=0.5, like_obspy=True, t_win=None, t_out=None):
    pre = (0.02, 0.04, 0.5, 0.7)
    interp = (Interpolate.from_utc(t_win, t_out, new_dt, n_out, like_obspy=like_obspy)
              if t_win is not None else Interpolate(new_dt, n_out, start_offset))
    return Pipeline([Detrend("linear"), Detrend("demean"), Taper(0.05), PreFilt(pre),
                     Detrend("linear"), Detrend("demean"), Taper(0.05), interp])


def obspy_window_chain(x, t0, start, end, t_out, n_out=300, new_dt=0.5):
    out = np.empty(x.shape[:-1] + (n_out,))
    for idx in np.ndindex(x.shape[:-1]):
        tr = Trace(x[idx].copy(), header=dict(delta=0.05, starttime=t0))
        tr.trim(start - 5 * 0.05, end + 5 * 0.05)
        tr.detrend("linear"); tr.detrend("demean"); tr.taper(0.05, type="cosine")
        n = tr.stats.npts
        nfft = __import__("obspy").signal.util._npts2nfft(n)
        X = np.fft.rfft(tr.data, n=nfft)
        X *= __import__("obspy").signal.invsim.cosine_sac_taper(
            np.linspace(0, 10.0, nfft // 2 + 1), flimit=(0.02, 0.04, 0.5, 0.7))
        X[-1] = abs(X[-1]) + 0j
        tr.data = np.fft.irfft(X)[:n]
        tr.detrend("linear"); tr.detrend("demean"); tr.taper(0.05, type="cosine")
        tr.interpolate(1 / new_dt, method="lanczos", a=12, starttime=t_out, npts=n_out)
        out[idx] = tr.data
    return out


def test_window_reproduces_obspy_at_tau0(raw):
    """At tau = 0 both routes agree with obspy's trim and processing to
    1e-12: the data route takes a slice of the trace, the synthetic route
    interpolates it with S(0). S(0) is the identity here, because the window
    lies on the trace's grid to the nanosecond. Off that grid it would read
    ~1e-9 / dt samples away."""
    x, t0 = raw
    start, end, t_out = t0 + 30.013, t0 + 250.0, t0 + 40.0
    w = FixedWindow.from_trim(t0, 0.05, start, end, pad_samples=5)
    ref = obspy_window_chain(x, t0, start, end, t_out)
    p = window_chain(None, t_win=w.first_time, t_out=t_out)
    y_data = p(x[..., w.indices_in(t0)], 0.05)[0]
    rel = lambda a, b: np.abs(a - b).max() / np.abs(b).max()
    assert rel(y_data, ref) <= 1e-12
    y_syn = p(w.shift_step(t0, tau=0.0)(x, 0.05), 0.05)[0]
    assert rel(y_syn, ref) <= 1e-12


def test_smooth_in_tau():
    """A fixed functional of P . S(tau) . x over a 0.05 s grid of tau on gf3d's
    3.92 s grid: max/median of the second differences <= 3 (the study measured
    1.8). Production instead shifts the trace's start time and trims at the
    nearest sample, so its window jumps by a whole sample as tau crosses half
    a sample: its ratio is far larger (the study measured 106)."""
    dt, n = 3.92, 1000
    x = smooth((3, n), dt=dt, fmax=0.02, seed=3)
    t_first = T_REF - 200.0
    start, end = T_REF + 100.0, T_REF + 3300.0
    w = FixedWindow.from_trim(t_first, dt, start, end, pad_samples=5)
    p = window_chain(None, n_out=1601, new_dt=2.0, t_win=w.first_time, t_out=start)
    p = Pipeline([PreFilt((0.003, 0.004, 0.0111, 0.0139)) if isinstance(s, PreFilt) else s
                  for s in p.steps])
    run = p.compiled(w.n, dt)
    g = np.random.default_rng(4).standard_normal((3, 1601))
    taus = np.arange(0.0, 8.0001, 0.05)
    f = np.array([np.sum(g * run(w.shift_step(t_first, tau)(x, dt))) for tau in taus])
    d2 = np.abs(np.diff(f, 2))
    assert d2.max() / np.median(d2) <= 3.0

    # production: the cut snaps to the shifted trace's samples
    def production(tau):
        wp = FixedWindow.from_trim(t_first + tau, dt, start, end, pad_samples=5)
        y = x[..., wp.indices_in(t_first + tau)]
        q = Pipeline(p.steps[:-1] + [Interpolate(2.0, 1601, float(start - wp.first_time))])
        return np.sum(g * q(y, dt)[0])
    fp = np.array([production(tau) for tau in taus])
    d2p = np.abs(np.diff(fp, 2))
    assert d2p.max() / np.median(d2p) > 10 * 3.0
