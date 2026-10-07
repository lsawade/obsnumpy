"""Tests of obsnumpy.pipeline against obspy (and obsproclib, when importable).

The tests come in tiers:

- T1: each step, and two whole chains, against obspy's Trace methods, on the
  raw test traces;
- T2: CuPy against numpy (skipped without CuPy or a GPU);
- T3: the compiled pipeline (one matrix) against the step-by-step pipeline;
- T4: linearity of every linear step.

The probes at the end show that the T1 tolerance catches the two easy
mistakes: the taper's rounding convention, and a filter run forwards only.

Errors are relative to each trace's peak, worst trace reported.
"""
import contextlib
import io
import os
import types

import numpy as np
import obspy
import pytest
from obspy import Trace
from obspy.signal.invsim import cosine_sac_taper
from obspy.signal.rotate import rotate_ne_rt
from obspy.signal.util import _npts2nfft

from obsnumpy import Dataset, Meta, Stations
from obsnumpy.pipeline import (Bandpass, Convolve, Cut, Detrend, Highpass, Interpolate,
                               Lowpass, Pipeline, PreFilt, RotateNE2RT, Step, StepConvolve,
                               Taper)
from obsnumpy.pipeline import tables

TESTDATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# The four raw stations with all of N, E and Z (G.SSB has no N), 20 Hz.
STATIONS = ["G.ECH", "II.BFO", "IU.GRFO", "IU.PAB"]
FIRST, NT, DT = 1000, 3000, 0.05          # samples 50 s .. 200 s of the record
BAZ = np.array([12.5, 97.0, 201.3, 333.3])
TOL = 1e-12                                # against obspy
TOL_SAME = 1e-13                           # CuPy against numpy, compiled against steps

# The IIR filters on these 20 Hz traces: the Butterworth band 0.05-0.5 Hz has
# poles at |p| = 0.995, close to the unit circle, and rounding in the
# recursion grows with 1 / (1 - |p|). Measured against the same sections in
# long double (80-bit):
# - scipy's sosfilt is off by 6.3e-14 of the peak, step by step, and 7.1e-14
#   compiled (P is built by the same sosfilt). Two float64 routes can differ
#   by about twice that.
# - cupyx's sosfilt (a block-parallel recursion) is off by 6.6e-12.
# At cmtinversion's band (1/150-1/40 Hz at dt = 3.92 s, |p| = 0.962) all three
# are off by 3-4e-14, and test_cupy_iir_at_long_periods holds CuPy to
# TOL_SAME there.
TOL_IIR = 3e-13
TOL_CUPY_IIR = 3e-11

# The cmtinversion chain, scaled to these traces: detrend, taper, Butterworth
# band, Lanczos, rotation.
BP = dict(freqmin=0.05, freqmax=0.5, corners=4, zerophase=True)
BP_START, BP_DT, BP_NPTS = 10.0, 0.25, 520

# The cmt3d chain, scaled: cut, detrend, demean, taper, pre_filt, detrend,
# demean, taper, Lanczos, rotation. The output start is off the input grid.
C0, C1 = 200, 2800
PRE = (0.02, 0.04, 0.5, 0.7)
C3_START, C3_DT, C3_NPTS = 20.123, 0.5, 200


# ----------------------------------------------------------------------------- helpers

def rel(a, b):
    """max over traces of max|a - b| / max|b|."""
    a, b = np.asarray(a), np.asarray(b)
    d = np.abs(a - b).max(axis=-1)
    p = np.abs(b).max(axis=-1)
    return float((d / np.where(p > 0, p, 1.0)).max())


def get(a):
    return a.get() if hasattr(a, "get") else a


def obspy_filter_trace(tr, flimit):
    """obsproclib's filter_trace, written out with obspy's own pieces."""
    n = tr.stats.npts
    nfft = _npts2nfft(n)
    freqs = np.linspace(0, 1.0 / (tr.stats.delta * 2.0), nfft // 2 + 1)
    X = np.fft.rfft(tr.data.astype(np.float64), n=nfft)
    X *= cosine_sac_taper(freqs, flimit=flimit)
    X[-1] = abs(X[-1]) + 0.0j
    tr.data = np.fft.irfft(X)[:n]


def obspy_each(x, t0, fn, dt=DT):
    """Apply ``fn(tr)`` to each trace of ``x`` (sta, comp, t) as an obspy Trace."""
    out = []
    for i in range(x.shape[0]):
        row = []
        for c in range(x.shape[1]):
            tr = Trace(x[i, c].copy(), header=dict(delta=dt, starttime=t0))
            fn(tr)
            row.append(tr.data)
        out.append(row)
    return np.array(out)


def obspy_rotate(y, baz):
    out = y.copy()
    for i in range(y.shape[0]):
        out[i, 0], out[i, 1] = rotate_ne_rt(y[i, 0], y[i, 1], baz[i])
    return out


@pytest.fixture(scope="module")
def raw():
    """(4 stations, NEZ, NT) float64 counts, one common start time ``t0``."""
    st = obspy.read(os.path.join(TESTDATA, "raw", "*.mseed"))
    st.merge()
    x = np.empty((len(STATIONS), 3, NT))
    for i, code in enumerate(STATIONS):
        net, sta = code.split(".")
        for c, comp in enumerate("NEZ"):
            tr = st.select(network=net, station=sta, component=comp)[0]
            x[i, c] = tr.data[FIRST:FIRST + NT].astype(np.float64)
    t0 = st.select(network="G", station="ECH")[0].stats.starttime + FIRST * DT
    return types.SimpleNamespace(x=x, t0=t0)


def bandpass_chain(t0, like_obspy=True):
    interp = Interpolate.from_utc(t0, t0 + BP_START, BP_DT, BP_NPTS, a=12, like_obspy=like_obspy)
    return Pipeline([Detrend("linear"), Taper(0.05), Bandpass(**BP), interp, RotateNE2RT(BAZ)])


def bandpass_chain_obspy(x, t0):
    def fn(tr):
        tr.detrend("linear")
        tr.taper(max_percentage=0.05, type="cosine")
        tr.filter("bandpass", **BP)
        tr.interpolate(1.0 / BP_DT, method="lanczos", a=12, starttime=t0 + BP_START,
                       npts=BP_NPTS)
    return obspy_rotate(obspy_each(x, t0, fn), BAZ)


def cmt3d_chain(t0, like_obspy=True):
    interp = Interpolate.from_utc(t0 + C0 * DT, t0 + C3_START, C3_DT, C3_NPTS, a=12,
                                  like_obspy=like_obspy)
    return Pipeline([Cut(C0, C1),
                     Detrend("linear"), Detrend("demean"), Taper(0.05),
                     PreFilt(PRE),
                     Detrend("linear"), Detrend("demean"), Taper(0.05),
                     interp, RotateNE2RT(BAZ)])


def cmt3d_chain_obspy(x, t0):
    def fn(tr):
        tr.detrend("linear"); tr.detrend("demean"); tr.taper(max_percentage=0.05, type="cosine")
        obspy_filter_trace(tr, PRE)
        tr.detrend("linear"); tr.detrend("demean"); tr.taper(max_percentage=0.05, type="cosine")
        tr.interpolate(1.0 / C3_DT, method="lanczos", a=12, starttime=t0 + C3_START,
                       npts=C3_NPTS)
    return obspy_rotate(obspy_each(x[..., C0:C1], t0 + C0 * DT, fn), BAZ)


# ----------------------------------------------------------------------------- T1, steps

@pytest.mark.parametrize("kind", ["linear", "demean"])
def test_detrend(raw, kind):
    y, _ = Pipeline([Detrend(kind)])(raw.x, DT)
    assert rel(y, obspy_each(raw.x, raw.t0, lambda tr: tr.detrend(kind))) <= TOL


@pytest.mark.parametrize("kw", [dict(max_percentage=0.05),
                                dict(max_percentage=0.1, side="left"),
                                dict(max_percentage=0.2, side="right"),
                                dict(max_percentage=0.3, max_length=3.0),
                                dict(max_percentage=0.5)])
def test_taper(raw, kw):
    y = Taper(**kw)(raw.x, DT)
    ref = obspy_each(raw.x, raw.t0, lambda tr: tr.taper(type="cosine", **kw))
    assert rel(y, ref) <= TOL


@pytest.mark.parametrize("step, args", [
    (Bandpass(0.05, 0.5, 4, True), ("bandpass", dict(freqmin=0.05, freqmax=0.5, corners=4,
                                                      zerophase=True))),
    (Bandpass(0.05, 0.5, 3, False), ("bandpass", dict(freqmin=0.05, freqmax=0.5, corners=3,
                                                       zerophase=False))),
    (Highpass(0.1, 4, True), ("highpass", dict(freq=0.1, corners=4, zerophase=True))),
    (Lowpass(1.0, 6, False), ("lowpass", dict(freq=1.0, corners=6, zerophase=False))),
])
def test_filters(raw, step, args):
    x, _ = Pipeline([Detrend("linear"), Taper(0.05)])(raw.x, DT)
    ref = obspy_each(x, raw.t0, lambda tr: tr.filter(args[0], **args[1]))
    assert rel(step(x, DT), ref) <= TOL


def test_bandpass_above_nyquist_is_highpass(raw):
    with pytest.warns(UserWarning, match="Nyquist"):
        y = Bandpass(0.1, 20.0, 4, True)(raw.x, DT)
    with pytest.warns(UserWarning, match="Nyquist"):
        ref = obspy_each(raw.x, raw.t0, lambda tr: tr.filter("bandpass", freqmin=0.1,
                                                              freqmax=20.0, zerophase=True))
    assert rel(y, ref) <= TOL


@pytest.mark.parametrize("n", list(range(1, 3000, 37)) + [2500, 2501, 4999, 5003, 7919,
                                                           10007, 65537, 199999])
def test_npts2nfft(n):
    assert tables.npts2nfft(n) == _npts2nfft(n)


# 2503 is prime, so its nfft is nudged; the second flimit reaches past Nyquist
# (10 Hz), where the taper is 0.5 and the Nyquist bin's real part counts.
@pytest.mark.parametrize("flimit", [PRE, (0.02, 0.04, 5.0, 15.0)])
@pytest.mark.parametrize("n", [NT, NT - 1, 2503])
def test_prefilt(raw, n, flimit):
    x = raw.x[..., :n]
    ref = obspy_each(x, raw.t0, lambda tr: obspy_filter_trace(tr, flimit))
    assert rel(PreFilt(flimit)(x, DT), ref) <= TOL


@pytest.mark.production
def test_prefilt_against_obsproclib(raw):
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from obsproclib.process import filter_trace
    except ImportError:
        pytest.skip("obsproclib is not importable")
    ref = obspy_each(raw.x, raw.t0, lambda tr: filter_trace(tr, list(PRE)))
    assert rel(PreFilt(PRE)(raw.x, DT), ref) <= TOL


def test_cut(raw):
    y = Cut(C0, C1)(raw.x, DT)
    ref = obspy_each(raw.x, raw.t0, lambda tr: tr.trim(raw.t0 + C0 * DT,
                                                       raw.t0 + (C1 - 1) * DT))
    assert np.array_equal(y, ref)
    assert Cut(C0, C1).out_npts(NT) == C1 - C0 and Cut(C0, C1).out_start(DT) == C0 * DT
    assert Cut(-100).out_npts(NT) == 100 and Cut(stop=-1).out_npts(NT) == NT - 1


@pytest.mark.parametrize("start, new_dt, npts", [(5.0, 0.25, 400),     # on the grid
                                                 (5.013, 0.25, 400),   # off the grid
                                                 (0.0, 0.05, NT),      # identity
                                                 (7.77, 0.0333, 2000)])
def test_interpolate(raw, start, new_dt, npts):
    x, _ = Pipeline([Lowpass(1.5, 6, True)])(raw.x, DT)
    step = Interpolate.from_utc(raw.t0, raw.t0 + start, new_dt, npts, a=12, like_obspy=True)
    ref = obspy_each(x, raw.t0, lambda tr: tr.interpolate(1.0 / new_dt, method="lanczos", a=12,
                                                          starttime=raw.t0 + start, npts=npts))
    assert rel(step(x, DT), ref) <= TOL


def test_rotate(raw):
    y = RotateNE2RT(BAZ)(raw.x, DT)
    assert rel(y, obspy_rotate(raw.x, BAZ)) <= TOL
    assert RotateNE2RT(BAZ).out_components(["N", "E", "Z"]) == ["R", "T", "Z"]
    with pytest.raises(ValueError):
        RotateNE2RT(BAZ).out_components(["Z", "N", "E"])
    # components in another order
    zne = raw.x[:, [2, 0, 1]]
    y2 = RotateNE2RT(BAZ, n=1, e=2)(zne, DT)
    assert np.array_equal(y2[:, [1, 2, 0]], y)


# ----------------------------------------------------------------------------- T1, chains

def test_bandpass_chain(raw):
    y, dt = bandpass_chain(raw.t0)(raw.x, DT)
    assert dt == BP_DT
    assert rel(y, bandpass_chain_obspy(raw.x, raw.t0)) <= TOL


def test_cmt3d_chain(raw):
    y, dt = cmt3d_chain(raw.t0)(raw.x, DT)
    assert dt == C3_DT
    assert rel(y, cmt3d_chain_obspy(raw.x, raw.t0)) <= TOL


def test_obspy_offset_is_the_gap(raw):
    """The prototype's 3.0e-9 against obspy came from obspy's offset arithmetic.

    ``Trace.interpolate`` takes the offset as a difference of two float64 epoch
    timestamps (``starttime.timestamp - stats.starttime.timestamp``), each
    rounded to half a unit in the last place: ~1.2e-7 s in 2004. With that
    same offset the chain agrees to 1e-12 (test_cmt3d_chain). With the exact
    offset (UTCDateTime's nanoseconds) it differs by the offset error times
    the signal's slope, and by no more.
    """
    old, new = raw.t0 + C0 * DT, raw.t0 + C3_START
    delta = (new.timestamp - old.timestamp) - (new.ns - old.ns) * 1e-9
    ulp = np.spacing(new.timestamp)
    assert 0 < abs(delta) <= ulp                     # only the timestamps' rounding

    ref = cmt3d_chain_obspy(raw.x, raw.t0)
    y_exact, _ = cmt3d_chain(raw.t0, like_obspy=False)(raw.x, DT)
    slope = np.abs(np.diff(ref, axis=-1)).max(axis=-1) / C3_DT     # per trace, per second
    bound = float((abs(delta) * slope / np.abs(ref).max(axis=-1)).max())
    gap = rel(y_exact, ref)
    assert gap > 1e3 * TOL                           # the gap is real, and this is it:
    assert gap <= 1.5 * bound + TOL


# ----------------------------------------------------------------------------- convolutions

def _kernels():
    k = np.exp(-0.5 * (np.arange(-8, 9) / 3.0) ** 2)
    rng = np.random.default_rng(1)
    return [np.array([1.0]), k / k.sum(), rng.standard_normal(7)]   # incl. an asymmetric one


@pytest.mark.parametrize("w", _kernels())
def test_convolve(raw, w):
    K = (len(w) - 1) // 2
    ref = np.apply_along_axis(lambda t: np.convolve(t, w)[K:K + NT], -1, raw.x)
    assert rel(Convolve(w)(raw.x, DT), ref) <= TOL


@pytest.mark.parametrize("w", _kernels())
def test_step_convolve(raw, w):
    """Against np.convolve with the explicit step response: dt*w on -K..K, dt after."""
    K = (len(w) - 1) // 2
    h = DT * np.concatenate([w, np.ones(NT - K - 1)])                 # taps -K .. NT-1
    ref = np.apply_along_axis(lambda t: np.convolve(t, h)[K:K + NT], -1, raw.x)
    assert rel(StepConvolve(w)(raw.x, DT), ref) <= TOL


def test_kernel_must_be_odd():
    with pytest.raises(ValueError):
        StepConvolve(np.ones(4))


# ----------------------------------------------------------------------------- pipeline

def test_leading_axes_are_free(raw):
    p = cmt3d_chain(raw.t0)
    y, _ = p(raw.x, DT)
    y2, _ = p(np.stack([raw.x, -2.0 * raw.x]), DT)
    assert np.array_equal(y2[0], y) and rel(y2[1], -2.0 * y) <= TOL_SAME


def test_integer_input_is_float64(raw):
    """A rotation writes into a copy of its input: on integers it would truncate."""
    p = Pipeline([RotateNE2RT(BAZ)])
    y, _ = p(raw.x.astype(np.int64), DT)
    assert y.dtype == np.float64 and np.array_equal(y, p(raw.x, DT)[0])


def test_out_and_split(raw):
    p = cmt3d_chain(raw.t0)
    assert p.out_dt(DT) == C3_DT and p.out_npts(NT) == C3_NPTS
    assert p.out_start(DT) == pytest.approx(C3_START, abs=1e-6)
    a, b = p.split(4)
    assert len(a) == 4 and len(b) == len(p) - 4
    ya, dta = a(raw.x, DT)
    assert np.array_equal(b(ya, dta)[0], p(raw.x, DT)[0])
    # PreFilt(nyquist="abs") is linear here (its taper vanishes at Nyquist), not always
    assert p.is_linear(NT, DT) and not p.linear


def test_matrix_refuses():
    class Square(Step):
        linear = False

        def __call__(self, x, dt):
            return x * x

    with pytest.raises(ValueError, match="RotateNE2RT"):
        Pipeline([Detrend(), RotateNE2RT(BAZ)]).matrix(10, DT)
    with pytest.raises(ValueError, match="Square"):
        Pipeline([Square()]).matrix(10, DT)
    assert not Pipeline([Square()]).linear


def test_dataset_apply(raw):
    meta = Meta(starttime=raw.t0, npts=NT, delta=DT, components=["N", "E", "Z"],
                stations=Stations(codes=np.array(STATIONS)))
    ds = Dataset(data=raw.x.copy(), meta=meta)
    p = cmt3d_chain(raw.t0)
    out = ds.apply(p)
    assert out is ds
    assert np.array_equal(ds.data, p(raw.x, DT)[0])
    assert ds.meta.delta == C3_DT and ds.meta.npts == C3_NPTS
    assert ds.meta.components == ["R", "T", "Z"]
    assert abs(ds.meta.starttime - (raw.t0 + C3_START)) < 1e-6


# ----------------------------------------------------------------------------- T3, compiled

# The cmt3d chain filters by FFT; its two routes agree to 4.6e-15. The
# bandpass chain's differ by the IIR's rounding (TOL_IIR).
@pytest.mark.parametrize("chain, tol", [(bandpass_chain, TOL_IIR), (cmt3d_chain, TOL_SAME)])
def test_compiled(raw, chain, tol):
    p = chain(raw.t0)
    timepart, comppart = p.split(len(p) - 1)
    run = timepart.compiled(NT, DT)
    assert run.P.shape == (NT, timepart.out_npts(NT)) and run.dt == timepart.out_dt(DT)
    y = comppart(run(raw.x), run.dt)[0]
    assert rel(y, p(raw.x, DT)[0]) <= tol
    with pytest.raises(ValueError):
        run(raw.x[..., :-1])


# ----------------------------------------------------------------------------- T4, linearity

def _is_iir(step):
    return isinstance(step, (Bandpass, Highpass, Lowpass))


def _linear_steps(t0):
    return [Detrend("linear"), Detrend("demean"), Taper(0.05), Bandpass(**BP),
            Highpass(0.1, 4, True), Lowpass(1.0, 4, False), PreFilt(PRE), Cut(C0, C1),
            Interpolate(0.25, 500, 3.3), RotateNE2RT(BAZ), Convolve(_kernels()[2]),
            StepConvolve(_kernels()[1])]


@pytest.mark.parametrize("step", _linear_steps(None), ids=lambda s: type(s).__name__)
def test_linearity(raw, step):
    z = np.random.default_rng(2).standard_normal(raw.x.shape) * np.abs(raw.x).max()
    a, b = 1.7, -0.3
    lhs = step(a * raw.x + b * z, DT)
    rhs = a * step(raw.x, DT) + b * step(z, DT)
    assert step.is_linear(DT, NT) and rel(lhs, rhs) <= (TOL_IIR if _is_iir(step) else TOL_SAME)


# ----------------------------------------------------------------------------- T2, CuPy

@pytest.mark.gpu
@pytest.mark.parametrize("step", _linear_steps(None), ids=lambda s: type(s).__name__)
def test_cupy_steps(raw, cp, step):
    y = step(cp.asarray(raw.x), DT)
    assert isinstance(y, cp.ndarray)
    assert rel(get(y), step(raw.x, DT)) <= (TOL_CUPY_IIR if _is_iir(step) else TOL_SAME)


@pytest.mark.gpu
@pytest.mark.parametrize("step", [Bandpass(1 / 150, 1 / 40, 4, True), Highpass(1 / 150, 4, True),
                                  Lowpass(1 / 40, 4, False)], ids=lambda s: type(s).__name__)
def test_cupy_iir_at_long_periods(raw, cp, step):
    """cmtinversion's band at gf3d's sampling (dt = 3.92 s): CuPy as numpy."""
    dt = 3.92
    x, _ = Pipeline([Detrend("linear"), Taper(0.05)])(raw.x, dt)
    assert rel(get(step(cp.asarray(x), dt)), step(x, dt)) <= TOL_SAME


@pytest.mark.gpu
@pytest.mark.parametrize("chain, tol", [(bandpass_chain, TOL_CUPY_IIR), (cmt3d_chain, TOL_SAME)])
def test_cupy_chains(raw, cp, chain, tol):
    p = chain(raw.t0)
    y, dt = p(cp.asarray(raw.x), DT)
    assert isinstance(y, cp.ndarray) and dt == p.out_dt(DT)
    assert rel(get(y), p(raw.x, DT)[0]) <= tol


@pytest.mark.gpu
@pytest.mark.parametrize("chain, tol", [(bandpass_chain, TOL_CUPY_IIR), (cmt3d_chain, TOL_SAME)])
def test_cupy_compiled(raw, cp, chain, tol):
    p = chain(raw.t0)
    timepart, comppart = p.split(len(p) - 1)
    run = timepart.compiled(NT, DT, xp=cp)
    assert isinstance(run.P, cp.ndarray)
    y = comppart(run(cp.asarray(raw.x)), run.dt)[0]
    assert rel(get(y), p(raw.x, DT)[0]) <= tol


# ----------------------------------------------------------------------------- probes

def test_probe_taper_rounding_fails_T1(raw):
    """obspy's cosine_taper with p < 1 rounds the ramp length up (frac + 0.5);
    Trace.taper passes p = 1, which truncates. The rounding version (the
    prototype's first, wrong convention) must fail T1's tolerance."""
    n = NT
    wlen = int(0.05 * n)
    m = 2 * wlen + 1
    frac = int(m / 2.0 + 0.5)                              # the wrong convention
    i2, i3 = frac - 1, m - frac
    side = np.zeros(m)
    side[:i2 + 1] = 0.5 * (1.0 - np.cos(np.pi * np.arange(0, i2 + 1) / i2))
    side[i2 + 1:i3] = 1.0
    side[i3:] = 0.5 * (1.0 + np.cos(np.pi * (i3 - np.arange(i3, m)) / (m - 1 - i3)))
    wrong = np.hstack((side[:wlen], np.ones(n - 2 * wlen), side[len(side) - wlen:]))
    ref = obspy_each(raw.x, raw.t0, lambda tr: tr.taper(max_percentage=0.05, type="cosine"))
    assert rel(raw.x * wrong, ref) > 1e3 * TOL
    assert rel(Taper(0.05)(raw.x, DT), ref) <= TOL


def test_probe_forward_only_filter_fails_T1(raw):
    ref = obspy_each(raw.x, raw.t0, lambda tr: tr.filter("bandpass", **BP))
    forward_only = Bandpass(BP["freqmin"], BP["freqmax"], BP["corners"], zerophase=False)
    assert rel(forward_only(raw.x, DT), ref) > 1e3 * TOL
