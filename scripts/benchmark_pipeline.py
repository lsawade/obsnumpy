"""Benchmark obsnumpy.pipeline against an obspy trace loop.

The chain is cmt3d's (process.yml): cut 100-3300 s after the centroid with five
samples of padding, detrend, demean, taper, pre_filt, detrend, demean, taper,
Lanczos to 0.5 Hz, rotation to RT. It runs on gf3d's sampling (dt = 3.92 s).

The data are the raw test traces (tests/data/raw, four NEZ stations), tiled to
``--stations`` stations and relabelled at dt = 3.92 s; filters depend only on
the normalised frequency, so the arithmetic is that of a real run. ``--array``
takes a (nsta, 3, npts) .npy array instead.

    numactl --cpunodebind=0 --membind=0 \\
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        python scripts/benchmark_pipeline.py

It prints the thread settings, the devices, and wall and CPU time per call.
"""
import argparse
import os
import platform
import time

import numpy as np
import obspy
from obspy import Trace, UTCDateTime
from obspy.signal.invsim import cosine_sac_taper
from obspy.signal.rotate import rotate_ne_rt
from obspy.signal.util import _npts2nfft

from obsnumpy.pipeline import (Cut, Detrend, Interpolate, Pipeline, PreFilt, RotateNE2RT, Taper,
                               cupy_or_none)

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "tests", "data", "raw", "*.mseed")
DT, NT = 3.92, 3725
PRE = (0.0030, 0.0040, 0.0111, 0.0139)
T0 = UTCDateTime(2023, 4, 14, 9, 55, 51.48) - 200.0          # first sample; centroid at 200 s
C0, C1 = int(np.floor(300 / DT)) - 5, int(np.ceil(3500 / DT)) + 6
NEW_DT, NOUT = 2.0, int(3200 * 0.5) + 1


def load(nsta):
    st = obspy.read(RAW)
    st.merge()
    codes = ["G.ECH", "II.BFO", "IU.GRFO", "IU.PAB"]
    x = np.empty((len(codes), 3, NT))
    for i, code in enumerate(codes):
        net, sta = code.split(".")
        for c, comp in enumerate("NEZ"):
            x[i, c] = st.select(network=net, station=sta, component=comp)[0].data[:NT]
    return np.ascontiguousarray(np.resize(x, (nsta, 3, NT)))


def cmt3d_chain(baz):
    interp = Interpolate.from_utc(T0 + C0 * DT, T0 + 300.0, NEW_DT, NOUT, a=12, like_obspy=True)
    return Pipeline([Cut(C0, C1),
                     Detrend("linear"), Detrend("demean"), Taper(0.05),
                     PreFilt(PRE),
                     Detrend("linear"), Detrend("demean"), Taper(0.05),
                     interp, RotateNE2RT(baz)])


def obspy_loop(x, baz):
    out = np.empty((x.shape[0], 3, NOUT))
    for i in range(x.shape[0]):
        trs = []
        for c in range(3):
            tr = Trace(x[i, c, C0:C1].copy(), header=dict(delta=DT, starttime=T0 + C0 * DT))
            tr.detrend("linear"); tr.detrend("demean"); tr.taper(0.05, type="cosine")
            n = tr.stats.npts
            nfft = _npts2nfft(n)
            X = np.fft.rfft(tr.data, n=nfft)
            X *= cosine_sac_taper(np.linspace(0, 1 / (2 * DT), nfft // 2 + 1), flimit=PRE)
            X[-1] = abs(X[-1]) + 0j
            tr.data = np.fft.irfft(X)[:n]
            tr.detrend("linear"); tr.detrend("demean"); tr.taper(0.05, type="cosine")
            tr.interpolate(1 / NEW_DT, method="lanczos", a=12, starttime=T0 + 300.0, npts=NOUT)
            trs.append(tr.data)
        r, t = rotate_ne_rt(trs[0], trs[1], baz[i])
        out[i] = np.stack([r, t, trs[2]])
    return out


def timeit(f, n, sync=None):
    f()
    if sync:
        sync()
    w, c = time.perf_counter(), time.process_time()
    for _ in range(n):
        f()
    if sync:
        sync()
    return (time.perf_counter() - w) / n * 1e3, (time.process_time() - c) / n * 1e3


def cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stations", type=int, default=142)
    ap.add_argument("--array", help="a (nsta, 3, npts) .npy array to use instead")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--no-obspy", action="store_true", help="skip the obspy loop")
    args = ap.parse_args()

    x = np.load(args.array) if args.array else load(args.stations)
    if x.shape[-1] != NT:
        raise SystemExit(f"the chain is set up for {NT} samples, the array has {x.shape[-1]}")
    baz = np.random.default_rng(0).uniform(0, 360, x.shape[0])
    p = cmt3d_chain(baz)
    timepart, comppart = p.split(len(p) - 1)

    print("obsnumpy.pipeline benchmark: cmt3d chain, dt 3.92 s -> 2 s, rotation to RT")
    print(f"  data      {x.shape[0]} stations x 3 x {NT} samples ({x.shape[0] * 3} traces)")
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        print(f"  {v:<21}{os.environ.get(v, '(unset)')}")
    print(f"  CPU       {cpu_name()}")
    cp = cupy_or_none()
    if cp is not None:
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        print(f"  GPU       {props['name'].decode()} (device {cp.cuda.Device().id}), "
              f"CuPy {cp.__version__}")
    else:
        print("  GPU       none (CuPy or a device not available)")
    print(f"  measured, mean of {args.reps} calls after one warm-up; [ms] wall / CPU\n")

    rows = []
    y_ref = p(x, DT)[0]
    if not args.no_obspy:
        rows.append(("obspy trace loop", *timeit(lambda: obspy_loop(x, baz), 1)))
        err = np.abs(obspy_loop(x, baz) - y_ref).max() / np.abs(y_ref).max()
        print(f"  (numpy steps against obspy: {err:.1e} of the peak)")
    rows.append(("numpy, step by step", *timeit(lambda: p(x, DT), args.reps)))
    run = timepart.compiled(NT, DT)
    rows.append(("numpy, compiled", *timeit(lambda: comppart(run(x), run.dt), args.reps)))
    if cp is not None:
        g = cp.asarray(x)
        sync = cp.cuda.Device().synchronize
        rows.append(("CuPy, step by step", *timeit(lambda: p(g, DT), args.reps * 4, sync)))
        runc = timepart.compiled(NT, DT, xp=cp)
        rows.append(("CuPy, compiled", *timeit(lambda: comppart(runc(g), runc.dt),
                                               args.reps * 4, sync)))
        err = float(cp.abs(comppart(runc(g), runc.dt)[0] - cp.asarray(y_ref)).max()
                    / np.abs(y_ref).max())
        print(f"  (CuPy compiled against numpy steps: {err:.1e} of the peak)")
    print()
    for name, w, c in rows:
        print(f"  {name:<22}{w:10.2f} / {c:10.2f}")


if __name__ == "__main__":
    main()
