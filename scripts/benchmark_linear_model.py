"""Benchmark obsnumpy.linear_model at the size of a gf3d basis for cmt3d.

The basis is synthetic, the size of one gf3d element for the cmt3d settings:
142 receivers x 3 components x 917 samples x 375 basis vectors, float32
(0.59 GB), at dt = 3.92 s. The window is cmt3d's: 827 samples, from 82.24 s
after the centroid, processed by its chain (detrend, demean, taper,
pre_filt, detrend, demean, taper, Lanczos to 2 s, 1601 samples) and rotated.
The misfit has one 400 s window per trace. The rotation to RT is applied to
the basis once (``Basis.rotated``), as a basis filler would; with
``--rotate-each-call`` it runs on every evaluation instead.

It reports, per evaluation and for K chains at once:

- forward: the cost only;
- gradient: cost, dC/dW and dC/dtau by the adjoint;
- jacobian: s and the derivatives along nine weight directions and tau;

on the CPU (set the thread variables before running) and on the GPU.

    CUDA_VISIBLE_DEVICES=1 numactl --cpunodebind=1 --membind=1 \\
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        python scripts/benchmark_linear_model.py
"""
import argparse
import os
import platform
import time

import numpy as np

from obsnumpy.linear_model import Basis, LinearBasisModel
from obsnumpy.misfit import WindowedMisfit, windows_to_tapers
from obsnumpy.pipeline import (Detrend, FixedWindow, Interpolate, Pipeline, PreFilt, RotateNE2RT,
                               Taper, cupy_or_none)

NREC, NCOMP, NT, NB, NDW = 142, 3, 917, 375, 9
DT, T_FIRST = 3.92, -200.0                      # basis: first sample 200 s before the centroid


def setup(xp, prune, rotate_basis=True):
    rng = np.random.default_rng(0)
    U = rng.standard_normal((NREC, NCOMP, NT, NB), dtype=np.float32)
    basis = Basis(U=U, t0=T_FIRST, dt=DT, scale=rng.uniform(0.5, 2, NREC)).to(xp)
    window = FixedWindow.from_trim(T_FIRST, DT, 100.0, 3300.0, pad_samples=5)
    processing = Pipeline([Detrend("linear"), Detrend("demean"), Taper(0.05),
                           PreFilt((0.0030, 0.0040, 0.0111, 0.0139)),
                           Detrend("linear"), Detrend("demean"), Taper(0.05),
                           Interpolate(2.0, 1601, 100.0 - window.first_time)])
    wins = [[[(int(a), int(a) + 200)] for a in rng.integers(0, 1400, NCOMP)] for _ in range(NREC)]
    d = rng.standard_normal((NREC, NCOMP, 1601))
    misfit = WindowedMisfit.build(d, 2.0, windows_to_tapers(wins, 1601))
    rotation = RotateNE2RT(rng.uniform(0, 360, NREC))
    if rotate_basis:                          # rotated once, as a basis filler would
        basis, rotation = basis.rotated(rotation), None
    model = LinearBasisModel(window, processing, rotation, misfit, prune=prune)
    return basis, model, window


def timeit(f, reps, sync):
    f()
    sync()
    w, c = time.perf_counter(), time.process_time()
    for _ in range(reps):
        f()
    sync()
    return (time.perf_counter() - w) / reps * 1e3, (time.process_time() - c) / reps * 1e3


def cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def run(xp, label, Ks, reps, jac_max_K, prune, rotate_basis):
    basis, model, window = setup(xp, prune, rotate_basis)
    sync = (lambda: xp.cuda.Device().synchronize()) if xp is not np else (lambda: None)
    rng = np.random.default_rng(1)
    print(f"\n{label}: basis {basis.nbytes / 1e9:.2f} GB, window {window.n} samples, "
          f"prune={prune}, rotation {'in the basis' if rotate_basis else 'per evaluation'}; "
          f"[ms] per call wall / CPU, and per evaluation")
    for K in Ks:
        W = xp.asarray(rng.standard_normal((K, NB)))
        tau = rng.uniform(-5, 5, K)
        dW = xp.asarray(rng.standard_normal((K, NDW, NB)))
        rows = [("forward", lambda: model.forward(basis, W, tau, return_s=False)),
                ("gradient", lambda: model.gradient(basis, W, tau))]
        if K <= jac_max_K:
            rows.append(("jacobian", lambda: model.jacobian(basis, W, tau, dW)))
        for name, f in rows:
            w, c = timeit(f, reps, sync)
            print(f"  K = {K:3d}  {name:<9} {w:9.2f} / {c:9.2f}   {w / K * 1e3:9.1f} us per evaluation")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--K", type=int, nargs="+", default=[1, 8, 64])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--no-cpu", action="store_true")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--prune", action="store_true", help="prune P to the windows' samples")
    ap.add_argument("--rotate-each-call", action="store_true",
                    help="rotate per evaluation instead of rotating the basis once")
    args = ap.parse_args()

    print("obsnumpy.linear_model benchmark")
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES"):
        print(f"  {v:<21}{os.environ.get(v, '(unset)')}")
    print(f"  CPU       {cpu_name()}")
    cp = None if args.no_gpu else cupy_or_none()
    if cp is not None:
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        print(f"  GPU       {props['name'].decode()}, CuPy {cp.__version__}")
    print("  measured: mean of the repeats after one warm-up")

    if not args.no_cpu:
        run(np, "CPU", args.K, max(1, args.reps // 2), jac_max_K=8, prune=args.prune,
            rotate_basis=not args.rotate_each_call)
    if cp is not None:
        run(cp, "GPU", args.K, args.reps * 4, jac_max_K=64, prune=args.prune,
            rotate_basis=not args.rotate_each_call)


if __name__ == "__main__":
    main()
