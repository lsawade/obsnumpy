# Combining obspy and numpy to accelerate processing.

This is an intermediate package to streamline the processing from
obspy's incredible slow loops to actions in from of numpy operations.
The main goal however is to immitate.

The objective is to translate Stream.
- rotate
- interpolate
- bandpass filter

The package will assume that the user did the bare minimum of processing the data into NEZ traces and removed the instrument response.

There will be a function that takes in a stream as an argument that will interpolate each trace separately to make it a single array, but that's really not the purpose of this package, but rather a convenience function.


For the source inversion one could think of the following:

After the initial windowing, we can select the traces that have windows and compute corresponding ArrayStreams from the windows

- a taper array
- a weight array
- a normalizing array
    * We want this to be an array because we cannot divide the array after
      summation. We only have to compute this once. since we are normalizing by the data (See Sawade et al. 2022)

Then, the array formulation for the cost function simplifies to

C(m) = 0.5 * sum_k { sum(W * [Dk-Sk(m)]**2 / Nk), }

where W is the combined window and weight array, D is the Data array, S is the simulated data array, and N is the Data normalization array. k denotes the category.



## Pipelines

`obsnumpy.pipeline` runs seismogram processing over whole arrays, on numpy or
CuPy. A `Pipeline` is a list of steps, declared once and applied to an array
of any leading shape `(..., npts)` in one call: stations, components, partial
derivatives, chains, or the traces of a Green-function basis. The array module
follows the input: numpy in, numpy out; CuPy in, CuPy out.

Every step reproduces an obspy operation, and `tests/test_pipeline.py` checks
it against obspy to 1e-12 of each trace's peak:

| step | reproduces |
|---|---|
| `Detrend("linear" \| "demean")` | `Trace.detrend` |
| `Taper(max_percentage, type="cosine", max_length=None, side="both")` | `Trace.taper` (obspy's p = 1 window) |
| `Bandpass`, `Highpass`, `Lowpass` | `Trace.filter(..., zerophase=)`; `cupyx.scipy.signal.sosfilt` on CuPy |
| `PreFilt(flimit)` | `remove_response(pre_filt=)` without a response (obsproclib's `filter_trace`) |
| `Cut(start, stop)` | slicing by sample |
| `Interpolate(new_dt, npts, offset_s, a=12)` | `Trace.interpolate(method="lanczos")`, as a sparse linear operator |
| `RotateNE2RT(baz)` | `obspy.signal.rotate.rotate_ne_rt`, on the component axis |
| `Convolve(kernel)`, `StepConvolve(kernel)` | a centred FIR, and a running sum plus a centred kernel (the Heaviside STF of a Green-function database) |

obspy's default `weighted_average_slopes` interpolation is not a step, because
it is nonlinear. `obsnumpy.process.interpolate` keeps it.

`examples/pipeline.py` runs a pipeline on the test data, checks it against obspy,
compiles it, and plots the result.

The cmt3d chain, on a `Dataset` of NEZ traces:

```python
from obsnumpy.pipeline import (Pipeline, Cut, Detrend, Taper, PreFilt, Interpolate,
                               RotateNE2RT)

interp = Interpolate.from_utc(ds.meta.starttime + c0 * dt, centroid + 100.0,
                              new_dt=2.0, npts=1601, a=12)
p = Pipeline([Cut(c0, c1),
              Detrend("linear"), Detrend("demean"), Taper(0.05),
              PreFilt((0.003, 0.004, 0.0111, 0.0139)),
              Detrend("linear"), Detrend("demean"), Taper(0.05),
              interp, RotateNE2RT(ds.meta.stations.back_azimuths)])

ds.apply(p)          # data, delta, npts, starttime and components follow the steps
```

**Compiling.** When every step is linear and acts on time, `Pipeline.matrix`
pushes the identity through the steps. It returns one `(n_in, n_out)` matrix
`P`, so that processing is `y = x @ P`, and `Pipeline.compiled` returns that
product as a callable. Split the rotation off first:

```python
timepart, rotation = p.split(len(p) - 1)
run = timepart.compiled(n_in, dt, xp=cupy)          # P lives on the GPU
y, _ = rotation(run(cupy.asarray(x)), run.dt)
```

**Matching obspy's interpolation exactly.** `Trace.interpolate` takes the
offset as a difference of two float64 epoch timestamps, each rounded to about
1e-7 s. `Interpolate.from_utc(..., like_obspy=True)` reproduces that offset,
and with it obspy's output to rounding. The default uses `UTCDateTime`'s exact
nanosecond difference.

**Measured** (with `scripts/benchmark_pipeline.py`, removed after ce4b5f5): the cmt3d chain on 426 traces
× 3725 samples, `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`,
`numactl --cpunodebind=0 --membind=0`, Intel Xeon Gold 6548Y+ and an NVIDIA
H100 NVL. Times are wall, in ms:

| route | time [ms] |
|---|---|
| obspy trace loop | 1316 |
| numpy, step by step | 19.8 |
| numpy, compiled | 61 |
| CuPy, step by step | 0.94 |
| CuPy, compiled | 0.22 |

CuPy is optional (`pip install obsnumpy[gpu]`). Nothing imports it unless a
CuPy array is passed in.


## Differentiating through processing

Every linear step has an adjoint, `step.adjoint(g, dt, npts)`, and so does a
pipeline, `p.adjoint(g, dt, n_in)` (its steps' adjoints, in reverse order).
Each maps a cotangent on the output back onto the input, so that
`<p(x), g> = <x, p.adjoint(g)>`. A compiled pipeline's adjoint is
`run.adjoint(g) = g @ P.T`. The tests check every one of them with dot
products.

**Processing stays in absolute time.** Observed data are cut at absolute
times, and the detrend, taper and filters act on that window wherever the
source is. Cut and taper do not commute with a time shift, so a synthetic
that moves with a shift `tau` is not processed in a frame that moves with
it. Instead it is first interpolated onto the data's fixed window, `S(tau)`,
and then processed, `P`:

    s(tau)  = P . S(tau) . v
    ds/dtau = P . S'(tau) . v
    dC/dv   = S(tau).T . P.T . dC/ds

- `FixedWindow.from_trim(trace_start, dt, t0, t1, pad_samples)` is the grid
  that obspy's `Trace.trim(nearest_sample=True)` keeps after obsproclib's
  `flex_cut_trace(..., dynamic_npts=pad_samples)`: an exact half sample
  rounds outward. Floor/ceil keeps one sample fewer for cmt3d's cut.
- `window.shift_step(trace_start, tau)` is `S(tau)`, a `Shift` step. A
  positive `tau` delays the trace. `.d_dtau()` is `S'(tau)`, built from the
  derivative of the Lanczos kernel on the same taps.
- `Interpolate(..., deriv=1)` is the interpolation's derivative in its
  offset.

`s(tau)` is then smooth in `tau`. Production instead shifts the trace's
start time and trims at the nearest sample, so its window jumps by a whole
sample as `tau` crosses half a sample. The tests measure the second
differences of a fixed functional over a 0.05 s grid of `tau`: max/median
1.8, against more than 30 for production's route.

A gradient in `tau` and in the synthetic, checked by central differences:

```python
import numpy as np
from obspy import UTCDateTime
from obsnumpy.pipeline import (Pipeline, Detrend, Taper, PreFilt, Interpolate, FixedWindow)

dt, centroid = 3.92, UTCDateTime(2023, 4, 14, 9, 55, 51.48)
t_first = centroid - 200.0                              # the synthetic's first sample
v = np.random.default_rng(0).standard_normal((3, 1000))  # stand-in synthetic, (..., npts)
d = np.random.default_rng(1).standard_normal((3, 1601))  # stand-in data on the output grid

# the window the data were cut on, and its processing
w = FixedWindow.from_trim(t_first, dt, centroid + 100, centroid + 3300, pad_samples=5)
P = Pipeline([Detrend("linear"), Detrend("demean"), Taper(0.05),
              PreFilt((0.003, 0.004, 0.0111, 0.0139)),
              Detrend("linear"), Detrend("demean"), Taper(0.05),
              Interpolate.from_utc(w.first_time, centroid + 100, 2.0, 1601)])

def misfit(tau):
    S = w.shift_step(t_first, tau)                      # S(tau): onto the fixed window
    s, _ = P(S(v, dt), dt)
    return 0.5 * np.sum((s - d) ** 2)

def gradient(tau):
    S = w.shift_step(t_first, tau)
    s, _ = P(S(v, dt), dt)
    r = s - d                                           # dC/ds
    ds_dtau, _ = P(S.d_dtau()(v, dt), dt)               # P . S'(tau) . v
    dC_dv = S.adjoint(P.adjoint(r, dt, w.n), dt, v.shape[-1])   # S^T . P^T . r
    return np.sum(r * ds_dtau), dC_dv

tau, h = 1.3, 1e-3
g_tau, g_v = gradient(tau)
print(g_tau, (misfit(tau + h) - misfit(tau - h)) / (2 * h))   # -0.0268698 twice
```

**PreFilt and Nyquist.** obspy and obsproclib make the Nyquist bin real with
`abs()`, which flips the sign of a negative bin. That is nonlinear once the
SAC taper reaches Nyquist. `PreFilt` keeps obspy's behaviour by default and
reports it per input: `is_linear(dt, npts)` is True when the taper vanishes
at Nyquist, the usual case. `matrix()` and `adjoint()` refuse the step where
it is not linear. `PreFilt(..., nyquist="real")` is linear always.


## Windowed misfit

`obsnumpy.misfit` is the weighted, window-tapered, data-normalised L2 misfit
of Sawade et al. (2022), eq. (5), on arrays. Per trace and per group of
windows k (e.g. one per wave type):

    C   = sum_k sum_traces weight_k * 1/2 sum_w int tap_kw (s - d)^2 dt / N_k
    N_k = sum_w int tap_kw d^2 dt                   (not divided where N_k = 0)

This is the quantity of cmt3d's `CostGradHess.cost`, summed over wave types.
Everything but the synthetic `s` is fixed by the data, so `from_groups` collects it
once into an effective weight trace per (station, component):

    Omega = sum_k weight_k * sum_w tap_kw / N_k,    C = 1/2 sum int Omega (s - d)^2 dt

Overlapping windows add. Wave types that share one processing collapse into
one `Omega`, so an evaluation is one weighted sum. The taper of a window is
obswinlib's `tukey(right - left, alpha=0.25)` on `[left, right)`.

```python
import numpy as np
from obsnumpy.misfit import WindowedMisfit, windows_to_tapers

rng = np.random.default_rng(0)
dt, npts = 2.0, 1601
d = rng.standard_normal((142, 3, npts))                  # processed data (sta, RTZ, t)
s = d + 0.1 * rng.standard_normal(d.shape)               # a synthetic
J = rng.standard_normal((10,) + d.shape)                 # its processed derivatives

# windows per wave type: windows[sta][comp] = [(left, right), ...] in samples
body = [[[(100, 300)], [(120, 320)], [(100, 300)]] for _ in range(142)]
surface = [[[(600, 1100)], [(550, 1000)], [(600, 1100)]] for _ in range(142)]
w_body, w_surf = rng.uniform(0.5, 1.5, (142, 3)), rng.uniform(0.5, 1.5, (142, 3))

m = WindowedMisfit.from_groups(d, dt, [(windows_to_tapers(body, npts), w_body),
                                        (windows_to_tapers(surface, npts), w_surf)])
C = m.cost(s)                       # float; or (K,) for s of shape (K, 142, 3, npts)
g = m.residual_weight(s)            # dC/ds: the adjoint source
grad = m.gradient(s, J)             # dC/dm, (10,)
H = m.gauss_newton(J)               # J^T Omega J dt, (10, 10)
```

`s` and `J` may be CuPy arrays; `d` and `Omega` move to the device once.
`m.mask` (`Omega > 0`) marks the samples any window covers. `build(d, dt,
tapers, weights)` is the one-group case.
`examples/windowed_misfit.py` builds P and S groups on real data and checks
the adjoint gradient in a time shift against a finite difference.

`gauss_newton(J)` is `J^T Omega J dt`, the Gauss-Newton Hessian of `C`.
cmt3d's `hess` instead weights each window by its taper *squared*; the two
differ on the tapers' ramps, by a few percent of `H`.

**Adding a misfit.** `obsnumpy/misfit/` holds one module per misfit
(`windowed.py`), each a subclass of `obsnumpy.misfit.Misfit` (`base.py`). A
subclass implements `trace_cost(s)` (per trace, `(..., nsta, ncomp)`),
`_residual_weight(s)` (`dC/ds`), `mask` and `take(cols)`, and inherits
`cost`, `residual_weight` and `gradient`, with per-trace `trace_weights`.
That is the interface `LinearBasisModel` uses.


## Linear basis model

`obsnumpy.linear_model` is a forward model for synthetics that are linear in a
basis, with its adjoint gradient and its Jacobian, batched over chains:

    v = scale * U W                  contraction (one GEMM over the chains)
    y = S(tau) v                     Lanczos onto the data's FixedWindow
    s = Rot P y                      the data's processing, compiled into P
    C = misfit.cost(s)

W can be a moment tensor contracted with strain Green functions, a force,
or weights on precomputed elementary seismograms. The caller maps its
parameters to W, and maps `dC/dW` back through its own `dW/dtheta`.

- `Basis(U, t0, dt, scale)`: `U` is `(nrec, ncomp, nt, nbasis)`, float32 by
  default; everything after the contraction is float64.
- `BasisStore(fill, max_items, max_bytes, xp)`: an LRU of bases on one device,
  filled on demand, with hit, miss and eviction counts.
- `LinearBasisModel(window, processing, rotation, misfit)`:
  - `forward(basis, W, tau)` returns `s` and `C`;
  - `gradient(...)` returns `C`, `dC/dW` and `dC/dtau`, by the adjoint
    `S^T P^T Rot^T dC/ds`;
  - `jacobian(..., dW)` returns `s` and the processed derivatives along the
    caller's weight directions, plus `tau`.

  `W` is `(K, nbasis)` and `tau` `(K,)`, one row per chain. A `tau` beyond
  what the basis covers raises; it does not silently zero-fill.

  `forward` and `gradient` also take `trace_weights` `(K, nrec, ncomp)`:
  each chain's own factor on `omega` per trace, for a likelihood whose
  per-trace variances are parameters. `per_trace=True` returns the
  per-trace costs `(K, nrec, ncomp)` (`WindowedMisfit.trace_cost`, without
  the weights) in place of `C`, while the gradient stays that of the
  weighted `C`.
- `Basis.rotated(RotateNE2RT(baz))`: the rotation commutes with the rest, so
  a basis rotated once at fill time, with `rotation=None`, saves a rotation
  per evaluation.

```python
import numpy as np
from obsnumpy.linear_model import Basis, BasisStore, LinearBasisModel
from obsnumpy.misfit import WindowedMisfit, windows_to_tapers
from obsnumpy.pipeline import (Pipeline, Detrend, Taper, PreFilt, Interpolate, FixedWindow,
                               RotateNE2RT)

rng = np.random.default_rng(0)
nrec, nt, nbasis, dt = 20, 917, 375, 3.92
baz = rng.uniform(0, 360, nrec)

def fill(key):
    """The caller's: build the basis for `key` (here random), rotated once."""
    U = rng.standard_normal((nrec, 3, nt, nbasis), dtype=np.float32)
    return Basis(U=U, t0=-200.0, dt=dt, scale=np.ones(nrec)).rotated(RotateNE2RT(baz))

window = FixedWindow.from_trim(-200.0, dt, 100.0, 3300.0, pad_samples=5)   # the data's cut
processing = Pipeline([Detrend("linear"), Detrend("demean"), Taper(0.05),
                       PreFilt((0.003, 0.004, 0.0111, 0.0139)),
                       Detrend("linear"), Detrend("demean"), Taper(0.05),
                       Interpolate(2.0, 1601, 100.0 - window.first_time)])
d = rng.standard_normal((nrec, 3, 1601))                                   # processed data, RTZ
wins = [[[(200, 600)]] * 3 for _ in range(nrec)]
misfit = WindowedMisfit.build(d, 2.0, windows_to_tapers(wins, 1601))
model = LinearBasisModel(window, processing, rotation=None, misfit=misfit)
store = BasisStore(fill, max_items=4)

basis = store.get("element 1985")
W = rng.standard_normal((8, nbasis))                     # 8 chains
tau = rng.uniform(-5, 5, 8)                              # their time shifts [s]
s, C = model.forward(basis, W, tau)                      # (8, nrec, 3, 1601), (8,)
C, dC_dW, dC_dtau = model.gradient(basis, W, tau)        # (8,), (8, nbasis), (8,)
s, J = model.jacobian(basis, W, tau, dW=rng.standard_normal((8, 9, nbasis)))  # J: (8, 10, ...)
```

On the GPU, `S(tau)` and its adjoint are small CUDA kernels. When the window's
sampling interval is the basis's, or an integer multiple of it, the tap
weights depend on the chain only and are computed on the host.

**Measured** (with `scripts/benchmark_linear_model.py`, removed after ce4b5f5): a basis the size of one
gf3d element for cmt3d, 142 x 3 x 917 x 375 float32 (0.59 GB); the cmt3d
window and chain; the basis rotated once. One socket of an Intel Xeon Gold
6548Y+ (`numactl` on the GPU's NUMA node), and an NVIDIA H100 NVL. Per call,
wall time:

| | forward | gradient | jacobian (9 + tau) |
|---|---|---|---|
| CPU, 1 thread, K = 1 | 61 ms | 143 ms | 410 ms |
| CPU, 32 threads, K = 1 | 17 ms | 60 ms | 212 ms |
| GPU, K = 1 | 0.43 ms | 0.79 ms | 0.95 ms |
| GPU, K = 64, per evaluation | 69 us | 146 us | 695 us |

At K = 64 most of an evaluation is the float64 processing GEMM: `P` maps 827
window samples to 1601 for each of 426 traces, about 34 us at 33 TFLOP/s.
The contraction is 13 us. `prune=True` drops `P`'s columns outside every
window. It changes no result, but it saves nothing when the traces' windows
jointly cover the record.
