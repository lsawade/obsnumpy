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

**Measured** (`scripts/benchmark_pipeline.py`): the cmt3d chain on 426 traces
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
