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
