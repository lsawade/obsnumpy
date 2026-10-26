"""Process a small dataset with an obsnumpy pipeline.

The data are the test traces of the 2004-12-05 M4.2 event in the Black
Forest (tests/data/interp): velocity, pre-filtered to 0.1-1 Hz, at 5 Hz,
starting at the centroid. The pipeline detrends, tapers, bandpasses,
resamples to 4 Hz and rotates to RTZ, on all traces at once. It is then
checked against obspy on one trace, and run again compiled into a single
matrix. IU.PAB, at 12 degrees, is mostly noise for an M4.2 and is left out.

    python examples/pipeline.py          # writes examples/pipeline.png
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import obspy

from obsnumpy import Dataset
from obsnumpy.pipeline import Bandpass, Detrend, Interpolate, Pipeline, RotateNE2RT, Taper

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "tests", "data")
EVENT_LAT, EVENT_LON = 48.03, 8.04

# 1. The data: (station, N/E/Z, time) arrays with their metadata. G.SSB has no
#    N component and is dropped; so is everything beyond 5 degrees.
st = obspy.read(os.path.join(DATA, "interp", "*.mseed"))
inv = obspy.read_inventory(os.path.join(DATA, "stations", "*.xml"))
ds = Dataset.from_stream(st, inv=inv, event_latitude=EVENT_LAT, event_longitude=EVENT_LON)
ds = ds.subset(stations=np.flatnonzero(ds.meta.stations.distances < 5.0))   # drops IU.PAB
raw = ds.copy()

# 2. The pipeline, declared once. Each step is the obspy operation of the
#    same name, applied to the whole array.
p = Pipeline([
    Detrend("linear"),
    Detrend("demean"),
    Taper(0.05),
    Bandpass(0.1, 1.0, corners=4, zerophase=True),
    Interpolate(new_dt=0.25, npts=800),                      # the first 200 s
    RotateNE2RT(ds.meta.stations.back_azimuths),
])
ds.apply(p)                      # data and meta (delta, npts, starttime, RTZ) in place
print(f"{raw.data.shape} at {raw.meta.delta} s  ->  {ds.data.shape} at {ds.meta.delta} s, "
      f"components {ds.meta.components}")

# 3. The same on one trace with obspy (Z is not rotated).
i = 0
tr = st.select(id=f"{ds.meta.stations.codes[i]}.*Z")[0].copy()
tr.detrend("linear").detrend("demean").taper(0.05, type="cosine")
tr.filter("bandpass", freqmin=0.1, freqmax=1.0, corners=4, zerophase=True)
tr.interpolate(4.0, method="lanczos", a=12, npts=800)
err = np.abs(ds.data[i, 2] - tr.data).max() / np.abs(tr.data).max()
print(f"{tr.id} against obspy: {err:.1e} of the peak")

# 4. Compiled: every step but the rotation is linear in time, so they collapse
#    into one (n_in, n_out) matrix P, and processing is y = x @ P.
timepart, rotation = p.split(len(p) - 1)
run = timepart.compiled(raw.meta.npts, raw.meta.delta)
y, _ = rotation(run(raw.data), run.dt)
print(f"compiled against step by step: {np.abs(y - ds.data).max() / np.abs(ds.data).max():.1e}")

# 5. A record section of the result.
order = np.argsort(ds.meta.stations.distances)
t = np.arange(ds.meta.npts) * ds.meta.delta
fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True)
for ax, c, comp in zip(axes, range(3), ds.meta.components):
    for k, i in enumerate(order):
        ax.plot(t, k + 0.45 * ds.data[i, c] / np.abs(ds.data[i]).max(), "k", lw=0.5)
    ax.set(title=comp, xlabel="time after centroid [s]", xlim=(t[0], t[-1]))
axes[0].set_yticks(range(len(order)),
                   [f"{ds.meta.stations.codes[i]} {ds.meta.stations.distances[i]:.1f}°"
                    for i in order])
fig.tight_layout()
fig.savefig(os.path.join(HERE, "pipeline.png"), dpi=150)
