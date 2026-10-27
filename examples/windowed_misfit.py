"""The windowed misfit between processed data and a synthetic.

The data are those of examples/pipeline.py: three stations of the 2004-12-05
M4.2 Black Forest event, processed to 4 Hz and RTZ, here at 2.5-10 s
periods. The "synthetic" is the same record, delayed by tau and scaled by
0.8, through the same processing. So the cost is smallest at tau = 0, and
the adjoint gradient in tau can be checked against a finite difference.

Windows come in two groups, P (R and Z) and S (R, T and Z), from crustal
velocities. Each group is normalised by its own data energy, and both
collapse into one weight trace, omega.

    python examples/windowed_misfit.py   # writes examples/windowed_misfit.png
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import obspy

from obsnumpy import Dataset
from obsnumpy.misfit import WindowedMisfit, windows_to_tapers
from obsnumpy.pipeline import Bandpass, Detrend, Interpolate, Pipeline, RotateNE2RT, Taper

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "tests", "data")
EVENT_LAT, EVENT_LON = 48.03, 8.04
TIME_SHIFT = 3.9                           # the record starts at the centroid, 3.9 s after the origin
DT, NPTS = 0.25, 800                       # the processed grid: 4 Hz, 200 s

# 1. Data, and processing with the start time as a parameter: a synthetic
#    delayed by tau reads the record tau earlier.
st = obspy.read(os.path.join(DATA, "interp", "*.mseed"))
inv = obspy.read_inventory(os.path.join(DATA, "stations", "*.xml"))
ds = Dataset.from_stream(st, inv=inv, event_latitude=EVENT_LAT, event_longitude=EVENT_LON)
ds = ds.subset(stations=np.flatnonzero(ds.meta.stations.distances < 5.0))
x, dt_raw = ds.data, ds.meta.delta
sta = ds.meta.stations


def process(tau=0.0, deriv=0):
    p = Pipeline([Detrend("linear"), Detrend("demean"), Taper(0.05),
                  Bandpass(0.1, 0.4, corners=4, zerophase=True),
                  Interpolate(DT, NPTS, offset_s=-tau, deriv=deriv),
                  RotateNE2RT(sta.back_azimuths)])
    return p(x, dt_raw)[0]


d = process()


def synthetic(tau):
    return 0.8 * process(tau)


def dsynthetic_dtau(tau):
    return -0.8 * process(tau, deriv=1)          # d offset / d tau = -1


# 2. Windows, in samples, per station and component (R, T, Z): P from 6.0 km/s,
#    S from 3.5 km/s, each from 2 s before the arrival. Overlapping windows add.
km = sta.distances * 111.19


def windows(v, length, comps):
    out = []
    for dist in km:
        t0 = dist / v - TIME_SHIFT - 2.0
        win = [(int(t0 / DT), int((t0 + length) / DT))]
        out.append([win if c in comps else [] for c in "RTZ"])
    return out


P = windows_to_tapers(windows(6.0, 10.0, "RZ"), NPTS)
S = windows_to_tapers(windows(3.5, 30.0, "RTZ"), NPTS)
m = WindowedMisfit.from_groups(d, DT, [(P, 1.0), (S, 1.0)])

# 3. The cost over tau, and its gradient by the adjoint source.
taus = np.linspace(-3.0, 3.0, 121)
C = np.array([m.cost(synthetic(tau)) for tau in taus])
print(f"C is smallest at tau = {taus[C.argmin()]:+.2f} s")

tau, h = 0.5, 1e-4
adjoint = np.sum(m.residual_weight(synthetic(tau)) * dsynthetic_dtau(tau))
fd = (m.cost(synthetic(tau + h)) - m.cost(synthetic(tau - h))) / (2 * h)
print(f"dC/dtau at {tau} s: adjoint {adjoint:.6g}, finite difference {fd:.6g}")

# 4. Data, the synthetic at tau, and the windows of each group; and C(tau).
s = synthetic(tau)
t = np.arange(NPTS) * DT
order = np.argsort(sta.distances)
fig, axes = plt.subplots(1, 4, figsize=(13, 4.5), gridspec_kw=dict(width_ratios=[3, 3, 3, 2]))
for ax, c, comp in zip(axes, range(3), "RTZ"):
    for k, i in enumerate(order):
        for tap, color in ((P, "C0"), (S, "C2")):
            ax.fill_between(t, k - 0.45 * tap[i, c], k + 0.45 * tap[i, c], color=color,
                            alpha=0.2, lw=0)
        scale = 0.45 / np.abs(d[i]).max()
        ax.plot(t, k + scale * d[i, c], "k", lw=0.8)
        ax.plot(t, k + scale * s[i, c], "C3", lw=0.8)
    ax.set(title=comp, xlabel="time after centroid [s]", xlim=(0, 130), ylim=(-0.6, 2.6),
           yticks=range(len(order)), yticklabels=[])
axes[0].set_yticklabels([f"{sta.codes[i]} {sta.distances[i]:.1f}°" for i in order])
axes[3].plot(taus, C, "k")
axes[3].axvline(tau, color="C3", lw=0.8)
axes[3].set(title="C(tau)", xlabel="tau [s]")
handles = [plt.Line2D([], [], color="k", label="data"),
           plt.Line2D([], [], color="C3", label=f"synthetic, tau = {tau} s"),
           plt.Rectangle((0, 0), 1, 1, color="C0", alpha=0.2, label="P windows"),
           plt.Rectangle((0, 0), 1, 1, color="C2", alpha=0.2, label="S windows")]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(os.path.join(HERE, "windowed_misfit.png"), dpi=150)
