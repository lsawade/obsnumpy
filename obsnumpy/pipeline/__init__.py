"""Seismogram processing over whole arrays, on numpy or CuPy.

A :class:`Pipeline` is a list of :class:`Step` objects, declared once and
applied to a whole array in one call. Each step reproduces an obspy operation
and is tested against it. When every step is linear and acts on time, the
pipeline compiles into a single matrix.

    >>> from obsnumpy.pipeline import Pipeline, Detrend, Taper, Bandpass
    >>> p = Pipeline([Detrend("linear"), Taper(0.05), Bandpass(1/150, 1/40, 4, True)])
    >>> y, dt_out = p(x, dt)                     # x: (..., npts), numpy or CuPy
"""
from .backend import xp_of, to_xp, cupy_or_none
from .steps import (Step, Detrend, Taper, Bandpass, Highpass, Lowpass, PreFilt, Cut,
                    Interpolate, RotateNE2RT, Convolve, StepConvolve)
from .pipeline import Pipeline, apply_to_dataset

__all__ = ["Pipeline", "apply_to_dataset", "Step", "Detrend", "Taper", "Bandpass", "Highpass",
           "Lowpass", "PreFilt", "Cut", "Interpolate", "RotateNE2RT", "Convolve",
           "StepConvolve", "xp_of", "to_xp", "cupy_or_none"]
