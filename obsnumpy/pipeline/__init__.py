"""Seismogram processing over whole arrays, on numpy or CuPy.

A :class:`Pipeline` is a list of :class:`Step` objects, declared once and
applied to a whole array in one call. Each step reproduces an obspy operation
and is tested against it. When every step is linear and acts on time, the
pipeline compiles into a single matrix. Every linear step has an adjoint, and
:class:`FixedWindow` with :class:`Shift` processes a shifted synthetic on a
window fixed in absolute time, with its derivative in the shift.

    >>> from obsnumpy.pipeline import Pipeline, Detrend, Taper, Bandpass
    >>> p = Pipeline([Detrend("linear"), Taper(0.05), Bandpass(1/150, 1/40, 4, True)])
    >>> y, dt_out = p(x, dt)                     # x: (..., npts), numpy or CuPy
"""
from ..backend import xp_of, to_xp, to_host, cupy_or_none
from .steps import (Step, Detrend, Taper, Bandpass, Highpass, Lowpass, PreFilt, Cut,
                    Interpolate, Shift, RotateNE2RT, Convolve, StepConvolve)
from .pipeline import Pipeline, apply_to_dataset
from .window import FixedWindow

__all__ = ["Pipeline", "apply_to_dataset", "FixedWindow", "Step", "Detrend", "Taper",
           "Bandpass", "Highpass", "Lowpass", "PreFilt", "Cut", "Interpolate", "Shift",
           "RotateNE2RT", "Convolve", "StepConvolve", "xp_of", "to_xp", "to_host",
           "cupy_or_none"]
