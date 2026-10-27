"""Misfits between synthetics and data, on whole arrays, on numpy or CuPy.

- :mod:`.base`: :class:`Misfit`, the interface every misfit implements;
- :mod:`.windowed`: :class:`WindowedMisfit`, the windowed, weighted,
  data-normalised L2 misfit of Sawade et al. (2022).

A new misfit goes in its own module, subclasses :class:`Misfit`, and is
exported here.
"""
from .base import Misfit
from .windowed import WindowedMisfit, windows_to_tapers

__all__ = ["Misfit", "WindowedMisfit", "windows_to_tapers"]
