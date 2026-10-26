"""What every misfit provides, and what a new one has to implement.

A misfit compares a synthetic ``s`` with fixed data. ``s`` is a numpy or CuPy
array ``(..., nsta, ncomp, npts)``; leading axes are a batch (chains), and
results follow ``s`` onto its device. The cost is a sum of per-trace costs,
which ``trace_weights`` ``(..., nsta, ncomp)`` may reweight per batch entry
(a likelihood with per-trace variances).

A subclass implements

- :meth:`Misfit.trace_cost`: the per-trace costs ``(..., nsta, ncomp)``;
- :meth:`Misfit._residual_weight`: ``dC/ds``, the shape of ``s``;
- :attr:`Misfit.mask`: the samples the cost depends on;
- :meth:`Misfit.take`: the same misfit on a subset of the time axis;

and gets :meth:`Misfit.cost`, :meth:`Misfit.residual_weight` and
:meth:`Misfit.gradient` from them. Fixed data arrays can be fetched with
:meth:`Misfit._on`, which copies them to a device once.
"""
from __future__ import annotations

import numpy as np

from ..backend import device_key, xp_of


class Misfit:

    # -- to implement ------------------------------------------------------------------

    def trace_cost(self, s):
        """Per trace, the cost of ``s``: ``(..., nsta, ncomp)``."""
        raise NotImplementedError

    def _residual_weight(self, s):
        """``dC/ds`` of the unweighted cost, the shape of ``s``."""
        raise NotImplementedError

    @property
    def mask(self) -> np.ndarray:
        """``(nsta, ncomp, npts)``, True where the cost depends on ``s``."""
        raise NotImplementedError

    def take(self, cols) -> "Misfit":
        """The misfit on the samples ``cols`` of the time axis only: the same
        cost on synthetics cut to ``cols``."""
        raise NotImplementedError

    # -- provided ----------------------------------------------------------------------

    def cost(self, s, trace_weights=None):
        """``C(s) = sum trace_weights * trace_cost(s)``: a float, or an array over
        the leading (batch) axes."""
        c = self.trace_cost(s)
        if trace_weights is not None:
            c = c * xp_of(s).asarray(trace_weights, dtype=np.float64)
        c = c.sum(axis=(-2, -1))
        return float(c) if c.ndim == 0 else c

    def residual_weight(self, s, trace_weights=None):
        """``dC/ds``, the shape of ``s``: the adjoint source."""
        rw = self._residual_weight(s)
        if trace_weights is None:
            return rw
        return rw * xp_of(s).asarray(trace_weights, dtype=np.float64)[..., None]

    def gradient(self, s, J):
        """``dC/dm = <dC/ds, J_m>`` for ``J`` ``(..., nparam, nsta, ncomp, npts)``,
        the derivatives of ``s``. Returns ``(..., nparam)``."""
        rw = self.residual_weight(s)
        return (J * rw[..., None, :, :, :]).sum(axis=(-3, -2, -1))

    def _on(self, xp, name):
        """The numpy attribute ``name`` on ``xp``'s current device, copied once."""
        if xp is np:
            return getattr(self, name)
        cache = self.__dict__.setdefault("_device_cache", {})
        key = (name, device_key(xp))
        if key not in cache:
            cache[key] = xp.asarray(getattr(self, name))
        return cache[key]
