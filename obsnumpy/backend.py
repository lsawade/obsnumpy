"""numpy or CuPy, chosen by the input array's type.

CuPy is optional. Nothing here imports it unless a CuPy array is passed in (or
:func:`cupy_or_none` is asked for it), so a CPU-only install never touches it.
"""
from __future__ import annotations

from collections import OrderedDict
import importlib

import numpy as np


def cupy_or_none():
    """The ``cupy`` module, or None when it is not installed or has no device."""
    try:
        cp = importlib.import_module("cupy")
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        return None
    return cp


def _is_cupy(x) -> bool:
    return type(x).__module__.split(".", 1)[0] in ("cupy", "cupyx")


def xp_of(x):
    """The array module of ``x``: ``cupy`` for a CuPy array, else ``numpy``."""
    if _is_cupy(x):
        return importlib.import_module("cupy")
    return np


def device_key(xp):
    """``None`` for numpy, else the current CUDA device's id: the key for
    per-device caches of arrays."""
    return None if xp is np else xp.cuda.Device().id


def to_host(x):
    """``x`` as a numpy array (copied from the device for CuPy)."""
    return x.get() if _is_cupy(x) else np.asarray(x)


def signal_of(xp):
    """``scipy.signal`` or ``cupyx.scipy.signal``, matching ``xp``."""
    if xp is np:
        return importlib.import_module("scipy.signal")
    return importlib.import_module("cupyx.scipy.signal")


def sparse_of(xp):
    """``scipy.sparse`` or ``cupyx.scipy.sparse``, matching ``xp``."""
    if xp is np:
        return importlib.import_module("scipy.sparse")
    return importlib.import_module("cupyx.scipy.sparse")


def to_xp(xp, a, dtype=None):
    """``a`` (a host array) on ``xp``'s device, optionally cast to ``dtype``."""
    return xp.asarray(a, dtype=dtype)


# Small host tables (taper windows, filter sections, interpolation operators)
# are built once in float64 on the host. Their device copies are kept here, so
# that a pipeline applied many times does not copy them every call.
_DEVICE_CACHE: OrderedDict = OrderedDict()
_DEVICE_CACHE_SIZE = 64


def cached_to_xp(xp, key, build, dtype=None):
    """``build()`` moved to ``xp``'s current device and cast to ``dtype``,
    cached by ``(key, dtype, device)``. ``build`` returns a host array or a
    scipy sparse matrix; on numpy only the cast is applied (and cached)."""
    ckey = (key, None if dtype is None else np.dtype(dtype).str, device_key(xp))
    try:
        out = _DEVICE_CACHE[ckey]
        _DEVICE_CACHE.move_to_end(ckey)
        return out
    except KeyError:
        pass
    host = build()
    sparse = importlib.import_module("scipy.sparse")
    if sparse.issparse(host):
        if dtype is not None:
            host = host.astype(dtype)
        out = host if xp is np else sparse_of(xp).csr_matrix(host)
    elif xp is np:
        # a private copy: the host table is shared and read-only, and some
        # consumers (scipy's sosfilt) refuse read-only buffers
        out = np.array(host, dtype=dtype)
    else:
        out = to_xp(xp, host, dtype)
    _DEVICE_CACHE[ckey] = out
    if len(_DEVICE_CACHE) > _DEVICE_CACHE_SIZE:
        _DEVICE_CACHE.popitem(last=False)
    return out
