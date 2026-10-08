"""Moved to :mod:`obsnumpy.backend`; kept so that old imports work."""
from ..backend import (cached_to_xp, cupy_or_none, device_key, signal_of, sparse_of,  # noqa: F401
                       to_host, to_xp, xp_of)
