"""Regression tests for Stations, Meta and Dataset fixes."""
import numpy as np
from obspy import UTCDateTime

import obsnumpy as onp


def _dataset(nsta=3, npts=200):
    rng = np.random.default_rng(0)
    stations = onp.Stations(codes=np.array([f"N.S{i}" for i in range(nsta)]),
                            latitudes=rng.uniform(-50, 50, nsta),
                            longitudes=rng.uniform(-150, 150, nsta),
                            elevations=rng.uniform(0, 1000, nsta))
    meta = onp.Meta(starttime=UTCDateTime(2020, 1, 1), npts=npts, delta=1.0,
                    components=["N", "E", "Z"], stations=stations)
    return onp.Dataset(data=rng.standard_normal((nsta, 3, npts)), meta=meta)


def test_stations_from_lists():
    s = onp.Stations(codes=np.array(["A.B", "C.D"]), latitudes=[1.0, 2.0], longitudes=(3, 4))
    assert isinstance(s.latitudes, np.ndarray) and s.latitudes.tolist() == [1.0, 2.0]
    assert s.longitudes.tolist() == [3, 4]


def test_meta_dict_round_trip():
    meta = _dataset().meta
    back = onp.Meta.from_dict(meta.to_dict())
    assert np.array_equal(back.stations.elevations, meta.stations.elevations)
    assert list(back.stations.codes) == list(meta.stations.codes)


def test_taper_default_and_side():
    ds = _dataset()
    ref = ds.copy()
    ref.taper(0.1, type="cosine")
    ds.taper(0.1)
    assert np.array_equal(ds.data, ref.data)
    left = _dataset()
    left.taper(0.1, side="left")
    raw = _dataset().data
    assert np.array_equal(left.data[..., -10:], raw[..., -10:])     # right end untouched
    assert not np.array_equal(left.data[..., :10], raw[..., :10])


def test_subset_single_element_lists():
    ds = _dataset()
    one = ds.subset(stations=[1])
    assert one.data.shape == (1, 3, 200) and list(one.meta.stations.codes) == ["N.S1"]
    assert np.array_equal(one.data[0], ds.data[1])
    z = ds.subset(components=["Z"])
    assert z.data.shape == (3, 1, 200) and z.meta.components == ["Z"]
    assert np.array_equal(z.data[:, 0], ds.data[:, 2])


def test_subset_stations_and_components_cross():
    ds = _dataset(nsta=4)
    sub = ds.subset(stations=np.array([0, 3]), components=["N", "Z"])
    assert sub.data.shape == (2, 2, 200)
    assert np.array_equal(sub.data, ds.data[[0, 3]][:, [0, 2]])
    assert sub.meta.components == ["N", "Z"] and list(sub.meta.stations.codes) == ["N.S0", "N.S3"]
    s = ds.subset(stations="N.S2")
    assert list(s.meta.stations.codes) == ["N.S2"] and s.meta.components == ["N", "E", "Z"]
