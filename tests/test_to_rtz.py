import obspy
from obsnumpy import Dataset, Meta, Stations
import numpy as np
from copy import deepcopy


def test_tortz_single_station():

    # Station geometry is north-south
    event_latitude = 0
    event_longitude = 10

    station_latitude = np.array([0])
    station_longitude = np.array([0])

    stations = Stations(
        codes=np.array(["station1"]),
        latitudes=station_latitude,
        longitudes=station_longitude,
    )

    meta = Meta(
        starttime=obspy.UTCDateTime(0),
        npts=10,
        delta=0.01,
        components=["N", "E", "Z"],
        stations=stations,
    )

    # Data is going to have all the energy on the N and Z components
    data = np.zeros((1, 3, 10))

    # North component
    data[0, 0, :] = 1
    # Z component
    data[0, 2, :] = 1

    # Create array stream with the data
    ast = Dataset(data=deepcopy(data), meta=meta)

    print(ast)

    # If we rotate to RTZ the T component should be the same as the N component
    # of the original data
    ast.rotate("->RT", event_latitude, event_longitude)

    print(ast)

    # Check that the R component is the same as the E component of the original data
    assert np.allclose(data[0, 1, :], ast.data[0, 0, :])

    # Check that the T component is the same as the N component of the original data
    assert np.allclose(data[0, 0, :], ast.data[0, 1, :])

    # Check that the Z component is the same as the Z component of the original data
    assert np.allclose(data[0, 2, :], ast.data[0, 2, :])


def test_tortz_single_station_2():

    # Station geometry is north-south
    event_latitude = 0
    event_longitude = 10

    station_latitude = np.array([10])
    station_longitude = np.array([10])

    stations = Stations(
        codes=np.array(["station1"]),
        latitudes=station_latitude,
        longitudes=station_longitude,
    )

    meta = Meta(
        starttime=obspy.UTCDateTime(0),
        npts=10,
        delta=0.01,
        components=["N", "E", "Z"],
        stations=stations,
    )

    # Data is going to have all the energy on the N and Z components
    data = np.zeros((1, 3, 10))

    # North component
    data[0, 1, :] = 1
    # Z component
    data[0, 2, :] = 1

    # Create array stream with the data
    ast = Dataset(data=deepcopy(data), meta=meta)

    # If we rotate to RTZ the T component should be the same as the N component
    # of the original data
    ast.rotate("->RT", event_latitude, event_longitude)

    # Check that the R component is the same as the N component of the original data
    assert np.allclose(data[0, 0, :], ast.data[0, 0, :])

    # Check that the T component is the same as the E component of the original data
    assert np.allclose(data[0, 1, :], ast.data[0, 1, :])

    # Check that the Z component is the same as the Z component of the original data
    assert np.allclose(data[0, 2, :], ast.data[0, 2, :])


def test_tortz_two_stations():

    # Station geometry is north-south
    event_latitude = 0
    event_longitude = 10

    station_latitude = np.array([0, 10])
    station_longitude = np.array([0, 10])

    stations = Stations(
        codes=np.array(["station1", "station2"]),
        latitudes=station_latitude,
        longitudes=station_longitude,
    )

    meta = Meta(
        starttime=obspy.UTCDateTime(0),
        npts=10,
        delta=0.01,
        components=["N", "E", "Z"],
        stations=stations,
    )

    # Data is going to have all the energy on the N and Z components
    data = np.zeros((2, 3, 10))

    # North component for station 1
    data[0, 0, :] = 1
    # Z component
    data[0, 2, :] = 1

    # North component for station 2
    data[1, 1, :] = 1
    # Z component
    data[1, 2, :] = 0

    # Create array stream with the data
    ast = Dataset(
        meta=deepcopy(meta),
        data=deepcopy(data),
    )

    # Station 1 check
    # If we rotate to RTZ the T component should be the same as the N component
    # of the original data
    ast.rotate("->RT", event_latitude, event_longitude)

    # Check that the R component is the same as the E component of the original data
    assert np.allclose(data[0, 1, :], ast.data[0, 0, :])

    # Check that the T component is the same as the N component of the original data
    assert np.allclose(data[0, 0, :], ast.data[0, 1, :])

    # Check that the Z component is the same as the Z component of the original data
    assert np.allclose(data[0, 2, :], ast.data[0, 2, :])

    # Station 2 check
    # If we rotate to RTZ the T component should be the same as the E component
    # of the original data for station 2, because it's aligned with the event

    # Check that the R component is the same as the N component of the original data
    assert np.allclose(data[1, 0, :], ast.data[1, 0, :])

    # Check that the T component is the same as the E component of the original data
    assert np.allclose(data[1, 1, :], ast.data[1, 1, :])

    # Check that the Z component is the same as the Z component of the original data
    assert np.allclose(data[1, 2, :], ast.data[1, 2, :])
