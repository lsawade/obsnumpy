from obspy.geodetics.base import gps2dist_azimuth
from obsnumpy.utils import distazbaz
import numpy as np
import time


def test_distaz():

    lat0 = 5
    lon0 = 5
    lat = np.array([5, 10, 20, 50, 60])
    lon = np.array([5, 10, 20, 50, 60])

    correct_dist = []
    correct_az = []
    correct_baz = []

    t0 = time.time()
    for _lat, _lon in zip(lat, lon):
        d, a, b = gps2dist_azimuth(lat0, lon0, _lat, _lon)
        correct_dist.append(d)
        correct_az.append(a)
        correct_baz.append(b)

    correct_dist = np.array(correct_dist)
    correct_az = np.array(correct_az)
    t1 = time.time()

    print(f"Elapsed time for for loop: {t1-t0}", flush=True)

    t0 = time.time()

    dist, az, baz = distazbaz(lat0, lon0, lat, lon)

    t1 = time.time()
    print(f"Elapsed time for vectorized function: {t1-t0}", flush=True)

    assert np.allclose(dist, correct_dist)
    assert np.allclose(az, correct_az)
    assert np.allclose(baz, correct_baz)
