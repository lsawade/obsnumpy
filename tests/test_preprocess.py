import os
import obspy
import numpy as np
from obsnumpy import preprocess, Dataset


testfolder = os.path.dirname(os.path.abspath(__file__))
testdatafolder = os.path.join(testfolder, "data")
testdata_raw = os.path.join(testfolder, "data", "raw")
testdata_preprocessed = os.path.join(testfolder, "data", "preprocessed")
testdata_interp = os.path.join(testfolder, "data", "interp")
testdata_stations = os.path.join(testfolder, "data", "stations")
testdata_event = os.path.join(testfolder, "data", "germany_event.cmt")


def test_preprocess_no_interpolation():
    """Test just reads the data and checks if the preprocessing is done and
    the data is correctly stored in the ArrayStream object."""
    # Get event
    event = obspy.read_events(testdata_event)[0]

    # Get event data
    origin = event.origins[0]

    # Read inventory
    inv = obspy.read_inventory(os.path.join(testdata_stations, "*.xml"))

    # Read the raw stream
    st = obspy.read(os.path.join(testdata_raw, "*.mseed"))

    # Read the preprocessed stream
    correct_st = obspy.read(os.path.join(testdata_preprocessed, "*.mseed"))

    # Preprocess
    pre_filt = (0.05, 0.1, 1, 2)

    # Define sampling rate as a function of the pre_filt
    sampling_rate = pre_filt[3] * 2.5
    npts = int(1000 * sampling_rate)

    # Response output
    rr_output = "VEL"
    water_level = 60

    # Initialize output stream
    pst = st.copy()

    # Length of the seismograms
    length_in_s = npts / sampling_rate

    print(length_in_s, sampling_rate, npts)

    # Preprocess the stream
    pst = preprocess(
        pst,
        starttime=origin.time,
        length_in_s=length_in_s,
        sps=sampling_rate,
        inv=inv,
        pre_filt=pre_filt,
        water_level=water_level,
        rr_output=rr_output,
        interpolate=False,
    )

    for _tr in pst:

        net = _tr.stats.network
        sta = _tr.stats.station
        comp = _tr.stats.component

        tr = correct_st.select(network=net, station=sta, component=comp)[0]
        assert np.allclose(_tr.data, tr.data)
        assert net == tr.stats.network
        assert sta == tr.stats.station
        assert comp == tr.stats.component
        print(tr.stats.network, tr.stats.station, tr.stats.component)


def test_preprocess_interpolation():
    # Get event
    event = obspy.read_events(testdata_event)[0]

    # Get event data
    origin = event.origins[0]

    # Read inventory
    inv = obspy.read_inventory(os.path.join(testdata_stations, "*.xml"))

    # Read stream
    st = obspy.read(os.path.join(testdata_raw, "*.mseed"))

    # Read the preprocessed stream
    correct_st = obspy.read(os.path.join(testdata_interp, "*.mseed"))

    # Preprocess
    pre_filt = (0.05, 0.1, 1, 2)

    # Define sampling rate as a function of the pre_filt
    sampling_rate = pre_filt[3] * 2.5
    npts = int(1000 * sampling_rate)

    # Response output
    rr_output = "VEL"
    water_level = 60

    # Copy base stream
    pst = st.copy()

    # Length of the seismograms
    length_in_s = npts / sampling_rate

    print(length_in_s, sampling_rate, npts)

    # Preprocess the stream
    pst = preprocess(
        pst,
        starttime=origin.time,
        length_in_s=length_in_s,
        sps=sampling_rate,
        inv=inv,
        pre_filt=pre_filt,
        water_level=water_level,
        rr_output=rr_output,
        filter=False,
        interpolate=True,
    )

    for _tr in pst:

        net = _tr.stats.network
        sta = _tr.stats.station
        comp = _tr.stats.component

        tr = correct_st.select(network=net, station=sta, component=comp)[0]
        assert np.allclose(_tr.data, tr.data)
        assert net == tr.stats.network
        assert sta == tr.stats.station
        assert comp == tr.stats.component
        print(tr.stats.network, tr.stats.station, tr.stats.component)


def test_preprocess_corrects_misoriented_ne_channels():
    """Channels named N/E whose inventory azimuths are not 0/90 deg are
    rotated to true north/east, as obsproclib does (obspy's default
    rotate("->ZNE") would leave them alone). II.BFO's BHN/BHE are given
    azimuths 5.6/95.6 deg in a copy of the inventory; the result must be the
    true-north result seen from the turned sensor, rotated back."""
    import copy

    event = obspy.read_events(testdata_event)[0]
    inv = obspy.read_inventory(os.path.join(testdata_stations, "*.xml"))
    st = obspy.read(os.path.join(testdata_raw, "*.mseed")).select(network="II", station="BFO")
    kw = dict(starttime=event.origins[0].time, length_in_s=1000.0, sps=5.0,
              pre_filt=(0.05, 0.1, 1, 2), water_level=60, rr_output="VEL", interpolate=False)
    ref = preprocess(st.copy(), inv=inv, **kw)

    a = 5.6
    turned = copy.deepcopy(inv)
    for ch in turned.select(network="II", station="BFO")[0][0]:
        if ch.code == "BHN":
            ch.azimuth = a
        elif ch.code == "BHE":
            ch.azimuth = 90.0 + a
    out = preprocess(st.copy(), inv=turned, **kw)

    n0, e0 = (ref.select(component=c)[0].data for c in "NE")
    n1, e1 = (out.select(component=c)[0].data for c in "NE")
    # the same records, now read as a sensor turned by a: true N/E are the
    # original N/E rotated by a
    r = np.radians(a)
    n_true = n0 * np.cos(r) - e0 * np.sin(r)
    e_true = n0 * np.sin(r) + e0 * np.cos(r)
    peak = max(np.abs(n0).max(), np.abs(e0).max())
    assert np.abs(n1 - n_true).max() <= 1e-10 * peak
    assert np.abs(e1 - e_true).max() <= 1e-10 * peak
    z0, z1 = (x.select(component="Z")[0].data for x in (ref, out))
    assert np.abs(z1 - z0).max() <= 1e-12 * np.abs(z0).max()      # recomputed, same values
