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
    preprocess(
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
    preprocess(
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
