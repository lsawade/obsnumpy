import os
import obspy
import numpy as np
from obsnumpy import Dataset
import obsnumpy.utils
from obspy.geodetics.base import gps2dist_azimuth


testfolder = os.path.dirname(os.path.abspath(__file__))
testdatafolder = os.path.join(testfolder, "data")
testdata_raw = os.path.join(testfolder, "data", "raw")
testdata_interp = os.path.join(testfolder, "data", "interp")
testdata_stations = os.path.join(testfolder, "data", "stations")
testdata_event = os.path.join(testfolder, "data", "germany_event.cmt")


def test_read_interpolated_stream():

    # Read inventory
    inv = obspy.read_inventory(os.path.join(testdata_stations, "*.xml"))

    # Read stream
    st = obspy.read(os.path.join(testdata_interp, "*.mseed"))

    # Convert stream to ArrayStream
    ds = Dataset.from_stream(st, inv=inv)

    for i, station in enumerate(ds.meta.stations.codes):

        net, sta = station.split(".")

        for j, comp in enumerate(ds.meta.components):
            assert comp in ["N", "E", "Z"]

            tr = st.select(network=net, station=sta, component=comp)[0]
            assert np.allclose(tr.data, ds.data[i, j, :])
            assert net == tr.stats.network
            assert sta == tr.stats.station
            assert comp == tr.stats.component
            print(tr.stats.network, tr.stats.station, tr.stats.component)


def test_rotate_interpolated_stream():

    # Get event data
    event = obspy.read_events(testdata_event)[0]
    origin = event.origins[0]
    event_latitude = origin.latitude
    event_longitude = origin.longitude

    # Read inventory
    inv = obspy.read_inventory(os.path.join(testdata_stations, "*.xml"))

    # Convert inv to a dictionary of stations and locations
    station_dict = obsnumpy.utils.inventory2stationdict(inv)

    # Read stream
    st = obspy.read(os.path.join(testdata_interp, "*.mseed"))

    # Stream to rotate
    rst = st.copy()

    # Rotate stream
    for network, stations in station_dict.items():
        for station, sdict in stations.items():

            subst = rst.select(network=network, station=station)
            _, _, b = gps2dist_azimuth(
                event_latitude,
                event_longitude,
                sdict["latitude"],
                sdict["longitude"],
            )
            subst.rotate("NE->RT", inventory=inv, back_azimuth=b)

    # Convert stream to ArrayStream

    ds = Dataset.from_stream(st, inv=inv)

    # Rotate ArrayStream
    ds.rotate("->RT", event_latitude, event_longitude)

    # Loop ove
    for i, (station) in enumerate(ds.meta.stations.codes):

        net, sta = station.split(".")

        for j, comp in enumerate(ds.meta.components):
            assert comp in ["R", "T", "Z"]

            tr = rst.select(network=net, station=sta, component=comp)[0]
            assert np.allclose(tr.data, ds.data[i, j, :])
            assert net == tr.stats.network
            assert sta == tr.stats.station
            assert comp == tr.stats.component
            print(tr.stats.network, tr.stats.station, tr.stats.component)

    del ds


def test_bandpass_interpolated_stream():

    # Read inventory
    inv = obspy.read_inventory(os.path.join(testdata_stations, "*.xml"))

    # Read stream
    st = obspy.read(os.path.join(testdata_interp, "*.mseed"))

    # Stream to rotate
    rst = st.copy()
    rst.filter("bandpass", freqmin=0.1, freqmax=1.0, corners=4, zerophase=True)

    # Convert stream to Dataset
    ds = Dataset.from_stream(st, inv=inv, components=["N", "E", "Z"])

    print("\n BANDPASS COMPONENTS", ds.meta.components, "\n")

    # Bandpass filter
    ds.filter("bandpass", freqmin=0.1, freqmax=1.0, corners=4, zerophase=True)

    # Loop over stations
    for i, station in enumerate(ds.meta.stations.codes):

        net, sta = station.split(".")

        for j, comp in enumerate(ds.meta.components):
            assert comp in ["N", "E", "Z"]

            tr = rst.select(network=net, station=sta, component=comp)[0]
            assert np.allclose(tr.data, ds.data[i, j, :])
            assert net == tr.stats.network
            assert sta == tr.stats.station
            assert comp == tr.stats.component
            print(tr.stats.network, tr.stats.station, tr.stats.component)


def test_lowpass_interpolated_stream():

    # Read inventory
    inv = obspy.read_inventory(os.path.join(testdata_stations, "*.xml"))

    # Read stream
    st = obspy.read(os.path.join(testdata_interp, "*.mseed"))

    # Stream to rotate
    rst = st.copy()
    rst.filter("lowpass", freq=1.0, corners=4, zerophase=True)

    # Convert stream to Dataset
    lds = Dataset.from_stream(st, inv=inv)

    # Bandpass filter
    lds.filter("lowpass", freq=1.0, corners=4, zerophase=True)

    # Loop over stations
    for i, station in enumerate(lds.meta.stations.codes):

        net, sta = station.split(".")

        for j, comp in enumerate(lds.meta.components):
            assert comp in ["N", "E", "Z"]

            tr = rst.select(network=net, station=sta, component=comp)[0]
            assert np.allclose(tr.data, lds.data[i, j, :])
            assert net == tr.stats.network
            assert sta == tr.stats.station
            assert comp == tr.stats.component
            print(tr.stats.network, tr.stats.station, tr.stats.component)


def test_highpass_interpolated_stream():

    # Read inventory
    inv = obspy.read_inventory(os.path.join(testdata_stations, "*.xml"))

    # Read stream
    st = obspy.read(os.path.join(testdata_interp, "*.mseed"))

    # Stream to rotate
    rst = st.copy()
    rst.filter("highpass", freq=1.0, corners=4, zerophase=True)

    # Convert stream to Dataset
    ds = Dataset.from_stream(st, inv=inv)

    # Bandpass filter
    ds.filter("highpass", freq=1.0, corners=4, zerophase=True)

    # Loop over stations
    for i, station in enumerate(ds.meta.stations.codes):

        net, sta = station.split(".")

        for j, comp in enumerate(ds.meta.components):
            assert comp in ("N", "E", "Z")

            tr = rst.select(network=net, station=sta, component=comp)[0]
            assert np.allclose(tr.data, ds.data[i, j, :])
            assert net == tr.stats.network
            assert sta == tr.stats.station
            assert comp == tr.stats.component
            print(tr.stats.network, tr.stats.station, tr.stats.component)


def test_interpolate_weighted_average_slopes_interpolated_stream():

    # Read inventory
    inv = obspy.read_inventory(os.path.join(testdata_stations, "*.xml"))

    # Read stream
    st = obspy.read(os.path.join(testdata_interp, "*.mseed"))

    # Stream to interpolate
    rst = st.copy()
    rst.interpolate(
        starttime=st[0].stats.starttime,
        sampling_rate=1 / st[0].stats.delta * 2,
        npts=2 * st[0].stats.npts - 2,
        method="weighted_average_slopes",
    )

    # Convert stream to Dataset
    ds = Dataset.from_stream(st.copy(), inv=inv)

    # Interpolate
    ds.interpolate(
        starttime=st[0].stats.starttime,
        sampling_rate=1 / ds.meta.delta * 2,
        npts=2 * st[0].stats.npts - 2,
        method="weighted_average_slopes",
    )
    # Loop over stations
    for i, station in enumerate(ds.meta.stations.codes):

        net, sta = station.split(".")

        for j, comp in enumerate(ds.meta.components):
            assert comp in ["N", "E", "Z"]

            tr = rst.select(network=net, station=sta, component=comp)[0]
            print(
                comp,
                "MISFIT:",
                np.sum(np.abs(tr.data - ds.data[i, j, :]) / np.abs(tr.data)),
            )

            print("TRTYPE", type(tr.data), "DSTYPE", type(ds.data[i, j, :]))
            assert np.allclose(tr.data, ds.data[i, j, :])
            assert net == tr.stats.network
            assert sta == tr.stats.station
            assert comp == tr.stats.component
            print(tr.stats.network, tr.stats.station, tr.stats.component)


def test_interpolate_linear_interpolated_stream():

    # Read inventory
    inv = obspy.read_inventory(os.path.join(testdata_stations, "*.xml"))

    # Read stream
    st = obspy.read(os.path.join(testdata_interp, "*.mseed"))

    # Stream to interpolate
    rst = st.copy()
    rst.interpolate(
        starttime=st[0].stats.starttime,
        sampling_rate=1 / st[0].stats.delta * 2,
        npts=2 * st[0].stats.npts - 2,
        method="linear",
    )

    # Convert stream to Dataset
    ds = Dataset.from_stream(st.copy(), inv=inv)

    # Interpolate
    ds.interpolate(
        starttime=st[0].stats.starttime,
        sampling_rate=1 / ds.meta.delta * 2,
        npts=2 * st[0].stats.npts - 2,
        method="linear",
    )
    # Loop over stations
    for i, station in enumerate(ds.meta.stations.codes):

        net, sta = station.split(".")

        for j, comp in enumerate(ds.meta.components):
            assert comp in ["N", "E", "Z"]

            tr = rst.select(network=net, station=sta, component=comp)[0]
            print(
                comp,
                "MISFIT:",
                np.sum(np.abs(tr.data - ds.data[i, j, :]) / np.abs(tr.data)),
            )

            print("TRTYPE", type(tr.data), "DSTYPE", type(ds.data[i, j, :]))
            assert np.allclose(tr.data, ds.data[i, j, :])
            assert net == tr.stats.network
            assert sta == tr.stats.station
            assert comp == tr.stats.component
            print(tr.stats.network, tr.stats.station, tr.stats.component)


def test_interpolate_cubic_interpolated_stream():

    # Read inventory
    inv = obspy.read_inventory(os.path.join(testdata_stations, "*.xml"))

    # Read stream
    st = obspy.read(os.path.join(testdata_interp, "*.mseed"))

    # Stream to interpolate
    rst = st.copy()
    rst.interpolate(
        starttime=st[0].stats.starttime,
        sampling_rate=1 / st[0].stats.delta * 2,
        npts=2 * st[0].stats.npts - 2,
        method="cubic",
    )

    # Convert stream to Dataset
    ds = Dataset.from_stream(st.copy(), inv=inv)

    # Interpolate
    ds.interpolate(
        starttime=st[0].stats.starttime,
        sampling_rate=1 / ds.meta.delta * 2,
        npts=2 * st[0].stats.npts - 2,
        method="cubic",
    )
    # Loop over stations
    for i, station in enumerate(ds.meta.stations.codes):

        net, sta = station.split(".")

        for j, comp in enumerate(ds.meta.components):
            assert comp in ["N", "E", "Z"]

            tr = rst.select(network=net, station=sta, component=comp)[0]
            print(
                comp,
                "MISFIT:",
                np.sum(np.abs(tr.data - ds.data[i, j, :]) / np.abs(tr.data)),
            )

            print("TRTYPE", type(tr.data), "DSTYPE", type(ds.data[i, j, :]))
            assert np.allclose(tr.data, ds.data[i, j, :])
            assert net == tr.stats.network
            assert sta == tr.stats.station
            assert comp == tr.stats.component
            print(tr.stats.network, tr.stats.station, tr.stats.component)


def test_taper_cosine_interpolated_stream():

    # Read inventory
    inv = obspy.read_inventory(os.path.join(testdata_stations, "*.xml"))

    # Read stream
    st = obspy.read(os.path.join(testdata_interp, "*.mseed"))

    # Stream to interpolate
    rst = st.copy()
    rst.taper(type="cosine", max_percentage=0.4)

    # Convert stream to Dataset
    ds = Dataset.from_stream(st.copy(), inv=inv)

    # Interpolate
    ds.taper(type="cosine", max_percentage=0.4)

    # Loop over stations
    for i, station in enumerate(ds.meta.stations.codes):

        net, sta = station.split(".")

        for j, comp in enumerate(ds.meta.components):
            assert comp in ["N", "E", "Z"]

            tr = rst.select(network=net, station=sta, component=comp)[0]
            print(
                comp,
                "MISFIT:",
                np.sum(np.abs(tr.data - ds.data[i, j, :]) / np.abs(tr.data)),
            )

            print("TRTYPE", type(tr.data), "DSTYPE", type(ds.data[i, j, :]))
            assert np.allclose(tr.data, ds.data[i, j, :])
            assert net == tr.stats.network
            assert sta == tr.stats.station
            assert comp == tr.stats.component
            print(tr.stats.network, tr.stats.station, tr.stats.component)


def test_from_raw():
    # Get event
    event = obspy.read_events(testdata_event)[0]

    # Get event data
    origin = event.origins[0]

    # Read inventory
    inv = obspy.read_inventory(os.path.join(testdata_stations, "*.xml"))

    # Read stream
    st = obspy.read(os.path.join(testdata_raw, "*.mseed"))

    # Read the correct interpolated stream
    correct_st = obspy.read(os.path.join(testdata_interp, "*.mseed"))

    # Preprocess
    pre_filt = (0.05, 0.1, 1, 2)

    # Define sampling rate as a function of the pre_filt
    sampling_rate = pre_filt[3] * 2.5
    npts = int(1000 * sampling_rate)

    # Response output
    rr_output = "VEL"

    # Convert stream to ArrayStream
    ds = Dataset.from_raw(
        st,
        starttime=origin.time,
        length_in_s=npts / sampling_rate,
        sps=sampling_rate,
        inv=inv,
        pre_filt=pre_filt,
        rr_output=rr_output,
        filter=False,
        interpolate=True,
    )

    for i, station in enumerate(ds.meta.stations.codes):

        net, sta = station.split(".")

        for j, comp in enumerate(ds.meta.components):
            assert comp in ["N", "E", "Z"]

            tr = correct_st.select(network=net, station=sta, component=comp)[0]
            assert np.allclose(tr.data, ds.data[i, j, :])
            assert net == tr.stats.network
            assert sta == tr.stats.station
            assert comp == tr.stats.component
            print(tr.stats.network, tr.stats.station, tr.stats.component)


if __name__ == "__main__":
    test_read_interpolated_stream()
    test_rotate_interpolated_stream()
    test_lowpass_interpolated_stream()
    test_bandpass_interpolated_stream()
    test_highpass_interpolated_stream()
