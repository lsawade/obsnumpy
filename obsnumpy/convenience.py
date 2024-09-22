import obspy
import typing as tp


def preprocess(
    st: obspy.Stream,
    starttime: float,
    length_in_s: float,
    sps: float,
    inv: obspy.Inventory = None,
    freqmin: float = 0.004,
    freqmax: float = 1 / 17.0,
    pre_filt: tp.Tuple[float, float, float, float] = (0.001, 0.005, 1.0, 1 / 0.5),
    water_level: float = 100.0,
    rr_output: str = "DISP",
    interpolate: bool = False,
    filter: bool = False,
):
    """Function that does bare minimum processing of observed seismograms
    to make them ready for further processing.

    Parameters
    ----------
    st : obspy.Stream
        Input stream of seismograms
    starttime : float
        Start time of the seismograms
    length_in_s : float
        Length of the seismograms in seconds
    sps : float
        Sampling rate of the seismograms in Hz
    inv : obspy.Inventory, optional
        Inventory if the traces require response removal, by default None
    freqmin : float, optional
        minimum frequency for the bandpass filter, by default 0.004
    freqmax : float, optional
        maxinum frequency for the bandpass filtera, by default 1/17.0
    pre_filt : tp.Tuple[float, float, float, float], optional
        pre_filt tuple for response removal, by default (0.001, 0.005, 1.0, 1 / 0.5)
    water_level : float, optional
        deconvolution water_level for the response removal, by default 100.0
    rr_output : str, optional
        output of the response removal, by default "DISP"
    interpolate : bool, optional
        Interpolate the seismograms to the given sampling rate, by default False

    Returns
    -------
    Dataset

    """
    
    st = st.copy()

    # Generic
    st.detrend("linear")
    st.detrend("demean")

    # Taper the seismograms
    st.taper(max_percentage=0.05, type="cosine")

    # Remove response if inventory is given given
    if inv:
        st.remove_response(
            inventory=inv, output=rr_output, pre_filt=pre_filt, water_level=water_level
        )
        
        # Get station codes
        stations = set()
        for tr in st:
            station = (tr.stats.network, tr.stats.station)
            stations.add(station)
        
        # Create a new stream to add rotated seismograms to
        new_st = obspy.Stream()
        
        for station in stations:
            st_temp = st.select(network=station[0], station=station[1])
            
            try:
                st_temp.rotate("->ZNE", inventory=inv)
                new_st += st_temp
            except ValueError as e:
                print(f"Error rotating seismograms for station {station}: {e}")
        
        print(f'Reduced stream length from {len(st)} to {len(new_st)}')
        
        # Reassign the stream
        st = new_st

        for tr in st:
            station = inv.select(network=tr.stats.network, station=tr.stats.station)[0][
                0
            ]
            tr.stats.latitude = station.latitude
            tr.stats.longitude = station.longitude

    if filter:
        # Filter the seismograms
        st.filter(
            "bandpass", freqmin=freqmin, freqmax=freqmax, corners=4, zerophase=True
        )

    # Interpolate the seismograms
    if interpolate:
        print(f"Interpolating seismograms to {sps} sps")
        new_st = obspy.Stream()
        
        for tr in st:
            try:    
                tr.interpolate(
                    sampling_rate=sps, starttime=starttime, npts=int((length_in_s) * sps)
                )
                new_st += tr
            except ValueError as e:
                print(f"Error interpolating seismogram {tr.id}: {e}")
            
        st = new_st
        
        # Check whether all seismograms have the same length
        for tr in st:
            if len(tr.data) != int((length_in_s) * sps):
                print(f"Seismogram {tr.id} has length {len(tr.data)}")
            
    return st
                    
