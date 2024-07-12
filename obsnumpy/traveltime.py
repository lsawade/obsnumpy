from __future__ import annotations
import typing as tp
import numpy as np
from obspy.taup import TauPyModel
from scipy import signal
from .meta import AttrDict

if tp.TYPE_CHECKING:
    from .dataset import Dataset


def get_arrivals(cmt, ds: Dataset, phase='P'):
    """Compute teleseismic arrivals for a given cmt and distances and azimuths."""

    # Make model
    model = TauPyModel(model='ak135')

    # Just get the meta data for the stations to add the arrivals
    stations = ds.meta.stations

    # Check if the attributes field exists
    if not hasattr(stations, 'attributes') or stations.attributes is None:
        stations.attributes = AttrDict()

    # Reference to the meta data
    attd = stations.attributes
    attd.origin_time = cmt.origin_time

    if phase == 'P':
        phaselist = ['P']
    elif phase == 'anyP':
        # With these three phases taup should return an arrival time for any epicentral distance
        phaselist = ['P', 'Pdiff', 'PKP', 'PKIKP']
    elif phase == 'S':
        phaselist = ['S']
    elif phase == 'anyS':
        # With these three phases taup should return an arrival time for any epicentral distance
        phaselist = ['S', 'ScS', 'SKS', 'Sdiff', 'SKIKS']
    elif phase == 'S':
        phaselist = ['S']
    elif phase == 'Rayleigh':
        minvel = 3.0
        maxvel = 4.2
    elif phase == 'Love':
        minvel = 3.0
        maxvel = 5.0
    else:
        raise ValueError(f'Phase {phase} not recognized')

    # Check if the arrivals field exists
    if not hasattr(attd, 'arrivals'):
        attd.arrivals = AttrDict()

    # Get separate function to compute min and max surface wave windows.
    if phase in ['Rayleigh', 'Love']:

        minarrivals = get_surface_wave_arrivals(stations.distances, minvel, ncircles=1)[:, 0]
        maxarrivals = get_surface_wave_arrivals(stations.distances, maxvel, ncircles=1)[:, 0]

        if not hasattr(attd.arrivals, phase):
            attd.arrivals[phase] =  AttrDict()

        attd.arrivals[phase]['min'] = minarrivals
        attd.arrivals[phase]['max'] = maxarrivals

    else:

        # Get the arrivals
        phase_arrivals = []
        for dist in stations.distances:

            arrivals = model.get_travel_times(source_depth_in_km=cmt.depth,
                                                distance_in_degree=dist,
                                                phase_list=phaselist)
            if len(arrivals) > 0:
                arrivals = sorted(arrivals, key=lambda x: x.time)
                phase_arrivals.append(arrivals[0].time)
            else:
                phase_arrivals.append(np.nan)

        attd.arrivals[phase] = np.array(phase_arrivals)


def get_surface_wave_arrivals(dist_in_deg, vel, ncircles=1):
    """
    Calculate the arrival time of surface waves, based on the distance
    and velocity range (min_vel, max_vel).
    This function will calculate both minor-arc and major-arc surface
    waves. It further calcualte the surface orbit multiple times
    if you set the ncircles > 1.

    Returns the list of surface wave arrivals in time order.
    """

    earth_circle = 111.11*360.0
    dt1 = earth_circle / vel

    # 1st arrival: minor-arc arrival
    minor_dist_km = 111.11*dist_in_deg  # major-arc distance
    t_minor = minor_dist_km / vel

    # 2nd arrival: major-arc arrival
    major_dist_km = 111.11*(360.0 - dist_in_deg)
    t_major = major_dist_km / vel

    # prepare the arrival list
    arrivals = np.zeros((len(dist_in_deg), ncircles*2))

    for i in range(ncircles):

        ts = t_minor + i * dt1
        arrivals[:, 2*i] = ts

        ts = t_major + i * dt1
        arrivals[:, 2*i + 1] = ts

    return arrivals


def select_traveltime_subset(ds: Dataset, mindist=30.0, maxdist=np.inf, component='Z', phase='P'):
    """Selects subset of the array based on the distance range and component.
    And where we have a valid arrival time."""

    # Just get short forms of the relevant objects
    stations = ds.meta.stations
    arrivals =  ds.meta.stations.attributes.arrivals

    # Get distance selection station is at least mindist away and at most
    # maxdist away
    selection = (stations.distances > mindist ) & (stations.distances < maxdist)

    if phase == 'Ptrain':
        selection = selection & (~np.isnan(arrivals.anyP)) & (~np.isnan(arrivals.anyS))

    elif phase == 'P':
        selection = selection & (~np.isnan(arrivals.P))

    elif phase == 'Strain':
        selection = selection & (~np.isnan(arrivals.anyS))

        if component in ['Z', 'R']:
            selection = selection & (~np.isnan(arrivals.Rayleigh.min))
        elif component == 'T':
            selection = selection & (~np.isnan(arrivals.Love.min))
        else:
            raise ValueError("Component must be Z, R or T")

    elif phase == 'S':
        selection = selection & (~np.isnan(arrivals.P))

    elif phase == 'Rayleigh':
        selection = selection & (~np.isnan(arrivals.Rayleigh.min))
        selection = selection & (~np.isnan(arrivals.Rayleigh.max))

    elif phase == 'Love':
        selection = selection & (~np.isnan(arrivals.Love.min))
        selection = selection & (~np.isnan(arrivals.Love.max))

    elif phase == 'body':
        if component in ['Z', 'R']:
            selection = selection & (~np.isnan(arrivals.anyP)) & (~np.isnan(arrivals.Rayleigh.min))
        elif component == 'T':
            selection = selection & (~np.isnan(arrivals.anyP)) & (~np.isnan(arrivals.Love.min))

    else:
        raise ValueError("Component must be Z, R or T")

    # Get the indeces that match the selection
    idx = np.where(selection)[0]

    # Sort the subset by distance
    idx2 = np.argsort(stations.distances[idx])

    # Get final indeces
    pos = idx[idx2]

    # Return reindexed subset
    return ds.subset(stations=pos, components=component)


def get_windows(arrivals, phase='P', window=200.0):
    """Uses specfic traveltimes to compute windows for different phases, wave trains, etc.

    Parameters
    ----------
    arrivals : AttrDict
        Meta data for corresponding traces.
    phase : str, optional
        phase or wavetrain ID, by default 'P'
    window : float, optional
        time window for capturing specific windows only, e.g. P or S,
        by default 200.0

    Returns
    -------
    tuple(ndarray, ndarray)
        start_arrivals, end_arrivals
    """

    # First we need to compute the slice for each trace
    if phase in ['Love', 'Rayleigh']:

        start_arrivals = arrivals[phase].max
        end_arrivals = arrivals[phase].min

    elif phase == 'Ptrain':

        start_arrivals = arrivals.anyP
        end_arrivals = arrivals.anyS

    elif phase == 'P':

        start_arrivals = arrivals.P
        end_arrivals = start_arrivals + window

    elif phase == 'Strain':

        start_arrivals = arrivals.anyS
        end_arrivals = arrivals.Love.max

    elif phase == 'S':

        start_arrivals = arrivals.S
        end_arrivals = start_arrivals + window

    elif phase == 'body':

        start_arrivals = arrivals.anyP - 200
        end_arrivals = arrivals.Love.max

    return start_arrivals, end_arrivals



def construct_taper(npts, taper_type="tukey", alpha=0.2):
    """
    Construct taper based on npts

    :param npts: the number of points
    :param taper_type:
    :param alpha: taper width
    :return:
    """
    taper_type = taper_type.lower()
    _options = ['hann', 'boxcar', 'tukey', 'hamming']
    if taper_type not in _options:
        raise ValueError("taper type option: %s" % taper_type)
    if taper_type == "hann":
        taper = signal.windows.hann(npts)
    elif taper_type == "boxcar":
        taper = signal.windows.boxcar(npts)
    elif taper_type == "hamming":
        taper = signal.windows.hamming(npts)
    elif taper_type == "tukey":
        taper = signal.windows.tukey(npts, alpha=alpha)
    else:
        raise ValueError("Taper type not supported: %s" % taper_type)
    return taper


def taper_dataset(ds: Dataset, phase, tshift, taper_perc=0.5):

    outds = ds.copy()
    # Get sampling interval
    delta = outds.meta.delta
    npts = outds.meta.npts
    length_in_s = npts * delta
    t = np.arange(0,npts) * delta - tshift

    # Get the corresponding windows
    start_arrival, end_arrival = get_windows(outds.meta.stations.attributes.arrivals, phase=phase)

    # Get the taper based on the window
    outds.data = np.zeros_like(ds.data)

    for _i, (_start, _end) in enumerate(zip(start_arrival, end_arrival)):
        print(_end - _start)
        idx = np.where((_start <= t) & (t <= _end))[0]
        npts = len(idx)
        outds.data[_i, 0, idx] = ds.data[_i, 0, idx] * construct_taper(npts, taper_type="tukey", alpha=taper_perc)

    return outds