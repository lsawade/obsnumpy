from copy import deepcopy
import numpy as np
import typing as tp
import obspy
from obspy.geodetics.base import gps2dist_azimuth

from dataclasses import fields, is_dataclass
from .meta import AttrDict


def distazbaz(lat0, lon0, lat, lon):

    # Compute azimuth
    def azfunc(lat1, lon1):
        return gps2dist_azimuth(lat0, lon0, lat1, lon1)

    # Vecotrize the function (not faster, but no loop necessary)
    vazfunc = np.vectorize(azfunc)

    return vazfunc(lat, lon)


def inventory2stationdict(inv: obspy.Inventory) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    """Very simple function to get a dictionary with the station coordinates

    Parameters
    ----------
    inv : obspy.Inventory
        Normal obspy inventory

    Returns
    -------
    tp.Dict[str, tp.Dict[str, tp.Any]]
        _description_
    """
    # Initialize
    station_dict = {}

    for network in inv:
        if network.code not in station_dict:
            station_dict[network.code] = {}

        for station in network:
            if station.code not in station_dict[network.code]:
                station_dict[network.code][station.code] = {
                    "latitude": station.latitude,
                    "longitude": station.longitude,
                }
            else:
                pass
    return station_dict


def stream2stationdict(st: obspy.Stream) -> tp.Dict[str, tp.Dict[str, tp.Any]]:
    """Very simple function to get a dictionary with the station coordinates

    Parameters
    ----------
    inv : obspy.Stream
        Normal obspy stream

    Returns
    -------
    tp.Dict[str, tp.Dict[str, tp.Any]]
        _description_
    """
    # Initialize
    station_dict = {}

    # Loop over traces
    for tr in st:

        if tr.stats.network not in station_dict:
            station_dict[tr.stats.network] = []

        if tr.stats.station not in station_dict[tr.stats.network]:
            station_dict[tr.stats.network].append(tr.stats.station)

    return station_dict


def reindex(indc, idx, N_original, debug=False):

    dc = deepcopy(indc)


    # Get the fields
    if is_dataclass(dc):
        fields = dc.__dataclass_fields__.keys()

    elif isinstance(dc, dict) or isinstance(dc, AttrDict):
        fields = dc.keys()

    #Loop over fields
    for field in fields:

        do_index=True

        # Get attribute depending on the input type
        if is_dataclass(dc):
            att = dc.__getattribute__(field)

        elif isinstance(dc, dict):
            att = dc[field]

        elif isinstance(dc, AttrDict):
            att = dc.__getattribute__(field)

        if debug:
            print(field, type(att))

        # Recursive calling of the original function if the attribute is a dataclass
        if is_dataclass(att):
            if debug:
                print("--> Entering ")

            newatt = reindex(att, idx, N_original, debug=debug)

        elif isinstance(att, dict):
            if debug:
                print("--> Entering ")

            newatt = reindex(att, idx, N_original, debug=debug)

        elif isinstance(att, AttrDict):
            if debug:
                print("--> Entering ")

            newatt = reindex(att, idx, N_original, debug=debug)

        # Index list or numpy array
        elif isinstance(att, list) and len(att) == N_original:
            if debug:
                print("--> Reindexing", field, type(att), len(att), N_original)

            newatt = [att[i] for i in idx]

        elif isinstance(att, np.ndarray) and len(att) == N_original:
            if debug:
                print("--> Reindexing", field, type(att), len(att), N_original)

            newatt = att[idx]

        # Index single value
        else:
            if debug:
                print("--> Not reindexing.")

            do_index=False

        if do_index:
            if isinstance(dc, dict):
                dc[field] = newatt
            elif isinstance(dc, AttrDict):
                dc.__setattr__(field, newatt)
            else:
                dc.__setattr__(field, newatt)

    return dc



def L2(obs, syn, normalize=True):

    l2 = np.sum((syn.data - obs.data)**2, axis=-1)

    if normalize:
        l2 /= np.sum(obs.data**2, axis=-1)

    return l2


# %%
# remove where the misfit is too large

def remove_misfits(obs, syn):

    # Get misfits
    misfit = L2(obs, syn, normalize=True)
    
    # Get the threshold
    misfit_threshold = np.quantile(misfit, 0.975)
    
    # Removal of anomalously low or large data traces
    ratio = np.sum(syn.data**2,axis=-1) / np.sum(obs.data**2,axis=-1)
    
    # Get ratio threshold
    ratio_threshold_above = np.quantile(ratio, 0.95)
    ratio_threshold_below = np.quantile(ratio, 0.05)
    
    # Get indices
    idx = np.where((misfit < misfit_threshold) & (ratio < ratio_threshold_above) & (ratio > ratio_threshold_below))[0]
    
    return obs.subset(stations=idx), syn.subset(stations=idx), idx
