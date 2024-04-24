from copy import deepcopy
import numpy as np
import typing as tp
import obspy
from obspy.geodetics.base import gps2dist_azimuth

from dataclasses import fields, is_dataclass


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


def reindex_dataclass(indc, idx, N_original, debug=False):

    dc = deepcopy(indc)

    # Get the fields
    fields = dc.__dataclass_fields__.keys()

    for field in fields:

        att = dc.__getattribute__(fields)

        if debug:
            print(field, type(att))

        # Recursive calling of the original function if the attribute is a dataclass
        if is_dataclass(att):
            if debug:
                print("--> Entering ")
            dc.__setattribute__(
                field, reindex_dataclass(att, idx, N_original, debug=debug)
            )

        # Index list or numpy array
        elif isinstance(att, list) and len(att) == N_original:
            if debug:
                print("--> Reindexing", field, type(att), len(att), N_original)

            dc.__setattribute__(field, [att[i] for i in idx])

        elif isinstance(att, np.ndarray) and len(att) == N_original:
            if debug:
                print("--> Reindexing", field, type(att), len(att), N_original)

            dc.__setattribute__(field, att[idx])

        # Index single value
        else:
            if debug:
                print("--> Not reindexing.")

    return dc
