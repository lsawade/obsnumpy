import numpy as np
import obspy
from dataclasses import dataclass, fields
from copy import deepcopy

# from .utils import reindex_dataclass


@dataclass()
class AttrDict:
    """Dataclass to store attributes for a dataset."""

    a: int | None = None


@dataclass
class Stations:

    codes: np.ndarray
    latitudes: np.ndarray | None = None
    longitudes: np.ndarray | None = None
    elevations: np.ndarray | None = None
    depths: np.ndarray | None = None
    azimuths: np.ndarray | None = None
    back_azimuths: np.ndarray | None = None
    distances: np.ndarray | None = None
    attributes: AttrDict | None = None

    def __post_init__(self):

        print(self)

        attr = self.__dict__.keys()

        for _attname in attr:

            _att = self.__getattribute__(_attname)
            if _att is not None:
                if len(self.codes) != len(_att):
                    raise ValueError(
                        f"Number of stations and {_attname}s do not match."
                    )

    def copy(self):
        return deepcopy(self)

    def __len__(self):
        return len(self.codes)


@dataclass
class Meta:
    """Dataclass to store metadata for a dataset."""

    starttime: obspy.UTCDateTime
    npts: int
    delta: float
    components: list
    stations: Stations
    origin: obspy.UTCDateTime = None

    @classmethod
    def from_dict(cls, meta_dict: dict):
        """Create a Meta object from a dictionary.
        The meta dictionary is expected to have following keys:

        - starttime: obspy.UTCDateTime
        - npts: int
        - delta: float
        - components: list
        - stations: list
        - latitudes: list
        - longitudes: list
        - origin: obspy.UTCDateTime


        """
        if "origin" not in meta_dict:
            meta_dict["origin"] = None

        return cls(
            starttime=meta_dict["starttime"],
            npts=meta_dict["npts"],
            delta=meta_dict["delta"],
            components=meta_dict["components"],
            stations=Stations(
                codes=meta_dict["stations"],
                latitudes=meta_dict["latitudes"],
                longitudes=meta_dict["longitudes"],
                elevations=meta_dict["elevations"],
            ),
            origin=meta_dict["origin"],
        )

    def to_dict(self):
        return {
            "starttime": self.starttime,
            "npts": self.npts,
            "delta": self.delta,
            "components": self.components,
            "stations": self.stations.codes,
            "latitudes": self.stations.latitudes,
            "longitudes": self.stations.longitudes,
            "elevations": self.stations.elevations,
            "origin": self.origin,
        }


#     def
#         # Check overlapping stations
#         istations = list(set(obs_meta['stations']).intersection(set(gf__meta['stations'])))


#     def isolate_stations(array, meta, stations):
#         idx = np.array([meta['stations'].index(station) for station in stations])
#         print(idx)
#         outmeta = osl.utils.reindex_dict(meta, idx)

#     return array[idx, ...], outmeta

# obs_array, obs_meta = isolate_stations(obs_array, obs_meta, istations)
# gf__array, gf_meta = isolate_stations(gf__array, gf__meta, istations)
