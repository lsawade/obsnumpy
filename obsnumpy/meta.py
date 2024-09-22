import numpy as np
import obspy
import dataclasses
from dataclasses import dataclass, fields
from copy import deepcopy
import json

# from .utils import reindex_dataclass


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        try:
            if isinstance(o, AttrDict):
                return o.to_dict()
            elif isinstance(o, np.ndarray):
                return o.tolist()
            elif isinstance(o, obspy.UTCDateTime):
                return o.isoformat()
            elif dataclasses.is_dataclass(o):
                return dataclasses.asdict(o)
        except TypeError as e:
            print(o)
            print(e)
            raise e
            
        return super().default(o)


def convert_to_ndarray(d: dict, length: int) -> dict:
    """This function convertes every ``list`` entry of a ``dict`` of a given length to a numpy array.
    This is used to load a dictionary from a json file and convert it to a dictionary of numpy arrays.
    Unless it's a list 
    """
    
    for key, value in d.items():
        if isinstance(value, list):
            # Only convert numeric lists
            if len(value) == length and isinstance(value[0], (int, float)):
                d[key] = np.array(value)
            else:
                pass
        elif isinstance(value, dict):
            d[key] = convert_to_ndarray(value, length)
            
    return d

class AttrDict(dict):
    """Dictionary to store attributes for a dataset."""
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError from e
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


@dataclass
class Stations:

    codes: np.ndarray
    latitudes: np.ndarray | None = None
    longitudes: np.ndarray | None = None
    elevations: np.ndarray | None = None
    burials: np.ndarray | None = None
    depths: np.ndarray | None = None
    azimuths: np.ndarray | None = None
    back_azimuths: np.ndarray | None = None
    distances: np.ndarray | None = None
    attributes: AttrDict | None = None

    def __post_init__(self):

        attr = self.__dict__.keys()

        for _attname in attr:

            _att = self.__getattribute__(_attname)
            
            
            if _att is not None and _attname != "codes":
                
                if len(self.codes) != len(_att) and _attname != "attributes":                   
                    raise ValueError(
                        f"Number of stations and {_attname}s do not match."
                    )
                if not isinstance(_att, np.ndarray) and _attname != "attributes":
                    try:
                        self.__setattr__(_attname, np.ndarray(_att.tolist()))
                    except Exception as e:
                        print("+++++++++++++++++++++++++++++++++++++++++++++++")
                        print(_attname, _att)
                        print("+++++++++++++++++++++++++++++++++++++++++++++++")
                        raise e
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
    extras: AttrDict = None

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
                codes=np.array(meta_dict["stations"]),
                latitudes=np.array(meta_dict["latitudes"]),
                longitudes=np.array(meta_dict["longitudes"]),
                burials=np.array(meta_dict["burials"]),
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

    def copy(self):
        return deepcopy(self)
    
    def write(self, filename):
        with open(filename, "w") as f:
            json.dump(self, f, cls=EnhancedJSONEncoder)

    @classmethod
    def read(cls, filename):
        
        with open(filename, "r") as f:
            meta_dict = json.load(f)
            
        if "origin" not in meta_dict or meta_dict["origin"] is None:
            meta_dict["origin"] = None
        else:
            meta_dict["origin"] = obspy.UTCDateTime(meta_dict["origin"])
            
        # Fix the starttime
        meta_dict["starttime"] = obspy.UTCDateTime(meta_dict["starttime"])
        
        # Make lists in the stations dictionary into numpy arrays
        statdict = convert_to_ndarray(meta_dict["stations"], len(meta_dict["stations"]['codes']))
        
        # Make attribute dictionary into AttrDict        
        if "attributes" in statdict and statdict["attributes"] is not None:
            __att = AttrDict()
            __att.update(statdict['attributes'])
            statdict['attributes'] = __att
        
        # Fix origin in station dictionary if it exists
        if "origin_time" in statdict["attributes"] and statdict["attributes"]["origin_time"] is not None:
            statdict["attributes"]["origin_time"] = obspy.UTCDateTime(statdict["attributes"]["origin_time"])
    
        # Create station object
        stations = Stations(**statdict)
        
        return cls(
            starttime=meta_dict["starttime"],
            npts=int(meta_dict["npts"]),
            delta=float(meta_dict["delta"]),
            components=meta_dict["components"],
            stations=stations,
            origin=meta_dict["origin"],
        )

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
