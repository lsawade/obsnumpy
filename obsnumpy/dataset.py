from __future__ import annotations
import obspy
import numpy as np
import typing as tp
from copy import deepcopy
from . import utils
from collections import OrderedDict
from .meta import Meta
from .meta import Stations
from .process import filter, interpolate, taper
from .convenience import preprocess
from warnings import warn

from dataclasses import dataclass


@dataclass
class Dataset:

    data: np.ndarray
    meta: Meta

    def __post_init__(self):

        # Check whether data has the right shape
        data_shape = self.data.shape
        if len(data_shape) != 3:
            raise ValueError(
                f"Data must have shape (ntraces, 3, npts), but has shape {data_shape}"
            )

        if data_shape[1] != 3:
            raise ValueError(
                f"Data must have shape (ntraces, 3, npts), but has shape {data_shape}"
            )

    @classmethod
    def from_stream(
        cls,
        st: obspy.Stream,
        components: list | None = ["N", "E", "Z"],
        inv: obspy.Inventory | None = None,
        event_latitude: float | None = None,
        event_longitude: float | None = None,
        station_dict: tp.Dict[str, tp.Dict[str, tp.Any]] | None = None,
    ):
        """This function expectes the stream to already be preprocessed, such
        that the data is filtered, detrended, demeaned, and possibly
        interpolated. They should be uniformly sampled in time, and have the
        same starttime.

        We expect that this function will mainly be used for observed data that
        has been preprocessed in Obspy, which is then converted to an Stream
        object.

        Parameters
        ----------
        st : obspy.Stream
            input stream
        components : list, optional
            components to be stored in array format, by default ['N', 'E', 'Z']
        inv : obspy.Inventory, optional
            inventory with station information, by default None
        event_latitude : float, optional
            event latitude, by default None
        event_longitude : float, optional
            event longitude, by default None
        station_latitudes : dict, optional
            dictionary with station latitudes, by default None
        station_longitudes : dict, optional
            dictionary with station longitudes, by default None
        """
        if components is None:
            components = ["N", "E", "Z"]
        else:
            components = components.copy()

        # First get all unqiue station ids
        stations = set()
        for tr in st:

            station = f"{tr.stats.network}.{tr.stats.station}"
            stations.add(station)

        # Get stations with less than 3 components
        remove = []
        for i, station in enumerate(stations):
            _net, _sta = station.split(".")
            substream = st.select(network=_net, station=_sta)

            if len(substream) < 3 or len(substream) > 3:
                print(f"Station {station} has {len(substream)} components. Removing.")
                remove.append(station)

        # Remove those stations
        for station in remove:
            stations.remove(station)

        # Get number of stations, points and components to get array size
        Nstations = len(stations)
        Npts = st[0].stats.npts
        Ncomponents = len(components)

        # Create array
        array = np.zeros((Nstations, Ncomponents, Npts))

        # Final stations
        fstations = []
        for i, station in enumerate(stations):
            _net, _sta = station.split(".")
            substream = st.select(network=_net, station=_sta)

            if len(substream) < 3 or len(substream) > 3:
                print(
                    f"Station {station} has {len(substream)} components. \n   --> Station selection not working properly!."
                )
                continue

            for j, component in enumerate(components):
                subtr = substream.select(component=component)
                if len(subtr) > 0:
                    array[i, j, :] = subtr[0].data
                else:
                    print(f"Did not find component {component} for station {station}")

            fstations.append(station)

        # Metadata
        if station_dict is None and inv is not None:
            station_dict = utils.inventory2stationdict(inv)

        # Use station dictionary to get latitudes and longitudes
        if station_dict is not None:

            # Get latitudes and longitudes
            latitudes = []
            longitudes = []

            # Get latitudes and longitudes
            for _station in fstations:
                _net, _sta = _station.split(".")
                latitudes.append(station_dict[_net][_sta]["latitude"])
                longitudes.append(station_dict[_net][_sta]["longitude"])

            # Convert to numpy arrays
            latitudes = np.array(latitudes)
            longitudes = np.array(longitudes)

        else:

            latitudes = None
            longitudes = None

        # Compute Geometry if the event latitude and longitude are set and
        # station latitudes and longitudes are set
        if (event_latitude is not None and event_longitude is not None) and (
            latitudes is not None and longitudes is not None
        ):
            distances, azimuths, back_azimuths = utils.distazbaz(
                event_latitude, event_longitude, latitudes, longitudes
            )

            # Convert distance to degrees
            distances = distances / 1000.0 / 111.11

        else:
            distances, azimuths, back_azimuths = None, None, None

        elevations = None
        depths = None
        # Define the stations attributes as computed
        station_attributes = Stations(
            codes=np.array(fstations),
            latitudes=latitudes,
            longitudes=longitudes,
            elevations=elevations,
            depths=depths,
            azimuths=azimuths,
            distances=distances,
            back_azimuths=back_azimuths,
        )

        meta = Meta(
            starttime=st[0].stats.starttime,
            npts=Npts,
            delta=st[0].stats.delta,
            components=components,
            stations=station_attributes,
            origin=None,
        )

        print("STREAM CREATION:", components)
        return cls(data=array, meta=meta)

    @classmethod
    def from_raw(
        cls,
        *args,
        **kwargs,
    ):
        """Create a dataset from raw data. The main function that makes the input
        data uniform is the preprocess function from the convenience module.

        Check the input parameters for the preprocess function. It controls
        starttime and length_in_s, and the sampling rate, etc.

        There are certain parameters that are only used for the creation of the
        dataset:
        - components: list of components to be stored in the array
        - event_latitude: latitude of the event
        - event_longitude: longitude of the event
        - station_dict: dictionary with station information
        These can be parsed as kwargs so that they are not used as input to the
        preprocess function.
        """

        # Pop items that are only inpus to self.from_stream
        if "components" in kwargs:
            components = kwargs.pop("components")
        else:
            components = None

        if "event_latitude" in kwargs:
            event_latitude = kwargs.pop("event_latitude")
        else:
            event_latitude = None

        if "event_longitude" in kwargs:
            event_longitude = kwargs.pop("event_longitude")
        else:
            event_longitude = None

        if "station_dict" in kwargs:
            station_dict = kwargs.pop("station_dict")
        else:
            station_dict = None

        # Split stream from other arguments
        st = args[0]
        args = args[1:]

        # Make sure that st is a stream
        if not isinstance(st, obspy.Stream):
            raise ValueError("Input data must be an Obspy Stream.")

        if interpolate not in kwargs:
            kwargs["interpolate"] = True
        else:
            if kwargs["interpolate"] is False:

                # Check if all traces have the same number of points,
                # starttime and sampling rate
                npts = st[0].stats.npts
                delta = st[0].stats.delta
                starttime = st[0].stats.starttime

                for tr in st:
                    if tr.stats.npts != npts:
                        raise ValueError(
                            "If interpolate is explicitly set to False, all traces must have the same number of points."
                        )
                    if tr.stats.delta != delta:
                        raise ValueError(
                            "If interpolate is explicitly set to False, all traces must have the same sampling rate."
                        )
                    if tr.stats.starttime != starttime:
                        raise ValueError(
                            "If interpolate is explicitly set to False, all traces must have the same starttime."
                        )

        # Preprocess stream
        preprocess(st, *args, **kwargs)

        # Get inv if it is in kwargs
        if "inv" in kwargs:
            inv = kwargs.pop("inv")
        else:
            inv = None

        # Create array stream
        return cls.from_stream(
            st,
            inv=inv,
            components=components,
            event_latitude=event_latitude,
            event_longitude=event_longitude,
            station_dict=station_dict,
        )

    def rotate(self, rtype="->RT", event_latitude=None, event_longitude=None):
        """If rotation is performed components are replaced. The replacement
        map for each type of rotations is:

        ->RT:
            NE -> RT -- That is R replaces N and T replaces E

        """

        # Check whether rotation type is supported
        rtypes = ["->RT"]

        # Throw error if the rotatino type is not supported
        if rtype not in rtypes:
            raise ValueError(
                f"Rotation type not understood. Please use one of {rtypes}"
            )

        # Check which components are present
        components = self.meta.components

        if rtype == "->RT" and all([comp in components for comp in "NE"]):
            fulltype = "NE->RT"
        else:
            raise ValueError("Only NE->RT is supported at the moment.")

        if fulltype == "NE->RT":

            # Check current components

            if not all([comp in "ZNE" for comp in self.meta.components]):
                raise ValueError("Components must be  ZNE to rotate to RTZ")

            # Get Back azimuths
            if self.meta.stations.back_azimuths is not None:

                baz = self.meta.stations.back_azimuths

            else:

                if event_latitude is None or event_longitude is None:
                    raise ValueError(
                        "Event latitude and longitude must be set "
                        "to compute back azimuths if back azimuths "
                        "are not computed."
                    )

                warn(
                    "Back azimuths not set. Computing back_azimuths from origin "
                    "and latitudes and longitudes."
                )

                # Get latitudes
                latitudes = self.meta.stations.latitudes
                longitudes = self.meta.stations.longitudes

                # Compute azimuth and distance
                dist, az, baz = utils.distazbaz(
                    event_latitude, event_longitude, latitudes, longitudes
                )

                # Compute distance in degrees
                self.meta.stations.distances = dist / 1000.0 / 111.11
                self.meta.stations.azimuths = az
                self.meta.stations.back_azimuths = baz

            # Convert azimuth to radians
            baz = np.radians(baz)

            # Compute rotation matrix
            Nidx = self.meta.components.index("N")
            Eidx = self.meta.components.index("E")

            dRdN = -np.cos(baz)[:, np.newaxis] * self.data[:, Nidx, :]
            dRdE = -np.sin(baz)[:, np.newaxis] * self.data[:, Eidx, :]
            dTdN = np.sin(baz)[:, np.newaxis] * self.data[:, Nidx, :]
            dTdE = -np.cos(baz)[:, np.newaxis] * self.data[:, Eidx, :]

            # R replaces N and T replaces E
            self.data[:, Nidx, :] = dRdN + dRdE
            self.data[:, Eidx, :] = dTdN + dTdE

            # Update components
            self.meta.components[Nidx] = "R"
            self.meta.components[Eidx] = "T"

    def filter(self, ftype="bandpass", **kwargs):
        """
        Filter data.

        This function is a wrapper around the other filter functions in this module.
        It selects the correct filter function based on the input arguments.

        :type data: numpy.ndarray
        :param data: Data to filter.
        :param kwargs: Keyword arguments for the filter functions.
        :return: Filtered data.
        """
        if ftype == "bandpass":

            # Double check whether freqmin and freqmax are set
            if "freqmin" not in kwargs or "freqmax" not in kwargs:
                raise ValueError(
                    "For bandpass filter, 'freqmin' and 'freqmax' must be set."
                )

            # Get them
            freqmin = kwargs.pop("freqmin")
            freqmax = kwargs.pop("freqmax")

            # Filter trace
            self.data = filter.bandpass(
                self.data, freqmin, freqmax, df=1 / self.meta.delta, **kwargs
            )

        elif ftype == "lowpass":

            # Double check whether freq is set
            if "freq" not in kwargs:
                raise ValueError("For lowpass filter, 'freq' must be set.")

            # Get freq
            freq = kwargs.pop("freq")

            # Filter the data
            self.data = filter.lowpass(
                self.data, freq, df=1 / self.meta.delta, **kwargs
            )

        elif ftype == "highpass":

            # Double check whether freq is set
            if "freq" not in kwargs:
                raise ValueError("For lowpass filter, 'freq' must be set.")

            # Get freq
            freq = kwargs.pop("freq")

            # Filter the data
            self.data = filter.highpass(
                self.data, freq, df=1 / self.meta.delta, **kwargs
            )

        else:
            raise ValueError(
                f"{ftype} is not a valid filter type. Use 'bandpass', 'lowpass', or 'highpass'."
            )

    def interpolate(
        self,
        starttime: obspy.UTCDateTime,
        sampling_rate: float,
        npts: int,
        method="weighted_average_slopes",
    ):
        """Interpolate the stream to a new starttime, sampling rate and number of points.

        Parameters
        ----------
        starttime : obspy.UTCDateTime
            new starttime
        sampling_rate : float, optional
            new sampling rate, by default None
        npts : int, optional
            new number of samples, by default None
        method : str, optional
            interpolation method, available methods: "linear", "cubic",
            "weighted_average_slopes", by default "weighted_average_slopes"
        """

        new_dt = 1 / sampling_rate

        self.data = interpolate.interp1d(
            self.data,
            old_dt=self.meta.delta,
            old_start=self.meta.starttime,
            new_start=starttime,
            new_dt=new_dt,
            new_npts=npts,
            itype=method,
            axis=-1,
        )

        self.meta.starttime = starttime
        self.meta.delta = 1 / sampling_rate
        self.meta.npts = npts

    def taper(
        self, max_percentage, type="hann", max_length=None, side="both", **kwargs
    ):
        """Taper the data.

        Parameters
        ----------
        taper_type : str, optional
            taper type, by default "cosine"
        max_percentage : float, optional
            maximum percentage of taper, by default 0.05
        """
        self.data = taper.taper(
            self.data,
            max_percentage,
            sampling_rate=1 / self.meta.delta,
            type=type,
            max_length=None,
            side="both",
            **kwargs,
        )

    def __len__(self):
        return len(self.meta.stations)

    def subset(self, idx) -> Dataset:
        """Given a set of indeces this function will create a new dataset with.
        only the selected stations."""

        # Reindex the input metadata
        new_meta = utils.reindex_dataclass(self.meta, idx, len(self))

        # New data
        new_data = self.data[idx, ...].copy()

        # Check whether len of data is equal the length station codes
        if len(new_meta.stations.codes) != new_data.shape[0]:
            raise ValueError(
                "Number of stations and number of data points do not match."
            )

        return Dataset(data=new_data, meta=new_meta)

    def intersect(self, other: Dataset) -> tp.Tuple[Dataset, Dataset]:
        """You did well copilot!"""

        # Get common stations
        common = set(self.meta.stations.codes).intersection(
            set(other.meta.stations.codes)
        )

        # Get indeces
        idx1 = [self.meta.stations.codes.index(code) for code in common]
        idx2 = [other.meta.stations.codes.index(code) for code in common]

        # Return subsets of the datasets
        return self.subset(idx1), other.subset(idx2)
