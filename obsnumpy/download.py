import logging
import os
from typing import Union, List
from obspy import Inventory
from obspy import UTCDateTime
from obspy.clients.fdsn.mass_downloader import (
    RectangularDomain,
    Restrictions,
    MassDownloader,
)


def download_waveforms_to_storage(
    datastorage: str,
    starttime: UTCDateTime,
    endtime: UTCDateTime,
    minimum_length: float = 0.9,
    reject_channels_with_gaps: bool = True,
    network: Union[str, None] = "IU,II,G",
    station: Union[str, None] = None,
    channel: Union[str, None] = None,
    location: Union[str, None] = None,
    providers: Union[List[str], None] = ["IRIS"],
    minlatitude: float = -90.0,
    maxlatitude: float = 90.0,
    minlongitude: float = -180.0,
    maxlongitude: float = 180.0,
    location_priorities=None,
    channel_priorities=None,
    limit_stations_to_inventory: Union[Inventory, None] = None,
    waveform_storage: str = None,
    station_storage: str = None,
    logfile: str = None,
    **kwargs,
):

    domain = RectangularDomain(
        minlatitude=minlatitude,
        maxlatitude=maxlatitude,
        minlongitude=minlongitude,
        maxlongitude=maxlongitude,
    )

    # Create Dictionary with the settings
    rdict = dict(
        starttime=starttime,
        endtime=endtime,
        reject_channels_with_gaps=True,
        # Trace needs to be almost full length
        minimum_length=minimum_length,
        network=network,
        station=station,
        location=location,
        channel=channel,
        location_priorities=location_priorities,
        channel_priorities=channel_priorities,
        limit_stations_to_inventory=limit_stations_to_inventory,
    )

    # Remove unset settings
    if not location_priorities:
        rdict.pop("location_priorities")
    if not channel_priorities:
        rdict.pop("channel_priorities")

    restrictions = Restrictions(**rdict)

    # Datastorage:
    if waveform_storage is None:
        waveform_storage = os.path.join(datastorage, "waveforms")
    if station_storage is None:
        station_storage = os.path.join(datastorage, "stations")

    # Get the logger from the obspy package
    logger = logging.getLogger("obspy.clients.fdsn.mass_downloader")

    # Setup the logger to print to file instead of stdout/-err
    if logfile is not None:
        # Remove Stream handler (prints stuff to stdout)
        logger.handlers = []

        # Add File handler (prints stuff to file)
        fh = logging.FileHandler(logfile, mode="w")
        fh.setLevel(logging.DEBUG)

        # Add file handler
        logger.addHandler(fh)

    # Create massdownloader
    mdl = MassDownloader(providers=providers)
    logger.debug(f"\n")
    logger.debug(f"{' Downloading data to: ':*^72}")
    logger.debug(f"MSEEDs: {waveform_storage}")
    logger.debug(f"XMLs:   {station_storage}")

    mdl.download(
        domain,
        restrictions,
        mseed_storage=waveform_storage,
        stationxml_storage=station_storage,
        **kwargs,
    )

    logger.debug("\n")
    logger.debug(72 * "*")
    logger.debug("\n")
