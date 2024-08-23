from copy import deepcopy
import numpy as np
import typing as tp
import datetime
import obspy
from obspy.geodetics.base import gps2dist_azimuth

from dataclasses import fields, is_dataclass
from .meta import AttrDict
from . import traveltime as tt

def log(msg):
    length = 80
    length_msg = len(msg)
    length_right = (length - length_msg) - 1
    if length_right < 0:
        fill = ""
    else:
        fill = "-" * length_right
    print(f'[{datetime.datetime.now()}] {msg} {fill}', flush=True)
    
    
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


def reindex(indc, idx, N_original, debug=True):

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


def misfit_reduction(m0, m1): 
    # Compute overall misfit reduction
    total_misfit_reduction = 100*(np.sum(m0) - np.sum(m1))/np.sum(m0)
    return total_misfit_reduction    


# %%
# remove where the misfit is too large

def compute_snr(ds, tshift, period=17, phase='P'):
    """Computes the SNR base on the the integrated squares and absmax of the seismic signal 
    pre- and post P arrival. It is important to compute this prior to tapering!!!!
    The noise end index is stored in the meta data, so are the SNR values.
    snr_int and snr_max
    """
    
    # Get the index of the P arrival
    starts, ends = tt.get_windows(ds.meta.stations.attributes.arrivals, phase=phase)
    
    # Get the index of the P arrival
    idx = np.argmin(np.abs(ds.t[None, :]-(starts[:, None]+tshift-period)), axis=-1)
    
    
    # Assign noise end indeces
    ds.meta.stations.attributes.noise_end = idx
    
    # Since the noise is of different length for each trace, we need to loop over the traces
    snr_int = np.zeros((ds.data.shape[0], ds.data.shape[1]))
    snr_max = np.zeros((ds.data.shape[0], ds.data.shape[1]))
    
    for i in range(ds.data.shape[0]):
            
            # Get the noise
            noise = ds.data[i, :, 0:idx[i]]
            
            # Get the signal
            signal = ds.data[i, :, idx[i]:]
            
            # Compute the SNR and normalize by lenght of signal and noise
            snr_int[i, :] = np.sum(signal**2, axis=-1)/signal.shape[-1] / (np.sum(noise**2, axis=-1)/noise.shape[-1])
    
            # Compute the SNR by the maximum of the absolute values
            snr_max[i, :] = np.max(np.abs(signal), axis=-1)/np.max(np.abs(noise), axis=-1)
            
    # Add to meta data
    ds.meta.stations.attributes.snr_int = snr_int
    ds.meta.stations.attributes.snr_max = snr_max


def remove_snr(ds, 
               snr_int_min_threshold=2.0, snr_int_max_threshold=np.inf, 
               snr_max_min_threshold=2.0, snr_max_max_threshold=np.inf, 
               component='Z'):
    
    comp = ds.meta.components.index(component)
    
    idx = np.where((ds.meta.stations.attributes.snr_int[:, comp] >= snr_int_min_threshold) 
                   & (ds.meta.stations.attributes.snr_int[:, comp] <= snr_int_max_threshold)
                   & (ds.meta.stations.attributes.snr_max[:, comp] >= snr_max_min_threshold)
                   & (ds.meta.stations.attributes.snr_max[:, comp] <= snr_max_max_threshold) 
                   )[0]
    
    return ds.subset(stations=idx), idx
    
    
def remove_misfits(obs, syn, misfit_quantile_threshold=0.975, ratio_quantile_threshold_above=0.95, ratio_quantile_threshold_below=0.05):
    
    # Get misfits
    misfit = L2(obs, syn, normalize=True)
    
    if misfit.shape[0] <= 5:
        return obs, syn, np.arange(len(obs))
    
    # Get the threshold
    misfit_threshold = np.quantile(misfit, misfit_quantile_threshold)
    
    # Print some statistics and the threshold
    log(f"Min, Mean, Median, Max misfit: {np.min(misfit):.2f}, {np.mean(misfit):.2f}, {np.median(misfit):.2f}, {np.max(misfit):.2f}")
    log(f"Misfit threshold: {misfit_threshold:.2f}")
    
    # Removal of anomalously low or large data traces
    ratio = np.sum(syn.data**2,axis=-1) / np.sum(obs.data**2,axis=-1)
    
    # Get ratio threshold
    ratio_threshold_above = np.quantile(ratio, ratio_quantile_threshold_above)
    ratio_threshold_below = np.quantile(ratio, ratio_quantile_threshold_below)
    
    
    log(f"Min, Mean, Median, Max ratio: {np.min(ratio):.2f}, {np.mean(ratio):.2f}, {np.median(ratio):.2f}, {np.max(ratio):.2f}")
    log(f"Ratio thresholds: {ratio_threshold_below:.2f} -- {ratio_threshold_above:.2f}")
    
    # Get indices
    idx = np.where((misfit <= misfit_threshold) & (ratio <= ratio_threshold_above) & (ratio >= ratio_threshold_below))[0]
    
    return obs.subset(stations=idx), syn.subset(stations=idx), idx


def remove_zero_traces(ds):
    
    idx = []
    
    for i in range(ds.data.shape[0]):
        
        zero_trace = []
        
        for j in range(ds.data.shape[1]):
            
            if np.isclose(np.trapz(np.abs(ds.data[i,j,:])), 0.0):
                zero_trace.append(True)
            else:
                zero_trace.append(False)
                
        if not any(zero_trace):
            idx.append(i)
            
    idx = np.array(idx, dtype=int)
    
    return ds.subset(stations=idx)
