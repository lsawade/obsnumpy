import numpy as np
import warnings
from scipy.fftpack import hilbert
from scipy.signal import cheb2ord, cheby2, convolve, get_window, iirfilter, remez

from scipy.signal import sosfilt
from scipy.signal import zpk2sos


def bandpass(data: np.ndarray, freqmin, freqmax, df, corners=4, zerophase=False):
    """
    Butterworth-Bandpass Filter.

    Filter data from ``freqmin`` to ``freqmax`` using ``corners``
    corners.
    The filter uses :func:`scipy.signal.iirfilter` (for design)
    and :func:`scipy.signal.sosfilt` (for applying the filter).

    :type data: numpy.ndarray
    :param data: Data to filter.
    :param freqmin: Pass band low corner frequency.
    :param freqmax: Pass band high corner frequency.
    :param df: Sampling rate in Hz.
    :param corners: Filter corners / order.
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the filter order but zero phase shift in
        the resulting filtered trace.
    :return: Filtered data.
    """
    return _apply_sos(bandpass_sos(freqmin, freqmax, df, corners), data, zerophase)


def highpass(data: np.ndarray, freq, df, corners=4, zerophase=False):
    """
    Butterworth-Highpass Filter.

    Filter data removing data below certain frequency ``freq`` using
    ``corners`` corners.
    The filter uses :func:`scipy.signal.iirfilter` (for design)
    and :func:`scipy.signal.sosfilt` (for applying the filter).

    :type data: numpy.ndarray
    :param data: Data to filter.
    :param freq: Filter corner frequency.
    :param df: Sampling rate in Hz.
    :param corners: Filter corners / order.
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the number of corners but zero phase shift in
        the resulting filtered trace.
    :return: Filtered data.
    """
    return _apply_sos(highpass_sos(freq, df, corners), data, zerophase)


def lowpass(data: np.ndarray, freq, df, corners=4, zerophase=False):
    """
    Butterworth-Lowpass Filter.

    Filter data removing data over certain frequency ``freq`` using ``corners``
    corners.
    The filter uses :func:`scipy.signal.iirfilter` (for design)
    and :func:`scipy.signal.sosfilt` (for applying the filter).

    :type data: numpy.ndarray
    :param data: Data to filter.
    :param freq: Filter corner frequency.
    :param df: Sampling rate in Hz.
    :param corners: Filter corners / order.
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the number of corners but zero phase shift in
        the resulting filtered trace.
    :return: Filtered data.
    """
    return _apply_sos(lowpass_sos(freq, df, corners), data, zerophase)


def _apply_sos(sos, data, zerophase):
    """Apply second-order sections along the last axis, as obspy does: once
    forwards, and with ``zerophase`` once more over the reversed result."""
    firstpass = sosfilt(sos, data, axis=-1)
    if not zerophase:
        return firstpass
    return sosfilt(sos, firstpass[..., ::-1], axis=-1)[..., ::-1]


def bandpass_sos(freqmin, freqmax, df, corners=4):
    """Second-order sections of :func:`bandpass` (obspy's
    ``obspy.signal.filter.bandpass``): a Butterworth band of ``corners``
    corners. A high corner at or above Nyquist gives the highpass sections,
    with obspy's warning."""
    fe = 0.5 * df
    low = freqmin / fe
    high = freqmax / fe
    # raise for some bad scenarios
    if high - 1.0 > -1e-6:
        msg = (
            "Selected high corner frequency ({}) of bandpass is at or "
            "above Nyquist ({}). Applying a high-pass instead."
        ).format(freqmax, fe)
        warnings.warn(msg)
        return highpass_sos(freqmin, df, corners)
    if low > 1:
        msg = "Selected low corner frequency is above Nyquist."
        raise ValueError(msg)
    z, p, k = iirfilter(
        corners, [low, high], btype="band", ftype="butter", output="zpk"
    )
    return zpk2sos(z, p, k)


def highpass_sos(freq, df, corners=4):
    """Second-order sections of :func:`highpass` (obspy's
    ``obspy.signal.filter.highpass``)."""
    fe = 0.5 * df
    f = freq / fe
    # raise for some bad scenarios
    if f > 1:
        msg = "Selected corner frequency is above Nyquist."
        raise ValueError(msg)
    z, p, k = iirfilter(corners, f, btype="highpass", ftype="butter", output="zpk")
    return zpk2sos(z, p, k)


def lowpass_sos(freq, df, corners=4):
    """Second-order sections of :func:`lowpass` (obspy's
    ``obspy.signal.filter.lowpass``)."""
    fe = 0.5 * df
    f = freq / fe
    # raise for some bad scenarios
    if f > 1:
        f = 1.0
        msg = (
            "Selected corner frequency is above Nyquist. "
            + "Setting Nyquist as high corner."
        )
        warnings.warn(msg)
    z, p, k = iirfilter(corners, f, btype="lowpass", ftype="butter", output="zpk")
    return zpk2sos(z, p, k)
