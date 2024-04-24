import sys, inspect
import numpy as np
import warnings


def taper(
    data,
    max_percentage,
    sampling_rate,
    type="cosine",
    max_length=None,
    side="both",
    **kwargs,
):
    """
    Taper the trace.

    Optional (and sometimes necessary) options to the tapering function can
    be provided as kwargs. See respective function definitions in
    `Supported Methods`_ section below.

    :type type: str
    :param type: Type of taper to use for detrending. Defaults to
        ``'hann'``.  See the `Supported Methods`_ section below for
        further details.
    :type max_percentage: None, float
    :param max_percentage: Decimal percentage of taper at one end (ranging
        from 0. to 0.5).
    :type max_length: None, float
    :param max_length: Length of taper at one end in seconds.
    :type side: str
    :param side: Specify if both sides should be tapered (default, "both")
        or if only the left half ("left") or right half ("right") should be
        tapered.

    .. note::

        To get the same results as the default taper in SAC, use
        `max_percentage=0.05` and leave `type` as `hann`.

    .. note::

        If both `max_percentage` and `max_length` are set to a float, the
        shorter tape length is used. If both `max_percentage` and
        `max_length` are set to `None`, the whole trace will be tapered.

    .. note::

        This operation is performed in place on the actual data arrays. The
        raw data is not accessible anymore afterwards. To keep your
        original data, use :meth:`~obspy.core.trace.Trace.copy` to create
        a copy of your trace object.
        This also makes an entry with information on the applied processing
        in ``stats.processing`` of this trace.

    .. rubric:: _`Supported Methods`

    ``'cosine'``
        Cosine taper, for additional options like taper percentage see:
        :func:`obspy.signal.invsim.cosine_taper`.
    ``'barthann'``
        Modified Bartlett-Hann window. (uses:
        :func:`scipy.signal.windows.barthann`)
    ``'bartlett'``
        Bartlett window. (uses: :func:`scipy.signal.windows.bartlett`)
    ``'blackman'``
        Blackman window. (uses: :func:`scipy.signal.windows.blackman`)
    ``'blackmanharris'``
        Minimum 4-term Blackman-Harris window. (uses:
        :func:`scipy.signal.windows.blackmanharris`)
    ``'bohman'``
        Bohman window. (uses: :func:`scipy.signal.windows.bohman`)
    ``'boxcar'``
        Boxcar window. (uses: :func:`scipy.signal.windows.boxcar`)
    ``'chebwin'``
        Dolph-Chebyshev window.
        (uses: :func:`scipy.signal.windows.chebwin`)
    ``'flattop'``
        Flat top window. (uses: :func:`scipy.signal.windows.flattop`)
    ``'gaussian'``
        Gaussian window with standard-deviation std. (uses:
        :func:`scipy.signal.windows.gaussian`)
    ``'general_gaussian'``
        Generalized Gaussian window. (uses:
        :func:`scipy.signal.windows.general_gaussian`)
    ``'hamming'``
        Hamming window. (uses: :func:`scipy.signal.windows.hamming`)
    ``'hann'``
        Hann window. (uses: :func:`scipy.signal.windows.hann`)
    ``'kaiser'``
        Kaiser window with shape parameter beta. (uses:
        :func:`scipy.signal.windows.kaiser`)
    ``'nuttall'``
        Minimum 4-term Blackman-Harris window according to Nuttall.
        (uses: :func:`scipy.signal.windows.nuttall`)
    ``'parzen'``
        Parzen window. (uses: :func:`scipy.signal.windows.parzen`)
    ``'slepian'``
        Slepian window. (uses: :func:`scipy.signal.windows.slepian`)
    ``'triang'``
        Triangular window. (uses: :func:`scipy.signal.windows.triang`)
    """
    type = type.lower()
    side = side.lower()
    side_valid = ["both", "left", "right"]
    npts = data.shape[-1]
    if side not in side_valid:
        raise ValueError("'side' has to be one of: %s" % side_valid)
    # retrieve function call from entry points
    current_module = sys.modules[__name__]
    current_callables = dict(inspect.getmembers(current_module, inspect.isfunction))
    for key, value in current_callables.items():
        print(key)
        if callable(value) and value.__module__ == __name__ and key == f"_{type}":
            func = value
            break
    else:
        raise ValueError("Taper type '%s' not supported." % type)

    # store all constraints for maximum taper length
    max_half_lenghts = []
    if max_percentage is not None:
        max_half_lenghts.append(int(max_percentage * npts))
    if max_length is not None:
        max_half_lenghts.append(int(max_length * sampling_rate))
    if np.all([2 * mhl > npts for mhl in max_half_lenghts]):
        msg = (
            "The requested taper is longer than the trace. "
            "The taper will be shortened to trace length."
        )
        warnings.warn(msg)
    # add full trace length to constraints
    max_half_lenghts.append(int(npts / 2))
    # select shortest acceptable window half-length
    wlen = min(max_half_lenghts)
    # obspy.signal.cosine_taper has a default value for taper percentage,
    # we need to override is as we control percentage completely via npts
    # of taper function and insert ones in the middle afterwards
    if type == "cosine":
        kwargs["p"] = 1.0
    # tapering. tapering functions are expected to accept the number of
    # samples as first argument and return an array of values between 0 and
    # 1 with the same length as the data
    if 2 * wlen == npts:
        taper_sides = func(2 * wlen, **kwargs)
    else:
        taper_sides = func(2 * wlen + 1, **kwargs)
    if side == "left":
        taper = np.hstack((taper_sides[:wlen], np.ones(npts - wlen)))
    elif side == "right":
        taper = np.hstack(
            (np.ones(npts - wlen), taper_sides[len(taper_sides) - wlen :])
        )
    else:
        taper = np.hstack(
            (
                taper_sides[:wlen],
                np.ones(npts - 2 * wlen),
                taper_sides[len(taper_sides) - wlen :],
            )
        )

    # Convert data if it's not a floating point type.
    if not np.issubdtype(data.dtype, np.floating):
        data = np.require(data, dtype=np.float64)

    # Apply taper to specified axis
    data *= taper[..., :]
    return data


def _cosine(npts, p=0.1, freqs=None, flimit=None, halfcosine=True, sactaper=False):
    """
    Cosine Taper.

    :type npts: int
    :param npts: Number of points of cosine taper.
    :type p: float
    :param p: Decimal percentage of cosine taper (ranging from 0 to 1). Default
        is 0.1 (10%) which tapers 5% from the beginning and 5% form the end.
    :rtype: float NumPy :class:`~numpy.ndarray`
    :return: Cosine taper array/vector of length npts.
    :type freqs: NumPy :class:`~numpy.ndarray`
    :param freqs: Frequencies as, for example, returned by fftfreq
    :type flimit: list(float, float, float, float) or
        tuple(float, float, float, float)
    :param flimit: The list or tuple defines the four corner frequencies
        (f1, f2, f3, f4) of the cosine taper which is one between f2 and f3 and
        tapers to zero for f1 < f < f2 and f3 < f < f4.
    :type halfcosine: bool
    :param halfcosine: If True the taper is a half cosine function. If False it
        is a quarter cosine function.
    :type sactaper: bool
    :param sactaper: If set to True the cosine taper already tapers at the
        corner frequency (SAC behavior). By default, the taper has a value
        of 1.0 at the corner frequencies.

    .. rubric:: Example

    >>> tap = cosine_taper(100, 1.0)
    >>> tap2 = 0.5 * (1 + np.cos(np.linspace(np.pi, 2 * np.pi, 50)))
    >>> np.allclose(tap[0:50], tap2)
    True
    >>> npts = 100
    >>> p = 0.1
    >>> tap3 = cosine_taper(npts, p)
    >>> (tap3[int(npts*p/2):int(npts*(1-p/2))]==np.ones(int(npts*(1-p)))).all()
    True
    """
    if p < 0 or p > 1:
        msg = "Decimal taper percentage must be between 0 and 1."
        raise ValueError(msg)
    if p == 0.0 or p == 1.0:
        frac = int(npts * p / 2.0)
    else:
        frac = int(npts * p / 2.0 + 0.5)

    if freqs is not None and flimit is not None:
        fl1, fl2, fl3, fl4 = flimit
        idx1 = np.argmin(abs(freqs - fl1))
        idx2 = np.argmin(abs(freqs - fl2))
        idx3 = np.argmin(abs(freqs - fl3))
        idx4 = np.argmin(abs(freqs - fl4))
    else:
        idx1 = 0
        idx2 = frac - 1
        idx3 = npts - frac
        idx4 = npts - 1
    if sactaper:
        # in SAC the second and third
        # index are already tapered
        idx2 += 1
        idx3 -= 1

    # Very small data lengths or small decimal taper percentages can result in
    # idx1 == idx2 and idx3 == idx4. This breaks the following calculations.
    if idx1 == idx2:
        idx2 += 1
    if idx3 == idx4:
        idx3 -= 1

    # the taper at idx1 and idx4 equals zero and
    # at idx2 and idx3 equals one
    cos_win = np.zeros(npts)
    if halfcosine:
        # cos_win[idx1:idx2+1] =  0.5 * (1.0 + np.cos((np.pi * \
        #    (idx2 - np.arange(idx1, idx2+1)) / (idx2 - idx1))))
        cos_win[idx1 : idx2 + 1] = 0.5 * (
            1.0
            - np.cos(
                (np.pi * (np.arange(idx1, idx2 + 1) - float(idx1)) / (idx2 - idx1))
            )
        )
        cos_win[idx2 + 1 : idx3] = 1.0
        cos_win[idx3 : idx4 + 1] = 0.5 * (
            1.0
            + np.cos(
                (np.pi * (float(idx3) - np.arange(idx3, idx4 + 1)) / (idx4 - idx3))
            )
        )
    else:
        cos_win[idx1 : idx2 + 1] = np.cos(
            -(np.pi / 2.0 * (float(idx2) - np.arange(idx1, idx2 + 1)) / (idx2 - idx1))
        )
        cos_win[idx2 + 1 : idx3] = 1.0
        cos_win[idx3 : idx4 + 1] = np.cos(
            (np.pi / 2.0 * (float(idx3) - np.arange(idx3, idx4 + 1)) / (idx4 - idx3))
        )

    # if indices are identical division by zero
    # causes NaN values in cos_win
    if idx1 == idx2:
        cos_win[idx1] = 0.0
    if idx3 == idx4:
        cos_win[idx3] = 0.0
    return cos_win
