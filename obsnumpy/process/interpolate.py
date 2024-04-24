import numpy as np
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d as scp_interp1d


def interp1d(
    data,
    old_dt,
    old_start,
    new_dt,
    new_start,
    new_npts,
    itype="linear",
    axis=-1,
    **kwargs,
):
    """1-D interpolation conveniene function.

    Parameters
    ----------
    data : np.ndarray
        data in a multidimensional array
    old_dt : float
        sampling interval of the data
    old_start : UTCDateTime
        starttime of the trace
    new_dt : float
        new samplings interval
    new_start : UTCDateTime
        new starttime
    new_npts : int
        new number of points
    itype : str, optional
        interpoaltion type. right new 'linear' or 'cubic' are supported, by default 'linear'

    Returns
    -------
    data_new : np.ndarray
        data interpolated to new sampling rate, starttime and number of points

    Raises
    ------
    ValueError
        _description_
    """

    # interpolate
    t = np.arange(0, data.shape[-1] * old_dt, old_dt)

    # Setup new time vector
    t0_new = new_start - old_start
    tnew = np.arange(t0_new, new_npts * new_dt + t0_new, new_dt)
    print("New time vector:", len(tnew))

    # Interpolate
    if itype == "linear":
        data_new = scp_interp1d(t, data, axis=-1, **kwargs)(tnew)

    elif itype == "cubic":
        f = CubicSpline(t, data, axis=axis, **kwargs)
        data_new = f(tnew)

    elif itype == "weighted_average_slopes":
        data_new = weighted_average_slopes(t, data, tnew)

    else:
        raise ValueError("Interpolation type not recognized.")

    return data_new


def weighted_average_slopes(t, data, tnew):
    r"""
    Implements the weighted average slopes interpolation scheme proposed in
    [Wiggins1976]_ for evenly sampled data. The scheme guarantees that there
    will be no additional extrema after the interpolation in contrast to
    spline interpolation.

    The slope :math:`s_i` at each knot is given by a weighted average of the
    adjacent linear slopes :math:`m_i` and :math:`m_{i+j}`:

    .. math::

        s_i = (w_i m_i + w_{i+1} m_{i+1}) / (w_i + w_{i+1})

    where

    .. math::

        w = 1 / max \left\{ \left\| m_i \\right\|, \epsilon \\right\}

    The value at each data point and the slope are then plugged into a
    piecewise continuous cubic polynomial used to evaluate the interpolated
    sample points.

    :type data: array_like
    :param data: Array to interpolate.
    :type old_start: float
    :param old_start: The start of the array as a number.
    :type old_start: float
    :param old_dt: The time delta of the current array.
    :type new_start: float
    :param new_start: The start of the interpolated array. Must be greater
        or equal to the current start of the array.
    :type new_dt: float
    :param new_dt: The desired new time delta.
    :type new_npts: int
    :param new_npts: The new number of samples.
    """
    old_dt = t[1] - t[0]
    m = np.diff(data, axis=-1) / old_dt
    w = np.abs(m)
    w = 1.0 / np.clip(w, np.spacing(np.ones_like(w)), np.max(w, axis=-1, keepdims=True))

    slope = np.empty(data.shape, dtype=np.float64)
    slope[..., 0] = m[..., 0]
    slope[..., 1:-1] = (w[..., :-1] * m[..., :-1] + w[..., 1:] * m[..., 1:]) / (
        w[..., :-1] + w[..., 1:]
    )
    slope[..., -1] = m[..., -1]

    # If m_i and m_{i+1} have opposite signs then set the slope to zero.
    # This forces the curve to have extrema at the sample points and not
    # in-between.
    sign_change = np.diff(np.sign(m), axis=-1).astype(bool)
    slope[..., 1:-1][sign_change] = 0.0

    # derivatives = np.empty(* data.shape, 2), dtype=np.float64)
    # derivatives[:, 0] = data
    # derivatives[:, 1] = slope

    # Create interpolated value using hermite interpolation. In this case
    # it is directly applicable as the first derivatives are known.
    # Using scipy.interpolate.piecewise_polynomial_interpolate() is too
    # memory intensive
    # return_data = np.empty(np.hstack((*data.shape[:-1], len(tnew))), dtype=np.float64)
    # clibsignal.hermite_interpolation(data, slope, tnew, return_data,
    #                                  len(data), len(return_data), old_dt,
    #                                  0.0)
    return hermite_interpolation(data, slope, tnew, old_dt, 0.0)


def hermite_interpolation(data, slope, tnew, h, old_start):
    """Copied from C function in obspy and adapted to numpy.

    Multiple operations can be consolidated here which will remove memory
    consumption. But for now I'll keep it.

    a_0 = data[..., i_0];
    a_1 = data[..., i_1];
    b_minus_1 = h * slope[..., i_0];
    b_plus_1 = h * slope[..., i_1];
    b_0 = a_1 - a_0;
    c_0 = b_0 - b_minus_1;
    c_1 = b_plus_1 - b_0;
    d_0 = c_1 - c_0;

    --->

    a_0 = data[..., i_0];
    b_0 = data[..., i_1] - a_0;
    c_0 = b_0 - h * slope[..., i_0];
    d_0 = h * slope[..., i_1] - b_0 - c_0;

    """

    # h = old dt
    # x_start = old_start

    # This gets the the distance between the new time and the old time
    i = (tnew - old_start) / h

    # This gets the index of the lower bound of the interval
    i_0 = i.astype(int)

    # This gets the index of the upper bound of the interval
    i_1 = i_0 + 1

    # # No need to interpolate if exactly at start of the interval.
    # exact_match = np.where(i == i_0.astype(float))

    # # This should skip interpolation everywhere we have an exact match in time
    # return_data[..., exact_match] = data[..., i_0[exact_match]]

    # This gets the difference in index between the new time and the lower bound of the interval
    # So t is between 0 and 1
    t = i - i_0

    # These are now just adapted to get the same shape as the output data
    # as
    a_0 = data[..., i_0]
    a_1 = data[..., i_1]
    b_minus_1 = h * slope[..., i_0]
    b_plus_1 = h * slope[..., i_1]
    b_0 = a_1 - a_0
    c_0 = b_0 - b_minus_1
    c_1 = b_plus_1 - b_0
    d_0 = c_1 - c_0

    return a_0 + (b_0 + (c_0 + d_0 * t) * (t - 1.0)) * t
