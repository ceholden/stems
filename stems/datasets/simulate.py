""" Tools for generating simulated datasets

The intent of this module is to provide tools for generating simulated data
that is usefulf for testing, debugging, and learning. This module is inspired
by the :py:func:`sklearn.datasets.make_classification` and
:py:func:`sklearn.datasets.make_regression`, among others.
"""
import numpy as np
import pandas as pd


# Defaults
_N_SAMPLES = 1000
_N_SERIES = 3
_N_SEGMENT = 2
_DATE_START = '2000-01-01'
_DATE_END = '2010-01-01'
_FREQ = '16D'


# =============================================================================
# Segments
def make_segments(date_start=None, date_end=None, date_freq=None,
                  n_series=3, n_segments=2, seg_sep=None,
                  means=None, stds=None, trends=None,
                  amplitudes=None, phases=None):
    """ Simulate data from multiple temporal segments

    Parameters
    ----------
    date_start : str, datetime, and more, optional
        Starting date (in a format known to Pandas)
    date_end : str, datetime, and more, optional
        Ending date (in a format known to Pandas)
    date_freq : str, optional
        Date frequency
    n_series : int
        Number of series/spectral bands to simulate
    n_segments : int
        Number of segments to simulate
    seg_sep : Sequence[float]
        Separability of segments (i.e., the size of the disturbance)
    means : Sequence[float]
        The mean value for each series/spectral band (passed to
        :py:func:`make_time_series_mean`)
    stds : Sequence[float]
        The standard deviation for each series/spectral band (passed to
        :py:func:`make_time_series_mean`)
    trends : Sequence[float]
        The time trend for each series/spectral band (passed to
        :py:func:`make_time_series_trend`)
    amplitudes : Sequence[float]
        The harmonic amplitude value for each series/spectral band (passed to
        :py:func:`make_time_series_harmonic`)
    phases : Sequence[float]
        The harmonic phase value for each series/spectral band (passed to
        :py:func:`make_time_series_harmonic`)

    Returns
    -------
    xr.DataArray
        Simulated data for ``n_segments`` across ``n_series`` series/spectral
        bands
    np.ndarray
        Array of ``datetime64`` indicating the dates of change
        (size=``n_segments - 1``)
    """
    assert isinstance(n_series, int)
    assert isinstance(n_segments, int) and n_segments >= 1
    # TODO
    pass


# =============================================================================
# Time series
def make_dates(date_start=None, date_end=None, date_freq=None):
    """ Return ``datetime64`` dates

    Parameters
    ----------
    date_start : str, datetime, and more, optional
        Starting date (in a format known to Pandas)
    date_end : str, datetime, and more, optional
        Ending date (in a format known to Pandas)
    date_freq : str, optional
        Date frequency

    Returns
    -------
    np.ndarray
        Dates as ``np.datetime64``

    See Also
    --------
    pandas.date_range
    """
    return pd.date_range(date_start, date_end, freq=date_freq).values


def make_time_series_mean(dates, mean=None, std=None):
    pass


def make_time_series_trend(dates, trend=None):
    pass


def make_time_series_harmonic(dates, amplitude=None, phase=None):
    pass


def make_time_series_noise(dates, mean=0., std=1.):
    """ Generate a time series of noise
    """
    # TODO: implement Gaussian noise with mean/std
    # TODO: add kwarg to parametrize noise from clouds/shadows
    pass
