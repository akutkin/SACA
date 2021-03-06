#!/usr/bin python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import gammaln
from scipy import optimize


class KnuthHist(object):

    def __init__(self, data):
        self.data = np.array(data, copy=True)
        if self.data.ndim != 1:
            raise ValueError("data should be 1-dimensional")
        self.data.sort()
        self.n = self.data.size

        dx0, bins0 = freedman_bin_width(data, True)
        self.M0 = len(bins0) - 1

    def __call__(self):

        try:
            M = optimize.fmin(self.lnpost, self.M0)[0]
        except IndexError:
            M = self.M0

        bins = self.bins(M)
        dx = bins[1] - bins[0]

        return dx, bins

    def bins(self, M):
        """
        Return the bin edges given a number of bins ``M``.
        """
        return np.linspace(self.data[0], self.data[-1], int(M) + 1)

    def lnpost(self, M):
        """
        Logarithm of posterior for M bins.
        """

        M = int(M)
        bins = self.bins(M)
        nk, bins = np.histogram(self.data, bins)

        return -(self.n * np.log(M)
                 + gammaln(0.5 * M)
                 - M * gammaln(0.5)
                 - gammaln(self.n + 0.5 * M)
                 + np.sum(gammaln(nk + 0.5)))


def freedman_bin_width(data, return_bins=False):
    r"""Return the optimal histogram bin width using the Freedman-Diaconis rule

    Parameters
    ----------
    data : array-like, ndim=1
    observed (one-dimensional) data
    return_bins : bool (optional)
    if True, then return the bin edges

    Returns
    -------
    width : float
    optimal bin width using Scott's rule
    bins : ndarray
    bin edges: returned if `return_bins` is True

    Notes
    -----
    The optimal bin width is

    .. math::
    \Delta_b = \frac{2(q_{75} - q_{25})}{n^{1/3}}

    where :math:`q_{N}` is the :math:`N` percent quartile of the data, and
    :math:`n` is the number of data points.
    """
    data = np.asarray(data)
    if data.ndim != 1:
        raise ValueError("data should be one-dimensional")

    n = data.size
    if n < 4:
        raise ValueError("data should have more than three entries")

    dsorted = np.sort(data)
    v25 = dsorted[n / 4 - 1]
    v75 = dsorted[(3 * n) / 4 - 1]

    dx = 2 * (v75 - v25) * 1. / (n ** (1. / 3))

    if return_bins:
        Nbins = np.ceil((dsorted[-1] - dsorted[0]) * 1. / dx)
        Nbins = max(1, Nbins)
        bins = dsorted[0] + dx * np.arange(Nbins + 1)
        return dx, bins
    else:
        return dx


def histogram(a, range_=None, **kwargs):

    a = np.asarray(a)
    # if range is specified, we need to truncate the data for
    # the bin-finding routines
    if range_ is not None:
        a = a[(a >= range_[0]) & (a <= range_[1])]
    knuth = KnuthHist(a)
    da, bins = knuth()

    return np.histogram(a, bins, range_, **kwargs)
