#!/usr/bin python2
# -*- coding: utf-8 -*-

from fits_formats import UV_FITS


class Data(object):
    """
    Represent VLBI observational data.
    """

    def __init__(self, fits_format=UV_FITS()):
        """
        format - instance of UV_FITS or FITS_IDI class used for io.
        """
        self._fits_format = fits_format
        self._recarray = None
        # Recarray - nice container of interferometric data
        #(u, v, w, t, bl, dt, wght, cmatrix(if, chan, stok))

    def load(self, fname):
        """
        Load data from FITS-file.
        """
        self._recarray = self._fits_format.load(fname)

    def save(self, fname):
        """
        Save data to FITS-file.
        """
        #TODO: put data from self.records (recarray) to HDU data
        # if recarray is not a view of HDU.data
        self._fits_format.save(fname)

    @property
    def uvw(self):
        """
        Returns (u, v, w) for data.
        """
        pass

    def uvplot(self, posangle=None):
        """
        Plot data vs. uv-plot distance.
        """
        pass

    def vplot(self):
        """
        Plot data vs. time.
        """
        pass

    def fit(self, imodel):
        """
        Fit visibility data with image plane model.
        """
        pass

    def __add__(self, data):
        """
        Add data to self.
        """
        pass

    def __multiply__(self, gains):
        """
        Applies complex antenna gains from instance of Gains class to
        the visibilities of self.
        """
        pass

    #TODO: how to substitute data to model only on one baseline?
    def substitute(self, model):
        """
        Substitue data of self with visibilities of the model.
        """
        uv_correlations = model.correlations(uvws=self.uvw)
        uv_correlations.broadcast(self._recarray)
        self._recarray = uv_correlations

    def calculate_noise(self, split_scans=False):
        """
        Calculate noise on all baselines.
        """
        pass

    def add_noise(self, stds, split_scans=False):
        """
        Add gaussian noise to self using specified std for each baseline/
        scan.
        """
        pass
