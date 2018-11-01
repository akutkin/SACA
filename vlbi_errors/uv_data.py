import os
import math
import copy
import numpy as np
import astropy.io.fits as pf
from astropy.time import Time, TimeDelta
from astropy.stats import biweight_midvariance, mad_std
from sklearn.cluster import DBSCAN
from collections import OrderedDict
from utils import (baselines_2_ants, index_of, get_uv_correlations,
                   find_card_from_header, get_key, to_boolean_array,
                   check_issubset, convert_an_hdu, convert_fq_hdu,
                   mask_boolean_with_boolean)

try:
    import pylab
except ImportError:
    pylab = None

import matplotlib

vec_complex = np.vectorize(np.complex)
vec_int = np.vectorize(np.int)
stokes_dict = {-8:'YX', -7:'XY', -6:'YY', -5:'XX',
               -4:'LR', -3:'RL', -2:'LL', -1:'RR',
                1: 'I', 2: 'Q', 3: 'U', 4: 'V'}


# FIXME: Handling FITS files with only one scan (used for CV)
class UVData(object):

    def __init__(self, fname, mode='readonly'):
        self.fname = fname
        self.hdulist = pf.open(fname, mode=mode, save_backup=True)
        self.hdu = self.hdulist[0]
        self._stokes_dict = {'RR': 0, 'LL': 1, 'RL': 2, 'LR': 3}
        self.learn_data_structure(self.hdu)
        self._uvdata = self.view_uvdata({'COMPLEX': 0}) +\
            1j * self.view_uvdata({'COMPLEX': 1})
        self._weights = self.view_uvdata({'COMPLEX': 2})

        # Numpy boolean arrays with shape of ``UVData.uvdata``.
        self._nw_indxs = self._weights < 0
        self._pw_indxs = self._weights >= 0

        self._error = None
        self._scans_bl = None
        self._stokes = None
        self._times = None

        self._frequency = None
        self._nchans = None
        self._freq_width = None
        self._freq_width_if = None
        self._band_center = None

        self.scale_uv = 1.0 # scale for uv scaling )

        # Dictionary with keys - baselines & values - boolean numpy arrays or
        # lists of boolean numpy arrays with indexes of that baseline (or it's
        # scans) in ``UVData.uvdata`` array
        self._indxs_baselines = dict()
        self._indxs_baselines_scans = dict()
        # Dictionary with keys - baselines & values - tuples or lists of tuples
        # of shapes of part for that baseline (or it's scans) in
        # ``UVData.uvdata`` array
        self._shapes_baselines = dict()
        self._shapes_baselines_scans = dict()
        self._get_baselines_info()
        self._noise_diffs = None
        self._noise_v = None

    def set_uv_scale(self, scale):
        self.scale_uv = scale

    def _get_baselines_info(self):
        """
        Count indexes of visibilities on each single baseline (for single IF &
        Stokes) in ``uvdata`` array.
        """
        self._indxs_baselines_scans = self.scans_bl
        for baseline in self.baselines:
            indxs = self._get_baseline_indexes(baseline)
            self._indxs_baselines[baseline] = indxs
            self._shapes_baselines[baseline] = np.shape(self.uvdata[indxs])
            self._shapes_baselines_scans[baseline] = list()
            try:
                for scan_indxs in self._indxs_baselines_scans[baseline]:
                    bl_scan_data = self.uvdata[scan_indxs]
                    self._shapes_baselines_scans[baseline].append(np.shape(bl_scan_data))
            except TypeError:
                pass

    def nw_indxs_baseline(self, baseline, average_bands=False, stokes=None,
                          average_stokes=False):
        """
        Shortcut to negative or zero weights visibilities on given baseline.

        :param baseline:
            Integer baseline number.
        :param average_bands: (optional)
            Average bands in that way that if any bands for current
            visibility/stokes has negative weight then this visibility/stokes
            has negative weight. (default: ``False``)
        :param stokes: (optional)
            Stokes parameters of ``self`` that output or use for calculation of
            frequency averaged values.
        :param average_stokes: (optional)
            Average Stokes parameters chosen in ``stokes`` kw argument or all
            present in data in that way that if any stokes for current
            visibility has negative weight then this visibility has negative
            weight. (default: ``False``)
        :return:
            Numpy boolean array with shape of ``(#vis, #bands, #stokes)`` or
            ``(#vis, #stokes)``, where #vis - number of visibilities for given
            baseline & #stokes - number of stokes parameters in ``self`` or
            ``len(stokes)`` in ``stokes`` is not ``None``. (default: ``None``)
        """
        result = self._nw_indxs[self._indxs_baselines[baseline]]
        stokes_indxs = list()
        if stokes is not None:
            for stoke in stokes:
                assert stoke in self.stokes
                stokes_indxs.append(self.stokes_dict_inv[stoke])
        result = result[:, :, stokes_indxs]
        if average_bands:
            result = np.asarray(~result, dtype=int)
            result = np.prod(result, axis=1)
            result = np.asarray(result, dtype=bool)
            result = ~result
        if average_stokes and not average_bands:
            result = np.asarray(~result, dtype=int)
            result = np.prod(result, axis=2)
            result = np.asarray(result, dtype=bool)
            result = ~result
        if average_stokes and average_bands:
            result = np.asarray(~result, dtype=int)
            result = np.prod(result, axis=1)
            result = np.asarray(result, dtype=bool)
            result = ~result
        return result

    def pw_indxs_baseline(self, baseline, average_bands=False, stokes=None,
                          average_stokes=False):
        """
        Shortcut to positive weights visibilities on given baseline.

        :param baseline:
            Integer baseline number.
        :return:
            Numpy boolean array with shape of ``(#vis, #bands, #stokes)``, where
            #vis - number of visibilities for given baseline.
        """
        return ~self.nw_indxs_baseline(baseline, average_bands=average_bands,
                                       stokes=stokes,
                                       average_stokes=average_stokes)

    # FIXME: Finish it!
    def sample_size(self, stokes='I', average_freq=True):
        assert self._check_stokes_present(stokes)
        if average_freq:
            size = self.uvdata_freq_averaged.mask[0]
        else:
            size = self.uvdata_weight_masked.shape[0] * self.uvdata_weight_masked[1]

        stokes_multiplyer = 1
        if stokes == 'I':
            stokes_multiplyer = 2

    def _check_stokes_present(self, stokes):
        """
        Check if ``stokes`` is present in data (could be calculated from data).
        :param stokes:
            String of Stokes parameters ("I, Q, U, V, RR, LL, RL, LR").
        :return:
            Boolean value.
        """
        stokes_present = self.stokes
        if stokes in stokes_present:
            return True
        elif stokes in ("I", "Q", "U", "V"):
            if stokes in ("I", "V"):
                return "RR" in stokes_present and "LL" in stokes_present
            # If "Q" or "U"
            else:
                return "RL" in stokes_present and "LR" in stokes_present
        elif stokes in ("RR", "LL", "RL", "LR"):
            return stokes in stokes_present
        else:
            raise Exception("stokes must be from I, Q, U, V, RR, LL, RL or LR!")


    def sync(self):
        """
        Sync internal representation with complex representation and update
        complex representation ``self._uvdata``. I need this because i don't
        know how to make a complex view to real numpy.ndarray
        """
        slices_dict = self.slices_dict.copy()
        slices_dict.update({'COMPLEX': 0})
        self.hdu.data.data[slices_dict.values()] = self.uvdata.real
        slices_dict.update({'COMPLEX': 1})
        self.hdu.data.data[slices_dict.values()] = self.uvdata.imag

    def save(self, fname=None, data=None, rewrite=False,
             downscale_by_freq=False):
        """
        Save uv-data to FITS-file.

        :param data: (optional)
            Numpy record array with uv-data & parameters info. If ``None`` then
            save current instance's uv-data. (default: ``None``)
        :param fname: (optional)
            Name of FITS-file to save. If ``None`` then use current instance's
            original file. (default: ``None``)
        :param rewrite: (optional)
            Boolean - rewrite file with original name if any? (default:
            ``False``)
        """
        fname = fname or self.fname
        if os.path.exists(fname) and rewrite:
            os.unlink(fname)
        if data is None:
            if downscale_by_freq:
                self._downscale_uvw_by_frequency()
            self.hdulist.writeto(fname)
        else:
            # datas = np.array(sorted(data, key=lambda x: x['DATE']+x['_DATE']),
            #                 dtype=data.dtype)
            new_hdu = pf.GroupsHDU(data)
            # PyFits updates header using given data (``GCOUNT`` key) anyway
            new_hdu.header = self.hdu.header

            hdulist = pf.HDUList([new_hdu])
            for hdu in self.hdulist[1:]:
                if hdu.header['EXTNAME'] == 'AIPS AN':
                    # FIXME:
                    try:
                        hdu = convert_an_hdu(hdu, new_hdu)
                    except IndexError:
                        print "You should fix that issue!"
                        pass
                if hdu.header['EXTNAME'] == 'AIPS FQ':
                    hdu = convert_fq_hdu(hdu)
                hdulist.append(hdu)
            # FIXME: Sometimes i need this to be commented
            if downscale_by_freq:
                self._downscale_uvw_by_frequency()
            hdulist.writeto(fname, output_verify='ignore')

    def save_fraction(self, fname, frac, random_state=0):
        """
        Save only fraction of of data on each baseline.

        :param fname:
            File path to save.
        :param frac:
            Float (0., 1.). Fraction of points from each baseline to save.
        """
        from sklearn.model_selection import ShuffleSplit
        ss = ShuffleSplit(n_splits=1, test_size=1-frac,
                          random_state=random_state)
        indxs = list()
        for bl in self.baselines:
            bl_indxs = self._indxs_baselines[bl]
            print "Baseline {} has {} samples".format(bl,
                                                      np.count_nonzero(bl_indxs))
            bl_indxs_pw = self.pw_indxs_baseline(bl, average_bands=True,
                                                 stokes=['RR', 'LL'],
                                                 average_stokes=True)
            bl_indxs = mask_boolean_with_boolean(bl_indxs, bl_indxs_pw)
            for train, test in ss.split(np.nonzero(bl_indxs)[0]):
                # tr = to_boolean_array(np.nonzero(bl_indxs)[0][train],
                #                       len(bl_indxs))
                tr = np.nonzero(bl_indxs)[0][train]
            indxs.append(tr)
        indxs = np.hstack(indxs)
        indxs = sorted(indxs)
        data = self.hdu.data[indxs]
        self.save(fname, data, rewrite=True)

    def save_uvrange(self, fname, uv_min):
        """
        Save only fraction of of data on each baseline.

        :param fname:
            File path to save.
        """
        indxs = list()
        for bl in self.baselines:
            bl_indxs = self._indxs_baselines[bl]
            bl_indxs_pw = self.pw_indxs_baseline(bl, average_bands=True,
                                                 stokes=['RR', 'LL'],
                                                 average_stokes=True)
            bl_indxs = mask_boolean_with_boolean(bl_indxs, bl_indxs_pw)
            uv = self.uv[np.nonzero(bl_indxs)[0]]
            uv_rad = np.hypot(uv[:, 0], uv[:, 1])
            tr = np.nonzero(bl_indxs)[0][uv_rad > uv_min]
            indxs.append(tr)
        indxs = np.hstack(indxs)
        indxs = sorted(indxs)
        data = self.hdu.data[indxs]
        self.save(fname, data, rewrite=True)

    # TODO: for IDI extend this method
    def learn_data_structure(self, hdu):
        # Learn parameters
        par_dict = OrderedDict()
        for i, par in enumerate(hdu.data.names):
            par_dict.update({par: i})
        self.par_dict = par_dict
        # Create mapping of FITS CTYPEi ``i`` number to dimensions of PyFits
        # hdu.data[`DATA`] (hdu.data.data) numpy.ndarray.
        data_dict = OrderedDict()
        data_dict.update({'GROUP': (0, hdu.header['GCOUNT'])})
        for i in range(hdu.header['NAXIS'], 1, -1):
            data_dict.update({hdu.header['CTYPE' + str(i)]:
                                     (hdu.header['NAXIS'] - i + 1,
                                      hdu.header['NAXIS' + str(i)])})
        # Save shape and dimensions of data recarray
        self.data_dict = data_dict
        self.nif = data_dict['IF'][1]
        self.nstokes = data_dict['STOKES'][1]

        # Create dictionary with necessary slices
        slices_dict = OrderedDict()
        for key, value in data_dict.items():
            # FIXME: Generally we should avoid removing dims
            if value[1] == 1 and key not in ['IF', 'STOKES']:
                slices_dict.update({key: 0})
            else:
                slices_dict.update({key: slice(None, None)})
        self.slices_dict = slices_dict
        uvdata_slices_dict = OrderedDict()
        for key, value in slices_dict.items():
            if value is not 0:
                uvdata_slices_dict.update({key: value})
        self.uvdata_slices_dict = uvdata_slices_dict

    def new_slices(self, key, key_slice):
        """
        Return VIEW of internal ``hdu.data.data`` numpy.ndarray with given
        slice.
        """
        slices_dict = self.slices_dict.copy()
        slices_dict.update({key: key_slice})
        return slices_dict

    def view_uvdata(self, new_slices_dict):
        """
        Return VIEW of internal ``hdu.data.data`` numpy.ndarray with given
        slices.

        :param new_slices_dict:
            Ex. {'COMPLEX': slice(0, 1), 'IF': slice(0, 2)}
        """
        slices_dict = self.slices_dict.copy()
        for key, key_slice in new_slices_dict.items():
            slices_dict.update({key: key_slice})
        return self.hdu.data.data[slices_dict.values()]

    @property
    def stokes(self):
        """
        Shortcut to correlations present (or Stokes parameters).
        """
        if self._stokes is None:
            ref_val = get_key(self.hdu.header, 'STOKES', 'CRVAL')
            ref_pix = get_key(self.hdu.header, 'STOKES', 'CRPIX')
            delta = get_key(self.hdu.header, 'STOKES', 'CDELT')
            n_stokes = get_key(self.hdu.header, 'STOKES', 'NAXIS')
            self._stokes = [stokes_dict[ref_val + (i - ref_pix) * delta] for i
                            in range(1, n_stokes + 1)]
        return self._stokes

    @property
    def stokes_dict(self):
        return {i: stokes for i, stokes in enumerate(self.stokes)}

    @property
    def stokes_dict_inv(self):
        return {stokes: i for i, stokes in enumerate(self.stokes)}

    @property
    def uvdata(self):
        """
        Returns (#groups, #if, #stokes,) complex numpy.ndarray with last
        dimension - real&imag part of visibilities. It is A COPY of
        ``hdu.data.data`` numpy.ndarray.
        """
        # Always return complex representation of internal ``hdu.data.data``
        return self._uvdata

    @uvdata.setter
    def uvdata(self, other):
        # Updates A COPY of ``hdu.data.data`` numpy.ndarray (complex repr.)
        self._uvdata = other
        # Sync internal representation with changed complex representation.
        self.sync()

    @property
    def weights(self):
        """
        Returns (#groups, #if, #stokes,) complex numpy.ndarray with last
        dimension - weight of visibilities. It is A COPY of ``hdu.data.data``
        numpy.ndarray.
        """
        return self._weights

    @property
    def uvdata_weight_masked(self):
        return np.ma.array(self.uvdata, mask=self._nw_indxs)


    @property
    def uvdata_freq_averaged(self):
        """
        Returns ``self.uvdata`` averaged in IFs, that is complex numpy.ndarray
        with shape (#N, #stokes).
        """
        if self.nif > 1:
            result = np.ma.mean(self.uvdata_weight_masked, axis=1)
        # FIXME: if self.nif=1 then np.mean for axis=1 will remove this
        # dimension. So don't need this if-else
        else:
            result = self.uvdata_weight_masked[:, 0, :]
        return result


    @property
    def weights_nw_masked(self):
        """
        Returns (#groups, #if, #stokes,) complex numpy.ndarray with last
        dimension - weight of visibilities. It is A COPY of ``hdu.data.data``
        numpy.ndarray.
        """
        return np.ma.array(self._weights, mask=self._nw_indxs)

    @property
    def errors_from_weights(self):
        """
        Returns (#groups, #if, #stokes,) complex numpy.ndarray with last
        dimension - weight of visibilities. It is A COPY of ``hdu.data.data``
        numpy.ndarray.
        """
        return 1. / np.sqrt(self.weights_nw_masked)

    @property
    def errors_from_weights_masked_freq_averaged(self):
        if self.nif > 1:
            result = np.ma.mean(self.errors_from_weights, axis=1)
        else:
            result = self.errors_from_weights[:, 0, :]
        return result

    @property
    def baselines(self):
        """
        Returns list of baselines numbers.
        """
        result = list(set(self.hdu.data['BASELINE']))
        return vec_int(sorted(result))

    @property
    def antennas(self):
        """
        Returns list of antennas numbers.
        """
        return baselines_2_ants(self.baselines)

    @property
    def frequency(self):
        """
        Returns sky frequency in Hz.
        """
        if self._frequency is None:
            freq_card = find_card_from_header(self.hdu.header, value='FREQ')[0]
            self._frequency = self.hdu.header['CRVAL{}'.format(freq_card[0][-1])]
        return self._frequency

    @property
    def freq_width_if(self):
        """
        Returns width of IF in Hz.
        """
        if self._freq_width_if is None:
            freq_card = find_card_from_header(self.hdu.header, value='FREQ')[0]
            self._freq_width_if = self.hdu.header['CDELT{}'.format(freq_card[0][-1])]
        return self._freq_width_if

    @property
    def freq_width(self):
        """
        Returns width of all IFs in Hz.
        """
        if self._freq_width is None:
            freq_card = find_card_from_header(self.hdu.header, value='FREQ')[0]
            self._freq_width = self.nif * self.hdu.header['CDELT{}'.format(freq_card[0][-1])]
        return self._freq_width

    @property
    def band_center(self):
        """
        Returns center of frequency bandwidth in Hz.
        """
        if self._band_center is None:
            self._band_center = self.frequency + self.freq_width_if * (self.nif / 2. - 0.5)
        return self._band_center

    @property
    def times(self):
        """
        Returns array of ``astropy.time.Time`` instances.
        """
        if self._times is None:
            self._times = Time(self.hdu.data['DATE'] + self.hdu.data['_DATE'],
                               format='jd')
        return self._times

    @property
    def scans(self):
        """
        Returns list of times that separates different scans. If NX table is
        present in the original

        :return:
            numpy.ndarray with shape (#scans, 2,) with start & stop time for each
            of #scans scans.
        """
        try:
            indx = self.hdulist.index_of('AIPS NX')
            print "Found AIPS NX table!"
        except KeyError:
            indx = None
            print "No AIPS NX table are found!"

        if indx is not None:
            nx_hdu = self.hdulist[indx]
            scans = (np.vstack((nx_hdu.data['TIME'], nx_hdu.data['TIME'] +
                                nx_hdu.data['TIME INTERVAL']))).T
        else:
            scans = None

        return scans

    # FIXME: doesn't work for ``J0005+3820_X_1998_06_24_fey_vis.fits``
    # FIXME: Sometimes only 1 measurement in `scan`. It results in noise =
    # ``nan`` for that scan
    # FIXME: It would be better to output indexes of different scans for each
    # baselines
    @property
    def __scans_bl(self):
        """
        Calculate scans for each baseline separately.

        It won't coincide with UVData.scans because different baselines have
        different number of scans.

        :return:
            Dictionary with scans borders for each baseline.
        """
        scans_dict = dict()
        all_times = self.hdu.columns[self.par_dict['DATE']].array
        all_a, all_b = np.histogram(all_times[1:] - all_times[:-1])
        for bl in self.baselines:
            # print "Processing baseline ", bl
            bl_indxs = self._choose_uvdata(baselines=bl)[1]
            bl_times = self.hdu.columns[self.par_dict['DATE']].array[bl_indxs]
            a, b = np.histogram(bl_times[1:] - bl_times[:-1])
            # If baseline consists only of 1 scan
            if b[-1] < all_b[1]:
                scans_dict.update({bl: np.atleast_2d([bl_times[0],
                                                      bl_times[-1]])})
            # If baseline has > 1 scan
            else:
                scan_borders = bl_times[(np.where((bl_times[1:] -
                                                   bl_times[:-1]) > b[1])[0])]
                scans_list = [[bl_times[0], scan_borders[0]]]
                for i in range(len(scan_borders) - 1):
                    scans_list.append([float(bl_times[np.where(bl_times == scan_borders[i])[0] + 1]),
                                       scan_borders[i + 1]])
                scans_list.append([float(bl_times[np.where(bl_times == scan_borders[i + 1])[0] + 1]),
                                   bl_times[-1]])
                scans_dict.update({bl: np.asarray(scans_list)})

        return scans_dict

    @property
    def scans_bl(self):
        if self._scans_bl is None:
            scans_dict = dict()
            for bl in self.baselines:
                bl_scans = list()
                bl_indxs = self._get_baseline_indexes(bl)
                # JD-formated times for current baseline
                bl_times = self.hdu.data['DATE'][bl_indxs] +\
                           self.hdu.data['_DATE'][bl_indxs]
                bl_times = bl_times.reshape((bl_times.size, 1))
                db = DBSCAN(eps=TimeDelta(100., format='sec').jd, min_samples=10,
                            leaf_size=5).fit(bl_times)
                core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
                core_samples_mask[db.core_sample_indices_] = True
                labels = db.labels_
                if -1 in set(labels):
                    ant1, ant2 = baselines_2_ants([bl])
                    # print "Non-typical scan structure for baseline" \
                    #       " {}-{}".format(ant1, ant2)
                    scans_dict[bl] = None
                else:
                    bl_indxs_ = np.array(bl_indxs, dtype=int)
                    bl_indxs_[bl_indxs] = labels + 1
                    for i in set(labels):
                        bl_scans.append(bl_indxs_ == i + 1)
                    scans_dict[bl] = bl_scans
            self._scans_bl = scans_dict
        return self._scans_bl

    def _downscale_uvw_by_frequency(self):
        suffix = '--'
        try:
            u = self.hdu.columns[self.par_dict['UU{}'.format(suffix)]].array
            v = self.hdu.columns[self.par_dict['VV{}'.format(suffix)]].array
            w = self.hdu.columns[self.par_dict['WW{}'.format(suffix)]].array
        except KeyError:
            try:
                suffix = '---SIN'
                u = self.hdu.columns[self.par_dict['UU{}'.format(suffix)]].array
                v = self.hdu.columns[self.par_dict['VV{}'.format(suffix)]].array
                w = self.hdu.columns[self.par_dict['WW{}'.format(suffix)]].array
            except KeyError:
                suffix = ''
                u = self.hdu.columns[self.par_dict['UU{}'.format(suffix)]].array
                v = self.hdu.columns[self.par_dict['VV{}'.format(suffix)]].array
                w = self.hdu.columns[self.par_dict['WW{}'.format(suffix)]].array
        if abs(np.mean(u)) > 1.:
            self.hdu.columns[self.par_dict['UU{}'.format(suffix)]].array /= self.frequency
            self.hdu.columns[self.par_dict['VV{}'.format(suffix)]].array /= self.frequency
            self.hdu.columns[self.par_dict['WW{}'.format(suffix)]].array /= self.frequency

    def _upscale_uvw_by_frequency(self):
        suffix = '--'
        try:
            u = self.hdu.columns[self.par_dict['UU{}'.format(suffix)]].array
            v = self.hdu.columns[self.par_dict['VV{}'.format(suffix)]].array
            w = self.hdu.columns[self.par_dict['WW{}'.format(suffix)]].array
        except KeyError:
            try:
                suffix = '---SIN'
                u = self.hdu.columns[self.par_dict['UU{}'.format(suffix)]].array
                v = self.hdu.columns[self.par_dict['VV{}'.format(suffix)]].array
                w = self.hdu.columns[self.par_dict['WW{}'.format(suffix)]].array
            except KeyError:
                suffix = ''
                u = self.hdu.columns[self.par_dict['UU{}'.format(suffix)]].array
                v = self.hdu.columns[self.par_dict['VV{}'.format(suffix)]].array
                w = self.hdu.columns[self.par_dict['WW{}'.format(suffix)]].array
        if abs(np.mean(u)) < 1.:
            self.hdu.columns[self.par_dict['UU{}'.format(suffix)]].array *= self.frequency
            self.hdu.columns[self.par_dict['VV{}'.format(suffix)]].array *= self.frequency
            self.hdu.columns[self.par_dict['WW{}'.format(suffix)]].array *= self.frequency

    @property
    def uvw(self):
        """
        Shortcut for all (u, v, w)-elements of self.

        :return:
            Numpy.ndarray with shape (N, 3,), where N is the number of (u, v, w)
            points.
        """
        suffix = '--'
        try:
            u = self.hdu.columns[self.par_dict['UU{}'.format(suffix)]].array
            v = self.hdu.columns[self.par_dict['VV{}'.format(suffix)]].array
            w = self.hdu.columns[self.par_dict['WW{}'.format(suffix)]].array
        except KeyError:
            try:
                suffix = '---SIN'
                u = self.hdu.columns[self.par_dict['UU{}'.format(suffix)]].array
                v = self.hdu.columns[self.par_dict['VV{}'.format(suffix)]].array
                w = self.hdu.columns[self.par_dict['WW{}'.format(suffix)]].array
            except KeyError:
                suffix = ''
                u = self.hdu.columns[self.par_dict['UU{}'.format(suffix)]].array
                v = self.hdu.columns[self.par_dict['VV{}'.format(suffix)]].array
                w = self.hdu.columns[self.par_dict['WW{}'.format(suffix)]].array
        if abs(np.mean(u)) < 1.:
            u *= self.frequency
            v *= self.frequency
            w *= self.frequency
        return np.vstack((u, v, w)).T * self.scale_uv

    @property
    def uv(self):
        """
        Shortcut for (u, v) -coordinates of visibility values.

        :return:
            Numpy.ndarray with shape (N, 2,), where N is the number of (u, v, w)
            points.
        """
        return self.uvw[:, :2]


    @property
    def imsize_by_uv_coverage(self):
        """
        Calculate image size & pixel size using UV-plane coverage information.
        """
        raise NotImplementedError

    def _get_baseline_indexes(self, baseline):
        """
        Return boolean numpy array with indexes of given baseline in original
        record array.
        """
        assert baseline in self.baselines
        try:
            indxs = self._indxs_baselines[baseline]
        except KeyError:
            indxs = self.hdu.data['BASELINE'] == baseline
        return indxs

    def _get_baselines_indexes(self, baselines):
        """
        Return boolean numpy array with indexes of given baselines in original
        record array.
        """
        result = self._get_baseline_indexes(baseline=baselines[0])
        try:
            for baseline in baselines[1:]:
                result = np.logical_or(result, self._get_baseline_indexes(baseline))
        # When ``baselines`` consists of only one item
        except TypeError:
            pass
        return result

    def _get_times_indexes(self, start_time, stop_time):
        """
        Return numpy boolean array with indexes between given time in original
        record array.

        :param start_time:
            Instance of ``astropy.time.Time`` class.
        :param stop_time:
            Instance of ``astropy.time.Time`` class.
        """
        return np.logical_and(start_time <= self.times, stop_time >= self.times)

    def _conver_bands_to_indexes(self, bands):
        """
        Convert iterable of band numbers to boolean array with ``True`` values
        for given bands.

        :param bands:
            Iterable of integers (starting from zero) - band numbers.
        :return:
            Numpy boolean array with size equal to number of bands and ``True``
            values corresponding to specified band numbers.
        """
        assert set(bands).issubset(xrange(self.nif)), "Bands number must be "\
                                                      "from 0 to {}".format(self.nif)
        assert max(bands) <= self.nif
        return to_boolean_array(bands, self.nif)

    def _convert_stokes_to_indexes(self, stokes):
        """
        Convert iterable of correlations to boolean array with ``True`` values
        for given correlations.

        :param stokes:
            Iterable of strings - correlations.
        :return:
            Numpy boolean array with size equal to number of correlations and
            ``True`` values corresponding to specified correlations.
        """
#        assert check_issubset(stokes, self.stokes), "Must be RR, LL, RL or LR!"
        stokes_num = [self.stokes_dict_inv[stokes_] for stokes_ in stokes]
        return to_boolean_array(stokes_num, self.nstokes)

    def _get_uvdata_slice(self, baselines=None, start_time=None, stop_time=None,
                          bands=None, stokes=None):
        """
        Return tuple of index arrays that represent portion of ``UVData.uvdata``
        array with given values of baselines, times, bands, stokes.
        """
        if baselines is None:
            baselines = self.baselines
        indxs = self._get_baselines_indexes(baselines)

        if start_time is not None or stop_time is not None:
            indxs = np.logical_and(indxs, self._get_times_indexes(start_time,
                                                                  stop_time))

        if bands is None:
            bands_indxs = self._conver_bands_to_indexes(xrange(self.nif))
        else:
            bands_indxs = self._conver_bands_to_indexes(bands)

        if stokes is None:
            stokes = self.stokes
        stokes_indxs = self._convert_stokes_to_indexes(stokes)

        return np.ix_(indxs, bands_indxs, stokes_indxs)

    def _convert_uvdata_slice_to_bool(self, sl):
        """
        Convert indexing tuple to boolean array of ``UVData.uvdata`` shape.

        :param sl:
            Tuple of indexing arrays. Output of ``self._get_uvdata_slice``.
        :return:
            Boolean numpy array with shape of ``UVData.uvdata``.
        """
        boolean = np.zeros(self.uvdata.shape, dtype=bool)
        boolean[sl] = True
        return boolean

    # FIXME: Choose only one stokes parameter
    def _choose_uvdata(self, start_time=None, stop_time=None, baselines=None,
                       bands=None, stokes=None, freq_average=False):
        """
        Method that returns chosen data from ``_data`` numpy structured array
        based on user specified parameters.

        :param start_time: (optional)
            Instance of ``astropy.time.Time`` class. (default: ``None``)

        :param stop_time: (optional)
            Instance of ``astropy.time.Time`` class. (default: ``None``)

        :param baselines: (optional)
            One or iterable of baselines numbers or ``None``. If ``None`` then
            use all baselines. (default: ``None``)

        :param bands: (optional)
            Iterable of IF numbers (0 to #IF-1) or ``None``. If ``None`` then
            use all IFs. (default: ``None``)

        :param stokes: (optional)
            Any string of: ``I``, ``Q``, ``U``, ``V``, ``RR``, ``LL``, ``RL``,
            ``LR`` or ``None``. If ``None`` then use all available correlations.
            If ``I``, ``Q``, ``U``, ``V`` then must be iterable with only one
            item (any single Stokes parameter). (default: ``None``)

        :return:
            Numpy.ndarray that is part of (copy) ``UVData.uvdata`` array with
            shape (#N, #IF, #STOKES).
        """
        # Copy with shape (#N, #IF, #STOKES)
        uvdata = self.uvdata_weight_masked

        if start_time is None:
            start_time = self.times[0]
        if stop_time is None:
            stop_time = self.times[-1]

        # FIXME: Choose only one stokes parameter
        if stokes is None:
            stokes = self.stokes
        if stokes and (stokes[0] in self.stokes):
            sl = self._get_uvdata_slice(baselines, start_time, stop_time, bands,
                                        stokes)
            result = uvdata[sl]

        elif check_issubset(stokes, ('I', 'Q', 'U', 'V')):
            assert len(stokes) == 1, "Only one Stokes parameter allowed!"

            if stokes in ('I', 'V'):
                sl_rr = self._get_uvdata_slice(baselines, start_time, stop_time,
                                               bands, stokes=['RR'])
                sl_ll = self._get_uvdata_slice(baselines, start_time, stop_time,
                                               bands, stokes=['LL'])
                if stokes == 'I':
                    # I = 0.5 * (RR + LL)
                    result = 0.5 * (uvdata[sl_rr] + uvdata[sl_ll])
                else:
                    # V = 0.5 * (RR - LL)
                    result = 0.5 * (uvdata[sl_rr] - uvdata[sl_ll])

            if stokes in ('Q', 'U'):
                sl_rl = self._get_uvdata_slice(baselines, start_time, stop_time,
                                               bands, stokes=['RL'])
                sl_lr = self._get_uvdata_slice(baselines, start_time, stop_time,
                                               bands, stokes=['LR'])

                if stokes == 'Q':
                    # V = 0.5 * (LR + RL)
                    result = 0.5 * (uvdata[sl_lr] + uvdata[sl_rl])
                else:
                    # V = 0.5 * 1j * (LR - RL)
                    result = 0.5 * 1j * (uvdata[sl_lr] - uvdata[sl_rl])
        else:
            raise Exception("Stokes must be iterable consisting of following "
                            "items only: I, Q, U, V, RR, LL, RL, LR!")

        if freq_average:
            result = np.ma.mean(result, axis=1).squeeze()

        return result

    def noise_v(self, average_bands=False):
        """
        Calculate noise for each baseline using Stokes ``V`` data.

        :param average_bands: (optional)
            Boolean - average bands after noise calculation?

        :return:
            Dictionary with keys - baseline numbers & values - numpy arrays with
            shape (#bands, #stokes) or (#stokes,) if ``average_bands=True``.
        """
        if self._noise_v is None:
            baseline_noises = dict()
            for baseline in self.baselines:
                uvdata = self._choose_uvdata(baselines=[baseline])
                v = uvdata[..., 0] - uvdata[..., 1]
                mask = np.logical_or(np.isnan(v), v.mask)
                # #groups, #bands
                data = np.ma.array(v, mask=mask)
                mstd = list()
                for band_data in data.T:
                    mstd.append(0.5 * (biweight_midvariance(band_data.real) +
                                       biweight_midvariance(band_data.imag)))
                baseline_noises[baseline] =\
                    np.array(mstd).repeat(self.nstokes).reshape((self.nif,
                                                                 self.nstokes))
            self._noise_v = baseline_noises.copy()

        if average_bands:
            return {baseline: np.nanmean(mstd, axis=0) for baseline, mstd in
                    self._noise_v.items()}

        return self._noise_v

    def noise_diffs(self, average_bands=False):
        """
        Calculate noise for each baseline using successive differences approach
        (Brigg's dissertation).

        :param average_bands: (optional)
            Boolean - average bands after noise calculation?

        :return:
            Dictionary with keys - baseline numbers & values - numpy arrays with
            shape (#bands, #stokes) or (#stokes,) if ``average_bands=True``.
        """
        if self._noise_diffs is None:
            baseline_noises = dict()
            for baseline in self.baselines:
                uvdata = self._choose_uvdata(baselines=[baseline])
                diffs = uvdata[:-1, ...] - uvdata[1:, ...]
                mask = np.logical_or(np.isnan(diffs), diffs.mask)
                # #groups, #bands
                data = np.ma.array(diffs, mask=mask)
                mstd = np.zeros((self.nif, self.nstokes))
                for if_ in xrange(self.nif):
                    for stoke in xrange(self.nstokes):
                        data_ = data[:, if_, stoke]
                        # mstd[if_, stoke] += biweight_midvariance(data_.real)
                        # mstd[if_, stoke] += biweight_midvariance(data_.imag)
                        mstd[if_, stoke] += np.std(data_.real)
                        mstd[if_, stoke] += np.std(data_.imag)
                        mstd[if_, stoke] *= 0.5
                baseline_noises[baseline] = mstd
            self._noise_diffs = baseline_noises.copy()

        if average_bands:
            return {baseline: np.nanmean(mstd, axis=0) for baseline, mstd in
                    self._noise_diffs.items()}

        return self._noise_diffs

    def noise(self, split_scans=False, use_V=True, average_freq=False):
        """
        Calculate noise for each baseline. If ``split_scans`` is True then
        calculate noise for each scan too. If ``use_V`` is True then use stokes
        V data (`RR`` - ``LL``) for computation assuming no signal in V. Else
        use successive differences approach (Brigg's dissertation).

        :param split_scans: (optional)
            Should we calculate noise for each scan? (default: ``False``)

        :param use_V: (optional)
            Use stokes V data (``RR`` - ``LL``) to calculate noise assuming no
            signal in stokes V? If ``False`` then use successive differences
            approach (see Brigg's dissertation). (default: ``True``)

        :param average_freq: (optional)
            Use IF-averaged data for calculating noise? (default: ``False``)

        :return:
            Dictionary with keys - baseline numbers & values - arrays of shape
            ([#scans], [#IF], [#stokes]). It means (#scans, #IF) if
            ``split_scans=True`` & ``use_V=True``, (#IF, #stokes) if
            ``split_scans=False`` & ``use_V=False``, (#scans, #IF, #stokes) if
             ``split_scans=True``, ``use_V=False`` & ``average_freq=False`` etc.
        """
        baselines_noises = dict()
        if use_V:
            # Calculate dictionary {baseline: noise} (if split_scans is False)
            # or {baseline: [noises]} if split_scans is True.
            if not split_scans:
                for baseline in self.baselines:
                    baseline_uvdata = self._choose_uvdata(baselines=[baseline])
                    if average_freq:
                        baseline_uvdata = np.mean(baseline_uvdata, axis=1)
                    v = (baseline_uvdata[..., 0] - baseline_uvdata[..., 1]).real
                    mask = ~np.isnan(v)
                    baselines_noises[baseline] =\
                        np.asarray(mad_std(np.ma.array(v, mask=np.invert(mask)).data,
                                           axis=0))
                        # np.asarray(np.std(np.ma.array(v, mask=np.invert(mask)).data,
                        #                   axis=0))
            else:
                # Use each scan
                for baseline in self.baselines:
                    baseline_noise = list()
                    try:
                        for scan_bl_indxs in self.scans_bl[baseline]:
                            # (#obs in scan, #nif, #nstokes,)
                            scan_baseline_uvdata = self.uvdata[scan_bl_indxs]
                            if average_freq:
                                # (#obs in scan, #nstokes,)
                                scan_baseline_uvdata = np.mean(scan_baseline_uvdata,
                                                               axis=1)
                            v = (scan_baseline_uvdata[..., 0] -
                                 scan_baseline_uvdata[..., 1]).real
                            mask = ~np.isnan(v)
                            scan_noise = np.asarray(np.std(np.ma.array(v,
                                                                       mask=np.invert(mask)).data,
                                                           axis=0))
                            baseline_noise.append(scan_noise)
                        baselines_noises[baseline] = np.asarray(baseline_noise)
                    except TypeError:
                        baselines_noises[baseline] = None

        else:
            if not split_scans:
                for baseline in self.baselines:
                    # (#, #IF, #Stokes)
                    baseline_uvdata = self._choose_uvdata(baselines=[baseline])
                    if average_freq:
                        baseline_uvdata = np.mean(baseline_uvdata, axis=1)
                    # (#, #IF, #Stokes)
                    differences = (baseline_uvdata[:-1, ...] -
                                   baseline_uvdata[1:, ...])
                    mask = np.isnan(differences)
                    # (#IF, #Stokes)
                    baselines_noises[baseline] = \
                        np.asarray([mad_std(np.ma.array(differences,
                                                        mask=mask).real[..., i], axis=0) for i
                                    in range(self.nstokes)]).T
            else:
                # Use each scan
                for baseline in self.baselines:
                    baseline_noise = list()
                    try:
                        for scan_bl_indxs in self.scans_bl[baseline]:
                            # (#obs in scan, #nif, #nstokes,)
                            scan_baseline_uvdata = self.uvdata[scan_bl_indxs]
                            if average_freq:
                                # shape = (#obs in scan, #nstokes,)
                                scan_baseline_uvdata = np.mean(scan_baseline_uvdata,
                                                               axis=1)
                            # (#obs in scan, #nif, #nstokes,)
                            differences = (scan_baseline_uvdata[:-1, ...] -
                                           scan_baseline_uvdata[1:, ...])
                            mask = ~np.isnan(differences)
                            # (nif, nstokes,)
                            scan_noise = np.asarray([mad_std(np.ma.array(differences,
                                                                        mask=np.invert(mask)).real[..., i],
                                                            axis=0) for i in
                                                     range(self.nstokes)]).T
                            baseline_noise.append(scan_noise)
                        baselines_noises[baseline] = np.asarray(baseline_noise)
                    except TypeError:
                        baselines_noises[baseline] = None

        return baselines_noises

    def noise_add(self, noise=None, df=None, split_scans=False):
        """
        Add noise to visibilities. Here std - standard deviation of
        real/imaginary component.

        :param noise:
            Mapping from baseline number to:

            1) std of noise. Will use one value of std for all stokes and IFs.
            2) iterable of stds. Will use different values of std for different
            IFs.

        :param df: (optional)
            Number of d.o.f. for standard Student t-distribution used as noise
            model.  If set to ``None`` then use gaussian noise model. (default:
            ``None``)

        :param split_scans: (optional)
            Is parameter ``noise`` is mapping from baseline numbers to
            iterables of std of noise for each scan on baseline? (default:
            ``False``)
        """

        # TODO: if on df before generating noise values
        for baseline, baseline_stds in noise.items():
            # i - IF number, std (#IF, #Stokes)
            for i, std in enumerate(baseline_stds):
                # (#, 1, #stokes)
                for stokes in self.stokes:
                    j = self.stokes_dict_inv[stokes]
                    baseline_uvdata =\
                        self._choose_uvdata(baselines=[baseline], bands=[i],
                                            stokes=[stokes])
                    # (#, #IF, #CH, #stokes)
                    n = len(baseline_uvdata)
                    sl = self._get_uvdata_slice(baselines=[baseline], bands=[i],
                                                stokes=[stokes])
                    noise_to_add = vec_complex(np.random.normal(scale=std,
                                                                size=n),
                                               np.random.normal(scale=std,
                                                                size=n))
                    noise_to_add = np.reshape(noise_to_add,
                                              baseline_uvdata.shape)
                    baseline_uvdata += noise_to_add
                    self.uvdata[sl] = baseline_uvdata
        self.sync()

    # TODO: Optionally calculate noise by scans.
    def error(self, average_freq=False, use_V=True):
        """
        Shortcut for error associated with each visibility.

        It uses noise calculations based on zero V stokes or successive
        differences implemented in ``noise()`` method to infer sigma of
        gaussian noise.  Later it is supposed to add more functionality (see
        Issue #8).

        :param average_freq: (optional)
            Use IF-averaged data for calculating errors? (default: ``False``)
        :param use_V: (optional)
            Boolean. Calculate noise using Stokes `V` or successive differences?
            (default: ``True``)

        :return:
            Numpy.ndarray with shape (#N, [#IF,] #stokes,) where #N - number of
            groups.
        """
        if self._error is None:
            noise_dict = self.noise(use_V=use_V, split_scans=False,
                                    average_freq=average_freq)
            if not average_freq:
                self._error = np.empty((len(self.uvdata), self.nif,
                                        self.nstokes,), dtype=float)
            else:
                self._error = np.empty((len(self.uvdata), self.nstokes,),
                                       dtype=float)

            for i, baseline in enumerate(self.hdu.data['BASELINE']):
                # FIXME: Until ``UVData.noise`` always returns (#, [#IF],
                # #Stokes) even for ``use_V=True`` - i must repeat array for
                # each Stokes if ``use_V=True`` is used!
                self._error[i] = noise_dict[baseline]

        return self._error

    def scale_amplitude(self, scale):
        """
        Scale amplitude of uv-data by some scale factor.

        :param scale:
            Float. Factor of scaling.
        """
        self.uvdata *= scale

    # TODO: use different stokes and symmetry!
    def uv_coverage(self, antennas=None, baselines=None, sym='.k',
                    start_time=None, stop_time=None):
        """
        Make plots of uv-coverage for selected baselines/antennas.

        If ``antenna`` is not None, then plot tracs for all baselines of
        selected antenna with antennas specified in ``baselines``. It is like
        AIPS task UVPLOT with bparm=6,7,0.

        :param antennas: (optional)
            AIPS-like ``uvplt`` parameter.
        :param baselines: (optional)
            AIPS-like ``uvplt`` parameter.
        :param sym: (optional)
            Matplotlib symbols to plot. (default: ``.k``)
        :param start_time: (optional)
            Instance of ``astropy.time.Time`` class. (default: ``None``)
        :param stop_time: (optional)
            Instance of ``astropy.time.Time`` class. (default: ``None``)
        """
        if antennas is None:
            antennas = self.antennas

        if baselines is None:
            raise Exception("Provide some antenna num. for baselines!")
        else:
            baselines_list = list()
            # If ``baselines`` is iterable
            try:
                baselines_list.extend(baselines)
            # If ``baselines`` is not iterable (int)
            except TypeError:
                baselines_list.append(baselines)
            baselines = set(baselines_list)

        # Check that given baseline numbers are among existing ones
        assert(baselines.issubset(self.antennas))
        # Assert that we don't have one and the same antenna and baseline
        if len(baselines) == len(antennas) == 1:
            assert not baselines.issubset(antennas), "Zero spacing baseline!"

        # Find what baselines to display
        baselines_to_display = list()
        antennas_list = list()
        # If ``antennas`` is iterable
        try:
            antennas_list.extend(antennas)
        # If ``antennas`` is not iterable (int)
        except TypeError:
            antennas_list.append(antennas)
        for ant1 in antennas_list:
            for ant2 in baselines:
                if ant2 > ant1:
                    baselines_to_display.append(ant2 + 256 * ant1)
                elif ant2 < ant1:
                    baselines_to_display.append(ant1 + 256 * ant2)

        baselines_to_display = list(set(baselines_to_display))

        uvdata, indxs = self._choose_uvdata(baselines=baselines_to_display,
                                            start_time=start_time,
                                            stop_time=stop_time,
                                            freq_average=True)

        # FIXME: Use properties for u, v
        u = self.hdu.columns[self.par_dict['UU--']].array[indxs]
        v = self.hdu.columns[self.par_dict['VV--']].array[indxs]
        uv = np.vstack((u, v)).T
        matplotlib.pyplot.subplot(1, 1, 1)
        matplotlib.pyplot.plot(uv[:, 0], uv[:, 1], sym)
        # FIXME: This is right only for RR/LL!
        matplotlib.pyplot.plot(-uv[:, 0], -uv[:, 1], sym)
        # Find max(u & v)
        umax = max(abs(u))
        vmax = max(abs(v))
        uvmax = max(umax, vmax)
        uv_range = [-1.1 * uvmax, 1.1 * uvmax]
        matplotlib.pyplot.xlim(uv_range)
        matplotlib.pyplot.ylim(uv_range)
        matplotlib.pyplot.axes().set_aspect('equal')
        matplotlib.pyplot.xlabel('U, wavelengths')
        matplotlib.pyplot.ylabel('V, wavelengths')
        matplotlib.pyplot.show()

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return UVData(self.hdulist.filename(), mode='readonly')

    def __add__(self, other):
        """
        Add to self another instance of UVData.

        :param other:
            Instance of ``UVData`` class. Or object that has ``uvdata``
            attribute that is numpy structured array with the same ``dtype`` as
            ``self``.

        :return:
            Instance od ``UVData`` class with uv-data in ``uvdata`` attribute
            that is sum of ``self`` and other.
        """

        assert(self.uvdata.shape == other.uvdata.shape)
        assert(len(self.uvdata) == len(other.uvdata))

        self_copy = copy.deepcopy(self)
        self_copy.uvdata = self.uvdata + other.uvdata
        self_copy.sync()

        return self_copy

    def __sub__(self, other):
        """
        Substruct from self another instance of UVData.

        :param other:
            Instance of ``UVData`` class. Or object that has ``uvdata``
            attribute that is numpy structured array with the same ``dtype`` as
            ``self``.

        :return:
            Instance od ``UVData`` class with uv-data in ``uvdata`` attribute
            that is difference of ``self`` and other.
        """

        assert(self.uvdata.shape == other.uvdata.shape)
        assert(len(self.uvdata) == len(other.uvdata))

        self_copy = copy.deepcopy(self)
        self_copy.uvdata = self.uvdata - other.uvdata
        self_copy.sync()

        return self_copy

    def multiply(self, x):
        """
        Multiply visibilities on number.
        :param x:
        :return:
        """
        self_copy = copy.deepcopy(self)
        self_copy.uvdata = x * self.uvdata
        self_copy.sync()

        return self_copy

    # TODO: TEST ME!!!
    # TODO: Do i need the possibility of multiplying on any complex number?
    # FIXME: After absorbing gains and multiplying on UVData instance some
    # entries do contain NaN. Is that because of some data is flagged and no
    # gains solution are available for that data?
    def __mul__(self, gains):
        """
        Applies complex antenna gains to the visibilities of ``self``.

        :param gains:
            Instance of ``Gains`` class. Or object with ``data`` attribute
            that is structured numpy array and has ``dtype``:
            dtype=[('start', '<f8'),
                   ('stop', '<f8'),
                   ('antenna', 'int'),
                   ('gains', 'complex', (nif, npol,)),
                   ('weights', '<f8', (nif, npol,))]

        :return:
            Instance of ``UVData`` class with visibilities multiplyied by
            complex antenna gains.
        """

        self_copy = copy.deepcopy(self)

        assert(self.nif == np.shape(gains.nif))
        # TODO: Now we need this to calculating gain * gains*. But try to
        # exclude this assertion
        assert(self.nstokes == 4)

        for t in set(self.hdu.columns[self.par_dict['DATE']].array):

            # Find all uv-data entries with time t:
            indxs = np.where(self.hdu.columns[self.par_dict['DATE']].array
                             == t)[0]
            # Loop through uv_indxs (different baselines with the same ``t``)
            # and multiply visibility with baseline ant1-ant2 to
            # gain(ant1)*gain(ant2)^*.
            for indx in indxs:
                bl = self.hdu.columns[self.par_dict['BASELINE']].array[indx]
                try:
                    gains12 = gains.find_gains_for_baseline(t, bl)
                # If gains is the instance of ``Absorber`` class
                except AttributeError:
                    gains12 = gains.absorbed_gains.find_gains_for_baseline(t,
                                                                           bl)
                # FIXME: In substitute() ['hands'] then [indxs] does return
                # view.
                # print "gains12 :"
                # print gains12
                # Doesn't it change copying? Order of indexing [][] has changed
                self_copy.uvdata[indx] *= gains12.T
        self_copy.sync()

        return self_copy

    def zero_data(self):
        """
        Method that zeros all visibilities.
        """
        self.uvdata = np.zeros(np.shape(self.uvdata), dtype=self.uvdata.dtype)

    def cv(self, q, fname):
        """
        Method that prepares training and testing samples for q-fold
        cross-validation.

        Inputs:

        :param q:
            Number of folds for cross-validation.

        :param fname:
            Base of file names for output the results.

        :return:
            ``q`` pairs of files (format that of ``IO`` subclass that loaded
            current instance of ``UVData``) with training and testing samples
            prepaired in a such way that 1/``q``- part of visibilities from
            each baseline falls in testing sample and other part falls in
            training sample.
        """

        # List of lists of ``q`` blocks of each baseline
        baselines_chunks = list()

        # Split data of each baseline to ``q`` blocks
        for baseline in self.baselines:
            baseline_indxs = np.where(self.hdu.columns[self.par_dict['BASELINE']].array ==
                                     baseline)[0]
            # Shuffle indexes
            np.random.shuffle(baseline_indxs)
            # Indexes of ``q`` nearly equal chunks. That is list of ``q`` index
            # arrays
            q_indxs = np.array_split(baseline_indxs, q)
            # ``q`` blocks for current baseline
            baseline_chunks = [list(indx) for indx in q_indxs]
            baselines_chunks.append(baseline_chunks)

        # Combine ``q`` chunks to ``q`` pairs of training & testing datasets
        for i in range(q):
            print i
            # List of i-th chunk for testing dataset for each baseline
            testing_indxs = [baseline_chunks[i] for baseline_chunks in
                            baselines_chunks]
            # List of "all - i-th" chunk as training dataset for each baseline
            training_indxs = [baseline_chunks[:i] + baseline_chunks[i + 1:] for
                             baseline_chunks in baselines_chunks]

            # Combain testing & training samples of each baseline in one
            testing_indxs = np.sort([item for sublist in testing_indxs for item
                                     in sublist])
            training_indxs = [item for sublist in training_indxs for item in
                              sublist]
            training_indxs = [item for sublist in training_indxs for item in
                              sublist]
            # Save each pair of datasets to files
            # NAXIS changed!!!
            training_data=self.hdu.data[training_indxs]
            testing_data =self.hdu.data[testing_indxs]
            self.save(data=training_data,
                      fname=fname + '_train' + '_' + str(i + 1).zfill(2) + 'of'
                            + str(q) + '.FITS')
            self.save(data=testing_data,
                      fname=fname + '_test' + '_' + str(i + 1).zfill(2) + 'of' +
                            str(q) + '.FITS')

    # TODO: Refactor to general eatimating score (RMS) of ``Model`` instance of
    # ``self.``
    def cv_score(self, model, average_freq=True, baselines=None):
        """
        Method that returns cross-validation score for ``self`` (as testing
        cv-sample) and model (trained on training cv-sample).

        :param model:
            Model to cross-validate. Instance of ``Model`` class.

        :param average_freq: (optional)
            Boolean - average IFs before CV score calculation? (default:
            ``True``)

        :return:
            Cross-validation score between uv-data of current instance and
            model for stokes ``I``.
        """

        baselines_cv_scores = list()

        # noise = self.noise_diffs(average_bands=average_freq)

        data_copied = copy.deepcopy(self)
        data_copied.substitute([model])
        data_copied = self - data_copied

        if average_freq:
            uvdata = data_copied.uvdata_freq_averaged
        else:
            uvdata = data_copied.uvdata_weight_masked

        if baselines is None:
            baselines = self.baselines
        for baseline in baselines:
            # square difference for each baseline, divide by baseline noise
            # and then sum for current baseline
            indxs = data_copied._indxs_baselines[baseline]
            hands_diff = uvdata[indxs]
            # if average_freq:
            #     hands_diff = uvdata[indxs] / noise[baseline]
            # else:
            #     hands_diff = uvdata[indxs] / noise[baseline][None, :, None]
            # Construct difference for Stokes ``I`` parameter
            diff = 0.5 * (hands_diff[..., 0] + hands_diff[..., 1])
            # print np.shape(hands_diff)
            # diff = hands_diff[..., 0]
            diff = diff.flatten()
            diff *= np.conjugate(diff)
            try:
                baselines_cv_scores.append(float(diff.sum())/np.count_nonzero(~diff.mask[..., :2]))
            except ZeroDivisionError:
                continue

        return sum(baselines_cv_scores)

    # TODO: Use for-cycle on baseline indexes
    def substitute(self, models, baselines=None):
        """
        Method that substitutes visibilities of ``self`` with model values.

        :param models:
            Iterable of ``Model`` instances that substitute visibilities of
            ``self`` with it's own. There should be only one (or zero) model for
            each stokes parameter. If there are two, say I-stokes models, then
            sum them firstly using ``Model.__add__``.

        :param baseline (optional):
            Iterable of baselines on which to substitute visibilities. If
            ``None`` then substitute on all baselines.
            (default: ``None``)
        """

        if baselines is None:
            baselines = self.baselines
        # Indexes of hdu.data with chosen baselines
        indxs = np.hstack(index_of(baselines, self.hdu.columns[self.par_dict['BASELINE']].array))
        n = len(indxs)
        uv = self.uvw[indxs, :2]

        uv_correlations = get_uv_correlations(uv, models)
        for i, hand in self.stokes_dict.items():
            try:
                self.uvdata[indxs, :, i] = \
                    uv_correlations[hand].repeat(self.nif).reshape((n, self.nif))
                self.sync()
            # If model doesn't have some hands => pass it
            except KeyError:
                pass

    # TODO: convert time to datetime format and use date2num for plotting
    # TODO: make a kwarg argument - to plot in different symbols/colors
    def tplot(self, baselines=None, bands=None, stokes=None, style='a&p',
              freq_average=False, sym=None, start_time=None, stop_time=None):
        """
        Method that plots uv-data vs. time.

        :param baselines: (optional)
            Iterable of baselines numbers or ``None``. If ``None`` then
            use all baselines. (default: ``None``)
        :parm bands: (optional)
            Iterable of IF numbers (0-#IF-1) or ``None``. If ``None`` then
            use all IFs. (default: ``None``)
        :param stokes: (optional)
            Any string of: ``I``, ``Q``, ``U``, ``V``, ``RR``, ``LL``, ``RL``,
            ``LR`` or ``None``. If ``None`` then use ``I``.
            (default: ``None``)
        :param style: (optional)
            How to plot complex visibilities - real and imaginary part
            (``re&im``) or amplitude and phase (``a&p``). (default: ``a&p``)
        :param start_time: (optional)
            Instance of ``astropy.time.Time`` class. (default: ``None``)
        :param stop_time: (optional)
            Instance of ``astropy.time.Time`` class. (default: ``None``)

        .. note:: All checks are in ``_choose_uvdata`` method.
        """

        try:
            import pylab
        except:
            raise Exception('Install ``pylab`` for plotting!')

        if not stokes:
            stokes = 'I'

        if start_time is not None and stop_time is not None:
            times_indxs = self._get_times_indexes(start_time, stop_time)
            times = self.times[times_indxs]
        else:
            times= self.times

        uvdata = self._choose_uvdata(baselines=baselines, bands=bands,
                                     stokes=stokes, freq_average=freq_average,
                                     start_time=start_time, stop_time=stop_time)
        print uvdata.shape

        if style == 'a&p':
            a1 = np.angle(uvdata)
            a2 = np.real(np.sqrt(uvdata * np.conj(uvdata)))
        elif style == 're&im':
            a1 = uvdata.real
            a2 = uvdata.imag
        else:
            raise Exception('Only ``a&p`` and ``re&im`` styles are allowed!')

#        return times, a1, a2

        if not freq_average:

            # # of chosen IFs
            n_if = len(bands)

            # TODO: define colors
            try:
                syms = self.__color_list[:n_if]
            except AttributeError:
                print "Define self.__color_list to show in different colors!"
                syms = ['.k'] * n_if

            pylab.subplot(2, 1, 1)
            for _if in range(n_if):
                # TODO: plot in different colors and make a legend
                pylab.plot(times, a1[:, _if], syms[_if])
            pylab.subplot(2, 1, 2)
            for _if in range(n_if):
                pylab.plot(times, a2[:, _if], syms[_if])
                if style == 'a&p':
                    pylab.ylim([-math.pi, math.pi])
            pylab.show()

        else:
            if not sym:
                sym = '.k'
            pylab.subplot(2, 1, 1)
            # WARNING: set proper selection for first index of a1/a2
            pylab.plot(times, a1, sym)
            pylab.subplot(2, 1, 2)
            pylab.plot(times, a2[:,0], sym)
            if style == 'a&p':
                pylab.ylim([-math.pi, math.pi])
            pylab.show()

    # TODO: Implement PA[deg] slicing of uv-plane with keyword argument ``PA``.
    # TODO: Add ``model`` kwarg for plotting image plane model with data
    # together.
    # TODO: Add ``plot_noise`` boolean kwarg for plotting error bars also. (Use
    # ``UVData.noise()`` method for finding noise values.)
    # TODO: implement antennas/baselines arguments as in ``uv_coverage``.
    def uvplot(self, baselines=None, bands=None, stokes=None, style='a&p',
               freq_average=False, sym=None, phase_range=None, amp_range=None,
               re_range=None, im_range=None, colors=None, color='#4682b4',
               fig=None, start_time=None, stop_time=None, alpha=1.0):
        """
        Method that plots uv-data for given baseline vs. uv-radius.

        :param baselines: (optional)
            Iterable of baselines numbers or ``None``. If ``None`` then use all
            baselines. (default: ``None``)
        :parm bands (optional):
            Iterable of IF numbers (0 to #IF-1) or ``None``. If ``None`` then
            use all IFs. (default: ``None``)
        :param stokes: (optional)
            Any string of: ``I``, ``Q``, ``U``, ``V``, ``RR``, ``LL``, ``RL``,
            ``LR`` or ``None``. If ``None`` then use ``I``.
            (default: ``None``)
        :param start_time: (optional)
            Instance of ``astropy.time.Time`` class. (default: ``None``)
        :param stop_time: (optional)
            Instance of ``astropy.time.Time`` class. (default: ``None``)
        :param style: (optional)
            How to plot complex visibilities - real and imaginary part
            (``re&im``) or amplitude and phase (``a&p``). (default: ``a&p``)
        :param color: (optional)
            Default color.
        :param colors: (optional)
            Default colors for multi IF plotting.

        .. note:: All checks are in ``_choose_uvdata`` method.
        """

        if not pylab:
            raise Exception('Install ``pylab`` for plotting!')

        if stokes is None:
            stokes = 'I'

        if start_time is None:
            start_time = self.times[0]
        if stop_time is None:
            stop_time = self.times[-1]

        if baselines is None:
            baselines = self.baselines

        indxs = np.logical_and(self._get_baselines_indexes(baselines),
                               self._get_times_indexes(start_time, stop_time))
        uvdata = self._choose_uvdata(baselines=baselines, bands=bands,
                                     stokes=stokes, freq_average=freq_average)

        uvw_data = self.uvw[indxs]
        uv_radius = np.sqrt(uvw_data[:, 0] ** 2 + uvw_data[:, 1] ** 2)
        weights = self.weights[indxs]

        if style == 'a&p':
            a1 = np.angle(uvdata)
            a2 = np.real(np.sqrt(uvdata * np.conj(uvdata)))
        elif style == 're&im':
            a1 = uvdata.real
            a2 = uvdata.imag
        else:
            raise Exception('Only ``a&p`` and ``re&im`` styles are allowed!')

        if fig is None:
            fig, axes = matplotlib.pyplot.subplots(nrows=2, ncols=1, sharex=True,
                                                   sharey=False)
        else:
            axes = fig.get_axes()

        if not freq_average:
            # # of chosen IFs
            # TODO: Better use len(IF) if ``data`` shape will change sometimes.
            if bands is None:
                n_if = self.nif
            else:
                n_if = len(bands)

            # TODO: define colors
            if colors is not None:
                assert n_if <= len(colors)
                syms = colors[:n_if]
                syms = [".{}".format(color) for color in syms]
            else:
                print "Provide ``colors`` argument to show in different" \
                      " colors!"
                syms = ['.'] * n_if

            if n_if > 1 or stokes not in self.stokes:

                for _if in range(n_if):
                    # TODO: plot in different colors and make a legend
                    # Ignore default color if colors list is supplied
                    if colors is not None:
                        axes[0].plot(uv_radius, a2[:, _if], syms[_if])
                    else:
                        axes[0].plot(uv_radius, a2[:, _if], syms[_if],
                                     color=color, alpha=alpha)
                for _if in range(n_if):
                    # Ignore default color if colors list is supplied
                    if colors is not None:
                        axes[1].plot(uv_radius, a1[:, _if], syms[_if])
                    else:
                        axes[1].plot(uv_radius, a1[:, _if], syms[_if],
                                     color=color, alpha=alpha)
                axes[1].set_xlabel('UV-radius, wavelengths')

            elif n_if == 1 and stokes in self.stokes:

                sc_0 = axes[0].scatter(uv_radius, a2, cmap='gray_r',
                                       norm=matplotlib.colors.LogNorm(),
                                       c=weights[:, bands[0],
                                         self.stokes_dict_inv[stokes]])
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                divider_0 = make_axes_locatable(axes[0])
                cax_0 = divider_0.append_axes("right", size="2%", pad=0.00)
                cb_0 = fig.colorbar(sc_0, cax=cax_0)
                sc_1 = axes[1].scatter(uv_radius, a1, cmap='gray_r',
                                       norm=matplotlib.colors.LogNorm(),
                                       c=weights[:, bands[0],
                                         self.stokes_dict_inv[stokes]])
                cb_0.set_label('Weights')
                divider_1 = make_axes_locatable(axes[1])
                cax_1 = divider_1.append_axes("right", size="2%", pad=0.00)
                cb_1 = fig.colorbar(sc_1, cax=cax_1)
                cb_1.set_label('Weights')
                axes[1].set_xlabel('UV-radius, wavelengths')

            if style == 'a&p':
                axes[1].set_ylim([-math.pi, math.pi])
                axes[0].set_ylabel('Amplitude, [Jy]')
                axes[1].set_ylabel('Phase, [rad]')
                if amp_range is not None:
                    axes[0].set_ylim(amp_range)
                if phase_range is not None:
                    axes[1].set_ylim(phase_range)
            elif style == 're&im':
                axes[0].set_ylabel('Re, [Jy]')
                axes[1].set_ylabel('Im, [Jy]')
                if re_range is not None:
                    axes[0].set_ylim(re_range)
                if im_range is not None:
                    axes[1].set_ylim(im_range)
            axes[1].set_xlim(left=0)
            fig.show()
        else:
            if not sym:
                sym = '.'
            # axes[0].plot(uv_radius, a2, sym, color=color)
            # axes[1].plot(uv_radius, a1, sym, color=color)
            # FIXME: Doesn't work for stokes='I' (stokes_dict_inv)
            sc_0 = axes[0].scatter(uv_radius, a2, cmap='gray_r',
                                   norm=matplotlib.colors.LogNorm(),
                                   c=np.mean(weights[:, :, self.stokes_dict_inv[stokes]], axis=1))
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider_0 = make_axes_locatable(axes[0])
            cax_0 = divider_0.append_axes("right", size="2%", pad=0.00)
            cb_0 = fig.colorbar(sc_0, cax=cax_0)
            sc_1 = axes[1].scatter(uv_radius, a1, cmap='gray_r',
                                   norm=matplotlib.colors.LogNorm(),
                                   c=np.mean(weights[:, :, self.stokes_dict_inv[stokes]], axis=1))
            cb_0.set_label('Weights')
            divider_1 = make_axes_locatable(axes[1])
            cax_1 = divider_1.append_axes("right", size="2%", pad=0.00)
            cb_1 = fig.colorbar(sc_1, cax=cax_1)
            cb_1.set_label('Weights')
            axes[1].set_xlabel('UV-radius, wavelengths')
            if style == 'a&p':
                axes[1].set_ylim([-math.pi, math.pi])
                axes[1].set_xlim(left=0)
                axes[0].set_ylabel('Amplitude, [Jy]')
                axes[1].set_ylabel('Phase, [rad]')
                if amp_range is not None:
                    axes[0].set_ylim(amp_range)
                if phase_range is not None:
                    axes[1].set_ylim(phase_range)

            elif style == 're&im':
                axes[0].set_ylabel('Re, [Jy]')
                axes[1].set_ylabel('Im, [Jy]')
                if re_range is not None:
                    axes[0].set_ylim(re_range)
                if im_range is not None:
                    axes[1].set_ylim(im_range)

            fig.show()
        return fig

    def uvplot_model(self, model, baselines=None, stokes=None, style='a&p'):
        """
        Plot given image plain model.

        :param model:
            Instance of ``Model`` class.

        :param baselines: (optional)
            One or iterable of baselines numbers or ``None``. If ``None`` then
            use all baselines. (default: ``None``)

        :parm IF (optional):
            One or iterable of IF numbers (1-#IF) or ``None``. If ``None`` then
            use all IFs. (default: ``None``)

        :param stokes: (optional)
            Any string of: ``I``, ``Q``, ``U``, ``V``, ``RR``, ``LL``, ``RL``,
            ``LR`` or ``None``. If ``None`` then use ``I``.
            (default: ``None``)

        :param style: (optional)
            How to plot complex visibilities - real and imaginary part
            (``re&im``) or amplitude and phase (``a&p``). (default: ``a&p``)
        """
        # Copy ``model``, choose ``uvws`` given ``baselines`` and set ``_uvws``
        # atribute of ``model``'s copy to calculated ``uvws``. Use
        # ``model.uvplot()`` method to plot model.
        raise NotImplementedError

    def ft_to_image(self, image_params, baselines=None, IF=None, times=None,
                    freq_average=True):
        """
        FT uv-data to dirty map with specified parameters.

        :param image_params:
            Dictionary with image parameters.
        :param baselines: (optional)
            Baselines to use. If ``None`` then use all. (default: ``None``)
        :param IF: (optional)
            IFs to use. If ``None`` then use all. (default: ``None``)
        :param times: (optional)
            Time range to use. If ``None`` then use all. (default: ``None``)
        :param freq_average: (optional)
            Average IFs? (default: ``True``)

        :return:
            ``Image`` instance with dirty map.
        """
        # im(x, y) = vis(u, v) * np.exp(2. * math.pi * 1j * (u * x + v * y))
        # where x, y - distances from pase center [rad]
        raise NotImplementedError


if __name__ == '__main__':
    import os
    data_dir = '/home/ilya/code/vlbi_errors/bin'
    uvdata = UVData(os.path.join(data_dir, '0125+487_L.uvf_difmap'))
    noises = uvdata.noise(split_scans=True, use_V=False, average_freq=False)
