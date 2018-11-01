import os
import glob
import numpy as np
from uv_data import UVData
from sklearn.cross_validation import KFold
from utils import to_boolean_array, mask_boolean_with_boolean
from spydiff import import_difmap_model, modelfit_difmap, clean_difmap, clean_n
from model import Model
from from_fits import create_model_from_fits_file
import matplotlib.pyplot as plt


# TODO: Use only positive weighted data for CV
class KFoldCV(object):
    def __init__(self, uv_fits_path, k, basename='cv', seed=None,
                 baselines=None, stokes='I'):
        if stokes not in ('I', 'RR', 'LL'):
            raise Exception("Only stokes (I, RR, LL) supported!")
        self.stokes = stokes
        self.uv_fits_path = uv_fits_path
        self.uvdata = UVData(uv_fits_path)
        self.k = k
        self.seed = seed
        self.basename = basename
        self.test_fname_base = "{}_test".format(basename)
        self.train_fname_base = "{}_train".format(basename)
        self.baseline_folds = None
        self.create_folds(baselines)

    def create_folds(self, baselines=None):
        baseline_folds = dict()

        if baselines is None:
            baselines = self.uvdata.baselines

        if self.stokes == 'I':
            stokes = ['RR', 'LL']
            average_stokes = True
        elif self.stokes == 'RR':
            stokes = ['RR']
            average_stokes = False
        elif self.stokes == 'LL':
            stokes = ['LL']
            average_stokes = False
        else:
            raise Exception("Only stokes (I, RR, LL) supported!")

        for bl in baselines:
            bl_indxs = self.uvdata._indxs_baselines[bl]
            print "Baseline {} has {} samples".format(bl,
                                                      np.count_nonzero(bl_indxs))
            bl_indxs_pw = self.uvdata.pw_indxs_baseline(bl, average_bands=True,
                                                        stokes=stokes,
                                                        average_stokes=average_stokes)
            bl_indxs = mask_boolean_with_boolean(bl_indxs, bl_indxs_pw)
            print "Baseline {} has {} samples with positive weight".format(bl,
                                                      np.count_nonzero(bl_indxs))

            try:
                kfold = KFold(np.count_nonzero(bl_indxs), self.k, shuffle=True,
                              random_state=self.seed)
                baseline_folds[bl] = list()
                for train, test in kfold:
                    tr = to_boolean_array(np.nonzero(bl_indxs)[0][train],
                                          len(bl_indxs))
                    te = to_boolean_array(np.nonzero(bl_indxs)[0][test],
                                          len(bl_indxs))
                    baseline_folds[bl].append((tr, te))
            # When ``k`` more then number of baseline samples
            except ValueError:
                pass

        # Add all other baselines data w/o folding - all data to train & nothing
        # to test
        rest_baselines = list(self.uvdata.baselines)
        for bl in baselines:
            rest_baselines.remove(bl)
        for bl in rest_baselines:
            baseline_folds[bl] = list()
        for bl in rest_baselines:
            bl_indxs = self.uvdata._indxs_baselines[bl]
            for k in range(self.k):
                baseline_folds[bl].append((bl_indxs, np.zeros(len(bl_indxs),
                                                              dtype=bool)))

        self.baseline_folds = baseline_folds

    def create_train_test_data(self, outdir=None):
        if outdir is None:
            outdir = os.getcwd()
        for i in xrange(self.k):
            train_indxs = np.zeros(len(self.uvdata.hdu.data))
            test_indxs = np.zeros(len(self.uvdata.hdu.data))
            for bl, kfolds in self.baseline_folds.items():
                itrain, itest = kfolds[i]
                # itrain = to_boolean_array(itrain)
                train_indxs = np.logical_or(train_indxs, itrain)
                test_indxs = np.logical_or(test_indxs, itest)
            train_data = self.uvdata.hdu.data[train_indxs]
            test_data = self.uvdata.hdu.data[test_indxs]
            self.uvdata.save(os.path.join(outdir, self.test_fname_base + '_{}.fits'.format(i)),
                             test_data, rewrite=True)
            self.uvdata.save(os.path.join(outdir, self.train_fname_base + '_{}.fits'.format(i)),
                             train_data, rewrite=True)

    def cv_score(self, initial_dfm_model_path=None, data_dir=None, niter=100,
                 path_to_script=None, mapsize_clean=None):
        if data_dir is None:
            data_dir = os.getcwd()
        train_uv_fits_paths = sorted(glob.glob(os.path.join(data_dir,
                                                            self.train_fname_base+'*')))
        test_uv_fits_paths = sorted(glob.glob(os.path.join(data_dir,
                                                           self.test_fname_base+'*')))
        cv_scores = list()
        train_scores = list()
        if initial_dfm_model_path is not None:
            for i, (train_uv_fits_path, test_uv_fits_path) in enumerate(zip(train_uv_fits_paths,
                                                                            test_uv_fits_paths)):
                print "Calculating CV-score for {} of {} splits".format(i+1, self.k)
                print "Training FITS: {}".format(train_uv_fits_path)
                print "Testing FITS: {}".format(test_uv_fits_path)
                out_mdl_fname = 'train_{}.mdl'.format(i)
                dfm_model_dir, dfm_model_fname = os.path.split(initial_dfm_model_path)
                modelfit_difmap(train_uv_fits_path, dfm_model_fname,
                                out_mdl_fname, niter=niter,
                                path=data_dir, mdl_path=dfm_model_dir,
                                out_path=data_dir, stokes=self.stokes)
                cv_scores.append(score(test_uv_fits_path, os.path.join(data_dir, out_mdl_fname)))
                train_scores.append(score(train_uv_fits_path, os.path.join(data_dir, out_mdl_fname)))
        else:
            for i, (train_uv_fits_path, test_uv_fits_path) in enumerate(zip(train_uv_fits_paths,
                                                                            test_uv_fits_paths)):
                out_mdl_fname = 'train_{}.fits'.format(i)
                # This used when learning curves are created
                # clean_difmap(train_uv_fits_path, out_mdl_fname, 'I',
                #              mapsize_clean, data_dir, path_to_script,
                #              outpath=data_dir, show_difmap_output=True)
                # This used when different number of iterations are tested
                clean_n(train_uv_fits_path, out_mdl_fname, 'I',
                        mapsize_clean, niter=niter, path_to_script=path_to_script,
                        outpath=data_dir, show_difmap_output=True,)
                cv_scores.append(score(test_uv_fits_path, os.path.join(data_dir, out_mdl_fname)))
                train_scores.append(score(train_uv_fits_path, os.path.join(data_dir, out_mdl_fname)))

        return cv_scores, train_scores


def cv_difmap_models(dfm_model_files, uv_fits, K=5, baselines=None, n_iter=50,
                     out_dir=None, seed=None, n_rep=10, stokes='I'):
    """
    Function that Cross-Validates several difmap models of the same uv-data set.

    :param dfm_model_files:
        Iterable of paths to difmap-style model files.
    :param uv_fits:
        Path to FITS-file with uv-data.
    :param K: (optional)
        Number of CV folds to split the full uv-data set. One of the ``K`` folds
        will be test sample. (default: ``5``)
    :param baselines: (optional)
        Iterable of baseline number to consider in CV. If ``None`` then use all.
        (default: ``None``)
    :param n_iter: (optional)
        Number of iterations for difmap. (default: ``50``)
    :param out_dir:
        Directory to store temporal FITS-files with splitted data. If ``None``
        then use CWD. (default: ``None``)
    :param seed: (optional)
        Random seed to use. If ``None`` then use random integer - that is used
        when K-fold CV is repeated ``n_rep`` times to estimate the uncertainty
        of CV-scores. (default: ``None``)
    :param n_rep:  (optional)
        Number of times K-fold CV is repeated  to estimate the uncertainty of
        CV-scores. (default: ``10``)
    :param stokes: (optional)
        Stokes parameter string. ``I``, ``RR`` or ``LL`` are currently
        supported. (default: ``I``)
    :return:
        Dictionary with keys - number denoting the model and values - lists of
        ``n_rep`` values of CV-scores. One can use their mean and std to get
        single value and it's uncertainty for each of the models.
    """
    if seed is not None and n_rep > 1:
        raise Exception("You don't need fixed seed with n > 1!")
    if out_dir is None:
        out_dir = os.getcwd()
    mdl_dict = {i: mdl_file for (i, mdl_file) in enumerate(dfm_model_files)}

    cv_means = dict()
    cv_stds = dict()

    # If no seed is specified and no repetitions => choose some seed to compare
    # models using the same splits.
    if seed is None and n_rep == 1:
        seed = np.random.randint(0, 1000)
    for i in mdl_dict:
        print "Calculating CV scores for model {}".format(i)
        cv_means[i] = list()
        cv_stds[i] = list()
        for j in range(n_rep):
            print "Doing {} of {} repetitions".format(j+1, n_rep)
            if seed is None:
                seed_used = np.random.randint(0, 1000)
            else:
                seed_used = seed
            print "Using seed {}".format(seed_used)
            kfold = KFoldCV(uv_fits, K, seed=seed_used, baselines=baselines,
                            stokes=stokes)
            kfold.create_train_test_data(outdir=out_dir)
            cv_scores, train_scores =\
                kfold.cv_score(initial_dfm_model_path=mdl_dict[i],
                               data_dir=out_dir, niter=n_iter)
            print "Calculated scores (CV, train) :"
            print cv_scores
            print train_scores
            cv_means[i].append(np.mean(cv_scores))
            cv_stds[i].append(np.std(cv_scores))

    return cv_means, cv_stds


def score(uv_fits_path, mdl_path, stokes='I'):
    """
    Returns rms of model on given uv-data for stokes 'I'.

    :param uv_fits_path:
        Path to uv-fits file.
    :param mdl_path:
        Path to difmap model text file or FITS-file with CLEAN model.
    :param stokes: (optional)
        Stokes parameter string. ``I``, ``RR`` or ``LL`` currently supported.
        (default: ``I``)
    :return:
        Per-point rms between given data and model evaluated at given data
        points.
    """
    if stokes not in ('I', 'RR', 'LL'):
        raise Exception("Only stokes (I, RR, LL) supported!")
    uvdata = UVData(uv_fits_path)
    uvdata_model = UVData(uv_fits_path)
    try:
        model = create_model_from_fits_file(mdl_path)
    except IOError:
        dfm_mdl_dir, dfm_mdl_fname = os.path.split(mdl_path)
        comps = import_difmap_model(dfm_mdl_fname, dfm_mdl_dir)
        model = Model(stokes=stokes)
        model.add_components(*comps)
    uvdata_model.substitute([model])
    uvdata_diff = uvdata - uvdata_model
    if stokes == 'I':
        i_diff = 0.5 * (uvdata_diff.uvdata_weight_masked[..., 0] +
                        uvdata_diff.uvdata_weight_masked[..., 1])
    elif stokes == 'RR':
        i_diff = uvdata_diff.uvdata_weight_masked[..., 0]
    elif stokes == 'LL':
        i_diff = uvdata_diff.uvdata_weight_masked[..., 1]
    else:
        raise Exception("Only stokes (I, RR, LL) supported!")
    # 2 means that Re & Im are counted independently
    factor = 2 * np.count_nonzero(i_diff)
    # factor = np.count_nonzero(~uvdata_diff.uvdata_weight_masked.mask[:, :, :2])
    # squared_diff = uvdata_diff.uvdata_weight_masked[:, :, :2] * \
    #                uvdata_diff.uvdata_weight_masked[:, :, :2].conj()
    squared_diff = i_diff * i_diff.conj()
    return np.sqrt(float(np.sum(squared_diff)) / factor)


if __name__ == '__main__':

    dfm_mdl_files = ['/home/osh/Dropbox/Ilya/2CV/0235+164.q1.2008_08_15.mdl',
                     '/home/osh/Dropbox/Ilya/2CV/0235+164.q1.2008_08_15.new.mdl',
                     '/home/osh/Dropbox/Ilya/2CV/0235+164.q1.2008_08_15.Jorstad.mdl']
    uv_fits = '/home/osh/Dropbox/Ilya/2CV/0235+164.q1.2008_08_15.uvp'
#    dfm_mdl_files = ['/home/ilya/Dropbox/0235/tmp/q_models_for_CV/q_mod1.mdl',
#                     '/home/ilya/Dropbox/0235/tmp/q_models_for_CV/q_mod2.mdl',
#                     '/home/ilya/Dropbox/0235/tmp/q_models_for_CV/q_mod3.mdl',
#                     '/home/ilya/Dropbox/0235/tmp/q_models_for_CV/q_mod4.mdl',
#                     '/home/ilya/Dropbox/0235/tmp/q_models_for_CV/q_mod5.mdl']
#    uv_fits = '/home/ilya/Dropbox/0235/tmp/to_compare/0235+164.k1.2008_09_02.uvf_difmap'
    cv_means, cv_stds =\
        cv_difmap_models(dfm_mdl_files, uv_fits, baselines=None, K=5,
                         out_dir='/home/osh/Dropbox/Ilya/2CV',
                         n_iter=300, n_rep=30, stokes='I')
    # Plot CV-RMSE versus model number
    keys = sorted(cv_means.keys())
    plt.figure()
    plt.errorbar(keys, [cv_means[i][0] for i in keys],
                 yerr=[cv_stds[i][0] for i in keys])
    plt.xlabel(r'model number')
    plt.ylabel(r'RMSE')

    # a = np.array(cv_scores.values())
    # y = a[:, 0]
    # yerr = a[:, 1]
    # label_size = 12
    # import matplotlib
    # matplotlib.rcParams['xtick.labelsize'] = label_size
    # matplotlib.rcParams['ytick.labelsize'] = label_size
    # matplotlib.rcParams['axes.titlesize'] = label_size
    # matplotlib.rcParams['axes.labelsize'] = label_size
    # matplotlib.rcParams['font.size'] = 12
    # matplotlib.rcParams['legend.fontsize'] = 12

    # import matplotlib.pyplot as plt
    # plt.figure()
    # # plt.errorbar(np.arange(len(dfm_mdl_files))+1, y, yerr, lw=2)
    # plt.plot(np.arange(len(dfm_mdl_files))+1, y)
    # # plt.xlim([0.9, len(dfm_mdl_files) + 0.1])
    # plt.xlabel("Model number")
    # plt.ylabel("CV score")
    # # plt.xticks(range(len(dfm_mdl_files)))
    # plt.show()
    # # plt.savefig('/home/ilya/Dropbox/papers/boot/new_pics/cv_cc.eps',
    # #             bbox_inches='tight', format='eps', dpi=500)
    # # plt.savefig('/home/ilya/Dropbox/papers/boot/new_pics/cv_cc.svg',
    #             bbox_inches='tight', format='svg', dpi=500)


    # cv_scores_ = list()
    # for i in range(10):
    #     cv_scores = cv_model(dfm_mdl_files, uv_fits, baselines=[774, 1546], K=10,
    #                          dfm_model_dir='/home/ilya/code/vlbi_errors/bin_c1')
    #     cv_scores_.append(cv_scores)
    # print cv_scores_
    # import matplotlib.pyplot as plt
    # plt.figure()
    # a = np.array(cv_scores_.values())[..., 0].T
    # for ar in a:
    #     plt.plot(np.arange(len(dfm_mdl_files)) +
    #              np.random.normal(0, 0.03, size=3), ar, '.k', lw=2)
    # plt.xlim([-0.1, len(dfm_mdl_files) -0.9])
    # plt.xlabel("Model number")
    # plt.ylabel("CV score, lower - better")
    # plt.xticks(range(len(dfm_mdl_files)))
    # plt.show()


    # # Plot CV-score vs. N_clean
    # data_dir = '/home/ilya/Dropbox/papers/boot/new_pics/cv_cc/'
    # from spydiff import clean_n
    # # cc_pars = [50, 75, 100, 125, 150, 200, 300, 500, 1000, 2500, 5000, 10000]
    # cc_pars = [50, 100, 200, 500, 1000, 5000, 10000]
    # path_to_script = '/home/ilya/code/vlbi_errors/difmap/clean_n'
    # uv_fits_fname = '0055+300.u.2006_02_12.uvf'
    # uv_fits_path = os.path.join(data_dir, uv_fits_fname)
    # cv_scores = dict()
    # n_folds = 10
    # cv_scores_ncc = dict()
    # train_scores_ncc = dict()
    # for niter in cc_pars:
    #     print "===================="
    #     print "Using {} iterations!".format(niter)
    #     print "===================="
    #     cv_scores_ncc[niter] = list()
    #     train_scores_ncc[niter] = list()
    #     print "Using niter = {}".format(niter)
    #     for i in range(10):
    #         kfold = KFoldCV(uv_fits_path, n_folds, seed=np.random.randint(0, 1000))
    #         kfold.create_train_test_data(outdir=data_dir)
    #         cv_scores, train_scores = kfold.cv_score(initial_dfm_model_path=None,
    #                                                  data_dir=data_dir,
    #                                                  path_to_script=path_to_script,
    #                                                  mapsize_clean=(512, 0.1),
    #                                                  niter=niter)
    #         cv_scores_ncc[niter].append(np.mean(cv_scores))
    #         train_scores_ncc[niter].append(np.mean(train_scores))

    # import matplotlib.pyplot as plt
    # import matplotlib
    # label_size = 15
    # matplotlib.rcParams['xtick.labelsize'] = label_size
    # matplotlib.rcParams['ytick.labelsize'] = label_size
    # matplotlib.rcParams['axes.titlesize'] = label_size
    # matplotlib.rcParams['axes.labelsize'] = label_size
    # matplotlib.rcParams['font.size'] = label_size
    # matplotlib.rcParams['legend.fontsize'] = label_size
    # import matplotlib.pyplot as plt
    # plt.semilogy()
    # plt.semilogx()
    # plt.figure()
    # plt.errorbar(sorted(cv_scores_ncc.keys()),
    #              y=[np.mean(cv_scores_ncc[niter]) for niter in sorted(cv_scores_ncc.keys())],
    #              yerr=[np.std(cv_scores_ncc[niter]) for niter in sorted(cv_scores_ncc.keys())],
    #              label='CV', ls='solid')
    # plt.scatter(sorted(cv_scores_ncc.keys()),
    #             [np.mean(cv_scores_ncc[niter]) for niter in sorted(cv_scores_ncc.keys())],
    #             s=10, marker='o')
    # plt.errorbar(sorted(train_scores_ncc.keys()),
    #              y=[np.mean(train_scores_ncc[niter]) for frac in sorted(train_scores_ncc.keys())],
    #              yerr=[np.std(train_scores_ncc[niter]) for frac in sorted(train_scores_ncc.keys())],
    #              label='Train', ls='dashed')
    # plt.scatter(sorted(train_scores_ncc.keys()),
    #             [np.mean(train_scores_ncc[niter]) for frac in sorted(train_scores_ncc.keys())],
    #             s=10, marker='s')
    # plt.legend()
    # plt.xlabel(r"$N_{CC}$")
    # plt.ylabel(r"$RMSE$")
    # plt.show()

    # plt.savefig(os.path.join(data_dir, 'cv_cc.pdf'),
    #             bbox_inches='tight', format='pdf', dpi=1200)

