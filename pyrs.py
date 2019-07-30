#!/usr/bin/env python3
"""
**pyrs.py
** Copyright (C) 2019  Jose Sergio Hleap

Compute a polygenic risk score using the P+T method

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

E-mail: jose.hleaplozano@mcgill.ca

Python modules:
1. pandas
2. matplotlib
3. Dask
4. pandas_plink
"""
import argparse
import os
import time
from functools import reduce
from itertools import cycle, product
from multiprocessing.pool import ThreadPool

import dask
import dask.array as da
import gc
import h5py
import mpmath as mp
import numpy as np
import pandas as pd
import psutil
import statsmodels.formula.api as smf
from chest import Chest
from dask.diagnostics import ProgressBar
from dask_ml.decomposition import PCA
from pandas_plink import read_plink
from scipy.stats import linregress
from sklearn.model_selection import train_test_split

from qtraitsimulation import qtraits_simulation

# -----------------------------------------------------------------------------
def estimate_chunks(shape, threads, memory=None):
    """
    Estimate the appropriate chunks to split arrays in the dask format to made
    them fit in memory. If Memory is None, it will be set to a tenth of the
    total memory. It also takes into account the number of threads

    :param tuple shape: Shape of the array to be chunkenized
    :param threads: Number of threads intended to be used
    :param memory: Memory limit
    :return: The appropriate chunk in tuple form
    """
    total = psutil.virtual_memory().available  # a tenth of the memory
    avail_mem = total if memory is None else memory  # Set available memory
    size = (reduce(np.multiply, shape) * 8) / 1E6
    usage = size * threads  # Compute threaded estimated size
    # Determine number of chunks given usage and available memory
    n_chunks = np.ceil(usage / avail_mem).astype(int)
    # Mute divided by zero error only for this block of code
    with np.errstate(divide='ignore', invalid='ignore'):
        estimated = tuple(np.array(shape) / n_chunks)  # Get chunk estimation
    chunks = min(shape, tuple(estimated))            # Fix if n_chunks is 0
    return tuple(int(i) for i in chunks)  # Ensure chunks as tuple of integers


# -----------------------------------------------------------------------------
def read_geno(bedfileset, normalize=True):
    # read geno
    bim, fam, g = read_plink(bedfileset)
    if normalize:
        # normalize geno
        std = g.std(axis=1)
        mean = g.mean(axis=1)
        ng = (g.T - mean) / std
        return ng, bim, fam
    else:
        return g, bim, fam


# -----------------------------------------------------------------------------
def t_sf(t, df):
    """
    Student t distribution cumulative density function or survival function

    :param t: t statistic
    :param df: degrees of freedom
    :return: area under the PDF from -inf to t
    """
    t = -mp.fabs(t)
    lhs = mp.gamma((df + 1) / 2) / (mp.sqrt(df * mp.pi) * mp.gamma(df / 2))
    rhs = mp.quad(lambda x: (1 + (x * x) / df) ** (-df / 2 - 1 / 2),
                  [-mp.inf, t])
    gc.collect()
    return lhs * rhs


class GWAS(object):
    def __init__(self, filesetprefix, pheno, outprefix, threads, client=None,
                 check=False, freq_thresh=0.01, flip=False, max_memory=None,
                 seed=None, usable_snps=None, covs=None, **kwargs):
        self.bed = []
        self.bim = None
        self.fam = None
        self.cache = None
        self.seed = seed
        self.kwargs = kwargs
        self.usable_snps = usable_snps
        self.outpref = outprefix
        self.threads = threads
        self.client = client
        self.max_memory = max_memory
        self.flip = flip
        self.check = check
        self.freq_thresh = freq_thresh
        self.geno = filesetprefix
        self.pheno = pheno
        self.sum_stats = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.p_values = None
        self.covs = covs
        print(self.__dict__)

    @property
    def covs(self):
        return self.__geno

    @covs.setter
    def covs(self, covs):
        if isinstance(covs, str):
            self.__covs = pd.read_csv(covs, sep='\t')
        elif isinstance(covs, pd.core.frame.DataFrame):
            self.__covs = covs
        else:
            raise NotImplementedError


    @property
    def geno(self):
        return self.__geno

    @geno.setter
    def geno(self, filesetprefix):
        bed = '%s.bed' % filesetprefix
        op = dict(check=self.check, usable_snps=self.usable_snps,
                  max_memory=self.max_memory)
        (bim, fam, geno) = self.read_geno(bed, self.freq_thresh, self.threads,
                                          **op)
        if self.bed:
            self.bed.append(bed)
            self.bim = pd.concat((self.bim, bim), axis=0).reset_index(drop=True
                                                                      )
            pd.testing.assert_frame_equal(self.fam, fam)
            self.__geno = da.concatenate([self.geno, geno], axis=1)
        else:
            self.bed.append(bed)
            self.bim = bim
            self.fam = fam
            self.__geno = geno

    @property
    def pheno(self):
        return self.__pheno

    @pheno.setter
    def pheno(self, pheno):
        if pheno is None:
            options = dict(outprefix=self.outpref, bfile=self.geno, h2=0.5,
                           ncausal=10, normalize=True, uniform=False,
                           snps=None, seed=self.seed, bfile2=None,
                           flip=self.flip, max_memory=self.max_memory,
                           fam=self.fam, high_precision_on_zero=False,
                           bim=self.bim)
            # If pheno is not provided, simulate it using qtraits_simulation
            options.update(self.kwargs)
            pheno, h2, gen = qtraits_simulation(**options)
            (x, bim, truebeta, vec) = gen
            self.truebeta = truebeta
            self.causals = vec

        elif isinstance(pheno, str):
            # If pheno is provided as a string, read it
            pheno = pd.read_csv(pheno, delim_whitespace=True, header=None,
                                names=['fid', 'iid', 'PHENO'])
        # else:
        #     pheno = self.__pheno
        try:
            y = pheno.compute(num_workers=self.threads, cache=self.cache)
        except AttributeError:
            y = pheno
        self.__pheno = y

    @property
    def seed(self):
        return self.__seed

    @seed.setter
    def seed(self, seed):
        self.__seed = np.random.randint(1e4) if seed is None else seed

    @property
    def max_memory(self):
        return self.__max_memory

    @max_memory.setter
    def max_memory(self, max_memory):
        # set Cache to protect memory spilling
        if max_memory is not None:
            available_memory = max_memory
        else:
            available_memory = psutil.virtual_memory().available
        self.__max_memory = available_memory
        self.cache = Chest(available_memory=available_memory)

    @staticmethod
    def read_geno(bfile, freq_thresh, threads, flip=False, check=False,
                  max_memory=None, usable_snps=None):
        """
        Read the plink bed fileset, restrict to a given frequency (optional,
        freq_thresh), flip the sequence to match the MAF (optional; flip), and
        check if constant variants present (optional; check)

        :param max_memory: Maximum allowed memory
        :param bfile: Prefix of the bed (plink) fileset
        :param freq_thresh: If greater than 0, limit MAF to at least freq_thresh
        :param threads: Number of threads to use in computation
        :param flip: Whether to check for flips and to fix the genotype file
        :param check: Whether to check for constant sites
        :return: Dataframes (bim, fam) and array corresponding to the bed fileset
        """
        # set Cache to protect memory spilling
        if max_memory is not None:
            available_memory = max_memory
        else:
            available_memory = psutil.virtual_memory().available
        cache = Chest(available_memory=available_memory)
        (bim, fam, g) = read_plink(bfile)  # read the files using pandas_plink
        m, n = g.shape  # get the dimensions of the genotype
        # remove invariant sites
        if check:
            g_std = g.std(axis=1)
            with ProgressBar(), dask.config.set(pool=ThreadPool(threads)):
                print('Removing invariant sites')
                idx = (g_std != 0).compute(cache=cache)
            g = g[idx, :]
            bim = bim[idx].copy().reset_index(drop=True)
            bim.i = bim.index.tolist()
            del g_std, idx
            gc.collect()
        if usable_snps is not None:
            idx = bim[bim.snp.isin(usable_snps)].i.tolist()
            g = g[idx, :]
            bim = bim[bim.i.isin(idx)].copy().reset_index(drop=True)
            bim.i = bim.index.tolist()
        # compute the mafs if required
        mafs = g.sum(axis=1) / (2 * n) if flip or freq_thresh > 0 else None
        if flip:
            # check possible flips
            flips = np.zeros(bim.shape[0], dtype=bool)
            flips[np.where(mafs > 0.5)[0]] = True
            bim['flip'] = flips
            vec = np.zeros(flips.shape[0])
            vec[flips] = 2
            # perform the flipping
            g = abs(g.T - vec)
            del flips
            gc.collect()
        else:
            g = g.T
        # Filter MAF
        if freq_thresh > 0:
            print('Filtering MAFs smaller than', freq_thresh)
            print('    Genotype matrix shape before', g.shape)
            assert freq_thresh < 0.5
            good = (mafs < (1 - float(freq_thresh))) & (
                        mafs > float(freq_thresh))
            with ProgressBar():
                with dask.config.set(pool=ThreadPool(threads)):
                    good, mafs = dask.compute(good, mafs, cache=cache)
            g = g[:, good]
            print('    Genotype matrix shape after', g.shape)
            bim = bim[good]
            bim['mafs'] = mafs[good]
            del good
            gc.collect()
        bim = bim.reset_index(drop=True)  # Get the indices in order
        # Fix the i such that it matches the genotype indices
        bim['i'] = bim.index.tolist()
        # Get chunks apropriate with the number of threads
        g = g.rechunk(
            estimate_chunks(g.shape, threads, memory=available_memory))
        del mafs
        gc.collect()
        return bim, fam, g


    @staticmethod
    def nu_linregress(x, y, **kwargs):
        """
        Refactor of the scipy linregress with mpmath in the estimation of the
        pvalue, numba, and less checks for speed sake

        :param x: array for independent variable
        :param y: array for the dependent variable
        :return: dictionary with slope, intercept, r, pvalue and stderr
        """
        cols = ['slope', 'intercept', 'rvalue', 'pvalue', 'stderr']
        # Make sure x and y are arrays
        x = np.asarray(x)
        y = np.asarray(y)
        n = len(x)
        # means in vector form
        xmean = np.mean(x, None)
        ymean = np.mean(y, None)
        # average sum of squares:
        ssxm, ssxym, ssyxm, ssym = np.cov(x, y, bias=True).flat
        r_num = ssxym
        r_den = np.sqrt(ssxm * ssym)
        # Estimate correlation
        r = r_num / r_den
        # test for numerical error propagation
        if r > 1.0:
            r = 1.0
        elif r < -1.0:
            r = -1.0
        # estimate degrees of freedom
        df = n - 2
        slope = r_num / ssxm
        intercept = ymean - slope * xmean
        # Estimate t-statistic
        t = r * np.sqrt(df / ((1.0 - r) * (1.0 + r)))
        # Get the pvalue
        prob = 2 * t_sf(t, df)
        # get the estimated standard error
        sterrest = np.sqrt((1 - r * r) * ssym / ssxm / df)
        return dict(zip(cols, [slope, intercept, r, prob, sterrest]))

    @staticmethod
    def high_precision_pvalue(df, r):
        r = r if np.abs(r) != 1.0 else mp.mpf(0.9999999999999999) * mp.sign(r)
        den = ((1.0 - r) * (1.0 + r))
        t = r * np.sqrt(df / den)
        return t_sf(t, df) * 2

    def manhattan_plot(self, causal_pos=None, alpha=0.05):
        """
        Generates a manhattan plot for a list of p-values. Overlays a
        horizontal line indicating the Bonferroni significance threshold
        assuming all p-values derive from independent test.
        """
        # TODO: include coloring by chromosome
        # Compute the bonferrony corrected threshold
        bonferroni_threshold = alpha / len(self.p_values)
        # Make it log
        log_b_t = -np.log10(bonferroni_threshold)
        self.p_values[np.where(self.p_values < 1E-10)] = 1E-10
        # Make the values logaritm
        vals = -np.log10(self.p_values)
        # Plot it
        fig = plt.figure()
        ax2 = fig.add_subplot(111)
        # Add threshold line
        ax2.axhline(y=log_b_t, linewidth=1, color='r', ls='--')
        # Add shaded regions on the causal positions
        if causal_pos is not None:
            [ax2.axvspan(x - 0.2, x + 0.2, facecolor='0.8', alpha=0.8) for x in
             causal_pos]
        # Plot one point per value
        ax2.plot(vals, '.', ms=1)
        # Zoom-in / limit the view to different portions of the data
        ymax = max(vals)
        ax2.set_ylim(0, ymax + 0.2)  # most of the data
        ax2.set_xlim([-0.2, len(vals) + 1])
        plt.xlabel(r"marker index")
        plt.ylabel(r"-log10(p-value)")
        plt.savefig('%s.pdf' % self.outpref)
        plt.close()

    @staticmethod
    def st_mod(x, y, covs):
        """
        Linear regression using stats models. This module is very slow but
        allows to include covariates in the estimation.

        :param x: array for independent variable
        :param y: array for dependent variable
        :param covs: array for covariates
        :return: Regression results
        """
        df = pd.DataFrame({'geno': x, 'pheno': y})
        cols = ['slope', 'intercept', 'rvalue', 'pvalue', 'stderr', 'b_pval',
                'b_std_err']
        if np.allclose(x.var(), 0.0):
            linregress_result = dict(zip(cols, cycle([np.nan])))
        else:
            if covs is not None:
                c = []
                cols = covs.columns.tolist()
                cols = [x for x in cols if x not in ['fid', 'iid']]
                for col in cols:
                    n = 'Cov%d' % col
                    df[n] = covs[:, col]
                    c.append(n)
                formula = 'pheno ~ geno + %s' % ' + '.join(c)
            else:
                formula = 'pheno ~ geno'
            model = smf.ols(formula=formula, data=df)
            results = model.fit()
            vals = [results.params.Intercept, results.params.geno,
                    results.pvalues.Intercept, results.pvalues.geno,
                    results.rsquared, results.bse.Intercept, results.bse.geno]
            linregress_result = dict(zip(cols, vals))
        return linregress_result

    @staticmethod
    def linregress(x, y, **kwargs):
        linregress_result = linregress(x, y)
        return linregress_result

    @staticmethod
    def do_pca(g, n_comp):
        """
        Perform a PCA on the genetic array and return n_comp of it

        :param g: Genotype array
        :param n_comp: Number of components sought
        :return: components array
        """
        pca = PCA(n_components=n_comp)
        pca = pca.fit_transform(g)
        return pca

    def load_previous_run(self):
        """
        Load a previos GWAS run

        :param prefix: The prefix of the output files from the previous run
        :param threads: Number of threads to be used in the estimations
        :return: previous gwas results
        """
        # Get the file names
        pfn = '%s_phenos.hdf5' % self.outpref
        gfn = '%s.geno.hdf5' % self.outpref
        f = h5py.File(gfn, 'r')  # Read the genotype h5 file
        x_train = da.from_array(f.get('x_train'))
        x_train.rechunk((x_train.shape[0], 1))
        # Get the test set of the genotype array
        x_test = da.from_array(f.get('x_test'))
        x_test.rechunk((x_test.shape[0], 1))
        # Get the training set of the phenotype
        y_train = pd.read_hdf(pfn, key='y_train')
        # Get the testing set of the phenotype
        y_test = pd.read_hdf(pfn, key='y_test')
        # Read the resulting gwas table
        res = pd.read_csv('%s.gwas' % self.outpref, sep='\t')
        return res, x_train, x_test, y_train, y_test

    def plink_free_gwas(self, validate=None, plot=False, causal_pos=None,
                        pca=None, stmd=False, high_precision=False,
                        high_precision_on_zero=False, **kwargs):
        """
        Compute the least square regression for a genotype in a phenotype. This
        assumes that the phenotype has been computed from a nearly independent
        set of variants to be accurate (I believe that that is the case for
        most programs but it is not "advertised")
        """
        seed = self.seed
        print('Performing GWAS\n    Using seed', seed)
        now = time.time()
        pfn = '%s_phenos.hdf5' % self.outpref
        gfn = '%s.geno.hdf5' % self.outpref
        if os.path.isfile(pfn):
            res, x_train, x_test, y_train, y_test = self.load_previous_run()
        else:
            np.random.seed(seed=seed)
            if validate is not None:
                print('making the crossvalidation data')
                x_train, x_test, y_train, y_test = train_test_split(
                    self.geno, self.pheno, test_size=1 / validate,
                    random_state=seed)
            else:
                x_train, x_test = self.geno, self.geno
                y_train, y_test = self.pheno, self.pheno
            assert not da.isnan(x_train).any().compute(threads=self.threads)
            # write test and train IDs
            opts = dict(sep=' ', index=False, header=False)
            y_test.to_csv('%s_testIDs.txt' % self.outpref, **opts)
            y_train.to_csv('%s_trainIDs.txt' % self.outpref, **opts)
            if isinstance(x_train, dask.array.core.Array):
                x_train = x_train.rechunk((x_train.shape[0], 1)).astype(
                    np.float)
            if 'normalize' in kwargs:
                if kwargs['normalize']:
                    print('Normalizing train set to variance 1 and mean 0')
                    x_train = (x_train - x_train.mean(axis=0)) / x_train.std(
                        axis=0)
                    print('Normalizing test set to variance 1 and mean 0')
                    x_test = (x_test - x_test.mean(axis=0)) / x_test.std(axis=0
                                                                         )
            # Get apropriate function for linear regression
            func = self.nu_linregress if high_precision else self.st_mod \
                if stmd else self.linregress
            daskpheno = da.from_array(y_train.PHENO.values).astype(np.float)
            if pca is not None:
                print('Using %d PCs' % pca)
                # Perform PCA
                func = self.st_mod  # Force function to statsmodels
                pcs = pd.DataFrame(self.do_pca(x_train, pca))  # Estimate PCAs
                if self.covs is not None:
                    covs_train = y_train.reindex(columns='iid').merge(
                        self.covs,  on=['iid'], how='left')
                    assert covs_train.shape[0] == y_train.shape[0]
                    covs = pd.concat([covs_train, pcs], axis=1)
                else:
                    pcs['fid'] = y_train.fid
                    pcs['iid'] = y_train.iid
                    covs = pcs
                combos = product((x_train[:, x] for x in range(
                    x_train.shape[1])), [daskpheno], covs)
            else:
                combos = product((x_train[:, x] for x in range(
                    x_train.shape[1])), [daskpheno])
            print('Performing regressions')
            delayed_results = [dask.delayed(func)(x, y) for x, y in combos]
            with ProgressBar():
                r = dask.compute(*delayed_results, scheduler='threads')
            gc.collect()
            try:
                res = pd.DataFrame.from_records(list(r), columns=r[0]._fields)
            except AttributeError:
                res = pd.DataFrame(r)
            assert res.shape[0] == self.bim.shape[0]
            # Combine mapping and gwas
            res = pd.concat((res, self.bim.reset_index()), axis=1)
            # check precision issues and re-run the association
            zeros = res[res.pvalue == 0.0]
            if not zeros.empty and not stmd and high_precision_on_zero:
                print('    Processing zeros with arbitrary precision')
                df = x_train.shape[0] - 2
                combos = product(df, zeros.rvalue.values)
                with ThreadPool(self.threads) as p:
                    results = p.starmap(self.high_precision_pvalue, combos)
                zero_res = np.array(*results)
                res.loc[res.pvalue == 0.0, 'pvalue'] = zero_res
                res['pvalue'] = [mp.mpf(z) for z in res.pvalue]
            self.p_values = res.pvalue.values
            # Make a manhatan plot
            if plot:
                self.manhattan_plot(causal_pos, alpha=plot)
            # write files
            res.to_csv('%s.gwas' % self.outpref, sep='\t', index=False)
            labels = ['/x_train', '/x_test']
            arrays = [x_train, x_test]
            hdf_opt = dict(table=True, mode='a', format="table")
            y_train.to_hdf(pfn, 'y_train', **hdf_opt)
            y_test.to_hdf(pfn, 'y_test', **hdf_opt)
            assert len(x_train.shape) == 2
            assert len(x_test.shape) == 2
            chunks = np.array([x_train.shape, x_test.shape])
            np.save('chunks.npy', chunks)
            data = dict(zip(labels, arrays))
            da.to_hdf5(gfn, data)
        print('GWAS DONE after %.2f seconds !!' % (time.time() - now))
        self.sum_stats = res
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test


class PRS(object):
    def __init__(self, bedfileset, sum_stats, pheno=None, ld_range=None,
                 pval_range=None, check=True, memory=None, threads=1,
                 snp_list=None, outpref='prs', cv=3, freq_thresh=0.1,
                 normalize=True, **kargs):
        self.kwargs = kargs
        self.normalize = normalize
        self.cache = None
        self.memory = memory
        self.threads = threads
        self.check = check
        self.freq_thresh = freq_thresh
        self.bim = None
        self.fam = None
        self.geno = bedfileset
        self.sum_stats = sum_stats
        self.ld_range = ld_range
        self.outpref = outpref
        self.pheno = pheno
        self.pval_range = pval_range
        self.snp_list = snp_list
        self.rho = self.geno
        self.cv = cv
        self.index = None
        self.best = None

    def __deepcopy__(self):
        return self

    @property
    def pval_range(self):
        return self.__pval_range

    @pval_range.setter
    def pval_range(self, pval_range):
        if pval_range is None:
            r = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 10e-4, 10e-6, 10e-8]
            self.__pval_range = r
        else:
            self.__pval_range = pval_range

    @property
    def ld_range(self):
        return self.__ld_range

    @ld_range.setter
    def ld_range(self, ld_range):
        if ld_range is None:
            self.__ld_range = np.arange(0.1, 0.8, 0.1)
        elif isinstance(ld_range, tuple) or isinstance(ld_range, list):
            self.__ld_range = np.arange(ld_range[0], ld_range[1], ld_range[3])
        else:
            self.__ld_range = ld_range
            assert isinstance(ld_range, np.ndarray)

    @property
    def geno(self):
        return self.__geno

    @geno.setter
    def geno(self, geno):
        if isinstance(geno, str):
            op = dict(check=self.check, max_memory=self.memory,
                      normalize=self.normalize)
            out = self.read_geno(geno, self.freq_thresh, self.threads, **op)
            g, self.bim, self.fam = out
        elif isinstance(geno, tuple):
            self.bim, self.fam, g = geno
            if not self.is_transposed(g, self.bim.shape[0], self.fam.shape[0]):
                g = g.t
        else:
            assert isinstance(geno, da.core.Array)
            g = geno
        self.__geno = g
        print('Gentotype file with %d individuals and %d variants' % (g.shape[
            0], g.shape[1]))

    @staticmethod
    def is_transposed(g, famshape, bimshape):
        if g.shape[0] == famshape:
            return True
        else:
            assert g.shape[0] == bimshape
            return False

    @staticmethod
    def read_geno(bfile, freq_thresh, threads, check=False, max_memory=None,
                  usable_snps=None, normalize=False):
        # set Cache to protect memory spilling
        if max_memory is not None:
            available_memory = max_memory
        else:
            available_memory = psutil.virtual_memory().available
        cache = Chest(available_memory=available_memory)
        (bim, fam, g) = read_plink(bfile)  # read the files using pandas_plink
        g_std = g.std(axis=1)
        if check:
            with ProgressBar(), dask.config.set(pool=ThreadPool(threads)):
                print('Removing invariant sites')
                idx = (g_std != 0).compute(cache=cache)
            g = g[idx, :]
            bim = bim[idx].copy().reset_index(drop=True)
            bim.i = bim.index.tolist()
            del idx
            gc.collect()
        if usable_snps is not None:
            idx = bim[bim.snp.isin(usable_snps)].i.tolist()
            g = g[idx, :]
            bim = bim[bim.i.isin(idx)].copy().reset_index(drop=True)
            bim.i = bim.index.tolist()
        mafs = g.sum(axis=1) / (2 * n) if freq_thresh > 0 else None
        # Filter MAF
        if freq_thresh > 0:
            print('Filtering MAFs smaller than', freq_thresh)
            print('    Genotype matrix shape before', g.shape)
            assert freq_thresh < 0.5
            good = (mafs < (1 - float(freq_thresh))) & (mafs > float(
                freq_thresh))
            with ProgressBar():
                with dask.config.set(pool=ThreadPool(threads)):
                    good, mafs = dask.compute(good, mafs, cache=cache)
            g = g[good, :]
            print('    Genotype matrix shape after', g.shape)
            bim = bim[good]
            bim['mafs'] = mafs[good]
            del good
            gc.collect()
        if normalize:
            mean = g.mean(axis=1)
            g = (g.T - mean) / g_std
        else:
            g = g.T
        return g, bim, fam

    @property
    def pheno(self):
        return self.__pheno

    @pheno.setter
    def pheno(self, pheno):
        if isinstance(pheno, str):
            opt = dict(delim_whitespace=True, header=None,
                       names=['fid', 'iid', 'pheno'])
            self.__pheno = pd.read_csv(pheno, **opt)
        elif isinstance(pheno,  pd.core.frame.DataFrame):
            self.__pheno = pheno
        else:
            options = dict(outprefix=self.outpref, bfile=self.geno, h2=0.5,
                           ncausal=10, normalize=True, uniform=False,
                           snps=None, seed=self.seed, bfile2=None,
                           max_memory=self.max_memory, bim=self.bim,
                           fam=self.fam, high_precision_on_zero=False)
            self.__pheno, h2, gen = qtraits_simulation(**options)
            g, b, self.truebeta, self.causals = gen

    @property
    def sum_stats(self):
        return self.__sum_stats

    @sum_stats.setter
    def sum_stats(self, sum_stats):
        if isinstance(sum_stats, str):
            self.__sum_stats = pd.read_csv(sum_stats, sep='\t')
        elif isinstance(sum_stats, pd.core.frame.DataFrame):
            self.__sum_stats = sum_stats
        else:
            raise NotImplementedError

    @property
    def memory(self):
        return self.__memory

    @memory.setter
    def memory(self, memory):
        if memory is not None:
            self.__memory = memory
        else:
            self.__memory = psutil.virtual_memory().available / 2
        self.cache = Chest(available_memory=self.__memory)

    @property
    def rho(self):
        return self.__rho

    @rho.setter
    def rho(self, ng):
        self.__rho = (da.dot(ng.T, ng) / ng.shape[0]) ** 2

    def get_clumps(self, ld_thr):
        # get clumps
        G_sparse = csr_matrix((self.rho >= ld_thr).compute().astype(int))
        n_comp, lab = connected_components(csgraph=G_sparse, directed=False,
                                           return_labels=True)
        clump = self.bim.copy(deep=True)
        clump['clumps'] = lab
        return clump

    def pval_thresholding(self, clump, pv_thr):
        gwas = self.sum_stats[self.sum_stats.pvalue <= pv_thr]
        gwas = gwas[~pd.isnull(gwas.slope)]
        merged = clump.merge(gwas, on=['snp', 'i'])
        merged.sort_values(by='pvalue', ascending=True, inplace=True)
        return merged.groupby('clumps').first()

    def score(self, geno, pheno, ld_thr, pv_thr):
        clump = self.get_clumps(ld_thr)
        index = self.pval_thresholding(clump, pv_thr)
        prs = geno[:, index.i.values].dot(index.slope)
        pheno = pheno.copy()
        pheno['prs'] = prs
        r2 = pheno.reindex(columns=['pheno', 'prs']).corr().loc[
                 'pheno', 'prs'] ** 2
        return pheno, index, ld_thr, pv_thr, r2

    def compute_prs(self):
        param_space = product(self.pval_range, self.ld_range)
        out = train_test_split(self.geno, self.pheno, test_size=1/self.cv)
        train_g, test_g, train_p, test_p = out
        delayed_results = [dask.delayed(self.score)(train_g, train_p, ld, pv)
                           for pv, ld in param_space]
        with ProgressBar():
            print('Computing PRS')
            result = list(dask.compute(*delayed_results, scheduler='threads'))
        best = sorted(result, key=lambda tup: tup[-1], reverse = True)[0]
        print('Best result achieved with LD prunning over %.2f and evlaue of '
              '%.2e, rendering an R2 of %.3f' % (best[2], best[3], best[4]))
        print('Index snps in training set:')
        print(best[1])
        print('Applyting to test set')
        with ProgressBar():
            actual_r2 = self.score(test_g, test_p, best[2], best[3])
            print('R2 in testset is', actual_r2[-1])
        actual_r2[0].to_csv('%s.prs' % self.outpref, sep='\t', index=False)
        actual_r2[1].to_csv('%s.indices' % self.outpref, sep='\t', index=False)
        self.best = best
        return actual_r2


def just_score(index_snp, sumstats, pheno, geno):
    clump = sumstats[sumstats.snp.isin(index_snp)]
    idx = clump.i.values.astype(int)
    boole = da.isnan(geno[:, idx]).any(axis=0)
    idx = idx[~boole]
    try:
        genclump = geno[:, idx]
    except ValueError:
        print(type(idx), idx.shape, geno.shape)
        print(idx)
        print(geno)
        raise
    aclump = clump[clump.i.isin(idx.tolist())]
    assert not np.isnan(aclump.slope).any()
    try:
        assert not da.isnan(genclump).any()
    except AssertionError:
        print(da.isnan(geneclump).sum())
    prs = genclump.dot(aclump.slope)
    assert not da.isnan(prs).any()
    assert not pd.isna(pheno.PHENO).any()
    est = np.corrcoef(prs, pheno.PHENO)[1, 0] ** 2
    if np.isnan(est):
        print(genclump[0:10,:])
        print(prs.compute(), pheno.PHENO)
        print(prs.shape, pheno.shape)
        print(pheno.columns)
        raise Exception
    return est


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='PROG', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('geno', help='Genotype file (bed filename)')
    parser.add_argument('prefix', help='prefix for outputs')
    parser.add_argument('-f', '--pheno', help='Phenotype file', default=None)
    parser.add_argument('-v', '--validate', default=None, type=int,
                        help='Use pseudo crossvalidation')
    parser.add_argument('-t', '--threads', default=1, type=int,
                        help='Number of cpus')
    parser.add_argument('-M', '--maxmem', default=None, type=int)
    parser.add_argument('--f_thr', type=float, default=0,
                        help='Keyword argument for read_geno. The frequency '
                              'threshold to cleanup the genotype file')
    parser.add_argument('--flip', action='store_true', default=False,
                        help='Keyword argument for read_geno. Whether to flip '
                             'the genotypes when MAF is > 0.5')
    parser.add_argument('--check', action='store_true', default=False,
                        help='Keyword argument for read_geno. Whether to '
                             'check the genotypes for invariant columns.')
    parser.add_argument('--normalize', action='store_true',
                        help='Keyword argument for qtraits_simulation. '
                             'Whether to normalize the genotype or not.')
    parser.add_argument('-s', '--sumstats', default=None,
                        help='Filename with summary statistics (previous gwas)'
                        )
    parser.add_argument('-p', '--pval_range', default=None,
                        help='Range of pvalues to explore')
    parser.add_argument('-r', '--ld_range', default=None, help='Range of R2 to'
                                                               ' explore')
    parser.add_argument('--cluster', default="local", help='local or slurm')
    parser.add_argument('--processes', default=None, help='Number of processes'
                                                          ' in a slurm cluster'
                        )
    parser.add_argument('--project', default=None, help='Name of your cluster'
                                                        ' account')
    parser.add_argument('--walltime', default="01:00:00", help='Time required')

    args = parser.parse_args()

    # client = set_cluster_type(args.cluster, processes=args.processes,
    #                           project=args.project, walltime=args.walltime)
    main(args.geno, args.pheno, args.prefix, args.sumstats, client=args.cluster,
         threads=args.threads,  memory=args.maxmem, validate=args.validate,
         freq_thresh=args.f_thr, check=args.check)