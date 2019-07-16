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
import os
import gc
import h5py
import time
import dask
import dill
import psutil
import operator
import argparse
import warnings
import matplotlib
import numpy as np
import pandas as pd
import mpmath as mp
from numba import jit
from tqdm import tqdm
from chest import Chest
import dask.array as da
import dask.dataframe as dd
from igraph import Graph
from copy import deepcopy
import dask.dataframe as dd
from functools import reduce
from dask.array.core import Array
from collections import namedtuple
from scipy.stats import linregress
from pandas_plink import read_plink
from joblib import Parallel, delayed
from multiprocessing import cpu_count
#import statsmodels.formula.api as smf
from sklearn.decomposition import PCA
from dask_jobqueue import SLURMCluster
from dask.diagnostics import ProgressBar
from itertools import cycle, product, chain
from qtraitsimulation import qtraits_simulation
from multiprocessing.pool import ThreadPool, Pool
from sklearn.model_selection import train_test_split
from dask.distributed import Client, LocalCluster, as_completed, progress
from dask_ml.linear_model import LinearRegression


mp.dps = 25
mp.pretty = True
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

lr = jit(linregress, nopython=True)

# #@jit
# def linregress(tuple_param):#, pbar):
#     x, y = tuple_param
#     linregress_result = linregress(x, y)
#     #pbar.update()
#     return linregress_result


def single_locus_clump(clump, sum_stats):
    tag = clump.vs['label']
    sub_stats = sum_stats[sum_stats.snp.isin(tag)]
    index = sub_stats.nsmallest(1, 'pvalue')
    try:
        key = (index.snp.values[0], index.pvalue.values[0])
    except IndexError:
        print(tag)
        print(sum_stats)
        print(sub_stats)
        raise

    return key, sub_stats


def clumps(sum_stats, locus, ld_thr):
    """
    Get clumps from locus helper function to multiprocess
    :param sum_stats: subset of the summary statistics for the locus
    :param ld_threshold: the threshold for this run
    """
    snp_list, d = locus
    boole = snp_list.isin(sum_stats.snp)
    # Name the rows and columns
    snp_list = snp_list[boole].to_list()
    d2 = d[boole, :]
    d2 = d2[:, boole]
    try:
        gr = Graph.Adjacency((d2 ** 2 > ld_thr).tolist())
    except TypeError:
        print(snp_list)
        print(d.shape, d2.shape)
        raise
    gr.vs['label'] = snp_list
    grs = gr.components().subgraphs()
    clumped = [single_locus_clump(clump, sum_stats) for clump in grs]
    return ld_thr, dict(clumped)


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


def get_index(parameter_tuple):
    all_clumps, p_thr, sum_stats, train_p, train_g = parameter_tuple
    space = []
    for ld_thr in all_clumps.keys():
        clumped = all_clumps[ld_thr]
        index_snps = [k[0] for k in clumped.keys() if k[1] < p_thr]
        if not index_snps:
            r2 = 0
        else:
            try:
                r2 = just_score(index_snps, sum_stats, train_p, train_g)
            except Exception:
                with open('failed.pckl', 'wb') as F:
                    dill.dump((index_snps, sum_stats, train_p, train_g), F)
                    raise
        space.append((ld_thr, p_thr, index_snps, r2, pd.concat(clumped.values()
                                                               )))
    return space


class PRS(object):
    def __init__(self, geno, sum_stats, kbwindow=1000, ld_range=None,
                 pval_range=None, extend=False, check=True, memory=None,
                 threads=1, re_normalize=True, client=None):
        self.cache = None
        self.memory = memory
        self.threads = threads
        self.client = client
        self.extend = extend
        self.check = check
        self.bim = None
        self.fam = None
        self.geno = geno
        self.kbwindow = kbwindow
        self.sum_stats = sum_stats
        self.loci = None
        self.all_clumps = {}
        self.ld_range = ld_range
        self.re_normalize = re_normalize
        self.best = None
        self.pval_range = pval_range
        self.train_p = None
        self.test_p = None
        self.train_g = None
        self.test_g = None

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
            assert isinstance(a, np.ndarray)

    @property
    def geno(self):
        return self.__geno

    @geno.setter
    def geno(self, geno):
        if isinstance(geno, str):
            op = dict(check=self.check, max_memory=self.max_memory)
            out = read_geno(bed, self.freq_thresh, self.threads, **op)
            self.bim, self.fam, self.__geno = out
        elif isinstance(geno, tuple):
            self.bim, self.fam, self.__geno = geno
        else:
            assert isinstance(geno, da)
            self.__geno = geno

    @property
    def memory(self):
        return self.__memory

    @memory.setter
    def memory(self, memory):
        if memory is not None:
            self.__memory = max_memory
        else:
            self.__memory = psutil.virtual_memory().available / 2
        self.cache = Chest(available_memory=self.__memory)

    def single_window(self, g, subbim):
        """
        Helper function to compute the correlation between variants from a
        genotype array
        :param subbim: Subset of bim dataframe
        """
        if not subbim.empty:
            # set Cache to protect memory spilling

            # Make sure chunks make sense
            chunk_opts = dict(threads=self.threads, memory=self.memory)
            if not isinstance(g, np.ndarray):
                g = g.rechunk(estimate_chunks(shape=g.shape, **chunk_opts))
            # extend the genotype at both end to avoid edge effects
            if self.extend:
                # get the indices of the subset genotype array
                nidx = np.arange(g.shape[1])
                # Split the array in half (approximately)
                idx_a, idx_b = np.array_split(nidx, 2)
                # Get the extednded indices
                i = np.concatenate([idx_a[::-1][:-1], nidx, idx_b[::-1][1:]])
                # Re-subset the genotype arrays with the extensions
                g = g[:, i]
                # normalize for correlation computation
                g = (g - g.mean(axis=0)) / g.std(axis=0)
                # Compute the correlation as X'X/N
                rho = da.dot(g.T, g) / g.shape[0]
                # remove the extras
                idx = np.arange(i.shape[0])[
                      idx_a.shape[0] - 1: (nidx.shape[0] + idx_b.shape[0])]
                rho = rho[idx, :]
                rho = rho[:, idx]
            else:
                g = (g - g.mean(axis=0)) / g.std(axis=0)
                # Just compute the correlations
                rho = da.dot(g.T, g) / g.shape[0]
            gc.collect()
            return subbim.snp, rho

    def window_yielder(self, geno):
        """
        Iterator over snps windows
        """
        for window, subbim in self.bim.groupby('windows'):
            if not subbim.snp.empty:
                # Get the mapping indices
                idx = subbim.i.values
                # Subset the genotype arrays
                g = geno[:, idx]
                yield dask.delayed(g), dask.delayed(subbim)
    @property
    def loci(self):
        return self.__loci

    @loci.setter
    def loci(self, _):
        """
        Get the LD blocks in one population
        """
        # set Cache to protect memory spilling
        rp = 'r.pckl'
        if os.path.isfile(rp):
            with open(rp, 'rb') as pckl:
                r = dill.load(pckl)
        else:
            if os.path.isfile('ld.matrix'):
                print('Loading precomputed LD matrix')
                r = dd.read_parquet('ld.matrix')
            else:
                print('Computing LD score per window')
                # Get the number of bins or loci to be computed
                nbins = np.ceil(max(self.bim.pos)/(self.kbwindow * 1000)
                                ).astype(int)
                # Get the limits of the loci
                bins = np.linspace(0, max(self.bim.pos) + 1, num=nbins,
                                   endpoint=True, dtype=int)
                if bins.shape[0] == 1:
                    # Fix the special case in which the window is much bigger 
                    # than the range
                    bins = np.append(bins, self.kbwindow * 1000)
                # Get the proper intervals into the dataframe
                self.bim['windows'] = pd.cut(self.bim['pos'], bins, 
                                             include_lowest=True)
                # Compute each locus in parallel
                dask_geno = dask.delayed(self.geno)
                delayed_results = [dask.delayed(self.single_window)(g, df) for
                                   g, df in self.window_yielder(dask_geno)]
                opts = dict(num_workers=self.threads, cache=self.cache,
                            pool=ThreadPool(self.threads))
                with ProgressBar(), dask.config.set(**opts), open(rp,
                                                                  'wb') as pck:
                    r = tuple(dask.compute(*delayed_results))
                    dill.dump(r, pck)
        r = tuple(x for x in r if x is not None)
        self.__loci = r
        # r = pd.concat(r)
        # dd.to_parquet(r, 'ld.matrix')
        # self.loci = r

    def clumps(self, tuple_param, pbar):
        """
        Get clumps from locus
        :param sum_stats: subset of the summary statistics for the locus
        :param ld_threshold: the threshold for this run
        """
        all_clumps = {}
        locus, ld_thr = tuple_param
        pbar.set_description(desc="Clumping with %f LD threshold" % ld_thr)
        pbar.refresh()
        ascend = True
        # unpack the locus tuple
        snp_list, d = locus
        # Name the rows and columns
        snp_list = snp_list.to_list()
        with dask.config.set(num_workers=self.threads):
            nd = (d ** 2 > ld_thr).compute()
        gr = Graph.Adjacency(nd.tolist())
        gr.vs['label'] = snp_list
        # d = pd.DataFrame(d.compute(), index=snp_list, columns=snp_list)
        # subset sum_stats
        # Get the clumps pfr this locus
        grs = gr.components().subgraphs()
        for clump in grs:
            tag = clump.vs['label']
            sub_stats = self.sum_stats[self.sum_stats.snp.isin(tag)]
            index = sub_stats.nsmallest(1, 'pvalue')
            key = (index.snp.values[0], index.pvalue.values[0])
            all_clumps[key] = sub_stats
            pbar.update()
            #total = len(self.ld_range) * len(self.loci)
        # while not sum_stats.empty:
        #     # get the index snp
        #     index = sum_stats.nsmallest(1, 'pvalue')
        #     # get the clump around index for
        #     vec = (d ** 2).loc[index.snp, :]
        #     tag = vec[vec > ld_threshold].index.tolist()
        #     # Subset the sumary statistic dataframe with the snps in the clump
        #     sub_stats = sum_stats[sum_stats.snp.isin(tag)]
        #     key = (index.snp.values[0], index.pvalue.values[0])
        #     all_clumps[key] = sub_stats
        #     # remove the clumped snps from the summary statistics dataframe
        #     sum_stats = sum_stats[~sum_stats.snp.isin(tag)]

        return ld_thr, all_clumps

    def compute_clumps(self, ld_threshold):
        """

        :param loci: list of tuples with the LDs and snps per locus
        :param sum_stats: sumary statistics
        :param ld_threshold: trheshold for cumping
        :param h2: heritability of the trait
        :param avh2: average heritability
        :param n: number of samples
        :param threads: number of threads to use in multiprocessing
        :param cache: chest dictionary to avoid memory overflow
        :param memory: max memory to use
        :return: dictionary with the clumps
        """

        delayed_results = [dask.delayed(self.clumps)(locus, ld_threshold)
                           for locus in self.loci]
        pbar = ProgressBar()
        config = dask.config.set(num_workers=self.threads,
                                 memory_limit=self.memory, cache=self.cache,
                                 pool=ThreadPool(self.threads))
        with pbar, config:
            l = list(dask.compute(*delayed_results))
        return dict(pair for d in l for pair in d.items())

    def get_index(self, parameter_tuple):
        all_clumps, by_threshold = parameter_tuple
        rank = operator.lt
        index_snps = [k[0] for k in all_clumps.keys() if rank(k[0] + 1,
                                                              by_threshold)]
        if not index_snps:
            r2 = 0
        else:
            try:
                r2 = just_score(index_snps, self.sum_stats, self.train_p,
                                self.train_g)
            except Exception:
                with open('failed.pckl', 'wb') as F:
                    dill.dump((index_snps, self.sum_stats, self.train_p,
                                 self.train_g), F)
                    raise
        # pbar.update()
        return (index_snps, r2, pd.concat(all_clumps.values()))

    def optimize_it(self, test_geno, test_pheno):
        """
        Optimize the R2 based on summary statistics
        """
        # Optimize with one split, return reference score with the second
        out = train_test_split(test_geno, test_pheno, test_size=0.5)
        train_g, test_g, train_p, test_p = out
        if self.re_normalize:
            # re-normalize the genotypes
            std_train = train_g.std(axis=0).compute()
            std_test = test_g.std(axis=0).compute()
            boole = (std_train != 0) & (std_test !=0)
            train_g = train_g[:, boole]
            train_g = (train_g - train_g.mean(axis=0)) /std_train[boole]
            test_g = test_g[:, boole]
            test_g = (test_g - test_g.mean(axis=0)) / std_test[boole]
            sumstats = self.sum_stats[boole]
        else:
            sumstats = self.sum_stats
        self.train_g, self.test_g = train_g, test_g
        self.train_p, self.test_p = train_p, test_p
        assert not da.isnan(self.train_g).any().compute()
        try:
            assert (sumstats.shape[0] == self.train_g.shape[1])
        except AssertionError:
            print(sumstats.shape, self.train_g.shape)
            print((sumstats.shape[0] == self.train_g.shape[1]))
            raise
        newbool = ~np.isnan(sumstats.slope).values
        sumstats = sumstats[newbool]
        self.train_g = self.train_g[:, newbool]
        self.test_g = self.test_g[:, newbool]
        sumstats['i'] = range(sumstats.shape[0])
        # combos = product(self.loci, self.ld_range)
        combos = product(self.loci, self.ld_range)
        delayed_results = [
            dask.delayed(clumps)(sumstats[sumstats.snp.isin(locus[0])], locus,
                                 ld_thr) for locus, ld_thr in combos]
        opts = dict(num_workers=self.threads, cache=self.cache,
                    scheduler='threads')
        print('\tGet clumps')
        pcklfile = 'clumps.pckl'
        if os.path.isfile(pcklfile):
            with open(pcklfile, 'rb') as p:
                clumped = dill.load(p)
        else:
            with ProgressBar(), dask.config.set(**opts), open(pcklfile,
                                                              'wb') as p:
                clumped = list(dask.compute(*delayed_results))
                dill.dump(clumped, p)
        del combos, delayed_results
        gc.collect()
        clumped = dict(clumped)
        combos = product([clumped], self.pval_range, [sumstats],
                         [self.train_p], [self.train_g])
        delayed_results = [dask.delayed(get_index)(param_tuple)
                           for param_tuple in combos]
        print('\tGet index and score')
        with ProgressBar(), dask.config.set(**opts):
            results = list(dask.compute(*delayed_results))
        results = list(chain.from_iterable(results))
        del combos
        gc.collect()
        res = sorted(results, key=lambda x: x[3], reverse=True)
        with open('allresults.pckl', 'wb') as p:
            dill.dump(res, p)
        curr_best = res[0][2:]
        r2 = just_score(curr_best[0], sumstats, test_p, test_g)
        best = namedtuple('best', ('indices', 'r2', 'clumps', 'ld', 'pval'))
        self.best = best(curr_best[0], r2, curr_best[-1], res[0][0], res[0][1])


# ----------------------------------------------------------------------
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
    usage = estimate_size(shape) * threads     # Compute threaded estimated size
    # Determine number of chunks given usage and available memory
    n_chunks = np.ceil(usage / avail_mem).astype(int)
    # Mute divided by zero error only for this block of code
    with np.errstate(divide='ignore', invalid='ignore'):
        estimated = tuple(np.array(shape) / n_chunks)  # Get chunk estimation
    chunks = min(shape, tuple(estimated))            # Fix if n_chunks is 0
    return tuple(int(i) for i in chunks)  # Assure chunks is a tuple of integers


# ----------------------------------------------------------------------
def estimate_size(shape):
    """
    Estimate the potential size of an array
    :param shape: shape of the resulting array
    :return: size in Mb
    """
    total_bytes = reduce(np.multiply, shape) * 8
    return total_bytes / 1E6


# -----------------------------------------------------------------------------
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
    (bim, fam, g) = read_plink(bfile)   # read the files using pandas_plink
    m, n = g.shape                      # get the dimensions of the genotype
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
        good = (mafs < (1 - float(freq_thresh))) & (mafs > float(freq_thresh))
        with ProgressBar():
            with dask.config.set(pool=ThreadPool(threads)):
                good, mafs = dask.compute(good, mafs, cache=cache)
        g = g[:, good]
        print('    Genotype matrix shape after', g.shape)
        bim = bim[good]
        bim['mafs'] = mafs[good]
        del good
        gc.collect()
    bim = bim.reset_index(drop=True)    # Get the indices in order
    # Fix the i such that it matches the genotype indices
    bim['i'] = bim.index.tolist()
    # Get chunks apropriate with the number of threads
    g = g.rechunk(estimate_chunks(g.shape, threads, memory=available_memory))
    del mafs
    gc.collect()
    return bim, fam, g


class GWAS(object):
    def __init__(self, filesetprefix, pheno, outprefix, threads, client=None,
                 check=False, freq_thresh=0.01, flip=False, max_memory=None,
                 seed=None, usable_snps=None, **kwargs):
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
        print(self.__dict__)

    @property
    def geno(self):
        return self.__geno

    @geno.setter
    def geno(self, filesetprefix):
        bed = '%s.bed' % filesetprefix
        op = dict(check=self.check, usable_snps=self.usable_snps,
                  max_memory=self.max_memory)
        (bim, fam, geno) = read_geno(bed, self.freq_thresh, self.threads, **op)
        if self.bed:
            self.bed.append(bed)
            self.bim = pd.concat((self.bim, bim), axis=0).reset_index(drop=True
                                                                      )
            pd.testing.assert_frame_equal(self.fam, fam)
            self.__geno = da.concatenate((self.geno, geno), axis=1)
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

        elif isinstance(self.pheno, str):
            # If pheno is provided as a string, read it
            pheno = pd.read_csv(self.pheno, delim_whitespace=True,
                                  header=None, names=['fid', 'iid', 'PHENO'])
        else:
            pheno = self.pheno
        try:
            y = pheno.compute(num_workers=threads, cache=self.cache)
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

    # @staticmethod
    # @jit(nopython=True, parallel=True)
    # def logistic_regression(Y, X, w, iterations):
    #     for i in range(iterations):
    #         w -= np.dot(((1.0 /(1.0 + np.exp(-Y * np.dot(X, w))) - 1.0) * Y), X
    #                     )
    #     return w

    @staticmethod
    #@jit(nopython=True, parallel=True)
    def nu_linregress(param_tuple, pbar):
        """
        Refactor of the scipy linregress with mpmath in the estimation of the
        pvalue, numba, and less checks for speed sake

        :param x: array for independent variable
        :param y: array for the dependent variable
        :return: dictionary with slope, intercept, r, pvalue and stderr
        """
        x, y = param_tuple
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
        pbar.update()
        return dict(zip(cols, [slope, intercept, r, prob, sterrest]))

    @staticmethod
    #@jit(nopython=True, parallel=True)
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
    #@jit(parallel=True)
    def st_mod(param_tuple, pbar):
        """
        Linear regression using stats models. This module is very slow but
        allows to include covariates in the estimation.

        :param x: array for independent variable
        :param y: array for dependent variable
        :param covs: array for covariates
        :return: Regression results
        """
        if len(param_tuple) == 2:
            x, y = param_tuple
            covs = None
        else:
            x, y, covs = param_tuple
        df = pd.DataFrame({'geno': x, 'pheno': y})
        cols = ['slope', 'intercept', 'rvalue', 'pvalue', 'stderr', 'b_pval',
                'b_std_err']
        if np.allclose(x.var(), 0.0):
            linregress_result = dict(zip(cols, cycle([np.nan])))
        else:
            if covs is not None:
                c = []
                for col in range(covs.shape[1]):
                    df['Cov%d' % col] = covs[:, col]
                    c.append('Cov%d' % col)
                formula = 'pheno ~ geno + %s' % ' + '.join(c)
            else:
                formula = 'pheno ~ geno'
            model = smf.ols(formula=formula, data=df)
            results = model.fit()
            vals = [results.params.Intercept, results.params.geno,
                    results.pvalues.Intercept, results.pvalues.geno,
                    results.rsquared, results.bse.Intercept, results.bse.geno]
            linregress_result = dict(zip(cols, vals))
        pbar.update()
        return linregress_result

    @staticmethod
    def linregress(x, y):#param_tuple):
        #x, y = param_tuple
        linregress_result = linregress(x, y)
        return linregress_result

    @staticmethod
    #@jit(parallel=True)
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
        chunks = np.load('chunks.npy')  # Load the chunks stored
        # Estimate chunk sizes given the number of threads
        #chunks = [estimate_chunks(tuple(i), self.threads) for i in chunks]
        # Get training set of the genotype array
        #x_train = da.from_array(f.get('x_train'), chunks=tuple(chunks[0]))
        x_train = da.from_array(f.get('x_train'))
        x_train.rechunk((x_train.shape[0], 1))
        # Get the test set of the genotype array
        x_test = da.from_array(f.get('x_test'))
        x_test.rechunk((x_test.shape[0], 1))
        # x_test = da.from_array(f.get('x_test'), chunks=tuple(chunks[1]))
        # Get the training set of the phenotype
        y_train = pd.read_hdf(pfn, key='y_train')
        # Get the testing set of the phenotype
        y_test = pd.read_hdf(pfn, key='y_test')
        # Read the resulting gwas table
        res = pd.read_csv('%s.gwas' % self.outpref, sep='\t')
        return res, x_train, x_test, y_train, y_test
    
    def plink_free_gwas(self, validate=None, plot=False, causal_pos=None,
                        threads=8, pca=None, stmd=False,
                        high_precision=False, max_memory=None,
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
                #Perform PCA
                func = st_mod                   # Force function to statsmodels
                covs = self.do_pca(x_train, pca)     # Estimate PCAs
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


def do_gwas(geno, pheno, sum_stats, outprefix, threads, max_memory, validate=2,
            **kwargs):
    if sum_stats is not None:
        sum_stats = pd.read_csv(sum_stats, delim_whitespace=True)
        gwas = namedtuple(gwas, ('fam', 'bim', 'geno', 'sum_stats', 'x_train',
                                 'x_test', 'y_train', 'y_test'))
        gwas = gwas(kwargs['fam'], kwargs['bim'], geno, sum_stats, None, None,
                    None, None)
    else:
        gwas = GWAS(geno, pheno, outprefix, threads, max_memory, **kwargs)
        gwas.plink_free_gwas(validate=validate, **kwargs)

    return gwas


def set_cluster_type(cluster, **kwargs):
    if cluster.lower() == 'local':
        cluster = LocalCluster()
    else:
        cluster = SLURMCluster(**kwargs)
    client = Client(cluster)
    return client


def main(geno, pheno, outprefix, sum_stats=None, max_memory=None, threads=1,
         pval_range=None, ld_range=None, client=None, **kwargs):
    print('Performing P + T')
    if isinstance(geno, str):
        if geno.endswith('.bed'):
            geno = geno[: geno.find('.bed')]
    if pheno is not None:
        pheno = pd.read_csv(pheno, blocksize=25e6, delim_whitespace=True)
    if isinstance(geno, list):
        if geno[0].endswith('.bed'):
            geno = [g[: g.find('.bed')] for g in geno]
        gw = [do_gwas(g, pheno, sum_stats, outprefix, threads, max_memory,
                      **kwargs) for g in geno]
        bim, fam, geno, sum_stats = zip(*gw)
        sum_stats = pd.concat(sum_stats)
        x_test, y_test = geno, pheno
    else:
        gwas = do_gwas(geno, pheno, sum_stats, outprefix, threads, max_memory,
                       **kwargs)
        bim, fam, geno = gwas.bim, gwas.fam, gwas.geno
        sum_stats = gwas.sum_stats
        x_test, y_test = gwas.x_test, gwas.y_test
    prs = PRS((bim, fam, geno), sum_stats, threads=threads, memory=max_memory,
              pval_range=pval_range, ld_range=ld_range, client=client)
    prs.optimize_it(x_test, y_test)
    print(prs.best)


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