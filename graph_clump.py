import sys
from itertools import product

import argparse
import dask
import dask.array as da
import gc
import numpy as np
import pandas as pd
import psutil
from chest import Chest
from dask.diagnostics import ProgressBar
from multiprocessing.pool import ThreadPool
from pandas_plink import read_plink
from qtraitsimulation import qtraits_simulation
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import train_test_split


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


class PRS(object):
    def __init__(self, bedfileset, sum_stats, pheno=None, ld_range=None,
                 pval_range=None, check=True, memory=None, threads=1,
                 snp_list=None, outpref='prs', cv=3, freq_thresh=0.1,
                 normalize=True):
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
            assert isinstance(a, np.ndarray)

    @property
    def geno(self):
        return self.__geno

    @geno.setter
    def geno(self, geno):
        if isinstance(geno, str):
            op = dict(check=self.check, max_memory=self.memory,
                      normalize=self.normalize)
            out = self.read_geno(geno, self.freq_thresh, self.threads, **op)
            self.__geno, self.bim, self.fam = out
        elif isinstance(geno, tuple):
            self.bim, self.fam, self.__geno = geno
        else:
            assert isinstance(geno, da)
            self.__geno = geno

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
            opt = dict(delim_whitespace=True,header=None,
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
            self.truebeta = truebeta
            self.causals = vec

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
            self.__memory = max_memory
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


def compute_PRS(rho, pheno, gwas, ld_thr, pv_thr):
    # get clumps
    G_sparse = csr_matrix((rho >= ld_thr).compute().astype(int))
    n_comp, lab = connected_components(csgraph=G_sparse, directed=False,
                                       return_labels=True)
    bim = self.bim.copy(deep=True)
    bim['clumps'] = lab
    # merge bim and filtered gwas
    gwas = gwas[gwas.pvalue <= pv_thr]
    merged = bim.merge(gwas, on=['snp', 'i'])
    merged.sort_values(by='pvalue', ascending=True, inplace=True)
    index = merged.groupby('clumps').first()
    # COMPUTE PRS
    prs = ng[:,index.i.values].dot(index.slope).compute()
    pheno['prs'] = prs
    r2 = pheno.reindex(columns=['pheno', 'prs']).corr().loc['pheno', 'prs']**2
    print(r2)


def main(geno, pheno, outpref, gwas, pval_range, ld_range, threads, memory,
         validate, freq_thresh, check, snp_subset):
        p = PRS(geno, gwas, pheno=pheno, ld_range=ld_range, check=check,
                pval_range=pval_range, memory=memory, threads=threads,
                snp_list=snp_subset, outpref=outpref, cv=validate,
                freq_thresh=freq_thresh)
        p.compute_prs()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='PROG', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('geno', help='Genotype file (bed filename)')
    parser.add_argument('prefix', help='prefix for outputs')
    parser.add_argument('-f', '--pheno', help='Phenotype file', default=None)
    parser.add_argument('-v', '--validate', default=None, type=int,
                        help='Use pseudo crossvalidation with this number of '
                             'folds')
    parser.add_argument('-t', '--threads', default=1, type=int,
                        help='Number of cpus')
    parser.add_argument('-M', '--maxmem', default=None, type=int)
    parser.add_argument('--f_thr', type=float, default=0,
                        help='Keyword argument for read_geno. The frequency '
                              'threshold to cleanup the genotype file')
    parser.add_argument('--check', action='store_false', default=True,
                        help='Disable checking the genotypes for invariant '
                             'columns.')
    parser.add_argument('--normalize', action='store_false',
                        help='Disable genotype normalization')
    parser.add_argument('-s', '--sumstats', default=None,
                        help='Filename with summary statistics (previous gwas)'
                        )
    parser.add_argument('-p', '--pval_range', default=None,
                        help='Range of pvalues to explore')
    parser.add_argument('-r', '--ld_range', default=None, help='Range of R2 to'
                                                               ' explore')
    parser.add_argument('-S', '--snp_subset', default=None,
                        help='subset of SNPs to analyse')
    args = parser.parse_args()
    main(args.geno, args.pheno, args.prefix, args.sumstats, args.pval_range,
         args.ld_range, check=args.check, threads=args.threads,
         memory=args.maxmem, validate=args.validate, freq_thresh=args.f_thr,
         snp_subset=args.snp_subset)