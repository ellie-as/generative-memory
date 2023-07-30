"""
The code in this file is based on https://github.com/ml-jku/hopfield-layers.
This repo accompanies Ramsauer et al. (2020) (https://arxiv.org/abs/2008.02217).
"""
import numpy as np
from itertools import product
from tqdm import tqdm
from scipy.special import softmax
import math


class DenseHopfield:
    def __init__(self, pat_size, beta=1, normalization_option=1):
        self.size = pat_size
        self.beta = beta
        self.max_norm = np.sqrt(self.size)
        if normalization_option == 0:
            self.energy = self.energy_unnormalized
        elif normalization_option == 1: # normalize dot product of patterns by 1/sqrt(pattern_size)
            self.energy = self.energy_normalized
        elif normalization_option == 2: # normalize dot product of patterns by shifting and clamping low exponentials
            self.energy = self.energy_normalized2
        else:
            raise ValueError('unkown option for normalization: %d'% normalization_option)

        return

    def learn(self, patterns):
        """expects patterns as numpy arrays and stores them col-wise in pattern matrix 
        """
        self.num_pat = len(patterns)
        assert(all(type(x) is np.ndarray for x in patterns)), 'not all input patterns are numpy arrays'
        assert(all(len(x.shape) == 2 for x in patterns)), 'not all input patterns have dimension 2'
        assert(all(1 == x.shape[1] for x in patterns)), 'not all input patterns have shape (-1,1) '
        self.patterns = np.array(patterns).squeeze(axis=-1).T # save patterns col-wise
        # without squeeze axis would result in problem with one pattern
        self.max_pat_norm = max(np.linalg.norm(x) for x in patterns)

    def retrieve(self, partial_pattern, max_iter=np.inf, thresh=0.5):
        # partial patterns have to be provided with None/0 at empty spots
        if partial_pattern.size != self.size:
            raise ValueError("Input pattern %r does not match state size: %d vs %d" 
                %(partial_pattern, len(partial_pattern), self.size))
        
        if None in partial_pattern:
            raise NotImplementedError("None elements not supported")

        assert type(partial_pattern) == np.ndarray, 'test pattern was no numpy array'
        assert len(partial_pattern.shape) <=2 and 1 == partial_pattern.shape[1], 'test pattern with shape %r is not a col-vector' %(partial_pattern.shape,)

        pat_old = partial_pattern.copy()
        iters = 0

        for iters in tqdm(range(max_iter)):
            pat_new = np.zeros(partial_pattern.shape)
            # jj = np.random.randint(self.size)
            for jj in range(self.size):
                # simple variant:
                E = 0
                temp = pat_old[jj].copy()
                pat_old[jj] = +1
                E -= self.energy(pat_old)
                pat_old[jj] = -1
                E += self.energy(pat_old)

                pat_old[jj] = temp
                pat_new[jj] = np.where(E >0 , 1, -1)
            
            if np.count_nonzero(pat_old != pat_new)<= thresh:
                break
            else:
                pat_old = pat_new
            
        return pat_new

    @staticmethod
    def _lse(z, beta):
        return 1/beta * np.log(np.sum(np.exp(beta*z)))

    def energy_unnormalized(self, pattern):
        # return -1*np.exp(self._lse(self.patterns.T @pattern, beta=self.beta))
        # this is equal, but faster
        return -1*np.sum(np.exp(self.patterns.T @pattern ))
    
    def energy_normalized(self, pattern):
        # normalize dot product of patterns by 1/sqrt(pattern_size)
        return -1*np.sum(np.exp((self.patterns.T @pattern)/self.max_norm ))
    
    def energy_normalized2(self, pattern):
        # normalize dot product of patterns by shifting by -sqrt(pattern_size)
        # also clamp exponential for exponents smaller then -73 to 0 
        exponents = self.patterns.T @pattern
        norm_exponents = exponents - self.max_pat_norm
        norm_exponents[norm_exponents < -73] = -np.inf

        return -1*np.sum(np.exp(norm_exponents))

    
    def energy_landscape(self):
        for pat in product([1,-1], repeat=self.size):
            pat = np.array(pat)
            print("energy(%r)=%.3f"%(pat, self.energy(pat)))

class ContinuousHopfield:
    def __init__(self, pat_size, beta=1, do_normalization=True):
        self.size = pat_size # size of individual pattern
        self.beta = beta
        print(self.beta)
        self.max_norm = np.sqrt(self.size)
        if do_normalization:
            self.softmax = self.softmax_normalized
            self.energy = self.energy_normalized
        else:
            self.softmax = self.softmax_unnormalized
            self.energy = self.energy_unnormalized
        
        return

    def learn(self, patterns):
        """expects patterns as numpy arrays and stores them col-wise in pattern matrix 
        """
        self.num_pat = len(patterns)
        assert(all(type(x) is np.ndarray for x in patterns)), 'not all input patterns are numpy arrays'
        assert(all(len(x.shape) == 2 for x in patterns)), 'not all input patterns have dimension 2'
        assert(all(1 == x.shape[1] for x in patterns)), 'not all input patterns have shape (-1,1) '
        self.patterns = np.array(patterns).squeeze(axis=-1).T # save patterns col-wise
        # without squeeze axis would result in problem with one pattern
        # return -1*np.sum(np.exp([(self.patterns[:,ii].T @pattern)/self.max_norm for ii in range(self.patterns.shape[1])]))
        self.M = max(np.linalg.norm(vec) for vec in patterns)# maximal norm of actually stored patterns
        return

    def retrieve(self, partial_pattern, max_iter=np.inf, thresh=0.5):
        # partial patterns have to be provided with None/0 at empty spots
        if partial_pattern.size != self.size:
            raise ValueError("Input pattern %r does not match state size: %d vs %d" 
                %(partial_pattern, len(partial_pattern), self.size))
        
        if None in partial_pattern:
            raise NotImplementedError("None elements not supported")

        assert type(partial_pattern) == np.ndarray, 'test pattern was not numpy array'
        assert len(partial_pattern.shape) ==2 and 1 == partial_pattern.shape[1], 'test pattern with shape %r is not a col-vector' %(partial_pattern.shape,)

        pat_old = partial_pattern.copy()
        iters = 0

        while iters < max_iter:
            pat_new = self.patterns @ self.softmax(self.beta*self.patterns.T @ pat_old)

            if np.count_nonzero(pat_old != pat_new)<= thresh: # converged
                break
            else:
                pat_old = pat_new
            iters += 1
            
        return pat_new

    def softmax_unnormalized(self, z):
        return softmax(z)  # Scipy's softmax is numerically stable

    def softmax_normalized(self, z):
        return softmax(z / self.max_norm)

    @staticmethod
    def _lse(z, beta):
        return 1/beta * np.log(np.sum(np.exp(beta*z)))

    def energy_unnormalized(self, pattern):
        return -1*self._lse(self.patterns.T @pattern, 1) + 0.5 * pattern.T @ pattern\
            + 1/self.beta*np.log(self.num_pat)\
            + 0.5*self.M**2
    
    def energy_normalized(self, pattern):
        # normalize dot product of patterns by 1/sqrt(pattern_size)
        return -1*self._lse((self.patterns.T @pattern )/self.max_norm, 1) + 0.5 * pattern.T @ pattern\
            + 1/self.beta*np.log(self.num_pat)\
            + 0.5*self.M**2

    def energy_landscape(self):
        for pat in product([1,-1], repeat=self.size):
            pat = np.array(pat)
            print("energy(%r)=%.3f"%(pat, self.energy(pat)))