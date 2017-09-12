# Author: Brooke Husic <brookehusic@gmail.com>
# Contributors: 
# Copyright (c) 2017, Stanford University and the Authors
# All rights reserved.

from __future__ import print_function, division, absolute_import

import numpy as np
from msmbuilder.msm import MarkovStateModel
import scipy.spatial.distance


class BACE(MarkovStateModel):
    """Bayesian agglomerative clustering engine (BACE) for coarse-graining
    (lumping) microstates into macrostates.

    BACE is implemented with defaults chosen to produce results consistent with
    the MSMBuilder2 code written by Greg Bowman. The BACE method was introduced
    in Bowman, J. Chem. Phys. 2012, dx.doi.org/10.1063/1.4755751.

    Right now, our implementaton of BACE does not account for statistically
    undersampled states by merging them with the kinetically closest neighbor,
    which IS available in the MSMBuilder2 code.

    To accommodate MSMs created with sliding windows, the counts matrix
    is multiplied by the lag time. This may not be the mathematically best
    approach.

    Parameters
    ----------
    n_macrostates : int (default : 2)
    kwargs : optional
        Additional keyword arguments to be passed to MarkovStateModel.  See
        msmbuilder.msm.MarkovStateModel for possible options.

    Attributes
    ----------
    microstate_mapping_ : np.array, [number of microstates]
    bayes_factors_ : np.array, [number microstates - number macrostates + 1]

    Notes
    -----
    BACE is a subclass of MarkovStateModel.  However, the MSM properties
    and attributes on BACE refer to the MICROSTATE properties--e.g.
    bace.transmat_ is the microstate transition matrix.  To get the
    macrostate transition matrix, you must fit a new MarkovStateModel
    object on the output (assignments) of BACE().
    """

    def __init__(self, n_macrostates=2, pseudocount=None, **kwargs):
        self.n_macrostates = n_macrostates
        self.pseudocount = pseudocount
        super(BACE, self).__init__(**kwargs)

    def fit(self, sequences, y=None):
        """Fit a BACE lumping model using a sequence of cluster assignments.

        Parameters
        ----------
        sequences : list(np.ndarray(dtype='int'))
            List of arrays of cluster assignments
        y : None
            Unused, present for sklearn compatibility only.

        Returns
        -------
        self
        """
        super(BACE, self).fit(sequences, y=y)
        self._do_lumping()

        return self

    def _distance_helper(self, w_i, w_j, c_i, c_j):
        print(w_i, w_j)
        p_i = c_i/w_i
        p_j = c_j/w_j
        cp = (c_i + c_j) / (w_i + w_j)
        return c_i.dot(np.log(p_i/cp)) + c_j.dot(np.log(p_j/cp))

    def _get_bayes_factor(self, i, j):
        cmat = np.copy(self.cmat_)

        # Unconnected states are infinitely distant
        if cmat[i,j] == 0:
            if cmat[j,i] == 0:
                return np.inf, None
 
        cmat += self.pseudocount
        w = np.sum(cmat, axis=1)

        w_i = w[i]
        w_j = w[j]
        c_i = cmat[i]
        c_j = cmat[j]

        d = self._distance_helper(w_i, w_j, c_i, c_j)
    
        return d, w

    def _get_initial_pdist(self):
        d = np.full((self.n_states_, self.n_states_), np.inf)
        w_list = np.full((self.n_states_, self.n_states_), None)
        
#         n = len(self.cmat_)
#         d = np.empty((n,n))
         #w_list = []
        
        for i in range(self.n_states_):
            for j in range(i+1,self.n_states_):
                dist, weights = self._get_bayes_factor(i, j)
                d[i, j] = dist
                #w_list.append(weights)
                w_list[i, j] = weights
        
        return d, w_list

#         return (scipy.spatial.distance.squareform(d, checks=False),
#                 np.array(w_list))

    def _get_pair(self, pdist_out, n):
        pdist_vec, weights = pdist_out
        #print(pdist_vec)
        #print("get pair weights: ", weights)
        min_bf = np.min(pdist_vec)
        #print(min_bf)
        self.bayes_factors_.append((int(n-1), min_bf))  
        
        i, j = np.where(pdist_vec == min_bf)
        i = i[0]
        j = j[0]
#         print(k, self.n_states_)
#         i = np.triu_indices(self.n_states_,k=1)[0][k]
#         j = np.triu_indices(self.n_states_,k=1)[1][k]
#         i = int(n - 2 - int(sqrt(-8*k + 4*n*(n-1)-7)/2.0 - 0.5))
#         j = int(k + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2)

        self.microstate_mapping_[i] = np.min((i,j))
        self.microstate_mapping_[j] = np.min((i,j))
        return (i, j, weights[i,j])

    def _merge_states(self, pair_out, cmat_orig):
        cmat = np.copy(cmat_orig)
        if len(self.removed_) == 0:
            cmat += self.pseudocount
        
        pair = pair_out[0:2]
        w = pair_out[2]
        
        i = np.min(pair)
        j = np.max(pair)
        #print('pair out: ', i, j)
        w[i] += w[j]
        w[j] = 0
        self.removed_.append(j)
        
        for row_ind, row in enumerate(cmat):
            if row_ind != j:
                if row_ind != i:
                    row[i] += row[j]
                    
        new_row_i = np.zeros(cmat.shape[0])
        for ele_ind, _ in enumerate(new_row_i):
            if ele_ind != i:
                if ele_ind != j:
                    new_row_i[ele_ind] = cmat[i, ele_ind] + cmat[j, ele_ind]
        new_row_i[i] = cmat[i,i] + cmat[j,j] + cmat[i,j] + cmat[j,i]
        new_row_i[j] = 0.
        cmat[i] = new_row_i
        
        for rind in self.removed_:
            cmat[rind] = np.zeros(cmat.shape[0])
            cmat[:,rind] = np.zeros(cmat.shape[0])

        return cmat, w, i, j

    def _get_i_and_j(self, w, i, j, ii, jj, cmat):
        print('~~hello from get i and j~~')
        if ii in [i, j]:
            if w[i] > w[j]:
                my_i = i
            else:
                my_i = j
            print('my i is: ', my_i)
            w_i = w[my_i]
            
            if w[jj] == 0:
                jj = self.microstate_mapping_[jj]

            w_j = w[jj]
            c_i = cmat[my_i]
            c_j = cmat[jj]
                                
        elif jj in [i,j]:
            if w[i] > w[j]:
                my_j = i
            else:
                my_j = j
            print('my j is: ', my_j)
            
            if w[ii] == 0:
                ii = self.microstate_mapping_[ii]
                print("ii switched to ", ii, " !")

            w_i = w[ii]    
            print("w_i is ", w_i, " from index ii = ", ii)
            w_j = w[my_j]
            print("w_j is ", w_j)
            c_i = cmat[ii]
            c_j = cmat[my_j]
            
        return w_i, w_j, c_i, c_j, ii, jj

    def _update_helper(self, w, i, j, ii, jj, cmat):
        #print(w)
        w_i, w_j, c_i, c_j, ii, jj = self._get_i_and_j(w, i, j, ii, jj, cmat)

        for del_ind in sorted(np.where(w == 0)[0])[::-1]:
            c_i = np.delete(c_i,del_ind)
            c_j = np.delete(c_j,del_ind)

        d = self._distance_helper(w_i, w_j, c_i, c_j)
        return d, ii, jj

    def _update_pdists(self, merge_output, pdist_output):
        cmat = np.copy(merge_output[0])
        weights = np.copy(merge_output[1])
        i = merge_output[2]
        j = merge_output[3]
        
        new_pdist = pdist_output[0]
        new_pdist_weights = pdist_output[1]
#         print(new_pdist)
#         print(new_pdist_weights)
        for row in new_pdist_weights:
            for pw in row:
                if pw is not None:
                    pw[i] += pw[j]
                    pw[j] = 0
            
        n = cmat.shape[0]
        #print(cmat)
        
        print(new_pdist)
        
        for ii in range(n):
            for jj in range(ii+1,n):
                if (ii in [i, j] and jj not in [i, j]) or (
                    jj in [i, j] and ii not in [i, j]):              
                    #k = int((n*(n-1)/2) - (n-ii)*((n-ii)-1)/2 + jj - ii - 1)
                    if new_pdist[ii, jj] != np.inf:
                        print('weights: ', weights)
                        print("now that ", i, j, " have been merged,")
                        print("need to update the distance between ", ii, jj)
                        d, new_ii, new_jj = self._update_helper(weights, i, j, ii, jj, cmat)
                        
                        if (new_ii, new_jj) != (ii, jj):
                            new_pdist[ii, jj] = np.inf
                            new_pdist_weights[ii, jj] = None
                            ii = new_ii
                            jj = new_jj
                        
                        new_pdist[ii, jj] = d
                        print("MY NEW PDIST: ", new_pdist[ii,jj])
                        new_pdist_weights[ii, jj] = weights
        
        # set bf between merged clusters to inf
#         k = int((n*(n-1)/2) - (n-i)*((n-i)-1)/2 + j - i - 1)
#         new_pdist[k] = np.inf
#         new_pdist_weights[k] = None
        new_pdist[i, j] = np.inf
        new_pdist_weights[i, j] = None

        #print(new_pdist)
        return new_pdist, new_pdist_weights

    def _fix_mapping(self):
        unique_map = {}
        for i, v in enumerate(np.unique(list(self.microstate_mapping_.values()))):
            unique_map[v] = i
        
        new_mapping = []
        for i in self.microstate_mapping_.keys():
            new_mapping.append(unique_map[self.microstate_mapping_[i]])

        return np.array(new_mapping)

    def _do_lumping(self):
        """Do the BACE lumping.
        """
        # for this implementation we require integer counts, so we
        # adjust for sliding window
        if self.sliding_window:
            self.cmat_ = self.countsmat_ * self.lag_time
        else:
            self.cmat_ = self.countsmat_

        # default pseudocount is 1/n_states_
        if self.pseudocount is None:
            self.pseudocount = 1/self.n_states_

        self.bayes_factors_ = []
        self.removed_ = []
        self.microstate_mapping_ = {}
        for i in range(self.n_states_):
            self.microstate_mapping_[i] = i

        pdist_output = self._get_initial_pdist()
        #print(pdist_output)
        #print(pdist_output[0])
        cmat = self.cmat_
        n_macro = self.n_states_
        pair_output = self._get_pair(pdist_output, n_macro)
        merge = self._merge_states(pair_output, cmat)
        pdist_output = self._update_pdists(merge, pdist_output)
        n_macro = self.bayes_factors_[-1][0]        

        while n_macro > self.n_macrostates:
            #print(pdist_output[0])
            print(n_macro)
            pair_output = self._get_pair(pdist_output, n_macro)
            print('pair_output')
            merge = self._merge_states(pair_output, merge[0])
            print('merge')
            pdist_output = self._update_pdists(merge, pdist_output)
            print('pdist_output')
            print(self.microstate_mapping_)
            print(self.removed_)
            n_macro = self.bayes_factors_[-1][0]
        
        self.microstate_mapping_ = self._fix_mapping()

        if self.n_macrostates == 2:
            self.bayes_factors_.append((1, np.min(pdist_output[0])))

        self.bayes_factors_ = np.array(self.bayes_factors_)

    def partial_transform(self, sequence, mode='clip'):
        trimmed_sequence = super(BACE, self).partial_transform(sequence, mode)
        if mode == 'clip':
            return [self.microstate_mapping_[seq] for seq in trimmed_sequence]
        elif mode == 'fill':
            def nan_get(x):
                try:
                    x = int(x)
                    return self.microstate_mapping_[x]
                except ValueError:
                    return np.nan

            return np.asarray([nan_get(x) for x in trimmed_sequence])
        else:
            raise ValueError

    @classmethod
    def from_msm(cls, msm, n_macrostates=2):
        """Create and fit lumped model from pre-existing MSM.

        Parameters
        ----------
        msm : MarkovStateModel
            The input microstate msm to use.
        n_macrostates : int
            The number of macrostates (default : 2)

        Returns
        -------
        lumper : cls
            The fit BACE object.
        """
        params = msm.get_params()
        lumper = cls(n_macrostates=n_macrostates, **params)

        lumper.transmat_ = msm.transmat_
        lumper.populations_ = msm.populations_
        lumper.mapping_ = msm.mapping_
        lumper.countsmat_ = msm.countsmat_
        lumper.n_states_ = msm.n_states_

        lumper._do_lumping()

        return lumper
