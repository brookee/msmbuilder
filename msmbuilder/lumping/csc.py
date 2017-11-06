# Author: Brooke Husic <brookehusic@gmail.com>
# Contributors: John Dabiri
# Copyright (c) 2017, Stanford University and the Authors
# All rights reserved.

from __future__ import print_function, division, absolute_import

import numpy as np
from msmbuilder.cluster import LandmarkAgglomerative
from msmbuilder.msm import MarkovStateModel
from msmbuilder.cluster.agglomerative import pdist
from msmbuilder.utils.divergence import js_metric_array
import scipy.cluster.hierarchy


class CSC(MarkovStateModel):
    """Coherent Structure Coloring (CSC) for coarse-graining (lumping)
    microstates into macrostates.

    Parameters
    ----------
    n_macrostates : int or None
        The desired number of macrostates in the lumped model. If None,
        only the linkages are calcluated (see ``use_scipy``)
    linkage : {'single', 'complete', 'average', 'ward'}, default='average'
        Which linkage criterion to use. The linkage criterion determines which
        distance to use between sets of observation. The algorithm will merge
        the pairs of cluster that minimize this criterion.
            - average uses the average of the distances of each observation of
              the two sets.
            - complete or maximum linkage uses the maximum distances between
              all observations of the two sets.
            - single uses the minimum distance between all observations of the
              two sets.
            - ward linkage minimizes the within-cluster variance
        The linkage also effects the predict() method and the use of landmarks.
        After computing the distance from each new data point to the landmarks,
        the new data point will be assigned to the cluster that minimizes the
        linkage function between the new data point and each of the landmarks.
        (i.e with ``single``, new data points will be assigned the label of
        the closest landmark, with ``average``, it will be assigned the label
        of the landmark s.t. the mean distance from the test point to all the
        landmarks with that label is minimized, etc.)
    eig : int, default=0
        The eigenprocess along which to lump
    metric : string or callable, default=js_metric_array
        Function to determine pairwise distances. Can be custom.
    fit_only : boolean, default=False
        If True, the fit landmark points will be returned. If False,
        the predicted labels will be returned. In general these are not equal.
    n_landmarks : int, optional
        Memory-saving approximation. Instead of actually clustering every
        point, we instead select n_landmark points either randomly or by
        striding the data matrix (see ``landmark_strategy``). Then we cluster
        the only the landmarks, and then assign the remaining dataset based
        on distances to the landmarks. Note that n_landmarks=None is equivalent
        to using every point in the dataset as a landmark.
    landmark_strategy : {'stride', 'random'}, default='stride'
        Method for determining landmark points. Only matters when n_landmarks
        is not None. "stride" takes landmarks every n-th data point in X, and
        random selects them  uniformly at random.
    random_state : integer or numpy.RandomState, optional
        The generator used to select random landmarks. Only used if
        landmark_strategy=='random'. If an integer is given, it fixes the seed.
        Defaults to the global numpy random number generator.
    kwargs : optional
        Additional keyword arguments to be passed to MarkovStateModel.  See
        msmbuilder.msm.MarkovStateModel for possible options.

    Attributes
    ----------
    microstate_mapping_ : np.array, [number of microstates]

    Notes
    -----
    This method is described in the following manuscripts:
        1. Schlueter-Kuck, K. L. and Dabiri, J. O., J. Fluid Mech. 811,
           469 (2017).
        2. Schlueter-Kuck, K. L. and Dabiri, J. O., Chaos 27, 091101 (2017).

    CSC is a subclass of MarkovStateModel.  However, the MSM properties
    and attributes on CSC refer to the MICROSTATE properties--e.g.
    mvca.transmat_ is the microstate transition matrix.  To get the
    macrostate transition matrix, you must fit a new MarkovStateModel
    object on the output (assignments) of CSC().
    CSC will scale poorly with the number of microstates. Consider
    use_scipy=False and n_landmarks < number of microstates.
    """
    def __init__(self, n_macrostates, linkage='average',
                 eig=0, metric=js_metric_array, fit_only=False,
                 adjacency_mat=None, n_landmarks=None,
                 landmark_strategy='stride',
                 random_state=None, **kwargs):
        self.n_macrostates = n_macrostates
        self.linkage = linkage
        self.eig = eig
        self.metric = metric
        self.fit_only = fit_only
        self.adjacency_mat = adjacency_mat
        self.n_landmarks = n_landmarks
        self.landmark_strategy = landmark_strategy
        self.random_state = random_state
        self._CSC_fullset = None
        super(CSC, self).__init__(**kwargs)

    def fit(self, sequences, y=None):
        """Fit a CSC lumping model using a sequence of cluster assignments.

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
        super(CSC, self).fit(sequences, y=y)
        if self.n_macrostates is not None:
            self._do_lumping()
        else:
            raise RuntimeError('n_macrostates must not be None to fit')

        return self

    def _do_lumping(self):
        """Do the CSC lumping.
        """
        if self.adjacency_mat is None:
            A = scipy.spatial.distance.squareform(
                        pdist(self.transmat_, metric=self.metric))
            self.adjacency_mat = A

        else:
            A = self.get_adj_mat(dim=2)

        Adegree = np.sum(A, axis=1)
        DD = np.diag(Adegree)
        L = DD - A 
        eigval, eigvec = scipy.linalg.eig(L, DD)
        lambdaindex = np.argsort(eigval)[::-1]
        sortedeigs = eigval[lambdaindex]
        CSC_fullset = eigvec[:, lambdaindex]
        self._CSC_fullset = CSC_fullset

        process = np.array([[i] for i in self._CSC_fullset[:,self.eig]])

        model = LandmarkAgglomerative(linkage=self.linkage,
                                      n_clusters=self.n_macrostates,
                                      n_landmarks=self.n_landmarks,
                                      landmark_strategy=self.landmark_strategy,
                                      random_state=self.random_state)
        model.fit([process])

        if self.fit_only:
            microstate_mapping_ = model.landmark_labels_

        else:
            microstate_mapping_ = model.transform([process])[0]

        self.microstate_mapping_ = microstate_mapping_

    def partial_transform(self, sequence, mode='clip'):
        if self.n_macrostates is None:
            raise RuntimeError('n_macrostates must not be None to transform')
        trimmed_sequence = super(CSC, self).partial_transform(sequence, mode)
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

    def get_adj_mat(self, dim=1):
        A = np.array(self.adjacency_mat)

        if len(A.shape) not in [1, 2]:
            raise RuntimeError('Adjacency matrix must be square or linear')

        if dim == 1:
            if len(A.shape) == 2:
                A = scipy.spatial.distance.squareform(A)
            return A

        elif dim == 2:
            if len(A.shape) == 1:
                A = scipy.spatial.distance.squareform(A)
            return A

    @classmethod
    def from_msm(cls, msm, n_macrostates, linkage='average', eig=0,
                 metric=js_metric_array, fit_only=False,
                 adjacency_mat=None, n_landmarks=None,
                 landmark_strategy='stride', random_state=None,
                 get_linkage=False):
        """Create and fit lumped model from pre-existing MSM.

        Parameters
        ----------
        msm : MarkovStateModel
            The input microstate msm to use.
        n_macrostates : int
            The number of macrostates
        get_linkage : boolean, default=False
            Whether to return linkage and elbow data objects. Warning:
            This will compute n choose 2 pairwise distances

        Returns
        -------
        lumper : cls
            The fit CSC object.
        pairwise_dists : if get_linkage is True, np.array,
                         [number of microstates choose 2]
        linkage : if get_linkage is True, scipy linkage object
        elbow_data : if get_linkage is True, np.array,
                     [number of microstates - 1]. Change in updated Ward
                     objective function, indexed by n_macrostates - 1

        Example
        -------
        plt.figure()
        scipy.cluster.hierarchy.dendrogram(mvca.linkage)

        scatter(arange(1,n_microstates), mvca.elbow_data)
        """
        params = msm.get_params()
        lumper = cls(n_macrostates, linkage=linkage, eig=eig, metric=metric,
                 fit_only=fit_only, adjacency_mat=adjacency_mat,
                 n_landmarks=n_landmarks, landmark_strategy=landmark_strategy,
                 random_state=random_state, **params)

        lumper.transmat_ = msm.transmat_
        lumper.populations_ = msm.populations_
        lumper.mapping_ = msm.mapping_
        lumper.countsmat_ = msm.countsmat_
        lumper.n_states_ = msm.n_states_

        if n_macrostates is not None:
            lumper._do_lumping()

        if get_linkage:
            if lumper.adjacency_mat is None:
                p = pdist(msm.transmat_, metric=metric)
                lumper.adjacency_mat = p

            else:
                p = lumper.get_adj_mat(dim=1)

            #p = pdist(msm.transmat_, metric=metric)
            l = scipy.cluster.hierarchy.linkage(p, linkage)

            lumper.pairwise_dists = p
            lumper.linkage = l
            lumper.elbow_data = l[:, 2][::-1]

        else:
            lumper.pairwise_dists = None
            lumper.linkage = None
            lumper.elbow_data = None

        return lumper

