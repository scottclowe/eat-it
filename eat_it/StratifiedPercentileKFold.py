"""
The :mod:`sklearn.cross_validation` module includes utilities for cross-
validation and performance evaluation.
"""

# Author: Scott Lowe
# License: BSD 3 clause

from __future__ import print_function
from __future__ import division

import warnings
from itertools import chain, combinations
from math import ceil, floor, factorial
import numbers
import time
from abc import ABCMeta, abstractmethod

import numpy as np
import scipy.sparse as sp

from sklearn.base import is_classifier, clone
#from sklearn.utils import indexable, check_random_state, safe_indexing
from sklearn.utils import check_random_state, safe_indexing
#from sklearn.utils.validation import (_is_arraylike, _num_samples, check_array, column_or_1d)
from sklearn.utils.validation import (_num_samples, column_or_1d)
from sklearn.utils.multiclass import type_of_target
from sklearn.externals.joblib import Parallel, delayed, logger
from sklearn.externals.six import with_metaclass
from sklearn.externals.six.moves import zip
from sklearn.metrics.scorer import check_scoring
#from sklearn.utils.fixes import bincount


from sklearn.cross_validation import _BaseKFold as _BaseKFold


class StratifiedPercentileKFold(_BaseKFold):
    """Stratified Percentile K-Folds cross validation iterator
    Provides train/test indices to split data in train test sets.
    This cross-validation object is a variation of KFold that
    returns stratified folds for a continuous variable. The folds
    are made by placing one sample in every K samples in the test
    set, which preserves the overall distribution of the target
    variable present both the training and test sets of each fold.
    Parameters
    ----------
    y : array-like, [n_samples]
        Samples to split in K folds.
    n_folds : int, default=3
        Number of folds. Must be at least 2.
    shuffle : boolean, optional
        Whether to shuffle each stratification of the data before splitting
        into batches. If shuffle is enabled, 
    random_state : None, int or RandomState
        Pseudo-random number generator state used for random
        sampling. If None, use default numpy RNG for shuffling
    Examples
    --------
    >>> from sklearn import cross_validation
    >>> X = np.array([[11, 12], [13, 14], [11, 12], [13, 14]])
    >>> y = np.array([0.1, 0.4, 0.8, 0.2])
    >>> cv = cross_validation.StratifiedPercentileKFold(y, n_folds=2, shuffle=True, random_state=0)
    >>> len(cv)
    2
    >>> print(cv)
    StratifiedPercentileKFold(labels=[ 0.1  0.4  0.8  0.2], n_folds=2, shuffle=True, random_state=0)
    >>> for train_index, test_index in cv:
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    ...     X_train, X_test = X[train_index], X[test_index]
    ...     y_train, y_test = y[train_index], y[test_index]
    ...     print(X_test, y_test)
    TRAIN: [1 3] TEST: [0 2]
    [[11 12]
     [11 12]] [ 0.1  0.8]
    TRAIN: [0 2] TEST: [1 3]
    [[13 14]
     [13 14]] [ 0.4  0.2]
    Notes
    -----
    Each fold is sized either the floor or ceil of (n_samples / n_folds), in a
    random order. The total number of samples in all the folds is n_samples,
    with each sample appearing exactly once.
    """
    
    def __init__(self, y, n_folds=3, indices=None, shuffle=False,
                 random_state=None, shuffle_windows=False):
        super(StratifiedPercentileKFold, self).__init__(
            len(y), n_folds, indices, shuffle, random_state)
        
        self.shuffle_windows = shuffle_windows
        
        y = np.asarray(y)
        self.y = y
        
        n_samples = y.shape[0]
        
        if self.shuffle:
            rng = check_random_state(self.random_state)
        else:
            rng = self.random_state
        
        # For each X percentile,
        # Put each of the values into a unique fold
        # Do this by shuffling an array, holding indices from 0 to num_in_pc
        # in dim0, repeated K times in dim1
        
        n_windows = ceil(n_samples / n_folds)
        
        # Make a matrix identifying which fold each sample is in test set
        v = np.arange(n_folds)
        v = np.expand_dims(v, 0)
        v = np.tile(v, [n_windows,1])
        #print(v)
        
        # get number of samples in each window (method 1)
        #split_edges = np.linspace(0, n_samples, n_folds+1)
        #split_edges = np.round(split_edges)
        #n_samples_per_window = np.diff(split_edges)
        
        # get number of samples in each window (method 2)
        split_edges = np.arange(n_windows+1) * n_samples / n_windows
        split_edges = np.round(split_edges)
        n_samples_per_window = np.diff(split_edges)
        
        if self.shuffle_windows:
            # randomly wrap this vector
            n_samples_per_window = np.roll(n_samples_per_window, rng.randint(n_windows))
        
        # drop some folds as possibilities
        windows_to_shorten = np.where(n_samples_per_window<n_folds)[0]
        # need to know which order to drop each fold number
        # drop each fold once at most
        # This doesn't need to be picked at random
        if False and self.shuffle_windows:
            fold_drop_order = rng.permutation(n_folds)
        else:
            fold_drop_order = np.arange(n_folds)
        j = 0
        for i in windows_to_shorten:
            v[i,fold_drop_order[j]] = -1
            j += 1
        
        # Shuffle which fold each sample is in test set for
        if self.shuffle:
            for i in range(n_windows):
                rng.shuffle(v[i])
        
        # Wrap the matrix into a flat array
        v = np.ravel(v)
        # removing unnecessary values from the middle
        v = v[v!=-1]
        
        # Sort the regression targets in ascending order
        sort_idx = np.argsort(y)
        # sorted_y = y[sort_idx] is now in ascending order
        # We need to convert sorted_y back to y
        # Sort the sort index to turn the indices back into ascending order
        inverse_of_sort_idx = np.argsort(sort_idx)
        # now we should have y = sorted_y[inverse_of_sort_idx]
        
        # v refers to when to include each element in their sorted order
        # So use the inverse of the sorting indices to map v to the ordering of y
        self.test_folds = v[inverse_of_sort_idx]
        

    def _iter_test_masks(self):
        for i in range(self.n_folds):
            yield self.test_folds == i

    def __repr__(self):
        return '%s.%s(labels=%s, n_folds=%i, shuffle=%s, random_state=%s, shuffle_windows=%s)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.y,
            self.n_folds,
            self.shuffle,
            self.random_state,
            self.shuffle_windows
        )

    def __len__(self):
        return self.n_folds

