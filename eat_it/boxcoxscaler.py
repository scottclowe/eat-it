# Author: Scott Lowe
#
# License: BSD 3 clause

from itertools import chain, combinations
import numbers
import warnings

import numpy as np
from scipy import sparse
from scipy import stats

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import six
from sklearn.utils import check_array
from sklearn.utils import warn_if_not_float
from sklearn.utils.extmath import row_norms
from sklearn.utils.fixes import (combinations_with_replacement as combinations_w_r,
                           bincount)
from sklearn.utils.fixes import isclose
from sklearn.utils.sparsefuncs_fast import (inplace_csr_row_normalize_l1,
                                      inplace_csr_row_normalize_l2)
from sklearn.utils.sparsefuncs import (inplace_column_scale, mean_variance_axis)
from sklearn.utils.validation import check_is_fitted


zip = six.moves.zip
map = six.moves.map
range = six.moves.range


class BoxCoxScaler(BaseEstimator, TransformerMixin):
    """Standardizes features by scaling each feature with Box-Cox
    Parameters
    ----------
    known_min: array-like, default=None
        A-priori known minium values of each feature.
        If none, then feature minima are inferred from the training
        dataset when fitting.
        NB: Only non-positive features are offset-corrected.
    copy : boolean, optional, default True
        Set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array).
    Attributes
    ----------
    offset_ : ndarray, shape (n_features,)
        Per feature adjustment for minimum.
    lambda_ : ndarray, shape (n_features,)
        Per feature lambda power fit for Box-Cox transformation.
    """

    def __init__(self, known_min=None, copy=True):
        self.known_min = None
        self.copy = copy

    def fit(self, X, y=None):
        """Compute the Box-Cox lambda value to be used for later scaling.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to compute the per-feature minimum and maximum
            used for later scaling along the features axis.
        """
        X = check_array(X, copy=self.copy, ensure_2d=True)
        warn_if_not_float(X, estimator=self)
        
        # Take the minimum of each feature
        data_min = np.min(X, axis=0)
        # Sanity check
        if self.known_min is not None and np.any(self.known_min > data_min):
            raise Warning("The minimum of the data is less than the supplied"
                          " 'known' minimum value.")
        
        if self.known_min is not None:
            data_min = np.minimum(data_min, self.known_min)
        else:
            # Since the user didn't know how negative the values could be,
            # let's err on the side of caution a little bit
            data_min = data_min*2
            # Note: this has no effect is the data is non-negative.
        
        # Need to offset by the negative of the minima so all values are +ve
        offset = -data_min
        # And we need to ensure 0 gets mapped correctly
        offset[offset >= 0] += 1
        # We want to change inputs which are always +ve
        offset = np.maximum(offset, 0)
        
        # Store this array of feature offsets
        self.offset_ = offset
        
        # Find the optimal Box-Cox transform for each feature
        n_samples = X.shape[0]
        n_features = X.shape[1]
        self.lambda_values
        for i in range(n_features):
            # Apply the offset to the raw data
            tXi, lambda_values[i] = stats.boxcox(X[:,i])
            
            # Sanity check:
            # Make sure the transformed values are not all the same if the original
            # data wasn't like that
            # (this is a bug which can happen if lambda was chosen badly)
            if not np.allclose(X[:,i], X[0,i]*np.ones(n_samples)) and
                np.allclose(tXi, tXi[0]*np.ones(n_samples)):
                raise ValueError("Lambda was badly chosen for feature {}."
                                 "Values became singular!" % i)
        
        # Store the lambda value
        self.lambda_ = lambda_values
        return self

    def transform(self, X):
        """Box-Cox transforming of X with fitted lambda.
        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            Input data that will be transformed.
        """
        check_is_fitted(self, 'lambda_')
        check_is_fitted(self, 'offset_')
        
        # Apply offset to fix negative values and zero
        X += self.offset_
        # Transform each feature with Box-Cox transform
        X = (np.power(X,self.lambda_) - 1) / self.lambda_
        # But the features with lambda==0 are a special case
        # These are just log-transform of original data
        li = self.lambda_==0
        X[li] = np.log(X[li])
        return X

    def inverse_transform(self, X):
        """Undo the Box-Cox transform of X.
        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            Input data that will be transformed.
        """
        check_is_fitted(self, 'lambda_')
        check_is_fitted(self, 'offset_')

        X = check_array(X, copy=self.copy, ensure_2d=False)
        
        # Undo Box-Cox transform for each feature
        X = np.power((X * self.lambda_) + 1, 1 / self.lambda_)
        # But the features with lambda==0 are a special case
        # These were just log-transform of original data
        li = self.lambda_==0
        X[li] = np.exp(X[li])
        # Return negative values appropriately
        X -= self.offset_
        return X
