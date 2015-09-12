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
#from sklearn.utils import check_array
from sklearn.utils import warn_if_not_float
from sklearn.utils.extmath import row_norms
#from sklearn.utils.fixes import (combinations_with_replacement as combinations_w_r, bincount)
from sklearn.utils.fixes import (combinations_with_replacement as combinations_w_r)
from sklearn.utils.fixes import isclose
from sklearn.utils.sparsefuncs_fast import (inplace_csr_row_normalize_l1,
                                      inplace_csr_row_normalize_l2)
#from sklearn.utils.sparsefuncs import (inplace_column_scale, mean_variance_axis)
from sklearn.utils.sparsefuncs import (inplace_column_scale)

#from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import *
from sklearn.utils.validation import (_assert_all_finite, _num_samples)
from sklearn.preprocessing.data import *


zip = six.moves.zip
map = six.moves.map
range = six.moves.range

#####################################################################################
# Should be imported from sklearn, but can't work. Might be version difference?
#####################################################################################

def check_array(array, accept_sparse=None, dtype="numeric", order=None,
                copy=False, force_all_finite=True, ensure_2d=True,
                allow_nd=False, ensure_min_samples=1, ensure_min_features=1):
    """Input validation on an array, list, sparse matrix or similar.
    By default, the input is converted to an at least 2nd numpy array.
    If the dtype of the array is object, attempt converting to float,
    raising on failure.
    Parameters
    ----------
    array : object
        Input object to check / convert.
    accept_sparse : string, list of string or None (default=None)
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc.  None means that sparse matrix input will raise an error.
        If the input is sparse but not in the allowed format, it will be
        converted to the first listed format.
    dtype : string, type or None (default="numeric")
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
    order : 'F', 'C' or None (default=None)
        Whether an array will be forced to be fortran or c-style.
    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.
    force_all_finite : boolean (default=True)
        Whether to raise an error on np.inf and np.nan in X.
    ensure_2d : boolean (default=True)
        Whether to make X at least 2d.
    allow_nd : boolean (default=False)
        Whether to allow X.ndim > 2.
    ensure_min_samples : int (default=1)
        Make sure that the array has a minimum number of samples in its first
        axis (rows for a 2D array). Setting to 0 disables this check.
    ensure_min_features : int (default=1)
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when the input data has effectively 2
        dimensions or is originally 1D and ``ensure_2d`` is True. Setting to 0
        disables this check.
    Returns
    -------
    X_converted : object
        The converted and validated X.
    """
    if isinstance(accept_sparse, str):
        accept_sparse = [accept_sparse]

    # store whether originally we wanted numeric dtype
    dtype_numeric = dtype == "numeric"

    if sp.issparse(array):
        if dtype_numeric:
            dtype = None
        array = _ensure_sparse_format(array, accept_sparse, dtype, order,
                                      copy, force_all_finite)
    else:
        if ensure_2d:
            array = np.atleast_2d(array)
        if dtype_numeric:
            if hasattr(array, "dtype") and getattr(array.dtype, "kind", None) == "O":
                # if input is object, convert to float.
                dtype = np.float64
            else:
                dtype = None
        array = np.array(array, dtype=dtype, order=order, copy=copy)
        # make sure we actually converted to numeric:
        if dtype_numeric and array.dtype.kind == "O":
            array = array.astype(np.float64)
        if not allow_nd and array.ndim >= 3:
            raise ValueError("Found array with dim %d. Expected <= 2" %
                             array.ndim)
        if force_all_finite:
            _assert_all_finite(array)

    shape_repr = _shape_repr(array.shape)
    if ensure_min_samples > 0:
        n_samples = _num_samples(array)
        if n_samples < ensure_min_samples:
            raise ValueError("Found array with %d sample(s) (shape=%s) while a"
                             " minimum of %d is required."
                             % (n_samples, shape_repr, ensure_min_samples))

    if ensure_min_features > 0 and array.ndim == 2:
        n_features = array.shape[1]
        if n_features < ensure_min_features:
            raise ValueError("Found array with %d feature(s) (shape=%s) while"
                             " a minimum of %d is required."
                             % (n_features, shape_repr, ensure_min_features))
    return array


def _shape_repr(shape):
    """Return a platform independent reprensentation of an array shape
    Under Python 2, the `long` type introduces an 'L' suffix when using the
    default %r format for tuples of integers (typically used to store the shape
    of an array).
    Under Windows 64 bit (and Python 2), the `long` type is used by default
    in numpy shapes even when the integer dimensions are well below 32 bit.
    The platform specific type causes string messages or doctests to change
    from one platform to another which is not desirable.
    Under Python 3, there is no more `long` type so the `L` suffix is never
    introduced in string representation.
    >>> _shape_repr((1, 2))
    '(1, 2)'
    >>> one = 2 ** 64 / 2 ** 64  # force an upcast to `long` under Python 2
    >>> _shape_repr((one, 2 * one))
    '(1, 2)'
    >>> _shape_repr((1,))
    '(1,)'
    >>> _shape_repr(())
    '()'
    """
    if len(shape) == 0:
        return "()"
    joined = ", ".join("%d" % e for e in shape)
    if len(shape) == 1:
        # special notation for singleton tuples
        joined += ','
    return "(%s)" % joined
    
def check_is_fitted(estimator, attributes, msg=None, all_or_any=all):
    """Perform is_fitted validation for estimator.
    Checks if the estimator is fitted by verifying the presence of
    "all_or_any" of the passed attributes and raises a NotFittedError with the
    given message.
    Parameters
    ----------
    estimator : estimator instance.
        estimator instance for which the check is performed.
    attributes : attribute name(s) given as string or a list/tuple of strings
        Eg. : ["coef_", "estimator_", ...], "coef_"
    msg : string
        The default error message is, "This %(name)s instance is not fitted
        yet. Call 'fit' with appropriate arguments before using this method."
        For custom messages if "%(name)s" is present in the message string,
        it is substituted for the estimator name.
        Eg. : "Estimator, %(name)s, must be fitted before sparsifying".
    all_or_any : callable, {all, any}, default all
        Specify whether all or any of the given attributes must exist.
    """
    if msg is None:
        msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
               "appropriate arguments before using this method.")

    if not hasattr(estimator, 'fit'):
        raise TypeError("%s is not an estimator instance." % (estimator))

    if not isinstance(attributes, (list, tuple)):
        attributes = [attributes]

    if not all_or_any([hasattr(estimator, attr) for attr in attributes]):
        raise NotFittedError(msg % {'name': type(estimator).__name__})


def _transform_selected(X, transform, selected="all", copy=True):
    """Apply a transform function to portion of selected features
    Parameters
    ----------
    X : array-like or sparse matrix, shape=(n_samples, n_features)
        Dense array or sparse matrix.
    transform : callable
        A callable transform(X) -> X_transformed
    copy : boolean, optional
        Copy X even if it could be avoided.
    selected: "all" or array of indices or mask
        Specify which features to apply the transform to.
    Returns
    -------
    X : array or sparse matrix, shape=(n_samples, n_features_new)
    """
    if selected == "all":
        return transform(X)

    X = check_array(X, accept_sparse='csc', copy=copy)

    if len(selected) == 0:
        return X

    n_features = X.shape[1]
    ind = np.arange(n_features)
    sel = np.zeros(n_features, dtype=bool)
    sel[np.asarray(selected)] = True
    not_sel = np.logical_not(sel)
    n_selected = np.sum(sel)

    if n_selected == 0:
        # No features selected.
        return X
    elif n_selected == n_features:
        # All features selected.
        return transform(X)
    else:
        X_sel = transform(X[:, ind[sel]])
        X_not_sel = X[:, ind[not_sel]]

        if sparse.issparse(X_sel) or sparse.issparse(X_not_sel):
            return sparse.hstack((X_sel, X_not_sel))
        else:
            return np.hstack((X_sel, X_not_sel))
#####################################################################################



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
    standardise: boolean, optional, default True
        Set to False to perform BoxCox transformation without
        standardisation. By default, the transformation is standardised
        to have mean=0 and std=1.
    Attributes
    ----------
    offset_ : ndarray, shape (n_features,)
        Per feature adjustment for minimum.
    lambda_ : ndarray, shape (n_features,)
        Per feature lambda power fit for Box-Cox transformation.
    """

    def __init__(self, known_min=None, copy=True, standardise=True):
        self.known_min = known_min
        self.copy = copy
        self.standardise = standardise
        
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
        data_min = np.min(X, axis=0, keepdims=True)
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
        
        # Apply the offset to the raw data
        X += self.offset_
        
        # Find the optimal Box-Cox transform for each feature
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        lambda_values = np.zeros((n_features))
        Xt = np.zeros(X.shape)
        
        for i in range(n_features):
            # Fit the BoxCox transform to the data in this column
            Xt[:,i], lambda_values[i] = stats.boxcox(X[:,i])
            
            # Sanity check:
            # Make sure the transformed values are not all the same if the original
            # data wasn't like that
            # (this is a bug which can happen if lambda was chosen badly by scipy)
            if not np.allclose(X[:,i], X[0,i]*np.ones(n_samples)) and np.allclose(Xt[:,i], Xt[0,i]*np.ones(n_samples)):
                raise ValueError("Lambda was badly chosen for feature {}. Values became singular!".format(i))
            # We should fix this issue by finding a better lambda value ourselves
        
        # Store the lambda value
        self.lambda_ = lambda_values
        
        # Fit the to z-score with standard scaler
        if self.standardise:
            self.standardiser = StandardScaler()
            # Fit on the transformed data
            self.standardiser.fit(Xt, y)
            
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
        Xt = (np.power(X,self.lambda_) - 1) / self.lambda_
        # But the features with lambda==0 are a special case
        # These are just log-transform of original data
        li = self.lambda_==0
        Xt[li] = np.log(X[li])
        
        # Convert to Z-score if necessary
        if self.standardise:
            Xt = self.standardiser.transform(Xt)
            
        return Xt

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
        
        # Swap from Z-score to regular box-cox
        if self.standardise:
            X = self.standardiser.inverse_transform(X, copy=self.copy)
        
        # Undo Box-Cox transform for each feature
        Xt = np.power((X * self.lambda_) + 1, 1 / self.lambda_)
        # But the features with lambda==0 are a special case
        # These were just log-transform of original data
        li = self.lambda_==0
        Xt[li] = np.exp(X[li])
        # Return negative values appropriately
        Xt -= self.offset_
        return Xt


class OrdinalHotEncoder(BaseEstimator, TransformerMixin):
    """Encode ordinal features using a hot-if-greater-than scheme.
    The input to this transformer should be a matrix of integers, denoting
    the values taken on by categorical (discrete) features. The output will be
    a sparse matrix where each column corresponds to one possible value of one
    feature. It is assumed that input features take on values in the range
    [0, n_values).
    This encoding is needed for feeding categorical data to many scikit-learn
    estimators, notably linear models and SVMs with the standard kernels.
    Parameters
    ----------
    n_values : 'auto', int or array of ints
        Number of values per feature.
        - 'auto' : determine value range from training data.
        - int : maximum value for all features.
        - array : maximum value per feature.
    categorical_features: "all" or array of indices or mask
        Specify what features are treated as categorical.
        - 'all' (default): All features are treated as categorical.
        - array of indices: Array of categorical feature indices.
        - mask: Array of length n_features and with dtype=bool.
        Non-categorical features are always stacked to the right of the matrix.
    dtype : number type, default=np.float
        Desired dtype of output.
    sparse : boolean, default=True
        Will return sparse matrix if set True else will return an array.
    handle_unknown : str, 'error' or 'ignore'
        Whether to raise an error or ignore if a unknown categorical feature is
        present during transform.
    Attributes
    ----------
    active_features_ : array
        Indices for active features, meaning values that actually occur
        in the training set. Only available when n_values is ``'auto'``.
    feature_indices_ : array of shape (n_features,)
        Indices to feature ranges.
        Feature ``i`` in the original data is mapped to features
        from ``feature_indices_[i]`` to ``feature_indices_[i+1]``
        (and then potentially masked by `active_features_` afterwards)
    n_values_ : array of shape (n_features,)
        Maximum number of values per feature.
    Examples
    --------
    Given a dataset with three features and two samples, we let the encoder
    find the maximum value per feature and transform the data to a binary
    one-hot encoding.
    >>> from sklearn.preprocessing import OneHotEncoder
    >>> enc = OneHotEncoder()
    >>> enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], \
[1, 0, 2]])  # doctest: +ELLIPSIS
    OneHotEncoder(categorical_features='all', dtype=<... 'float'>,
           handle_unknown='error', n_values='auto', sparse=True)
    >>> enc.n_values_
    array([2, 3, 4])
    >>> enc.feature_indices_
    array([0, 2, 5, 9])
    >>> enc.transform([[0, 1, 1]]).toarray()
    array([[ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.]])
    See also
    --------
    sklearn.feature_extraction.DictVectorizer : performs a one-hot encoding of
      dictionary items (also handles string-valued features).
    sklearn.feature_extraction.FeatureHasher : performs an approximate one-hot
      encoding of dictionary items or strings.
    """
    def __init__(self, edge_values="auto", categorical_features="all",
                 dtype=np.float, handle_unknown='error'):
        self.edge_values = edge_values
        self.categorical_features = categorical_features
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        """Fit OneHotEncoder to X.
        Parameters
        ----------
        X : array-like, shape=(n_samples, n_feature)
            Input array of type int.
        Returns
        -------
        self
        """
        return _transform_selected(X, self._fit,
                                   self.categorical_features, copy=True)

    def _fit(self, X):
        """Assumes X contains only categorical features."""
        X = check_array(X)
        
        n_samples, n_features = X.shape
        
        if self.edge_values == 'auto':
            edge_values = []
            for i in range(n_features):
                unq = np.unique(X[:,i])
                edg = (unq[1:] + unq[:-1])/2
                edge_values.append(edg)
        else:
            raise NotImplementedError
        
        self.edge_values_ = edge_values

        return self

    def fit_transform(self, X, y=None):
        """Fit encoder to X, then transform X.
        Equivalent to self.fit(X).transform(X), but more convenient.
        See fit for the parameters, transform for the return value.
        """
        self.fit(X, y)
        return _transform_selected(X, self._transform,
                                   self.categorical_features, copy=True)

    def _transform(self, X):
        """Assumes X contains only ordinal features."""
        X = check_array(X)
        
        n_samples, n_features = X.shape
        
        X_t = []
        
        for i in range(n_features):
            n_val = len(self.edge_values_[i])
            Xti = np.zeros((n_samples, n_val), dtype=self.dtype)
            for j in range(n_val):
                Xti[:,j] = X[:,i] > self.edge_values_[i][j]
            X_t.append(Xti)
            
        X_t = np.concatenate(X_t, axis=1)
        
        return X_t

    def transform(self, X):
        """Transform X using one-hot encoding.
        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Input array of type int.
        Returns
        -------
        X_out : sparse matrix if sparse=True else a 2-d array, dtype=int
            Transformed input.
        """
        return _transform_selected(X, self._transform,
                                   self.categorical_features, copy=True)
