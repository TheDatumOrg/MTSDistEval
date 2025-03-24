import itertools
import warnings
from functools import partial
from numbers import Integral, Real

import numpy as np
from joblib import effective_n_jobs,Parallel,delayed
from scipy.sparse import csr_matrix, issparse
from scipy.spatial import distance



def pairwise_distances(X,Y=None,metric:callable=None,adaptive_scaling=False,n_jobs=-1,**kwds):
    func = partial(_pairwise_callable,metric=metric,adaptive_scaling=adaptive_scaling, **kwds)
    return _parallel_pairwise(X, Y, func, n_jobs, **kwds) 

def _dist_wrapper(dist_func, *args, **kwargs):
    """Write in-place to a slice of a distance matrix."""
    return dist_func(*args, **kwargs)

def _parallel_pairwise(X, Y, func, n_jobs, **kwds):
    """Break the pairwise matrix in n_jobs even slices
    and compute them in parallel."""

    if Y is None:
        Y = X

    if effective_n_jobs(n_jobs) == 1:
        return func(X, Y, **kwds)

    # enforce a threading backend to prevent data communication overhead
    fd = delayed(_dist_wrapper)
    # ret = np.empty((X.shape[0], Y.shape[0]), order="F")
    ret = Parallel(backend="loky", n_jobs=n_jobs)(
        fd(func, X, Y[s], **kwds) for s in gen_even_slices(len(Y), effective_n_jobs(n_jobs))
    )

    ret = np.concatenate(ret, axis=1)

    return ret


def _pairwise_callable(X, Y, metric,adaptive_scaling=False, **kwds):
    """Handle the callable case for pairwise_{distances,kernels}."""
    if X is Y:
        # Only calculate metric for upper triangle
        out = np.zeros((X.shape[0], Y.shape[0]), dtype="float")
        iterator = itertools.combinations(range(X.shape[0]), 2)
        for i, j in iterator:
            # scipy has not yet implemented 1D sparse slices; once implemented this can
            # be removed and `arr[ind]` can be simply used.
            x = X[[i], :] if issparse(X) else X[i]
            y = Y[[j], :] if issparse(Y) else Y[j]

            if adaptive_scaling:
                a = np.sum(np.multiply(x,y),axis=1,keepdims=True) / np.sum(np.multiply(y,y),axis=1,keepdims=True)
                y = a*y
            out[i, j] = metric(x, y, **kwds)

        # Make symmetric
        # NB: out += out.T will produce incorrect results
        out = out + out.T

        # Calculate diagonal
        # NB: nonzero diagonals are allowed for both metrics and kernels
        for i in range(X.shape[0]):
            # scipy has not yet implemented 1D sparse slices; once implemented this can
            # be removed and `arr[ind]` can be simply used.
            x = X[[i], :] if issparse(X) else X[i]
            out[i, i] = metric(x, x, **kwds)

    else:
        # Calculate all cells
        out = np.empty((X.shape[0], Y.shape[0]), dtype="float")
        iterator = itertools.product(range(X.shape[0]), range(Y.shape[0]))
        for i, j in iterator:
            # scipy has not yet implemented 1D sparse slices; once implemented this can
            # be removed and `arr[ind]` can be simply used.
            x = X[[i], :] if issparse(X) else X[i]
            y = Y[[j], :] if issparse(Y) else Y[j]

            if adaptive_scaling:
                a = np.sum(np.multiply(x,y),axis=1,keepdims=True) / np.sum(np.multiply(y,y),axis=1,keepdims=True)
                y = a*y
            out[i, j] = metric(x, y, **kwds)

    return out

def gen_even_slices(n, n_packs, *, n_samples=None):
    """Generator to create `n_packs` evenly spaced slices going up to `n`.

    If `n_packs` does not divide `n`, except for the first `n % n_packs`
    slices, remaining slices may contain fewer elements.

    Parameters
    ----------
    n : int
        Size of the sequence.
    n_packs : int
        Number of slices to generate.
    n_samples : int, default=None
        Number of samples. Pass `n_samples` when the slices are to be used for
        sparse matrix indexing; slicing off-the-end raises an exception, while
        it works for NumPy arrays.

    Yields
    ------
    `slice` representing a set of indices from 0 to n.

    See Also
    --------
    gen_batches: Generator to create slices containing batch_size elements
        from 0 to n.

    Examples
    --------
    >>> from sklearn.utils import gen_even_slices
    >>> list(gen_even_slices(10, 1))
    [slice(0, 10, None)]
    >>> list(gen_even_slices(10, 10))
    [slice(0, 1, None), slice(1, 2, None), ..., slice(9, 10, None)]
    >>> list(gen_even_slices(10, 5))
    [slice(0, 2, None), slice(2, 4, None), ..., slice(8, 10, None)]
    >>> list(gen_even_slices(10, 3))
    [slice(0, 4, None), slice(4, 7, None), slice(7, 10, None)]
    """
    start = 0
    for pack_num in range(n_packs):
        this_n = n // n_packs
        if pack_num < n % n_packs:
            this_n += 1
        if this_n > 0:
            end = start + this_n
            if n_samples is not None:
                end = min(n_samples, end)
            yield slice(start, end, None)
            start = end