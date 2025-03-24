from sklearn.decomposition import PCA
import numpy as np
import math
from src.distance_measures.dist import distance_matrix


"""
Wrapper for the PCA function from sklearn
A: an MTS with shape (C,T)
Return is the PCA object
"""
def pca(A):
    return PCA().fit(A.T)

def get_weights(dataset):
    """
    Gets the weights for Eros; the average eigenvalues of the covariance matrices of the whole dataset.

    Parameters
    ----------
    dataset : List or array of shape=(n_mts, n_dimensions, n_samples)
        The dataset to compute the weights for.
    """
    n, d, m = len(dataset), max([mts.shape[0] for mts in dataset]), max([mts.shape[-1] for mts in dataset])
    n_components = min(d, m)

    weights = np.zeros((n,n_components)) # shape=(n_mts, n_dimensions)
    exp_variance_ratios = np.zeros((n,n_components)) # shape=(n_mts, n_dimensions)
    pcas = np.zeros((n,n_components,d))

    for i,mts in enumerate(dataset):
        pca_obj = pca(mts)

        s = pca_obj.singular_values_
        Vt = pca_obj.components_ 
        exvar = pca_obj.explained_variance_ratio_

        nc, nf = Vt.shape

        # Only fill the relevant components, leave the rest as 0
        weights[i, :len(s)] = s
        nc = Vt.shape[0]
        pcas[i, :nc, :nf] = Vt
        exp_variance_ratios[i, :len(exvar)] = exvar

    # Average the weights and variances over the MTS
    weights = np.mean(weights, axis=0)
    exp_variance_ratios = np.median(exp_variance_ratios, axis=0)

    # Normalize the weights
    weights /= np.sum(weights)

    return weights, pcas, exp_variance_ratios

def pca_dist(pca1, pca2):
    return 1 / (1 + np.sum(np.square(pca1 @ pca2.T)))

def eros(pca1, pca2, w):
    """
    Compute the extended frobenius norm between the principal components of two MTS.
    The weights are based on the average eigenvalues of the covariance matrices of the whole dataset.
    """
    # Compute the weighted sum of dots of the eigenvectors
    s = 0
    for i in range(pca1.shape[0]):
        s += w[i] * np.abs(np.dot(pca1[i], pca2[i]))
    return s

def eros_dist(pca1, pca2, w):
    sim = min(eros(pca1, pca2, w), 1) # Set max similarity to 1
    return math.sqrt(2 - 2*sim)


def pca_all(X,Y, exvar=0.90, **kwargs):
    """
    Compute the distance matrix between the principal components of all MTS using the PCA similarity factor
    X: an MTS dataset with shape (n_mts, n_dimensions, n_samples)
    Y: an MTS dataset with shape (n_mts, n_dimensions, n_samples)
    exvar: the minimum explained variance to keep the components
    return: the distance matrix
    """
    # Add feature to support Json input. Json input is a dictionary and we need to adapt to it.
    if isinstance(exvar, dict):
        exvar = exvar['exvar']
    else: pass
    
    if exvar < 0 or exvar > 1:
        raise ValueError("exvar must be between 0 and 1")
    
    # Combine the datasets to compute the weights
    ALL = np.concatenate([X, Y], axis=0)

    # Get pcas
    _, pcas, exvar_ratios = get_weights(ALL)

    # Only keep the components that cover 90% of the variance
    n_components = np.argmax(np.cumsum(exvar_ratios) >= exvar) + 1
    pcas = pcas[:, :n_components, :n_components]

    # Get the X and Y pcas
    pcasX = pcas[:X.shape[0]]
    pcasY = pcas[X.shape[0]:]

    return distance_matrix(pcasX,pcasY, pca_dist)

def eros_all(X,Y, **kwargs):
    """
    Compute the distance matrix between the principal components of all MTS using Eros
    X: an MTS dataset with shape (n_mts, n_dimensions, n_samples)
    Y: an MTS dataset with shape (n_mts, n_dimensions, n_samples)
    return: the distance matrix
    """
    
    # Combine the datasets to compute the weights
    ALL = np.concatenate([X, Y], axis=0)

    # Get pcas
    weights, pcas, _ = get_weights(ALL)

    # Get the X and Y pcas
    pcasX = pcas[:X.shape[0]]
    pcasY = pcas[X.shape[0]:]

    # Compute the distance matrix. Updated by Haojun, the original distance_matrix does not support w=weigts.
    return distance_matrix(pcasX, pcasY, eros_dist, w=weights)
# Eros OK