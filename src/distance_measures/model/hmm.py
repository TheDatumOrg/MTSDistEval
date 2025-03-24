import numpy as np
from hmmlearn import hmm
import math
from scipy import linalg
from hmmlearn import _hmmc
from src.distance_measures.dist import distance_matrix1d, distance_matrix

def fit_hmm(X, tries=3, n_states=2, covariance_type="full", n_iter=20, tol=0.1):
    """
    Fit a HMM to the data X.

    Parameters
    ----------
    X : np.ndarray, shape=(n_dimensions, n_samples)
        The data to fit the HMM to.
    tries : int
        The number of times to try to fit the model.
    n_states : int
        The number of states of the HMM.
    covariance_type : str
        The type of covariance matrix to use.
    n_iter : int
        The number of iterations to fit the model.
    tol : float
        The tolerance of the model.
    """

    s = tries
    max_likelihood = -np.inf
    best_model = None
    for i in range(s):
        model = hmm.GaussianHMM(n_components=n_states, covariance_type=covariance_type, n_iter=n_iter, tol=tol)
        model.n_features = X.shape[0]

        # Fit the model
        model.fit(X.T)

        # Get the likelihood of the data given the model
        log_likelihood = np.array(model.monitor_.history)[-1]

        # Save the model if it improved the likelihood
        if log_likelihood > max_likelihood:
            max_likelihood = log_likelihood
            best_model = model

    return best_model

def fit_hmms(dataset, **kwargs):
    return [fit_hmm(X, **kwargs) for X in dataset]

def _log_multivariate_normal_density_full(X, means, covars, min_covar=1.e-7):
    """
    Log probability for full covariance matrices.
    !!! X possibly has less dimensions than the model was trained on.
    """

    # Map both the model and the data to the same number of dimensions
    nf = min(X.shape[1], means.shape[1])
    X = X[:, :nf]
    local_means = means[:, :nf]
    local_covars = covars[:, :nf, :nf]

    log_prob = []
    for c, (mu, cv) in enumerate(zip(local_means, local_covars)):
        try:
            cv_chol = linalg.cholesky(cv, lower=True)
        except linalg.LinAlgError:
            # The model is most probably stuck in a component with too
            # few observations, we need to reinitialize this components
            try:
                cv_chol = linalg.cholesky(cv + min_covar * np.eye(nf),
                                          lower=True)
            except linalg.LinAlgError:
                raise ValueError("'covars' must be symmetric, "
                                 "positive-definite")

        cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
        cv_sol = linalg.solve_triangular(cv_chol, (X - mu).T, lower=True).T
        log_prob.append(-.5 * (nf * np.log(2 * np.pi)
                               + (cv_sol ** 2).sum(axis=1)
                               + cv_log_det))

    return np.transpose(log_prob)

def _gaussian_norm_factor(cov):
    k = cov.shape[0]
    return (2 * math.pi)**(-k/2) * np.linalg.det(cov)**(-0.5)

def _get_subset_norm_factor(cov, k):
    k_factor = _gaussian_norm_factor(cov[:k,:k])
    full_factor = _gaussian_norm_factor(cov)
    return math.log(full_factor / k_factor)

def _score_log(model, X):
        """
        Compute the log probability under the model, as well as posteriors if
        *compute_posteriors* is True (otherwise, an empty array is returned
        for the latter).
        Arguments:
        ---------
        model: hmmlearn.hmm.GaussianHMM
            The HMM to score.
        X : array-like, shape (n_dimensions, n_samples)
            Feature matrix of individual samples.

        """
        log_frameprob = _log_multivariate_normal_density_full(X.T, model.means_, model.covars_)
        k = X.shape[0]

        # Compute the normalization factor for the subset of dimensions
        if k < model.n_features:
            k_norm_factor = _get_subset_norm_factor(model.covars_[0], k)

            # Add the normalization factor to the log probabilities
            log_frameprob += k_norm_factor

        log_prob, fwdlattice = _hmmc.forward_log(
            model.startprob_, model.transmat_, log_frameprob)
        return log_prob

def _score_hmms(models,R, normalize=True):
    """
    Compute the scores (log likelihoods) of HMMs on a dataset.

    Parameters
    ----------
    models : List[hmmlearn.hmm.GaussianHMM]
        The HMMs to score.
    R : np.ndarray, shape=(n_mts, n_dimensions, n_samples)
        The reference dataset to score the HMMs on.
    """

    # Shape of the scores array: (n_models, n_mts)
    scores = np.array([[_score_log(model, X) for X in R] for model in models])

    # Normalize by dividing by the log-likelihoods by the maximum log-likelihood of each model
    if normalize:
        # scores /= np.abs(np.diag(scores))[:, None]
        scores /= np.max(scores, axis=1)[:, None]
    
    probs = np.clip(np.exp(scores), 1e-100, 1e20)
    probs /= np.sum(probs, axis=1, keepdims=True)
    return probs

def get_probs_all(X,R, normalize = False, **kwargs):
    """
    Fit a HMM to the dataset and return the probabilities of each MTS belonging to each state.

    Parameters
    ----------
    X : np.ndarray, shape=(n_mts, n_dimensions, n_samples)
        The train dataset of MTS, used to fit the HMMs.
    R : np.ndarray, shape=(n_mts, n_dimensions, n_samples)
        Reference dataset of MTS, used to score the HMMs.
    n_states : int
        The number of states of the HMM.
    covariance_type : str
        The type of covariance matrix to use.
    n_iter : int
        The number of iterations to fit the model.
    tol : float
        The tolerance of the model.
    """
    models = fit_hmms(X, **kwargs)

    # Get the probabilities of each MTS belonging to each state
    return _score_hmms(models, R, normalize=normalize)

def compare_hmms(H1_probs: np.ndarray, H2_probs: np.ndarray):
    """
    Compare two HMMs by computing the Kullback leibner divergence on their estimated probability densities, following the approach of [Ghassempour13].

    Parameters
    ----------
    H1_scores : np.ndarray
        The scores of the first HMM.
    H2_scores : np.ndarray
        The scores of the second HMM.
    """
    # Compute DKL(H1||H2) using the precomputed scores
    dkl_1 = np.sum(H1_probs * np.log(H1_probs / H2_probs))

    # Compute DKL(H2||H1) using the precomputed scores
    dkl_2 = np.sum(H2_probs * np.log(H2_probs / H1_probs))

    # Compute the average DKL
    dkl = (dkl_1 + dkl_2) / 2

    return dkl

def hmmdiv_d_all(X,Y, normalize_probs = True, adaptive_scaling=False, **kwargs):
    """
    Compute the divergence between all pairs of HMMs in X and Y.

    Parameters
    ----------
    X : np.ndarray, shape=(n_mts, n_dimensions, n_samples)
        The train dataset of MTS.
    Y : np.ndarray, shape=(n_mts, n_dimensions, n_samples)
        The test dataset of MTS.
    normalize_probs: bool. 
        Whether to normalize the probabilities of the HMMs.
    n_states : int
        The number of states of the HMM.
    covariance_type : str
        The type of covariance matrix to use.
    n_iter : int
        The number of iterations to fit the model.
    tol : float
        The tolerance of the model.
    """

    probsX = get_probs_all(X, R=X, normalize=normalize_probs, **kwargs)
    probsY = get_probs_all(Y, R=X, normalize=normalize_probs, **kwargs)
    return distance_matrix(probsX, probsY, compare_hmms,{})

def hmmdiv_i_all(X, Y, normalize_probs = True, adaptive_scaling=False, **kwargs):
    """
    Compute the divergence between all pairs of HMMs in X and Y.

    Parameters
    ----------
    X : np.ndarray, shape=(n_mts, n_dimensions, n_samples)
        The train dataset of MTS.
    Y : np.ndarray, shape=(n_mts, n_dimensions, n_samples)
        The test dataset of MTS.
    normalize_probs: bool. 
        Whether to normalize the probabilities of the HMMs.
    n_states : int
        The number of states of the HMM.
    covariance_type : str
        The type of covariance matrix to use.
    n_iter : int
        The number of iterations to fit the model.
    tol : float
        The tolerance of the model.
    """
    
    distmat = np.zeros((X.shape[0],Y.shape[0]))
    n,c,t = X.shape
    for i in range(c):
        Xc = X[:,i,:][:, np.newaxis, :]
        Yc = Y[:,i,:][:, np.newaxis, :]

        probsX = get_probs_all(Xc, R=Xc, normalize=normalize_probs, **kwargs)
        probsY = get_probs_all(Yc, R=Xc, normalize=normalize_probs, **kwargs)
        distmat += distance_matrix(probsX, probsY, compare_hmms, {})
    return distmat