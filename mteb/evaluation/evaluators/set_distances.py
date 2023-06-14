import logging

import numpy as np
from sklearn.metrics.pairwise import paired_cosine_distances, paired_distances, pairwise_distances
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

from corrsim.similarity import get_similarity_by_name

logger = logging.getLogger(__name__)

from .Evaluator import Evaluator


def _reshape_matrix_list(matrices, max_n=None):
    """
   `matrices` is an N-length list of matrices of shape (n_i, d) for i=1,...,N

    We want to collect these into two arrays of shape (N, d*max(n_i)), zero-padded
    """
    # first get the number of rows in each matrix
    n_i = np.array([m.shape[0] for m in matrices])

    max_n = max_n or n_i.max()
    dn = matrices[0].shape[1] * max_n

    # "pack" the lists into matrices of shape (N, d*max(n_i)), zero-padded
    M = np.array([np.pad(m.flatten(), (0, dn - m.size)) for m in matrices])

    # store the original lengths as the first column in each array
    M = np.hstack([n_i.reshape(-1, 1), M])

    return M


def paired_distances_vector_set(e1, e2, d, metric):
    """
    Compute the `metric` between two sets of vectors, where each set is a list of matrices of shape (n_i, d)
    """
    assert isinstance(e1, np.ndarray) and isinstance(e2, np.ndarray), "e1 and e2 must be numpy arrays, reshape them first"
    # the `metric` is designed to work over two matrices of shape (n_i, d),
    # so now we "unpack" the matrices
    def _wrapped_metric(v1, v2):
        n1 = int(v1[0])
        n2 = int(v2[0])
        m1 = v1[1:n1*d+1].reshape(n1, d)
        m2 = v2[1:n2*d+1].reshape(n2, d)
        return metric(m1, m2)

    return paired_distances(e1, e2, metric=_wrapped_metric)


def _cross_prod_cosine(m1, m2, agg_fn):
    return agg_fn(pairwise_distances(m1, m2, metric="cosine"))


def cross_prod_cosine_q10(m1, m2):
    return _cross_prod_cosine(m1, m2, lambda x: np.quantile(x, 0.1))


def cross_prod_cosine_min(m1, m2):
    return _cross_prod_cosine(m1, m2, np.min)


def compute_all_set_distances(embeddings1, embeddings2):

    logger.info("Evaluating...")
    # means
    logger.info("Calculating mean set dists")
    embeddings_mean1 = np.array([np.mean(embedding, axis=0) for embedding in embeddings1])
    embeddings_mean2 = np.array([np.mean(embedding, axis=0) for embedding in embeddings2])

    mean_cosine_scores = 1 - (paired_cosine_distances(embeddings_mean1, embeddings_mean2))
    mean_pearson_scores = paired_distances(embeddings_mean1, embeddings_mean2, metric=lambda v1, v2: pearsonr(v1, v2)[0])
    mean_spearman_scores = paired_distances(embeddings_mean1, embeddings_mean2, metric=lambda v1, v2: spearmanr(v1, v2)[0])

    # max
    logger.info("Calculating max set dists")
    embeddings_max1 = np.array([np.max(embedding, axis=0) for embedding in embeddings1])
    embeddings_max2 = np.array([np.max(embedding, axis=0) for embedding in embeddings2])

    max_cosine_scores = 1 - (paired_cosine_distances(embeddings_max1, embeddings_max2))
    max_pearson_scores = paired_distances(embeddings_max1, embeddings_max2, metric=lambda v1, v2: pearsonr(v1, v2)[0])
    max_spearman_scores = paired_distances(embeddings_max1, embeddings_max2, metric=lambda v1, v2: spearmanr(v1, v2)[0])

    # sets
    d = embeddings1[0].shape[1]
    max_n = max([m.shape[0] for e in [embeddings1, embeddings2] for m in e])
    embeddings1 = _reshape_matrix_list(embeddings1, max_n=max_n)
    embeddings2 = _reshape_matrix_list(embeddings2, max_n=max_n)

    # cross-products
    logger.info("Calculating aggregated pairwise set dists")
    cdist_cosine_q10 = 1 - paired_distances_vector_set(embeddings1, embeddings2, d, metric=cross_prod_cosine_q10)
    cdist_cosine_min = 1 - paired_distances_vector_set(embeddings1, embeddings2, d, metric=cross_prod_cosine_min)

    # from Zhelezniak 2019,2020 (https://github.com/babylonhealth/corrsim/)
    logger.info("Calculating MI set dists")
    setdist_ksg3 = paired_distances_vector_set(embeddings1, embeddings2, d, metric=get_similarity_by_name("ksg3"))
    setdist_ksg10 = paired_distances_vector_set(embeddings1, embeddings2, d, metric=get_similarity_by_name("ksg10"))
    setdist_mean_ksg10 = paired_distances_vector_set(embeddings1, embeddings2, d, metric=get_similarity_by_name("mean_ksg10"))
    setdist_max_ksg10 = paired_distances_vector_set(embeddings1, embeddings2, d, metric=get_similarity_by_name("max_ksg10"))

    logger.info("Calculating CKA set dists")
    setdist_cka_linear = paired_distances_vector_set(embeddings1, embeddings2, d, metric=get_similarity_by_name("cka_linear"))
    setdist_cka_gaussian = paired_distances_vector_set(embeddings1, embeddings2, d, metric=get_similarity_by_name("cka_gaussian"))
    setdist_cka_dcorr = paired_distances_vector_set(embeddings1, embeddings2, d, metric=get_similarity_by_name("cka_dcorr"))

    return {
        "mean_cosine_scores": mean_cosine_scores,
        "mean_pearson_scores": mean_pearson_scores,
        "mean_spearman_scores": mean_spearman_scores,
        "max_cosine_scores": max_cosine_scores,
        "max_pearson_scores": max_pearson_scores,
        "max_spearman_scores": max_spearman_scores,
        "cdist_cosine_q10": cdist_cosine_q10,
        "cdist_cosine_min": cdist_cosine_min,
        "setdist_ksg3": setdist_ksg3,
        "setdist_ksg10": setdist_ksg10,
        "setdist_mean_ksg10": setdist_mean_ksg10,
        "setdist_max_ksg10": setdist_max_ksg10,
        "setdist_cka_linear": setdist_cka_linear,
        "setdist_cka_gaussian": setdist_cka_gaussian,
        "setdist_cka_dcorr": setdist_cka_dcorr,
    }
