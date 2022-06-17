from similarity_metrics import cka_core
from similarity_metrics.cca_core import get_cca_similarity
from similarity_metrics.pwcca import compute_pwcca

import numpy as np


def normalize(acts, axis):
    acts = acts - np.mean(acts, axis=axis, keepdims=True)
    acts = acts / np.linalg.norm(acts)
    return acts


def cca(acts1, acts2):
    acts1 = normalize(np.transpose(acts1), 1)
    acts2 = normalize(np.transpose(acts2), 1)
    cca_sim = get_cca_similarity(acts1, acts2, verbose=False)["cca_coef1"]
    return np.mean(cca_sim)


def pwcca(acts1, acts2):
    acts1 = normalize(np.transpose(acts1), 1)
    acts2 = normalize(np.transpose(acts2), 1)
    return compute_pwcca(acts1, acts2)[0]


def svcca(acts1, acts2, n_dims=20):
    acts1 = normalize(np.transpose(acts1), 1)
    acts2 = normalize(np.transpose(acts2), 1)

    U1, s1, V1 = np.linalg.svd(acts1, full_matrices=False)
    U2, s2, V2 = np.linalg.svd(acts2, full_matrices=False)

    svacts1 = np.dot(s1[:n_dims] * np.eye(n_dims), V1[:n_dims])
    svacts2 = np.dot(s2[:n_dims] * np.eye(n_dims), V2[:n_dims])

    return np.mean(get_cca_similarity(svacts1, svacts2, verbose=False)["cca_coef1"])


# TODO: Delete this function as its not being used
def cka(acts1, acts2):
    acts1 = normalize(acts1, 0)
    acts2 = normalize(acts2, 0)
    return cka_core.feature_space_linear_cka(acts1, acts2)


def gram_cka(acts1, acts2):
    acts1 = normalize(acts1, 0)
    acts2 = normalize(acts2, 0)
    acts1_gram = cka_core.gram_linear(acts1)
    acts2_gram = cka_core.gram_linear(acts2)
    return cka_core.cka(acts1_gram, acts2_gram)


def orthogonal_procrustes(acts1, acts2):
    acts1 = normalize(acts1, 0)
    acts2 = normalize(acts2, 0)
    n1 = np.power(np.linalg.norm(acts1, ord="fro"), 2)
    n2 = np.power(np.linalg.norm(acts2, ord="fro"), 2)
    n3 = 2 * np.linalg.norm(np.matmul(acts1.T, acts2), ord="nuc")
    return n1 + n2 - n3


def get_similarity_metric(metric_name):
    if metric_name == "cca":
        metric = cca
    elif metric_name == "pwcca":
        metric = pwcca
    elif metric_name == "svcca":
        metric = svcca
    elif metric_name == "cka":
        metric = cka
    elif metric_name == "gram_cka":
        metric = gram_cka
    elif metric_name == "procrustes":
        metric = orthogonal_procrustes
    return metric
