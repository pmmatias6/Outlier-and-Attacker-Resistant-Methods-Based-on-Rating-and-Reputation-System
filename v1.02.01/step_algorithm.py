# -*- coding: utf-8 -*-

import numpy as np
import essentials_v2 as e


##############################################################################

def estimate_consensus(vec: np.ndarray, weight_matrix: np.ndarray):
    w, v = np.linalg.eig(weight_matrix.T)
    v_norm = (v[:, 0] / sum(v[:, 0]))
    return np.round(np.real(np.dot(v_norm, vec)), 5)


##############################################################################

def consensus_step(vec: np.ndarray, node_matrix: np.ndarray) -> np.ndarray:
    weight_matrix = e.get_weight_matrix(node_matrix)
    return weight_matrix.dot(vec)


##############################################################################

def dextra_step(k: int, x: np.ndarray, y: np.ndarray, z: np.ndarray, node_matrix: np.ndarray, gradient,
                alpha: float = 0.1, theta: float = 0.1):
    n = len(node_matrix)
    weight_matrix = e.get_weight_matrix(node_matrix)
    tilde_matrix = theta * np.eye(n) + (1 - theta) * weight_matrix
    d = lambda power: np.diag(np.linalg.matrix_power(weight_matrix, power).dot(np.ones(len(weight_matrix))))

    z[k] = np.linalg.inv(d(k)).dot(x[k])
    if k == 0:
        x[k + 1] = weight_matrix.dot(x[k]) - alpha * gradient(z[k])
    else:
        x[k + 1] = x[k] + weight_matrix.dot(x[k]) - tilde_matrix.dot(x[k - 1]) - alpha * (
                gradient(z[k]) - gradient(z[k - 1]))
    y[k + 1] = weight_matrix.dot(y[k])
    return x, y, z
