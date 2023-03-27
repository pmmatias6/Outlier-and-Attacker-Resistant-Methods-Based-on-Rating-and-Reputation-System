# -*- coding: utf-8 -*-

from copy import deepcopy
from types import FunctionType

import essentials_v2 as ex
import numpy as np
from sklearn.covariance import EllipticEnvelope as ee
from sklearn.svm import OneClassSVM

##############################################################################

'''
OCSVM:
    - Description: Algorithm receives a training and a testing set, trains
    over the training set and predicts the outlying degree of the testing set
    according to the One Class Support Vector Machines Algorithm
    - INPUTS:
        - V: The testing set of size N
        - H: The training set
        - Q: maximum size of the H array to be maintained in memory
    - OUTPUTS:
        - array: numeric binary array of size N with outliers marked as 0
'''


def ocsvm_step(V: list, H: list, Q: int, **kwargs):
    if len(H) > Q:
        H = H[-Q:]
    clf = OneClassSVM(gamma='scale').fit(np.array(H).reshape((-1, 1)))
    result = np.array([max(r, 0) for r in clf.predict(np.array(V).reshape(-1, 1))])
    return result, H


##############################################################################

'''
MCD:
    - Description: Algorithm receives a training and a testing set, trains
    over the training set and predicts the outlying degree of the testing set
    according to the Minimum Covariance Determinant Algorithm
    - INPUTS:
        - input_vector: The testing set of size N
        - history: The training set
    - OUTPUTS:
        - array: numeric binary array of size N with outliers marked as 0
'''


def mcd_step(V: list, H: list, Q: int, **kwargs):
    if len(H) > Q:
        H = H[-Q:]
    try:
        mcd = ee(random_state=0).fit(np.array(H).reshape((-1, 1)))
        acc = mcd.predict(np.array(V).reshape((-1, 1)))
    except ValueError:
        return np.array([1 for _ in V]), H
    return np.array([max(a, 0) for a in acc]), H


##############################################################################

'''
RATING:
    - Description:
    - INPUTS:
    - OUTPUTS:
'''


def rating(v, H):
    return sum(np.linalg.norm([v - h]) for h in H)


def rateAll(V, H):
    return [rating(v, H) for v in V]


def rating_step(V, H, **kwargs):
    Rt = [1 / r if r > 0.01 else 100 for r in rateAll(V, V)]
    #Rt_norm = [(r - min(Rt)) / max(Rt) for r in Rt]
    Rt_bound = [r/max(Rt) for r in Rt]
    # TODO: save Rt pre & post normalization in a file
    return Rt_bound, H#_norm, H


##############################################################################


'''
REPUTATION:
    - Description:
    - INPUTS:
    - OUTPUTS:
'''


def ratingRep(v, history, IDs, rep):
    return sum(
        rep[IDs[i]] * np.linalg.norm([v - h]) for i, h in enumerate(history)
    )


def rateAllRep(V, H, IDs, rep):
    return [ratingRep(v, H, IDs, rep) for v in V]


def reputation_step(V, reputation, IDs, H, **kwargs):
    Rpt = [1 / r if r > 0.01 else 100 for r in rateAllRep(V, V, IDs, reputation)]
   # Rpt_norm = [(r - min(Rpt)) / max(Rpt) for r in Rpt]
    Rpt_bound = [r / max(Rpt) for r in Rpt]
    return Rpt_bound, H#_norm, H


##############################################################################

'''
CLASSIFICATION:
    - Description: Algorithm receives an array X, a dictionary H*, and a 
    node matrix A, then calls a function to classify received values as 
    inliers or outliers. Its output is the node matrix W, which is the altered 
    weighed version of matrix A.
    - INPUTS:
        - X: list or array of values of size N
        - H: dictionary of previous values accepted in previous iterations
        - A: numerical binary matrix of size NxN
        - Classifier: function to classify the set or None if no classification
        is provided.
    - OUTPUTS:
        - W: weighed matrix of A, after classification if it was provided
'''


def classification(X: np.array, H: dict, A: np.ndarray, *, reputation: dict,
                   Classifier: FunctionType, Q_size: int = 20) -> object:
    W = deepcopy(A)
    for i, node in enumerate(A):
        rec = np.array([X[k] for k, n in enumerate(node) if n > 0])
        IDs = [k for k, n in enumerate(node) if n > 0]
        hist = np.concatenate([H[i], rec])
        F, H[i] = Classifier(V=rec, H=hist, Q=Q_size, reputation=reputation[i], IDs=IDs)
        for k, j in enumerate(IDs):
            W[i][j] = F[k]
        W[i][i] = 1
        reputation[i] = dict(enumerate(W[i]))
   # print(W)
   # W = ex.get_weight_matrix(W)
   # print(W)
    return W, H, reputation
