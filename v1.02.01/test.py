# -*- coding: utf-8 -*-


from copy import deepcopy
import step_algorithm as sa
import numpy as np
from matplotlib import pyplot as plt
import essentials_v2 as ex

LOOP = 200

# DEBUG VALUES #######################################
x_0 = np.array([10, 9, 9, 8, 7], dtype=float)
A = np.array([
    [1, 0, 1, 1, 0],
    [1, 1, 0, 0, 1],
    [0, 1, 1, 0, 1],
    [1, 0, 1, 1, 0],
    [1, 1, 0, 0, 1]
], dtype=float)
######################################################

N = 5
TARGET = 0

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
    Rt_norm = [(r - min(Rt)) / max(Rt) for r in Rt]
    # Rt_bound = [r/max(Rt) for r in Rt]
    # TODO: save Rt pre & post normalization in a file
    return Rt_norm, H


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


def reputation_step(V, reputation, IDs, **kwargs):
    Rpt = [1 / r if r > 0.01 else 100 for r in rateAllRep(V, V, IDs, reputation)]
    Rpt_norm = [(r - min(Rpt)) / max(Rpt) for r in Rpt]
    #Rpt_bound = [r / max(Rpt) for r in Rpt]
    return Rpt_norm
##############################################################################

## Testing loop
x = [deepcopy(x_0)]
W = deepcopy(A)
for k in range(LOOP):
    
    for ni, node in enumerate(A):
        rec = [xi for xi, n in zip(x[-1],node) if n > 0]
        IDs = [i for i,n in enumerate(node) if n!=0]
        rpt = reputation_step(rec, W[ni], IDs)
        #W[ni] = [rpt[i]  if n>0 else n for i, n in enumerate(node)]
        
        for k, j in enumerate(IDs):
            W[ni][j] = rpt[k]
        W[ni][ni] = 1
        
    x.append(sa.consensus_step(x[-1], W))

for xn in np.transpose(x):
    plt.plot(xn)
goal = sa.estimate_consensus(x_0, ex.get_weight_matrix(A))*np.ones(LOOP)
plt.plot(goal)
plt.legend(np.concatenate([[f'node {n+1}' for n in range(len(W))], ["goal"]]))
plt.show()
    