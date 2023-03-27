# -*- coding: utf-8 -*-

import classification as clf
import step_algorithm as sa
from copy import deepcopy
import attack as atk
import numpy as np


##############################################################################

def algorithm(init_x, A_init, GOAL, c=None, step="consensus", atk_mode=None, nb_iter=100, atkd_node=0):
    x_step = deepcopy(init_x)
    W = deepcopy(A_init)
    X = {i: [x_step[i]] for i in range(len(init_x))}
    hist = {i: [] for i in range(len(init_x))}

    #goal = sa.estimate_consensus(init_x, A_init)

    rep = {i: np.ones(len(init_x)) for i in range(len(init_x))}

    if step == "dextra":
        x = np.zeros((nb_iter + 1, len(init_x)))
        y = np.zeros((nb_iter + 1, len(init_x)))
        z = np.zeros((nb_iter + 1, len(init_x)))

        x[0] = deepcopy(init_x)
        y[0] = np.ones(len(init_x))

        gradient = lambda x: 2 * x - 2 * GOAL

    for k in range(nb_iter):
        
        if c != 'simple':
            [W, hist, rep] = clf.classification(X=x_step, A=A_init, Classifier=c, H=hist, reputation=rep)

        if step == "consensus":
            x_step = sa.consensus_step(x_step, W)

        elif step == "dextra":
            x, y, z = sa.dextra_step(k, x, y, z, W, gradient)
            x_step = x[k]

        x_step = atk.mode(x_step, atk_mode, atkd_node, k)

        for i in range(len(x_step)):
            X[i].append(x_step[i])

    return X
