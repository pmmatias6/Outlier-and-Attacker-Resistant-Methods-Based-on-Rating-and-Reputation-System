# -*- coding: utf-8 -*-

import numpy as np
import random

##############################################################################
ATK_VAL = 50


def mode(x: np.array, name: str, node: int, index: int):
    """

    :param x: 1-D np.array
    :param name: attack name from list of possible names, if name is not any of the known types, x is returned unchanged
    :param node: node to be attacked in persistent or single named attacks.
    :param index: index of the current iteration of the algorithm, required to identify the moment to perform the single attack
    :return: x
    """
    # An attack that propagates continuously in time, without caring for the system's state
    if name == "persistent" or (name == "single" and index == 10):
        x[node] = ATK_VAL

    elif name == "random":
        node = np.random.choice(np.arange(len(x)))
        x[node] = random.randrange(min(x), max(x))

    elif name == "copy":
        A = np.arange(len(x))
        n1 = np.random.choice(A)
        A = np.delete(A, n1)
        n2 = np.random.choice(A)
        x[n1] = x[n2]
    return x
