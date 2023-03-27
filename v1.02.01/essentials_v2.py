# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as m
from matplotlib import pyplot as plt
import itertools


# import pandas as pd
# import os

##############################################################################

def get_weight_matrix(node_matrix: np.ndarray) -> np.ndarray:
    return (node_matrix.T / node_matrix.T.sum(axis=0)).T


##############################################################################

# def strongly_connected_matrix(size: int, atkd_node: int = 0 ) -> np.ndarray:
#     matrix = ((np.add((np.random.random((size, size)) <= 1 / 3) * 1, np.eye(size))) >= 1) * 1.0
#     while np.any(m.matrix_power(matrix, size) == 0):
#         matrix = ((np.add((np.random.random((size, size)) <= 1 / 3) * 1, matrix)) >= 1) + 0.0
#     inner_matrix = np.delete(matrix, atkd_node,axis=0)
#     inner_matrix = np.delete(inner_matrix, atkd_node,axis=1)
#     while np.any(m.matrix_power(inner_matrix, size) == 0):
#         inner_matrix = ((np.add((np.random.random((size, size)) <= 1 / 3) * 1, inner_matrix)) >= 1) + 0.0
#     return matrix

def strongly_connected_matrix(size: int, atk_node: int = 0) -> np.ndarray:
    matrix = np.eye(size)
    while np.any(m.matrix_power(matrix, size) == 0):
        matrix = ((np.add((np.random.random((size, size)) <= 1 / 3) * 1, matrix)) >= 1) + 0
    inner_matrix = np.delete(matrix, atk_node, axis=0)
    inner_matrix = np.delete(inner_matrix, atk_node, axis=1)
    # print(inner_matrix)
    while np.any(m.matrix_power(inner_matrix, size) == 0):
        inner_matrix = ((np.add((np.random.random((size - 1, size - 1)) <= 1 / 3) * 1, inner_matrix)) >= 1) + 0

    for k, n in enumerate(matrix):
        for i, _ in enumerate(n):
            if k != atk_node and i != atk_node:
                if k == 0:
                    matrix[k][i] = inner_matrix[k][i] if i == 0 else inner_matrix[k][i - 1]
                else:
                    matrix[k][i] = inner_matrix[k - 1][i - 1]
    return matrix


##############################################################################

def plot_iter(result, goal, title, LOOP):
    lg = []
    fig, ax = plt.subplots()
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for key in result:
        # print(key, result[key])
        ax.plot(result[key])
        lg.append(f"node {int(key) + 1}")  #: {np.round(result[key][-1],5)}")
    ax.plot(np.linspace(0, LOOP, num=10), goal * np.ones(10), 'k*')
    lg.append("goal")
    ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.2, alpha=0.5)
    plt.legend(lg, bbox_to_anchor=(1, 1.03))
    plt.title(title)
    #plt.xscale("log")
    #plt.yscale("log")
    ax.set_ylabel("state values")
    ax.set_xlabel("iteration")
    plt.show()


##############################################################################

def max_difference_evo(dataset: dict, cx_function, atk_mode, classifier_list, objective_val, Length, atkd_node=None):
    fig, ax = plt.subplots()
    lg = []
    error_list = {}
    mx_h = 50
    for cl_str in classifier_list:
        ds = dataset[cl_str][cx_function][atk_mode]
        error = []
        for k in range(Length):
            x = (
                [ds[key][k] for key in ds if int(key) != atkd_node]
                if atk_mode != "clean"
                else [ds[key][k] for key in ds if int(key)]
            )
            diffs = [np.abs(xi - xj) for xi, xj in itertools.product(x, x)]
            error.append(max(diffs))
        if error[-1] < mx_h:
            mx_h = error[-1]+3
        ax.plot(error)
        lg.append(cl_str)
        error_list[cl_str] = error
    plt.legend(lg, bbox_to_anchor=(1, 1.03))
    #plt.yscale("log")
    # plt.xscale("log")
    
     # grid
    ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.2, alpha=0.5)

    # spine removal
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
        
    plt.ylim([-1,mx_h])
    plt.title(
        f'Maximum Difference between non-attacked nodes\nConsensus Function: {cx_function} | Attack Mode: {atk_mode}\n')
    plt.show()
    return error_list

##############################################################################

def max_goal_difference_evo(dataset: dict, cx_function, atk_mode, classifier_list, objective_val, Length,
                            atkd_node=None):
    fig, ax = plt.subplots()
    lg = []
    error_list = {}
    mx_h = 50
    for cl_str in classifier_list:
        ds = dataset[cl_str][cx_function][atk_mode]
        error = []
        for k in range(Length):
            if atk_mode != "clean":
                x = [ds[key][k] for key in ds if int(key) != atkd_node]
            else:
                x = [ds[key][k] for key in ds if int(key)]

            error.append(abs(np.mean(x) - objective_val))
        if error[-1] < mx_h:
            mx_h = error[-1]+3
        ax.plot(error)
        lg.append(cl_str)
        error_list[cl_str] = error
    plt.legend(lg, bbox_to_anchor=(1, 1.03))
    plt.ylim([-1,mx_h])
    #plt.yscale("log")
    # plt.xscale("log")
     # grid
    ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.2, alpha=0.5)

    # spine removal
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
        
    plt.title(f'Distance to Expected Value\nConsensus Function: {cx_function} | Attack Mode: {atk_mode}\n')
    plt.show()
    return error_list


##############################################################################

def plot_last_node_comp(data, goal, title, atkd_node):
    fig, ax = plt.subplots()
    nodes = {key: abs(np.mean(data[key]) - goal) for key in data}
    methods = list(nodes.keys())
    values = list(nodes.values())

    ax.bar(methods, values, width=0.5)

    # grid
    ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.2, alpha=0.5)
    
    # spine removal
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)

        # difference value annotations
    for i in ax.patches:
        plt.text(i.get_x(), i.get_height() + 0.01 * i.get_height(), str(round(i.get_height(), 6)))

    ax.set_xlabel("Attack Mode")
    ax.set_ylabel(f"Distance to Goal ({np.round(goal, 5)})")
    ax.set_title(title)
    plt.show()

##############################################################################

def plot_differences(collection, N, TARGET, goal):
    for key1 in collection:
        fig, axs = plt.subplots(len(collection[key1].keys()), 1, sharey=True, sharex=True, constrained_layout=True)
        fig.suptitle(f'{key1}: Distance to goal comparison', fontsize=14)
        for k, key2 in enumerate(collection[key1]):
            values = []
            nodes = []
            for key3 in collection[key1][key2]:
                if key3 == "persistent" and TARGET:
                    dtx = [collection[key1][key2][key3][i][-1] for i in range(N) if i != TARGET]
                else:
                    dtx = [collection[key1][key2][key3][i][-1] for i in range(N)]
                values.append(np.abs(np.mean(dtx) - goal))
                nodes.append(key3)

            axs[k].bar(nodes, values, width=0.5)
            axs[k].set_title(f'using {key2}')
            axs[k].grid(b=True, color='grey', linestyle='-.', linewidth=0.2, alpha=0.5)

            # spine removal
            for s in ['top', 'right']:
                axs[k].spines[s].set_visible(False)

                # difference value annotations
            for i in axs[k].patches:
                plt.text(i.get_x(), i.get_height() + 0.01 * i.get_height(), str(round(i.get_height(), 6)))

            axs[k].set_xlabel("Attack Mode")
            # axs[k].set_ylabel(f"Distance to Goal ({np.round(goal, 5)})")
        plt.show()
