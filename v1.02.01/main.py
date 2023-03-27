# -*- coding: utf-8 -*-

import itertools
from copy import deepcopy

import essentials_v2 as ex
import file_manager as fm
import linear_A1 as LA1
# from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

##############################################################################

LOOP = 200

# DEBUG VALUES #######################################
# x_0 = np.array([10, 9, 9, 8, 7], dtype=float)
# A = np.array([
#     [1, 0, 1, 1, 0],
#     [1, 1, 0, 0, 1],
#     [0, 1, 1, 0, 1],
#     [1, 0, 1, 1, 0],
#     [1, 1, 0, 0, 1]
# ], dtype=float)
######################################################

N = 5
TARGET = 0

C = ["simple", LA1.clf.mcd_step, LA1.clf.ocsvm_step, LA1.clf.rating_step, LA1.clf.reputation_step]
ATK = ["clean", "single", "persistent"]
Consensus = ["consensus", "dextra"]

##############################################################################

c_str_list = [str(cl).split()[1] if callable(cl) else cl for cl in C]

# atk_mode = {ax: [] for ax in ATK}
# consensus_result = {cx: atk_mode for cx in Consensus}
# collection = {c: consensus_result for c in c_str_list}

# DF = pd.DataFrame(index=np.concatenate([ATK, ['persistent without attacker']]), columns=Consensus)
DF = pd.DataFrame(index=ATK, columns=Consensus)
result = {c: deepcopy(DF) for c in c_str_list}
# last_value = {c: deepcopy(DF) for c in c_str_list}
# last_value_avg = {c: deepcopy(DF) for c in c_str_list}
# diff_to_goal = {c: deepcopy(DF) for c in c_str_list}

##############################################################################
collectMode = False
#collectMode = True

if collectMode:
    x_0 = np.round(np.random.rand(N) * 10, 2)
    A = ex.strongly_connected_matrix(N, TARGET)
    goal = LA1.sa.estimate_consensus(x_0, A)

    fm.save_starting_state_and_network(x_0, A, "results", "csv")

    for cl, cx, a in itertools.product(C, Consensus, ATK):
        c_str = str(cl).split()[1] if callable(cl) else cl
        result[c_str][cx][a] = LA1.algorithm(x_0, A, GOAL = goal, nb_iter=LOOP, c=cl, step=cx, atk_mode=a, atkd_node=TARGET)
        file_path = fm.new_filepath(c_str, folder_key=f"results\\{cx}\\{a}", filetype="csv")
        fm.Write_to_File(file_path, pd.DataFrame(result[c_str][cx][a]))

else:
    # [A, x_0] = fm.get_files("results", "net_goal").values.tolist()
    state_net = fm.get_files("results", "net_goal", "last")
    A = np.array(state_net.values.tolist()[:N])
    x_0 = state_net.values.tolist()[-1]
    goal = LA1.sa.estimate_consensus(x_0, A)
    for cl, cx, a in itertools.product(c_str_list, Consensus, ATK):
        # print(f"results\\{cx}\\{a}\\{cl}")
        result[cl][cx][a] = fm.get_files(f"results\\{cx}\\{a}", cl, "last").to_dict(orient="list")
        # print(result[cl][cx][a])

##############################################################################

print(A)
print("\n")
print(x_0)
print("\n")
print(goal)

n = 0
errors = {cx: {a: {}} for cx, a in itertools.product(Consensus, ATK)}
goal_diff = {cx: {a: {}} for cx, a in itertools.product(Consensus, ATK)}

##############################################################################

for cx, a, c_str in itertools.product(Consensus, ATK, c_str_list):
    title = f'Consensus Function: {cx} | Attack Mode: {a} | Classification: {c_str}\n'
    ex.plot_iter(result[c_str][cx][a], goal, title, LOOP)
    n += 1
    if n >= len(c_str_list):
        # errors[cx][a] = ex.show_error(result, cx, a, c_str_list, goal, LOOP, TARGET)
        errors[cx][a] = ex.max_difference_evo(result, cx, a, c_str_list, goal, LOOP, TARGET)
        goal_diff[cx][a] = ex.max_goal_difference_evo(result, cx, a, c_str_list, goal, LOOP, TARGET)
        n = 0

##############################################################################
#     last_value[c_str][cx][a] = [res[r][-1] for r in res]
#     last_value_avg[c_str][cx][a] = np.mean([res[r][-1] for r in res])
#     diff_to_goal[c_str][cx][a] = np.abs(last_value_avg[c_str][cx][a]-goal)

#     if a == "persistent":
#         last_value_avg[c_str][cx]['persistent without attacker'] = np.mean(
#             [res[i][-1] for i,r in enumerate(result) if i != TARGET])

#         diff_to_goal[c_str][cx]['persistent without attacker'] = np.abs(
#             last_value_avg[c_str][cx]['persistent without attacker']-goal)

# ##############################################################################       
# for key in last_value:
#     print('\n\n\t\t ' + key)
#     print(last_value[key])
#     print('\n\n\t Average Result on Last Node')
#     print(last_value_avg[key])
#     print(f'\n\n\t Difference to Established Goal ({goal})')
#     print(diff_to_goal[key])
