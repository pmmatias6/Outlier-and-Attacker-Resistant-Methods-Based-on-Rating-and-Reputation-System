# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np

def new_filepath(name, folder_key=None, d_path=os.getcwd(), filetype="", nb=0):
    fileExists = True
    if folder_key:
        d_path = get_folder(folder_key)

    while fileExists:
        f_path = os.path.join(d_path, f"{str(nb)}-{name}.{filetype}")
        if os.path.isfile(f_path):
            nb += 1
        else:
            return f_path


def get_folder(name):
    dir_path = os.path.join(os.getcwd(), name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path   


def Write_to_File(f_name: str, data):
    if f_name.split(".")[-1] == "csv":
        data.to_csv(f_name, index=False)
    else:
        with open(f_name, 'w') as f:
            f.write(data)


def save_starting_state_and_network(x, network, fk, ft):
    n = list(network)
    n.append(x)
    df = pd.DataFrame(np.array(n))
    fp = new_filepath("net_goal", folder_key=fk, filetype=ft)
    Write_to_File(fp, df)


def Read_from_File(f_name):
    return pd.read_csv(f_name)


def check_dir(dir_path):
    # return os.listdir(dir_path)
    # for old in os.listdir(dir_path):
    #     print(old)
    return [old.split("-")[1].split(".")[0] if os.path.isfile(old) else old for old in os.listdir(dir_path)]
    

def search_for_folder(name, directory):
    if "\\" in name:
        name = name.split("\\")
        new_dir = ""
        for n in name[:-1]:
            new_dir = os.path.join(new_dir, n)
        directory = os.path.join(directory, new_dir)
        name = name[-1]
    return [os.path.join(sl[0], name) for sl in list(os.walk(directory)) if name in sl[1]]


# how to create a file searcher without knowing the exact file name??
def search_for_file(name, directory):
    dir_list = check_dir(directory)
    # print(dir_list)
    #    return [dir_list.index(name) if name in dir_list]
    return [i for i in range(len(dir_list)) if name in dir_list[i]]


def get_files(folder_key, file_key, return_call="all"):
    found = search_for_folder(folder_key, os.getcwd())
    # print(found[0])
    if return_call == "all":
        results = []
        for f in found:
            index_list = search_for_file(file_key, f) 
            results.extend(
                Read_from_File(os.path.join(f, os.listdir(f)[i]))
                for i in index_list
            )
        return results
    elif return_call == "first":
        index_list = search_for_file(file_key, found[0])
        # print(index_list)
        return Read_from_File(os.path.join(found[0], os.listdir(found[0])[index_list[0]]))
    elif return_call == "last":
        index_list = search_for_file(file_key, found[-1])
        return Read_from_File(os.path.join(found[-1], os.listdir(found[-1])[index_list[-1]]))
