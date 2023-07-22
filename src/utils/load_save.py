import os
import json
import zipfile
import numpy as np
import pickle

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def write_json(data, filename, save_pretty=False, sort_keys=False):
    with open(filename, "w") as f:
        if save_pretty:
            f.write(json.dumps(data, indent=4, sort_keys=sort_keys))
        else:
            json.dump(data, f)

def concat_json_list(filepaths, save_path):
    json_lists = []
    for p in filepaths:
        json_lists += load_json(p)
    write_json(json_lists, save_path)


def save_lines(list_of_str, filepath):
    with open(filepath, "w") as f:
        f.write("\n".join(list_of_str))


def read_lines(filepath):
    with open(filepath, "r") as f:
        return [e.strip("\n") for e in f.readlines()]


def get_rounded_percentage(float_number, n_floats=2):
    return round(float_number * 100, n_floats)


def save_parameters(path,opt):
    '''Write parameters setting file'''
    with open(os.path.join(path, 'params.txt'), 'w') as file:
        file.write('Training Parameters: \n')
        file.write(str(opt) + '\n')