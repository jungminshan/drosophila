import numpy as np
from scipy import stats, linalg
import os
import pandas as pd
import neurolab as nl
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle
import itertools as it

def make_data(data, n_bin=6):
    data_bin = np.vsplit(data, n_bin)
    data_init = [data_bin[i] for i in range(n_bin-1)]
    data_fin = [data_bin[i+1] for i in range(n_bin-1)]
    data_midpt = [(data_bin[i]+data_bin[i+1])*0.5 for i in range(n_bin-1)]
    data_deriv = [np.sign(data_bin[i+1]-data_bin[i]) for i in range(n_bin-1)]
    return np.vstack(data_init), np.vstack(data_fin), np.vstack(data_midpt), np.vstack(data_deriv)

def make_data_diff(data, n_bin=6):
    data_bin = np.vsplit(data, n_bin)
    data_diff = [data_bin[i+1]-data_bin[i] for i in range(n_bin-1)]
    return np.vstack(data_diff)

def make_data_boundary(data, boundary=0, n_bin=6):
    data_bin = np.vsplit(data, n_bin)
    data_init = [data_bin[i] for i in range(n_bin-1)]
    data_fin = [data_bin[i+1] for i in range(n_bin-1)]
    data_midpt = [(data_bin[i]+data_bin[i+1])*0.5 for i in range(n_bin-1)]
    data_diff = [data_bin[i+1]-data_bin[i] for i in range(n_bin-1)]
    data_deriv = np.copy(np.vstack(data_diff))
    data_deriv[data_deriv <= boundary] = -1
    data_deriv[data_deriv > boundary] = 1
    return np.vstack(data_init), np.vstack(data_fin), np.vstack(data_midpt), np.vstack(data_deriv)
    