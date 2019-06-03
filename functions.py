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

tor=0.75
epochs=50
epoch_increment=20
nn_error=40

def nn_interpolate(train_in, train_out, test_in, epochs=50, nn_error=nn_error):
    in_size = train_in.shape[1]
    combined = np.append(train_in, test_in)
    error, num_iter=[100000], 0
    while error[-1]>=nn_error and num_iter<2:
        net = nl.net.newff([[np.min(combined),np.max(combined)] for i in range(in_size)],[int(in_size/2),1])
        error = net.train(train_in, train_out/np.max(train_out),\
                          epochs=epochs+epoch_increment*num_iter,\
                          show=100, goal=1e-5)
        num_iter += 1
        h_preder = net.sim(test_in).reshape(test_in.shape[0])
        factor = np.max(train_out)/np.max(h_preder)
        def netsim(v):
            return net.sim(v).reshape(v.shape[0])*factor
    return h_preder*factor, netsim

def nn_train(train_data, comp_ind, epochs=epochs, redo_ind=[], nn_error=nn_error):
    if len(redo_ind) > 0:
        incomp_ind = np.array(redo_ind,dtype='int')
    else:
        incomp_ind = np.array([i for i in range(train_data.shape[1]) if i not in set(comp_ind)])
    train_row_ind = {i:np.array([j for j in range(train_data.shape[0])\
                                 if train_data[j,i] != 0]) for i in incomp_ind}
    test_row_ind = {i:np.array([j for j in range(train_data.shape[0])\
                                if train_data[j,i] == 0]) for i in incomp_ind}
    
    gene_train_input = np.copy(train_data[:,comp_ind]) #(3039 cells * 6 timepoints) x 27 genes
    nn_pred = {i:nn_interpolate(gene_train_input[train_row_ind[i],:],\
                        train_data[train_row_ind[i],i:i+1],\
                        gene_train_input[test_row_ind[i],:], epochs=epochs, nn_error=nn_error)\
               for i in incomp_ind}
    for i in incomp_ind:
        train_data[test_row_ind[i],i] = nn_pred[i][0]
    return {i:(nn_pred[i][1], np.copy(test_row_ind[i])) for i in incomp_ind}
    
def nn_complete(train_data, test_data, comp_ind):
    n_error = nn_error
    incomp_ind = np.array([i for i in range(test_data.shape[1]) if i not in set(comp_ind)])
    train_row_ind = {i:np.array([j for j in range(test_data.shape[0])\
                                 if test_data[j,i] != 0]) for i in incomp_ind}
    test_row_ind = {i:np.array([j for j in range(test_data.shape[0])\
                                if test_data[j,i] == 0]) for i in incomp_ind}
    
    nn_pred_functions = nn_train(train_data, comp_ind, redo_ind=[])
    
    gene_test_input = np.copy(test_data[:, comp_ind])
    repeat=[]
    rep_count = 0
    corr_v = [0]*len(incomp_ind)
    ind = 0
    for i in nn_pred_functions.keys():
        test_data[test_row_ind[i],i] = nn_pred_functions[i][0](gene_test_input[test_row_ind[i],:])
        corr = stats.linregress(test_data[train_row_ind[i],i],\
                                      nn_pred_functions[i][0](gene_test_input[train_row_ind[i],:]))[-3]
        if corr < tor:
            repeat.append(i)
            train_data[nn_pred_functions[i][1],i] = 0
            test_data[test_row_ind[i],i] = 0
        if corr > tor:
            corr_v[ind] = corr
            ind += 1
    while len(repeat) > 0:
#         print(rep_count)
#         print(len(repeat))
        rep_count += 1
        nn_pred_functions = nn_train(train_data, comp_ind, epochs=epochs+5*rep_count,\
                                     redo_ind=repeat, nn_error=n_error+rep_count)
        repeat = []
        for i in nn_pred_functions.keys():
            test_data[test_row_ind[i],i] = nn_pred_functions[i][0](gene_test_input[test_row_ind[i],:])
            corr = stats.linregress(test_data[train_row_ind[i],i],\
                                      nn_pred_functions[i][0](gene_test_input[train_row_ind[i],:]))[-3]
            if corr < tor:
                print(i, corr)
                repeat.append(i)
                train_data[nn_pred_functions[i][1], i] = 0
                test_data[test_row_ind[i],i] = 0
            if corr > tor:
                corr_v[ind] = corr
                ind += 1
            
    return corr_v

def make_data(data, n_bin=6):
    data_bin = np.vsplit(data, n_bin)
    data_init = [data_bin[i] for i in range(n_bin-1)]
    data_fin = [data_bin[i+1] for i in range(n_bin-1)]
    data_midpt = [(data_bin[i]+data_bin[i+1])*0.5 for i in range(n_bin-1)]
    data_deriv = [np.sign(data_bin[i+1]-data_bin[i]) for i in range(n_bin-1)]
    return np.vstack(data_init), np.vstack(data_fin), np.vstack(data_midpt), np.vstack(data_deriv)

def make_deriv(data, boundary=0, n_bin=6):
    data_bin = np.vsplit(data, n_bin)
    data_diff = [data_bin[i+1]-data_bin[i] for i in range(n_bin-1)]
    data_deriv = np.copy(np.vstack(data_diff))
    data_deriv[data_deriv <= boundary] = -1
    data_deriv[data_deriv > boundary] = 1
    return np.vstack(data_diff), data_deriv
    

def fit(x,y,niter_max=100,regu=0.1):    
    n = x.shape[1]
    
    x_av = np.mean(x,axis=0)
    dx = x - x_av
    c = np.cov(dx,rowvar=False,bias=True)
    c += regu*np.identity(n)
    c_inv = linalg.inv(c)

    # initial values
    h0 = 0.
    w = np.random.normal(0.0,1./np.sqrt(n),size=(n))
    
    cost = np.full(niter_max,100.)
    for iloop in range(niter_max):
        h = h0 + x.dot(w)
        y_model = np.tanh(h)    

        # stopping criterion
        cost[iloop] = ((y[:]-y_model[:])**2).mean()
        if iloop>0 and cost[iloop] >= cost[iloop-1]: break

        # update local field
        t = h!=0    
        h[t] *= y[t]/y_model[t]
        h[~t] = y[~t]

        # find w from h    
        h_av = h.mean()
        dh = h - h_av 
        dhdx = dh[:,np.newaxis]*dx[:,:]

        dhdx_av = dhdx.mean(axis=0)
        w = c_inv.dot(dhdx_av)
        h0 = h_av - x_av.dot(w)
        
    return h0,w
