import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib
import scipy.optimize as spo
from scipy.special import erf as sperf
from sklearn.linear_model import ElasticNet
import numpy.linalg as npl
import numpy.random as npr
import pickle
import timeit

root2over = 1/np.sqrt(2)
erf_max = sperf(root2over)
weights_limit = sperf(1e-10)*1e10
complete_all = ([int(x) - 1 for x in open('../indices_complete.txt','r').readline().split()])
comp_ind = list(map(int, list((np.array(complete_all)[::6]-3)/6)))

def find_sigma(y,h):
    s_sample, s_target = y.shape
    sigma = np.std(y-h,axis=0)
    #     sigma = np.random.rand(1,size) + 0.5
    #     for index in range(size):
    #         def f0(sig):
    #             return (1-np.std(y[:,index]/np.abs(sig) - h[:,index]))**2
    #         res = spo.minimize(f0,sigma[0,index])
    #         sigma[0,index] = np.abs(res.x)
    return(sigma.reshape(1,s_target))

def bias_update(y,h,b_in,pp):
    y_median = np.median(y)
    y_plus = y>y_median
    if pp==0:
        def f0(bias):
            return np.mean(y[y_plus]-np.tanh(bias+h[y_plus]))**2 + np.mean(y[~y_plus]-np.tanh(bias+h[~y_plus]))**2
    else:
        def f0(bias):
            return np.mean(y[y_plus]-odd_power(bias + h[y_plus],pp))**2 + \
                np.mean(y[~y_plus]-odd_power(bias + h[~y_plus],pp))**2
    res = spo.minimize(f0,b_in)
    return res.x

def odd_power(h,power=3):
    sign = np.sign(h)
    return sign*np.power(np.abs(h),1/power)

def enet_solve(c, b, a=0.1, ratio=0):
    regr = ElasticNet(alpha=a, l1_ratio=ratio, random_state=0, max_iter=10000, normalize=False)
    regr.fit(c,b)
    return regr.coef_

def infer_drosophila(x, y, max_iter = 100, tol=1e-8, func=npl.solve, power=1, l=0.1):
    s_sample, s_pred = x.shape
    s_sample, s_target = y.shape
    x0 = np.copy(x)
    y_max = np.max(np.abs(y),axis=0)
    if power == 0:
        y /= y_max[None,:]#now y is definitely within +/- 1
        x0 = x0/y_max[None,:]
    s = np.sign(y)
    c = np.cov(x0,rowvar=False)
    w = npr.rand(s_pred,s_target) - 0.5
    bias = npr.rand(1,s_target) - 0.5
    if power == 0:
        h = np.tanh(bias + x0.dot(w))
    else:
        h = odd_power(bias + x0.dot(w),power)
    for index in range(s_target):
        err_old,error = 0,np.inf
        #         print(index)
        counter = 0
        while np.abs(error-err_old) > tol and counter < max_iter:
            counter += 1 
            zeros = np.abs(bias[0,index] + x0.dot(w[:,index])) < 1e-12
            if power == 0:
                ratio = np.sqrt(np.pi/2.0)*np.ones((s_sample))
            else:
                ratio = np.sqrt(np.pi/2.0)*np.ones((s_sample))*h[:,index]**(power-1)
            ratio[~zeros] = h[~zeros,index]/sperf(h[~zeros,index]*root2over)
#             ratio[~zeros] = y[~zeros,index]/sperf(h[~zeros,index]*root2over)
#             weight[~zeros] = sperf((y[~zeros,index]-h[~zeros,index])*root2over)/(y[~zeros,index] - h[~zeros,index])
#             ratio[~zeros] = (h[~zeros,index]/sperf(h[~zeros,index]*root2over))*weight
            w[:,index] = func(c+l*np.eye(s_pred),\
                              np.mean((x0-np.mean(x0,axis=0)[None,:])*(s[:,index]*ratio)[:,np.newaxis],axis=0))
            h_temp = x0.dot(w[:,index])
            bias[0,index] = bias_update(y[:,index],h_temp,bias[0,index],pp=power)
            err_old = error
            if power == 0:
                h[:,index] = np.tanh(bias[0,index] + h_temp)
                error = npl.norm(s[:,index]-sperf(h[:,index]*root2over)/erf_max)
            else:
                h[:,index] = odd_power(bias[0,index] + h_temp,power)
                error = npl.norm(s[:,index]-sperf(h[:,index]*root2over))
#             if np.abs(error-err_old) < tol or counter >= max_iter:
#                 print(index, np.abs(error-err_old))
#     print(c.shape, np.mean((x0-np.mean(x0,axis=0)[None,:])*(s[:,index]*ratio)[:,np.newaxis],axis=0).shape)
    sigma = find_sigma(y,h)#*y_max[None,:]
    return w,sigma,bias

def infer_LAD_v(x, y, x_test, y_test, tol=1e-8, max_iter=5000):
    s_sample, s_pred = x.shape
    s_sample, s_target = y.shape
    w_sol = 0.0*(npr.rand(s_pred,s_target) - 0.5)
    b_sol = npr.rand(1,s_target) - 0.5
#     print(weights.shape)
    for index in range(s_target):
        error, old_error = np.inf, 0
        weights = np.ones((s_sample, 1))
        cov = np.cov(np.hstack((x,y[:,index][:,None])), rowvar=False, ddof=0, aweights=weights.reshape(s_sample))
        cov_xx, cov_xy = cov[:s_pred,:s_pred],cov[:s_pred,s_pred:(s_pred+1)]
#         print(cov.shape, cov_xx.shape, cov_xy.shape)
        counter = 0
        while np.abs(error-old_error) > tol and counter < max_iter:
            counter += 1
#             old_error = np.mean(np.abs(b_sol[0,index] + x.dot(w_sol[:,index]) - y[:,index]))
            old_error = np.mean(np.abs(b_sol[0,index] + x_test.dot(w_sol[:,index]) - y_test[:,index]))
#             print(w_sol[:,index].shape, npl.solve(cov_xx, cov_xy).reshape(s_pred).shape)
            w_sol[:,index] = npl.solve(cov_xx,cov_xy).reshape(s_pred)
            b_sol[0,index] = np.mean(y[:,index]-x.dot(w_sol[:,index]))
            weights = (b_sol[0,index] + x.dot(w_sol[:,index]) - y[:,index])
            sigma = np.std(weights)
#             error = np.mean(np.abs(weights))
            error = np.mean(np.abs(b_sol[0,index] + x_test.dot(w_sol[:,index]) - y_test[:,index]))
            weights_eq_0 = np.abs(weights) < 1e-10
            weights[weights_eq_0] = weights_limit
            weights[~weights_eq_0] = sigma*sperf(weights[~weights_eq_0]/sigma)/weights[~weights_eq_0]
            cov = np.cov(np.hstack((x,y[:,index][:,None])), rowvar=False, ddof=0, aweights=weights.reshape(s_sample))
            cov_xx, cov_xy = cov[:s_pred,:s_pred],cov[:s_pred,s_pred:(s_pred+1)]
#             print(old_error,error)
#     print(index, '\t', round(np.abs(old_error-error),10), '\t', round(error,6), counter)
    return w_sol,b_sol

def infer_LAD(x, y, tol=1e-8, max_iter=5000):
    s_sample, s_pred = x.shape
    s_sample, s_target = y.shape
    w_sol = 0.0*(npr.rand(s_pred,s_target) - 0.5)
    b_sol = npr.rand(1,s_target) - 0.5
#     print(weights.shape)
    for index in range(s_target):
        error, old_error = np.inf, 0
        weights = np.ones((s_sample, 1))
        cov = np.cov(np.hstack((x,y[:,index][:,None])), rowvar=False, ddof=0, aweights=weights.reshape(s_sample))
        cov_xx, cov_xy = cov[:s_pred,:s_pred],cov[:s_pred,s_pred:(s_pred+1)]
#         print(cov.shape, cov_xx.shape, cov_xy.shape)
        counter = 0
        while np.abs(error-old_error) > tol and counter < max_iter:
            counter += 1
            old_error = np.mean(np.abs(b_sol[0,index] + x.dot(w_sol[:,index]) - y[:,index]))
#             old_error = np.mean(np.abs(b_sol[0,index] + x_test.dot(w_sol[:,index]) - y_test[:,index]))
#             print(w_sol[:,index].shape, npl.solve(cov_xx, cov_xy).reshape(s_pred).shape)
            w_sol[:,index] = npl.solve(cov_xx,cov_xy).reshape(s_pred)
            b_con = np.max(-(x[:,index] + x.dot(w_sol[:,index])))
            b_sol[0,index] = np.mean(y[:,index]-x.dot(w_sol[:,index]))
            if b_sol[0,index] < b_con:
                b_sol[0,index] = b_con
            weights = (b_sol[0,index] + x.dot(w_sol[:,index]) - y[:,index])
            sigma = np.std(weights)
            error = np.mean(np.abs(weights))
#             error = np.mean(np.abs(b_sol[0,index] + x_test.dot(w_sol[:,index]) - y_test[:,index]))
            weights_eq_0 = np.abs(weights) < 1e-10
            weights[weights_eq_0] = weights_limit
            weights[~weights_eq_0] = sigma*sperf(weights[~weights_eq_0]/sigma)/weights[~weights_eq_0]
            cov = np.cov(np.hstack((x,y[:,index][:,None])), rowvar=False, ddof=0, aweights=weights.reshape(s_sample))
            cov_xx, cov_xy = cov[:s_pred,:s_pred],cov[:s_pred,s_pred:(s_pred+1)]
#             print(old_error,error)
#         print(index, '\t', round(np.abs(old_error-error),10), '\t', round(error,6), counter)
    return w_sol,b_sol

def infer_LAD_p(x, y, p, tol=1e-8, max_iter=100):
    s_sample, s_pred = x.shape
    s_sample, s_target = y.shape
    w_sol = 0.0*(npr.rand(s_pred,s_target) - 0.5)
    b_sol = npr.rand(1,s_target) - 0.5
#     print(weights.shape)
    for index in range(s_target):
        error, old_error = np.inf, 0
        weights = np.ones((s_sample, 1))
        cov = np.cov(np.hstack((x,y[:,index][:,None])), rowvar=False, ddof=0, aweights=weights.reshape(s_sample))
        cov_xx, cov_xy = cov[:s_pred,:s_pred],cov[:s_pred,s_pred:(s_pred+1)]
#         print(cov.shape, cov_xx.shape, cov_xy.shape)
        counter = 0
        while np.abs(error-old_error) > tol and counter < max_iter:
            counter += 1
            old_error = np.mean(np.abs(b_sol[0,index] + x.dot(w_sol[:,index]) - y[:,index]))
#             print(w_sol[:,index].shape, npl.solve(cov_xx, cov_xy).reshape(s_pred).shape)
            w_sol[:,index] = npl.solve(cov_xx,cov_xy).reshape(s_pred)
            b_sol[0,index] = np.mean(y[:,index]-x.dot(w_sol[:,index]))
            weights = (b_sol[0,index] + x.dot(w_sol[:,index]) - y[:,index])
            sigma = np.std(weights)
            error = np.mean(np.abs(weights))
            weights_eq_0 = np.abs(weights) < 1e-10
            weights[weights_eq_0] = weights_limit
            weights[~weights_eq_0] = (sigma*sperf(weights[~weights_eq_0]/sigma)/weights[~weights_eq_0])**(2-p)
            cov = np.cov(np.hstack((x,y[:,index][:,None])), rowvar=False, ddof=0, aweights=weights.reshape(s_sample))
            cov_xx, cov_xy = cov[:s_pred,:s_pred],cov[:s_pred,s_pred:(s_pred+1)]
#             print(old_error,error)
    return w_sol,b_sol

def infer_drosophila_quad(x, y, max_iter = 100, tol=1e-8, func=npl.solve, power=1, l=0.1):
    s_sample, s_pred = x.shape
    s_sample, s_target = y.shape
    x0 = np.copy(x)
    y_max = np.max(np.abs(y),axis=0)
    if power<3 and power !=0:
        y /= y_max[None,:]#now y is definitely within +/- 1
        if y.shape[1] == 99:
            x0[:,:99] = x0[:,:99]/y_max[None,:]
        elif y.shape[1] == 27:
            x0[:, comp_ind] = x0[:, comp_ind]/y_max[None,:]
    s = np.sign(y)
    c = np.cov(x0,rowvar=False)
    w = npr.rand(s_pred, s_target) - 0.5
    bias = npr.rand(1,s_target) - 0.5
    if power == 0:
        h = bias + x0.dot(w)
    elif power<3 and power != 0:
        h = np.tanh(bias + x0.dot(w))
    else:
        h = odd_power(bias + x0.dot(w),power)
    for index in range(s_target):
        err_old,error = 0,np.inf
        #         print(index)
        counter = 0
        while np.abs(error-err_old) > tol and counter < max_iter:
            counter += 1 
            zeros = np.abs(bias[0,index] + x0.dot(w[:,index])) < 1e-12
            ratio = np.sqrt(np.pi/2.0)*np.ones((s_sample))
            ratio[~zeros] = (bias[0,index] + x0[~zeros,:].dot(w[:,index]))/sperf(h[~zeros,index]*root2over)
            w[:,index] = func(c+l*np.eye(s_pred),np.mean((x0-np.mean(x0,axis=0)[None,:])*(s[:,index]*ratio)[:,np.newaxis],axis=0))
            h_temp = x0.dot(w[:,index])
            bias[0,index] = bias_update(y[:,index],h_temp,bias[0,index],pp=power)
            err_old = error
            if power == 0:
                h[:,index] = bias[0,index] + h_temp
                error = npl.norm(s[:,index]-sperf(h[:,index]*root2over)/erf_max)
            elif power<3 and power != 0:
                h[:,index] = np.tanh(bias[0,index] + h_temp)
                error = npl.norm(s[:,index]-sperf(h[:,index]*root2over)/erf_max)
            else:
                h[:,index] = odd_power(bias[0,index] + h_temp,power)
                error = npl.norm(s[:,index]-sperf(h[:,index]*root2over))
#             if np.abs(error-err_old) < tol or counter >= max_iter:
#                 print(index, np.abs(error-err_old))
    sigma = find_sigma(y,h)#*y_max[None,:]
    return w,sigma,bias

def ER_quad_elastic(x, y, max_iter = 100, tol=1e-8, func=enet_solve, power=1, alpha_=0.1, ratio_=0):
    s_sample, s_pred = x.shape
    s_sample, s_target = y.shape
    x0 = np.copy(x)
    y_max = np.max(np.abs(y),axis=0)
    if power<3 and power !=0:
        y /= y_max[None,:]#now y is definitely within +/- 1
        x0 = x0/y_max[None,:]
    s = np.sign(y)
    c = np.cov(x0,rowvar=False)
    w = npr.rand(s_pred, s_target) - 0.5
    bias = npr.rand(1,s_target) - 0.5
    if power == 0:
        h = bias + x0.dot(w)
    elif power<3 and power != 0:
        h = np.tanh(bias + x0.dot(w))
    else:
        h = odd_power(bias + x0.dot(w),power)
    for index in range(s_target):
        err_old,error = 0,np.inf
        #         print(index)
        counter = 0
        while np.abs(error-err_old) > tol and counter < max_iter:
            counter += 1 
            zeros = np.abs(bias[0,index] + x0.dot(w[:,index])) < 1e-12
            ratio = np.sqrt(np.pi/2.0)*np.ones((s_sample))
            ratio[~zeros] = (bias[0,index] + x0[~zeros,:].dot(w[:,index]))/sperf(h[~zeros,index]*root2over)
            w[:,index] = enet_solve(x0-np.mean(x0,axis=0), (s[:,index]*ratio)[:,np.newaxis], a=alpha_, ratio=ratio_)
#             elastic = ElasticNet(alpha=alpha_, l1_ratio=ratio_, normalize=False)
#             elastic.fit(np.mean((x0-np.mean(x0,axis=0)[None,:]), (s[:,index]*ratio)[:,np.newaxis],axis=0))
#             w[:,index] =  elastic.coef_
            h_temp = x0.dot(w[:,index])
            bias[0,index] = bias_update(y[:,index],h_temp,bias[0,index],pp=power)
            err_old = error
            if power == 0:
                h[:,index] = bias[0,index] + h_temp
                error = npl.norm(s[:,index]-sperf(h[:,index]*root2over)/erf_max)
            elif power<3 and power != 0:
                h[:,index] = np.tanh(bias[0,index] + h_temp)
                error = npl.norm(s[:,index]-sperf(h[:,index]*root2over)/erf_max)
            else:
                h[:,index] = odd_power(bias[0,index] + h_temp,power)
                error = npl.norm(s[:,index]-sperf(h[:,index]*root2over))
#             if np.abs(error-err_old) < tol or counter >= max_iter:
#                 print(index, np.abs(error-err_old))
    sigma = find_sigma(y,h)#*y_max[None,:]
    return w,sigma,bias