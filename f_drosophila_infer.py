import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib
import scipy.optimize as spo
from scipy.special import erf as sperf
from scipy.special import erf
from sklearn.linear_model import ElasticNet
import numpy.linalg as npl
import numpy.random as npr
import pickle as pkl

root2over = 1/np.sqrt(2)
erf_max = sperf(root2over)

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
            return np.mean(y[y_plus]-(bias+h[y_plus]))**2 + np.mean(y[~y_plus]-(bias+h[~y_plus]))**2
    elif pp==1:
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
    if power<3:
        y /= y_max[None,:]#now y is definitely within +/- 1
        x0 = x0/y_max[None,:]
    s = np.sign(y)
    c = np.cov(x0,rowvar=False)
    w = npr.rand(s_pred,s_target) - 0.5
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
            ratio[~zeros] = (bias[0,index] + x0[~zeros,:].dot(w[:,index]))/sp.special.erf(h[~zeros,index]*root2over)
            w[:,index] = func(c+l*np.eye(s_target),np.mean((x0-np.mean(x0,axis=0)[None,:])*(s[:,index]*ratio)[:,np.newaxis],axis=0))
            h_temp = x0.dot(w[:,index])
            bias[0,index] = bias_update(y[:,index],h_temp,bias[0,index],pp=power)
            err_old = error
            if power == 0:
                h[:,index] = bias[0,index] + h_temp
                error = npl.norm(s[:,index]-sp.special.erf(h[:,index]*root2over)/erf_max)
            elif power<3 and power != 0:
                h[:,index] = np.tanh(bias[0,index] + h_temp)
                error = npl.norm(s[:,index]-sp.special.erf(h[:,index]*root2over)/erf_max)
            else:
                h[:,index] = odd_power(bias[0,index] + h_temp,power)
                error = npl.norm(s[:,index]-sp.special.erf(h[:,index]*root2over))
#             if np.abs(error-err_old) < tol or counter >= max_iter:
#                 print(index, np.abs(error-err_old))
    sigma = find_sigma(y,h)#*y_max[None,:]
    return w,sigma,bias

def infer_drosophila_quad(x, y, max_iter = 100, tol=1e-8, func=npl.solve, power=1, l=0.1):
    s_sample, s_pred = x.shape
    s_sample, s_target = y.shape
    x0 = np.copy(x)
    y_max = np.max(np.abs(y),axis=0)
    if power<3:
        y /= y_max[None,:]#now y is definitely within +/- 1
#         x0 = x0/y_max[None,:]
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
            ratio[~zeros] = (bias[0,index] + x0[~zeros,:].dot(w[:,index]))/sp.special.erf(h[~zeros,index]*root2over)
            w[:,index] = func(c+l*np.eye(s_pred),np.mean((x0-np.mean(x0,axis=0)[None,:])*(s[:,index]*ratio)[:,np.newaxis],axis=0))
            h_temp = x0.dot(w[:,index])
            bias[0,index] = bias_update(y[:,index],h_temp,bias[0,index],pp=power)
            err_old = error
            if power == 0:
                h[:,index] = bias[0,index] + h_temp
                error = npl.norm(s[:,index]-sp.special.erf(h[:,index]*root2over)/erf_max)
            elif power<3 and power != 0:
                h[:,index] = np.tanh(bias[0,index] + h_temp)
                error = npl.norm(s[:,index]-sp.special.erf(h[:,index]*root2over)/erf_max)
            else:
                h[:,index] = odd_power(bias[0,index] + h_temp,power)
                error = npl.norm(s[:,index]-sp.special.erf(h[:,index]*root2over))
#             if np.abs(error-err_old) < tol or counter >= max_iter:
#                 print(index, np.abs(error-err_old))
    sigma = find_sigma(y,h)#*y_max[None,:]
    return w,sigma,bias

def ER_quad_elastic(x, y, max_iter = 100, tol=1e-8, func=enet_solve, power=1, alpha_=0.1, ratio_=0):
    s_sample, s_pred = x.shape
    s_sample, s_target = y.shape
    x0 = np.copy(x)
    y_max = np.max(np.abs(y),axis=0)
    if power<3:
        y /= y_max[None,:]#now y is definitely within +/- 1
#         x0 = x0/y_max[None,:]
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
            ratio[~zeros] = (bias[0,index] + x0[~zeros,:].dot(w[:,index]))/sp.special.erf(h[~zeros,index]*root2over)
            w[:,index] = enet_solve(x0-np.mean(x0,axis=0), (s[:,index]*ratio)[:,np.newaxis], a=alpha_, ratio=ratio_)
#             elastic = ElasticNet(alpha=alpha_, l1_ratio=ratio_, normalize=False)
#             elastic.fit(np.mean((x0-np.mean(x0,axis=0)[None,:]), (s[:,index]*ratio)[:,np.newaxis],axis=0))
#             w[:,index] =  elastic.coef_
            h_temp = x0.dot(w[:,index])
            bias[0,index] = bias_update(y[:,index],h_temp,bias[0,index],pp=power)
            err_old = error
            if power == 0:
                h[:,index] = bias[0,index] + h_temp
                error = npl.norm(s[:,index]-sp.special.erf(h[:,index]*root2over)/erf_max)
            elif power<3 and power != 0:
                h[:,index] = np.tanh(bias[0,index] + h_temp)
                error = npl.norm(s[:,index]-sp.special.erf(h[:,index]*root2over)/erf_max)
            else:
                h[:,index] = odd_power(bias[0,index] + h_temp,power)
                error = npl.norm(s[:,index]-sp.special.erf(h[:,index]*root2over))
#             if np.abs(error-err_old) < tol or counter >= max_iter:
#                 print(index, np.abs(error-err_old))
    sigma = find_sigma(y,h)#*y_max[None,:]
    return w,sigma,bias