import sys
import numpy as np
import scipy as sp
import scipy.optimize as spo
from scipy.special import erf as sperf
import numpy.linalg as npl
import numpy.random as npr
import pickle

k = int(sys.argv[1])

i = int(k/10) #column index
j = np.mod(k,10) #CV group

root2over = 1/np.sqrt(2)
erf_max = sperf(root2over)
weights_limit = sperf(1e-10)*1e10


def infer_LAD_v(x, y, x_test, y_test, tol=1e-8, max_iter=5000):
    s_sample, s_pred = x.shape
    s_sample, s_target = y.shape
    w_sol = 0.0*(npr.rand(s_pred,s_target) - 0.5)
    b_sol = npr.rand(1,s_target) - 0.5
    for index in range(s_target):
        error, old_error = np.inf, 0
        weights = np.ones((s_sample, 1))
        cov = np.cov(np.hstack((x,y[:,index][:,None])), rowvar=False, ddof=0, aweights=weights.reshape(s_sample))
        cov_xx, cov_xy = cov[:s_pred,:s_pred],cov[:s_pred,s_pred:(s_pred+1)]
        counter = 0
        error_v = []
        while np.abs(error-old_error) > tol and counter < max_iter:
            counter += 1
            old_error = np.mean(np.abs(b_sol[0,index] + x_test.dot(w_sol[:,index]) - y_test[:,index]))
            w_sol[:,index] = npl.solve(cov_xx,cov_xy).reshape(s_pred)
            b_sol[0,index] = np.mean(y[:,index]-x.dot(w_sol[:,index]))
            weights = (b_sol[0,index] + x.dot(w_sol[:,index]) - y[:,index])
            sigma = np.std(weights)
            error = np.mean(np.abs(b_sol[0,index] + x_test.dot(w_sol[:,index]) - y_test[:,index]))
            error_v = np.hstack((error_v, error))
            weights_eq_0 = np.abs(weights) < 1e-10
            weights[weights_eq_0] = weights_limit
            weights[~weights_eq_0] = sigma*sperf(weights[~weights_eq_0]/sigma)/weights[~weights_eq_0]
            cov = np.cov(np.hstack((x,y[:,index][:,None])), rowvar=False, ddof=0, aweights=weights.reshape(s_sample))
            cov_xx, cov_xy = cov[:s_pred,:s_pred],cov[:s_pred,s_pred:(s_pred+1)]
    return w_sol,b_sol, error_v

quad_tr = np.load('./numpy_files/LAD99/quad_tr_%s.npy' % j)
y_tr = np.load('./numpy_files/LAD99/y_tr_%s.npy' % j)
quad_te = np.load('./numpy_files/LAD99/quad_te_%s.npy' % j)
y_te = np.load('./numpy_files/LAD99/y_te_%s.npy' % j)

w, bias, e = infer_LAD_v(quad_tr, y_tr[:,i:(i+1)], quad_te, y_te[:,i:(i+1)])

res = [w, bias, e]

with open('./pickles/res/res_%s.pkl' % k, 'wb') as f:
	pickle.dump(res, f)



 
