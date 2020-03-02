from f_data_prep import *
from f_drosophila_infer import *
from f_train import *
import multiprocessing
from joblib import Parallel, delayed

data_all = np.loadtxt('../data_complete.txt')
all_bin = np.vsplit(data_all, 6)
all_init = np.vstack([all_bin[i] for i in range(5)])
all_diff = np.vstack([all_bin[i+1]-all_bin[i] for i in range(5)])

complete_all = ([int(x) - 1 for x in open('../indices_complete.txt','r').readline().split()])
comp_ind = list(map(int, list((np.array(complete_all)[::6]-3)/6)))

data_comp = np.copy(data_all[:, comp_ind])
comp_bin = np.vsplit(data_comp, 6)
comp_diff = np.vstack([comp_bin[i+1] - comp_bin[i] for i in range(5)])

def make_quad(X):
    quad = np.zeros((int(X.shape[0]), int(X.shape[1] + (X.shape[1]*(X.shape[1]-1))/2)))
    quad[:, :X.shape[1]] = np.copy(X)
    col = 99
    for i in range(X.shape[1]-1):
        for j in range(i+1, X.shape[1]):
            quad[:,col] = (X[:,i]*X[:,j])
            col += 1
    return quad

quad = make_quad(all_init)

with open('./pickles/validation_cells.pkl', 'rb') as f:
    cells_v = pickle.load(f)
    
val = np.hstack([cells_v+(6078*i) for i in range(5)])
cells_tt = np.delete(range(6078), cells_v)
tr_te = np.hstack([cells_tt+(6078*i) for i in range(5)])

X_v = all_init[val]
quad_v = quad[val]
y_v = all_diff[val]
quad_tt = quad[tr_te]
y_tt = all_diff[tr_te]

quad_tr = []
y_tr = []
quad_te = []
y_te = []

kfold = KFold(n_splits=10, shuffle=False, random_state=1)
for (cell_tr, cell_te) in (kfold.split(range(5470))):
    te = np.hstack([cell_te+(5470*i) for i in range(5)])
    tr = np.delete(range(27350), te)
    quad_tr.append(quad_tt[tr])
    y_tr.append(y_tt[tr])
    quad_te.append(quad_tt[te])
    y_te.append(y_tt[te])
    
    num_cores = multiprocessing.cpu_count()

def test_parallel(i):
    w, bias = infer_LAD_v(quad_tr[i], y_tr[i][:,0:1], quad_te[i], y_te[i][:,0:1])
    return w, bias
results = Parallel(n_jobs=num_cores, verbose=11)(delayed(test_parallel)(i) for i in range(10))