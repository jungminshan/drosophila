import sys
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

col = int(sys.argv[1])

quad_t = np.load('./numpy_files/quad_t.npy')
y_t = np.load('./numpy_files/y_t99.npy')

quad_v = np.load('./numpy_files/quad_v.npy')

model = Sequential()
model.add(Dense(1, activation='linear', use_bias=True, input_dim=4950))
model.compile(optimizer='SGD', loss='mean_squared_error')

es = EarlyStopping(monitor='val_loss', min_delta=1e-8, mode='min', verbose=0, patience=200)
mc = ModelCheckpoint(('./savedmodels/mse/best_model%s.h5' % col), monitor='val_loss', mode='min', verbose=0, save_best_only=True)

hist = model.fit(quad_t, y_t[:,col:(col+1)], batch_size=32, epochs=5000, validation_split=0.1, verbose=0, callbacks=[es,mc])
saved_model = load_model('./savedmodels/mse/best_model%s.h5' % col)
yp = saved_model.predict(quad_v)

res = {'pred': yp,
        'hist': hist.history
        }

with open('./pickles/keras/ttv_mse/%s.pkl' % col, 'wb') as f:
    pickle.dump(res,f)

