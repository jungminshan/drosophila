import sys
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

n_hidden = int(sys.argv[1])

quad_t = np.load('./numpy_files/quad_t.npy')
y_t = np.load('./numpy_files/y_t99.npy')

quad_v = np.load('./numpy_files/quad_v.npy')

model = Sequential()
model.add(Dense(n_hidden, activation='linear', input_dim=4950))
model.add(Dense(99, activation='linear'))
model.compile(optimizer='SGD', loss='mean_absolute_error')

es = EarlyStopping(monitor='val_loss', min_delta=0, mode='min', verbose=0, patience=200)
mc = ModelCheckpoint(('./savedmodels/hidden_%s.h5' % str(n_hidden)), monitor='val_loss', mode='min', verbose=0, save_best_only=True)

hist = model.fit(quad_t, y_t, batch_size=32, epochs=5000, validation_split=0.1, verbose=0, callbacks=[es,mc])
saved_model = load_model('./savedmodels/hidden_%s.h5' % str(n_hidden))
yp = saved_model.predict(quad_v)

res = {'pred': yp,
        'hist': hist.history
        }

with open('./pickles/keras/hidden/%s.pkl' % str(n_hidden), 'wb') as f:
    pickle.dump(res,f)

