import sys
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

col = sys.argv[1]

quad_t = np.load('./numpy_files/quad_t.npy')
y_t = np.load('./numpy_files/y_t%s.npy' % col)

quad_v = np.load('./numpy_files/quad_v.npy')
y_v = np.load('./numpy_files/y_v%s.npy' % col)

model = Sequential()
model.add(Dense(1, activation='linear', use_bias=True, input_dim=4950))
model.compile(optimizer='SGD', loss='mean_absolute_error')

hist = model.fit(quad_t, y_t, batch_size=50, epochs=200, validation_data=(quad_v, y_v), verbose=0)
yp = model.predict(quad_v)

res = {'pred': yp,
        'hist': hist.history
        }

with open('./pickles/keras/test_%s.pkl' % col, 'wb') as f:
    pickle.dump(res,f)
