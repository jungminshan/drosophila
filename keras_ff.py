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
model.add(Dense(50000, activation='linear', input_dim=4950))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='SGD', loss='mean_absolute_error')

es = EarlyStopping(monitor='val_loss', min_delta=1e-8, mode='min', verbose=0, patience=200)
mc = ModelCheckpoint(('./savedmodels/best_model_hidden%s.h5' % col), monitor='val_loss', mode='min', verbose=0, save_best_only=True)

hist = model.fit(quad_t, y_t, batch_size=32, epochs=5000, validation_split=0.1, shuffle=False, verbose=0, callbacks=[es,mc])
saved_model = load_model('./savedmodels/best_model_hidden%s.h5' % col)
yp = saved_model.predict(quad_v)

res = {'pred': yp,
        'hist': hist.history
        }

with open('./pickles/keras/%s.pkl' % col, 'wb') as f:
    pickle.dump(res,f)
