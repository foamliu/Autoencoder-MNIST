import numpy as np
import keras.backend as K
from keras.layers import Input, Dense, UpSampling2D, BatchNormalization, ZeroPadding2D, MaxPooling2D
from keras.models import Model

def custom_loss_wrapper(input_tensor):
    def custom_loss(y_true, y_pred):
        return K.binary_crossentropy(y_true, y_pred) + K.mean(input_tensor)
    return custom_loss

input_tensor = Input(shape=(10,))
hidden = Dense(100, activation='relu')(input_tensor)
out = Dense(1, activation='sigmoid')(hidden)
model = Model(input_tensor, out)
model.compile(loss=custom_loss_wrapper(input_tensor), optimizer='adam')

X = np.random.rand(1000, 10)
y = np.random.randint(2, size=1000)
model.test_on_batch(X, y)  # => 1.1974642

X *= 1000
model.test_on_batch(X, y)  # => 511.15466