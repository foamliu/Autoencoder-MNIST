import os
import random

import cv2 as cv
import keras.backend as K
import numpy as np
from keras.datasets import mnist

from autoencoder import create

if __name__ == '__main__':
    model, encoder = create()
    model.load_weights('models/model.119-0.01.hdf5')

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if not os.path.exists('images'):
        os.makedirs('images')

    num_rows, num_cols = 10, 30
    num_samples = num_rows * num_cols

    indices = random.sample(range(len(x_test)), num_samples)

    origin = np.empty((28 * num_rows, 28 * num_cols), dtype=np.float32)
    rec = np.empty((28 * num_rows, 28 * num_cols), dtype=np.float32)
    for i in range(num_rows):
        for j in range(num_cols):
            pos_x = j * 28
            pos_y = i * 28
            origin[pos_y:pos_y + 28, pos_x:pos_x + 28] = x_test[indices[i * num_cols + j]]
            x = np.empty((num_samples, 28, 28, 1), dtype=np.float32)
            x[0, :, :, 0] = x_test[indices[i * num_cols + j]] / 255.
            out = model.predict(x)
            out = out * 255.0
            rec[pos_y:pos_y + 28, pos_x:pos_x + 28] = out[0, :, :, 0]
    origin = origin.astype(np.uint8)
    rec = rec.astype(np.uint8)
    cv.imshow('origin', origin)
    cv.imshow('rec', rec)
    cv.imwrite('images/origin.png', origin)
    cv.imwrite('images/rec.png', rec)
    cv.waitKey(0)

    x = np.empty((1, 28, 28, 1), dtype=np.float32)
    x[0, :, :, 0] = x_test[0] / 255.
    code = encoder.predict(x)[0]
    print(code)
    print(code.shape)

    K.clear_session()
