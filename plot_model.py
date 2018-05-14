import keras.backend as K
from keras.layers import Input, Conv2D, UpSampling2D, MaxPooling2D, Reshape, Concatenate, Lambda, Multiply, Permute, \
    Activation
from keras.models import Model
from keras.utils import plot_model

num_channels = 3
img_h = img_w = 224
n_labels = 10
img_input = Input(shape=(num_channels, img_h, img_w))

x = Conv2D(512, (3, 3), padding="same")(img_input)

orig = x  # Save output x
x = MaxPooling2D()(x)

x = UpSampling2D()(x)

# here we're going to reshape the data for a concatenation:
# xReshaped and origReshaped are now split branches
xReshaped = Reshape((1, 512, 224, 224))(x)
origReshaped = Reshape((1, 512, 224, 224))(orig)

# concatenation - here, you unite both branches again
# normally you don't need to reshape or use the axis var,
# but here we want to keep track of what was x and what was orig.
together = Concatenate(axis=1)([origReshaped, xReshaped])

bool_mask = Lambda(lambda t: K.greater_equal(t[:, 0], t[:, 1]),
                   output_shape=(512, 224, 224))(together)
mask = Lambda(lambda t: K.cast(t, dtype='float32'))(bool_mask)

x = Multiply()([mask, x])
x = Conv2D(512, (3, 3), padding="same")(x)

x = Reshape((n_labels, img_h * img_w))(x)
x = Permute((2, 1))(x)
main_output = Activation('softmax')(x)

model = Model(inputs=img_input, outputs=main_output)

print(model.summary())
plot_model(model, to_file='model.svg', show_layer_names=True, show_shapes=True)
