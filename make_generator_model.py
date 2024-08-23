import sys
import tensorflow as tf
from tensorflow.keras import layers

def make_generator_model():
    output_shape = (160, 160, 3,1)
    model = tf.keras.Sequential()
    initializer = tf.random_normal_initializer(0., 0.02)

    start_dim = 3  # You can adjust this depending on the desired output size
    depth = 160   # Number of filters in the first Conv3DTranspose layer

    model.add(layers.Dense(10 * 10 * 1 * depth, use_bias=False, input_shape=(400,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((10, 10, 1, depth)))
    print(model.output_shape)
    assert model.output_shape == (None, 10, 10, 1, depth)  # Note: None is the batch size

    model.add(layers.Conv3DTranspose(64, (4, 4, 4), strides=(2, 2, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    print(model.output_shape)
    assert model.output_shape == (None, 20, 20, 1, 64)

    model.add(layers.Conv3DTranspose(64, (4, 4, 4), strides=(2, 2, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    print(model.output_shape)
    assert model.output_shape == (None, 40, 40, 1, 64)

    model.add(layers.Conv3DTranspose(32, (4, 4, 4), strides=(2, 2, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    print(model.output_shape)
    assert model.output_shape == (None, 80, 80, 1, 32)

    model.add(layers.Conv3DTranspose(32, (4, 4, 4), strides=(2, 2, 3), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    print(model.output_shape)
    assert model.output_shape == (None, 160, 160, start_dim, 32)

    model.add(layers.Conv3DTranspose(16, (4, 4, 4), strides=(1, 1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    print(model.output_shape)
    assert model.output_shape == (None, 160, 160, start_dim, 16)

    model.add(layers.Conv3DTranspose(8, (4, 4, 4), strides=(1, 1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    print(model.output_shape)
    assert model.output_shape == (None, 160, 160, start_dim, 8)

    model.add(layers.Conv3DTranspose(4, (4, 4, 4), strides=(1, 1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    print(model.output_shape)
    assert model.output_shape == (None, 160, 160, start_dim, 4)

    model.add(layers.Conv3D(output_shape[-1], (4, 4, 1), strides=(1, 1, 1), padding='same', use_bias=False, activation='tanh', kernel_initializer=initializer))
    print(model.output_shape)
    assert model.output_shape == (None, output_shape[0], output_shape[1], output_shape[2], output_shape[3])

    return model

sys.modules[__name__] = make_generator_model