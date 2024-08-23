import sys
import tensorflow as tf
from tensorflow.keras import layers

def make_discriminator_model():
    model = tf.keras.Sequential()

    model.add(layers.Conv3D(1, (1, 1, 1), strides=(1, 1, 1), padding='same', input_shape=[160,160,3, 1]))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))
    print(model.output_shape)

    model.add(layers.Conv3D(30, (4, 4, 4), strides=(2, 2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))
    print(model.output_shape)

    model.add(layers.Conv3D(60, (2, 2, 2), strides=(2, 2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv3D(120, (4, 4, 4), strides=(2, 2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))
    print(model.output_shape)

    model.add(layers.Conv3D(240, (4, 4, 4), strides=(2, 2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))
    print(model.output_shape)

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

sys.modules[__name__] = make_discriminator_model