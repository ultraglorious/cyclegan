import tensorflow as tf
import layers
import layers.shortcuts as sc


def discriminator(shape: tf.TensorShape) -> tf.keras.Model:
    """Discriminator architecture is C64-C128-C256-C512"""
    inputs = tf.keras.layers.Input(shape=shape[1:], batch_size=shape[0])
    x = sc.ck(64, normalize=False)(inputs)  # (bs, res/2, res/2, 64)
    x = sc.ck(128)(x)  # (bs, res/4, res/4, 128)
    x = sc.ck(256)(x)  # (bs, res/8, res/8, 256)
    x = sc.ck(512)(x)  # (bs, res/16, res/16, 512)
    # This next layer is unlisted in the paper but is allegedly in it (bs, res/16, res/16, 512)
    # See: https://machinelearningmastery.com/how-to-develop-cyclegan-models-from-scratch-with-keras/
    x = layers.ConvolutionBlock(kernel_size=4, stride=1, n_filters=512, normalize=True)(x)
    # Final convolution to produce 1D output (bs, res/16, res/16, 1)
    x = tf.keras.layers.Conv2D(filters=1, kernel_size=4, strides=1, padding="same",
                               kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False)(x)
    return tf.keras.Model(inputs=inputs, outputs=x)
