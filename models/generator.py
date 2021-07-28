import tensorflow as tf
import layers.shortcuts as sc


class GeneratorException(Exception):
    pass


def generator(shape: tf.TensorShape) -> tf.keras.Model:
    """
    Functional (as opposed to subclassed) version of Generator model.

    From paper:
    "We use 6 residual blocks for 128x128 training images and 9 residual blocks for 256x256 or higher-resolution
    training images..."

    "The network with 6 residual blocks consists of:
    c7s1-64, d128, d256, R256, R256, R256, R256, R256, R256, u128, u64, c7s1-3"

    "The network with 9 residual blocks consists of:
    c7s1-64, d128, d256, R256, R256, R256, R256, R256, R256, R256, R256, R256, u128, u64, c7s1-3"
    """

    if shape.ndims == 4:
        longest_dimension = tf.reduce_max(shape[1:3])
    else:
        raise GeneratorException("Number of dimensions must be 4.")

    if tf.math.equal(longest_dimension, 128):
        n_res_blocks = 6
    elif tf.math.greater_equal(longest_dimension, 256):
        n_res_blocks = 9
    else:
        # In case of something smaller, just use the six layer block
        n_res_blocks = 6

    inputs = tf.keras.layers.Input(shape=shape[1:], batch_size=shape[0])
    x = sc.c7s1k(64)(inputs)  # (bs, res, res, 64)
    x = sc.dk(128)(x)  # (bs, res/2, res/2, 128)
    x = sc.dk(256)(x)  # (bs, res/4, res/4, 256)
    for i in tf.range(n_res_blocks):
        x = sc.rk(256)(x)  # shape stays constant through residual layers
    x = sc.uk(128)(x)  # (bs, res/2, res/2, 128)
    x = sc.uk(64)(x)  # (bs, res, res, 64)
    x = sc.c7s1k(3, activation="tanh")(x)  # (bs, res, res, 3)
    return tf.keras.Model(inputs=inputs, outputs=x)
