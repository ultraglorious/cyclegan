import numpy as np
import tensorflow as tf
import get_data
from models import generator, discriminator
from image_operations import plot
from buffer import Buffer
import matplotlib.pyplot as plt
from layers import ConvolutionBlock, ResidualBlock


if __name__ == "__main__":
    train_horses, train_zebras, test_horses, test_zebras = get_data.load()
    sample_horse = next(iter(train_horses))
    sample_zebra = next(iter(train_zebras))

    res = 256
    n = res*res*3
    x = tf.constant(np.arange(0., n, 1).reshape([1, res, res, 3]) / n, dtype=tf.float32)

    gen = generator(sample_horse.shape)
    disc = discriminator(x.shape)
    g_out = gen(x)
    d_out = disc(x)

    # buff = Buffer(buffer_size=10, generator=gen, training_data=train_horses)

    # plot(gen, sample_horse)

    cb = ConvolutionBlock(kernel_size=3, stride=1, n_filters=256, reflect_padding=True, normalize=True)
