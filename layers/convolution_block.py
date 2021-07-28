from typing import Optional
import tensorflow as tf
from layers import ReflectionPadding2D
from tensorflow_addons.layers import InstanceNormalization


class ConvolutionBlock(tf.keras.layers.Layer):

    """Creates composite Conv2D(Transpose)-BatchNormalization-(Leaky)ReLU layer."""

    def __init__(self, kernel_size: int, stride: int, n_filters: int,
                 reflect_padding: bool = False,
                 transpose: bool = False,
                 normalize: bool = True,
                 activation: str = "relu",
                 leaky_slope: Optional[float] = None,
                 *args, **kwargs):
        """
        Parameters
        ----------
        kernel_size: int
            Kernel size.
        stride: int
            Stride size.
        n_filters: int
            Number of filters.
        reflect_padding: bool = False
            Set to True to do reflect padding instead of built-in "same" method.
        transpose: bool
            Set to True to do a transpose (upscaling) convolution instead.
            This should be what 'fractional stride' means.
        normalize: bool
            If set to True, normalize using InstanceNormalization.
        activation: str = "relu"
            String name of activation type.  Defaults to "relu".
        leaky_slope: float = None
            Slope of LeakyReLU layer.  This is the slope for x < 0.  Defaults to None.
        """
        super(ConvolutionBlock, self).__init__(*args, **kwargs)
        initializer = tf.random_normal_initializer(0., 0.02)

        if reflect_padding:
            self.rpad = ReflectionPadding2D(padding=(1, 1))
            method = "valid"
        else:
            self.rpad = None
            method = "same"

        if transpose:
            self.conv = tf.keras.layers.Conv2DTranspose(filters=n_filters, kernel_size=kernel_size, strides=stride,
                                                        padding=method, kernel_initializer=initializer, use_bias=False)
        else:
            self.conv = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=kernel_size, strides=stride,
                                               padding=method, kernel_initializer=initializer, use_bias=False)

        self.norm = None
        if normalize:
            self.norm = InstanceNormalization(axis=-1)

        if (activation == "relu") and (leaky_slope is not None):
            self.acti = tf.keras.layers.LeakyReLU(alpha=leaky_slope)
        else:
            self.acti = tf.keras.layers.Activation(activation=activation)

    def call(self, inputs: tf.Tensor):
        x = inputs
        if self.rpad is not None:
            x = self.rpad(x)
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        return self.acti(x)
