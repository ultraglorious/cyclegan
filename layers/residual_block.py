import tensorflow as tf
from tensorflow_addons.layers import InstanceNormalization


class ResidualBlock(tf.keras.layers.Layer):

    """ResNet residual block"""

    def __init__(self, n_filters: int, stride: int, change_n_channels: bool = False, *args, **kwargs):
        """
        Parameters
        ----------
        n_filters: int
        stride: int
        change_n_channels: bool
            If the convolutions change the number of channels/filters from the input, then a 1x1 kernel size
            convolution is needed to resize the inputs when added at the end of the block.  Defaults to False.
        """
        super(ResidualBlock, self).__init__(*args, **kwargs)
        initializer = tf.random_normal_initializer(0., 0.02)
        # Apparently BatchNormalization makes the use of bias terms unnecessary.  Not sure why yet.
        self.conv1 = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=3, strides=stride, padding="same",
                                            kernel_initializer=initializer, use_bias=False)
        self.n1 = InstanceNormalization(axis=-1)
        # In Residual block implementations I have seen, the stride here is hardcoded as 1.
        self.conv2 = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=3, strides=1, padding="same",
                                            kernel_initializer=initializer, use_bias=False)
        self.n2 = InstanceNormalization(axis=-1)
        self.conv3 = None
        if change_n_channels:
            self.conv3 = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=1, strides=stride,
                                                kernel_initializer=initializer, use_bias=False)

    def call(self, inputs: tf.Tensor):
        x = tf.keras.activations.relu(self.n1(self.conv1(inputs)))
        x = self.n2(self.conv2(x))
        if self.conv3 is not None:
            x = self.conv3(x)
        x += inputs
        # Try depth-concatenating here instead of adding and see how that does.
        # Depth-concatenating seems superior but the number of trainable parameters increases a lot.
        # x = tf.keras.layers.Concatenate()([x, inputs])
        return tf.keras.activations.relu(x)
