from typing import Tuple
import tensorflow as tf


class ReflectionPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding: Tuple[int, int] = (1, 1)):
        """
        Implements reflection padding for Conv2D layer.
        Modelled after https://stackoverflow.com/questions/50677544/reflection-padding-conv2d with changes.

        The padding might lead to a output with a slightly different shape if the kernel is even.

        Parameters
        ----------
        padding : Tuple[int, int]
            Tuple of vertical and horizontal paddings, in that order.
            Total size change to dimension is twice the pad size (padding in both directions).
        """
        self.padding = tuple(padding)
        self.input_spec = [tf.keras.layers.InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__()

    def compute_output_shape(self, input_shape) -> tuple:
        # Check for undefined dimensions
        if input_shape[1] is None:
            return None, None, None, input_shape[3]
        # Do nothing to first and last dimensions (batch size and number of features)
        output_shape = (
            input_shape[0],
            input_shape[1] + 2 * self.padding[0],
            input_shape[2] + 2 * self.padding[1],
            input_shape[3]
        )
        return output_shape

    def call(self, input_tensor, *args, **kwargs):
        v_pad, h_pad = self.padding
        paddings = [[0, 0], [v_pad, v_pad], [h_pad, h_pad], [0, 0]]
        return tf.pad(input_tensor, paddings, "REFLECT")
