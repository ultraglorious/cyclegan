import tensorflow as tf


class Buffer:

    """The data buffer is a Tensor of past generator data to select new discriminator images from."""

    def __init__(self, buffer_size: int, generator: tf.keras.Model, training_data: tf.data.Dataset):
        """Populate the buffer with an initial set of 50 images from the untrained generator."""
        self.buffer_size = buffer_size
        self.rng = tf.random.Generator.from_non_deterministic_state()
        for i, image in training_data.take(self.buffer_size).enumerate():
            if i == 0:
                buffer = tf.TensorArray(dtype=image.dtype, size=self.buffer_size)
            buffer = buffer.write(i, generator(image)[0, ...])
        self.buffer_variable = tf.Variable(buffer.stack())

    def get(self):
        """Converts buffer tf.Variable to Tensor."""
        return tf.convert_to_tensor(self.buffer_variable)

    @tf.function
    def update_buffer(self, images: tf.Tensor) -> tf.Tensor:
        """
        Update pool of fake data for training the discriminator model.

        Parameters
        ----------
        images: tf.Tensor
            Latest batch of training images to update the buffer with.
        Returns
        -------
        Tuple[tf.Tensor, tf.TensorArray]
            Tuple of the tf.Tensor containing the images for training the discriminator, and the tf.TensorArray buffer.
        """
        output_images = tf.TensorArray(dtype=images.dtype, size=images.shape[0])
        n_output = tf.constant(0)

        for image in images:
            if self.rng.uniform(shape=(1,), minval=0, maxval=1, dtype=tf.float32) < 0.5:
                # 50% chance of using image and not adding it to the buffer.
                output_images = output_images.write(n_output, image)
                n_output += 1
            else:
                # 50% chance of replacing an image in the buffer and using the replaced image.
                buffer = self.get()
                i = self.rng.uniform(shape=[], minval=0, maxval=self.buffer_size, dtype=tf.int32)
                replaced_image = buffer[i, ...]
                output_images = output_images.write(n_output, replaced_image)
                new_buffer = tf.concat(
                    [buffer[0:i, ...],
                     image[tf.newaxis, ...],
                     buffer[i + 1:, ...]],
                    axis=0)
                self.buffer_variable.assign(new_buffer)
                n_output += 1
        return output_images.stack()
