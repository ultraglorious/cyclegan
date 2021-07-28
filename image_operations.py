import os
from typing import Optional, List
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec as gridspec


class ImageOps:
    def __init__(self, img_dimensions: List[int], resized_dimensions: List[int]):
        """
        Parameters
        ----------
        img_dimensions: List[int]
            Output image (height, width) in pixels.
        resized_dimensions: List[int]
            Resized image (to introduce jitter) (height, width) in pixels.
        """
        self.img_height = img_dimensions[0]
        self.img_width = img_dimensions[1]
        self.resized_height = resized_dimensions[0]
        self.resized_width = resized_dimensions[1]

    def resize(self, input_image):
        """Resize image"""
        dims = [self.resized_height, self.resized_width]
        im_out = tf.image.resize(input_image, dims, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return im_out

    @staticmethod
    def normalize(image):
        """Normalize the images to [-1, 1]."""
        image = tf.cast(image, tf.float32)
        image = (image / 127.5) - 1
        return image

    def random_crop(self, image):
        """Randomly crop image down to desired output dimensions."""
        return tf.image.random_crop(image, size=[self.img_height, self.img_width, 3])

    @tf.function
    def random_jitter(self, image_in):
        """Resize and then crop image down to desired output dimensions."""
        # resizing to 286 x 286 x 3
        image_out = self.resize(image_in,)
        # randomly cropping to 256 x 256 x 3
        image_out = self.random_crop(image_out)
        # random mirroring
        image_out = tf.image.random_flip_left_right(image_out)
        return image_out

    @tf.function
    def preprocess_image_train(self, image, label):
        """Mapping function for training images."""
        image = self.random_jitter(image)
        image = self.normalize(image)
        return image

    @tf.function
    def preprocess_image_test(self, image, label):
        """Mapping function for testing images (no jitter)."""
        image = self.normalize(image)
        return image


def normalize_tensor(t: tf.Tensor) -> tf.Tensor:
    """Adjust the pixel values to between [0, 1] before plotting it."""
    return (t - tf.reduce_min(t)) / (tf.reduce_max(t) - tf.reduce_min(t))


def plot(generator_g: tf.keras.Model, test_input: tf.Tensor,
         epoch: Optional[int] = None, directory: Optional[str] = None):

    if epoch is not None:
        backend = "Agg"
    else:
        backend = "TkAgg"
    if mpl.get_backend() != backend:
        mpl.use(backend)  # Do not create a plotting window but plot to a buffer

    filename = "image_at_epoch_{:04d}.png"
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, figure=fig)

    ax1 = fig.add_subplot(gs[:, 0])
    ax1.imshow(normalize_tensor(test_input[0]))
    ax1.set_title("Input Image")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[:, 1])
    ax2.imshow(normalize_tensor(generator_g(test_input)[0]))
    ax2.set_title("Predicted Image")
    ax2.axis("off")

    if epoch is not None:
        if directory is not None:
            os.makedirs(directory, exist_ok=True)
            fp = os.path.join(directory, filename)
        else:
            fp = filename
        plt.savefig(fp.format(epoch))
    else:
        plt.show()
