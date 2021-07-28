import os
import tensorflow as tf
import tensorflow_datasets as tfds

from image_operations import ImageOps


def download():
    """Download and load dataset."""
    data_dir = os.path.join(os.getcwd(), "data")
    dataset, metadata = tfds.load("cycle_gan/horse2zebra", with_info=True, as_supervised=True, data_dir=data_dir)
    train_horses, train_zebras = dataset["trainA"], dataset["trainB"]
    test_horses, test_zebras = dataset["testA"], dataset["testB"]
    return train_horses, train_zebras, test_horses, test_zebras


def preprocess(train_horses_in: tf.data.Dataset, train_zebras_in: tf.data.Dataset,
               test_horses_in: tf.data.Dataset, test_zebras_in: tf.data.Dataset):
    """Preprocess data."""
    ops = ImageOps(img_dimensions=[256, 256], resized_dimensions=[286, 286])
    autotune = tf.data.AUTOTUNE
    buffer_size = 2000
    batch_size = 1  # Might require retuning to use other batch sizes..  Also, uses too much GPU memory.
    seed = 2

    train_horses = train_horses_in.map(
        ops.preprocess_image_train, num_parallel_calls=autotune).cache().shuffle(buffer_size, seed).batch(batch_size)
    train_zebras = train_zebras_in.map(
        ops.preprocess_image_train, num_parallel_calls=autotune).cache().shuffle(buffer_size, seed).batch(batch_size)
    test_horses = test_horses_in.map(
        ops.preprocess_image_test, num_parallel_calls=autotune).cache().shuffle(buffer_size, seed).batch(batch_size)
    test_zebras = test_zebras_in.map(
        ops.preprocess_image_test, num_parallel_calls=autotune).cache().shuffle(buffer_size, seed).batch(batch_size)

    return train_horses, train_zebras, test_horses, test_zebras


def load():
    """Download and preprocess data."""
    train_horses, train_zebras, test_horses, test_zebras = download()
    return preprocess(train_horses, train_zebras, test_horses, test_zebras)
