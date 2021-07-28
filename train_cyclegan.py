import os
import tensorflow as tf
from models import CycleGan
import get_data
from image_operations import plot as ioplot
from buffer import Buffer


def train(model: CycleGan,
          x_data: tf.data.Dataset, y_data: tf.data.Dataset,
          sample_x: tf.Tensor,
          n_epochs: tf.Tensor, dataset_name: str):
    # Restore latest checkpoint
    start_epoch = tf.constant(0)
    if model.checkpoint_manager.latest_checkpoint:
        start_epoch = tf.cast(model.checkpoint.save_counter, dtype=tf.int32) * model.save_every_nth
        model.checkpoint.restore(model.checkpoint_manager.latest_checkpoint)

    # initialize buffers here
    size = 50
    x_buffer = Buffer(buffer_size=size, generator=model.generator_g, training_data=x_data)
    y_buffer = Buffer(buffer_size=size, generator=model.generator_f, training_data=y_data)

    for epoch in tf.range(start_epoch, n_epochs):
        start = tf.timestamp()

        n = 0
        for image_x, image_y in tf.data.Dataset.zip((x_data, y_data)):
            # manage buffer here
            # TODO: test out training model with buffer implemented
            buffer_image_x = x_buffer.update_buffer(image_x)
            buffer_image_y = y_buffer.update_buffer(image_y)

            model.train_step(image_x, image_y, buffer_image_x, buffer_image_y)
            if tf.math.equal(tf.math.floormod(n, 10), 0):
                tf.print('.', end='')
            n += 1

        # Using a consistent image (sample_x) so that the progress of the model is clearly visible.
        directory = os.path.join(os.getcwd(), "output", dataset_name)
        ioplot(model.generator_g, sample_x, epoch, directory)

        if tf.equal(tf.math.floormod(epoch + 1, model.save_every_nth), 0):
            save_path = model.checkpoint_manager.save()
            tf.print(f"Saving checkpoint for epoch {epoch + 1} at {save_path}")
        running_time = tf.timestamp() - start
        tf.print(f"Time taken for epoch {epoch + 1} is {running_time:.1f} sec\n")


if __name__ == "__main__":
    train_horses, train_zebras, test_horses, test_zebras = get_data.load()
    sample_horse = next(iter(train_horses))
    sample_zebra = next(iter(train_zebras))

    dataset_name = "horse2zebra"
    n_epochs = tf.constant(15)
    cyclegan = CycleGan(dataset_name=dataset_name, shape=sample_horse.shape)
    train(model=cyclegan, x_data=train_horses, y_data=train_zebras, sample_x=sample_horse,
          n_epochs=n_epochs, dataset_name=dataset_name)
