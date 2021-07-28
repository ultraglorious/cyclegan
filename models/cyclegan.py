import os
import tensorflow as tf
import models


class CycleGan(tf.keras.Model):

    def __init__(self, dataset_name: str, shape: tf.TensorShape):
        """
        Parameters
        ----------
        dataset_name: str
            String name of dataset.  Used for checkpoint directory navigation.
        shape: tf.TensorShape
            TensorShape of batch.
        """
        super(CycleGan, self).__init__()
        self.loss_enhancement = tf.constant(10.)
        self.loss_reduction = tf.constant(0.5)
        self.loss_object = tf.keras.losses.MeanSquaredError()
        self.models(shape)
        self.optimizers()
        self.checkpointing(dataset_name)
        self.restore_checkpoint()

    def models(self, shape: tf.TensorShape):
        """
        Set up generator (artist) and discriminator (critic) models.
        G(x image) -> new image in style of y
        F(y image) -> new image in style of x
        D(new x) -> is new x image in true style of x?
        D(new y) -> is new y image in true style of y?
        """
        self.generator_g = models.generator(shape)
        self.generator_f = models.generator(shape)
        self.discriminator_x = models.discriminator(shape)
        self.discriminator_y = models.discriminator(shape)

    def optimizers(self):
        """"Set up optimizers"""
        learning_rate = 2e-4
        beta1 = 0.5  # The exponential decay rate for the 1st moment estimates. (Not sure what this is really)
        self.generator_g_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=beta1)
        self.generator_f_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=beta1)
        self.discriminator_x_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=beta1)
        self.discriminator_y_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=beta1)

    def checkpointing(self, dataset_name: str):
        """Set up checkpointing"""
        self.save_every_nth = tf.constant(5)  # Save every nth checkpoint
        checkpoint_path = os.path.join(os.getcwd(), "checkpoints", dataset_name, "train")
        self.checkpoint = tf.train.Checkpoint(generator_g=self.generator_g,
                                              generator_f=self.generator_f,
                                              discriminator_x=self.discriminator_x,
                                              discriminator_y=self.discriminator_y,
                                              generator_g_optimizer=self.generator_g_optimizer,
                                              generator_f_optimizer=self.generator_f_optimizer,
                                              discriminator_x_optimizer=self.discriminator_x_optimizer,
                                              discriminator_y_optimizer=self.discriminator_y_optimizer)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, checkpoint_path, max_to_keep=5)

    def restore_checkpoint(self):
        """If a checkpoint exists, restore the latest checkpoint."""
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print("Latest checkpoint restored.")

    def discriminator_loss(self, real_image: tf.Tensor, generated_image: tf.Tensor) -> tf.Tensor:
        """
        Discriminator wants to get better at detecting that the real image is real, and that generated images are not.
        """
        real_loss = self.loss_object(tf.ones_like(real_image), real_image)  # Real images are real
        gen_loss = self.loss_object(tf.zeros_like(generated_image), generated_image)  # Generated images are not
        return (real_loss + gen_loss) * self.loss_reduction

    def generator_loss(self, generated_image: tf.Tensor) -> tf.Tensor:
        """Generator wants to get closer to creating a realistic image."""
        return self.loss_object(tf.ones_like(generated_image), generated_image)

    def cycle_loss(self, real_image: tf.Tensor, cycled_image: tf.Tensor) -> tf.Tensor:
        """We want to minimize the difference between the real image, x, and its cycled state, F(G(x))."""
        return tf.reduce_mean(tf.abs(real_image - cycled_image)) * self.loss_enhancement

    def identity_loss(self, real_image: tf.Tensor, same_image: tf.Tensor) -> tf.Tensor:
        """
        We want to minimize the difference between image y and G(y) (G normally makes image x y-like).
        Generator model G should know it doesn't have to do much work to convert y to be y-like.
        """
        return tf.reduce_mean(tf.abs(real_image - same_image)) * self.loss_reduction * self.loss_enhancement

    @tf.function
    def train_step(self, real_x: tf.Tensor, real_y: tf.Tensor, buffer_x: tf.Tensor, buffer_y: tf.Tensor):
        """All inputs are images."""
        # We are calculating the gradient of multiple 'functions' so persistent is set to True.
        with tf.GradientTape(persistent=True) as tape:
            # Generator G translates X -> Y
            # Generator F translates Y -> X.

            fake_y = self.generator_g(real_x, training=True)  # Fake zebra: horse -> zebra
            cycled_x = self.generator_f(fake_y, training=True)  # Cycled horse: horse -> zebra -> horse

            fake_x = self.generator_f(real_y, training=True)  # Fake horse: zebra -> horse
            cycled_y = self.generator_g(fake_x, training=True)  # Cycled zebra: zebra -> horse -> zebra

            same_x = self.generator_f(real_x, training=True)  # Horse -> horse (ideally no change to image)
            same_y = self.generator_g(real_y, training=True)  # Zebra -> zebra

            disc_real_x = self.discriminator_x(real_x, training=True)  # Disc. assesses whether real horse is a horse
            disc_real_y = self.discriminator_y(real_y, training=True)  # Disc. assesses whether real zebra is a zebra

            disc_fake_x = self.discriminator_x(fake_x, training=True)  # Disc. assesses whether fake horse is a horse
            disc_fake_y = self.discriminator_y(fake_y, training=True)  # Disc. assesses whether fake zebra is a zebra

            # Calculate the generator's loss
            gen_g_loss = self.generator_loss(disc_fake_y)  # How close is fake zebra to real zebra?
            gen_f_loss = self.generator_loss(disc_fake_x)  # How close is fake horse to real horse?
            # How close is cycled zebra (horse) to real zebra (horse)? Sum of both directions
            total_cycle_loss = self.cycle_loss(real_x, cycled_x) + self.cycle_loss(real_y, cycled_y)
            # Total generator loss = adversarial loss + cycle loss
            total_gen_g_loss = gen_g_loss + total_cycle_loss + self.identity_loss(real_y, same_y)
            total_gen_f_loss = gen_f_loss + total_cycle_loss + self.identity_loss(real_x, same_x)

            # Calculate the discriminator's loss - using buffer image instead of latest fake image
            # Can the discriminator tell if fake horse or zebra belongs to the set of horse and zebra images?
            disc_buffer_x = self.discriminator_x(buffer_x, training=True)  # Disc. assesses whether fake horse is a horse
            disc_buffer_y = self.discriminator_y(buffer_y, training=True)  # Disc. assesses whether fake zebra is a zebra

            disc_x_loss = self.discriminator_loss(disc_real_x, disc_buffer_x)
            disc_y_loss = self.discriminator_loss(disc_real_y, disc_buffer_y)

        # Calculate the gradients for generator and discriminator
        generator_g_gradients = tape.gradient(total_gen_g_loss, self.generator_g.trainable_variables)
        generator_f_gradients = tape.gradient(total_gen_f_loss, self.generator_f.trainable_variables)
        discriminator_x_gradients = tape.gradient(disc_x_loss, self.discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss, self.discriminator_y.trainable_variables)

        # Ask the optimizers to apply the gradients to the trainable variables
        self.generator_g_optimizer.apply_gradients(zip(generator_g_gradients, self.generator_g.trainable_variables))
        self.generator_f_optimizer.apply_gradients(zip(generator_f_gradients, self.generator_f.trainable_variables))
        self.discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                           self.discriminator_x.trainable_variables))
        self.discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                           self.discriminator_y.trainable_variables))
