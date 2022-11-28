""" losses and model class"""
import pathlib
import tensorflow as tf
import generator_ETH
import discriminator
import read_data
import sys
from datetime import datetime
import numpy as np
import os



class DGMR():

  def __init__(self, strategy = None,
               generator_obj = None,
                discriminator_obj = None,
                generator_optimizer = tf.keras.optimizers.Adam(5e-5, beta_1=0.0), # adapted
                discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.0), # adapted
                epochs=3, batch_size=12,
                num_samples_per_input=2, eval_step=400, writer=None,
                save_model=False, num_valid_runs=100,
               ):
    self._generator = generator_obj
    self._discriminator = discriminator_obj
    self._gen_op = generator_optimizer
    self._disc_op = discriminator_optimizer
    self._epochs = epochs
    self.strategy = strategy
    self.batch_size = batch_size
    self.num_samples_per_input = num_samples_per_input
    self.steps_per_epoch = 10000000
    self.eval_step = eval_step
    self.writer = writer
    self.save_model = save_model
    self.num_valid = num_valid_runs
    tf.print("Epochs: {}, batch_size: {}, number of samples: {}"
             ", eval every {} steps".format(self._epochs, self.batch_size,
                                            self.num_samples_per_input, self.eval_step))
    tf.print("Stragey:", self.strategy)

  # predicts 18 frames from input of four frames
  @tf.function
  def predict(self, single_input):
    prediction = self._generator(single_input)
    return prediction

  @tf.function
  def distributed_train_step(self, dataset_inputs):
    per_replica_losses = self.strategy.run(self.train_step, args=(dataset_inputs,))
    print("per rp loss", per_replica_losses)
    x = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
    print("summed ", x)
    return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

  @tf.function
  def distributed_validation_step(self, dataset_inputs):
    per_replica_losses = self.strategy.run(self.validation_step, args=(dataset_inputs,))
    print("per rp loss", per_replica_losses)
    x = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
    print("summed ", x)
    return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

  def run(self, train_dataset, validation_dataset, example_images=None, ckpt_mg=None):
    ### added ####
    with self.strategy.scope():
      ####      ####

      create_images = False
      if example_images is not None:
        create_images = True
        # create evaluation images
      if create_images:
        images_iterator = iter(example_images)
        image_1 = next(images_iterator)
        image_2 = next(images_iterator)
        image_3 = next(images_iterator)
        image_4 = next(images_iterator)
        idx = 0

      for epoch in range(self._epochs):
        train_iterator = iter(train_dataset)
        tf.print("------------------------------------------------")
        tf.print("Epoch:", epoch)

        # run step
        for step in tf.range(self.steps_per_epoch):
          tf.print(step, "step")
          data = train_iterator.get_next_as_optional()
          if not data.has_value():
            break
          train_loss = self.distributed_train_step(data.get_value())
          summed_disc_loss, summed_gen_loss = tf.split(train_loss, 2, axis=0)
          with self.writer.as_default():
            tf.summary.scalar('Discriminator Loss', data=summed_disc_loss[0], step=tf.cast(step, tf.int64))
            tf.summary.scalar('Generator Loss', data=summed_gen_loss[0], step=tf.cast(step, tf.int64))
          tf.print("isc_loss", summed_disc_loss)
          tf.print("gen_loss", summed_gen_loss)

          # save model weights
          if step % 200 == 0:
            if self.save_model:
              save_path = ckpt_mg.save()
              tf.print("Saved checkpoint for step {}: {}".format(int(step), save_path))

          # evaluate performance
          if step % self.eval_step == 0:
            tf.print("Evaluating Validation Data")
            # create visualization
            if create_images:
              tf.print("Creating Visualization")
              save_observation = False
              if idx == 0:
                save_observation = True
              self.eval_image(image_1, "Example 1 ", idx, save_observation)
              self.eval_image(image_2, "Example 2 ", idx, save_observation)
              self.eval_image(image_3, "Example 3 ", idx, save_observation)
              self.eval_image(image_4, "Example 4 ", idx, save_observation)
              self.eval_image(next(images_iterator), "random Example", idx, True)
              idx += 1

            # calculate validation loss
            valid_iterator = iter(validation_dataset)
            first_loss = True
            for valid_step in tf.range(self.num_valid):
              tf.print("Eval step", valid_step)
              valid_data = valid_iterator.get_next_as_optional()
              if not data.has_value():
                break
              loss_ = self.distributed_validation_step(valid_data.get_value())
              if first_loss:
                loss = loss_
                first_loss = False
              else:
                loss = loss + loss_
            loss = loss / self.num_valid
            valid_disc_loss, valid_gen_loss = tf.split(loss, 2, axis=0)
            with self.writer.as_default():
              tf.summary.scalar('Validation Disc Loss', data=valid_disc_loss[0], step=tf.cast(step, tf.int64))
              tf.summary.scalar('Validation Gen Loss', data=valid_gen_loss[0], step=tf.cast(step, tf.int64))
            tf.print("valid_disc_loss", valid_disc_loss)
            tf.print("valid_gen_loss", valid_gen_loss)

  def train_step(self, frames):
      frames = tf.expand_dims(frames, -1)
      radar_frames = frames[0]
      eth_frames = frames[1]
      radar_batch_inputs, batch_targets = tf.split(radar_frames, [4, 18], axis=1)
      eth_batch_inputs, _ = tf.split(eth_frames, [4, 18], axis=1)
      real_sequence = tf.concat([radar_batch_inputs, batch_targets], axis=1)
      # train discriminator twice
      for _ in range(2):
        batch_predictions = self._generator(radar_batch_inputs, eth_batch_inputs)
        gen_sequence = tf.concat([radar_batch_inputs, batch_predictions], axis=1)
        concat_inputs = tf.concat([real_sequence, gen_sequence], axis=0)
        with tf.GradientTape() as disc_tape:
          concat_outputs = self._discriminator(concat_inputs)
          score_real, score_generated = tf.split(concat_outputs, 2, axis=0)
          disc_loss_dist = self.loss_hinge_disc_dist(score_generated, score_real)
        gradients_of_discriminator = disc_tape.gradient(disc_loss_dist, self._discriminator.trainable_variables)
        self._disc_op.apply_gradients(zip(gradients_of_discriminator, self._discriminator.trainable_variables))
      # train generator once
      for _ in range(1):
        with tf.GradientTape() as gen_tape:
          gen_samples = [
            self._generator(radar_batch_inputs, eth_batch_inputs) for _ in range(self.num_samples_per_input)]
          grid_cell_reg_dist = self.grid_cell_regularizer_dist(tf.stack(gen_samples, axis=0),
                                                          batch_targets)
          gen_sequences = [tf.concat([radar_batch_inputs, x], axis=1) for x in
                           gen_samples]  # from here on numpys as tf tensors
          gen_real_sequences = [tf.concat([x, real_sequence], axis=0) for x in gen_sequences]
          # Excpect error in pseudocode:
          #  gen_disc_loss = loss_hinge_gen(tf.concat(gen_sequences, axis=0))
          # changed to call discriminator on gen_sequence and caluculate loss on this output
          disc_output = [self._discriminator(x) for x in gen_real_sequences]
          gen_outputs = [tf.split(i, 2, axis=0)[0] for i in disc_output] # to only take output of gen samples
          gen_disc_loss_dist = self.loss_hinge_gen_dist(tf.concat(gen_outputs, axis=0))
          gen_loss = gen_disc_loss_dist + 20.0 * grid_cell_reg_dist
          tf.print("gen_loss", gen_loss)
        gradients_of_generator = gen_tape.gradient(gen_loss, self._generator.trainable_variables)
        self._gen_op.apply_gradients(zip(gradients_of_generator, self._generator.trainable_variables))

        return tf.stack([disc_loss_dist, gen_loss], axis=0)


  def validation_step(self, frames):

    frames = tf.expand_dims(frames, -1)
    radar_frames = frames[0]
    eth_frames = frames[1]
    radar_batch_inputs, batch_targets = tf.split(radar_frames, [4, 18], axis=1)
    eth_batch_inputs, _ = tf.split(eth_frames, [4, 18], axis=1)
    batch_predictions = self._generator(radar_batch_inputs, eth_batch_inputs)
    gen_sequence = tf.concat([radar_batch_inputs, batch_predictions], axis=1)
    real_sequence = tf.concat([radar_batch_inputs, batch_targets], axis=1)
    concat_inputs = tf.concat([real_sequence, gen_sequence], axis=0)
    concat_outputs = self._discriminator(concat_inputs)
    score_real, score_generated = tf.split(concat_outputs, 2, axis=0)
    # discriminator loss
    disc_loss = self.loss_hinge_disc_dist(score_generated, score_real)

    gen_samples = [
      self._generator(radar_batch_inputs, eth_batch_inputs) for _ in range(self.num_samples_per_input)]
    grid_cell_reg_dist = self.grid_cell_regularizer_dist(tf.stack(gen_samples, axis=0),
                                                    batch_targets)
    gen_sequences = [tf.concat([radar_batch_inputs, x], axis=1) for x in gen_samples]
    gen_real_sequences = [tf.concat([x, real_sequence], axis=0) for x in gen_sequences]
    disc_output = [self._discriminator(x) for x in gen_real_sequences]
    gen_outputs = [tf.split(i, 2, axis=0)[0] for i in disc_output] # to only take output of gen samples
    gen_disc_loss_dist = self.loss_hinge_gen_dist(tf.concat(gen_outputs, axis=0))
    # generator loss
    gen_loss = gen_disc_loss_dist + 20.0 * grid_cell_reg_dist

    return  tf.stack([disc_loss, gen_loss], axis=0)

  # save model prediction of given frames
  def eval_image(self, frames, name, idx, save_target = False):

      if save_target:
        frames = tf.expand_dims(frames, -1)
        real_radar_sequence = frames[0]
        with self.writer.as_default():
          target = np.reshape(real_radar_sequence, (-1, 256, 256, 1))
          tf.summary.image(name + " Observation", target, max_outputs=50, step=idx)

      frames = tf.expand_dims(frames, -1)
      radar_frames = frames[0]
      eth_frames = frames[1]
      radar_inputs, _ = tf.split(radar_frames, [4, 18], axis=1)
      eth_inputs, _ = tf.split(eth_frames, [4, 18], axis=1)
      predictions = self._generator(radar_inputs, eth_inputs)
      gen_sequence = tf.concat([radar_inputs, predictions], axis=1)
      with self.writer.as_default():
        generated = np.reshape(gen_sequence, (-1, 256, 256, 1))
        tf.summary.image(name, generated, max_outputs=50, step=idx)


  def loss_hinge_disc_dist(self, score_generated, score_real):
    """Discriminator hinge loss."""
    l1 = tf.nn.relu(1. - score_real)
    # divided by BATCH_SIZE * 2, as every loss score of two values: temporal and spatial
    loss = tf.reduce_sum(l1) * (1. / (self.batch_size * 2))
    l2 = tf.nn.relu(1. + score_generated)
    loss += tf.reduce_sum(l2) * (1. / (self.batch_size * 2))
    tf.print("loss hinge_disc", loss)
    return loss

  def loss_hinge_gen_dist(self, score_generated):
    """Generator hinge loss."""
    loss = -tf.reduce_sum(score_generated) * (1. / (self.batch_size * 2 * self.num_samples_per_input))
    tf.print("loss hinge_gen", loss)
    return loss

  def grid_cell_regularizer_dist(self, generated_samples, batch_targets):
    """Grid cell regularizer.
    Args:
      generated_samples: Tensor of size [n_samples, batch_size, 18, 256, 256, 1].
      batch_targets: Tensor of size [batch_size, 18, 256, 256, 1].
    Returns:
      loss: A tensor of shape [batch_size].
    """
    gen_mean = tf.reduce_mean(generated_samples, axis=0)
    # TODO check if clip at 24 could be raised
    weights = tf.clip_by_value(batch_targets, 0.0, 24.0)
    loss = tf.reduce_mean(tf.math.abs(gen_mean - batch_targets) * weights,
                          axis=list(range(1, len(batch_targets.shape))))
    loss = tf.reduce_sum(loss) * (1 / self.batch_size)
    tf.print("loss grid cell", loss)

    return loss

  @property
  def disc_op(self):
    return self._disc_op