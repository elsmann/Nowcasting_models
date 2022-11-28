"""Run DGMR model

  Args:
    1: directory of TFR files
    2: directory for tensorboard logs
    3: Optional: if "eager": turns eager execution on for debugging purposes

    example: $HOME/TFR  $HOME/TFR/logs
    example: $HOME/TFR  $HOME/TFR/logs eager """

import pathlib
import tensorflow as tf
import ETH_model
import read_data
import sys
import generator_ETH
import discriminator
from datetime import datetime
import numpy as np
import os

print("------Import successful------")

# os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

###  Training parameters ####

#############
Num_samples_per_input = 2  # default 6
Epochs = 3
Batch_size = 12
Steps_per_epoch = 15
Eval_step = 5
load_old_weights = False
Save_weights = False
checkpoints_dir = '/ETH_training_checkpoints_00'

############

base_directory = pathlib.Path(sys.argv[1])
log_dir = sys.argv[2]

tf.config.run_functions_eagerly(False)
if len(sys.argv) > 3:
  if sys.argv[3] == "eager":
    tf.config.run_functions_eagerly(True)
    mirrored_strategy = tf.distribute.get_strategy()
    tf.print("Running with eager execution")
    tf.print("Running without distribution")

else:
  mirrored_strategy = tf.distribute.MirroredStrategy()
  tf.print("Running with graph execution")

tf.print("Path of data used: {}".format(base_directory))
tf.print("Path of log: {}".format(log_dir))
tf.print("Checkpoints saved in {}".format(checkpoints_dir))
tf.print("{} Epochs \n{} Samples per Input\n"
         "{} Batch Size".format(Epochs, Num_samples_per_input, Batch_size))
tf.print('Number of devices: {}'.format(mirrored_strategy.num_replicas_in_sync))
tf.print(tf.config.list_physical_devices(
  device_type=None
))
print("----------------------------------")

# train dataset
train_dataset = read_data.read_TFR(base_directory, ETH=True, batch_size=Batch_size, ISS=200)
train_dataset = mirrored_strategy.experimental_distribute_dataset(train_dataset)

# validation dataset
validation_dataset = read_data.read_TFR(base_directory, ETH=True, batch_size=Batch_size, window_shift=40, ISS=400)
validation_dataset = mirrored_strategy.experimental_distribute_dataset(validation_dataset)

# dataset visualising images
image_set = read_data.read_TFR(base_directory, ETH=True, batch_size=1, window_shift=40, ISS=600)

stamp = datetime.now().strftime("%m%d-%H%M")
logdir = log_dir + "/func/ETH%s" % stamp + "B" + str(Batch_size)
writer = tf.summary.create_file_writer(logdir)

discriminator_o = discriminator.Discriminator()
generator_o = generator_ETH.Generator(lead_time=90, time_delta=5, strategy=mirrored_strategy)
model = ETH_model.DGMR(generator_obj=generator_o, discriminator_obj=discriminator_o,
                       epochs=Epochs, strategy=mirrored_strategy, batch_size=Batch_size, writer=writer,
                       eval_step=Eval_step,
                       save_model=Save_weights, num_samples_per_input=Num_samples_per_input)

checkpoint_dir = log_dir + checkpoints_dir
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
  generator_optimizer=model._gen_op,
  discriminator_optimizer=model._disc_op,
  generator=model._generator,
  discriminator=model._discriminator

)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=2)
if load_old_weights:
  checkpoint.restore(manager.latest_checkpoint)
  if manager.latest_checkpoint:
    tf.print("Restored from {}".format(manager.latest_checkpoint))
else:
  tf.print("Initializing from scratch.")

model.run(train_dataset=train_dataset, validation_dataset=validation_dataset,
          example_images=image_set, ckpt_mg=manager)
