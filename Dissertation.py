system_path = './'
checkpoint_dir = './ProstateX/training_checkpoints'
img_dir = './ProstateX/AugmentedImages'

import sys
sys.path.insert(0, system_path)

# Check that imports for the rest of the file work.
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from datetime import date, datetime
import make_generator_model
import make_discriminator_model
import TrainingUtils
import Eval

# Allow matplotlib images to render immediately.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Disable noisy outputs.

dataset = './ProstateX/np_dataset_2.npy'

BUFFER_SIZE = 1000
BATCH_SIZE = 30
EPOCHS = 1000
noise_dim = 400
num_examples_to_generate = BATCH_SIZE
seed = np.random.normal(0,1,size=[num_examples_to_generate, noise_dim])
utils = TrainingUtils(dataset, seed=seed)

generator_loss_arr = []
discriminator_loss_arr = []
fid_scores = []
steps = []
timestamp = int(datetime.now().timestamp())
today = date.today()
id = utils.get_random_id()
today = today.strftime("%d-%m-%Y")

save_gen_dir = './ProstateX/training_checkpoints/gen-model-{}-{}'.format(today, id)
save_disc_dir = './ProstateX/training_checkpoints/discriminator-model-{}-{}'.format(today, id)
save_fid_dir = './ProstateX/fid_{}_{}_{}'.format(today, id, timestamp)
save_gen_loss_dir = './ProstateX/generator_loss_{}_{}_{}'.format(today, id, timestamp)
save_disc_loss_dir = './ProstateX/discriminator_loss_{}_{}_{}'.format(today, id, timestamp)
augmented_images_dir = './ProstateX/AugmentedImages/{}-{}-gan_images/'.format(today, id)
augmented_image_preview_dir = './ProstateX/image_preview'
save_seed_dir = './ProstateX/seed_{}_{}_{}'.format(today, id, timestamp)

generator = make_generator_model()
discriminator = make_discriminator_model()

training_dataset = utils.get_training_dataset()
train_dataset = tf.data.Dataset.from_tensor_slices(training_dataset).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

generator_optimizer = utils.get_optimizer()
discriminator_optimizer = utils.get_optimizer()

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

def view_images(lists, date="", epoch=0, run_id="", save = False):
  fig, axes = plt.subplots(nrows=1, ncols=lists.shape[3], figsize=(20, 20), squeeze=False)
  plt.subplots_adjust(wspace=10, hspace=10)
  fig.tight_layout()
  for i, ax_row in enumerate(axes):
    for j, ax_col in enumerate(ax_row):
      sl = j;
      pred = lists[i]
      ax_col.imshow(pred[:,:,sl],cmap='gray')
      ax_col.axis('off')

      if(save):
        image_dir = augmented_images_dir
        file_name = '{}/image_at_epoch_{:04d}.png'.format(image_dir, epoch)
        if run_id == "":
          image_dir = augmented_image_preview_dir
          file_name = '{}/image_{}.png'.format(image_dir,utils.get_random_id())

        utils.save_plt_image(plt, image_dir, file_name)
  plt.show()

def generate_and_save_images(model, epoch, test_input, run_id, date, timestamp):
  predictions = model(test_input, training=False)
  view_images(predictions, date, epoch, run_id, True)

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(utils.get_seed(), training=True)
      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)
      gen_loss = utils.compute_generator_loss(fake_output)
      disc_loss = utils.compute_discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss, disc_loss

def compute_stats():
  real_data =train_dataset.take(1)
  real_data = next(iter(train_dataset))

  generated_images = generator(utils.get_seed(), training=True)
  real_output = discriminator(real_data, training=True)
  fake_output = discriminator(generated_images, training=True)
  gen_loss = utils.compute_generator_loss(fake_output)
  disc_loss = utils.compute_discriminator_loss(real_output, fake_output)

  eval = Eval()
  casted_gen_data = tf.cast(generated_images, dtype=tf.float64)
  fid_score = eval.get_fid(real_data, casted_gen_data)
  print("fid score: {}".format(fid_score))

  fid_scores.append(fid_score)
  generator_loss_arr.append(tf.keras.backend.get_value(gen_loss))
  discriminator_loss_arr.append(tf.keras.backend.get_value(disc_loss))

def train(dataset, epochs):
  overall_start = time.time()
  for epoch in range(epochs):
    start = time.time()
    for image_batch in dataset:
      gen_loss, disc_loss = train_step(image_batch)

    if (epoch + 1) % 1000 == 0:
      print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-overall_start))
      print("saving checkpoint")
      checkpoint.save(file_prefix = checkpoint_prefix)
      generate_and_save_images(generator,
                           epoch+1,
                           utils.get_seed(), id, today, timestamp)
      compute_stats()
      print("stats computed")
      overall_start = time.time()

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  generate_and_save_images(generator,
                           epochs,
                           utils.get_seed(), id, today, timestamp)

train(train_dataset, EPOCHS)

np.save(save_gen_loss_dir, generator_loss_arr)
np.save(save_disc_loss_dir, discriminator_loss_arr)
np.save(save_fid_dir, fid_scores)

plt.title('Generator loss per step')
plt.plot(generator_loss_arr, 'g--', discriminator_loss_arr, 'r--')
plt.legend(['Generator Loss', 'Discriminator loss'], loc='upper left')
utils.save_plt_image(plt, augmented_images_dir, '{}/{}'.format(augmented_images_dir, id))
plt.figure()

plt.title('Fid Scores')
plt.plot(fid_scores)
utils.save_plt_image(plt, augmented_images_dir, '{}/fid-{}'.format(augmented_images_dir, id))
plt.figure()

img_dir = '{}/{}'.format(img_dir, id)
utils.save_images(generator, img_dir)

generator.save("{}.keras".format(save_gen_dir))
discriminator.save("{}.keras".format(save_disc_dir))

seed = utils.get_seed()
np.save(save_seed_dir, seed)
