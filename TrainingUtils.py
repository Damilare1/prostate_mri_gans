import sys
import tensorflow as tf
import random
import string
import numpy as np
import os
import cv2

class TrainingUtils:

  __crossEntropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

  __optimizer = tf.keras.optimizers.Adam

  __training_dataset_path = ''

  __seed = tf.clip_by_value(np.random.normal(0,1,size=[2, 160]),clip_value_min=-1.0, clip_value_max=1.0)

  def __init__(self, dataset_path, crossEntropy = None, opt = None, seed = None) -> None:
    self.__training_dataset_path = dataset_path
    if crossEntropy is not None:
      self.__crossEntropy = crossEntropy
    if opt is not None:
      self.__optimizer = opt
    if seed is not None:
      self.__seed = seed
  
  @tf.autograph.experimental.do_not_convert 
  def get_cross_entropy(self):
    return self.__crossEntropy

  def get_training_dataset_path(self):
    return self.__training_dataset_path

  def get_training_dataset(self):
    return np.load(self.get_training_dataset_path())

  @tf.autograph.experimental.do_not_convert
  def get_seed(self):
    return self.__seed
  
  def set_seed(self, seed):
    seed = tf.clip_by_value(seed,clip_value_min=-1.0, clip_value_max=1.0)
    self.__seed = seed
    return self

  def compute_discriminator_loss(self, real_output, fake_output):
    cross_entropy = self.get_cross_entropy()
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

  def compute_generator_loss(self, fake_output):
    cross_entropy = self.get_cross_entropy()
    return cross_entropy(tf.ones_like(fake_output), fake_output)

  def get_optimizer(self, opt= 'Adam', **kwargs):
    opt_fn = None
    if hasattr(tf.keras.optimizers, opt):
        opt_fn = getattr(tf.keras.optimizers, opt)
    elif hasattr(tf.keras.optimizers.legacy, opt):
        opt_fn = getattr(tf.keras.optimizers.legacy, opt)
    else:
        raise ValueError(f"Optimizer '{opt}' not found.")
    optimizer = opt_fn(**kwargs)
    return optimizer
  
  def __id_generator(self, size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

  def get_random_id(self, size=6):
    return self.__id_generator(size)
  
  def normalize3DArray(self, array3d):
    return (array3d - 127.5)/127.5
  
  def save_plt_image(self, plot, dir, file_name):
    os.makedirs(dir, exist_ok=True)
    plot.savefig(file_name)

  def save_images(self, model, img_dir):
    predictions = model(self.get_seed(), training=False)
    arr = predictions.numpy()
    for i in range(arr.shape[0]):
      sub_dir = i
      dir = '{}/{}'.format(img_dir, sub_dir)
      os.makedirs(dir, exist_ok=True)
      for j in range(arr.shape[3]):
        cv2.imwrite('{}/{}.jpg'.format(dir, j), (arr[i,:,:,j,0]*127.5) + 127.5)

sys.modules[__name__] = TrainingUtils
