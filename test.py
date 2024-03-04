#!/usr/bin/python3

from absl import flags, app
from os import mkdir, listdir
from os.path import join, exists, splitext
import pickle
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from models import Trainer
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('dataset', default = None, help = 'path to dataset')
  flags.DEFINE_integer('batch_size', default = 32, help = 'batch size')
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to checkpoint')

def parse_function(serialized_example):
  feature = tf.io.parse_single_example(
    serialized_example,
    features = {
      'x': tf.io.FixedLenFeature((), dtype = tf.string),
      'y': tf.io.FixedLenFeature((), dtype = tf.string)
    })
  x = tf.io.parse_tensor(feature['x'], out_type = tf.float64)
  x = tf.reshape(x, (1800, 2))
  y = tf.io.parse_tensor(feature['y'], out_type = tf.float64)
  y = tf.reshape(y, (35, 2))
  return tf.cast(x, dtype = tf.float32), tf.cast(y, dtype = tf.float32)

def main(unused_argv):
  trainer = Trainer()
  trainer.load_weights(join(FLAGS.ckpt, 'ckpt', 'variables', 'variables'))

  dataset = tf.data.TFRecordDataset([join(FLAGS.dataset, 'trainset.tfrecord'), join(FLAGS.dataset, 'valset.tfrecord')]).map(parse_function).prefetch(FLAGS.batch_size).shuffle(FLAGS.batch_size).batch(FLAGS.batch_size)
  global_index = 0
  max_dist = tf.constant(0, dtype = tf.float32)
  for pulse, label in dataset:
    eis = trainer(pulse)
    eis = tf.stack([tf.math.exp(eis[:,0]), tf.math.log(eis[:,1])], axis = -1)
    for p, l in zip(eis, label):
      # p.shape = (35,2) l.shape = (35,2)
      plt.cla()
      plt.plot(p[:,0].numpy(),p[:,1].numpy(),label = 'prediction')
      plt.plot(l[:,0].numpy(),l[:,1].numpy(),label = 'ground truth')
      plt.legend()
      plt.savefig('%d.png' % global_index)
      global_index += 1
    dist = tf.math.sqrt(tf.math.reduce_sum((eis - label) ** 2, axis = -1)) # diff.shape = (batch, 35)
    m_dist = tf.math.reduce_max(dist, axis = (0,1)) # m_dist.shape = ()
    max_dist = tf.maximum(max_dist, m_dist)
  print("max distance: ", max_dist)

if __name__ == "__main__":
  add_options()
  app.run(main)

