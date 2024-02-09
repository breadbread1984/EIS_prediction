#!/usr/bin/python3

from absl import flags, app
from os import mkdir, listdir
from os.path import join, exists, splitext
import pickle
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from alternative_models import Trainer

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
  checkpoint = tf.train.Checkpoint(model = trainer)
  checkpoint.restore(tf.train.latest_checkpoint(join(FLAGS.ckpt, 'ckpt')))
  sos = tf.constant(np.load('sos.npy'))

  dataset = tf.data.TFRecordDataset([join(FLAGS.dataset, 'trainset.tfrecord'), join(FLAGS.dataset, 'valset.tfrecord')]).map(parse_function).prefetch(FLAGS.batch_size).shuffle(FLAGS.batch_size).batch(FLAGS.batch_size)
  max_diff = tf.zeros((2,))
  for pulse, label in dataset:
    eis = tf.tile(sos, (pulse.shape[0],1,1))
    for i in range(35):
      pred = trainer([pulse, eis])
      eis = tf.concat([eis, pred[:, -1:, :]], axis = -2)
    eis = eis[:,1:,:]
    diff = tf.reduce_max(tf.abs(eis - label), axis = (0,1))
    max_diff = tf.maximum(max_diff, diff)
  print("max real difference: ", max_diff[0], "max imaginary difference: ", max_diff[1])

if __name__ == "__main__":
  add_options()
  app.run(main)

