#!/usr/bin/python3

from os import listdir, mkdir
from os.path import exists, join,splitext
from absl import app, flags
import numpy as np
import tensorflow as tf
from models import Trainer

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_float('lr', default = 1e-2, help = 'learning rate')
  flags.DEFINE_integer('batch_size', default = 32, help = 'batch size')
  flags.DEFINE_integer('epoch', default = 600, help = 'epoch')
  flags.DEFINE_string('dataset', default = None, help = 'path to dataset')
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to checkpoint')
  flags.DEFINE_integer('save_freq', default = 100, help = 'save frequency')

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

def loss(label,pred):
  dists = tf.math.sqrt(tf.math.reduce_sum((label - pred)**2, axis = -1))
  loss = tf.math.reduce_mean(dists)
  return loss

def main(unused_argv):
  trainer = Trainer()
  optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.ExponentialDecay(FLAGS.lr, decay_steps = 1000, decay_rate = 0.96))
  metrics = [tf.keras.metrics.MeanSquaredError(), tf.keras.losses.MeanAbsoluteError()]

  trainset = tf.data.TFRecordDataset(join(FLAGS.dataset, 'trainset.tfrecord')).map(parse_function).prefetch(FLAGS.batch_size).shuffle(FLAGS.batch_size).batch(FLAGS.batch_size)
  valset = tf.data.TFRecordDataset(join(FLAGS.dataset, 'valset.tfrecord')).map(parse_function).prefetch(FLAGS.batch_size).shuffle(FLAGS.batch_size).batch(FLAGS.batch_size)

  if exists(FLAGS.ckpt): trainer.load_weights(join(FLAGS.ckpt, 'ckpt', 'variables', 'variables'))
  trainer.compile(optimizer = optimizer, loss = loss, metrics = metrics)
  callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir = FLAGS.ckpt),
    tf.keras.callbacks.ModelCheckpoint(filepath = join(FLAGS.ckpt, 'ckpt'), save_freq = FLAGS.save_freq)
  ]
  trainer.fit(trainset, epochs = FLAGS.epoch, validation_data = valset, callbacks = callbacks)

if __name__ == "__main__":
  add_options()
  app.run(main)
