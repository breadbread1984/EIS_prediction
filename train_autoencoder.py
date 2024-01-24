#!/usr/bin/python3

from os import listdir, mkdir
from os.path import exists, join, splitext
from absl import app, flags
import tensorflow as tf
from models import Trainer

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_enum('type', enum_values = {'pulse', 'eis'}, default = 'pulse', help = 'which type of encoder decoder is trained')
  flags.DEFINE_float('lr', default = 1e-3, help = 'learning rate')
  flags.DEFINE_integer('batch_size', default = 8, help = 'batch size')
  flags.DEFINE_integer('epoch', default = 200, help = 'epoch')
  flags.DEFINE_string('dataset', default = None, help = 'path to dataset')

def parse_function(serialized_example):
  feature = tf.io.parse_single_example(
    serialized_example,
    features = {
      'x': tf.io.FixedLenFeature((), dtype = tf.string),
    })
  if FLAGS.type == 'pulse':
    x = tf.io.parse_tensor(feature['x'], out_type = tf.float64)
    x = tf.reshape(x, (99,2))
  else:
    x = tf.io.parse_tensor(feature['x'], out_type = tf.float64)
    x = tf.reshape(x, (51,2))
  return tf.cast(x, dtype = tf.float32), tf.cast(x, dtype = tf.float32)

def main(unused_argv):
  trainer = Trainer()
  optimizer = tf.keras.optimizers.Adam(FLAGS.lr)
  dataset = tf.data.TFRecordDataset(join(FLAGS.dataset, 'trainset.tfrecord')).map(parse_function).prefetch(FLAGS.batch_size).shuffle(FLAGS.batch_size).batch(FLAGS.batch_size)
  for epoch in range(FLAGS.epoch):
    iterator = iter(dataset)
    for x, label in iterator:
      with tf.GradientTape() as tape:
        sample = trainer(x)
        loss = tf.reduce_mean(tf.abs(sample - label))
      print(loss)
      grads = tape.gradient(loss, trainer.trainable_variables)
      optimizer.apply_gradients(zip(grads, trainer.trainable_variables))
  trainer.save_weight('trainer.keras')

if __name__ == "__main__":
  add_options()
  app.run(main)
