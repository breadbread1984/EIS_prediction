#!/usr/bin/python3

from absl import flags, app
from shutil import rmtree
from os import mkdir, listdir
from os.path import join, exists, splitext
import pickle
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_dir', default = None, help = 'path to dataset')
  flags.DEFINE_string('output_dir', default = 'dataset', help = 'path to output directory')

def transformer():
  writer = tf.io.TFRecordWriter(join(FLAGS.output_dir, 'trainset.tfrecord'))
  for fname in listdir(join(FLAGS.input_dir, 'train_datasets')):
    stem, ext = splitext(fname)
    if ext != '.pkl': continue
    if not stem.startswith('train_pulse'): continue
    with open(join(FLAGS.input_dir, 'train_datasets', fname), 'rb') as f:
      pulse = pickle.load(f)
    with open(join(FLAGS.input_dir, 'train_datasets', fname.replace('pulse', 'eis')), 'rb') as f:
      eis = pickle.load(f)
    for SOC, pulse_samples in pulse.items():
      eis_samples = eis[SOC]
      x = tf.constant(np.stack([pulse_samples['Voltage'], pulse_samples['Current']], axis = -1))
      y = tf.constant(np.stack([eis_samples['Real'], eis_samples['Imaginary']], axis = -1))
      trainsample = tf.train.Example(features = tf.train.Features(
        feature = {
          'x': tf.train.Feature(bytes_list = tf.train.BytesList(value = [tf.io.serialize_tensor(x).numpy()])),
          'y': tf.train.Feature(bytes_list = tf.train.BytesList(value = [tf.io.serialize_tensor(y).numpy()]))
        }))
      writer.write(trainsample.SerializeToString())
  writer.close()

def main(unused_argv):
  if exists(FLAGS.output_dir): rmtree(FLAGS.output_dir)
  mkdir(FLAGS.output_dir)
  transformer()

if __name__ == "__main__":
  add_options()
  app.run(main)

