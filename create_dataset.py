#!/usr/bin/python3

from absl import flags, app
from shutil import rmtree
from os import mkdir, listdir
from os.path import join, exists, splitext
import pickle
import tensorflow as tf

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_dir', default = None, help = 'path to dataset')
  flags.DEFINE_string('output_dir', default = 'dataset', help = 'path to output directory')
  flags.DEFINE_enum('type', enum_values = {'pulse', 'eis', 'transformer'}, default = 'pulse', help = 'dataset type')

def pulse_eis():
  writer = tf.io.TFRecordWriter(join(FLAGS.output_dir, 'trainset.tfrecord'))
  for f in (listdir(join(FLAGS.input_dir, 'train_datasets')) + listdir(join(FLAGS.input_dir, 'test_datasets'))):
    stem, ext = splitext(f)
    if ext != '.pkl': continue
    if FLAGS.type == 'pulse' and not stem.startswith('train_pulse') and not stem.startswith('test_pulse'): continue
    if FLAGS.type == 'eis' and not stem.startswith('train_eis'): continue
    with open(join(FLAGS.input_dir, 'train_datasets', f), 'rb') as f:
      data = pickle.load(f)
    for SOC, sample in data.items():
      if FLAGS.type == 'pulse':
        sample = tf.constant(np.stack([sample['Voltage'], sample['Current']], axis = -1))
      else:
        sample = tf.constant(np.stack([sample['Real'], sample['Imaginary']], axis = -1))
      trainsample = tf.train.Example(features = tf.train.Features(
        feature = {
          'x': tf.train.Feature(bytes_list = tf.train.BytesList(value = [tf.io.serialize_tensor(sample).numpy()]))
        }))
      writer.write(trainsample.SerializeToString())
  writer.close()

def transformer():
  writer = tf.io.TFRecordWriter(join(FLAGS.output_dir, 'trainset.tfrecord'))


def main(unused_argv):
  if exists(FLAGS.output_dir): rmtree(FLAGS.output_dir)
  mkdir(FLAGS.output_dir)
  if FLAGS.type in ['pulse', 'eis']:
    pulse_eis()
  else:

