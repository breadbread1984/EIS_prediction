#!/usr/bin/python3

from os import listdir, mkdir
from os.path import exists, join,splitext
from absl import app, flags
import tensorflow as tf
from models import Trainer

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_float('lr', default = 1e-3, help = 'learning rate')
  flags.DEFINE_integer('batch_size', default = 8, help = 'batch size')
  flags.DEFINE_integer('epoch', default = 200, help = 'epoch')
  flags.DEFINE_string('dataset', default = None, help = 'path to dataset')
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to checkpoint')
  flags.DEFINE_integer('save_freq', default = 10, help = 'save frequency')

def parse_function(serialized_example):
  feature = tf.io.parse_single_example(
    serialized_example,
    features = {
      'x': tf.io.FixedLenFeature((), dtype = tf.string),
      'y': tf.io.FixedLenFeature((), dtype = tf.string)
    })
  x = tf.io.parse_tensor(feature['x'], out_type = tf.float64)
  x = tf.reshape(x, (99, 2))
  y = tf.io.parse_tensor(feature['y'], out_type = tf.float64)
  y = tf.reshape(y, (51, 2))
  return tf.cast(x, dtype = tf.float32), tf.cast(y, dtype = tf.float32)

def main(unused_argv):
  trainer = Trainer()
  sos = tf.Variable(tf.zeros((FLAGS.batch_size, 1, 2)))
  optimizer = tf.keras.optimizers.Adam(FLAGS.lr)
  optimizer.build(trainer.trainable_variables + [sos,])

  trainset = tf.data.TFRecordDataset(join(FLAGS.dataset, 'trainset.tfrecord')).map(parse_function).prefetch(FLAGS.batch_size).shuffle(FLAGS.batch_size).batch(FLAGS.batch_size)

  if not exists(FLAGS.ckpt): mkdir(FLAGS.ckpt)
  checkpoint = tf.train.Checkpoint(model = trainer)
  checkpoint.restore(tf.train.latest_checkpoint(join(FLAGS.ckpt, 'ckpt')))

  log = tf.summary.create_file_writer(FLAGS.ckpt)

  for epoch in range(FLAGS.epoch):
    train_metric = tf.keras.metrics.Mean(name = 'loss')
    train_iter = iter(trainset)
    for sample, label in train_iter:
      pulse = sample
      eis = sos
      with tf.GradientTape() as tape:
        for i in range(51):
          pred = trainer([pulse, eis])
          eis = tf.concat([eis, pred[:,-1:,:]], axis = -2) # pulse.shape = (batch, seq + 1, 2)
        eis = eis[:,1:,:] # pred.shape = (batch, 51, 2)
        loss = tf.reduce_mean(tf.abs(eis - label))
      train_metric.update_state(loss)
      grads = tape.gradient(loss, trainer.trainable_variables + [sos,])
      optimizer.apply_gradients(zip(grads, trainer.trainable_variables + [sos,]))
      print('Step #%d epoch %d: loss %f' % (optimizer.iterations, epoch, train_metric.result()))
      if optimizer.iterations % FLAGS.save_freq == 0:
        with log.as_default():
          tf.summary.scalar('loss', train_metric.result(), step = optimizer.iterations)
  checkpoint.save(join(FLAGS.ckpt, 'ckpt'))
  np.save('sos.npy', sos.numpy())

if __name__ == "__main__":
  add_options()
  app.run(main)
