#!/usr/bin/python3

from absl import flags, app
from os import mkdir, listdir
from os.path import join, exists, splitext
import pickle
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from models import Trainer

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('dataset', default = None, help = 'path to dataset')
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to checkpoint')

def main(unused_argv):
  trainer = Trainer()
  checkpoint = tf.train.Checkpoint(model = trainer)
  checkpoint.restore(tf.train.latest_checkpoint(join(FLAGS.ckpt, 'ckpt')))
  sos = tf.constant(np.load('sos.npy'))

  output = open('submission.csv', 'w')
  output.write('test_data_number,soc(%),EIS_real,EIS_imaginary\n')
  for f in tqdm(listdir(join(FLAGS.dataset, 'test_datasets'))):
    stem, ext = splitext(f)
    if ext != '.pkl': continue
    test_num = int(stem.replace('test_pulse_', ''))
    with open(join(FLAGS.dataset, 'test_datasets', f), 'rb') as f:
      data = pickle.load(f)
    for SOC, pulse_samples in data.items():
      soc = SOC.replace('%SOC','')
      pulse = tf.expand_dims(tf.stack([pulse_samples['Voltage'], pulse_samples['Current']], axis = -1), axis = 0) # pulse.shape = (1, seq, 2)
      eis = tf.tile(sos, (pulse.shape[0], 1, 1))
      for i in range(51):
        pred = trainer([pulse, eis])
        eis = tf.concat([eis, pred[:,-1:,:]], axis = -2)
      eis = eis[:,1:,:][0]
      for e in eis:
        output.write(','.join([str(test_num),soc,str(e[0].numpy().item()),str(e[1].numpy().item())]) + '\n')
  output.close()

if __name__ == "__main__":
  add_options()
  app.run(main)

