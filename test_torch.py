#!/usr/bin/python3

from absl import flags, app
from os import mkdir, listdir
from os.path import join, exists, splitext
import pickle
from tqdm import tqdm
import numpy as np
import torch
from torch import device, load
from torch.utils.data import DataLoader
from models_torch import Trainer
import matplotlib.pyplot as plt
from create_dataset_torch import EISDataset

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('dataset', default = None, help = 'path to dataset')
  flags.DEFINE_integer('batch_size', default = 32, help = 'batch size')
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to checkpoint')

def main(unused_argv):
  trainer = Trainer()
  trainer.eval()
  ckpt = load(join(FLAGS.ckpt, 'model.pth'))
  trainer.load_state_dict(ckpt['state_dict'])
  trainer.to(device('cuda'))
  evalset = EISDataset(join(FLAGS.dataset, 'val'))
  dataset = DataLoader(evalset, batch_size = FLAGS.batch_size, shuffle = True, num_workers = FLAGS.batch_size)
  global_index = 0
  max_dist = np.array(0., dtype = np.float32)
  for pulse, label in dataset:
    pulse, label = pulse.to(device('cuda')), label.numpy()
    eis = trainer(pulse)
    eis = eis.detach().cpu().numpy()
    for p, l in zip(eis, label):
      # p.shape = (35,2) l.shape = (35,2)
      plt.cla()
      plt.plot(p[:,0].numpy(),p[:,1],label = 'prediction')
      plt.plot(l[:,0].numpy(),l[:,1],label = 'ground truth')
      plt.legend()
      plt.savefig('%d.png' % global_index)
      global_index += 1
    dist = np.sqrt(np.sum((eis - label) ** 2, axis = -1)) # diff.shape = (batch, 35)
    m_dist = np.max(dist, axis = (0,1)) # m_dist.shape = ()
    max_dist = np.maximum(max_dist, m_dist)
  print("max distance: ", max_dist)

if __name__ == "__main__":
  add_options()
  app.run(main)
