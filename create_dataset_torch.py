#!/usr/bin/python3

from os import listdir
from os.path import join, splitext, isdir
import pickle
import numpy as np

class AutoEncoderDataset(Dataset):
  def __init__(self, dataset_dir, type = 'pulse'):
    assert type in {'pulse', 'eis'}
    self.samples = list()
    for split in listdir(dataset_dir):
      if not isdir(join(dataset_dir, split)): continue
      for f in listdir(join(split, 'train_datasets')):
        stem, ext = splitext(f)
        if ext != '.pkl': continue
        if stem.find(type) < 0: continue
        with open(join(dataset_dir, split, f), 'rb') as f:
          data = pickle.load(f)
        for soc, samples in data.items():
          if type == 'pulse':
            sample = np.stack([samples['Voltage'], samples['Current']], axis = -1)
          else:
            sample = np.stack([samples['Real'], samples['Imaginary']], axis = -1)
          self.samples.append(sample)
  def __len__(self):
    return len(self.samples)
  def __getitem__(self, idx):
    return self.samples[idx]

class TransformerDataset(Dataset):
  def __init__(self, dataset_dir, type = 'train'):
    assert type in {'train', 'val'}
    self.samples = list()
    split = 'train_datasets' if type == 'train' else 'test_datasets'
    for f in listdir(join(dataset_dir, split)):
      stem, ext = splitext(f)
      if ext != '.pkl': continue
      if stem.find('pulse') < 0: continue
      pulse_path = join(dataset_dir, split, f)
      eis_path = join(dataset_dir, split, f.replace('pulse', 'eis'))
      with open(pulse_path, 'rb') as f:
        pulse_samples = pickle.load(f)
      with open(eis_path, 'rb') as f:
        eis_samples = pickle.load(f)
      for soc, pulse_sample in pulse_samples.items():
        eis_sample = eis_samples[soc]
        pulse = np.stack([pulse_sample['Voltage'], pulse_sample['Current']], axis = -1)
        eis = np.stack([eis_sample['Real'], eis_sample['Imaginary']], axis = -1)
        self.samples.append((pulse, eis))
  def __len__(self):
    return len(self.samples)
  def __getitem__(self, idx):
    return self.samples[idx]

