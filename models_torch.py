#!/usr/bin/python3

import torch
from torch import nn

class Trainer(nn.Module):
  def __init__(self, **kwargs):
    super(Trainer, self).__init__()
    self.channels = kwargs.get('channels', 256)
    self.rate = kwargs.get('rate', 0.2)
    self.layer_num = kwargs.get('layer_num', 4)

    self.layernorm = nn.LayerNorm([self.channels,])
    self.dense = nn.Linear(self.channels, 35 * 2)
    layers = dict()
    for i in range(self.layer_num):
      layers['layernorm_%d' % i] = nn.LayerNorm([1800*2 if i == 0 else self.channels])
      layers['dense_%d' % i] = nn.Linear(1800*2 if i == 0 else self.channels, self.channels)
      layers['gelu_%d' % i] = nn.GELU()
      layers['dropout_%d' % i] = nn.Dropout(self.rate)
    self.layers = nn.ModuleDict(layers)
  def forward(self, pulse):
    # pulse.shape = (batch, seq_len, 2)
    results = torch.flatten(pulse, 1) # results.shape = (batch, seq_len * 2)
    for i in range(self.layer_num):
      results = self.layers['layernorm_%d' % i](results)
      results = self.layers['dense_%d' % i](results)
      results = self.layers['gelu_%d' % i](results)
      results = self.layers['dropout_%d' % i](results)
    results = self.layernorm(results)
    results = self.dense(results)
    eis = torch.reshape(results, (-1, 35, 2))
    return eis

if __name__ == "__main__":
  trainer = Trainer()
  a = torch.randn(4,1800,2)
  b = trainer(a)
  print(b.shape)
