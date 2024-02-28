#!/usr/bin/python3

import torch
from torch import nn

class Trainer(nn.Module):
  def __init__(self, **kwargs):
    super(Trainer, self).__init__()
    self.channels = kwargs.get('channels', 64)
    self.rate = kwargs.get('rate', 0.2)
    self.layer_num = kwargs.get('layer_num', 2)

    self.dense1 = nn.Linear(2, self.channels)
    self.dense2 = nn.Linear(self.channels, 2)
    self.rnns = nn.GRU(self.channels, self.channels, self.layer_num)
    self.embed = nn.Embedding(1, self.channels)
  def forward(self, pulse):
    # pulse.shape = (batch, seq_len, 2)
    batch = pulse.shape[0]
    pulse = self.dense1(pulse) # pulse.shape = (batch, seq_len, 2)
    pulse = torch.permute(pulse, (1,0,2)) # pulse.shape = (seq_len, batch, 2)
    _, state = self.rnns(pulse) # state.shape = (layer_num, batch, channels)
    sos = torch.zeros((1, batch)).to(torch.int32) # sos.shape = (1, batch)
    eis_embed = self.embed(sos) # eis_embed.shape = (1, batch, channels)
    latest_eis_embed = eis_embed
    for i in range(35):
      latest_eis_embed, state = self.rnns(latest_eis_embed, state)
      eis_embed = torch.cat([eis_embed, latest_eis_embed], dim = 0) # eis_embed.shape = (seq_len + 1, batch, channels)
    eis_embed = torch.permute(eis_embed, (1,0,2)) # eis_embed.shape = (batch, seq_len + 1, channels)
    pred = self.dense2(eis_embed) # pred.shape = (batch, seq_len + 1, 2)
    eis = pred[:,1:,:]
    return eis

if __name__ == "__main__":
  trainer = Trainer()
  a = torch.randn(4,1800,2)
  b = trainer(a)
  print(b.shape)
