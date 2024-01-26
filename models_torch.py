#!/usr/bin/python3

import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
  def __init__(self, seq_len, dict_size = 1024, drop_rate = 0.1):
    super(Encoder, self).__init__()
    self.layernorm1 = nn.LayerNorm([2, seq_len])
    self.layernorm2 = nn.LayerNorm([64, seq_len])
    self.layernorm3 = nn.LayerNorm([128, seq_len])
    self.linear1 = nn.Linear(2, 64)
    self.linear2 = nn.Linear(64, 128)
    self.linear3 = nn.Linear(128, dict_size)
    self.dropout1 = nn.Dropout(drop_rate)
    self.dropout2 = nn.Dropout(drop_rate)
    self.gelu1 = nn.GELU()
    self.gelu2 = nn.GELU()
    self.softmax = nn.Softmax(dim = dict_size)
  def forward(self, inputs):
    results = self.layernorm1(inputs)
    results = self.linear1(results)
    results = self.gelu1(results)
    results = self.dropout1(results)
    results = self.layernorm2(results)
    results = self.linear2(results)
    results = self.gelu2(results)
    results = self.dropout2(results)
    results = self.layernorm3(results)
    results = self.linear3(results)
    results = self.softmax(results)
    results = torch.argmax(results, dim = 1)
    return results

class Decoder(nn.Module):
  def __init__(self, seq_len, dict_size = 1024, drop_rate = 0.1):
    super(Decoder, self).__init__()
    self.dict_size = dict_size

    self.layernorm1 = nn.LayerNorm([dict_size, seq_len])
    self.layernorm2 = nn.LayerNorm([128, seq_len])
    self.layernorm3 = nn.LayerNorm([64, seq_len])
    self.linear1 = nn.Linear(dict_size, 128)
    self.linear2 = nn.Linear(128, 64)
    self.linear3 = nn.Linear(62, 2)
    self.dropout1 = nn.Dropout(drop_rate)
    self.dropout2 = nn.Dropout(drop_rate)
    self.gelu1 = nn.GELU()
    self.gelu2 = nn.GELU()
  def forward(self, inputs):
    results = F.one_hot(inputs, self.dict_size) # results.shape = (batch, seq_len, dict_size)
    results = self.layernorm1(results)
    results = self.linear1(results)
    results = self.gelu1(results)
    results = self.dropout1(results)
    results = self.layernorm2(results)
    results = self.linear2(results)
    results = self.gelu2(results)
    results = self.dropout2(results)
    results = self.layernorm3(results)
    results = self.linear3(results)
    return results

class AETrainer(nn.Module):
  def __init__(self, seq_len, dict_size = 1024, drop_rate = 0.1):
    super(AETrainer, self).__init__()
    self.encoder = Encoder(seq_len, dict_size, drop_rate)
    self.decoder = Decoder(seq_len, dict_size, drop_rate)
  def forward(self, inputs):
    results = self.encoder(inputs)
    results = self.decoder(results)
    return results


