#!/usr/bin/python3

import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
  def __init__(self, seq_len, dict_size = 1024, drop_rate = 0.1):
    super(Encoder, self).__init__()
    self.layernorm1 = nn.LayerNorm([seq_len, 2])
    self.layernorm2 = nn.LayerNorm([seq_len, 64])
    self.layernorm3 = nn.LayerNorm([seq_len, 128])
    self.linear1 = nn.Linear(2, 64)
    self.linear2 = nn.Linear(64, 128)
    self.linear3 = nn.Linear(128, dict_size)
    self.dropout1 = nn.Dropout(drop_rate)
    self.dropout2 = nn.Dropout(drop_rate)
    self.gelu1 = nn.GELU()
    self.gelu2 = nn.GELU()
  def forward(self, inputs):
    # inputs.shape = (batch, 2, seq)
    results = torch.transpose(inputs, 1, 2) # results.shape = (batch, seq, 2)
    results = self.layernorm1(inputs)
    results = self.linear1(results) # results.shape = (batch, seq, 64)
    results = self.gelu1(results)
    results = self.dropout1(results)
    results = self.layernorm2(results)
    results = self.linear2(results) # results.shape = (batch, seq, 128)
    results = self.gelu2(results)
    results = self.dropout2(results)
    results = self.layernorm3(results)
    results = self.linear3(results) # results.shape = (batch, seq, dict_size)
    results = F.softmax(results, dim = -1) # results.shape = (batch, seq, dict_size)
    results = torch.argmax(results, dim = -1) # results.shape = (batch, seq)
    return results

class Decoder(nn.Module):
  def __init__(self, seq_len, dict_size = 1024, drop_rate = 0.1):
    super(Decoder, self).__init__()
    self.dict_size = dict_size

    self.layernorm1 = nn.LayerNorm([seq_len, dict_size])
    self.layernorm2 = nn.LayerNorm([seq_len, 128])
    self.layernorm3 = nn.LayerNorm([seq_len, 64])
    self.linear1 = nn.Linear(dict_size, 128)
    self.linear2 = nn.Linear(128, 64)
    self.linear3 = nn.Linear(64, 2)
    self.dropout1 = nn.Dropout(drop_rate)
    self.dropout2 = nn.Dropout(drop_rate)
    self.gelu1 = nn.GELU()
    self.gelu2 = nn.GELU()
  def forward(self, inputs):
    # inputs.shape = (batch, seq_len)
    results = F.one_hot(inputs, self.dict_size) # results.shape = (batch, seq_len, dict_size)
    results = self.layernorm1(results)
    results = self.linear1(results) # results.shape = (batch, seq_len, 128)
    results = self.gelu1(results)
    results = self.dropout1(results)
    results = self.layernorm2(results)
    results = self.linear2(results) # results.shape = (batch, seq_len, 64)
    results = self.gelu2(results)
    results = self.dropout2(results)
    results = self.layernorm3(results)
    results = self.linear3(results) # results.shape = (batch, seq_len, 2)
    results = torch.transpose(results, 1, 2) # results.shape = (batch, 2, seq_len)
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

class SelfAttention(nn.Module):
  def __init__(self, hiddem_dim = 256, num_heads = 8, use_bias = False, drop_rate = 0.1, is_causal = True):
    super(SelfAttention, self).__init__()
    self.num_heads = num_heads
    self.hidden_dim = hidden_dim
    self.is_causal = is_causal

    self.linear1 = nn.Linear(hidden_dim, 3 * hidden_dim, bias = use_bias)
    self.linear2 = nn.Linear(hidden_dim, hidden_dim, bias = use_bias)
    self.dropout = nn.Dropout(drop_rate)
  def forward(self, inputs):
    # inputs.shape = (batch, hidden_dim, seq_len)
    results = torch.transpose(inputs, 1, 2) # results.shape = (batch, seq_len, hidden_dim)
    results = self.linear1(inputs) # results.shape = (batch, seq_len, 3 * hidden_dim)
    b, s, _ = results.shape
    results = torch.reshape(results, (b, s, 3, self.num_heads, self.hidden_dim // self.num_heads)) # results.shape = (batch, seq_len, 3, head_num, hidden_dim // head_num)
    results = torch.permute(results, (0,2,3,1,4)) # results.shape = (batch, 3, head_num, seq_len, hidden_dim // head_num)
    q,k,v = results[:,0,...], results[:,1,...], results[:,2,...] # shape = (batch, head_num, seq_len, hidden_dim // head_num)
    qk = torch.matmul(q, torch.transpose(k, 2,3)) * (self.hidden_dim // self.num_heads) ** -0.5 # qk.shape = (batch, head_num, seq_len, seq_len)
    if self.is_causal:
      mask = torch.unsqueeze(torch.unsqueeze(torch.tril(torch.ones(s,s)), 0), 0) # mask.shape = (1,1,seq_len, seq_len)
      mask = torch.where(mask, 0., torch.finfo(torch.float32).min)
      qk = qk + mask
    attn = F.softmax(qk, dim = -1)
    attn = self.dropout(attn) # attn.shape = (batch, head_num, seq_len, seq_len)
    qkv = torch.transpose(torch.matmul(attn, v), 1, 2) # qkv.shape = (batch, seq_len, head_num, hidden_dim // head_num)
    qkv = torch.reshape(qkv, (b, s, -1)) # qkv.shape = (batch, seq_len, hidden_dim)
    results = self.linear2(qkv) # results.shape = (batch, seq_len, hidden_dim)
    results = self.dropout(results) # results.shape = (batch, seq_len, hidden_dim)
    results = torch.transpose(results, 1, 2) # results.shape = (batch, hidden_dim, seq_len)
    return results

class 
