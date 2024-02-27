#!/usr/bin/python3

import torch
from torch import nn

class MLPMixer(nn.Module):
  def __init__(self, **kwargs):
    super(MLPMixer, self).__init__()
    self.hidden_dim = kwargs.get('hidden_dim', 768)
    self.num_blocks = kwargs.get('num_blocks', 12)
    self.tokens_mlp_dim = kwargs.get('tokens_mlp_dim', 384)
    self.channels_mlp_dim = kwargs.get('channels_mlp_dim', 3072)
    self.drop_rate = kwargs.get('drop_rate', 0.1)

    self.layernorm1 = nn.LayerNorm((1800, 2))
    self.dense1 = nn.Linear(2, self.hidden_dim)
    self.gelu = nn.GELU()
    self.dropout = nn.Dropout(self.drop_rate)
    layers = dict()
    for i in range(self.num_blocks):
      layers.update({
        'layernorm1_%d' % i: nn.LayerNorm((self.hidden_dim, 1800 if i == 0 else 35)),
        'linear1_%d' % i: nn.Linear(1800 if i == 0 else 35, self.tokens_mlp_dim),
        'gelu1_%d' % i: nn.GELU(),
        'linear2_%d' % i: nn.Linear(self.tokens_mlp_dim, 35),
        'layernorm2_%d' % i: nn.LayerNorm((35, self.hidden_dim)),
        'linear3_%d' % i: nn.Linear(self.hidden_dim, self.channels_mlp_dim),
        'gelu2_%d' % i: nn.GELU(),
        'linear4_%d' % i: nn.Linear(self.channels_mlp_dim, self.hidden_dim),
      })
    self.layers = nn.ModuleDict(layers)
    self.layernorm2 = nn.LayerNorm((35,self.hidden_dim))
    self.dense2 = nn.Linear(self.hidden_dim, 2)
  def forward(self, inputs):
    # inputs.shape = (batch, 1800, 2)
    results = self.layernorm1(inputs)
    results = self.dense1(results)
    results = self.gelu(results)
    results = self.dropout(results)

    for i in range(self.num_blocks):
      # 1) spatial mixing
      if i != 0: skip = results
      results = torch.permute(results, (0,2,1)) # results.shape = (batch, channel, 9**3)
      results = self.layers['layernorm1_%d' % i](results)
      results = self.layers['linear1_%d' % i](results)
      results = self.layers['gelu1_%d' % i](results)
      results = self.layers['linear2_%d' % i](results)
      results = torch.permute(results, (0,2,1)) # resutls.shape = (batch, 9**3, channel)
      if i != 0: results = results + skip
      # 2) channel mixing
      skip = results
      results = self.layers['layernorm2_%d' % i](results)
      results = self.layers['linear3_%d' % i](results)
      results = self.layers['gelu2_%d' % i](results)
      results = self.layers['linear4_%d' % i](results)
      results = results + skip

    results = self.layernorm2(results) # results.shape = (batch, 35, channel)
    results = self.dense2(results) # results.shape = (batch, 35, 2)
    return results

class Predictor(nn.Module):
  def __init__(self,):
    super(Predictor, self).__init__()
    kwargs = {'hidden_dim': 768, 'num_blocks': 2, 'tokens_mlp_dim': 384, 'channels_mlp_dim': 3072, 'drop_rate': 0.1}
    self.predictor = MLPMixer(**kwargs)
  def forward(self, inputs):
    results = self.predictor(inputs)
    return results

if __name__ == "__main__":
  mlpmixer = MLPMixer()
  a = torch.randn(4,1800,2)
  b = mlpmixer(a)
  print(b.shape)
