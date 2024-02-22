#!/usr/bin/python3

import tensorflow as tf

class Scale(tf.keras.layers.Layer):
  def __init__(self):
    super(Scale, self).__init__()
  def build(self, input_shapes):
    self.scale = self.add_weight(name = 'scale', shape = (1,1,2), trainable = True)
    self.bias = self.add_weight(name = 'bias', shape = (1,1,2), trainable = True)
  def call(self, inputs):
    # NOTE: inputs.shape = (batch, seq_len, 2)
    return inputs * self.scale + self.bias

class Trainer(tf.keras.Model):
  def __init__(self, hidden_dim = 256, layers = 1):
    super(Trainer, self).__init__()
    self.norm = tf.keras.layers.BatchNormalization()
    self.embed = tf.keras.layers.Embedding(1,hidden_dim)
    self.pulse_embed = tf.keras.layers.Dense(hidden_dim)
    self.eis_embed = tf.keras.layers.Dense(hidden_dim)
    self.lstm = tf.keras.layers.RNN([tf.keras.layers.LSTMCell(hidden_dim) for i in range(layers)], return_sequences = True, return_state = True)
    self.eis_mlp = tf.keras.layers.Dense(2)
    self.scale = Scale()
  def call(self, pulse):
    pulse = self.norm(pulse)
    pulse_embed = self.pulse_embed(pulse) # pulse_embed.shape = (batch, seq_len, channel)
    results = self.lstm(pulse_embed)
    state = results[1:]
    eis_embed = self.embed(tf.zeros(shape = (pulse.shape[0],1))) # sos.shape = (batch,1,hidden_dim)
    for i in range(35):
      results = self.lstm(eis_embed, initial_state = state) # pred.shape = (batch, seq_len, hidden_dim)
      hidden = results[0]
      eis_embed = tf.concat([eis_embed, hidden[:,-1:,:]], axis = -2) # eis.shape = (batch, 1+eis_length, hidden_dim)
    pred = self.eis_mlp(eis_embed)
    eis = pred[:,1:,:]
    eis = self.scale(eis)
    return eis

if __name__ == "__main__":
  trainer = Trainer(layers = 2)
  import numpy as np
  pulse = np.random.normal(size = (1, 5, 2)).astype(np.float32)
  eis = np.random.normal(size = (1, 3, 2)).astype(np.float32)
  results = trainer([pulse, eis])
  print(results.shape)
