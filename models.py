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

def Trainer(hidden_dim = 256, layers = 3):
  pulse = tf.keras.Input((None, 2)) # pulse.shape = (batch, seq_len, 2)
  pulse = tf.keras.layers.BatchNormalization()(pulse)
  pulse_embed = tf.keras.layers.Dense(hidden_dim)(pulse) # pulse_embed.shape = (batch, seq_len, channels)
  lstm = tf.keras.layers.RNN([tf.keras.layers.LSTMCell(hidden_dim) for i in range(layers)], return_sequences = True, return_state = True)
  state = lstm(pulse_embed)[1:]
  sos = tf.keras.layers.Lambda(lambda x: tf.zeros(shape = (tf.shape(x)[0],1)))(pulse) # sos.shape = (batch,1)
  eis_embed = tf.keras.layers.Embedding(1, hidden_dim)(sos) # eis_embed.shape = (batch,1,channels)
  latest_eis_embed = eis_embed
  for i in range(35):
    outputs = lstm(latest_eis_embed, initial_state = state) # results.shape = (batch, query_len, channels)
    latest_eis_embed, state = outputs[0], outputs[1:]
    eis_embed = tf.keras.layers.Lambda(lambda x: tf.concat([x[0],x[1]], axis = -2))([eis_embed, latest_eis_embed]) # eis_embed.shape = (batch, query_len + 1, channels)
  pred = tf.keras.layers.Dense(2)(eis_embed) # pred.shape = (batch, quey_len, 2)
  eis = tf.keras.layers.Lambda(lambda x: x[:,1:,:])(pred)
  eis = Scale()(eis)
  return tf.keras.Model(inputs = pulse, outputs = eis)

if __name__ == "__main__":
  trainer = Trainer(layers = 2)
  import numpy as np
  pulse = np.random.normal(size = (1, 5, 2)).astype(np.float32)
  results = trainer(pulse)
  trainer.save('trainer.h5')
  print(results.shape)
