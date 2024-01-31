#!/usr/bin/python3

import tensorflow as tf

def Trainer(hidden_dim = 256, layers = 1):
  pulse = tf.keras.Input((None, 2))
  eis = tf.keras.Input((None, 2))

  pulse_embed = tf.keras.layers.Dense(hidden_dim)(pulse)
  eis_embed = tf.keras.layers.Dense(hidden_dim)(eis)

  rnn = tf.keras.layers.RNN([tf.keras.layers.LSTMCell(hidden_dim) for i in range(layers)], return_sequences = True, return_state = True)
  results = rnn(pulse_embed)
  state = results[1:]
  results = rnn(eis_embed, initial_state = state)
  hidden = results[0]

  eis_update = tf.keras.layers.Dense(2)(hidden)

  return tf.keras.Model(inputs = (pulse, eis), outputs = eis_update)

if __name__ == "__main__":
  trainer = Trainer(layers = 2)
  import numpy as np
  pulse = np.random.normal(size = (1, 5, 2)).astype(np.float32)
  eis = np.random.normal(size = (1, 3, 2)).astype(np.float32)
  results = trainer([pulse, eis])
  print(results.shape)
