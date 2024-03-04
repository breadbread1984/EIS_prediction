#!/usr/bin/python3

import tensorflow as tf

def Trainer(hidden_dim = 256, drop_rate = 0.1, layer_num = 1):
  inputs = tf.keras.Input((1800, 2)) # pulse.shape = (batch, seq_len, 2)
  results = tf.keras.layers.Flatten()(inputs) # results.shape = (batch, seq_len * 2)
  for i in range(layer_num):
    results = tf.keras.layers.LayerNormalization()(results)
    results = tf.keras.layers.Dense(hidden_dim, activation = tf.keras.activations.gelu)(results)
    results = tf.keras.layers.Dropout(drop_rate)(results)
  results = tf.keras.layers.LayerNormalization()(results)
  results = tf.keras.layers.Dense(35 * 2)(results)
  results = tf.keras.layers.Reshape((35,2))(results)
  return tf.keras.Model(inputs = inputs, outputs = results)

if __name__ == "__main__":
  trainer = Trainer(layers = 2)
  import numpy as np
  pulse = np.random.normal(size = (1, 5, 2)).astype(np.float32)
  results = trainer(pulse)
  trainer.save('trainer.h5')
  print(results.shape)
