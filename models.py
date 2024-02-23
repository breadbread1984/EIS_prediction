#!/usr/bin/python3

import tensorflow as tf

def Trainer(channels = 256, rate = 0.2, layers = 2):
  inputs = tf.keras.Input((1800, 2)) # pulse.shape = (batch, seq_len, 2)
  results = inputs
  for i in range(layers):
    # spatial mix
    results = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0,2,1)))(results)
    results = tf.keras.layers.LayerNormalization()(results)
    results = tf.keras.layers.Dense(35,activation=tf.keras.activations.gelu)(results)
    results = tf.keras.layers.Dropout(rate)(results)
    results = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0,2,1)))(results)
    # channel mix
    results = tf.keras.layers.LayerNormalization()(results)
    results = tf.keras.layers.Dense(channels, activation=tf.keras.activations.gelu)(results)
    results = tf.keras.layers.Dropout(rate)(results)
    results = tf.keras.layers.LayerNormalization()(results)
    results = tf.keras.layers.Dense(2, activation=tf.keras.activations.gelu)(results)
    results = tf.keras.layers.Dropout(rate)(results)
  results = tf.keras.layers.Dense(2)(results)
  return tf.keras.Model(inputs = inputs, outputs = results)

if __name__ == "__main__":
  trainer = Trainer(layers = 2)
  import numpy as np
  pulse = np.random.normal(size = (1, 5, 2)).astype(np.float32)
  results = trainer(pulse)
  trainer.save('trainer.h5')
  print(results.shape)
