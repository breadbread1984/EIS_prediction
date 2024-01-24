#!/usr/bin/python3

import tensorflow as tf

def Encoder(dict_size = 1024, drop_rate = 0.1):
  inputs = tf.keras.Input((None, 2)) # inputs.shape = (batch, seq, 2)
  results = tf.keras.layers.Dense(64, activation = tf.keras.activations.gelu)(inputs) # inputs.shape = (batch, seq, 64)
  results = tf.keras.layers.Dropout(drop_rate)(results)
  results = tf.keras.layers.Dense(128, activation = tf.keras.activations.gelu)(results) # results.shape = (batch, seq, 128)
  results = tf.keras.layers.Dropout(drop_rate)(results)
  results = tf.keras.layers.Dense(dict_size, activation = tf.keras.activations.softmax)(results) # results.shape = (batch, seq, 1000)
  results = tf.keras.layers.Lambda(lambda x: tf.math.argmax(x, axis = -1))(results) # results.shape = (batch, seq)
  return tf.keras.Model(inputs = inputs, outputs = results)

def Decoder(dict_size = 1024, drop_rate = 0.1):
  inputs = tf.keras.Input((None,), dtype = tf.int32) # inputs.shape = (batch, seq)
  results = tf.keras.layers.Lambda(lambda x, s: tf.one_hot(x, s), arguments = {'s': dict_size})(inputs) # results.shape = (batch, seq, 1000)
  results = tf.keras.layers.Dense(128, activation = tf.keras.activations.gelu)(results) # results.shape = (batch, seq, 128)
  results = tf.keras.layers.Dropout(drop_rate)(results)
  results = tf.keras.layers.Dense(64, activation = tf.keras.activations.gelu)(results) # results.shape = (batch, seq, 64)
  results = tf.keras.layers.Dropout(drop_rate)(results)
  results = tf.keras.layers.Dense(2)(results)
  return tf.keras.Model(inputs = inputs, outputs = results)

def AETrainer(dict_size = 1024, drop_rate = 0.1):
  inputs = tf.keras.Input((None, 2))
  code = Encoder(dict_size, drop_rate)(inputs)
  results = Decoder(dict_size, drop_rate)(code)
  return tf.keras.Model(inputs = inputs, outputs = results)

def SelfAttention(hidden_dim = 256, num_heads = 8, use_bias = False, drop_rate = 0.1, is_causal = True):
  inputs = tf.keras.Input((None, hidden_dim)) # inputs.shape = (batch, seq, 256)
  results = tf.keras.layers.Dense(hidden_dim * 3, use_bias = use_bias)(inputs) # results.shape = (batch, seq, 3 * 256)
  results = tf.keras.layers.Reshape((-1, 3, num_heads, hidden_dim // num_heads))(results) # results.shape = (batch, seq, 3, 8, 32)
  results = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0,2,3,1,4)))(results) # results.shape = (batch, 3, 8, seq, 32)
  q, k, v = tf.keras.layers.Lambda(lambda x: (x[:,0,...], x[:,1,...], x[:,2,...]))(results) # shape = (batch, 8, seq, 32)
  qk = tf.keras.layers.Lambda(lambda x, s: tf.matmul(x[0], tf.transpose(x[1], (0,1,3,2))) * s, arguments = {'s': (hidden_dim // num_heads) ** -0.5})([q, k]) # qk.shape = (batch, 8, seq, seq)
  if is_causal:
    mask = tf.keras.layers.Lambda(lambda x: tf.expand_dims(
      tf.expand_dims(
        tf.where(
          tf.cast(tf.linalg.band_part(tf.ones((tf.shape(x)[2],tf.shape(x)[2])), -1, 0), dtype = tf.bool),
          tf.constant(0., dtype = tf.float32), tf.experimental.numpy.finfo(tf.float32).min),
        axis = 0),
      axis = 0))(k) # mask.shape = (1,1,seq,seq)
    qk = tf.keras.layers.Add()([qk, mask])
  attn = tf.keras.layers.Softmax(axis = -1)(qk)
  attn = tf.keras.layers.Dropout(rate = drop_rate)(attn)
  qkv = tf.keras.layers.Lambda(lambda x: tf.transpose(tf.matmul(x[0], x[1]), (0,2,1,3)))([attn, v]) # qkv.shape = (batch, seq, 8, 32)
  qkv = tf.keras.layers.Reshape((-1, hidden_dim))(qkv) # qkv.shape = (batch, seq, 256)
  results = tf.keras.layers.Dense(hidden_dim, use_bias = use_bias)(qkv)
  results = tf.keras.layers.Dropout(drop_rate)(results)
  return tf.keras.Model(inputs = inputs, outputs = results)

def TransformerEncoder(dict_size = 1024, hidden_dim = 256, num_heads = 8, use_bias = False, layers = 2, drop_rate = 0.1):
  inputs = tf.keras.Input((None,), dtype = tf.int32) # inputs.shape = (batch, seq)
  results = tf.keras.layers.Embedding(dict_size, hidden_dim)(inputs) # results.shape = (batch, seq, 256)
  results = tf.keras.layers.Dropout(drop_rate)(results)
  for i in range(layers):
    skip = results
    results = tf.keras.layers.Normalization()(results)
    results = SelfAttention(hidden_dim, num_heads, use_bias, drop_rate, is_causal = False)(results)
    results = tf.keras.layers.Add()([skip, results])
    skip = results
    results = tf.keras.layers.Normalization()(results)
    results = tf.keras.layers.Dense(4 * hidden_dim, activation = tf.keras.activations.gelu)(results)
    results = tf.keras.layers.Dense(hidden_dim)(results)
    results = tf.keras.layers.Dropout(drop_rate)(results)
    results = tf.keras.layers.Add()([skip, results])
  return tf.keras.Model(inputs = inputs, outputs = results)

def CrossAttention(hidden_dim = 256, num_heads = 8, use_bias = False, drop_rate = 0.1):
  code = tf.keras.Input((None, hidden_dim)) # code.shape = (batch, seq, 256)
  inputs = tf.keras.Input((None, hidden_dim)) # inputs.shape = (batch, seq, 256)
  code_results = tf.keras.layers.Dense(hidden_dim * 2, use_bias = use_bias)(code) # code_results.shape = (batch, seq, 2 * 256)
  code_results = tf.keras.layers.Reshape((-1, 2, num_heads, hidden_dim // num_heads))(code_results) # code_results.shape = (batch, seq, 2, 8, 32)
  code_results = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0,2,3,1,4)))(code_results) # code_results.shape = (batch, 2, 8, seq, 32)
  k, v = tf.keras.layers.Lambda(lambda x: (x[:,0,...], x[:,1,...]))(code_results) # shape = (batch, 8, seq, 32)
  results = tf.keras.layers.Dense(hidden_dim, use_bias = use_bias)(inputs) # results.shape = (batch, seq, 256)
  results = tf.keras.layers.Reshape((-1, num_heads, hidden_dim // num_heads))(results) # results.shape = (batch, seq, 8, 32)
  q = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0,2,1,3)))(results) # results.shape = (batch, 8, seq, 32)
  qk = tf.keras.layers.Lambda(lambda x, s: tf.matmul(x[0], tf.transpose(x[1], (0,1,3,2))) * s, arguments = {'s': (hidden_dim // num_heads) ** -0.5})([q, k]) # qk.shape = (batch, 8, seq, seq)
  attn = tf.keras.layers.Softmax(axis = -1)(qk)
  attn = tf.keras.layers.Dropout(rate = drop_rate)(attn)
  qkv = tf.keras.layers.Lambda(lambda x: tf.transpose(tf.matmul(x[0], x[1]), (0,2,1,3)))([attn, v]) # qkv.shape = (batch, seq, 8, 32)
  qkv = tf.keras.layers.Reshape((-1, hidden_dim))(qkv) # qkv.shape = (batch, seq, 256)
  results = tf.keras.layers.Dense(hidden_dim, use_bias = use_bias)(qkv)
  results = tf.keras.layers.Dropout(drop_rate)(results)
  return tf.keras.Model(inputs = (code, inputs), outputs = results) 

def TransformerDecoder(dict_size = 1024, hidden_dim = 256, num_heads = 8, use_bias = False, layers = 2, drop_rate = 0.1):
  code = tf.keras.Input((None, hidden_dim)) # code.shape = (batch, seq, 256)
  inputs = tf.keras.Input((None,), dtype = tf.int32) # inputs.shape = (batch, seq)
  results = tf.keras.layers.Embedding(dict_size, hidden_dim)(inputs) # results.shape = (batch, seq, hidden_dim)
  results = tf.keras.layers.Dropout(drop_rate)(results)
  for i in range(layers):
    skip = results
    results = tf.keras.layers.LayerNormalization()(results)
    results = SelfAttention(hidden_dim, num_heads, use_bias, drop_rate, is_causal = True)(results)
    results = tf.keras.layers.Add()([skip, results])
    skip = results
    results = tf.keras.layers.LayerNormalization()(results)
    results = CrossAttention(hidden_dim, num_heads, use_bias, drop_rate)([code, results])
    results = tf.keras.layers.Add()([skip, results])
    skip = results
    results = tf.keras.layers.Normalization()(results)
    results = tf.keras.layers.Dense(4 * hidden_dim, activation = tf.keras.activations.gelu)(results)
    results = tf.keras.layers.Dense(hidden_dim)(results)
    results = tf.keras.layers.Dropout(drop_rate)(results)
    results = tf.keras.layers.Add()([skip, results])
  return tf.keras.Model(inputs = (code, inputs), outputs = results)

def Trainer(dict_size = 1024, hidden_dim = 256, num_heads = 8, use_bias = False, layers = 1, drop_rate = 0.1):
  pulse = tf.keras.Input((None,2))
  eis = tf.keras.Input((None,2))

  pulse_encoder = Encoder(dict_size, drop_rate)
  pulse_encoder.load_weights('pulse_encoder.keras')
  pulse_encoder.trainable = False
  eis_encoder = Encoder(dict_size, drop_rate)
  eis_encoder.load_weights('eis_encoder.keras')
  eis_encoder.trainable = False
  eis_decoder = Decoder(dict_size, drop_rate)
  eis_decoder.load_weights('eis_decoder.keras')
  eis_decoder.trainable = False

  pulse_tokens = pulse_encoder(pulse) # pulse_tokens.shape = (batch, pulse_seq)
  code = TransformerEncoder(dict_size, hidden_dim, num_heads, use_bias, layers, drop_rate)(pulse_tokens) # code.shape = (batch, pulse_seq, 256)
  eis_tokens = eis_encoder(eis) # eis_tokens.shape = (batch, eis_seq)
  results = TransformerDecoder(dict_size, hidden_dim, num_heads, use_bias, layers, drop_rate)([code, eis_tokens]) # results.shape = (batch, eis_seq, 256)
  results = tf.keras.layers.Dense(dict_size, activation = tf.keras.activations.softmax)(results) # results.shape = (batch, eis_seq, dict_size)
  results = tf.keras.layers.Lambda(lambda x: tf.math.argmax(x, axis = -1))(results) # results.shape = (batch, eis_seq)
  eis_update = eis_decoder(results) # eis_tokens.shape = (batch, eis_seq, 2)
  return tf.keras.Model(inputs = (pulse, eis), outputs = (eis_update))

if __name__ == "__main__":
  inputs = tf.random.normal(shape = (1, 10, 256))
  results = SelfAttention()(inputs)
  print(results.shape)
  inputs = tf.random.uniform(minval = 0, maxval = 1024, shape = (2, 10), dtype = tf.int32)
  results = TransformerEncoder()(inputs)
  print(results.shape)
  trainer = Trainer()
  pulse = tf.random.normal(shape = (1, 10, 2))
  eis = tf.random.normal(shape = (1, 5, 2))
  eis_update = trainer([pulse, eis])
  print(eis_update.shape)
