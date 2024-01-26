#!/usr/bin/python3

from absl import flags, app

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_enum('type', enum_values = {'pulse', 'eis'}, default = 'pulse', help = 'which type of encoder decoder is trained')
  flags.DEFINE_float('lr', default = 1e-2, help = 'learning rate')
  flags.DEFINE_integer('batch_size', default = 8, help = 'batch size')
  flags.DEFINE_integer('epoch', default = 600, help = 'epoch')
  flags.DEFINE_string('dataset', default = None, help = 'path to dataset')
  flags.DEFINE_integer('workers', default = 4, help = 'number of workers')

 
