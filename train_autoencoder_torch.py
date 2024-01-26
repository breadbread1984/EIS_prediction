#!/usr/bin/python3

from absl import flags, app
from os import mkdir
from os.path import exists, join
from creat_dataset_torch import AutoEncoderDataset
from models_torch import AETrainer
from torch import device
from torch.nn import L1Loss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_enum('type', enum_values = {'pulse', 'eis'}, default = 'pulse', help = 'which type of encoder decoder is trained')
  flags.DEFINE_float('lr', default = 1e-2, help = 'learning rate')
  flags.DEFINE_integer('batch_size', default = 8, help = 'batch size')
  flags.DEFINE_integer('epoch', default = 600, help = 'epoch')
  flags.DEFINE_string('dataset', default = None, help = 'path to dataset')
  flags.DEFINE_integer('workers', default = 4, help = 'number of workers')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cuda', 'cpu'}, help = 'device')

def main(unused_argv):
  trainset = AutoEncoder(FLAGS.dataset, FLAGS.type)
  print('trainset size: %d' % len(trainset))
  trainset = DataLoader(trainset, batch_size = FLAGS.batch_size, shuffle = True, num_workers = FLAGS.workers)
  model = AETrainer(99 if FLAGS.type == 'pulse' else 51)
  model.to(device(FLAGS.device))
  mae = L1Loss()
  optimizer = Adam(model.parameters(), lr = FLAGS.lr)
  tb_writer = SummaryWriter(log_dir = join(FLAGS.ckpt, 'summaries'))
  global_steps = 0
  if not exists(FLAGS.ckpt): mkdir(FLAGS.ckpt)
  if exists(join(FLAGS.ckpt, 'model.pth')):
    ckpt = load(join(FLAGS.ckpt, 'model.pth'))
    model.load_state_dict(ckpt['state_dict'])
    global_steps = ckpt['global_steps']
  for epoch in range(FLAGS.epoch):
    model.train()
    for batch, x in enumerate(trainset):
      optimizer.zero_grad()
      x = x.to(device(FLAGS.device))
      pred = model(x)
      loss = mae(pred, x)
      loss.backward()
      optimizer.step()
      global_steps += 1
      print('Step #%d loss %f' % (global_steps, loss.cpu().detach().numpy().item()))

if __name__ == "__main__":
  add_options()
  app.run(main)

