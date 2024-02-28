#!/usr/bin/python3

from absl import flags, app
from os import mkdir
from os.path import exists, join
from torch import device, save, load, no_grad, any, isnan, autograd
from torch.nn import L1Loss
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
from create_dataset_torch import EISDataset
from models_torch import Trainer

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('dataset', default = None, help = 'path to directory containing train and test set')
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to directory for checkpoints')
  flags.DEFINE_integer('batch_size', default = 32, help = 'batch size')
  flags.DEFINE_integer('save_freq', default = 100, help = 'save frequency')
  flags.DEFINE_integer('epoch', default = 600, help = 'epoch')
  flags.DEFINE_float('lr', default = 1e-3, help = 'learning rate')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = ['cpu', 'cuda'], help = 'device')

def main(unused_argv):
  autograd.set_detect_anomaly(True)
  trainset = EISDataset(join(FLAGS.dataset, 'train'))
  evalset = EISDataset(join(FLAGS.dataset, 'val'))
  print('trainset size: %d, evalset size: %d' % (len(trainset), len(evalset)))
  train_dataloader = DataLoader(trainset, batch_size = FLAGS.batch_size, shuffle = True, num_workers = FLAGS.batch_size)
  eval_dataloader = DataLoader(evalset, batch_size = FLAGS.batch_size, shuffle = True, num_workers = FLAGS.batch_size)
  model = Trainer()
  model.to(device(FLAGS.device))
  mae = L1Loss()
  optimizer = Adam(model.parameters(), lr = FLAGS.lr)
  scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = 2, T_mult = 2)
  tb_writer = SummaryWriter(log_dir = join(FLAGS.ckpt, 'summaries'))
  start_epoch = 0
  if not exists(FLAGS.ckpt): mkdir(FLAGS.ckpt)
  if exists(join(FLAGS.ckpt, 'model.pth')):
    ckpt = load(join(FLAGS.ckpt, 'model.pth'))
    model.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler = ckpt['scheduler']
    start_epoch = ckpt['epoch']
  for epoch in range(start_epoch, FLAGS.epoch):
    model.train()
    for step, (x,y) in enumerate(train_dataloader):
      optimizer.zero_grad()
      pulse, eis = x.to(device(FLAGS.device)), y.to(device(FLAGS.device))
      preds = model(pulse)
      if any(isnan(preds)):
        print('there is nan in prediction results!')
        continue
      loss = torch.max((eis - preds)**2) # dists.shape = (batch, )
      #loss = mae(eis, preds)
      if any(isnan(loss)):
        print('there is nan in loss!')
        continue
      loss.backward()
      optimizer.step()
      global_steps = epoch * len(train_dataloader) + step
      if global_steps % 100 == 0:
        print('Step #%d Epoch #%d: loss %f, lr %f' % (global_steps, epoch, loss, scheduler.get_last_lr()[0]))
        tb_writer.add_scalar('loss', loss, global_steps)
      if global_steps % FLAGS.save_freq == 0:
        ckpt = {'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler}
        save(ckpt, join(FLAGS.ckpt, 'model.pth'))
    scheduler.step()
    with no_grad():
      model.eval()
      for x, y in eval_dataloader:
        pulse, eis = x.to(device(FLAGS.device)), y.to(device(FLAGS.device))
        preds = model(pulse)
        print('evaluate: loss %f' % torchmetrics.functional.mean_absolute_error(preds, eis))

if __name__ == "__main__":
  add_options()
  app.run(main)

