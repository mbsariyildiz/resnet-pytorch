from os.path import join
import argparse
import numpy as np
from tqdm import trange
from scipy.io import savemat

import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import classifier
import utils
import nn_ops

np.random.seed(67)
torch.manual_seed(67)

def main(hps):
  
  if hps.dataset == 'cifar10':
    n_class = 10

    train_transform = transforms.Compose([
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_set = torchvision.datasets.CIFAR10(root=hps.data_dir, train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=8)

    test_transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_set = torchvision.datasets.CIFAR10(root=hps.data_dir, train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)

  clf = classifier.ConvNet(n_class,
                           hps.init_lr,
                           hps.momentum,
                           hps.weight_decay,
                           hps.device,
                           hps.exp_dir)
  
  test_accs = np.zeros([hps.n_epochs], 'float32')
  train_accs = np.zeros([hps.n_epochs], 'float32')
  train_losses = np.zeros([hps.n_epochs], 'float32')
  learning_rates = np.zeros([hps.n_epochs], 'float32')

  if hps.device == 'cuda':
    cudnn.benchmark = True

  if hps.tensorboard:
    from tensorboard_logger import log_value

  try:
    for epoch in trange(hps.n_epochs):
      # arrange the learning rate
      lr = clf.adjust_learning_rate(epoch, hps.lr_dec_rate, hps.lr_dec_int)

      ########## Classifier update ##########
      train_loss, train_acc = clf.train_epoch(train_loader)

      ########## Evaluation ##########
      test_acc = clf.test(test_loader)

      if hps.tensorboard:
        log_value('test_acc', test_acc, epoch)
        log_value('train_acc', train_acc, epoch)
        log_value('train_loss', train_loss, epoch)
        log_value('learning_rate', lr, epoch)

      test_accs[epoch] = test_acc
      train_accs[epoch] = train_acc
      train_losses[epoch] = train_loss
      learning_rates[epoch] = lr

      torch.save(
        { 'epoch': epoch, 'clf': clf.net.state_dict() },
        join(hps.exp_dir, 'state_dict.ckpt'))

  except KeyboardInterrupt:
    print ('Early exit')
    import ipdb; ipdb.set_trace()

  savemat(
    join(hps.exp_dir, 'logs.mat'),
    {'train_accs':train_accs, 'test_accs':test_accs, 'train_losses':train_losses, 'learning_rates': learning_rates})

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--exp_dir', type=str, default='./log')
  parser.add_argument('--data_dir', type=str, default='./data')
  parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10'])
  parser.add_argument('--init_lr', type=float, default=0.1)
  parser.add_argument('--lr_dec_rate', type=float, default=0.1)
  parser.add_argument('--lr_dec_int', type=int, default=100)
  parser.add_argument('--weight_decay', type=float, default=5e-4)
  parser.add_argument('--momentum', type=float, default=0.9 )
  parser.add_argument('--tensorboard', type=bool, default=True)
  parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
  parser.add_argument('--batch_size', type=int, default=128)
  parser.add_argument('--test_batch_size', type=int, default=200)
  parser.add_argument('--n_epochs', type=int, default=300)
  FLAGS = parser.parse_args()

  utils.write_logs(FLAGS)
  if FLAGS.tensorboard: 
    from tensorboard_logger import configure
    configure(FLAGS.exp_dir)

  main(FLAGS)
  