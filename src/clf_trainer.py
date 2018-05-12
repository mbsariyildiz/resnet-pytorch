from os.path import join
import argparse
import numpy as np
from tqdm import tqdm, trange
from scipy.io import savemat

import torch
import torch.backends.cudnn as cudnn

import classifier
import utils
import config

np.random.seed(67)
torch.manual_seed(67)

def main(args):
  
  train_loader, test_loader, n_classes = config.create_loaders(args)

  clf = classifier.ConvNet(n_classes,
                           args.init_lr,
                           args.momentum,
                           args.weight_decay,
                           args.device,
                           args.exp_dir,
                           '',
                           args.model,
                           args.multi_gpu)
  
  test_accs = np.zeros([args.n_epochs], 'float32')
  train_accs = np.zeros([args.n_epochs], 'float32')
  train_losses = np.zeros([args.n_epochs], 'float32')
  learning_rates = np.zeros([args.n_epochs], 'float32')    

  if args.tensorboard:
    from tensorboard_logger import log_value

  try:
    for epoch in trange(args.n_epochs, ncols=100):
      # arrange the learning rate
      lr = clf.adjust_learning_rate(epoch, args.lr_dec_rate, args.lr_dec_int)

      ########## Classifier update ##########
      train_loss, train_acc = clf.train_epoch(train_loader)

      ########## Evaluation ##########
      test_acc = clf.test(test_loader)

      if args.tensorboard:
        log_value('test_acc', train_acc, epoch)
        log_value('train_acc', train_acc, epoch)
        log_value('train_loss', train_loss, epoch)
        log_value('learning_rate', lr, epoch)
      else:
        tqdm.write('train_acc:{:.1f}, test_acc{:.1f}'.format(
          train_acc, train_acc))

      test_accs[epoch] = test_acc
      train_accs[epoch] = train_acc
      train_losses[epoch] = train_loss
      learning_rates[epoch] = lr

      torch.save(
        { 'epoch': epoch, 'clf': clf.net.state_dict() },
        join(args.exp_dir, 'state_dict.ckpt'))

  except KeyboardInterrupt:
    print ('Early exit')
    import ipdb; ipdb.set_trace()

  savemat(
    join(args.exp_dir, 'logs.mat'),
    {'train_accs':train_accs, 'test_accs':test_accs, 'train_losses':train_losses, 'learning_rates': learning_rates})

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--exp_dir', type=str, default='./log')
  parser.add_argument('--data_dir', type=str, default='./data')
  parser.add_argument('--dataset', type=str, default='cifar10', 
    choices=['cifar10', 'cifar100'])
  parser.add_argument('--model', type=str, default='resnet-10', 
    choices=['resnet-10', 'resnet-18', 'resnet-34', 'resnet-50'])
  parser.add_argument('--device', type=str, default='cuda', 
    choices=['cpu', 'cuda'])
  parser.add_argument('--multi_gpu', type=bool, default=True)
  parser.add_argument('--init_lr', type=float, default=0.1)
  parser.add_argument('--lr_dec_rate', type=float, default=0.1)
  parser.add_argument('--lr_dec_int', type=int, default=100)
  parser.add_argument('--weight_decay', type=float, default=5e-4)
  parser.add_argument('--momentum', type=float, default=0.9 )
  parser.add_argument('--tensorboard', type=bool, default=True)
  parser.add_argument('--batch_size', type=int, default=128)
  parser.add_argument('--test_batch_size', type=int, default=128)
  parser.add_argument('--n_epochs', type=int, default=300)
  FLAGS = parser.parse_args()
  
  if FLAGS.device == 'cuda' and torch.cuda.is_available():
    cudnn.benchmark = True
  else:
    FLAGS.device = 'cpu'

  utils.write_logs(FLAGS)
  if FLAGS.tensorboard: 
    from tensorboard_logger import configure
    configure(FLAGS.exp_dir)

  main(FLAGS)
  