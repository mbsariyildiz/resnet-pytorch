from os.path import join
import argparse
import numpy as np
from tqdm import trange

import torch
import torchvision
import torchvision.transforms as transforms

import classifier
import nn_ops

np.random.seed(67)
torch.manual_seed(67)

N_IMAGES_TO_PLOT = 20

def main(args):
  
  if args.dataset == 'cifar10':
    n_class = 10

    test_transform = transforms.Compose([
      transforms.ToTensor(), 
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_set = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

  clf = classifier.ConvNet(n_class,
                           device=args.device,
                           ckpt_file=args.ckpt_file)
  clf.net.eval()
  for p in clf.net.parameters(): p.requires_grad = False                           

  n_ch, sp_size = test_set[0][0].size()[:2]
  n_features = clf.n_planes[-1]
  n_images = len(test_set)

  print ('Extracting features from the test set ... ')
  test_features = torch.zeros(n_images, n_features, device='cpu')
  for b_ix, (x, y) in enumerate(test_loader):
    x = x.to(args.device)
    ix_from = b_ix * args.batch_size
    ix_to = ix_from + args.batch_size
    test_features[ix_from:ix_to, :] = clf.net.features(x).to('cpu')
  test_features = test_features.numpy()

  print ('Selecting and ploting samples based on their feature magnitudes ... ')
  images_to_save = torch.zeros(n_features * N_IMAGES_TO_PLOT, n_ch, sp_size, sp_size, device='cpu')
  for f_ix in trange(n_features):
    ranks = test_features[:, f_ix].argsort()
    inds = np.hstack([ranks[:N_IMAGES_TO_PLOT//2], ranks[-N_IMAGES_TO_PLOT//2:]])

    for im_ix in trange(N_IMAGES_TO_PLOT):
      images_to_save[f_ix * N_IMAGES_TO_PLOT + im_ix, :, :, :] = test_set[inds[im_ix]][0]

  torchvision.utils.save_image(
    images_to_save,
    join(args.log_dir, '%s_images_selected_by_features.png' % args.dataset), 
    N_IMAGES_TO_PLOT,
    normalize=True)

  torch.save(
    {'images': images_to_save},
    join(args.log_dir, '%s_images_selected_by_features.torch' % args.dataset)
  )

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--log_dir', type=str, default='./log')
  parser.add_argument('--ckpt_file', type=str, default='./log/state_dict.ckpt')
  parser.add_argument('--data_dir', type=str, default='./data')
  parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10'])
  parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
  parser.add_argument('--batch_size', type=int, default=100)
  args = parser.parse_args()

  main(args)
  