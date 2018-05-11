from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

def create_loaders(args):
  if args.dataset == 'cifar10':
    n_classes = 10
    norm_mean, norm_std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    dset = torchvision.datasets.CIFAR10
  elif args.dataset == 'cifar100':
    n_classes = 100
    norm_mean, norm_std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    dset = torchvision.datasets.CIFAR100
    
  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)])
  train_set = dset(root=args.data_dir, train=True, download=True, transform=train_transform)
  train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8)

  test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)])
  test_set = dset(root=args.data_dir, train=False, download=True, transform=test_transform)
  test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

  return train_loader, test_loader, n_classes
