from os.path import join
import torch
import resnet
import utils
import clf_tools

class BaseClassifier(object):
  def __init__(self, n_class,
                     init_lr=0.1,
                     momentum=0.9,
                     weight_decay=5e-4,
                     device='cpu'):
    self.n_class = n_class
    self.init_lr = init_lr
    self.momentum = momentum
    self.weight_decay = weight_decay
    self.device = device
    self.net = None
    self.criterion = None
    self.optim = None

  def adjust_learning_rate(self, epoch, dec_rate=0.1, dec_int=10):
    """Decreases the initial learning rate by dec_rate every dec_int epochs"""

    lr = self.init_lr * (dec_rate ** (epoch // dec_int))
    # update the learing rate of the optimizer
    for param_group in self.optim.param_groups:
      param_group['lr'] = lr

    return lr

  def train_step(self, x, y):
    self.net.zero_grad()

    logits = self.net(x)
    loss = self.criterion(logits, y)
    loss.backward()
    self.optim.step()

    prec1 = clf_tools.accuracy(logits.data, y.data)[0]
    return prec1.item(), loss.item()

  def train_epoch(self, iterator):
    train_loss = utils.AverageMeter()
    train_acc = utils.AverageMeter()
    self.net.train()

    if isinstance(iterator, torch.utils.data.DataLoader):
      for x, y in iterator:
        x, y = x.to(self.device), y.to(self.device)
        
        prec1, loss = self.train_step(x, y)
        train_acc.update(prec1, x.size(0))
        train_loss.update(loss, x.size(0))

    else:
      raise NotImplementedError('Data feed can only be torch.utils.data.DataLoader.')

    return train_loss.avg, train_acc.avg

  def test(self, iterator):
    top1 = utils.AverageMeter()
    self.net.eval()

    if isinstance(iterator, torch.utils.data.DataLoader):
      for x, y in iterator:
        x, y = x.to(self.device), y.to(self.device)
        
        logits = self.net(x)
        prec1 = clf_tools.accuracy(logits, y)[0]
        top1.update(prec1.item(), x.size(0))

    else:
      raise NotImplementedError('Data feed can only be torch.utils.data.DataLoader.')

    self.net.train()
    return top1.avg

class ConvNet(BaseClassifier):

  def __init__(self, n_class,
                     init_lr=0.1,
                     momentum=0.9,
                     weight_decay=5e-4,
                     device='cpu',
                     log_dir='',
                     ckpt_file=''):
    super().__init__(n_class, init_lr, momentum, weight_decay, device) 
    
    self.n_planes = [64, 128, 256, 512]
    self.net = resnet.ResNet10(n_classes=self.n_class,
                               n_output_planes=self.n_planes).to(self.device)
    self.optim = torch.optim.SGD(self.net.parameters(),
                                 self.init_lr,
                                 momentum=self.momentum,
                                 weight_decay=self.weight_decay)
    self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

    if ckpt_file:
      print ('loading pretrained classifier checkpoint')
      if device == 'cpu':
        ckpt = torch.load(ckpt_file, map_location=lambda storage, loc: storage)
      else:
        ckpt = torch.load(ckpt_file)
      self.net.load_state_dict(ckpt['clf'])

    print('Number of dnn parameters: {}'.format(
      sum([p.data.nelement() for p in self.net.parameters()])))
    if log_dir:
      utils.save_model_desc(self.net, join(log_dir, 'classifier_desc.txt'))
