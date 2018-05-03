from torch import nn

def conv3x3(in_planes, out_planes, stride=1, bias=False):
  """3x3 convolution with padding"""
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=1, bias=bias)

def variable_init(m, neg_slope=0.0):
  if isinstance(m, (nn.Linear, nn.Conv2d)):
    nn.init.kaiming_uniform_(m.weight.data, neg_slope)
    if m.bias is not None:
      m.bias.data.zero_()
  elif isinstance(m, nn.BatchNorm2d):
    if m.weight is not None:
      m.weight.data.fill_(1)
    if m.bias is not None:
      m.bias.data.zero_()
    m.running_mean.zero_()
    m.running_var.zero_()

class PlusMinusOne(object):
  """ Scales values that are between [0, 1] into [-1, 1]. """
  
  def __init__(self):
    pass

  def __call__(self, x):
    x = 2.0 * x - 1.0
    return x    