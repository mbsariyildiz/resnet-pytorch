import torch
import torch.nn as nn
import nn_ops

def _down_sample(x):
  return nn.functional.avg_pool2d(x, 2, 2)

def _increase_planes(x, n_out_planes):
  n_samples, n_planes, spatial_size = x.size()[:-1]
  x_zeros = torch.zeros(
    n_samples, n_out_planes - n_planes, spatial_size, spatial_size, 
    dtype=x.dtype, device=x.device)
  return torch.cat([x, x_zeros], 1)

def _downsample_and_increase_planes(x, n_out_planes):
  x = _down_sample(x)
  x = _increase_planes(x, n_out_planes)
  return x

def identity_func(n_in_planes, n_out_planes, stride):
  identity = lambda x: x
  if stride == 2 and n_in_planes != n_out_planes:
    identity = lambda x: _downsample_and_increase_planes(x, n_out_planes)
  elif stride == 2:
    identity = _down_sample
  elif n_in_planes != n_out_planes:
    identity = lambda x: _increase_planes(x, n_out_planes)
  return identity

class BasicBlock(nn.Module):

  expansion = 1

  def __init__(self, n_in_planes, n_out_planes, stride=1):
    super().__init__()
    assert stride == 1 or stride == 2

    self.block = nn.Sequential(
      nn_ops.conv3x3(n_in_planes, n_out_planes, stride),
      nn.BatchNorm2d(n_out_planes),
      nn.ReLU(inplace=True),
      nn_ops.conv3x3(n_out_planes, n_out_planes),
      nn.BatchNorm2d(n_out_planes)
    )

    self.identity = identity_func(n_in_planes, n_out_planes, stride)

  def forward(self, x):
    out = self.block(x)
    identity = self.identity(x)

    out += identity
    out = nn.functional.relu(out)
    return out

class Bottleneck(nn.Module):

  expansion = 4

  def __init__(self, n_in_planes, n_out_planes, stride=1):
    super().__init__()
    
    self.conv1 = nn.Conv2d(n_in_planes, n_out_planes, kernel_size=1)
    self.bn1 = nn.BatchNorm2d(n_out_planes)

    self.conv2 = nn_ops.conv3x3(n_out_planes, n_out_planes, stride)
    self.bn2 = nn.BatchNorm2d(n_out_planes)

    self.conv3 = nn.Conv2d(n_out_planes, n_out_planes * 4, kernel_size=1)
    self.bn3 = nn.BatchNorm2d(n_out_planes * 4)

    self.relu = nn.ReLU(inplace=True)
    self.identity = identity_func(n_in_planes, n_out_planes * 4, stride)

  def forward(self, x):
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    identity = self.identity(x)
    out += identity
    out = self.relu(out)

    return out

class ResNet(nn.Module):

  def __init__(self, block, 
                     n_blocks, 
                     n_output_planes, 
                     n_classes):
    super(ResNet, self).__init__()
    assert len(n_blocks) == 4
    assert len(n_output_planes) == 4
    
    self.n_in_planes = n_output_planes[0]

    self.layer0 = nn.Sequential(
      nn_ops.conv3x3(3, self.n_in_planes),
      nn.BatchNorm2d(self.n_in_planes),
      nn.ReLU(inplace=True)
    )
    self.layer1 = self._make_layer(block, n_blocks[0], n_output_planes[0])
    self.layer2 = self._make_layer(block, n_blocks[1], n_output_planes[1], 2)
    self.layer3 = self._make_layer(block, n_blocks[2], n_output_planes[2], 2)
    self.layer4 = self._make_layer(block, n_blocks[3], n_output_planes[3], 2)
    self.fc = nn.Linear(n_output_planes[3] * block.expansion, n_classes, False)

    self.apply(nn_ops.variable_init)

  def _make_layer(self, block, n_blocks, n_out_planes, stride=1):
    layers = []
    layers.append(block(self.n_in_planes, n_out_planes, stride))
    self.n_in_planes = n_out_planes * block.expansion
    for i in range(1, n_blocks):
      layers.append(block(self.n_in_planes, n_out_planes))

    return nn.Sequential(*layers)

  def features(self, x):
    x = self.layer0(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    spatial_size = x.size(2)
    x = nn.functional.avg_pool2d(x, spatial_size, 1)
    x = x.view(x.size(0), -1)
    return x

  def forward(self, x):
    x = self.features(x)
    x = self.fc(x)
    return x

def ResNet10(**kwargs):
  return ResNet(BasicBlock, [1,1,1,1], **kwargs)

def ResNet18(**kwargs):
  return ResNet(BasicBlock, [2,2,2,2], **kwargs)

def ResNet34(**kwargs):
  return ResNet(BasicBlock, [3,4,6,3], **kwargs)

def ResNet50(**kwargs):
  return ResNet(Bottleneck, [3,4,6,3], **kwargs)
