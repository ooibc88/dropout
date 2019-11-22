import torch
import torch.nn as nn
import torch.nn.functional as F
from models import conv_block

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, args, stride=1):
        super(wide_basic, self).__init__()
        self.block1 = conv_block(in_planes, planes, kernel_size=3, block_type=args.block_type, use_gn=args.use_gn,
                        gn_groups=args.gn_groups, drop_type=args.drop_type, drop_rate=args.drop_rate, padding=1)
        self.block2 = conv_block(planes, planes, kernel_size=3, block_type=args.block_type, use_gn=args.use_gn,
                        gn_groups=args.gn_groups, drop_type=args.drop_type, drop_rate=0, stride=stride, padding=1)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out += self.shortcut(x)
        return out

class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, args):
        super(WideResNet, self).__init__()
        assert ((depth-4)%6 ==0), 'WideResnet depth should be 6n+4'
        n = int((depth-4)/6)
        k = widen_factor

        self.in_planes = 16
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, args, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, args, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, args, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], args.class_num)

    def _wide_layer(self, block, planes, num_blocks, args, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, args, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

def get_wrn(args):
    return WideResNet(args.depth, args.arg1, args)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='WideResNet')
    args = parser.parse_args()
    args.class_num = 10
    args.block_type = 1
    args.use_gn = True
    args.gn_groups = 16
    args.drop_type = 0
    args.drop_rate = 0.3

    net=WideResNet(28, 10, args)
    y = net(torch.randn(1,3,32,32))
    print(y.size())
    print(net)


    from convBlock import Norm2d, norm2d_stats, norm2d_track_stats

    # norm2d_track_stats(net, False)
    mean, var = norm2d_stats(net)
    print(mean)
    print(var)
