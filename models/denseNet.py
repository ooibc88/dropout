import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import conv_block

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, args):
        super(BasicBlock, self).__init__()
        self.conv = conv_block(in_planes, out_planes, 3, args.block_type,
                        args.use_gn, args.gn_groups, args.drop_type,
                        args.drop_rate, padding=1, track_stats=args.report_ratio)

    def forward(self, x):
        out = self.conv(x)
        return torch.cat([x, out], 1)

class Bottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, args):
        super(Bottleneck, self).__init__()
        inter_planes = out_planes * 4
        self.conv1 = conv_block(in_planes, inter_planes, 1, args.block_type,
                                args.use_gn, args.gn_groups, args.drop_type, args.drop_rate,
                                track_stats=args.report_ratio)
        self.conv2 = conv_block(inter_planes, out_planes, 3, args.block_type,
                                args.use_gn, args.gn_groups, args.drop_type, args.drop_rate,
                                padding=1, track_stats=args.report_ratio)

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, args):
        super(TransitionBlock, self).__init__()
        self.conv = conv_block(in_planes, out_planes, 1, args.block_type,
                        args.use_gn, args.gn_groups, args.drop_type, args.drop_rate,
                        track_stats=args.report_ratio)

    def forward(self, x):
        out = self.conv(x)
        return F.avg_pool2d(out, 2)

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_planes, growth_rate, block, args):
        super(DenseBlock, self).__init__()
        self.layer = nn.Sequential(*[block(in_planes+i*growth_rate, growth_rate, args)
                                     for i in range(num_layers)])

    def forward(self, x):
        return self.layer(x)

# For CIFAR-10/100 dataset
class DenseNet(nn.Module):
    def __init__(self, args, growth_rate=12,
                 reduction=0.5, bottleneck=True):
        super(DenseNet, self).__init__()
        in_planes = 2 * growth_rate
        n = int((args.depth - 4) / 3)
        if bottleneck == True:
            n = n//2
            block = Bottleneck
        else:
            block = BasicBlock
        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = DenseBlock(n, in_planes, growth_rate, block, args)
        in_planes = int(in_planes+n*growth_rate)
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), args)
        in_planes = int(math.floor(in_planes*reduction))
        # 2nd block
        self.block2 = DenseBlock(n, in_planes, growth_rate, block, args)
        in_planes = int(in_planes+n*growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), args)
        in_planes = int(math.floor(in_planes*reduction))
        # 3rd block
        self.block3 = DenseBlock(n, in_planes, growth_rate, block, args)
        in_planes = int(in_planes+n*growth_rate)
        # global average pooling and classifier
        self.bn = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_planes, args.class_num)
        self.in_planes = in_planes

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_planes)
        return self.fc(out)


# https://github.com/liuzhuang13/DenseNet#results-on-cifar
# CIFAR DenseNet3(depth=100, num_classes=10., growth_rate=12, reduction=0.5, bottleneck=True, drop_rate=0.2)
# SVHN  DenseNet3(depth=100, num_classes=10., growth_rate=24, reduction=0.5, bottleneck=True, drop_rate=0.2)
# DenseNet3(depth=250, num_classes=10., growth_rate=24, reduction=0.5, bottleneck=True, drop_rate=0.2)
# DenseNet3(depth=190, num_classes=10., growth_rate=40, reduction=0.5, bottleneck=True, drop_rate=0.2)
def get_densenet(args):
    return DenseNet(args, args.arg1)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='WideResNet')
    args = parser.parse_args()
    args.depth = 100
    args.class_num = 10
    args.block_type = 0
    args.use_gn = False
    args.gn_groups = 6
    args.drop_type = 1
    args.drop_rate = 0.1
    args.report_ratio = True
    args.arg1 = 12

    net = DenseNet(args, args.arg1)
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
    print(net)
    print(sum([p.data.nelement() for p in net.parameters()]))

    from convBlock import Norm2d, norm2d_stats, norm2d_track_stats

    # norm2d_track_stats(net, False)
    mean, var = norm2d_stats(net)
    print(len(mean), mean)
    print(var)
