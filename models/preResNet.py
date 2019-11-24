import torch
import torch.nn as nn
from models import conv_block

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, args, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.block2 = conv_block(planes, planes, 3, args.block_type, args.use_gn, args.gn_groups,
                                 args.drop_type, args.drop_rate, stride=stride, padding=1,
                                 track_stats=args.report_ratio)
        self.block3 = conv_block(planes, planes*Bottleneck.expansion, 1, block_type=0,
                                 use_gn=False, drop_rate=0., track_stats=False)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(x))

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)
        out = self.block2(out)
        out = self.block3(out)

        out += residual

        return out


class preResNet(nn.Module):
    def __init__(self, args, widen_factor=1.):
        super(preResNet, self).__init__()
        self.inplanes = int(16*widen_factor)
        n = int((args.depth - 2) / 9)

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(int(16*widen_factor), n, args)
        self.layer2 = self._make_layer(int(32*widen_factor), n, args, stride=2)
        self.layer3 = self._make_layer(int(64*widen_factor), n, args, stride=2)
        self.bn = nn.BatchNorm2d(int(64 * widen_factor) * Bottleneck.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(int(64*widen_factor) * Bottleneck.expansion, args.class_num)

    def _make_layer(self, planes, blocks, args, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )

        layers = []
        layers.append(Bottleneck(self.inplanes, planes, args, stride, downsample))
        self.inplanes = planes * Bottleneck.expansion
        for i in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, args))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu(self.bn(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def get_resnet(args):
    return preResNet(args, args.arg1)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PreResNet')
    args = parser.parse_args()
    args.depth=110
    args.class_num = 10
    args.block_type = 0
    args.use_gn = False
    args.gn_groups = 16
    args.drop_type = 1
    args.drop_rate = 0.1
    args.report_ratio = True

    net = preResNet(args)
    y = net((torch.randn(1, 3, 32, 32)))
    print(y.size())
    print(net)
    print(sum([p.data.nelement() for p in net.parameters()]))

    from convBlock import Norm2d, norm2d_stats, norm2d_track_stats

    # norm2d_track_stats(net, False)
    mean, var = norm2d_stats(net)
    print(mean)
    print(var)