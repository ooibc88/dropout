import torch
import torch.nn as nn
import torch.nn.functional as F
from models import conv_block

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, out_planes, cardinality, d_width, stride, args):
        super(Bottleneck, self).__init__()
        conv_width = (cardinality*d_width)*(out_planes//256)
        self.block1 = conv_block(in_planes, conv_width, 1, args.block_type, args.use_gn,
                                 args.gn_groups, args.drop_type, args.drop_rate,
                                 track_stats=args.report_ratio)
        self.block2 = conv_block(conv_width, conv_width, 3, args.block_type, args.use_gn,
                                 args.gn_groups, args.drop_type, args.drop_rate, stride, 1,
                                 cardinality, track_stats=args.report_ratio)
        self.conv3 = nn.Conv2d(conv_width, out_planes, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, 1, stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        out = self.block2(self.block1(x))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNeXt(nn.Module):
    def __init__(self, args, cardinality, d_width):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.d_width = d_width
        self.block_num = (args.depth - 2) // 9
        stages = [64, 64*Bottleneck.expansion, 128*Bottleneck.expansion, 256*Bottleneck.expansion]

        self.conv1 = nn.Conv2d(3, stages[0], 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(stages[0])
        self.layer1 = self._make_layer(stages[0], stages[1], 1, args)
        self.layer2 = self._make_layer(stages[1], stages[2], 2, args)
        self.layer3 = self._make_layer(stages[2], stages[3], 2, args)
        self.classifier = nn.Linear(stages[3], args.class_num)

    def _make_layer(self, in_planes, out_planes, stride, args):
        strides = [stride] + [1]*(self.block_num-1)
        layers = []
        for idx, stride in enumerate(strides):
            layers.append(Bottleneck(in_planes if idx==0 else out_planes, out_planes,
                                     self.cardinality, self.d_width, stride, args))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.avg_pool2d(x, 8)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def get_resnext(args):
    return ResNeXt(args, args.arg1, args.arg2)
