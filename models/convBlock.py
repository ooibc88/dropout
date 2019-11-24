import torch
import torch.nn as nn

class drop_op(nn.Module):
    def __init__(self, drop_type=0, drop_rate=0., inplace=False):
        super(drop_op, self).__init__()
        assert (drop_type in (0, 1, 2, 3) and 0.<=drop_rate<1.)
        self.drop_type, self.keep_rate = drop_type, 1.-drop_rate
        if drop_rate == 0.:
            self.drop_op = nn.Sequential();return
        if drop_type == 0:
            self.drop_op = nn.Dropout(p=drop_rate, inplace=inplace)
        elif drop_type == 1:
            self.drop_op = nn.Dropout2d(p=drop_rate, inplace=inplace)

    def forward(self, x):
        if self.keep_rate == 1.: return x
        if self.drop_type in (0, 1): return self.drop_op(x)
        # drop-branch/layer, x in [b_0, b_1, ...], b_i: B*C_i*H*W
        if self.training:
            mask = torch.FloatTensor(len(x)).to(x[0].device).\
                bernoulli_(self.keep_rate)*(1./self.keep_rate)
            x = [x[idx]*mask[idx] for idx in range(len(x))]
        return torch.cat(x, dim=1)

class Norm2d(nn.Module):
    def __init__(self, num_features, use_gn, gn_groups, track_stats,
                 eps=1e-5, momentum=0.1):
        super(Norm2d, self).__init__()
        self.norm = nn.GroupNorm(gn_groups, num_features) if use_gn else \
            nn.BatchNorm2d(num_features)
        self.track_stats = track_stats
        if track_stats:
            self.eps, self.momentum = eps, momentum
            self.register_buffer('train_running_mean', torch.zeros(num_features))
            self.register_buffer('train_running_var', torch.ones(num_features))
            self.register_buffer('test_running_mean', torch.zeros(num_features))
            self.register_buffer('test_running_var', torch.ones(num_features))

    def stats(self):
        if not self.track_stats: return None
        return float(torch.mean(self.test_running_mean)/torch.mean(self.train_running_mean)),\
               float(torch.mean(self.test_running_var)/torch.mean(self.train_running_var))

    def forward(self, input):
        if self.track_stats:
            with torch.no_grad():
                mean = input.mean([0, 2, 3])
                var = input.var([0, 2, 3], unbiased=False)
                n = input.numel()/input.size(1)
                if self.training:
                    self.train_running_mean.mul_(1 - self.momentum).add_(self.momentum*mean)
                    self.train_running_var.mul_(1-self.momentum).add_(self.momentum*var*n/(n-1))
                else:
                    self.test_running_mean.mul_(1 - self.momentum).add_(self.momentum*mean)
                    self.test_running_var.mul_(1-self.momentum).add_(self.momentum*var*n/(n-1))
        return self.norm(input)

def norm2d_track_stats(model, is_track):
    model.apply(lambda module: setattr(module, 'track_stats', is_track)
        if hasattr(module, 'track_stats') else None)

def norm2d_stats(model):
    modules = filter(lambda module: hasattr(module, 'track_stats') and module.track_stats,
                     model.modules())
    stats = list(map(lambda module: module.stats(), modules))
    mean_var = list(map(list, zip(*stats)))
    if len(mean_var)==0: return [0.], [0.]
    return mean_var[0], mean_var[1]

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, block_type=0,
                 use_gn=False, gn_groups=8,  drop_type=0, drop_rate=0.,
                 stride=1, padding=0, groups=1, bias=False, track_stats=False):
        super(conv_block, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.drop = drop_op(drop_type, drop_rate)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                groups=groups, stride=stride, padding=padding, bias=bias)
        bn_channels = in_channels if block_type in [0, 1] else out_channels
        self.norm = Norm2d(bn_channels, use_gn, gn_groups, track_stats)

        if block_type==0:       # bn/gn-relu-drop-conv, recommended
            self.ops = nn.Sequential(self.norm, self.relu,
                                     self.drop, self.conv)
        elif block_type==1:     # bn/gn-relu-conv-drop
            self.ops = nn.Sequential(self.norm, self.relu,
                                     self.conv, self.drop)
        elif block_type==2:     # drop-conv-bn/gn-relu, recommended
            self.ops = nn.Sequential(self.drop, self.conv,
                                     self.norm, self.relu)
        elif block_type==3:     # conv-drop-bn/gn-relu
            self.ops = nn.Sequential(self.conv, self.drop,
                                     self.norm, self.relu)

    def forward(self, x):
        return self.ops(x)

