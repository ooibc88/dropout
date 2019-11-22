import os
import torch
import time
import math
import logging
import sys
import shutil

# setup logger
def logger(log_dir, need_time=True, need_stdout=False):
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_dir)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y-%I:%M:%S')
    if need_stdout:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        log.addHandler(ch)
    if need_time:
        fh.setFormatter(formatter)
        if need_stdout:
            ch.setFormatter(formatter)
    log.addHandler(fh)
    return log

def timeSince(since=None, s=None):
    if s is None:
        s = int(time.time() - since)
    m = math.floor(s / 60)
    s %= 60
    h = math.floor(m / 60)
    m %= 60
    return '%dh %dm %ds' %(h, m, s)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size).item())

    return res

def load_checkpoint(args):
    if os.path.isfile(args.resume): return torch.load(args.resume)
    elif args.resume == 'checkpoint':
        if args.resume_best: return torch.load('{0}{1}'.format(args.checkpoint_dir, 'best_model.ckpt'))
        else: return torch.load('{0}{1}'.format(args.checkpoint_dir, 'model.ckpt'))
    else: sys.exit("=> no checkpoint found at '{}'".format(args.resume))

def save_checkpoint(ckpt, is_best, file_dir, file_name='model.ckpt'):
    if not os.path.exists(file_dir): os.makedirs(file_dir)
    ckpt_name = "{0}{1}".format(file_dir, file_name)
    torch.save(ckpt, ckpt_name)
    if is_best: shutil.copyfile(ckpt_name, "{0}{1}".format(file_dir, 'best_'+file_name))

def list_formatter(l, format='.4f'):
    return '['+', '.join([f'{i:{format}}' for i in l])+']'
