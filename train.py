import os
import time
import argparse

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim import lr_scheduler
import torch.utils.data
import torch.utils.data.distributed

from models import *
from data_loader import data_loader
from utils import logger, AverageMeter, timeSince, list_formatter
from utils import load_checkpoint, save_checkpoint, accuracy
from tensorboardX import SummaryWriter
start_time = time.time()

parser = argparse.ArgumentParser(description='Dropout CIFAR10/100, SVHN, ImageNet')
parser.add_argument('--exp_name', default='test', type=str, help='exp name used to store log and checkpoint')
parser.add_argument('--net_type', default='vgg', type=str, help='network type: vgg, resnet, resnext, densenet etc')
parser.add_argument('--depth', type=int, default=100, help='depth of network')
parser.add_argument('--arg1', type=int, default=10, help='addition arg1, e.g. widen_factor')
parser.add_argument('--arg2', type=int, default=64, help='addition arg2, e.g. base_width (ResNeXt)')

parser.add_argument('--block_type', type=int, default=1, help='specify block_type (default: 1)')
parser.add_argument('--use_gn', action='store_true', default=False, help='whether to use group norm (default: False)')
parser.add_argument('--gn_groups', type=int, default=8, help='group norm groups')
parser.add_argument('--drop_type', type=int, default=0, help='0-drop-neuron, 1-drop-channel, 2-drop-path, 3-drop-layer')
parser.add_argument('--drop_rate', default=0.0, type=float, help='dropout rate')

parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lr_scheduler', type=str, default='step', help='learning rate scheduler, default: step lr')
parser.add_argument('--batch_size', type=int, default=128, help='batch size (default: 128)')
parser.add_argument('--epoch', type=int, default=300, help='number of epochs to train (default: 300)')
parser.add_argument('--start_epoch', default=0, type=int, help='start epoch number (resume training)')

parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', action='store_true', default=False, help='whether to use nesterov (default: False)')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay (default: 1e-4)')

parser.add_argument('--dataset', type=str, default='cifar10', help='dataset name for data_loader')
parser.add_argument('--data-dir', default='./data/', type=str, metavar='PATH', help='path to dataset')
parser.add_argument('--no-augment', dest='augment', action='store_false', help='use standard augmentation (D: True)')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')

parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--resume_best', action='store_true', default=False, help='resume best_checkpoint (default: False)')
parser.add_argument('--checkpoint', action='store_true', default=False, help='whether to checkpoint (default: False)')
parser.add_argument('--checkpoint_dir', default='~/checkpoint/', type=str, metavar='PATH', help='path to checkpoint')
parser.add_argument('--tensorboard_dir', default='~/tensorboard/', type=str, metavar='PATH', help='path to tensorboard')

parser.add_argument('--report_freq', type=float, default=20, help='report frequency')
parser.add_argument('--report_ratio', action='store_true', default=False, help='report mean_var shift ratio (D: False)')
parser.set_defaults(augment=True)

args = parser.parse_args()
# init global variables, load dataset
best_err1, best_err5 = 100., 100.
if not os.path.isdir(f'log/{args.exp_name}'): os.mkdir(f'log/{args.exp_name}')
args.checkpoint_dir = f'{os.path.expanduser(args.checkpoint_dir)}{args.exp_name}/'
writer = SummaryWriter(f'{os.path.expanduser(args.tensorboard_dir)}{args.exp_name}/')
train_loader, val_loader, args.class_num = data_loader(args)

def creat_model(print_logger):
    print_logger.info("=> creating model '{}'".format(args.net_type))
    if args.dataset.startswith('cifar'):
        model_map = {'wrn': get_wrn,
                     'resnet': get_resnet,
                     'densenet': get_densenet,
                     'resnext': get_resnext,
                     }
    elif args.dataset == 'imagenet':
        model_map = {'wrn': None,
                     }
    model = model_map[args.net_type](args)
    model = torch.nn.DataParallel(model).cuda()
    print_logger.info(f'model parameters: {sum([p.data.nelement() for p in model.parameters()])}')
    return model

lr_sheduler_map = {
	'cosine': lambda optimizer: lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, last_epoch=args.last_epoch),
	'exp': lambda optimizer: lr_scheduler.ExponentialLR(
		optimizer, gamma=0.97, last_epoch=args.last_epoch),
	'step': lambda optimizer: lr_scheduler.MultiStepLR(optimizer,
        [int(args.epoch*0.5), int(args.epoch*0.75)], gamma=0.1),
    'imagenet': lambda optimizer: lr_scheduler.MultiStepLR(optimizer,
        [int(args.epoch*0.3), int(args.epoch*0.6), int(args.epoch*0.9)], gamma=0.1),
    'wrn': lambda optimizer: lr_scheduler.MultiStepLR(optimizer,
        [int(args.epoch*0.3), int(args.epoch*0.6), int(args.epoch*0.8)], gamma=0.2),
}

def main():
    global args, best_err1, best_err5, lr_sheduler_map
    plogger = logger('log/{}/stdout.log'.format(args.exp_name), True, True)
    plogger.info(vars(args))

    model = creat_model(plogger)
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    scheduler = lr_sheduler_map[args.lr_scheduler](optimizer)

    # resume from checkpoint
    if args.resume:
        ckpt = load_checkpoint(args)
        args.start_epoch = ckpt['epoch']+1
        best_err1 = ckpt['best_err1']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        plogger.info(f"=> finish loading checkpoint {args.resume} (epoch {ckpt['epoch']})")

    cudnn.benchmark = True
    for epoch in range(args.start_epoch, args.epoch):
        plogger.info(f'Epoch [{epoch:3d}/{args.epoch:3d}]\tlr: {scheduler.get_lr()[0]:.6f}')

        # train and eval
        run(epoch, model, train_loader, criterion, plogger, optimizer=optimizer)
        err1, err5 = run(epoch, model, val_loader, criterion, plogger)

        scheduler.step(epoch)
        # record best prec@1 and save checkpoint
        is_best = (err1 <= best_err1)
        best_err1 = min(err1, best_err1); best_err5 = min(err5, best_err5)
        if is_best: plogger.info(f'best err:\ttop1 = {best_err1:.4f} top5 = {best_err5:.4f}')
        if args.checkpoint:
            save_checkpoint({
                'epoch': epoch,
                'best_err1': best_err1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, file_dir=args.checkpoint_dir)
        if args.report_ratio:
            mean_var = norm2d_stats(model)
            plogger.info(f'\nrunning mean\t{list_formatter(mean_var[0])}\n'
                         f'running var\t{list_formatter(mean_var[1])}')
            writer.add_scalar(f'mean_shift_ratio/1st', mean_var[0][0], epoch)
            writer.add_scalar(f'var_shift_ratio/1st', mean_var[1][0], epoch)
            writer.add_scalar(f'mean_shift_ratio/mid', mean_var[0][len(mean_var[0])//2], epoch)
            writer.add_scalar(f'var_shift_ratio/mid', mean_var[1][len(mean_var[0])//2], epoch)
            writer.add_scalar(f'mean_shift_ratio/last', mean_var[0][-1], epoch)
            writer.add_scalar(f'var_shift_ratio/last', mean_var[1][-1], epoch)

    plogger.info(f'best err:\ttop1 = {best_err1:.4f} top5 = {best_err5:.4f}')
    plogger.info(f'Total training time: {timeSince(since=start_time)}')

#  one epoch of train/val/test
def run(epoch, model, data_loader, criterion, plogger, optimizer=None):
    global args
    is_train = True if optimizer!=None else False
    if is_train: model.train()
    else: model.eval()
    namespace = 'Train' if is_train else 'Test'

    time_avg = AverageMeter()
    loss_avg, top1_avg, top5_avg = AverageMeter(), AverageMeter(), AverageMeter()

    timestamp = time.time()
    for idx, (input, target) in enumerate(data_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        if is_train:
            output = model(input)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                output = model(input)
                loss = criterion(output, target)

        err1, err5 = accuracy(output, target, topk=(1, 5))
        loss_avg.update(loss.item(), input.size(0))
        top1_avg.update(err1, input.size(0))
        top5_avg.update(err5, input.size(0))
        # tensorboardX
        writer.add_scalar(f'{namespace}/loss', loss_avg.val, idx+len(data_loader)*epoch)
        writer.add_scalar(f'{namespace}/top-1', top1_avg.val, idx+len(data_loader)*epoch)
        writer.add_scalar(f'{namespace}/top-5', top5_avg.val, idx+len(data_loader)*epoch)

        time_avg.update(time.time()-timestamp)
        timestamp = time.time()

        if idx % args.report_freq == 0:
            plogger.info(f'Epoch [{epoch:3d}/{args.epoch:3d}][{idx:3d}/{len(data_loader):3d}]\t'
                         f'{time_avg.val:.3f} ({time_avg.avg:.3f})\t'
                         f'Loss {loss_avg.val:4f} ({loss_avg.avg:4f})\t'
                         f'top1 {top1_avg.val:8.4f} ({top1_avg.avg:8.4f})\t'
                         f'top5 {top5_avg.val:8.4f} ({top5_avg.avg:8.4f})')

    plogger.info(f'{namespace}\tTime {timeSince(s=time_avg.sum):>12s}\t'
                 f'top1 {top1_avg.avg:8.4f}\ttop5 {top5_avg.avg:8.4f}\tLoss {loss_avg.avg:4f}')
    return top1_avg.avg, top5_avg.avg

if __name__ == '__main__':
    main()
