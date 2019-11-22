import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def data_loader(args):
    args.data_dir += args.dataset
    if args.dataset.startswith('cifar'):
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        if args.augment:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                normalize,
                ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
            ])

        if args.dataset == 'cifar100':
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100(args.data_dir, train=True, download=False, transform=transform_train),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100(args.data_dir, train=False, transform=transform_test),
                batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
            class_num = 100
        elif args.dataset == 'cifar10':
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(args.data_dir, train=True, download=False, transform=transform_train),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(args.data_dir, train=False, transform=transform_test),
                batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
            class_num = 10
        else:
            raise Exception ('unknown dataset: {}'.format(args.dataset))
        return train_loader, val_loader, class_num

    elif args.dataset == 'svhn':
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_dataset = datasets.SVHN(root=args.data_dir,
                                      split='train',
                                      download=False,
                                      transform=transform, )
        extra_dataset = datasets.SVHN(root=args.data_dir,
                                      split='extra',
                                      download=False,
                                      transform=transform, )
        train_dataset += extra_dataset

        train_loader = torch.utils.data.DataLoader(train_dataset,
                           batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                           pin_memory=True)

        valid_loader = torch.utils.data.DataLoader(
            datasets.SVHN(root=args.data_dir, split='test', download=False, transform=transform, ),
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        return train_loader, valid_loader, 10

    elif args.dataset == 'imagenet':
        traindir = os.path.join(args.data_dir, 'train')
        valdir = os.path.join(args.data_dir, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        return train_loader, val_loader, 1000

    else:
        raise Exception ('unknown dataset: {}'.format(args.dataset))