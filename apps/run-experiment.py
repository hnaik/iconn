#!/usr/bin/env python3

import sys
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from argparse import ArgumentParser
from pathlib import Path

from iconn import models as ic_models


def train_model(model):
    train_dir = args.data_dir / 'train'
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    dataset_train = datasets.ImageFolder(
        train_dir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        sampler=None,
    )

    model = ic_models.make_model(model_name)
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(
        model.parameters(), args.learning_rate, weight_decay=args.weight_decay
    )
    for epoch in range(0, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        train(loader_train, model, criterion, optimizer, epoch)


def train(loader, model, criterion, optimizer, epoch):
    model.eval()

    gpu = None
    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images = images.cuda(gpu, non_blocking=True)
            target = target.cuda(gpu, non_blocking=True)

            output = model(images)
            loss = criterion(output, target)

            losses.update(loss.item(), images.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def adjust_learning_rate(optimizer, epoch, args):
    """Adjust learning rate"""
    lr = args.learning_rate * (0.1 ** epoch // 30)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    train_model(args.arch)


if __name__ == '__main__':
    parser = ArgumentParser(sys.argv)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('-a', '--arch', choices=ic_models.supported_models)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=1e-4)

    args = parser.parse_args()

    main()
