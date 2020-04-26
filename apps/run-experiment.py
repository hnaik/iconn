#!/usr/bin/env python3

"""
iconn: Interpretable Convolutional Neural Networks
Copyright (C) 2020 Harish Naik <harishgnaik@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import logging
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from argparse import ArgumentParser
from pathlib import Path

from iconn import models as ic_models
from iconn import utils as ic_utils

ic_utils.init_logging()

logger = logging.getLogger('run-experiment')


def train_model(model_name):
    logging.debug(f'Setting up training model {model_name}')

    train_dir = args.data_dir / 'train'
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    logger.debug('Initializing dataset')
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

    logger.debug('Initializing data loder')
    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        sampler=None,
    )

    model = ic_models.make_model(model_name, num_classes=1000)
    model = torch.nn.DataParallel(model).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(
        model.parameters(), args.learning_rate, weight_decay=args.weight_decay
    )

    logger.debug('Starting training')
    for epoch in range(0, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        train(loader_train, model, criterion, optimizer, epoch)
        logger.debug(f'Finished epoch {epoch}')


class Driver:
    def __init__(self, data_loader, model, optimizer, criterion):
        self.__dl = data_loader
        self.__model = model
        self.__optimizer = optimizer
        self.__criterion = criterion

    def train(self, loader):
        self.__model.train()

        with torch.no_grad():
            for i, (images, target) in enumerate(loader):
                loss = self.__compute_loss(images, target)
                loss.requires_grad = True

    def validate(self, loader):
        self.__model.eval()
        with torch.no_grad():
            for i, (images, target) in enumerate(loader):
                loss = self.__compute_loss(images, target)

    def __compute_loss(self, data, target):
        data = data.cuda()
        target = target.cuda()

        pred = self.__model(data)
        loss = self.__criterion(pred, target)


def train(loader, model, criterion, optimizer, epoch):
    model.train()

    gpu = None
    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images = images.cuda()
            target = target.cuda()

            output = model(images)
            loss = criterion(output, target)
            loss.requires_grad = True

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                logger.info(f'Finished training {i + 1} images')


def validate(loader, model, criterion):
    model.eval()

    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images = images.cuda()
            target = target.cuda()

            output = model(images)
            loss = criterion(output, target)


def adjust_learning_rate(optimizer, epoch, args):
    """Adjust learning rate"""
    lr = args.learning_rate * (0.1 ** epoch // 30)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    logger.debug('Beginning training model')
    train_model(args.arch)


if __name__ == '__main__':
    parser = ArgumentParser(sys.argv)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument(
        '-a', '--arch', choices=ic_models.supported_models, default='alexnet'
    )
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=1e-4)

    args = parser.parse_args()

    main()
