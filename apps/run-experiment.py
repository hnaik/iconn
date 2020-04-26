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
import numpy as np
import shutil
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import warnings

from argparse import ArgumentParser
from pathlib import Path
from PIL import ImageFile
from sklearn.metrics import precision_recall_fscore_support

from iconn import models as ic_models
from iconn import utils as ic_utils

ic_utils.init_logging()
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger('run-experiment')
warnings.filterwarnings('always')


class Driver:
    def __init__(self, model, data_loader, optimizer, criterion):
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


def save_checkpoint(state, filename):
    torch.save(state, filename)
    # if is_best:
    shutil.copyfile(filename, 'model_best.pth.tar')


def train_model(model_name):
    logger.debug(f'Setting up training model {model_name}')

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

    logger.debug('Initializing data loader')
    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        sampler=None,
    )

    val_dir = args.data_dir / 'val'
    loader_val = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            val_dir,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize
                ]
            )
        )
    )

    model = ic_models.make_model(model_name, num_classes=1000)
    model = torch.nn.DataParallel(model).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    logger.debug('Starting training')
    checkpoint_dir = args.output_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_dir / 'checkpoint.pth.tar'
    for epoch in range(0, args.epochs):
        epoch_idx = epoch + 1
        adjust_learning_rate(optimizer, epoch_idx, args)
        logger.debug(f'Starting epoch {epoch_idx} training')
        train(loader_train, model, criterion, optimizer, epoch)

        logger.debug(f'Starting epoch {epoch_idx} validation')
        correct, total, y_true, y_pred = validate(loader_val, model, criterion)
        print_accuracy(epoch_idx, correct, total, y_true, y_pred)

        save_checkpoint({
            'epoch': epoch_idx,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, filename=checkpoint_file)


def print_accuracy(epoch, correct, total, y, y_preds):
    p, r, f, s = precision_recall_fscore_support(y, y_preds)

    accuracy = correct * 100 / total
    precision = p.mean() * 100
    recall = r.mean() * 100
    f_score = f.mean() * 100

    logger.info(
        f'[Epoch={epoch:4d}] Accuracy={accuracy}, Precision={precision}, ' +
        f'Recall={recall}, F-Score={f_score}'
    )


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

            if (i + 1) % args.log_frequency == 0:
                logger.info(
                    f'Finished training {(i + 1) * args.batch_size} ' + f'images'
                )


def get_preds(out):
    return np.argmax(out)
    # res = np.zeros(out.shape[0])
    # res[idx] = 1
    # return res


def validate(loader, model, criterion):
    model.eval()

    y_correct_list = []
    y_pred_list = []
    correct = 0

    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            try:
                images = images.cuda()
                target = target.cuda()

                output = model(images)
                loss = criterion(output, target)

                y_pred = output.argmax(dim=1, keepdim=True)
                y_correct = y_pred.eq(target.view_as(y_pred)).sum().item()

                y_correct_list.append(target.to('cpu'))
                y_pred_list.append(get_preds(output.squeeze().to('cpu')))

                correct += (1 if y_pred_list[-1] == y_correct_list[-1] else 0)
            except OSError as err:
                logger.error(f'{err}')

    return correct, len(y_correct_list), y_correct_list, y_pred_list




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
    parser.add_argument('-e', '--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('-w', '--workers', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--log-frequency', type=int, default=100)
    parser.add_argument('-m','--momentum', type=float, default=0.9)

    args = parser.parse_args()

    try:
        main()
        logger.debug('End of experiment')
    except:
        logger.error(f'Program exited with an error')
        raise
