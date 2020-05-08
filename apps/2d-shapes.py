#!/usr/bin/env python3

# iconn: Interpretable Convolutional Neural Networks
# Copyright (C) 2020 Harish G. Naik <hnaik2@uic.edu>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import cv2
import json
import logging
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import sys

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from PIL import Image
from sklearn import preprocessing
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from types import SimpleNamespace

from iconn import utils as ic_utils
from iconn.models import interpretable as ip_models

ic_utils.init_logging()

logger = logging.getLogger('2d-shapes')


torch.manual_seed(0)


def plot_heatmap(weights, output_dir, stem):
    # file_path = output_dir / f'{stem}.png'
    # cv2.imwrite(str(output_path), weights)

    for idx, weight in enumerate(weights):
        plt.imshow(weight, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        file_path = output_dir / f'{stem}_channel-{idx}.png'
        plt.savefig(file_path)


def plot_layers(layers, output_dir):
    for layer_idx, layer in enumerate(layers):
        output_path = output_dir / str(layer_idx + 1)
        output_path.mkdir(parents=True, exist_ok=True)
        for idx, w in enumerate(layer):
            plot_heatmap(w, output_path, f'layer-{idx + 1}')


class Image64Dataset(Dataset):
    def __init__(self, X, y):
        # self.__df = df
        self.__X = X
        self.__y = y
        self.__n = len(X)

        unique_labels = np.unique(self.__y)
        label_encoder = preprocessing.LabelEncoder()
        targets = label_encoder.fit_transform(unique_labels)
        self.targets = torch.as_tensor(targets)

        self.label_dict = {
            label: idx for idx, label in enumerate(unique_labels)
        }

        # self.transform = transforms.Compose(
        #     [transforms.Grayscale(), transforms.ToTensor()]
        # )
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.__n

    def __getitem__(self, idx):
        try:
            image_path = self.__X[idx]
            image = Image.open(image_path)
            y = self.__y[idx]
            image = self.transform(image)
            y_ = torch.zeros(4)
            y_[self.targets[self.label_dict[y]]] = 1.0
            return image, y_
        except KeyError as err:
            logger.info(
                f'ERROR {err} index={idx} ({err}), shape={self.__X.shape}'
            )


class DataSplitter:
    def __init__(
        self, input_dir, test_size, batch_size, shuffle, x_labels, y_label
    ):
        self.batch_size = batch_size
        self.__labels = set()
        self.train_loader, self.train_size = self.load_dataset(
            input_dir, 'train', batch_size=batch_size, shuffle=True
        )
        self.test_loader, self.test_size = self.load_dataset(input_dir, 'test')

    @property
    def num_classes(self):
        return len(self.__labels)

    def __load_metadata(self, input_dir):
        with open(input_dir / 'metadata.json') as f:
            return json.load(f)

    def load_dataset(self, input_dir, dataset_id, batch_size=1, shuffle=False):
        dataset_dir = input_dir / dataset_id
        md = self.__load_metadata(dataset_dir)
        label_paths = md['label_paths']
        labels = label_paths.keys()

        X = []
        y = []
        for key, values in label_paths.items():
            for value in values:
                X.append(dataset_dir / value)
                y.append(key)
                self.__labels.add(key)

        loader = DataLoader(
            Image64Dataset(X, y), batch_size=batch_size, shuffle=shuffle
        )

        return loader, len(y)


def initialize_model(
    input_dir, output_dir, test_size, device, batch_size, shuffle=True
):
    splitter = DataSplitter(
        input_dir=input_dir,
        test_size=test_size,
        batch_size=batch_size,
        shuffle=shuffle,
        x_labels='image_path',
        y_label='label',
    )
    net = ip_models.make_model(
        arch=args.arch,
        num_classes=splitter.num_classes,
        output_dir=output_dir,
        args=args,
    )

    if not args.gpu_id:
        logger.info(
            f'No GPU ID specified, using all {torch.cuda.device_count()} GPUs'
        )
        net = nn.DataParallel(net)

    net = net.to(device)

    # net = ip_models.DefaultNet(
    #     num_classes=splitter.num_classes, output_dir=args.output_dir
    # ).to(device)

    optimizer = optim.Adam(net.parameters())

    return SimpleNamespace(
        **dict(
            net=net,
            criterion=nn.BCELoss().cuda(0),
            optimizer=optimizer,
            scheduler=StepLR(optimizer, step_size=1),
            splitter=splitter,
        )
    )


def get_conv_weights(model):
    layers = []
    for m in model.modules():
        if type(m) == nn.Conv2d:
            layers.append(m.weight.data.squeeze().cpu().numpy())
    return layers


def get_preds(output, size):
    norm_output = np.zeros(size, dtype=np.float32)
    idx = np.argmax(output.cpu().detach().numpy())
    norm_output[idx] = 1.0
    return norm_output


def train_epoch(params, epoch, device):
    params.net.train()

    begin = datetime.now()
    for i, (data, label) in enumerate(params.splitter.train_loader):
        params.net.zero_grad()

        # logger.info(f'copying to device {device}')

        data = data.to(device)
        label = label.to(device)

        # logger.info(f'forward')
        output = params.net(data)

        # logger.info('loss')
        loss = params.criterion(output, label)

        params.optimizer.zero_grad()

        # logger.info('backward')
        loss.backward()

        params.optimizer.step()

        idx = i + 1
        processed = idx * params.splitter.batch_size
        if processed % args.log_frequency == 0:
            now = datetime.now()
            duration = now - begin

            logger.info(
                f'[{idx:5d}] Epoch {epoch + 1}, {processed:8d} of '
                + f'{params.splitter.train_size}, '
                + f'avg. time per item {duration / processed} '
                + f'avg. time per iteration {duration / idx}'
            )
        # logger.info(f'finished iteration {i + 1}')

        # Don't update, since we want to count since the beginning
        # begin = now


@ic_utils.timed_routine
def train(params, epochs, device, output_dir):
    params.net = params.net

    for epoch in range(epochs):
        logger.info(f'starting epoch {epoch + 1}')
        begin = datetime.now()
        train_epoch(params, epoch, device)
        duration = datetime.now() - begin
        logger.info(f'Finished epoch {epoch + 1}, time {duration}')

    output_dir.mkdir(parents=True, exist_ok=True)
    model_to_save = {
        'model': params.net.state_dict(),
        'optimizer': params.optimizer.state_dict(),
        'arch': args.arch,
    }
    saved_model_dir = output_dir / 'models'
    saved_model_dir.mkdir(parents=True, exist_ok=True)
    model_file_path = saved_model_dir / 'models.pt'
    torch.save(model_to_save, model_file_path)

    # torch.save(params.net.state_dict(), str(output_dir / 'models.pt'))


@ic_utils.timed_routine
def test(params, device):
    test_loss = 0
    correct = 0
    idx = 0
    test_size = params.splitter.test_size
    num_classes = params.splitter.num_classes
    y = np.zeros(test_size * num_classes).reshape(test_size, num_classes)
    y_preds = np.zeros(test_size * num_classes).reshape(test_size, num_classes)

    count_true = 0
    total = 0
    logger.info('Starting test')
    params.net.eval()
    with torch.no_grad():
        for idx, (data, target) in enumerate(params.splitter.test_loader):
            total = idx + 1
            data = data.to(device)
            target = target.to(device)
            output = params.net(data)

            y[idx] = target.to('cpu')
            y_preds[idx] = get_preds(output.squeeze(), num_classes)

            test_loss += params.criterion(output, target).item()
            y_pred = output.argmax(dim=1, keepdim=True)
            # y_correct = y_pred.eq(target.view_as(y_pred)).sum().item()

            correct += (
                1
                if (
                    y_preds[idx][0] == y[idx][0]
                    and y_preds[idx][1] == y[idx][1]
                )
                else 0
            )

            count_true += y[idx]

            if (idx + 1) % 10 == 0:
                logger.debug(f'done processing {idx + 1} test samples')

    logger.info(
        f'[{total}] correct={correct}, len(y)={len(y)}, '
        + f'len(y_preds)={len(y_preds)}'
    )

    return correct, total, y, y_preds


def main():
    logger.info('Starting 2D shapes experiment')

    if args.gpu_id:
        logger.info(f'Using GPU {args.gpu_id}')
    else:
        logger.info(
            f'No GPU specified, using {torch.cuda.device_count()} available GPUs'
        )

    output_dir = args.output_dir / datetime.now().strftime('%Y%m%d-%H%M%S')
    output_dir.mkdir(parents=True, exist_ok=True)
    params = initialize_model(
        args.input_dir,
        output_dir,
        test_size=args.test_fraction,
        device=args.device,
        batch_size=args.batch_size,
    )

    if not args.pretrained_model_path:
        logger.info('No pretrained model provided, starting training')
        train(
            params,
            epochs=args.epochs,
            device=args.device,
            output_dir=output_dir,
        )
    else:
        logger.info(
            f'Loading pre-trained model from {args.pretrained_model_path}'
        )

        saved = torch.load(args.pretrained_model_path)

        logger.info('loaded pretrained model')
        params.net.load_state_dict(saved['model'])
        params.optimizer.load_state_dict(saved['optimizer'])
        logger.info('loaded state dictionary')

        params.net = nn.DataParallel(params.net)

    correct, total, y, y_preds = test(params, device=args.device)

    accuracy = correct * 100.0 / total
    p, r, f, s = precision_recall_fscore_support(y, y_preds)
    precision = p.mean() * 100
    recall = r.mean() * 100
    f_score = f.mean() * 100
    logger.info(
        f'[{args.epochs}, {accuracy:0.2f}, {precision:0.2f}, {recall:0.2f}, '
        + f'{f_score:0.2f}]'
    )

    if args.plot:
        layers = get_conv_weights(params.net)
        plot_layers(
            layers, output_dir / f'visualizations/epochs-{args.epochs}'
        )


if __name__ == '__main__':
    parser = ArgumentParser(sys.argv)
    parser.add_argument(
        '--arch',
        type=str,
        default='default',
        choices=['default', 'interpretable'],
    )
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--device', choices=['cuda', 'cpu'], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--test-fraction', type=float, default=0.2)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('--log-frequency', type=int, default=1000)
    parser.add_argument('--write-plots', action='store_true', default=False)
    parser.add_argument(
        '--template-norm', choices=['l1', 'l2', 'original'], default='original'
    )
    parser.add_argument('--gpu-id', type=int, default=None)
    parser.add_argument(
        '--cache-dir', type=int, help='Directory to write temporary files'
    )
    parser.add_argument('--pretrained-model-path', type=Path)
    parser.add_argument('--template-cache-dir', type=Path, required=True)

    args = parser.parse_args()

    main()
