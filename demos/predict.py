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

import json
import logging
import sys
import torch
import urllib

from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
from torchvision import transforms

from iconn import models as ic_models
from iconn import utils as ic_utils

ic_utils.init_logging()

logger = logging.getLogger('predict')


def run_prediction():
    model = ic_models.create_model(args.arch)
    model.eval()

    input_image = Image.open(str(args.input_file))
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    # create a mini-batch as expected by the model
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to(args.device)
        model.to(args.device)

    with torch.no_grad():
        # Tensor of shape 1000, with confidence scores over
        # Imagenet's 1000 classes
        output = model(input_batch)

    # The output has unnormalized scores. To get probabilities, you can run a
    # softmax on it.
    prob = torch.nn.functional.softmax(output[0], dim=0)
    idx = prob.argmax(0)

    labels = []
    with open(args.labels_file) as f:
        d = json.load(f)
        labels = [v[1] for k, v in d.items()]

    logger.info(f'The given file {args.input_file} is of a {labels[idx]}')


def main():
    run_prediction()


if __name__ == '__main__':
    parser = ArgumentParser(sys.argv)
    parser.add_argument('--device', choices=['cuda', 'cpu'])
    parser.add_argument('--input-file', type=Path, required=True)
    parser.add_argument(
        '--labels-file', type=Path, default='imagenet_class_index.json'
    )
    parser.add_argument(
        '--arch', choices=ic_models.vanilla_model_names, default='alexnet'
    )

    args = parser.parse_args()
    main()
