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

import logging
import torch

from .alexnet import AlexNet, InterpretableAlexNet

logger = logging.getLogger(__name__)

vanilla_model_names = {
    'alexnet',
    'vgg11',
    'vgg11_bn',
    'vgg13',
    'vgg13_bn',
    'vgg16',
    'vgg16_bn',
    'vgg19',
    'vgg19_bn',
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152',
    'resnext50_32x4d',
    'resnext101_32x8d',
    'wide_resnet50_2',
    'wide_resnet101_2',
    'googlenet',
}
supported_models = {'alexnet'}


def create_model(model_name, pretrained=True):
    """Convenience function to create model - from torch hub"""

    return torch.hub.load(
        'pytorch/vision:v0.5.0', model_name, pretrained=pretrained
    )


def make_model(model_name, *args, **kwargs):
    """Factory funtion to create project supported models"""

    logger.info(f'Creating model {model_name}')

    if model_name == 'alexnet':
        return AlexNet(*args, **kwargs)
    elif model_name == 'i-alexnet':
        return InterpretableAlexNet(*args, **kwargs)

    raise RuntimeError(f'Model {model_name} not supported')
