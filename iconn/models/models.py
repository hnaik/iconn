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
