import logging
import torch

logger = logging.getLogger(__name__)

model_names = [
    # AlexNet
    'alexnet',
    # VGG
    'vgg11',
    'vgg11_bn',
    'vgg13',
    'vgg13_bn',
    'vgg16',
    'vgg16_bn',
    'vgg19',
    'vgg19_bn',
    # ResNet
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152',
    'resnext50_32x4d',
    'resnext101_32x8d',
    'wide_resnet50_2',
    'wide_resnet101_2',
    # GoogLeNet
    'googlenet',
    # # Inception V3 Google
    # 'inception_v3_google',
]


def create_model(model_name, pretrained=True):
    """Convenience function to create model"""

    return torch.hub.load(
        'pytorch/vision:v0.5.0', model_name, pretrained=pretrained
    )
