import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from iconn import utils as ic_utils
from iconn.models import functions as ic_func

ic_utils.init_logging()

logger = logging.getLogger(__name__)


class DefaultNet(nn.Module):
    def __init__(self, num_classes, output_dir=None):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),
            nn.Flatten(),
            nn.Linear(3136, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, X):
        if X.is_cuda:
            return nn.parallel.data_parallel(self.main, X, range(1))


class CorrectiveAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X):
        return X

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class InterpretableNet(nn.Module):
    def __init__(self, num_classes, output_dir, write_plots, template_norm):
        super().__init__()

        self.stage_1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),  #
            nn.ReLU(inplace=True),
        )

        self.stage_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  #
            nn.ReLU(inplace=True),
        )

        self.max_pool = nn.MaxPool2d(kernel_size=3)

        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(3136, num_classes), nn.Sigmoid()
        )

        self.template_norm = template_norm
        self.index = 0
        self.write_plots = write_plots

        logger.debug(f'Initialized network {self.__class__}')

    def forward(self, X):
        def apply_parallel_X(component):
            return nn.parallel.data_parallel(component, X, range(1))

        if not self.training:
            self.index += 1

        if X.is_cuda:
            X = apply_parallel_X(self.stage_1)

            if self.training:
                if self.template_norm == 'l1':
                    X = ic_func.Filter_Stage1_L1.apply(X)
                elif self.template_norm == 'l2':
                    X = ic_func.Filter_Stage1_L2.apply(X)
                # Else no change
            else:
                if self.template_norm == 'l1':
                    X = ic_func.IntermediateLogger_Stage1_L1.apply(X)
                elif self.template_norm == 'l2':
                    X = ic_func.IntermediateLogger_Stage1_L2.apply(X)
                else:
                    X = ic_func.IntermediateLogger_Stage1_Original.apply(X)

                if self.write_plots:
                    self.__output(
                        X,
                        output_dir=self.stage_1_dir,
                        stage=0,  # stage 0-indexed
                        tag=f'id-{self.index}_{self.template_norm}',
                    )

            X = apply_parallel_X(self.max_pool)
            X = apply_parallel_X(self.stage_2)

            if self.training:
                if self.template_norm == 'l1':
                    X = ic_func.Filter_Stage2_L1.apply(X)
                elif self.template_norm == 'l2':
                    X = ic_func.Filter_Stage2_L2.apply(X)
                # Else no change
            else:
                if self.template_norm == 'l1':
                    X = ic_func.IntermediateLogger_Stage2_L1.apply(X)
                elif self.template_norm == 'l2':
                    X = ic_func.IntermediateLogger_Stage2_L2.apply(X)
                else:
                    X = ic_func.IntermediateLogger_Stage2_Original.apply(X)

                if self.write_plots:
                    self.__output(
                        X,
                        output_dir=self.stage_2_dir,
                        stage=1,  # stage 0-indexed
                        tag=f'id-{self.index}_{self.template_norm}',
                    )

            X = apply_parallel_X(self.max_pool)
            return apply_parallel_X(self.classifier)

    def __output(self, X, output_dir, stage, tag):
        stage_id = f'stage-{stage}'
        for idx, output in enumerate(X[0]):
            prefix = f'{tag}_id-{idx}'
            image = output.cpu().detach().numpy()
            plt.imshow(image, cmap='gray')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_dir / f'{prefix}.png')


def make_model(arch, num_classes, output_dir, args):
    logger.debug(f'making model {arch} with {num_classes} classes')

    if arch == 'interpretable':
        return InterpretableNet(
            num_classes, output_dir, args.write_plots, args.template_norm
        )

    return DefaultNet(num_classes, output_dir)
