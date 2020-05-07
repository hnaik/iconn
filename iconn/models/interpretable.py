import logging
import math
import matplotlib.pyplot as plt
import multiprocessing as mp

from multiprocessing import Pool
import numpy as np
from queue import Queue
import time
import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms

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


class ImageProcessor:
    def __init__(self, n_queues):
        self.__queues = [mp.Queue()] * n_queues
        self.p_1 = mp.Process(target=self.process, args=(0,))
        self.p_2 = mp.Process(target=self.process, args=(1,))

    def start(self):
        logger.info('Starting image processor queues')
        self.p_1.start()
        self.p_2.start()

    def add_item(self, idx, item):
        self.__queues[int(idx)].put(item, block=False)

    def process(self, idx):
        while True:
            item = self.__queues[int(idx)].get(block=True, timeout=None)
            image = transforms.ToPILImage()(item['image'].cpu().detach())
            plt.imshow(image, cmap='gray')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(item['path'])


class InterpretableNet(nn.Module):
    iteration = 0
    # image_proc = ImageProcessor(2)

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

        if self.write_plots:
            self.stage_1_dir = output_dir / 'stage_1'
            self.stage_1_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                f'writing stage 1 visualizations to {self.stage_1_dir}'
            )

            self.stage_2_dir = output_dir / 'stage_2'
            self.stage_2_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                f'writing stage 2 visualizations to {self.stage_2_dir}'
            )

        # self.image_proc.start()
        logger.info(f'Initialized network {self.__class__}')

    def forward(self, X):
        def apply_parallel_X(component):
            # return nn.parallel.data_parallel(component, X, range(1))
            return component(X)

        InterpretableNet.iteration += 1

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
                    self.__output(X, output_dir=self.stage_1_dir, stage=1)

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
                    self.__output(X, output_dir=self.stage_2_dir, stage=2)

            X = apply_parallel_X(self.max_pool)
            return apply_parallel_X(self.classifier)

    def __output(self, X, output_dir, stage):
        tag = f'id-{InterpretableNet.iteration:07d}_{self.template_norm}'
        logger.info(
            f'plotting index {InterpretableNet.iteration} stage {stage}'
        )
        for i, X_i in enumerate(X):
            for j, X_ij in enumerate(X_i):
                # self.image_proc.add_item(
                #     idx=stage - 1,
                #     item={
                #         'image': X_ij,
                #         'path': output_dir / f'{tag}_{i:03d}-{j:03d}.png',
                #     },
                # )

                image = X_ij.cpu().detach().numpy()
                x = X_ij.cpu().detach()
                image = transforms.ToPILImage()(x)
                plt.imshow(image, cmap='gray')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(output_dir / f'{tag}_{i}-{j}.png')


def make_model(arch, num_classes, output_dir, args):
    logger.debug(f'making model {arch} with {num_classes} classes')

    if arch == 'interpretable':
        return InterpretableNet(
            num_classes, output_dir, args.write_plots, args.template_norm
        )

    return DefaultNet(num_classes, output_dir)
