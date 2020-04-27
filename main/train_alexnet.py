#!/usr/bin/env python3

import logging
import sys

from argparse import ArgumentParser

import iconn.utils as ic_utils

ic_utils.init_logging()

logger = logging.getLogger('train_alexnet')


def main():
    logger.info('Starting AlexNet training')
    logger.debug('DEBUG')


if __name__ == '__main__':
    parser = ArgumentParser(sys.argv)
    parser.add_argument(
        '--log-level', choices=ic_utils.get_log_levels(), default='info'
    )

    args = parser.parse_args()

    ic_utils.set_level(logger, args.log_level)

    main()
