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
