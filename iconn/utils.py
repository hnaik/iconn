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
import logging.config
import sys

from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

log_levels = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL,
    'fatal': logging.FATAL,
}


def init_logging():
    self_path = Path(sys.argv[0])
    root_dir = self_path.parent.parent.absolute()
    logging.config.fileConfig(
        root_dir / 'logging_conf.ini', disable_existing_loggers=False
    )

    logging.addLevelName(logging.CRITICAL, 'C')
    logging.addLevelName(logging.ERROR, 'E')
    logging.addLevelName(logging.WARNING, 'W')
    logging.addLevelName(logging.INFO, 'I')
    logging.addLevelName(logging.DEBUG, 'D')


def get_log_levels():
    return log_levels.keys()


def set_level(logger_instance, level_name):
    return logger_instance.setLevel(log_levels[level_name])


def timed_routine(func):
    def wrapper(*args, **kwargs):
        start = datetime.now()
        return_value = func(*args, **kwargs)
        end = datetime.now()
        logger.info(f'Elapsed time {end - start}')
        return return_value

    return wrapper
