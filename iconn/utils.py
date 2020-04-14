import logging
import logging.config
import sys

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
    logging.config.fileConfig(root_dir / 'logging_conf.ini')


def get_log_levels():
    return log_levels.keys()


def set_level(logger_instance, level_name):
    return logger_instance.setLevel(log_levels[level_name])
