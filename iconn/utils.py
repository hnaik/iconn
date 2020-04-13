import logging
import logging.config
import sys

from pathlib import Path


def init_logging():
    self_path = Path(sys.argv[0])
    root_dir = self_path.parent.parent.absolute()
    logging.config.fileConfig(root_dir / 'logging_conf.ini')
