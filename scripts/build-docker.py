#!/usr/bin/env python3

import logging
import subprocess
import sys

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

from iconn import utils as ic_utils

ic_utils.init_logging()

logger = logging.getLogger('build-docker')

project = 'iconn'

script_path = Path(sys.argv[0])
script_dir = script_path.parent
root_dir = script_dir.parent
docker_config_dir = root_dir / 'docker'
datetime = datetime.now().strftime('%Y%m%d-%H%M%D')


def run_command(cmd):
    logger.debug(' '.join([str(s) for s in cmd]))
    ret = subprocess.run(cmd, capture_output=True, cwd=root_dir, check=True)
    out, err = ret.stdout.decode(), ret.stderr.decode()
    if not out.isspace():
        logger.debug(f'command output\n{out}')
    if ret.returncode != 0:
        logger.error(f'ERROR\n{ret.stderr.decode()}')


def main():
    docker_file = docker_config_dir / f'Dockerfile.{args.variant}'

    if not docker_file.exists():
        logger.error(
            f'No docker file {docker_file} found, variant '
            + f'{args.variant} is invalid'
        )
        sys.exit(1)

    docker_tag = f'{project}:latest'
    run_command(['docker', 'build', '-t', docker_tag, '-f', docker_file, '.'])

    if args.upload:
        docker_host = 'iridium.evl.uic.edu'
        docker_port = 5000
        docker_target = f'{docker_host}:{docker_port}/{docker_tag}'
        run_command(
            ['docker', 'image', 'tag', f'{docker_tag}', f'{docker_target}']
        )
        run_command(['docker', 'push', f'{docker_target}'])


if __name__ == '__main__':
    parser = ArgumentParser(sys.argv)
    parser.add_argument('--variant', default='pytorch', choices=['pytorch'])
    parser.add_argument('--upload', action='store_true', default=False)
    args = parser.parse_args()

    main()
