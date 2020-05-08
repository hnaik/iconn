#!/usr/bin/env python

import logging
import json
import os
import shutil
import sys

from argparse import ArgumentParser
from pathlib import Path
from sklearn.model_selection import train_test_split

from iconn import utils as ic_utils

ic_utils.init_logging()

logger = logging.getLogger('dataset-splitter')


def read_metadata():
    with open(args.input_dir / 'metadata.json') as f:
        return json.load(f)


def make_labels(data):
    paths = []
    labels = []

    for key, value in data.items():
        for path in value:
            paths.append(path)
            labels.append(key)

    assert len(paths) == len(paths)

    return paths, labels


def write_dataset(X, y, md, name):
    assert len(X) == len(y)

    metadata = {
        'metadata': {
            'total': len(X),
            'dim': md['dim'],
            'size_begin': md['size_begin'],
            'size_end': md['size_end'],
            'thickness_begin': md['thickness_begin'],
            'thickness_end': md['thickness_end'],
        }
    }

    label_paths = {}
    output_dir = args.output_dir / name
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(len(X)):
        path = X[idx]
        label = y[idx]

        if label not in label_paths:
            label_paths.update({label: []})

        label_paths[label].append(path)

        src_path = args.input_dir / path

        if not src_path.exists():
            raise RuntimeError(f'{src_path} does not exist')

        dest_dir = output_dir / os.path.dirname(path)

        if not dest_dir.exists():
            dest_dir.mkdir(parents=True, exist_ok=True)

        dest_path = dest_dir / os.path.basename(path)
        logger.debug(f'copying {src_path} -> {dest_path}')
        shutil.copy(src_path, dest_path)

    metadata.update({'label_paths': label_paths})

    with open(output_dir / 'metadata.json', 'w') as metadata_file:
        json.dump(metadata, metadata_file, indent=4)

    logger.info(f'Wrote {name} data to {output_dir}')


def make_splits(**kwargs):
    write_dataset(
        X=kwargs['X_train'],
        y=kwargs['y_train'],
        md=kwargs['metadata'],
        name='train',
    )
    write_dataset(
        X=kwargs['X_val'], y=kwargs['y_val'], md=kwargs['metadata'], name='val'
    )
    write_dataset(
        X=kwargs['X_test'],
        y=kwargs['y_test'],
        md=kwargs['metadata'],
        name='test',
    )


def main():
    metadata = read_metadata()
    label_paths = metadata['label_paths']
    X, y = make_labels(metadata['label_paths'])
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=args.test_fraction, shuffle=args.shuffle
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=args.test_fraction, shuffle=args.shuffle
    )

    logger.info(
        f'total={len(X)}, train={len(X_train)}, test={len(X_test)}, '
        + f'val={len(X_val)}'
    )
    make_splits(
        metadata=metadata['metadata'],
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )


if __name__ == '__main__':
    parser = ArgumentParser(sys.argv)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--test-fraction', type=float, default=0.2)
    parser.add_argument('--shuffle', action='store_true', default=False)

    args = parser.parse_args()

    main()
