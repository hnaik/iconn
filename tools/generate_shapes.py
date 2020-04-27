#!/usr/bin/env python3

import cv2
import logging
import matplotlib.pylab as plt
import numpy as np
import sys

from argparse import ArgumentParser
from pathlib import Path

from iconn import utils as ic_utils

ic_utils.init_logging()

logger = logging.getLogger('generate_shape')

colors = {'red': (0, 0, 255), 'green': (0, 255, 0), 'blue': (255, 0, 0)}


def write_file(image, output_dir, stem):
    output_path = output_dir / f'{stem}.png'
    cv2.imwrite(str(output_path), image)


def write_square(begin, end, color, thickness, output_dir):
    image = np.zeros((args.dim, args.dim, 3))
    cv2.rectangle(image, begin, end, color, thickness)
    f_stem = f'r{begin[0]}-c{begin[1]}_sz{end[0] - begin[0]}_t{thickness}'
    write_file(image, output_dir, f_stem)


def write_circle(center, radius, color, thickness, output_dir):
    image = np.zeros((args.dim, args.dim, 3))
    cv2.circle(image, center, radius, color, thickness)
    f_stem = f'r{center[0]}-c{center[1]}_d{radius * 2}_t{thickness}'
    write_file(image, output_dir, f_stem)


def make_triangle(corner, side_length, color, thickness):
    image = np.zeros((args.dim, args.dim, 3))

    x1 = corner[0]
    y1 = corner[1]

    x2 = corner[0] - side_length // 2
    y2 = y1 + int(np.sqrt(3) / 2 * side_length)

    x3 = x2 + side_length
    y3 = y2

    cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    cv2.line(image, (x2, y2), (x3, y3), color, thickness)
    cv2.line(image, (x3, y3), (x1, y1), color, thickness)

    return image


def write_triangle(point, side_length, color, thickness, output_dir):
    image = make_triangle(point, side_length, color, thickness)
    f_stem = f'r{point[0]}-c{point[1]}_l{side_length}_t{thickness}'
    write_file(image, output_dir, f_stem)


def make_shapes(size_range, position_start=(0, 0), thickness_range=(1, 5)):
    square_output_root = args.output_dir / f'{args.dim}x{args.dim}/squares'
    circle_output_root = args.output_dir / f'{args.dim}x{args.dim}/circles'
    triangle_output_root = args.output_dir / f'{args.dim}x{args.dim}/triangles'

    for color, color_code in colors.items():
        square_output_dir = square_output_root / f'{color}'
        square_output_dir.mkdir(parents=True, exist_ok=True)

        circle_output_dir = circle_output_root / f'{color}'
        circle_output_dir.mkdir(parents=True, exist_ok=True)

        triangle_output_dir = triangle_output_root / f'{color}'
        triangle_output_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(f'color={color}, code={color_code}')

        for thickness in range(thickness_range[0], thickness_range[1]):
            for size in range(size_range[0], size_range[1]):
                for row_begin in range(position_start[0], args.dim - size):
                    for col_begin in range(position_start[1], args.dim - size):
                        image = np.zeros((args.dim, args.dim, 3))

                        position_begin = (row_begin, col_begin)
                        position_end = (row_begin + size, col_begin + size)

                        write_square(
                            position_begin,
                            position_end,
                            color_code,
                            thickness,
                            square_output_dir,
                        )

                        radius = size // 2
                        center = (row_begin + radius, col_begin + radius)
                        write_circle(
                            center,
                            radius,
                            color_code,
                            thickness,
                            circle_output_dir,
                        )

                        point = (row_begin, col_begin + (size // 2))
                        write_triangle(
                            point,
                            size,
                            color_code,
                            thickness,
                            triangle_output_dir,
                        )


def main():
    make_shapes(size_range=(16, args.dim))


if __name__ == '__main__':
    parser = ArgumentParser(sys.argv)
    parser.add_argument('-d', '--dim', type=int, default=32)
    parser.add_argument('-o', '--output-dir', type=Path, required=True)

    args = parser.parse_args()

    main()
