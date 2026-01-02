#!/usr/bin/env python

import argparse
import os
import sys

import numpy as np
from PIL import Image

if sys.version_info >= (3, 14):
    from compression import zstd
else:
    from backports import zstd

import gcsfs


def process_file(filename, fs):
    """Process a single raw file."""
    print(f"Processing {filename}...")

    # Read raw data as uint16
    # with zstd.open(filename, 'rb') as f:
    with fs.open(filename, 'rb', compression='infer') as f:
        decompressed_data = f.read()
        data = np.frombuffer(decompressed_data, '<u2')

    # Reshape to 4096x4096 image
    if len(data) != 4096 * 4096:
        print(f"Warning: Expected {4096*4096} pixels, got {len(data)} for {filename}")
        return None

    q00, q25, q50, q75, q100 = np.percentile(data, [0, 25, 50, 75, 100])
    return q00, q25, q50, q75, q100


def main():
    parser = argparse.ArgumentParser(description='Compute percentiles of each image, save to file')
    parser.add_argument('filename_pattern', help='Filename pattern to glob (image.rawl.zst files)')
    parser.add_argument('-n', '--dry-run', action='store_true', help='Only print image filenames that match the pattern. Do no processing.')
    parser.add_argument('--output', default='percentiles.csv', help='Name of CSV file for output')

    args = parser.parse_args()


    fs = gcsfs.GCSFileSystem(project='Sierra Nevada Corp')
    # filenames = fs.glob('ei_snc_data/data/STARLINK_30254/03/*/??/0-??????0-0-image.rawl*')
    filenames = fs.glob(args.filename_pattern)

    for filename in filenames:
        if args.dry_run:
            print(filename)
            continue

        try:
            percentiles = process_file(filename, fs)
            print(filename)
            print(percentiles)
            print()

        except Exception as e:
            print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    main()
