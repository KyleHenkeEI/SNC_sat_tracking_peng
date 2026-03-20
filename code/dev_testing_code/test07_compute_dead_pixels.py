#!/usr/bin/env python

import argparse
import glob
import itertools
import os
import subprocess
import sys
from multiprocessing import Pool

import numpy as np

if sys.version_info >= (3, 14):
    from compression import zstd
else:
    from backports import zstd

import gcsfs


def process_single_case(camera, tile, output_dir):
    """Process a single case with given parameters, processing all matching files."""
    # Create case-specific subdirectory
    case_dir = os.path.join(output_dir, f"cam{camera}_tile{tile}")
    os.makedirs(case_dir, exist_ok=True)

    # Create GCS filesystem (per-process)
    fs = gcsfs.GCSFileSystem(project='Sierra Nevada Corp')

    # Build filename pattern
    pattern = f'ei_snc_data/data/STARLINK_30254/03/*/??/{camera}-??????{tile}-{tile}-image.rawl*'
    print(f"Processing: cam={camera}, tile={tile}")
    print(f"  Output directory: {case_dir}")

    # Find matching files
    filenames = fs.glob(pattern)

    if not filenames:
        print(f"Warning: No files found for pattern {pattern}")
        return []

    print(f"Found {len(filenames)} file(s) matching pattern")

    results = []

    first_time = True

    max_files = 10

    for num_file, filename in enumerate(filenames):
        if num_file > max_files: break

        try:
            # Read raw data as uint16
            with fs.open(filename, 'rb', compression='infer') as f:
                decompressed_data = f.read()
                data = np.frombuffer(decompressed_data, '<u2')

            if first_time:
                first_data = data.copy()
                first_time = False
                data_has_not_changed = np.ones(len(first_data))

            data_has_not_changed = np.logical_and(data_has_not_changed, first_data == data)
            count = np.count_nonzero(data_has_not_changed)
            print(f"Cam{camera} tile{tile} # constant pixels: {count}")

            results.append(filename)

            # save the dead pixel map to file at each iteration to enable early stopping
            output_filename = os.path.join(case_dir, 'dead_pixels.txt')
            with open(output_filename, 'w') as f:
                for i, dead in enumerate(data_has_not_changed):
                    if dead:
                        f.write(str(i))
                        f.write('\n')

        except Exception as e:
            print(f"✗ Error processing {filename}: {e}")


    return results


def main():
    parser = argparse.ArgumentParser(description='Parallel processing of 4096x4096 raw images with parameter grid')
    parser.add_argument('--output-dir', default='downsampled',
                       help='Output directory for dead pixel information')
    parser.add_argument('--processes', type=int, default=4,
                       help='Number of parallel processes (default: 4)')

    args = parser.parse_args()

    # Create base output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Define parameter grid
    cameras = [0, 1]
    # cameras = [1]

    tiles = range(6)  # [0, 1, 2, 3, 4, 5]
    # tiles = [0]

    # Generate all combinations
    cases = list(itertools.product(cameras, tiles))

    print(f"Total cases to process: {len(cases)}")
    print(f"Running with {args.processes} parallel processes")
    print("-" * 80)

    # Prepare arguments for starmap (add output_dir to each case)
    starmap_args = [(cam, tile, args.output_dir)
                    for cam, tile in cases]

    # Process in parallel
    with Pool(processes=args.processes) as pool:
        results = pool.starmap(process_single_case, starmap_args)

    print(f"Processing complete:")


if __name__ == "__main__":
    main()
