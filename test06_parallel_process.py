#!/usr/bin/env python

import argparse
import glob
import itertools
import os
import subprocess
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image

if sys.version_info >= (3, 14):
    from compression import zstd
else:
    from backports import zstd

import gcsfs


def downsample_max(image, factor):
    """Downsample using maximum value in each block."""
    h, w = image.shape
    new_h, new_w = h // factor, w // factor

    downsampled = np.zeros((new_h, new_w), dtype=image.dtype)

    for i in range(new_h):
        for j in range(new_w):
            block = image[i*factor:(i+1)*factor, j*factor:(j+1)*factor]
            downsampled[i, j] = np.max(block)

    return downsampled

def downsample_min(image, factor):
    """Downsample using minimum value in each block."""
    h, w = image.shape
    new_h, new_w = h // factor, w // factor

    downsampled = np.zeros((new_h, new_w), dtype=image.dtype)

    for i in range(new_h):
        for j in range(new_w):
            block = image[i*factor:(i+1)*factor, j*factor:(j+1)*factor]
            downsampled[i, j] = np.min(block)

    return downsampled

def downsample_average(image, factor):
    """Downsample using average value in each block."""
    h, w = image.shape
    new_h, new_w = h // factor, w // factor

    downsampled = np.zeros((new_h, new_w), dtype=image.dtype)

    for i in range(new_h):
        for j in range(new_w):
            block = image[i*factor:(i+1)*factor, j*factor:(j+1)*factor]
            downsampled[i, j] = np.mean(block)

    return downsampled

def downsample_lanczos(image, factor):
    """Downsample using Lanczos resampling via PIL."""
    h, w = image.shape
    new_h, new_w = h // factor, w // factor

    # Convert to PIL Image
    pil_image = Image.fromarray(image)

    # Resize using Lanczos
    resized = pil_image.resize((new_w, new_h), Image.LANCZOS)

    return np.array(resized)

def process_single_case(camera, tile, method, iqr_factor, downsample_factor, output_dir):
    """Process a single case with given parameters, processing all matching files."""
    # Create case-specific subdirectory
    case_dir = os.path.join(output_dir, f"cam{camera}_tile{tile}_{method}_iqr{iqr_factor}_down{downsample_factor}")
    os.makedirs(case_dir, exist_ok=True)

    # Create GCS filesystem (per-process)
    fs = gcsfs.GCSFileSystem(project='Sierra Nevada Corp')

    # Build filename pattern
    pattern = f'ei_snc_data/data/STARLINK_30254/03/*/??/{camera}-??????{tile}-{tile}-image.rawl*'
    print(f"Processing: cam={camera}, tile={tile}, method={method}, iqr={iqr_factor}, factor={downsample_factor}")
    print(f"  Output directory: {case_dir}")

    # Find matching files
    filenames = fs.glob(pattern)

    if not filenames:
        print(f"Warning: No files found for pattern {pattern}")
        return []

    print(f"Found {len(filenames)} file(s) matching pattern")

    results = []
    for filename in filenames:
        try:
            # Extract the wildcard portion (6 characters between camera and tile)
            # Pattern: {camera}-{wildcard}{tile}-{tile}-image.rawl*
            basename = os.path.basename(filename)
            parts = basename.split('-')
            if len(parts) >= 3:
                wildcard_id = parts[1]  # The 6-character unique identifier
            else:
                wildcard_id = "unknown"

            # Read raw data as uint16
            with fs.open(filename, 'rb', compression='infer') as f:
                decompressed_data = f.read()
                data = np.frombuffer(decompressed_data, '<u2')

            # Reshape to 4096x4096 image
            if len(data) != 4096 * 4096:
                print(f"Warning: Expected {4096*4096} pixels, got {len(data)} for {filename}")
                continue

            # Apply IQR-based scaling with parameterized IQR factor
            q00, q25, q50, q75, q100 = np.percentile(data, [0, 25, 50, 75, 100])
            IQR = q75 - q25
            # min value is either the minimum of the data or the editing limit, whichever is greater
            min_value = max(q00,  q25 - iqr_factor*IQR)
            # max value is either the maximum of the data or the editing limit, whichever is less
            max_value = min(q100, q75 + iqr_factor*IQR)

            M = 2**16 - 1
            data = M/(max_value - min_value) * (data - min_value)
            data = np.clip(data, 0, M)
            data = data.astype(np.uint16)

            image = data.reshape(4096, 4096)

            # Downsample based on method
            if method == 'max':
                downsampled = downsample_max(image, downsample_factor)
            elif method == 'min':
                downsampled = downsample_min(image, downsample_factor)
            elif method == 'average':
                downsampled = downsample_average(image, downsample_factor)
            elif method == 'lanczos':
                downsampled = downsample_lanczos(image, downsample_factor)
            else:
                raise ValueError(f"Unknown method: {method}")

            # Ensure data is uint16 for 16-bit PGM
            if downsampled.dtype != np.uint16:
                downsampled = downsampled.astype(np.uint16)

            # Generate output filename with wildcard ID in case-specific directory
            output_filename = os.path.join(case_dir,
                                          f"{wildcard_id}.pgm")

            # Save as 16-bit grayscale PGM
            pil_image = Image.fromarray(downsampled, mode='I;16')
            pil_image.save(output_filename)

            print(f"✓ Saved: {output_filename} (shape: {downsampled.shape})")
            results.append(output_filename)

        except Exception as e:
            print(f"✗ Error processing {filename}: {e}")

    # Create MP4 video from PGM files if any were created
    if results:
        mp4_filename = f"{case_dir}.mp4"
        print(f"Creating MP4 video: {mp4_filename}")
        try:
            # Use ffmpeg to create video from PGM files
            cmd = [
                'ffmpeg', '-y',  # Overwrite output file if it exists
                '-framerate', '15',
                '-pattern_type', 'glob',
                '-i', f'{case_dir}/*.pgm',
                '-c:v', 'libx264',
                '-crf', '18',
                '-pix_fmt', 'yuv420p',
                mp4_filename
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"✓ Created video: {mp4_filename}")

            # Delete PGM files to save disk space
            print(f"Deleting PGM files from {case_dir}")
            pgm_files = glob.glob(os.path.join(case_dir, '*.pgm'))
            for pgm_file in pgm_files:
                os.remove(pgm_file)
            print(f"✓ Deleted {len(pgm_files)} PGM file(s)")

            # Remove the empty case directory
            os.rmdir(case_dir)
            print(f"✓ Removed directory: {case_dir}")

        except subprocess.CalledProcessError as e:
            print(f"✗ Error creating MP4: {e.stderr.decode()}")
        except Exception as e:
            print(f"✗ Error in cleanup: {e}")

    return results

def main():
    parser = argparse.ArgumentParser(description='Parallel processing of 4096x4096 raw images with parameter grid')
    parser.add_argument('--output-dir', default='downsampled',
                       help='Output directory for downsampled images')
    parser.add_argument('--processes', type=int, default=4,
                       help='Number of parallel processes (default: 4)')

    args = parser.parse_args()

    # Create base output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Define parameter grid
    cameras = [0, 1]
    tiles = range(6)  # [0, 1, 2, 3, 4, 5]
    # methods = ['average', 'max']
    methods = ['max']
    iqr_factors = [2.0, 2.5, 3.0]
    # factors = [16, 32]
    factors = [16]

    # Generate all combinations
    cases = list(itertools.product(cameras, tiles, methods, iqr_factors, factors))

    print(f"Total cases to process: {len(cases)}")
    print(f"Running with {args.processes} parallel processes")
    print("-" * 80)

    # Prepare arguments for starmap (add output_dir to each case)
    starmap_args = [(cam, tile, method, iqr, factor, args.output_dir)
                    for cam, tile, method, iqr, factor in cases]

    # Process in parallel
    with Pool(processes=args.processes) as pool:
        results = pool.starmap(process_single_case, starmap_args)

    # Summary - results is a list of lists (each case returns list of files processed)
    print("-" * 80)
    total_files = sum(len(r) for r in results)
    cases_with_results = sum(1 for r in results if len(r) > 0)
    print(f"Processing complete:")
    print(f"  Total parameter cases: {len(results)}")
    print(f"  Cases with successful files: {cases_with_results}/{len(results)}")
    print(f"  Total files processed: {total_files}")


if __name__ == "__main__":
    main()
