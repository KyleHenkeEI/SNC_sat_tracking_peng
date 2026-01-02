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

def process_file(filename, downsample_factor, method, fs):
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

    return downsampled

def main():
    parser = argparse.ArgumentParser(description='Downsample multiple 4096x4096 raw image files')
    parser.add_argument('filename_pattern', help='Filename pattern to glob (image.rawl.zst files)')
    parser.add_argument('-n', '--dry-run', action='store_true', help='Only print image filenames that match the pattern. Do no processing.')
    parser.add_argument('--factor', type=int, required=True,
                       help='Downsampling factor (must be power of 2)')
    parser.add_argument('--method', choices=['max', 'min', 'average', 'lanczos'],
                       default='average', help='Downsampling method')
    parser.add_argument('--output-dir', default='downsampled',
                       help='Output directory for downsampled images')

    args = parser.parse_args()


    fs = gcsfs.GCSFileSystem(project='Sierra Nevada Corp')
    # filenames = fs.glob('ei_snc_data/data/STARLINK_30254/03/*/??/0-??????0-0-image.rawl*')
    filenames = fs.glob(args.filename_pattern)

    # Validate factor is power of 2
    if args.factor <= 0 or (args.factor & (args.factor - 1)) != 0:
        raise ValueError("Factor must be a positive power of 2")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    for filename in filenames:
        if args.dry_run:
            print(filename)
            continue
        try:
            downsampled = process_file(filename, args.factor, args.method, fs)
            if downsampled is not None:
                # Generate output filename
                base_name = os.path.splitext(os.path.basename(filename))[0]
                output_filename = os.path.join(args.output_dir,
                                             f"{base_name}_down{args.factor}_{args.method}.pgm")

                # Ensure data is uint16 for 16-bit PGM
                if downsampled.dtype != np.uint16:
                    downsampled = downsampled.astype(np.uint16)

                # Save as 16-bit grayscale PGM

                # multiply by 4 if wanted to get 14 bits up to 16
                # pil_image = Image.fromarray(4 * downsampled, mode='I;16')

                # ran tool to find min_value=3530 and max_value=16332. Expand that out to full range, crop if outside
                # min_value = 3530 
                # min_value = 9667
                # min_value = 9517
                # min_value = 9153
                min_value = 0

                # max_value = 16332
                # max_value = 2**13-1
                # max_value = 11012
                # max_value = 11691
                max_value = 2**14 -1

                M = 2**16 - 1
                downsampled = M/(max_value - min_value) * (downsampled - min_value)
                downsampled = downsampled.astype(np.uint16)
                downsampled = np.clip(downsampled, 0, M)
                pil_image = Image.fromarray(downsampled, mode='I;16')

                pil_image.save(output_filename)
                print(f"Saved downsampled image to {output_filename}")
                print(f"Original size: 4096x4096, New size: {downsampled.shape}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    main()
