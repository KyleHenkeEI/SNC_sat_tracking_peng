#!/usr/bin/env python3
"""
Chebyshev Polynomial Fitting for Pixel-to-Bearing Calibration

This script ingests pixel coordinates and their corresponding azimuth/elevation
bearings from a JSONL file, then performs 2D Chebyshev polynomial fitting.

Assumes the input file has been downselected from the giant JSON with something like:
jq 'select(.frame_number == 5743 and .camera_id == "0" and .fsm_id == "0")' bearing_lines_gs2ir_20250503_025236.jsonl | tee bearing_lines_gs2ir_20250503_025236.jsonl-frame_5743_cam0_fsm0

"""

import json
import numpy as np
from numpy.polynomial import chebyshev as cheb
import argparse
import sys


def load_jsonl_data(filename):
    """
    Load data from JSONL file and extract pixel coordinates and bearings.
    Handles both single-line and multi-line (pretty-printed) JSON objects.

    Args:
        filename: Path to JSONL file

    Returns:
        Tuple of numpy arrays (pixel_x, pixel_y, azimuth_deg, elevation_deg)
    """
    pixel_x = []
    pixel_y = []
    azimuth_deg = []
    elevation_deg = []

    with open(filename, 'r') as f:
        content = f.read()

    # Split by closing braces followed by opening braces
    # Handle both single-line and multi-line JSON objects
    json_objects = []
    current_obj = ""
    brace_count = 0

    for char in content:
        current_obj += char
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and current_obj.strip():
                try:
                    data = json.loads(current_obj.strip())
                    json_objects.append(data)
                    current_obj = ""
                except json.JSONDecodeError:
                    pass

    # Extract data from parsed objects
    for data in json_objects:
        pixel_x.append(data['pixel_x'])
        pixel_y.append(data['pixel_y'])
        azimuth_deg.append(data['azimuth_deg'])
        elevation_deg.append(data['elevation_deg'])

    return (np.array(pixel_x), np.array(pixel_y),
            np.array(azimuth_deg), np.array(elevation_deg))


def normalize_coords(x, y):
    """
    Normalize coordinates to [-1, 1] range (Chebyshev domain).

    Args:
        x: Independent variable 1 (pixel_x)
        y: Independent variable 2 (pixel_y)

    Returns:
        Tuple of (normalized_x, normalized_y, x_params, y_params)
        where params are (min, max) for denormalization
    """
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    # Normalize to [-1, 1]
    x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
    y_norm = 2 * (y - y_min) / (y_max - y_min) - 1

    return x_norm, y_norm, (x_min, x_max), (y_min, y_max)


def fit_chebyshev_2d(x, y, z, degree):
    """
    Fit a 2D Chebyshev polynomial to the data.
    Automatically normalizes coordinates to [-1, 1] range.

    Args:
        x: Independent variable 1 (pixel_x)
        y: Independent variable 2 (pixel_y)
        z: Dependent variable (azimuth or elevation)
        degree: Polynomial degree for both dimensions

    Returns:
        Tuple of (coeff, x_params, y_params) where params are for denormalization
    """
    # Normalize coordinates to [-1, 1] (Chebyshev domain)
    x_norm, y_norm, x_params, y_params = normalize_coords(x, y)

    # Use the Vandermonde matrix approach for 2D Chebyshev fitting
    # Create the Vandermonde matrix
    vander = cheb.chebvander2d(x_norm, y_norm, [degree, degree])

    # Solve the least squares problem
    # vander @ coeff_flat = z
    coeff_flat, residuals, rank, s = np.linalg.lstsq(vander, z, rcond=None)

    # Reshape coefficients to 2D matrix
    coeff = coeff_flat.reshape(degree + 1, degree + 1)

    return coeff, x_params, y_params


def evaluate_fit(x, y, coeff, x_params, y_params):
    """
    Evaluate the Chebyshev polynomial at given coordinates.
    Automatically normalizes coordinates using the provided parameters.

    Args:
        x: Independent variable 1 (pixel_x)
        y: Independent variable 2 (pixel_y)
        coeff: Chebyshev coefficient matrix
        x_params: Tuple (x_min, x_max) for normalization
        y_params: Tuple (y_min, y_max) for normalization

    Returns:
        Evaluated values
    """
    # Normalize coordinates
    x_min, x_max = x_params
    y_min, y_max = y_params
    x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
    y_norm = 2 * (y - y_min) / (y_max - y_min) - 1

    return cheb.chebval2d(x_norm, y_norm, coeff)


def compute_fit_statistics(actual, predicted):
    """
    Compute statistics for the fit quality.

    Args:
        actual: Actual values
        predicted: Predicted values from the fit

    Returns:
        Dictionary with statistics
    """
    residuals = actual - predicted
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    max_error = np.max(np.abs(residuals))

    return {
        'rmse': rmse,
        'mae': mae,
        'max_error': max_error,
        'std_residuals': np.std(residuals)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Fit 2D Chebyshev polynomials to pixel-to-bearing calibration data'
    )
    parser.add_argument(
        'jsonl_file',
        help='Path to JSONL file containing calibration data'
    )
    parser.add_argument(
        '-d', '--degree',
        type=int,
        required=True,
        help='Degree of Chebyshev polynomial (same for both dimensions)'
    )
    parser.add_argument(
        '-o', '--output',
        help='Optional output file to save coefficients (numpy .npz format)',
        default=None
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed statistics'
    )

    args = parser.parse_args()

    if args.degree < 0:
        print("Error: Degree must be non-negative", file=sys.stderr)
        sys.exit(1)

    # Load data
    print(f"Loading data from {args.jsonl_file}...")
    pixel_x, pixel_y, azimuth_deg, elevation_deg = load_jsonl_data(args.jsonl_file)
    print(f"Loaded {len(pixel_x)} data points")

    # Print data ranges
    print(f"\nData ranges:")
    print(f"  pixel_x: [{pixel_x.min()}, {pixel_x.max()}]")
    print(f"  pixel_y: [{pixel_y.min()}, {pixel_y.max()}]")
    print(f"  azimuth_deg: [{azimuth_deg.min():.4f}, {azimuth_deg.max():.4f}]")
    print(f"  elevation_deg: [{elevation_deg.min():.4f}, {elevation_deg.max():.4f}]")

    # Fit azimuth
    print(f"\nFitting azimuth with degree {args.degree} Chebyshev polynomial...")
    azimuth_coeff, azimuth_x_params, azimuth_y_params = fit_chebyshev_2d(
        pixel_x, pixel_y, azimuth_deg, args.degree
    )
    azimuth_pred = evaluate_fit(pixel_x, pixel_y, azimuth_coeff,
                                azimuth_x_params, azimuth_y_params)
    azimuth_stats = compute_fit_statistics(azimuth_deg, azimuth_pred)

    print(f"Azimuth fit statistics:")
    print(f"  RMSE: {azimuth_stats['rmse']:.6f} degrees")
    print(f"  MAE:  {azimuth_stats['mae']:.6f} degrees")
    print(f"  Max Error: {azimuth_stats['max_error']:.6f} degrees")
    if args.verbose:
        print(f"  Std of Residuals: {azimuth_stats['std_residuals']:.6f} degrees")

    # Fit elevation
    print(f"\nFitting elevation with degree {args.degree} Chebyshev polynomial...")
    elevation_coeff, elevation_x_params, elevation_y_params = fit_chebyshev_2d(
        pixel_x, pixel_y, elevation_deg, args.degree
    )
    elevation_pred = evaluate_fit(pixel_x, pixel_y, elevation_coeff,
                                  elevation_x_params, elevation_y_params)
    elevation_stats = compute_fit_statistics(elevation_deg, elevation_pred)

    print(f"Elevation fit statistics:")
    print(f"  RMSE: {elevation_stats['rmse']:.6f} degrees")
    print(f"  MAE:  {elevation_stats['mae']:.6f} degrees")
    print(f"  Max Error: {elevation_stats['max_error']:.6f} degrees")
    if args.verbose:
        print(f"  Std of Residuals: {elevation_stats['std_residuals']:.6f} degrees")

    # Save coefficients if requested
    if args.output:
        print(f"\nSaving coefficients to {args.output}...")
        np.savez(
            args.output,
            azimuth_coeff=azimuth_coeff,
            azimuth_x_params=azimuth_x_params,
            azimuth_y_params=azimuth_y_params,
            elevation_coeff=elevation_coeff,
            elevation_x_params=elevation_x_params,
            elevation_y_params=elevation_y_params,
            degree=args.degree
        )
        print("Coefficients saved successfully")

    if args.verbose:
        print(f"\nCoefficient matrix shapes:")
        print(f"  Azimuth:   {azimuth_coeff.shape}")
        print(f"  Elevation: {elevation_coeff.shape}")


if __name__ == '__main__':
    main()
