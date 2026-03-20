#!/usr/bin/env python3
"""
Evaluate Chebyshev Polynomial Calibration

This script loads saved Chebyshev coefficients and evaluates them at specified
pixel coordinates to get azimuth and elevation bearings.
"""

import numpy as np
from numpy.polynomial import chebyshev as cheb
import argparse


def evaluate_calibration(pixel_x, pixel_y, coeff_file):
    """
    Evaluate the calibration at given pixel coordinates.

    Args:
        pixel_x: Pixel x coordinate(s) - can be scalar or array
        pixel_y: Pixel y coordinate(s) - can be scalar or array
        coeff_file: Path to .npz file containing coefficients

    Returns:
        Tuple of (azimuth_deg, elevation_deg)
    """
    # Load coefficients
    data = np.load(coeff_file)
    azimuth_coeff = data['azimuth_coeff']
    elevation_coeff = data['elevation_coeff']
    azimuth_x_params = tuple(data['azimuth_x_params'])
    azimuth_y_params = tuple(data['azimuth_y_params'])
    elevation_x_params = tuple(data['elevation_x_params'])
    elevation_y_params = tuple(data['elevation_y_params'])

    # Normalize coordinates for azimuth
    x_min, x_max = azimuth_x_params
    y_min, y_max = azimuth_y_params
    x_norm = 2 * (pixel_x - x_min) / (x_max - x_min) - 1
    y_norm = 2 * (pixel_y - y_min) / (y_max - y_min) - 1

    # Evaluate azimuth
    azimuth = cheb.chebval2d(x_norm, y_norm, azimuth_coeff)

    # Normalize coordinates for elevation
    x_min, x_max = elevation_x_params
    y_min, y_max = elevation_y_params
    x_norm = 2 * (pixel_x - x_min) / (x_max - x_min) - 1
    y_norm = 2 * (pixel_y - y_min) / (y_max - y_min) - 1

    # Evaluate elevation
    elevation = cheb.chebval2d(x_norm, y_norm, elevation_coeff)

    return azimuth, elevation


def evaluate_derivatives(pixel_x, pixel_y, coeff_file):
    """
    Evaluate the derivatives of the calibration at given pixel coordinates.

    Computes:
    - d(azimuth)/d(pixel_x), d(azimuth)/d(pixel_y)
    - d(elevation)/d(pixel_x), d(elevation)/d(pixel_y)

    Args:
        pixel_x: Pixel x coordinate(s) - can be scalar or array
        pixel_y: Pixel y coordinate(s) - can be scalar or array
        coeff_file: Path to .npz file containing coefficients

    Returns:
        Tuple of (daz_dx, daz_dy, del_dx, del_dy) in degrees per pixel
    """
    # Load coefficients
    data = np.load(coeff_file)
    azimuth_coeff = data['azimuth_coeff']
    elevation_coeff = data['elevation_coeff']
    azimuth_x_params = tuple(data['azimuth_x_params'])
    azimuth_y_params = tuple(data['azimuth_y_params'])
    elevation_x_params = tuple(data['elevation_x_params'])
    elevation_y_params = tuple(data['elevation_y_params'])

    # Compute derivatives of Chebyshev coefficients
    # chebder(c, axis=0) gives derivative with respect to first variable (x)
    # chebder(c, axis=1) gives derivative with respect to second variable (y)
    azimuth_coeff_dx = cheb.chebder(azimuth_coeff, m=1, axis=0)
    azimuth_coeff_dy = cheb.chebder(azimuth_coeff, m=1, axis=1)
    elevation_coeff_dx = cheb.chebder(elevation_coeff, m=1, axis=0)
    elevation_coeff_dy = cheb.chebder(elevation_coeff, m=1, axis=1)

    # Normalize coordinates for azimuth
    x_min, x_max = azimuth_x_params
    y_min, y_max = azimuth_y_params
    x_norm = 2 * (pixel_x - x_min) / (x_max - x_min) - 1
    y_norm = 2 * (pixel_y - y_min) / (y_max - y_min) - 1

    # Evaluate derivatives in normalized space
    daz_dx_norm = cheb.chebval2d(x_norm, y_norm, azimuth_coeff_dx)
    daz_dy_norm = cheb.chebval2d(x_norm, y_norm, azimuth_coeff_dy)

    # Apply chain rule: df/dx = (df/dx_norm) * (dx_norm/dx)
    # where dx_norm/dx = 2 / (x_max - x_min)
    dx_norm_dx = 2.0 / (x_max - x_min)
    dy_norm_dy = 2.0 / (y_max - y_min)
    daz_dx = daz_dx_norm * dx_norm_dx
    daz_dy = daz_dy_norm * dy_norm_dy

    # Normalize coordinates for elevation
    x_min, x_max = elevation_x_params
    y_min, y_max = elevation_y_params
    x_norm = 2 * (pixel_x - x_min) / (x_max - x_min) - 1
    y_norm = 2 * (pixel_y - y_min) / (y_max - y_min) - 1

    # Evaluate derivatives in normalized space
    del_dx_norm = cheb.chebval2d(x_norm, y_norm, elevation_coeff_dx)
    del_dy_norm = cheb.chebval2d(x_norm, y_norm, elevation_coeff_dy)

    # Apply chain rule
    dx_norm_dx = 2.0 / (x_max - x_min)
    dy_norm_dy = 2.0 / (y_max - y_min)
    del_dx = del_dx_norm * dx_norm_dx
    del_dy = del_dy_norm * dy_norm_dy

    return daz_dx, daz_dy, del_dx, del_dy


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Chebyshev polynomial calibration at pixel coordinates'
    )
    parser.add_argument(
        'coeff_file',
        help='Path to .npz file containing calibration coefficients'
    )
    parser.add_argument(
        'pixel_x',
        type=float,
        help='Pixel x coordinate'
    )
    parser.add_argument(
        'pixel_y',
        type=float,
        help='Pixel y coordinate'
    )
    parser.add_argument(
        '--human-readable',
        action='store_true',
        help='Output in human-readable format with labels'
    )
    parser.add_argument(
        '--derivatives',
        action='store_true',
        help='Compute and output derivatives (deg/pixel)'
    )

    args = parser.parse_args()

    # Evaluate calibration
    azimuth, elevation = evaluate_calibration(
        args.pixel_x, args.pixel_y, args.coeff_file
    )

    if args.derivatives:
        # Compute derivatives
        daz_dx, daz_dy, del_dx, del_dy = evaluate_derivatives(
            args.pixel_x, args.pixel_y, args.coeff_file
        )

        if args.human_readable:
            print(f"Pixel coordinates: ({args.pixel_x}, {args.pixel_y})")
            print(f"Azimuth:   {azimuth:.6f} degrees")
            print(f"Elevation: {elevation:.6f} degrees")
            print(f"\nDerivatives (degrees/pixel):")
            print(f"  dAz/dx:  {daz_dx:.9f}")
            print(f"  dAz/dy:  {daz_dy:.9f}")
            print(f"  dEl/dx:  {del_dx:.9f}")
            print(f"  dEl/dy:  {del_dy:.9f}")
        else:
            # Default: output values then derivatives
            print(f"{azimuth:.6f} {elevation:.6f} {daz_dx:.9f} {daz_dy:.9f} {del_dx:.9f} {del_dy:.9f}")
    else:
        if args.human_readable:
            print(f"Pixel coordinates: ({args.pixel_x}, {args.pixel_y})")
            print(f"Azimuth:   {azimuth:.6f} degrees")
            print(f"Elevation: {elevation:.6f} degrees")
        else:
            # Default: just output the numbers
            print(f"{azimuth:.6f} {elevation:.6f}")


if __name__ == '__main__':
    main()
