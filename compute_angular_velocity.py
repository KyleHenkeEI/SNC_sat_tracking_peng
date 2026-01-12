#!/usr/bin/env python3
"""
Compute Angular Velocity from Pixel Velocity

Given an object's velocity in pixel space (dx/dt, dy/dt), compute the total
angular velocity as seen by the observer using the Chebyshev calibration.
"""

import numpy as np
from numpy.polynomial import chebyshev as cheb
import argparse


def load_calibration(coeff_file):
    """Load calibration coefficients from file."""
    data = np.load(coeff_file)
    return {
        'azimuth_coeff': data['azimuth_coeff'],
        'elevation_coeff': data['elevation_coeff'],
        'azimuth_x_params': tuple(data['azimuth_x_params']),
        'azimuth_y_params': tuple(data['azimuth_y_params']),
        'elevation_x_params': tuple(data['elevation_x_params']),
        'elevation_y_params': tuple(data['elevation_y_params'])
    }


def evaluate_at_pixel(pixel_x, pixel_y, cal):
    """Evaluate azimuth and elevation at given pixel coordinates."""
    # Azimuth
    x_min, x_max = cal['azimuth_x_params']
    y_min, y_max = cal['azimuth_y_params']
    x_norm = 2 * (pixel_x - x_min) / (x_max - x_min) - 1
    y_norm = 2 * (pixel_y - y_min) / (y_max - y_min) - 1
    azimuth = cheb.chebval2d(x_norm, y_norm, cal['azimuth_coeff'])

    # Elevation
    x_min, x_max = cal['elevation_x_params']
    y_min, y_max = cal['elevation_y_params']
    x_norm = 2 * (pixel_x - x_min) / (x_max - x_min) - 1
    y_norm = 2 * (pixel_y - y_min) / (y_max - y_min) - 1
    elevation = cheb.chebval2d(x_norm, y_norm, cal['elevation_coeff'])

    return azimuth, elevation


def compute_derivatives(pixel_x, pixel_y, cal):
    """Compute partial derivatives at given pixel coordinates."""
    # Compute derivative coefficient matrices
    azimuth_coeff_dx = cheb.chebder(cal['azimuth_coeff'], m=1, axis=0)
    azimuth_coeff_dy = cheb.chebder(cal['azimuth_coeff'], m=1, axis=1)
    elevation_coeff_dx = cheb.chebder(cal['elevation_coeff'], m=1, axis=0)
    elevation_coeff_dy = cheb.chebder(cal['elevation_coeff'], m=1, axis=1)

    # Azimuth derivatives
    x_min, x_max = cal['azimuth_x_params']
    y_min, y_max = cal['azimuth_y_params']
    x_norm = 2 * (pixel_x - x_min) / (x_max - x_min) - 1
    y_norm = 2 * (pixel_y - y_min) / (y_max - y_min) - 1

    daz_dx_norm = cheb.chebval2d(x_norm, y_norm, azimuth_coeff_dx)
    daz_dy_norm = cheb.chebval2d(x_norm, y_norm, azimuth_coeff_dy)

    dx_norm_dx = 2.0 / (x_max - x_min)
    dy_norm_dy = 2.0 / (y_max - y_min)
    daz_dx = daz_dx_norm * dx_norm_dx
    daz_dy = daz_dy_norm * dy_norm_dy

    # Elevation derivatives
    x_min, x_max = cal['elevation_x_params']
    y_min, y_max = cal['elevation_y_params']
    x_norm = 2 * (pixel_x - x_min) / (x_max - x_min) - 1
    y_norm = 2 * (pixel_y - y_min) / (y_max - y_min) - 1

    del_dx_norm = cheb.chebval2d(x_norm, y_norm, elevation_coeff_dx)
    del_dy_norm = cheb.chebval2d(x_norm, y_norm, elevation_coeff_dy)

    dx_norm_dx = 2.0 / (x_max - x_min)
    dy_norm_dy = 2.0 / (y_max - y_min)
    del_dx = del_dx_norm * dx_norm_dx
    del_dy = del_dy_norm * dy_norm_dy

    return daz_dx, daz_dy, del_dx, del_dy


def compute_angular_velocity(pixel_x, pixel_y, dx_dt, dy_dt, coeff_file):
    """
    Compute total angular velocity from pixel velocity.

    Args:
        pixel_x: Current pixel x coordinate
        pixel_y: Current pixel y coordinate
        dx_dt: Pixel velocity in x direction (pixels/time)
        dy_dt: Pixel velocity in y direction (pixels/time)
        coeff_file: Path to calibration coefficients

    Returns:
        Dictionary with angular velocity components and total
    """
    cal = load_calibration(coeff_file)

    # Get current azimuth and elevation
    azimuth, elevation = evaluate_at_pixel(pixel_x, pixel_y, cal)

    # Get partial derivatives
    daz_dx, daz_dy, del_dx, del_dy = compute_derivatives(pixel_x, pixel_y, cal)

    # Apply chain rule to get angular rates
    daz_dt = daz_dx * dx_dt + daz_dy * dy_dt  # deg/time
    del_dt = del_dx * dx_dt + del_dy * dy_dt  # deg/time

    # Compute total angular velocity accounting for spherical geometry
    # The azimuth component needs to be scaled by cos(elevation)
    elevation_rad = np.deg2rad(elevation)
    omega_az = daz_dt * np.cos(elevation_rad)  # Corrected azimuth rate
    omega_el = del_dt                          # Elevation rate

    # Total angular velocity magnitude
    omega_total = np.sqrt(omega_az**2 + omega_el**2)

    return {
        'azimuth_deg': azimuth,
        'elevation_deg': elevation,
        'daz_dt': daz_dt,           # degrees/time
        'del_dt': del_dt,           # degrees/time
        'omega_az': omega_az,       # degrees/time (corrected)
        'omega_el': omega_el,       # degrees/time
        'omega_total': omega_total, # degrees/time (total angular velocity)
        'omega_total_rad': np.deg2rad(omega_total)  # radians/time
    }


def main():
    parser = argparse.ArgumentParser(
        description='Compute angular velocity from pixel velocity using Chebyshev calibration'
    )
    parser.add_argument(
        'coeff_file',
        help='Path to .npz calibration file'
    )
    parser.add_argument(
        'pixel_x',
        type=float,
        help='Current pixel x coordinate'
    )
    parser.add_argument(
        'pixel_y',
        type=float,
        help='Current pixel y coordinate'
    )
    parser.add_argument(
        'dx_dt',
        type=float,
        help='Pixel velocity in x direction (pixels/time)'
    )
    parser.add_argument(
        'dy_dt',
        type=float,
        help='Pixel velocity in y direction (pixels/time)'
    )
    parser.add_argument(
        '--human-readable',
        action='store_true',
        help='Output in human-readable format'
    )
    parser.add_argument(
        '--radians',
        action='store_true',
        help='Output angular velocities in radians instead of degrees'
    )

    args = parser.parse_args()

    # Compute angular velocity
    result = compute_angular_velocity(
        args.pixel_x, args.pixel_y,
        args.dx_dt, args.dy_dt,
        args.coeff_file
    )

    if args.human_readable:
        print(f"Position:")
        print(f"  Pixel: ({args.pixel_x}, {args.pixel_y})")
        print(f"  Azimuth:   {result['azimuth_deg']:.6f} deg")
        print(f"  Elevation: {result['elevation_deg']:.6f} deg")
        print(f"\nPixel Velocity:")
        print(f"  dx/dt: {args.dx_dt:.6f} pixels/time")
        print(f"  dy/dt: {args.dy_dt:.6f} pixels/time")

        if args.radians:
            print(f"\nAngular Rates (radians/time):")
            print(f"  dAz/dt:  {np.deg2rad(result['daz_dt']):.9f}")
            print(f"  dEl/dt:  {np.deg2rad(result['del_dt']):.9f}")
            print(f"\nCorrected Components (radians/time):")
            print(f"  ω_az:    {np.deg2rad(result['omega_az']):.9f} (azimuth × cos(el))")
            print(f"  ω_el:    {np.deg2rad(result['omega_el']):.9f}")
            print(f"\nTotal Angular Velocity:")
            print(f"  |ω|:     {result['omega_total_rad']:.9f} rad/time")
        else:
            print(f"\nAngular Rates (degrees/time):")
            print(f"  dAz/dt:  {result['daz_dt']:.9f}")
            print(f"  dEl/dt:  {result['del_dt']:.9f}")
            print(f"\nCorrected Components (degrees/time):")
            print(f"  ω_az:    {result['omega_az']:.9f} (azimuth × cos(el))")
            print(f"  ω_el:    {result['omega_el']:.9f}")
            print(f"\nTotal Angular Velocity:")
            print(f"  |ω|:     {result['omega_total']:.9f} deg/time")
    else:
        # Default: compact output
        if args.radians:
            print(f"{result['omega_total_rad']:.9f}")
        else:
            print(f"{result['omega_total']:.9f}")


if __name__ == '__main__':
    main()
