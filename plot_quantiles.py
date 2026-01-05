#!/usr/bin/env python3
"""
Plot quantile columns from a CSV file.

CSV format: filename, 0%, 25%, 50%, 75%, 100%
Plots columns 3, 4, 5 (25%, 50%, 75%) vs line number.
Prints min/max for columns 2, 6 (0%, 100%).
"""

import sys
import csv
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def main():
    if len(sys.argv) != 2:
        print("Usage: python plot_quantiles.py <csv_file>", file=sys.stderr)
        sys.exit(1)

    csv_file = sys.argv[1]

    # Read CSV data
    line_numbers = []
    col2_values = []  # 0% quantile
    col3_values = []  # 25% quantile
    col4_values = []  # 50% quantile
    col5_values = []  # 75% quantile
    col6_values = []  # 100% quantile

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader, start=1):
            line_numbers.append(i)
            col2_values.append(float(row[1]))
            col3_values.append(float(row[2]))
            col4_values.append(float(row[3]))
            col5_values.append(float(row[4]))
            col6_values.append(float(row[5]))

    # Print min/max for columns 2 and 6
    print(f"Column 2 (0% quantile): min={min(col2_values)}, max={max(col2_values)}")
    print(f"Column 6 (100% quantile): min={min(col6_values)}, max={max(col6_values)}")

    # Create plot
    plt.figure(figsize=(12, 8))
    plt.plot(line_numbers, col3_values, label='25% quantile', alpha=0.7)
    plt.plot(line_numbers, col4_values, label='50% quantile', alpha=0.7)
    plt.plot(line_numbers, col5_values, label='75% quantile', alpha=0.7)

    plt.xlabel('Line Number')
    plt.ylabel('Value')
    plt.title('Quantiles vs Line Number')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save to file
    output_file = csv_file.replace('.csv', '_plot.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")


if __name__ == '__main__':
    main()
