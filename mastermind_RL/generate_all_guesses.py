#!/usr/bin/env python3
"""Generate all possible Mastermind guesses and save to CSV."""

import csv
from pathlib import Path

from RL import ALL_CODES, COLORS, POSITIONS


def main() -> None:
    total_colors = len(COLORS)
    header = [f"pos_{i}" for i in range(POSITIONS)]

    for k in range(1, total_colors + 1):
        allowed_colors = set(range(k))
        codes = [code for code in ALL_CODES if all(c in allowed_colors for c in code)]
        output_path = Path(f"all_codes_{k}.csv")

        with output_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerows(codes)

        print(
            f"Wrote {len(codes)} guesses ({k} colors, {POSITIONS} positions) to {output_path}"
        )


if __name__ == "__main__":
    main()
