#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from palpation_sim.curve_plots import CENTER_SAMPLE_NAME, draw_run_curves, draw_run_group_curves


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw F-z curve PNGs for assembled center-sphere Newton runs.")
    parser.add_argument("run", type=Path, help="An assembled run directory or a parent directory of assembled runs.")
    parser.add_argument("--max-columns", type=int, default=17, help="Maximum center-row curves to draw per run.")
    parser.add_argument("--output-name", type=str, default=None, help="Override the output PNG filename.")
    parser.add_argument("--no-child-plots", action="store_true", help="For a parent run, only write the aggregate PNG.")
    args = parser.parse_args()

    if not args.run.exists():
        raise SystemExit(f"run path not found: {args.run}")

    if (args.run / CENTER_SAMPLE_NAME).exists():
        result = draw_run_curves(
            args.run,
            output_name=args.output_name or "fz_curves.png",
            max_columns=args.max_columns,
        )
    else:
        result = draw_run_group_curves(
            args.run,
            output_name=args.output_name or "all_run_fz_curves.png",
            max_columns=args.max_columns,
            draw_child_plots=not args.no_child_plots,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
