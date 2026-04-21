"""
Command-line case study: Rondônia (southwestern Amazon) deforestation 2018 → 2023.

Runs the full pipeline end-to-end and saves outputs to ./outputs/ as PNGs + a
markdown report. Useful for the 'featured case study' section of the README.

Usage:
    python examples/amazon_case_study.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path when running from examples/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import BBox, fetch_lulc
from src.change_detection import compute_change, format_change_report
from src.viz import (
    render_lulc_map, render_change_map, transition_bar_chart, fig_to_png_bytes
)


# Known deforestation frontier in Rondônia, Brazil.
BBOX = BBox(west=-63.2, south=-10.7, east=-62.5, north=-10.1)
BEFORE_YEAR = 2018
AFTER_YEAR = 2023
AOI_NAME = "Rondônia, Brazil (deforestation frontier)"


def main():
    out_dir = ROOT / "outputs" / "amazon"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Fetching {BEFORE_YEAR} land cover for {AOI_NAME}...")
    before = fetch_lulc(BBOX, BEFORE_YEAR).compute()

    print(f"[2/4] Fetching {AFTER_YEAR} land cover...")
    after = fetch_lulc(BBOX, AFTER_YEAR).compute()
    if after.shape != before.shape:
        after = after.interp_like(before, method="nearest").astype("int16")

    print("[3/4] Computing change detection...")
    result = compute_change(before, after, BEFORE_YEAR, AFTER_YEAR)

    print("[4/4] Rendering outputs...")
    before_fig = render_lulc_map(result.before, title=f"Land cover {BEFORE_YEAR}")
    (out_dir / "before.png").write_bytes(fig_to_png_bytes(before_fig))

    after_fig = render_lulc_map(result.after, title=f"Land cover {AFTER_YEAR}")
    (out_dir / "after.png").write_bytes(fig_to_png_bytes(after_fig))

    change_fig = render_change_map(result)
    (out_dir / "change.png").write_bytes(fig_to_png_bytes(change_fig))

    bar_fig = transition_bar_chart(result, n=8)
    (out_dir / "transitions.png").write_bytes(fig_to_png_bytes(bar_fig))

    report = format_change_report(result, aoi_name=AOI_NAME)
    (out_dir / "report.md").write_text(report)

    print(f"\n✓ Done. Outputs in: {out_dir}")
    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)


if __name__ == "__main__":
    main()
