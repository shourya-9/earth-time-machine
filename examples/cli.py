"""
Generic CLI for running the change-detection pipeline on any bounding box.

Usage:
    python examples/cli.py \\
        --bbox -63.2,-10.7,-62.5,-10.1 \\
        --before 2018 --after 2023 \\
        --name "Rondônia" \\
        --out outputs/rondonia

    python examples/cli.py \\
        --bbox 55.10,24.90,55.45,25.20 \\
        --before 2017 --after 2023 \\
        --name "Dubai" \\
        --out outputs/dubai
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import BBox, fetch_lulc
from src.change_detection import compute_change, format_change_report
from src.viz import (
    render_lulc_map, render_change_map, transition_bar_chart, fig_to_png_bytes
)


def parse_bbox(s: str) -> BBox:
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "bbox must be 'west,south,east,north' (4 numbers)"
        )
    return BBox(*parts)


def main():
    p = argparse.ArgumentParser(description="Satellite change detection CLI")
    p.add_argument("--bbox", type=parse_bbox, required=True,
                   help="Bounding box as 'west,south,east,north'")
    p.add_argument("--before", type=int, required=True, help="Before year (2017-2023)")
    p.add_argument("--after", type=int, required=True, help="After year (2017-2023)")
    p.add_argument("--name", type=str, default="AOI", help="Friendly AOI name for report")
    p.add_argument("--out", type=Path, default=Path("outputs"), help="Output directory")

    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Fetching {args.before} land cover for {args.name}...")
    before = fetch_lulc(args.bbox, args.before).compute()

    print(f"[2/4] Fetching {args.after} land cover...")
    after = fetch_lulc(args.bbox, args.after).compute()
    if after.shape != before.shape:
        after = after.interp_like(before, method="nearest").astype("int16")

    print("[3/4] Computing change detection...")
    result = compute_change(before, after, args.before, args.after)

    print("[4/4] Rendering outputs...")
    (args.out / "before.png").write_bytes(
        fig_to_png_bytes(render_lulc_map(result.before, title=f"{args.name} — {args.before}"))
    )
    (args.out / "after.png").write_bytes(
        fig_to_png_bytes(render_lulc_map(result.after, title=f"{args.name} — {args.after}"))
    )
    (args.out / "change.png").write_bytes(fig_to_png_bytes(render_change_map(result)))
    (args.out / "transitions.png").write_bytes(
        fig_to_png_bytes(transition_bar_chart(result, n=8))
    )
    report = format_change_report(result, aoi_name=args.name)
    (args.out / "report.md").write_text(report)

    print(f"\n✓ Done. Outputs in: {args.out}")
    print(f"  - before.png, after.png, change.png, transitions.png")
    print(f"  - report.md")


if __name__ == "__main__":
    main()
