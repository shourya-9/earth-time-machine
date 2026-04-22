"""
Change detection logic.

Given two land-cover maps (before / after) as xarray DataArrays with integer
class codes, compute:
  - a pixel-wise change mask
  - per-class area before and after
  - a full transition matrix (from-class x to-class -> hectares)
  - the top N most significant transitions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr


# Impact Observatory / Esri 9-class scheme used by io-lulc-annual-v02
CLASS_NAMES: Dict[int, str] = {
    0: "No Data",
    1: "Water",
    2: "Trees",
    4: "Flooded Vegetation",
    5: "Crops",
    7: "Built Area",
    8: "Bare Ground",
    9: "Snow/Ice",
    10: "Clouds",
    11: "Rangeland",
}

# Colors roughly matching the standard IO LULC color table.
CLASS_COLORS: Dict[int, str] = {
    0: "#000000",
    1: "#1a5bab",
    2: "#358221",
    4: "#87d19e",
    5: "#ffdb5c",
    7: "#ed022a",
    8: "#ede9e4",
    9: "#f2faff",
    10: "#c8c8c8",
    11: "#c6ad8d",
}

# Transitions that are "interesting" for deforestation / urbanization stories.
NOTABLE_TRANSITIONS: List[Tuple[int, int, str]] = [
    (2, 5, "Forest → Cropland (deforestation)"),
    (2, 7, "Forest → Built Area (urbanization into forest)"),
    (2, 8, "Forest → Bare Ground (clearing)"),
    (2, 11, "Forest → Rangeland (forest degradation)"),
    (11, 7, "Rangeland → Built Area (urban sprawl)"),
    (5, 7, "Cropland → Built Area (urban sprawl)"),
    (11, 5, "Rangeland → Cropland (agricultural expansion)"),
    (8, 7, "Bare Ground → Built Area (new development)"),
    (1, 7, "Water → Built Area (land reclamation)"),
]


@dataclass
class ChangeResult:
    """Container for change-detection outputs."""
    before: xr.DataArray
    after: xr.DataArray
    change_mask: xr.DataArray          # boolean: True where class changed
    transition_code: xr.DataArray      # int: (before * 100 + after) where changed
    # A human-readable label for the "before" and "after" periods. For the
    # IO-LULC data source these are simple year ints (e.g. 2018, 2023).
    # For Dynamic World they are period strings (e.g. "Apr 2024", "2026-04-01 → 2026-04-30").
    before_year: Union[int, str]
    after_year: Union[int, str]
    pixel_area_ha: float               # hectares per pixel (approx)
    before_area_ha: Dict[int, float]
    after_area_ha: Dict[int, float]
    transition_ha: Dict[Tuple[int, int], float]


def _get_xy_coords(da: xr.DataArray):
    """Return (x_values, y_values) regardless of whether coords are named
    x/y or longitude/latitude."""
    if "x" in da.coords:
        xs = da["x"].values
    elif "longitude" in da.coords:
        xs = da["longitude"].values
    elif "lon" in da.coords:
        xs = da["lon"].values
    else:
        raise KeyError(f"No x/longitude coord found in {list(da.coords)}")

    if "y" in da.coords:
        ys = da["y"].values
    elif "latitude" in da.coords:
        ys = da["latitude"].values
    elif "lat" in da.coords:
        ys = da["lat"].values
    else:
        raise KeyError(f"No y/latitude coord found in {list(da.coords)}")
    return xs, ys


def _pixel_area_ha(da: xr.DataArray) -> float:
    """Estimate hectares per pixel in EPSG:4326 using the mid-latitude."""
    lons, lats = _get_xy_coords(da)

    dlat_deg = float(np.abs(np.mean(np.diff(lats))))
    dlon_deg = float(np.abs(np.mean(np.diff(lons))))
    mid_lat = float(np.mean(lats))

    lat_m = dlat_deg * 111_320.0
    lon_m = dlon_deg * 111_320.0 * np.cos(np.deg2rad(mid_lat))
    m2 = lat_m * lon_m
    return m2 / 10_000.0  # hectares


def class_area_statistics(lulc: xr.DataArray) -> Dict[int, float]:
    """Return hectares per class for a single land-cover map."""
    pixel_ha = _pixel_area_ha(lulc)
    values, counts = np.unique(lulc.values, return_counts=True)
    return {int(v): float(c) * pixel_ha for v, c in zip(values, counts)}


def transition_matrix(before: xr.DataArray, after: xr.DataArray) -> pd.DataFrame:
    """
    Build a transition matrix (pandas DataFrame) of hectares moving from each
    `before` class to each `after` class.
    """
    pixel_ha = _pixel_area_ha(before)

    b = before.values.ravel().astype(np.int16)
    a = after.values.ravel().astype(np.int16)

    # Pair into a single integer key for fast counting.
    key = b.astype(np.int32) * 100 + a.astype(np.int32)
    uniq, counts = np.unique(key, return_counts=True)

    rows = []
    for k, c in zip(uniq, counts):
        from_c = int(k // 100)
        to_c = int(k % 100)
        rows.append(
            {
                "from_class": from_c,
                "to_class": to_c,
                "from_name": CLASS_NAMES.get(from_c, f"Class {from_c}"),
                "to_name": CLASS_NAMES.get(to_c, f"Class {to_c}"),
                "hectares": float(c) * pixel_ha,
                "changed": from_c != to_c,
            }
        )
    return pd.DataFrame(rows)


def compute_change(
    before: xr.DataArray,
    after: xr.DataArray,
    before_year: Union[int, str],
    after_year: Union[int, str],
) -> ChangeResult:
    """
    Compute all change statistics between two land-cover rasters.

    The two arrays must already be aligned to the same grid (same coords).
    """
    if before.shape != after.shape:
        # Try to align to the common grid.
        after = after.interp_like(before, method="nearest")

    # Ensure integer type for comparison.
    b = before.astype("int16")
    a = after.astype("int16")

    change_mask = (b != a) & (b != 0) & (a != 0)
    transition_code = xr.where(change_mask, b * 100 + a, 0)

    pixel_ha = _pixel_area_ha(b)
    before_area = class_area_statistics(b)
    after_area = class_area_statistics(a)

    # Build the transition dictionary.
    b_flat = b.values.ravel().astype(np.int32)
    a_flat = a.values.ravel().astype(np.int32)
    key = b_flat * 100 + a_flat
    uniq, counts = np.unique(key, return_counts=True)

    transition_ha: Dict[Tuple[int, int], float] = {}
    for k, c in zip(uniq, counts):
        from_c = int(k // 100)
        to_c = int(k % 100)
        if from_c == 0 or to_c == 0:
            continue
        transition_ha[(from_c, to_c)] = float(c) * pixel_ha

    return ChangeResult(
        before=b,
        after=a,
        change_mask=change_mask,
        transition_code=transition_code,
        before_year=before_year,
        after_year=after_year,
        pixel_area_ha=pixel_ha,
        before_area_ha=before_area,
        after_area_ha=after_area,
        transition_ha=transition_ha,
    )


def top_transitions(
    result: ChangeResult,
    n: int = 5,
    exclude_unchanged: bool = True,
) -> List[Tuple[int, int, float]]:
    """Return the N largest class-to-class transitions by hectares."""
    items = list(result.transition_ha.items())
    if exclude_unchanged:
        items = [(k, v) for k, v in items if k[0] != k[1]]
    items.sort(key=lambda kv: kv[1], reverse=True)
    return [(k[0], k[1], v) for k, v in items[:n]]


def notable_transitions_summary(result: ChangeResult) -> List[Dict]:
    """
    For each pre-defined notable transition (e.g. forest->cropland),
    return the hectares observed in the AOI. Useful for the report.
    """
    out = []
    for from_c, to_c, label in NOTABLE_TRANSITIONS:
        ha = result.transition_ha.get((from_c, to_c), 0.0)
        out.append(
            {
                "label": label,
                "from_class": from_c,
                "to_class": to_c,
                "hectares": ha,
            }
        )
    return out


def format_change_report(result: ChangeResult, aoi_name: str = "the area of interest") -> str:
    """Generate a markdown report summarizing the change."""
    total_pixels = int(np.prod(result.before.shape))
    changed_pixels = int(result.change_mask.sum().item())
    pct_changed = 100.0 * changed_pixels / max(total_pixels, 1)
    total_ha_changed = changed_pixels * result.pixel_area_ha

    lines = [
        f"# Change Detection Report",
        "",
        f"**Area:** {aoi_name}  ",
        f"**Period:** {result.before_year} → {result.after_year}  ",
        f"**Pixel size:** ~{result.pixel_area_ha:.3f} ha/pixel  ",
        "",
        f"## Summary",
        f"- Total area analyzed: **{total_pixels * result.pixel_area_ha:,.0f} ha**",
        f"- Pixels with class change: **{changed_pixels:,}** ({pct_changed:.2f}%)",
        f"- Total area changed: **{total_ha_changed:,.0f} ha**",
        "",
        f"## Top transitions ({result.before_year} → {result.after_year})",
        "",
        "| From | To | Hectares |",
        "|------|-----|----------|",
    ]

    for from_c, to_c, ha in top_transitions(result, n=8):
        lines.append(
            f"| {CLASS_NAMES.get(from_c, from_c)} | "
            f"{CLASS_NAMES.get(to_c, to_c)} | {ha:,.0f} |"
        )

    lines += [
        "",
        f"## Notable transitions",
        "",
        "| Transition | Hectares |",
        "|------------|----------|",
    ]
    for entry in notable_transitions_summary(result):
        if entry["hectares"] > 0:
            lines.append(f"| {entry['label']} | {entry['hectares']:,.0f} |")

    lines += [
        "",
        f"## Net class area change (hectares)",
        "",
        "| Class | Before | After | Δ |",
        "|-------|--------|-------|---|",
    ]
    all_classes = sorted(set(result.before_area_ha) | set(result.after_area_ha))
    for c in all_classes:
        if c == 0 or c not in CLASS_NAMES:
            continue
        b = result.before_area_ha.get(c, 0.0)
        a = result.after_area_ha.get(c, 0.0)
        delta = a - b
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"| {CLASS_NAMES[c]} | {b:,.0f} | {a:,.0f} | {sign}{delta:,.0f} |"
        )

    return "\n".join(lines)
