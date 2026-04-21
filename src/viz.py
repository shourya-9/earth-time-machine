"""
Visualization helpers: land-cover colormaps, change maps, charts, folium layers.
"""

from __future__ import annotations

from io import BytesIO
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import xarray as xr
from PIL import Image

from .change_detection import (
    CLASS_NAMES,
    CLASS_COLORS,
    ChangeResult,
    top_transitions,
    _get_xy_coords,
)


def _extent(da) -> list:
    """Matplotlib imshow extent (xmin, xmax, ymin, ymax) for any xarray
    object that has x/y or longitude/latitude coords."""
    xs, ys = _get_xy_coords(da)
    return [float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())]


def _build_lulc_cmap() -> Tuple[ListedColormap, BoundaryNorm]:
    """Matplotlib colormap keyed to IO LULC class codes (0..11)."""
    codes = sorted(CLASS_NAMES.keys())
    max_code = max(codes)
    colors = ["#000000"] * (max_code + 1)
    for code, color in CLASS_COLORS.items():
        colors[code] = color
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, max_code + 1.5, 1), cmap.N)
    return cmap, norm


def render_lulc_map(
    lulc: xr.DataArray,
    title: str = "Land cover",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Render a land-cover DataArray with the standard IO LULC colors."""
    cmap, norm = _build_lulc_cmap()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    ax.imshow(
        lulc.values,
        cmap=cmap,
        norm=norm,
        extent=_extent(lulc),
        origin="upper",
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Legend showing only classes present in the image.
    present = sorted(np.unique(lulc.values).tolist())
    handles = [
        Patch(facecolor=CLASS_COLORS[c], edgecolor="black", label=CLASS_NAMES[c])
        for c in present
        if c in CLASS_NAMES and c != 0
    ]
    if handles:
        ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)

    return fig


def render_change_map(
    result: ChangeResult,
    title: Optional[str] = None,
    highlight_forest_loss: bool = True,
) -> plt.Figure:
    """
    Render a change map:
      - base layer: "after" land cover (muted)
      - overlay:    red where class changed

    If highlight_forest_loss, forest→anything transitions are shown in bright red;
    other changes in orange.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Base LULC (after) faded to provide spatial context.
    cmap, norm = _build_lulc_cmap()
    extent = _extent(result.after)
    ax.imshow(
        result.after.values,
        cmap=cmap,
        norm=norm,
        extent=extent,
        origin="upper",
        interpolation="nearest",
        alpha=0.35,
    )

    # Build overlay: 0=no change, 1=other change, 2=forest loss
    overlay = np.zeros_like(result.before.values, dtype=np.uint8)
    changed = result.change_mask.values
    overlay[changed] = 1
    if highlight_forest_loss:
        forest_loss = (result.before.values == 2) & (result.after.values != 2) & changed
        overlay[forest_loss] = 2

    overlay_rgba = np.zeros((*overlay.shape, 4), dtype=np.float32)
    overlay_rgba[overlay == 1] = [1.0, 0.55, 0.0, 0.7]    # orange
    overlay_rgba[overlay == 2] = [0.85, 0.05, 0.05, 0.9]  # red

    ax.imshow(
        overlay_rgba,
        extent=extent,
        origin="upper",
        interpolation="nearest",
    )

    if title is None:
        title = f"Change {result.before_year} → {result.after_year}"
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    legend = [
        Patch(facecolor=(0.85, 0.05, 0.05), label="Forest loss"),
        Patch(facecolor=(1.0, 0.55, 0.0), label="Other class change"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=9, framealpha=0.9)

    return fig


def render_rgb_preview(rgb: xr.Dataset, title: str = "Sentinel-2 RGB") -> plt.Figure:
    """Render an RGB preview from a Sentinel-2 dataset (bands B04, B03, B02)."""
    r = rgb["B04"].values.astype(np.float32)
    g = rgb["B03"].values.astype(np.float32)
    b = rgb["B02"].values.astype(np.float32)

    stack = np.stack([r, g, b], axis=-1)

    # Percentile stretch for readable display.
    lo = np.nanpercentile(stack, 2)
    hi = np.nanpercentile(stack, 98)
    stack = np.clip((stack - lo) / max(hi - lo, 1e-6), 0, 1)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(
        stack,
        extent=_extent(rgb),
        origin="upper",
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    return fig


def transition_bar_chart(result: ChangeResult, n: int = 8) -> plt.Figure:
    """Horizontal bar chart of the top N class-to-class transitions."""
    top = top_transitions(result, n=n)
    if not top:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No changes detected", ha="center", va="center")
        ax.axis("off")
        return fig

    labels = [
        f"{CLASS_NAMES.get(a, a)} → {CLASS_NAMES.get(b, b)}"
        for a, b, _ in top
    ]
    values = [ha for _, _, ha in top]

    fig, ax = plt.subplots(figsize=(8, 0.5 * n + 1.5))
    ax.barh(labels, values, color="#d62728")
    ax.invert_yaxis()
    ax.set_xlabel("Hectares")
    ax.set_title(f"Top transitions {result.before_year} → {result.after_year}")
    for i, v in enumerate(values):
        ax.text(v, i, f"  {v:,.0f}", va="center", fontsize=9)
    plt.tight_layout()
    return fig


def fig_to_png_bytes(fig: plt.Figure, dpi: int = 120) -> bytes:
    """Convert a matplotlib figure to PNG bytes (for Streamlit / downloads)."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def lulc_to_rgba_image(lulc: xr.DataArray) -> Image.Image:
    """Convert a land-cover array directly to a PIL RGBA image (for folium overlay)."""
    arr = lulc.values
    h, w = arr.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for code, hex_color in CLASS_COLORS.items():
        mask = arr == code
        if not mask.any():
            continue
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        rgba[mask] = [r, g, b, 200]
    return Image.fromarray(rgba, mode="RGBA")


def change_to_rgba_image(result: ChangeResult) -> Image.Image:
    """Convert a change mask to an RGBA image for folium overlay."""
    arr = result.change_mask.values
    h, w = arr.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    forest_loss = (result.before.values == 2) & (result.after.values != 2) & arr
    other_change = arr & ~forest_loss

    rgba[other_change] = [255, 140, 0, 180]
    rgba[forest_loss] = [220, 20, 20, 220]
    return Image.fromarray(rgba, mode="RGBA")
