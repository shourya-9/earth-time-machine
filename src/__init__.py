"""Satellite change detection package."""

from .data import fetch_lulc, get_bbox_from_geojson
from .change_detection import (
    compute_change,
    class_area_statistics,
    transition_matrix,
    CLASS_NAMES,
    CLASS_COLORS,
)
from .overlays import (
    fetch_firms_fires,
    firms_period_description,
    check_firms_key_status,
)
from .viz import (
    render_lulc_map,
    render_change_map,
    render_rgb_preview,
    transition_bar_chart,
)

__version__ = "0.1.0"
