"""
Satellite Change Detection — Streamlit app.

Run with:
    streamlit run app.py

Features:
- Pick an area of interest by drawing a rectangle on a world map, or choose a preset.
- Choose two years (2017-2023).
- Analyze: fetches Impact Observatory land cover for both years, computes the
  change detection, renders maps, statistics, and a report.
- Optional FIRMS fire overlay if FIRMS_MAP_KEY is set in environment.
"""

from __future__ import annotations

import os
import re
import traceback
from datetime import date, timedelta
from io import BytesIO

import folium
import numpy as np
import pandas as pd
import streamlit as st
from folium.plugins import Draw
from streamlit_folium import st_folium

from src.data import BBox, fetch_lulc, fetch_s2_rgb_preview, available_years
from src.change_detection import (
    compute_change,
    format_change_report,
    top_transitions,
    notable_transitions_summary,
    CLASS_NAMES,
    CLASS_COLORS,
    _get_xy_coords,
)
from src.viz import (
    render_lulc_map,
    render_change_map,
    render_rgb_preview,
    transition_bar_chart,
    fig_to_png_bytes,
)
from src.overlays import (
    fetch_firms_fires,
    firms_period_description,
    check_firms_key_status,
)
from src.dynamic_world import (
    fetch_dynamic_world_mode,
    dynamic_world_period_description,
    check_earth_engine_auth,
)


st.set_page_config(
    page_title="Satellite Change Detection",
    page_icon="🛰️",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Deployment bootstrap: load secrets from st.secrets → os.environ
# ---------------------------------------------------------------------------
# On Streamlit Community Cloud there's no shell env, so anything we'd normally
# `export` locally must come from st.secrets. We convert those values into
# environment variables here so the rest of the code (which reads env vars)
# doesn't need to know whether it's running locally or in the cloud.
#
# Locally, `.streamlit/secrets.toml` is optional — if it's missing, this block
# is a no-op and your terminal `export` workflow continues to work.

def _bootstrap_secrets() -> None:
    import json as _json
    import tempfile as _tempfile

    def _safe_get(key):
        """Return secrets[key] or None, swallowing 'no secrets file' errors."""
        try:
            return st.secrets.get(key) if hasattr(st.secrets, "get") else st.secrets[key]
        except Exception:
            return None

    # Plain string secrets → env vars.
    for key in ("EARTHENGINE_PROJECT", "FIRMS_MAP_KEY"):
        val = _safe_get(key)
        if val and not os.environ.get(key):
            os.environ[key] = str(val)

    # Service account JSON → materialize to a tempfile, point
    # GOOGLE_APPLICATION_CREDENTIALS at it. Earth Engine's
    # ServiceAccountCredentials constructor requires a file path.
    sa = _safe_get("GCP_SERVICE_ACCOUNT_JSON")
    if sa and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        # st.secrets returns a Mapping-like AttrDict; convert to plain dict.
        sa_dict = dict(sa) if not isinstance(sa, dict) else sa
        fd, path = _tempfile.mkstemp(prefix="ee-sa-", suffix=".json")
        with os.fdopen(fd, "w") as f:
            _json.dump(sa_dict, f)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path


_bootstrap_secrets()


# ---------------------------------------------------------------------------
# Preset case studies
# ---------------------------------------------------------------------------

PRESETS = {
    "Rondônia deforestation (Brazil)": {
        "bbox": BBox(-63.2, -10.7, -62.5, -10.1),
        "before_year": 2018,
        "after_year": 2023,
        "story": (
            "A well-known deforestation frontier in the southwestern Amazon. "
            "Expect a strong Forest → Cropland / Rangeland signal."
        ),
    },
    "Dubai urban growth": {
        "bbox": BBox(55.10, 24.90, 55.45, 25.20),
        "before_year": 2017,
        "after_year": 2023,
        "story": (
            "Rapid coastal urbanization. Expect Bare Ground / Water → Built Area."
        ),
    },
    "Bengaluru sprawl (India)": {
        "bbox": BBox(77.45, 12.80, 77.80, 13.15),
        "before_year": 2017,
        "after_year": 2023,
        "story": (
            "Sprawl on the southern periphery. Expect Crops / Rangeland → Built Area."
        ),
    },
    "California Camp Fire area": {
        "bbox": BBox(-121.75, 39.70, -121.45, 39.95),
        "before_year": 2018,
        "after_year": 2022,
        "story": (
            "The 2018 Camp Fire burn scar. Expect Forest → Bare Ground / Rangeland."
        ),
    },
    "Borneo peatland (Indonesia)": {
        "bbox": BBox(113.30, -2.80, 113.70, -2.40),
        "before_year": 2017,
        "after_year": 2023,
        "story": (
            "Central Kalimantan peat-swamp conversion. Expect Forest / Flooded "
            "Vegetation → Crops."
        ),
    },
}


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _init_state():
    # Default DW windows: 30 days ending ~2 years ago, and 30 days ending today.
    today = date.today()
    dw_after_end_default = today
    dw_after_start_default = today - timedelta(days=29)
    dw_before_end_default = today - timedelta(days=365 * 2)
    dw_before_start_default = dw_before_end_default - timedelta(days=29)

    defaults = {
        "bbox": None,
        "data_source": "iolulc",    # "iolulc" or "dw"
        "before_year": 2018,
        "after_year": 2023,
        "dw_before_start": dw_before_start_default,
        "dw_before_end": dw_before_end_default,
        "dw_after_start": dw_after_start_default,
        "dw_after_end": dw_after_end_default,
        "ee_project": os.environ.get("EARTHENGINE_PROJECT", ""),
        "result": None,
        "aoi_name": "custom AOI",
        "preview_before": None,
        "preview_after": None,
        "fires_df": None,
        "fires_error": None,
        "fires_requested": False,   # did the last Analyze include the fires fetch?
        "last_error": None,
        "_had_drawing": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ---------------------------------------------------------------------------
# Sidebar: preset + controls
# ---------------------------------------------------------------------------

st.sidebar.title("🛰️ Controls")

st.sidebar.markdown("### Quick-start presets")
preset_choice = st.sidebar.selectbox(
    "Featured case study",
    ["(choose...)"] + list(PRESETS.keys()),
)
if preset_choice != "(choose...)":
    if st.sidebar.button(f"Load: {preset_choice}"):
        p = PRESETS[preset_choice]
        st.session_state.bbox = p["bbox"]
        st.session_state.before_year = p["before_year"]
        st.session_state.after_year = p["after_year"]
        st.session_state.aoi_name = preset_choice
        st.session_state.result = None
        st.session_state.fires_df = None
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### Data source")

DATA_SOURCE_OPTIONS = {
    "iolulc": "IO-LULC (annual, 2017-2024)",
    "dw": "Google Dynamic World (near-real-time, any 2015-today window)",
}
_data_source_keys = list(DATA_SOURCE_OPTIONS.keys())
_current_ds_idx = _data_source_keys.index(
    st.session_state.data_source
    if st.session_state.data_source in _data_source_keys
    else "iolulc"
)
data_source = st.sidebar.radio(
    "Choose a land-cover product",
    _data_source_keys,
    format_func=lambda k: DATA_SOURCE_OPTIONS[k],
    index=_current_ds_idx,
    help=(
        "IO-LULC: Impact Observatory annual 10m land cover via Planetary Computer "
        "— no auth, but ~12-month lag and once-per-year snapshots.\n\n"
        "Dynamic World: Google's 10m NRT land cover via Earth Engine — updated "
        "every 2-5 days, goes back to 2015. Requires a free Earth Engine "
        "account (one-time `earthengine authenticate`)."
    ),
)
st.session_state.data_source = data_source

# Clear stale results if the user switches data sources.
if data_source != (st.session_state.get("_last_data_source") or data_source):
    st.session_state.result = None
st.session_state._last_data_source = data_source


if data_source == "iolulc":
    st.sidebar.markdown("### Years")
    years = available_years()

    # Clamp stored preferences in case they fall outside the current range.
    if st.session_state.before_year not in years:
        st.session_state.before_year = years[0]
    if st.session_state.after_year not in years:
        st.session_state.after_year = years[-1]

    before_year = st.sidebar.selectbox(
        "Before year",
        years,
        index=years.index(st.session_state.before_year),
    )
    after_year = st.sidebar.selectbox(
        "After year",
        years,
        index=years.index(st.session_state.after_year),
    )
    st.session_state.before_year = before_year
    st.session_state.after_year = after_year

    st.sidebar.caption(
        "IO-LULC is an **annual** product (~12-month lag). "
        "Latest year may not be published for every region."
    )

    if before_year >= after_year:
        st.sidebar.warning("'After year' should be later than 'before year'.")

else:
    # Dynamic World mode: two date-window pickers.
    st.sidebar.markdown("### Date windows")
    st.sidebar.caption(
        "Dynamic World is updated every 2-5 days. Pick a **before** window and "
        "an **after** window — the modal (most common) class per pixel across "
        "each window is used. 30-60 days is a good default."
    )

    dw_before_start = st.sidebar.date_input(
        "Before: start",
        value=st.session_state.dw_before_start,
        min_value=date(2015, 6, 23),   # DW coverage starts mid-2015
        max_value=date.today(),
        key="dw_before_start_input",
    )
    dw_before_end = st.sidebar.date_input(
        "Before: end",
        value=st.session_state.dw_before_end,
        min_value=date(2015, 6, 23),
        max_value=date.today(),
        key="dw_before_end_input",
    )
    dw_after_start = st.sidebar.date_input(
        "After: start",
        value=st.session_state.dw_after_start,
        min_value=date(2015, 6, 23),
        max_value=date.today(),
        key="dw_after_start_input",
    )
    dw_after_end = st.sidebar.date_input(
        "After: end",
        value=st.session_state.dw_after_end,
        min_value=date(2015, 6, 23),
        max_value=date.today(),
        key="dw_after_end_input",
    )

    st.session_state.dw_before_start = dw_before_start
    st.session_state.dw_before_end = dw_before_end
    st.session_state.dw_after_start = dw_after_start
    st.session_state.dw_after_end = dw_after_end

    # Validation
    if dw_before_start > dw_before_end:
        st.sidebar.warning("'Before: start' must be on/before 'Before: end'.")
    if dw_after_start > dw_after_end:
        st.sidebar.warning("'After: start' must be on/before 'After: end'.")
    if dw_before_end > dw_after_start:
        st.sidebar.warning(
            "The 'before' window should end before the 'after' window starts."
        )

    st.sidebar.markdown("#### Earth Engine project")
    ee_project = st.sidebar.text_input(
        "Google Cloud project ID",
        value=st.session_state.ee_project,
        placeholder="e.g. my-ee-project-123456",
        help=(
            "Since 2023, Earth Engine requires a Google Cloud project ID.\n\n"
            "Get one (free) at https://console.cloud.google.com/, then enable "
            "the Earth Engine API:\n"
            "https://console.cloud.google.com/apis/library/earthengine.googleapis.com\n\n"
            "Alternatives: run `earthengine set_project YOUR_ID` once in a "
            "terminal, or export EARTHENGINE_PROJECT=YOUR_ID."
        ),
    )
    st.session_state.ee_project = ee_project

    # Auth status indicator
    if st.sidebar.button("Check Earth Engine auth", width="stretch"):
        with st.spinner("Checking Earth Engine..."):
            ee_status = check_earth_engine_auth(project=ee_project or None)
        if ee_status["ok"]:
            st.sidebar.success(ee_status["message"])
        else:
            st.sidebar.error(
                f"{ee_status['message']}\n\n{ee_status['how_to_fix']}"
            )

st.sidebar.markdown("### Optional overlays")
show_rgb = st.sidebar.checkbox(
    "🛰️ Fetch Sentinel-2 true-color imagery",
    value=False,
    help=(
        "Adds a real-Earth satellite preview (Red/Green/Blue bands) for each "
        "year in the Maps tab. Takes ~30-60s extra per year."
    ),
)
show_fires = st.sidebar.checkbox(
    "🔥 Fetch NASA FIRMS active fires (last 60 days)",
    value=False,
    help=(
        "Shows current near-real-time fire detections in the AOI (last ~60 "
        "days). Needs a free FIRMS MAP_KEY (paste below or set FIRMS_MAP_KEY). "
        "Historical FIRMS archive is not accessible via this API endpoint."
    ),
)

# Resolve the FIRMS key from three possible sources, in order:
#   1. Sidebar text input (most convenient)
#   2. Streamlit secrets (if .streamlit/secrets.toml exists)
#   3. Environment variable FIRMS_MAP_KEY
_firms_key_env = os.environ.get("FIRMS_MAP_KEY", "")
_firms_key_secrets = ""
try:
    if hasattr(st, "secrets") and "FIRMS_MAP_KEY" in st.secrets:
        _firms_key_secrets = str(st.secrets["FIRMS_MAP_KEY"])
except Exception:
    pass
_firms_key_default = _firms_key_env or _firms_key_secrets

firms_key_input = ""
if show_fires:
    firms_key_input = st.sidebar.text_input(
        "FIRMS MAP_KEY",
        value=_firms_key_default,
        type="password",
        help=(
            "Paste your FIRMS MAP_KEY here. Get one free at "
            "https://firms.modaps.eosdis.nasa.gov/api/area/ (instant)."
        ),
    )
    if not firms_key_input:
        st.sidebar.warning("Fires overlay requires a FIRMS MAP_KEY.")
    else:
        if st.sidebar.button("Check FIRMS key status", width="stretch"):
            with st.spinner("Checking FIRMS MAP_KEY..."):
                status = check_firms_key_status(firms_key_input)
            if status["ok"]:
                raw = status.get("raw") or {}
                if isinstance(raw, dict) and "current_transactions" in raw:
                    st.sidebar.success(
                        f"Key is active — "
                        f"{raw.get('current_transactions')}/"
                        f"{raw.get('transaction_limit')} transactions used "
                        f"in the last {raw.get('transaction_interval_minutes', '?')} min."
                    )
                else:
                    st.sidebar.success(status["message"])
            else:
                st.sidebar.error(status["message"])

st.sidebar.markdown("---")
st.sidebar.caption(
    "Data sources: Impact Observatory / Esri 10m LULC (Planetary Computer), "
    "Google Dynamic World 10m NRT (Earth Engine), Sentinel-2 L2A for RGB "
    "previews, NASA FIRMS for fires. Basemap: Esri World Imagery."
)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🛰️ Satellite Change Detection")
st.markdown(
    "Detect land-cover change anywhere on Earth between two years using 10m "
    "Sentinel-derived land cover maps. Pick a preset, or draw your own AOI."
)


# ---------------------------------------------------------------------------
# Map: AOI picker
# ---------------------------------------------------------------------------

st.markdown("### 1. Pick an area of interest")

# Decide the initial map center / zoom.
if st.session_state.bbox is not None:
    b = st.session_state.bbox
    center = [(b.south + b.north) / 2, (b.west + b.east) / 2]
    zoom = 9
else:
    center = [0, 0]
    zoom = 2

m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True)

# Base layers: Satellite (default) + Streets. Users can toggle in the top-right.
folium.TileLayer(
    tiles=(
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}"
    ),
    attr="Esri, Maxar, Earthstar Geographics, and the GIS User Community",
    name="Satellite (Esri World Imagery)",
    overlay=False,
    control=True,
).add_to(m)

folium.TileLayer(
    tiles=(
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
    ),
    attr="Esri",
    name="Labels (on top of satellite)",
    overlay=True,
    control=True,
    opacity=0.9,
).add_to(m)

folium.TileLayer(
    tiles="OpenStreetMap",
    name="Streets (OpenStreetMap)",
    overlay=False,
    control=True,
).add_to(m)

# Draw tool configured for rectangles only. Edit and delete buttons are
# disabled (they're flaky through streamlit-folium) — use the "Clear AOI"
# button below the map instead.
Draw(
    export=False,
    draw_options={
        "polyline": False, "polygon": False, "circle": False,
        "marker": False, "circlemarker": False,
        "rectangle": {"shapeOptions": {"color": "#d62728"}},
    },
    edit_options={"edit": False, "remove": False, "poly": False},
).add_to(m)

# Display the current AOI if any.
if st.session_state.bbox is not None:
    b = st.session_state.bbox
    folium.Rectangle(
        bounds=[[b.south, b.west], [b.north, b.east]],
        color="#d62728",
        weight=2,
        fill=True,
        fill_opacity=0.1,
    ).add_to(m)

folium.LayerControl(collapsed=True, position="topright").add_to(m)

with st.expander("ℹ️ How to use the map", expanded=False):
    st.markdown(
        """
        - **Base layers** (top-right icon): toggle between **Satellite**
          (real Earth imagery), **Labels** overlay, and **Streets**.
        - **Left toolbar**:
            - **▢ Rectangle tool** — draw a new AOI. Click once for the first
              corner, move the mouse, and click again for the opposite corner.
            - To change the AOI, use the **🗑 Clear AOI** button that appears
              below the map, then draw a new rectangle.
        - **Scale bar** (bottom-left) shows real distance.
        - Or skip the map entirely and pick a **preset** in the sidebar.
        """
    )

map_state = st_folium(m, height=500, width="stretch", key="aoi_map")


def _bbox_from_polygon_feature(feature: dict):
    """Extract a BBox from a folium Draw polygon/rectangle feature."""
    geom = feature.get("geometry") or {}
    if geom.get("type") != "Polygon":
        return None
    coords = geom.get("coordinates", [[]])[0]
    if not coords:
        return None
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return BBox(min(xs), min(ys), max(xs), max(ys))


def _bbox_approx_equal(a: BBox, b: BBox, tol: float = 1e-5) -> bool:
    return all(
        abs(x - y) < tol
        for x, y in zip(a.as_tuple(), b.as_tuple())
    )


# Sync session state from whatever the draw plugin currently shows.
# `all_drawings` reflects the full live state after draws/edits/deletes,
# whereas `last_active_drawing` only captures the most-recent draw (so it
# misses edit/delete operations — this is a long-standing streamlit-folium
# quirk).
if map_state:
    all_drawings = map_state.get("all_drawings") or []
    # Filter to polygon/rectangle features only.
    polys = [d for d in all_drawings if (d.get("geometry") or {}).get("type") == "Polygon"]

    if polys:
        # Use the most recent polygon as the AOI.
        new_bbox = _bbox_from_polygon_feature(polys[-1])
        if new_bbox is not None:
            current = st.session_state.bbox
            if current is None or not _bbox_approx_equal(new_bbox, current):
                st.session_state.bbox = new_bbox
                st.session_state.aoi_name = "custom AOI"
                st.session_state.result = None
                st.session_state.fires_df = None
                st.rerun()
    else:
        # The draw toolbar shows an empty state. Only clear the AOI if it
        # previously came from a *drawn* rectangle; don't clobber presets.
        if (
            st.session_state.bbox is not None
            and st.session_state.aoi_name == "custom AOI"
            and st.session_state.get("_had_drawing", False)
        ):
            st.session_state.bbox = None
            st.session_state.result = None
            st.session_state.fires_df = None
            st.session_state._had_drawing = False
            st.rerun()

    # Track whether there's a live drawing on the map — used above so delete
    # only clears the AOI after the user has actually drawn one.
    st.session_state._had_drawing = bool(polys)

if st.session_state.bbox is None:
    st.info(
        "👉 Draw a rectangle on the map (use the ▢ tool on the left of the map), "
        "or load a preset from the sidebar."
    )
else:
    b = st.session_state.bbox
    area = b.area_km2_approx()
    info_col, clear_col = st.columns([5, 1])
    with info_col:
        st.success(
            f"**AOI:** {st.session_state.aoi_name} — "
            f"({b.west:.3f}, {b.south:.3f}) → ({b.east:.3f}, {b.north:.3f}) — "
            f"~{area:,.0f} km²"
        )
    with clear_col:
        if st.button("🗑 Clear AOI", width="stretch"):
            st.session_state.bbox = None
            st.session_state.aoi_name = "custom AOI"
            st.session_state.result = None
            st.session_state.fires_df = None
            st.session_state.preview_before = None
            st.session_state.preview_after = None
            st.session_state._had_drawing = False
            st.rerun()
    if area > 15000:
        st.warning(
            f"AOI is large (~{area:,.0f} km²). Download may be slow or fail. "
            "Consider a smaller region (< 10,000 km²) for interactive use."
        )


# ---------------------------------------------------------------------------
# Analyze button
# ---------------------------------------------------------------------------

st.markdown("### 2. Run change detection")

if data_source == "iolulc":
    can_run = (
        st.session_state.bbox is not None
        and st.session_state.before_year < st.session_state.after_year
    )
else:
    can_run = (
        st.session_state.bbox is not None
        and st.session_state.dw_before_start <= st.session_state.dw_before_end
        and st.session_state.dw_after_start <= st.session_state.dw_after_end
        and st.session_state.dw_before_end < st.session_state.dw_after_start
    )

if st.button("▶ Analyze", type="primary", disabled=not can_run, width="stretch"):
    st.session_state.last_error = None
    st.session_state.result = None
    st.session_state.fires_df = None
    st.session_state.fires_error = None
    st.session_state.fires_requested = bool(show_fires)
    st.session_state.preview_before = None
    st.session_state.preview_after = None

    bbox = st.session_state.bbox

    try:
        if data_source == "iolulc":
            y1 = st.session_state.before_year
            y2 = st.session_state.after_year
            before_label: object = y1
            after_label: object = y2

            with st.spinner(f"Fetching {y1} land cover (IO-LULC)..."):
                lulc_before = fetch_lulc(bbox, y1)
                lulc_before = lulc_before.compute()

            with st.spinner(f"Fetching {y2} land cover (IO-LULC)..."):
                lulc_after = fetch_lulc(bbox, y2)
                lulc_after = lulc_after.compute()

                if lulc_after.shape != lulc_before.shape:
                    lulc_after = lulc_after.interp_like(
                        lulc_before, method="nearest"
                    ).astype("int16")

            # For RGB preview in IO-LULC mode, use the same two years.
            rgb_before_year, rgb_after_year = y1, y2

        else:
            # Dynamic World path.
            b_start = st.session_state.dw_before_start
            b_end = st.session_state.dw_before_end
            a_start = st.session_state.dw_after_start
            a_end = st.session_state.dw_after_end
            before_label = dynamic_world_period_description(b_start, b_end)
            after_label = dynamic_world_period_description(a_start, a_end)

            ee_project = st.session_state.ee_project or None
            with st.spinner(f"Fetching Dynamic World for {before_label}..."):
                lulc_before, n_before = fetch_dynamic_world_mode(
                    bbox, b_start, b_end, project=ee_project
                )
            with st.spinner(f"Fetching Dynamic World for {after_label}..."):
                lulc_after, n_after = fetch_dynamic_world_mode(
                    bbox, a_start, a_end, project=ee_project
                )
                if lulc_after.shape != lulc_before.shape:
                    lulc_after = lulc_after.interp_like(
                        lulc_before, method="nearest"
                    ).astype("int16")

            st.toast(
                f"Dynamic World: {n_before} scene(s) before, "
                f"{n_after} scene(s) after.",
                icon="🛰️",
            )

            # For RGB preview in DW mode: use the calendar year of the
            # midpoint of each window as the S2 summer composite year.
            rgb_before_year = b_start.year + (1 if b_start.month > 9 else 0)
            rgb_after_year = a_start.year + (1 if a_start.month > 9 else 0)

        with st.spinner("Computing change detection..."):
            result = compute_change(
                lulc_before, lulc_after, before_label, after_label
            )
            st.session_state.result = result

        if show_rgb:
            try:
                with st.spinner(f"Fetching Sentinel-2 RGB for {rgb_before_year}..."):
                    st.session_state.preview_before = fetch_s2_rgb_preview(
                        bbox, rgb_before_year
                    )
                    if st.session_state.preview_before is not None:
                        st.session_state.preview_before = st.session_state.preview_before.compute()
                with st.spinner(f"Fetching Sentinel-2 RGB for {rgb_after_year}..."):
                    st.session_state.preview_after = fetch_s2_rgb_preview(
                        bbox, rgb_after_year
                    )
                    if st.session_state.preview_after is not None:
                        st.session_state.preview_after = st.session_state.preview_after.compute()
            except Exception as e:
                st.warning(f"Could not fetch RGB preview: {e}")

        if show_fires:
            if not firms_key_input:
                st.session_state.fires_error = (
                    "FIRMS overlay skipped: no MAP_KEY provided. "
                    "Paste one in the sidebar and click Analyze again."
                )
            else:
                try:
                    period_str = firms_period_description()
                    with st.spinner(f"Fetching FIRMS active fires ({period_str})..."):
                        st.session_state.fires_df = fetch_firms_fires(
                            bbox,
                            map_key=firms_key_input,
                        )
                except Exception as e:
                    st.session_state.fires_error = (
                        f"Could not fetch FIRMS fires: {type(e).__name__}: {e}"
                    )

    except Exception as e:
        st.session_state.last_error = f"{e}\n\n{traceback.format_exc()}"


if st.session_state.last_error:
    with st.expander("❌ Error details", expanded=True):
        st.code(st.session_state.last_error)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

result = st.session_state.result

if result is not None:
    st.markdown("### 3. Results")

    total_pixels = int(np.prod(result.before.shape))
    changed_pixels = int(result.change_mask.sum().item())
    pct_changed = 100.0 * changed_pixels / max(total_pixels, 1)
    total_ha_changed = changed_pixels * result.pixel_area_ha

    forest_loss_ha = sum(
        ha for (fc, tc), ha in result.transition_ha.items()
        if fc == 2 and tc != 2
    )
    built_gain_ha = sum(
        ha for (fc, tc), ha in result.transition_ha.items()
        if tc == 7 and fc != 7
    )

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total area changed", f"{total_ha_changed:,.0f} ha")
    kpi2.metric("% area changed", f"{pct_changed:.2f}%")
    kpi3.metric("Forest loss", f"{forest_loss_ha:,.0f} ha")
    kpi4.metric("Built-up gain", f"{built_gain_ha:,.0f} ha")

    tab_maps, tab_charts, tab_report, tab_fires = st.tabs(
        ["🗺️ Maps", "📊 Charts", "📝 Report", "🔥 Fires"]
    )

    with tab_maps:
        c1, c2 = st.columns(2)
        with c1:
            fig_before = render_lulc_map(result.before, title=f"Land cover {result.before_year}")
            st.image(fig_to_png_bytes(fig_before), width="stretch")
        with c2:
            fig_after = render_lulc_map(result.after, title=f"Land cover {result.after_year}")
            st.image(fig_to_png_bytes(fig_after), width="stretch")

        fig_change = render_change_map(result)
        st.image(fig_to_png_bytes(fig_change), width="stretch")

        if st.session_state.preview_before is not None or st.session_state.preview_after is not None:
            st.markdown("#### Sentinel-2 RGB (true color)")
            c3, c4 = st.columns(2)
            if st.session_state.preview_before is not None:
                with c3:
                    fig = render_rgb_preview(
                        st.session_state.preview_before,
                        title=f"RGB {result.before_year}",
                    )
                    st.image(fig_to_png_bytes(fig), width="stretch")
            if st.session_state.preview_after is not None:
                with c4:
                    fig = render_rgb_preview(
                        st.session_state.preview_after,
                        title=f"RGB {result.after_year}",
                    )
                    st.image(fig_to_png_bytes(fig), width="stretch")

    with tab_charts:
        fig_bar = transition_bar_chart(result, n=8)
        st.image(fig_to_png_bytes(fig_bar), width="stretch")

        st.markdown("#### Net class area change")
        all_classes = sorted(
            set(result.before_area_ha) | set(result.after_area_ha)
        )
        rows = []
        for c in all_classes:
            if c == 0 or c not in CLASS_NAMES:
                continue
            b = result.before_area_ha.get(c, 0.0)
            a = result.after_area_ha.get(c, 0.0)
            rows.append({
                "Class": CLASS_NAMES[c],
                f"{result.before_year} (ha)": round(b, 1),
                f"{result.after_year} (ha)": round(a, 1),
                "Δ (ha)": round(a - b, 1),
            })
        st.dataframe(pd.DataFrame(rows), width="stretch")

        st.markdown("#### Notable transitions")
        notable = notable_transitions_summary(result)
        notable = [n for n in notable if n["hectares"] > 0]
        if notable:
            st.dataframe(
                pd.DataFrame(notable)[["label", "hectares"]].rename(
                    columns={"label": "Transition", "hectares": "Hectares"}
                ).round({"Hectares": 1}),
                width="stretch",
            )
        else:
            st.info("No predefined 'notable' transitions detected.")

    with tab_report:
        report_md = format_change_report(result, aoi_name=st.session_state.aoi_name)
        st.markdown(report_md)
        def _safe_label(s: object) -> str:
            return re.sub(r"[^0-9A-Za-z_-]+", "_", str(s)).strip("_") or "period"
        fname_before = _safe_label(result.before_year)
        fname_after = _safe_label(result.after_year)
        st.download_button(
            "⬇ Download report (Markdown)",
            data=report_md.encode("utf-8"),
            file_name=f"change_report_{fname_before}_to_{fname_after}.md",
            mime="text/markdown",
        )

    with tab_fires:
        fires = st.session_state.fires_df
        fires_error = st.session_state.fires_error
        fires_was_requested_at_analyze_time = st.session_state.fires_requested

        if fires_error:
            st.error(fires_error)
        elif fires is None:
            if not show_fires:
                st.info(
                    "Fires overlay is off. Tick **🔥 Fetch NASA FIRMS fire "
                    "detections** in the sidebar, then click **▶ Analyze**."
                )
            elif not fires_was_requested_at_analyze_time:
                # Fires is checked now, but the last Analyze run happened
                # before it was enabled. Tell the user to re-run.
                st.warning(
                    "Fires are enabled but the last analysis didn't include "
                    "them. Click **▶ Analyze** again to fetch fire detections."
                )
            elif not firms_key_input:
                st.warning(
                    "Fires are enabled but no MAP_KEY was provided. "
                    "Paste your key in the sidebar and click **▶ Analyze** again."
                )
            else:
                st.info("Fires were requested but no data was returned.")
        elif len(fires) == 0:
            st.info(
                f"No FIRMS active-fire detections in this AOI for "
                f"{firms_period_description()}."
            )
        else:
            st.success(
                f"**{len(fires):,}** active-fire detections in the AOI "
                f"(FIRMS NRT, {firms_period_description()})."
            )
            st.caption(
                "FIRMS shows current fire activity — useful as real-time "
                "context alongside the historical land-cover change above."
            )

            _xs, _ys = _get_xy_coords(result.before)
            fire_map = folium.Map(
                location=[
                    float((_ys.min() + _ys.max()) / 2),
                    float((_xs.min() + _xs.max()) / 2),
                ],
                zoom_start=9,
                tiles="CartoDB positron",
            )
            for _, row in fires.head(2000).iterrows():
                folium.CircleMarker(
                    location=[row["latitude"], row["longitude"]],
                    radius=2,
                    color="red",
                    fill=True,
                    fill_opacity=0.6,
                    weight=0,
                ).add_to(fire_map)
            # returned_objects=[] disables streamlit-folium's default behavior
            # of returning map state (and triggering a rerun) on every zoom/pan.
            # Without this, zooming the fires map causes Streamlit to dim the
            # whole results section while it reruns.
            st_folium(
                fire_map,
                height=450,
                width="stretch",
                key="fire_map",
                returned_objects=[],
            )

            st.dataframe(fires.head(200), width="stretch")


else:
    st.info("Run the analysis to see results.")
