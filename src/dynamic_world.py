"""
Google Dynamic World — near-real-time 10m land cover via Google Earth Engine.

Dynamic World (https://dynamicworld.app/) is a global 10m land cover product
generated every time Sentinel-2 passes (every 2-5 days), going back to 2015.
This is the main advantage over IO-LULC's 1-year resolution with a ~12-month
publication lag: with DW you can compare "April 2026" against "April 2024"
and see what's changed in nearly-real time.

Access is through Google Earth Engine (free, requires one-time auth):

    pip install earthengine-api
    earthengine authenticate            # opens a browser, one-time

Or for headless / CI use a service account:

    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

Classes (Dynamic World `label` band):
    0 = Water
    1 = Trees
    2 = Grass
    3 = Flooded vegetation
    4 = Crops
    5 = Shrub and scrub
    6 = Built
    7 = Bare
    8 = Snow and ice

To keep the rest of the change-detection pipeline (CLASS_NAMES, CLASS_COLORS,
NOTABLE_TRANSITIONS, reports) working unchanged, we remap DW codes to the
IO-LULC 9-class scheme used everywhere else. The remap is lossy (DW's Grass
and Shrub both fold into Rangeland) but keeps a single, consistent legend
across data sources.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import date
from typing import Optional, Tuple
from urllib.request import urlretrieve

import numpy as np
import xarray as xr

from .data import BBox


DW_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"

# Dynamic World → IO-LULC class code remap.
# IO-LULC scheme is what change_detection.CLASS_NAMES uses.
DW_TO_IOLULC = {
    0: 1,    # Water       → Water
    1: 2,    # Trees       → Trees
    2: 11,   # Grass       → Rangeland
    3: 4,    # Flooded veg → Flooded vegetation
    4: 5,    # Crops       → Crops
    5: 11,   # Shrub/scrub → Rangeland
    6: 7,    # Built       → Built Area
    7: 8,    # Bare        → Bare Ground
    8: 9,    # Snow/ice    → Snow/Ice
}


_ee_initialized_with_project: Optional[str] = None


def _resolve_project(explicit: Optional[str] = None) -> Optional[str]:
    """
    Work out which Google Cloud project to use for Earth Engine, in this order:
      1. The `explicit` argument (e.g. from the Streamlit sidebar text input)
      2. EARTHENGINE_PROJECT env var
      3. GOOGLE_CLOUD_PROJECT env var
    If none is set, returns None and lets EE fall back to whatever
    `earthengine set_project` wrote into the cached credentials.
    """
    if explicit:
        return explicit
    for var in ("EARTHENGINE_PROJECT", "GOOGLE_CLOUD_PROJECT"):
        val = os.environ.get(var)
        if val:
            return val
    return None


def check_earth_engine_auth(project: Optional[str] = None) -> dict:
    """
    Check whether Earth Engine is importable and authenticated for `project`.

    Returns:
        {"ok": bool, "message": str, "how_to_fix": str}
    """
    try:
        import ee  # noqa: F401
    except ImportError:
        return {
            "ok": False,
            "message": "earthengine-api is not installed.",
            "how_to_fix": "Run: pip install earthengine-api",
        }

    import ee

    resolved_project = _resolve_project(project)

    # Track which credential path we ended up using, so the UI can report it.
    auth_method: str = "unknown"
    auth_identity: Optional[str] = None

    # Try to initialize without triggering a browser popup.
    try:
        sa_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        init_kwargs = {}
        if resolved_project:
            init_kwargs["project"] = resolved_project

        if sa_path and os.path.exists(sa_path):
            credentials = ee.ServiceAccountCredentials(None, sa_path)
            ee.Initialize(credentials, **init_kwargs)
            auth_method = "service_account"
            # Pull the client_email out of the JSON key for display.
            try:
                with open(sa_path, "r") as f:
                    auth_identity = json.load(f).get("client_email")
            except Exception:
                auth_identity = sa_path
        else:
            # Uses cached credentials from `earthengine authenticate`.
            ee.Initialize(**init_kwargs)
            auth_method = "user_credentials"
    except Exception as e:
        msg = str(e)
        if "no project found" in msg.lower() or "project" in msg.lower():
            return {
                "ok": False,
                "message": f"Earth Engine needs a Google Cloud project ID: {e}",
                "how_to_fix": (
                    "Pick ONE of:\n"
                    "  1. Paste your project ID in the sidebar box below.\n"
                    "  2. Run once in a terminal: "
                    "`earthengine set_project YOUR_PROJECT_ID`\n"
                    "  3. Export EARTHENGINE_PROJECT=YOUR_PROJECT_ID in your shell.\n\n"
                    "To get/create a project: https://console.cloud.google.com/ "
                    "(any free-tier GCP project works; the EE API must be enabled "
                    "once at https://console.cloud.google.com/apis/library/earthengine.googleapis.com )."
                ),
            }
        return {
            "ok": False,
            "message": f"Earth Engine not authenticated: {e}",
            "how_to_fix": (
                "Run once: `earthengine authenticate` "
                "(or set GOOGLE_APPLICATION_CREDENTIALS to a service-account JSON)."
            ),
        }

    # A zero-cost sanity check: ping the DW collection metadata.
    try:
        ee.ImageCollection(DW_COLLECTION).limit(1).size().getInfo()
    except Exception as e:
        return {
            "ok": False,
            "message": f"Earth Engine auth OK, but DW query failed: {e}",
            "how_to_fix": "Check that your EE account has access to Dynamic World (it is public).",
        }

    _remember_initialized(resolved_project)

    project_part = f" (project: {resolved_project})" if resolved_project else ""
    return {
        "ok": True,
        "message": f"Earth Engine is authenticated{project_part}.",
        "how_to_fix": "",
        # Still expose the credential path programmatically in case anything
        # else wants to inspect it; just don't surface it in the user message.
        "auth_method": auth_method,
        "auth_identity": auth_identity,
        "project": resolved_project,
    }


def _remember_initialized(project: Optional[str]) -> None:
    global _ee_initialized_with_project
    _ee_initialized_with_project = project or ""


def _ensure_ee(project: Optional[str] = None):
    """Initialize EE once per process (per project), with a clean error if not set up."""
    global _ee_initialized_with_project
    resolved = _resolve_project(project)
    # Re-use the existing init if the project matches.
    if _ee_initialized_with_project is not None and (
        _ee_initialized_with_project == (resolved or "")
    ):
        return
    status = check_earth_engine_auth(project=resolved)
    if not status["ok"]:
        raise RuntimeError(
            f"Cannot use Dynamic World: {status['message']} {status['how_to_fix']}"
        )


def fetch_dynamic_world_mode(
    bbox: BBox,
    start: date,
    end: date,
    resolution_m: int = 10,
    project: Optional[str] = None,
) -> Tuple[xr.DataArray, int]:
    """
    Compute the per-pixel modal (most common) Dynamic World class over a date
    window, and return an xarray DataArray remapped to the IO-LULC scheme.

    Returns (dataarray, n_scenes_used).

    The mode composite is the standard way to summarize DW across a window:
    it filters out transient misclassifications (clouds, shadows) and gives
    you a robust "typical" land cover label per pixel.
    """
    _ensure_ee(project=project)
    import ee

    geom = ee.Geometry.Rectangle([bbox.west, bbox.south, bbox.east, bbox.north])
    ic = (
        ee.ImageCollection(DW_COLLECTION)
        .filterBounds(geom)
        .filterDate(start.isoformat(), end.isoformat())
        .select("label")
    )
    n = int(ic.size().getInfo())
    if n == 0:
        raise RuntimeError(
            f"No Dynamic World scenes for bbox={bbox.as_tuple()} "
            f"between {start} and {end}. Widen the date window (DW revisits "
            f"every 2-5 days; 30-60 days typically yields enough scenes)."
        )

    mode_img = ic.mode().toInt().rename("label")

    # Request a GeoTIFF download URL. EE caps single download at 50MB; at 10m
    # resolution that's roughly a 0.6° x 0.6° bbox. For larger AOIs the user
    # should pick a coarser `resolution_m` (e.g. 20 or 30).
    try:
        url = mode_img.getDownloadURL({
            "region": geom,
            "scale": resolution_m,
            "crs": "EPSG:4326",
            "format": "GEO_TIFF",
        })
    except Exception as e:
        raise RuntimeError(
            f"Earth Engine download URL request failed: {e}. "
            f"If the error mentions size limits, try a smaller AOI or "
            f"coarser resolution_m (e.g. 20 or 30)."
        )

    tmp_path: Optional[str] = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".tif")
        os.close(fd)
        urlretrieve(url, tmp_path)

        import rioxarray
        da = rioxarray.open_rasterio(tmp_path)
        # open_rasterio returns a (band, y, x) array; squeeze to (y, x).
        if "band" in da.dims:
            da = da.isel(band=0, drop=True)
        da = da.astype("int16")

        # Apply DW → IO-LULC remap via a lookup table (fast, vectorized).
        lut = np.zeros(max(DW_TO_IOLULC.keys()) + 1, dtype=np.int16)
        for dw_code, io_code in DW_TO_IOLULC.items():
            lut[dw_code] = io_code
        values = da.values
        # Clip to valid DW range, then index.
        safe = np.clip(values, 0, len(lut) - 1)
        mapped = lut[safe]
        # Preserve nodata (values outside DW range map to 0 = "No Data").
        mapped = np.where((values >= 0) & (values < len(lut)), mapped, 0)

        remapped = xr.DataArray(
            mapped,
            coords=da.coords,
            dims=da.dims,
            name="data",
        )
        return remapped, n
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def dynamic_world_period_description(start: date, end: date) -> str:
    """Human-readable period string for reports / captions."""
    return f"{start.isoformat()} → {end.isoformat()}"
