"""
Data fetching from Microsoft Planetary Computer.

Primary products:
- io-lulc-annual-v02 : 9-class annual land cover, 10m, 2017-2023 global
- sentinel-2-l2a     : surface-reflectance imagery used for RGB previews

All access is anonymous; no API key required. Planetary Computer signs asset
URLs on the fly via the `planetary_computer` package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import planetary_computer
import pystac_client
import xarray as xr
import odc.stac


STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
LULC_COLLECTION = "io-lulc-annual-v02"
S2_COLLECTION = "sentinel-2-l2a"


def _open_catalog() -> pystac_client.Client:
    """Open the Planetary Computer STAC catalog with URL signing enabled."""
    return pystac_client.Client.open(
        STAC_URL,
        modifier=planetary_computer.sign_inplace,
    )


def _normalize_xy(obj):
    """
    Rename spatial coords/dims to a consistent ('x', 'y') scheme.

    odc.stac.load returns 'longitude'/'latitude' when the CRS is geographic
    and 'x'/'y' when projected — we normalize so the rest of the code only
    has to deal with one convention.
    """
    rename = {}
    for old in ("longitude", "lon"):
        if old in obj.dims or old in getattr(obj, "coords", {}):
            rename[old] = "x"
            break
    for old in ("latitude", "lat"):
        if old in obj.dims or old in getattr(obj, "coords", {}):
            rename[old] = "y"
            break
    if rename:
        obj = obj.rename(rename)
    return obj


@dataclass
class BBox:
    """Simple lon/lat bounding box. Order: (west, south, east, north)."""
    west: float
    south: float
    east: float
    north: float

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.west, self.south, self.east, self.north)

    def area_km2_approx(self) -> float:
        """Very rough area estimate (good enough for UI warnings)."""
        lat_mid = (self.south + self.north) / 2
        lon_km = 111.32 * np.cos(np.deg2rad(lat_mid))
        lat_km = 110.57
        return abs(self.east - self.west) * lon_km * abs(self.north - self.south) * lat_km


def get_bbox_from_geojson(geojson: dict) -> BBox:
    """Extract a BBox from a GeoJSON Feature or geometry."""
    from shapely.geometry import shape

    if geojson.get("type") == "Feature":
        geom = shape(geojson["geometry"])
    elif "geometry" in geojson:
        geom = shape(geojson["geometry"])
    else:
        geom = shape(geojson)

    minx, miny, maxx, maxy = geom.bounds
    return BBox(minx, miny, maxx, maxy)


def fetch_lulc(bbox: BBox, year: int, resolution_deg: float = 0.0001) -> xr.DataArray:
    """
    Fetch Impact Observatory / Esri land cover for a given bbox + year.

    Returns an xarray DataArray with integer class codes. Coordinates in EPSG:4326.

    Classes (io-lulc-annual-v02):
        0  = No Data
        1  = Water
        2  = Trees
        4  = Flooded Vegetation
        5  = Crops
        7  = Built Area
        8  = Bare Ground
        9  = Snow/Ice
        10 = Clouds
        11 = Rangeland
    """
    catalog = _open_catalog()

    time_range = f"{year}-01-01/{year}-12-31"
    search = catalog.search(
        collections=[LULC_COLLECTION],
        bbox=bbox.as_tuple(),
        datetime=time_range,
    )
    items = list(search.items())
    if not items:
        raise RuntimeError(
            f"No IO-LULC items found for bbox={bbox.as_tuple()} year={year}. "
            f"The annual product is typically published with a ~12-month lag, "
            f"so the most recent year may not be available yet. "
            f"Try an earlier year, or widen the bbox slightly."
        )

    data = odc.stac.load(
        items,
        bbox=bbox.as_tuple(),
        crs="EPSG:4326",
        resolution=resolution_deg,
    )
    data = _normalize_xy(data)

    # The data variable is named "data" in this collection.
    da = data["data"]

    # Items within one year are identical; take the first time slice if present.
    if "time" in da.dims:
        da = da.isel(time=0)

    # Ensure values are integer class codes.
    return da.astype("int16")


def fetch_s2_rgb_preview(
    bbox: BBox,
    year: int,
    resolution_deg: float = 0.0005,
    max_cloud_cover: int = 20,
) -> Optional[xr.Dataset]:
    """
    Fetch a median RGB composite of Sentinel-2 for the given bbox + year
    (summer half of the year by default, to minimize seasonal variation).

    Returns None if no cloud-free imagery is available.
    """
    catalog = _open_catalog()

    # Prefer the drier/clearer half of the year. Northern vs Southern hemisphere
    # handling could be added; for now we use a generic mid-year window.
    time_range = f"{year}-05-01/{year}-09-30"
    search = catalog.search(
        collections=[S2_COLLECTION],
        bbox=bbox.as_tuple(),
        datetime=time_range,
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
    )
    items = list(search.items())
    if not items:
        return None

    # Limit to a reasonable count for quick previews.
    items = items[:15]

    ds = odc.stac.load(
        items,
        bands=["B04", "B03", "B02"],  # Red, Green, Blue
        bbox=bbox.as_tuple(),
        crs="EPSG:4326",
        resolution=resolution_deg,
    )
    ds = _normalize_xy(ds)

    # Median composite across time to reduce cloud artifacts.
    rgb = ds.median(dim="time", skipna=True)
    return rgb


def available_years() -> List[int]:
    """
    Return the list of supported analysis years.

    IO-LULC v02 is an annual product with ~12-month publication lag.
    Years 2017-2024 are generally available as of early 2026.
    If a user selects a year that hasn't been published yet, fetch_lulc
    will raise a clear "no items found" error.
    """
    return list(range(2017, 2025))
