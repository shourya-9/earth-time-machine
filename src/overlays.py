"""
External data overlays that add context to the detected changes.

Currently implemented:
- NASA FIRMS near-real-time active-fire detections (VIIRS_SNPP + VIIRS_NOAA20)

FIRMS has a free API but requires a MAP_KEY. Get one here (free, instant):
    https://firms.modaps.eosdis.nasa.gov/api/area/

Export as FIRMS_MAP_KEY, or paste directly into the Streamlit sidebar.

NOTE ON DATA PERIOD
-------------------
The /api/area/csv/ endpoint reliably serves NRT (near-real-time) data covering
roughly the last 60 days. Historical "Standard Processing" (SP) sources are
documented but return 400 Bad Request for older dates in practice, so this
module sticks to NRT and queries the most recent window (up to 60 days).

That means the fires overlay shows *current* fire activity in the region,
which provides real-time context to the land-cover change analysis rather
than historical co-occurrence. For a historical archive, consider:
  - Google Earth Engine's FIRMS collection
  - NASA's FIRMS Fire Archive Download service
  - MODIS Burned Area product (MCD64A1), monthly, free
"""

from __future__ import annotations

import io
import os
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import requests

from .data import BBox


FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
FIRMS_STATUS_URL = "https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/"
FIRMS_MAX_DAYS = 5               # API hard limit per call: "Invalid day range. Expects [1..5]."
FIRMS_NRT_WINDOW_DAYS = 60       # NRT data covers ~last 60 days

# NRT sources that work reliably on the /area/csv/ endpoint.
# Ordered by reliability / coverage — used as a fallback chain.
FIRMS_NRT_SOURCES = (
    "VIIRS_SNPP_NRT",
    "VIIRS_NOAA20_NRT",
    "VIIRS_NOAA21_NRT",
    "MODIS_NRT",
)


def check_firms_key_status(map_key: str) -> dict:
    """
    Check the status of a FIRMS MAP_KEY.

    Returns a dict with at least:
        {"ok": bool, "message": str, "raw": <raw response text or json>}

    The FIRMS status endpoint returns JSON like:
        {
          "mapkey": "xxxxx",
          "current_transactions": 15,
          "transaction_limit": 5000,
          "transaction_interval_minutes": 10
        }
    or an error string if the key is invalid / inactive.
    """
    try:
        resp = requests.get(
            FIRMS_STATUS_URL,
            params={"MAP_KEY": map_key},
            timeout=30,
        )
    except requests.RequestException as e:
        return {"ok": False, "message": f"Could not reach FIRMS status endpoint: {e}", "raw": None}

    text = resp.text.strip()
    if resp.status_code != 200:
        return {
            "ok": False,
            "message": f"FIRMS status endpoint returned HTTP {resp.status_code}: {text[:200]}",
            "raw": text,
        }

    # Try to parse as JSON. If it's a plain-text error, surface it.
    try:
        data = resp.json()
    except ValueError:
        # Non-JSON response usually means an error string.
        return {"ok": False, "message": f"FIRMS key check failed: {text[:200]}", "raw": text}

    # Heuristic: if the JSON mentions rate limit or error, flag it.
    if isinstance(data, dict):
        current = data.get("current_transactions")
        limit = data.get("transaction_limit")
        if current is not None and limit is not None and current >= limit:
            return {
                "ok": False,
                "message": (
                    f"FIRMS MAP_KEY is rate-limited: {current}/{limit} transactions "
                    f"used in the last {data.get('transaction_interval_minutes', '?')} minutes. "
                    f"Wait a few minutes and retry."
                ),
                "raw": data,
            }
        return {"ok": True, "message": "FIRMS MAP_KEY is active.", "raw": data}

    return {"ok": False, "message": f"Unexpected FIRMS status response: {text[:200]}", "raw": data}


def _fetch_firms_single_source(
    bbox: BBox,
    query_start: date,
    query_end: date,
    map_key: str,
    source: str,
) -> tuple[list[pd.DataFrame], Optional[str]]:
    """
    Try to fetch FIRMS data for one source across the full date range.
    Returns (list of chunk DataFrames, last_error_or_None).
    """
    # FIRMS area query bbox: "west,south,east,north"
    area = f"{bbox.west},{bbox.south},{bbox.east},{bbox.north}"

    frames: list[pd.DataFrame] = []
    last_error: Optional[str] = None

    cur = query_start
    while cur <= query_end:
        chunk_end = min(cur + timedelta(days=FIRMS_MAX_DAYS - 1), query_end)
        chunk_days = (chunk_end - cur).days + 1
        url = (
            f"{FIRMS_BASE}/{map_key}/{source}/{area}/"
            f"{chunk_days}/{cur.strftime('%Y-%m-%d')}"
        )
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code == 400:
                body = resp.text.strip()[:200] if resp.text else "(empty body)"
                last_error = (
                    f"FIRMS returned 400 for source={source} start={cur} "
                    f"chunk={chunk_days}d. Body: {body}"
                )
            elif resp.status_code == 401 or resp.status_code == 403:
                last_error = (
                    f"FIRMS returned {resp.status_code} (auth) for source={source}. "
                    f"MAP_KEY may be invalid or inactive."
                )
                break  # no point retrying other dates with same key
            elif resp.status_code == 429:
                last_error = (
                    f"FIRMS returned 429 (rate-limited) for source={source}. "
                    f"Wait a few minutes and retry."
                )
                break
            else:
                resp.raise_for_status()
                text = resp.text.strip()
                if text and not text.lower().startswith("invalid"):
                    try:
                        df_chunk = pd.read_csv(io.StringIO(text))
                        if not df_chunk.empty:
                            df_chunk["source"] = source
                            frames.append(df_chunk)
                    except Exception as e:
                        last_error = f"Could not parse FIRMS CSV for {cur}: {e}"
        except requests.RequestException as e:
            last_error = f"Network error for source={source} {cur}: {e}"

        cur = chunk_end + timedelta(days=1)

    return frames, last_error


def fetch_firms_fires(
    bbox: BBox,
    start: Optional[date] = None,
    end: Optional[date] = None,
    map_key: Optional[str] = None,
    source: str = "VIIRS_SNPP_NRT",
    window_days: int = FIRMS_NRT_WINDOW_DAYS,
    try_fallback_sources: bool = True,
) -> pd.DataFrame:
    """
    Fetch NASA FIRMS fire detections in a bbox.

    The `start` and `end` parameters are accepted for backwards compatibility
    but will be clipped to the most recent `window_days` (default 60) — this is
    the reliable window for the NRT endpoint.

    If `try_fallback_sources=True` (default) and the requested `source` returns
    no data, the other NRT sources are tried in order before giving up.

    Before any data queries, the MAP_KEY is checked against the FIRMS status
    endpoint so invalid or rate-limited keys surface a clear error immediately.

    Returns a DataFrame with columns:
        latitude, longitude, acq_date, acq_time, confidence, frp, ...
    plus a `source` column identifying the sensor.
    """
    map_key = map_key or os.environ.get("FIRMS_MAP_KEY")
    if not map_key:
        raise RuntimeError(
            "No FIRMS MAP_KEY provided. Get one free at "
            "https://firms.modaps.eosdis.nasa.gov/api/area/ "
            "and paste it into the Streamlit sidebar (or set FIRMS_MAP_KEY)."
        )

    if source not in FIRMS_NRT_SOURCES:
        raise ValueError(
            f"Unsupported source {source!r}. Must be one of {FIRMS_NRT_SOURCES}."
        )

    # Verify the key is active before burning chunks on 400s.
    status = check_firms_key_status(map_key)
    if not status["ok"]:
        raise RuntimeError(
            f"FIRMS MAP_KEY check failed: {status['message']} "
            f"Verify at https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY={map_key}"
        )

    today = date.today()
    # Clip the requested window to the last `window_days`.
    query_end = min(end or today, today)
    query_start = max(
        start or (today - timedelta(days=window_days - 1)),
        today - timedelta(days=window_days - 1),
    )
    if query_end < query_start:
        query_start = query_end

    # Build the source try-list: the requested one first, then the rest.
    if try_fallback_sources:
        sources_to_try = [source] + [s for s in FIRMS_NRT_SOURCES if s != source]
    else:
        sources_to_try = [source]

    all_frames: list[pd.DataFrame] = []
    errors: list[str] = []

    for src in sources_to_try:
        frames, err = _fetch_firms_single_source(
            bbox, query_start, query_end, map_key, src
        )
        if frames:
            all_frames.extend(frames)
            # Stop as soon as we have data from one source to keep the
            # result clean (no mixed-sensor duplicates).
            break
        if err:
            errors.append(err)

    if not all_frames:
        if errors:
            raise RuntimeError(
                f"No FIRMS data returned for bbox={bbox.as_tuple()} "
                f"{query_start} → {query_end}. "
                f"Tried sources {sources_to_try}. Errors: {' | '.join(errors)}"
            )
        # No errors, no data: the area is just quiet (no fires).
        return pd.DataFrame(
            columns=["latitude", "longitude", "acq_date", "acq_time", "confidence", "frp", "source"]
        )

    df = pd.concat(all_frames, ignore_index=True)
    if "acq_date" in df.columns:
        df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    return df


def firms_period_description(
    start: Optional[date] = None,
    end: Optional[date] = None,
    window_days: int = FIRMS_NRT_WINDOW_DAYS,
) -> str:
    """Return a human-readable description of the period actually queried."""
    today = date.today()
    query_end = min(end or today, today)
    query_start = max(
        start or (today - timedelta(days=window_days - 1)),
        today - timedelta(days=window_days - 1),
    )
    if query_end < query_start:
        query_start = query_end
    return f"{query_start.isoformat()} → {query_end.isoformat()}"
