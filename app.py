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

import base64
import os
import re
import traceback
from datetime import date, timedelta
from io import BytesIO

import folium
import numpy as np
import pandas as pd
import streamlit as st
from folium.plugins import Draw, Geocoder
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
    page_title="Earth Time Machine",
    page_icon="🌎",
    layout="wide",
)




# ---------------------------------------------------------------------------
# Global styling — custom CSS for a modern, polished look
# ---------------------------------------------------------------------------
# Everything below is pure presentation — no functional changes. Uses Google
# Fonts (Inter for body, Space Grotesk for headings), gradient brand colors,
# softer shadows, better buttons / metrics / tabs / sidebar.

st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

  html, body, [class*="css"], .stMarkdown, .stText, [data-testid="stAppViewContainer"] {
      font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
  }

  /* --- Animated starfield + aurora behind the whole app -------------- */
  /* Layer 0: drifting green/blue/violet aurora blobs.
     Layer 1-3: three parallax layers of tiny "stars" made from repeating
     radial-gradient dots, each drifting at a different speed so the sky
     looks alive. All purely CSS — no JS, no canvas. */
  [data-testid="stAppViewContainer"] {
      position: relative;
      background: #070b16 !important;
  }
  [data-testid="stAppViewContainer"]::before {
      content: "";
      position: fixed;
      inset: 0;
      z-index: 0;
      pointer-events: none;
      background:
          radial-gradient(ellipse 60% 40% at 15% 20%, rgba(16,185,129,0.18) 0%, transparent 60%),
          radial-gradient(ellipse 55% 45% at 85% 75%, rgba(59,130,246,0.15) 0%, transparent 60%),
          radial-gradient(ellipse 40% 30% at 70% 20%, rgba(139,92,246,0.10) 0%, transparent 60%);
      animation: auroraDrift 24s ease-in-out infinite alternate;
      filter: blur(20px);
  }
  [data-testid="stAppViewContainer"]::after {
      content: "";
      position: fixed;
      inset: 0;
      z-index: 0;
      pointer-events: none;
      background-image:
          radial-gradient(1.5px 1.5px at 20px 30px, rgba(255,255,255,0.85), transparent 50%),
          radial-gradient(1px 1px at 60px 120px, rgba(255,255,255,0.6), transparent 50%),
          radial-gradient(1.5px 1.5px at 130px 50px, rgba(200,230,255,0.7), transparent 50%),
          radial-gradient(1px 1px at 180px 150px, rgba(255,255,255,0.5), transparent 50%),
          radial-gradient(2px 2px at 240px 80px, rgba(167,243,208,0.8), transparent 50%),
          radial-gradient(1px 1px at 300px 200px, rgba(255,255,255,0.55), transparent 50%),
          radial-gradient(1.5px 1.5px at 360px 40px, rgba(255,255,255,0.7), transparent 50%),
          radial-gradient(1px 1px at 420px 180px, rgba(200,230,255,0.5), transparent 50%);
      background-size: 500px 300px;
      animation: starsDrift 120s linear infinite, twinkle 4s ease-in-out infinite;
      opacity: 0.75;
  }
  @keyframes auroraDrift {
      0%   { transform: translate3d(0, 0, 0) scale(1); }
      50%  { transform: translate3d(-20px, 15px, 0) scale(1.08); }
      100% { transform: translate3d(15px, -10px, 0) scale(1.02); }
  }
  @keyframes starsDrift {
      from { background-position: 0 0; }
      to   { background-position: -500px 300px; }
  }
  @keyframes twinkle {
      0%, 100% { opacity: 0.75; }
      50%      { opacity: 0.45; }
  }

  /* Make sure actual content sits above the background layers. The sidebar
     is deliberately NOT in this rule — Streamlit manages its own stacking
     context for scroll behavior, and forcing position/z-index breaks the
     internal overflow-y container. The sidebar's own opaque gradient
     background (defined below) already covers the animation layers. */
  .main .block-container,
  .stApp > header {
      position: relative;
      z-index: 1;
  }

  /* Streamlit's default top padding on the main block-container is ~6rem,
     which leaves too much empty space above the hero. Pull the hero up
     closer to the top without flush-mounting it against the edge. */
  .main .block-container,
  [data-testid="stMainBlockContainer"],
  [data-testid="block-container"] {
      padding-top: 3rem !important;
  }

  /* --- Hero banner --------------------------------------------------- */
  .hero {
      padding: 2.5rem 2.25rem 2rem 2.25rem;
      background:
          radial-gradient(circle at 12% 18%, rgba(16,185,129,0.18) 0%, transparent 45%),
          radial-gradient(circle at 88% 82%, rgba(59,130,246,0.16) 0%, transparent 45%),
          linear-gradient(135deg, rgba(15,21,40,0.85) 0%, rgba(19,26,44,0.85) 100%);
      backdrop-filter: blur(6px);
      -webkit-backdrop-filter: blur(6px);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 18px;
      margin-bottom: 1.75rem;
      position: relative;
      overflow: hidden;
  }
  /* Orbit arc sweeping across the hero — feels like a satellite track. */
  .hero::before {
      content: "";
      position: absolute;
      width: 900px;
      height: 900px;
      left: -300px;
      top: -650px;
      border-radius: 50%;
      border: 1px solid rgba(16,185,129,0.18);
      box-shadow: inset 0 0 80px rgba(16,185,129,0.04);
      pointer-events: none;
  }
  .hero::after {
      content: "";
      position: absolute;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: #34d399;
      box-shadow: 0 0 14px #10b981, 0 0 28px rgba(16,185,129,0.6);
      top: 50%;
      left: 50%;
      transform-origin: -300px -100px;
      animation: orbit 14s linear infinite;
      pointer-events: none;
  }
  @keyframes orbit {
      from { transform: rotate(0deg) translate(450px) rotate(0deg); }
      to   { transform: rotate(360deg) translate(450px) rotate(-360deg); }
  }
  .hero h1 {
      font-family: 'Space Grotesk', sans-serif !important;
      font-weight: 700;
      font-size: 2.75rem;
      margin: 0;
      line-height: 1.05;
      letter-spacing: -0.02em;
      background: linear-gradient(135deg, #10b981 0%, #3b82f6 55%, #8b5cf6 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
  }
  .hero p {
      font-size: 1.02rem;
      color: #9ca3af;
      margin: 0.75rem 0 0 0;
      max-width: 680px;
      line-height: 1.55;
  }
  .hero .pill {
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      background: rgba(16,185,129,0.1);
      color: #34d399;
      border: 1px solid rgba(16,185,129,0.25);
      padding: 0.3rem 0.8rem;
      border-radius: 999px;
      font-size: 0.78rem;
      font-weight: 500;
      margin-bottom: 1rem;
      letter-spacing: 0.02em;
  }
  .hero .pill .dot {
      width: 6px; height: 6px;
      background: #10b981;
      border-radius: 50%;
      box-shadow: 0 0 8px #10b981;
      animation: pulse 2s ease-in-out infinite;
  }
  @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.5; }
  }

  /* --- Section headings ---------------------------------------------- */
  h2, h3 {
      font-family: 'Space Grotesk', sans-serif !important;
      font-weight: 600 !important;
      color: #f3f4f6 !important;
      letter-spacing: -0.01em;
  }
  h3 {
      padding-bottom: 0.4rem;
      border-bottom: 1px solid rgba(255,255,255,0.06);
      margin-top: 1.5rem !important;
      margin-bottom: 1rem !important;
  }

  /* --- Compact horizontal stats strip (under hero) ------------------- */
  .stats-strip {
      display: flex;
      flex-wrap: wrap;
      gap: 0.6rem;
      margin: -0.5rem 0 1.25rem 0;
  }
  .stat-chip {
      display: inline-flex;
      align-items: center;
      gap: 0.6rem;
      background: rgba(255,255,255,0.025);
      border: 1px solid rgba(255,255,255,0.07);
      padding: 0.35rem 0.85rem;
      border-radius: 999px;
      font-size: 0.78rem;
      color: #94a3b8;
      transition: all 0.18s ease;
      backdrop-filter: blur(8px);
  }
  .stat-chip:hover {
      border-color: rgba(16,185,129,0.35);
      background: rgba(16,185,129,0.05);
      transform: translateY(-1px);
  }
  .stat-chip .k {
      font-size: 0.68rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #6b7280;
      font-weight: 500;
  }
  .stat-chip .v {
      font-family: 'Space Grotesk', sans-serif;
      font-weight: 600;
      color: #e5e7eb;
  }
  .stat-chip .live-dot {
      display: inline-block;
      width: 6px; height: 6px;
      background: #10b981;
      border-radius: 50%;
      margin-right: 4px;
      box-shadow: 0 0 6px #10b981;
      animation: pulse 1.8s ease-in-out infinite;
      vertical-align: middle;
  }

  /* --- Buttons -------------------------------------------------------- */
  .stButton > button {
      font-family: 'Inter', sans-serif !important;
      font-weight: 500;
      border-radius: 9px;
      border: 1px solid rgba(255,255,255,0.1);
      transition: all 0.18s ease;
      padding: 0.45rem 1rem;
  }
  .stButton > button:hover {
      transform: translateY(-1px);
      border-color: rgba(16,185,129,0.5);
      box-shadow: 0 4px 14px rgba(16,185,129,0.15);
  }
  /* Primary button — make the label pop in bright white, regardless of
     Streamlit's default muted foreground color. */
  .stButton > button[kind="primary"],
  .stButton > button[kind="primary"] p,
  .stButton > button[kind="primary"] div,
  .stButton > button[kind="primary"] span {
      color: #ffffff !important;
      text-shadow: 0 1px 2px rgba(0,0,0,0.25);
  }
  .stButton > button[kind="primary"] {
      background: linear-gradient(135deg, #10b981 0%, #059669 100%);
      border: none;
      font-weight: 700 !important;
      letter-spacing: 0.01em;
  }
  .stButton > button[kind="primary"]:hover {
      background: linear-gradient(135deg, #059669 0%, #047857 100%);
      box-shadow: 0 6px 20px rgba(16,185,129,0.3);
      transform: translateY(-1px);
  }
  .stButton > button[kind="primary"]:active {
      transform: translateY(0);
  }

  /* --- Metric cards --------------------------------------------------- */
  [data-testid="stMetric"] {
      background: rgba(255,255,255,0.025);
      border: 1px solid rgba(255,255,255,0.06);
      padding: 1rem 1.25rem;
      border-radius: 12px;
      transition: all 0.18s ease;
  }
  [data-testid="stMetric"]:hover {
      border-color: rgba(16,185,129,0.3);
      background: rgba(16,185,129,0.03);
  }
  [data-testid="stMetricLabel"] {
      color: #9ca3af !important;
      font-size: 0.82rem !important;
      font-weight: 500 !important;
      letter-spacing: 0.02em;
      text-transform: uppercase;
  }
  [data-testid="stMetricValue"] {
      font-family: 'Space Grotesk', sans-serif !important;
      font-weight: 700 !important;
      font-size: 1.85rem !important;
  }

  /* --- Sidebar: "mission control" panel ------------------------------ */
  [data-testid="stSidebar"] {
      background:
          radial-gradient(ellipse 80% 50% at 50% 0%, rgba(16,185,129,0.12) 0%, transparent 60%),
          linear-gradient(180deg, #0c1324 0%, #080c18 100%) !important;
      border-right: 1px solid rgba(16,185,129,0.12);
      box-shadow: inset -1px 0 0 rgba(255,255,255,0.02),
                  4px 0 24px rgba(0,0,0,0.4);
  }

  /* Force scroll on the sidebar's inner content container. Streamlit's
     default behavior got broken by one of our styling rules creating an
     extra stacking context; this belt-and-suspenders rule targets every
     inner wrapper Streamlit might use across versions so wheel/touch
     scrolling works regardless of DOM version. The scrollbar itself is
     hidden (scrollbar-width: none + webkit pseudo) so the sidebar looks
     clean — scroll still works via wheel, touch, and keyboard. */
  section[data-testid="stSidebar"] > div,
  section[data-testid="stSidebar"] > div > div,
  section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
      overflow-y: auto !important;
      overflow-x: hidden !important;
      max-height: 100vh;
      -webkit-overflow-scrolling: touch;
      scrollbar-width: none;      /* Firefox */
      -ms-overflow-style: none;    /* IE / old Edge */
  }
  /* Emerald accent stripe at the very top of the sidebar. */
  [data-testid="stSidebar"]::before {
      content: "";
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 2px;
      background: linear-gradient(90deg, transparent 0%, #10b981 30%, #3b82f6 70%, transparent 100%);
      opacity: 0.6;
      pointer-events: none;
      z-index: 10;
  }

  /* Sidebar title — big, gradient, spaced. */
  [data-testid="stSidebar"] h1 {
      font-family: 'Space Grotesk', sans-serif !important;
      font-size: 1.55rem !important;
      font-weight: 700 !important;
      letter-spacing: -0.01em;
      margin-bottom: 0.5rem !important;
      background: linear-gradient(135deg, #10b981 0%, #60a5fa 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      border-bottom: none !important;
      padding-bottom: 0 !important;
  }

  /* Section headings inside the sidebar: uppercase, small, emerald tick. */
  [data-testid="stSidebar"] h3 {
      font-family: 'Space Grotesk', sans-serif !important;
      font-size: 0.78rem !important;
      font-weight: 600 !important;
      letter-spacing: 0.12em !important;
      text-transform: uppercase;
      color: #9ca3af !important;
      margin-top: 1.6rem !important;
      margin-bottom: 0.6rem !important;
      padding: 0 0 0 0.75rem !important;
      border-bottom: none !important;
      position: relative;
  }
  [data-testid="stSidebar"] h3::before {
      content: "";
      position: absolute;
      left: 0; top: 50%;
      width: 3px; height: 14px;
      transform: translateY(-50%);
      background: linear-gradient(180deg, #10b981, #3b82f6);
      border-radius: 2px;
  }

  /* Hide the noisy default <hr> rule; we use h3 markers instead. */
  [data-testid="stSidebar"] hr {
      border: none !important;
      height: 1px !important;
      background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent) !important;
      margin: 1.25rem 0 !important;
  }

  /* Inputs in the sidebar — polished dark panels with emerald focus. */
  [data-testid="stSidebar"] [data-baseweb="select"] > div,
  [data-testid="stSidebar"] input,
  [data-testid="stSidebar"] textarea {
      background: rgba(255,255,255,0.03) !important;
      border: 1px solid rgba(255,255,255,0.08) !important;
      border-radius: 8px !important;
      transition: all 0.15s ease;
      color: #e5e7eb !important;
  }
  [data-testid="stSidebar"] [data-baseweb="select"] > div:hover,
  [data-testid="stSidebar"] input:hover,
  [data-testid="stSidebar"] textarea:hover {
      border-color: rgba(16,185,129,0.3) !important;
      background: rgba(16,185,129,0.04) !important;
  }
  [data-testid="stSidebar"] [data-baseweb="select"] > div:focus-within,
  [data-testid="stSidebar"] input:focus,
  [data-testid="stSidebar"] textarea:focus {
      border-color: rgba(16,185,129,0.55) !important;
      box-shadow: 0 0 0 3px rgba(16,185,129,0.12);
      background: rgba(16,185,129,0.05) !important;
  }

  /* Input labels in the sidebar — small, muted, uppercase-ish. */
  [data-testid="stSidebar"] label {
      font-size: 0.8rem !important;
      font-weight: 500 !important;
      color: #cbd5e1 !important;
      letter-spacing: 0.02em;
  }

  /* Caption text (st.caption) — slightly brighter than default for legibility. */
  [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
  [data-testid="stSidebar"] .stCaption,
  [data-testid="stSidebar"] small {
      color: #94a3b8 !important;
      line-height: 1.5 !important;
  }

  /* Radio & checkbox groups — card-like hit targets. */
  [data-testid="stSidebar"] [role="radiogroup"] > label,
  [data-testid="stSidebar"] [data-baseweb="checkbox"] {
      padding: 0.35rem 0.6rem !important;
      border-radius: 7px;
      transition: background 0.15s;
  }
  [data-testid="stSidebar"] [role="radiogroup"] > label:hover,
  [data-testid="stSidebar"] [data-baseweb="checkbox"]:hover {
      background: rgba(16,185,129,0.05);
  }

  /* Sidebar buttons — inherit the main button style but force full width. */
  [data-testid="stSidebar"] .stButton > button {
      width: 100%;
      background: rgba(16,185,129,0.08);
      border: 1px solid rgba(16,185,129,0.25);
      color: #a7f3d0;
      font-weight: 500;
  }
  [data-testid="stSidebar"] .stButton > button:hover {
      background: rgba(16,185,129,0.15);
      border-color: rgba(16,185,129,0.5);
      color: #ffffff;
      transform: translateY(-1px);
      box-shadow: 0 4px 14px rgba(16,185,129,0.2);
  }

  /* Expanders inside the sidebar (e.g. "How to use") - subtle panel. */
  [data-testid="stSidebar"] [data-testid="stExpander"] {
      border: 1px solid rgba(255,255,255,0.06);
      background: rgba(0,0,0,0.25);
      border-radius: 9px;
  }

  /* Hide the sidebar scrollbar in Chrome/Safari/Edge. Scroll still works
     — only the visual track/thumb are removed. */
  [data-testid="stSidebar"] ::-webkit-scrollbar,
  section[data-testid="stSidebar"] > div::-webkit-scrollbar,
  section[data-testid="stSidebar"] > div > div::-webkit-scrollbar {
      width: 0 !important;
      height: 0 !important;
      background: transparent !important;
      display: none !important;
  }

  /* --- Expanders ------------------------------------------------------ */
  [data-testid="stExpander"] {
      border-radius: 10px;
      border: 1px solid rgba(255,255,255,0.06);
      background: rgba(255,255,255,0.015);
  }

  /* --- Tabs ----------------------------------------------------------- */
  .stTabs [data-baseweb="tab-list"] {
      gap: 6px;
      border-bottom: 1px solid rgba(255,255,255,0.06);
  }
  .stTabs [data-baseweb="tab"] {
      border-radius: 9px 9px 0 0;
      padding: 0.55rem 1.3rem;
      font-weight: 500;
      background: transparent;
      transition: all 0.15s;
  }
  .stTabs [data-baseweb="tab"]:hover {
      background: rgba(255,255,255,0.03);
  }
  .stTabs [aria-selected="true"] {
      color: #10b981 !important;
      background: rgba(16,185,129,0.08);
  }

  /* --- Alerts / status boxes ----------------------------------------- */
  [data-testid="stAlert"] {
      border-radius: 10px;
      border-left-width: 3px;
  }

  /* --- Hide Streamlit default footer --------------------------------- */
  footer { visibility: hidden; }

  /* --- Smoother scrollbar -------------------------------------------- */
  ::-webkit-scrollbar { width: 10px; height: 10px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb {
      background: rgba(255,255,255,0.1);
      border-radius: 5px;
  }
  ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }
</style>
""",
    unsafe_allow_html=True,
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
# Guard against an out-of-range value from older session state. The radio
# itself reads/writes st.session_state.data_source via `key=`, so we don't
# need (and must not have) `index=` + a post-render write — that double-source
# pattern was causing a "first click reverts" bug.
if st.session_state.data_source not in _data_source_keys:
    st.session_state.data_source = "iolulc"
data_source = st.sidebar.radio(
    "Choose a land-cover product",
    _data_source_keys,
    format_func=lambda k: DATA_SOURCE_OPTIONS[k],
    key="data_source",
    help=(
        "IO-LULC: Impact Observatory annual 10m land cover via Planetary Computer "
        "— no auth, but ~12-month lag and once-per-year snapshots.\n\n"
        "Dynamic World: Google's 10m NRT land cover via Earth Engine — updated "
        "every 2-5 days, goes back to 2015. Requires a free Earth Engine "
        "account (one-time `earthengine authenticate`)."
    ),
)

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

    # On deployment EE is preconfigured via secrets/env — the text input
    # and debug button would just clutter the UI for end users, so we hide
    # them. Locally (no env var set) the full debug block is shown.
    _ee_preconfigured = bool(os.environ.get("EARTHENGINE_PROJECT"))

    if _ee_preconfigured:
        st.sidebar.caption("✅ Earth Engine ready.")
    else:
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
# On deployment FIRMS_MAP_KEY is preconfigured via secrets/env. Showing it
# in a text input — even as `type="password"` — still ships the value to the
# browser. Hide the input entirely in that case and just use the default
# silently.
_firms_preconfigured = bool(_firms_key_default)

if show_fires:
    if _firms_preconfigured:
        firms_key_input = _firms_key_default
        st.sidebar.caption("🔥 FIRMS key ready.")
    else:
        firms_key_input = st.sidebar.text_input(
            "FIRMS MAP_KEY",
            value="",
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

HERO_HTML = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { background: transparent; overflow: hidden; height: 100%; }
  body {
      font-family: 'Inter', system-ui, sans-serif;
      color: #e5e7eb;
  }

  #hero-wrap {
      position: relative;
      width: 100%;
      height: 260px;
      border-radius: 16px;
      overflow: hidden;
      background:
          radial-gradient(ellipse 70% 60% at 15% 20%, rgba(16,185,129,0.18) 0%, transparent 60%),
          radial-gradient(ellipse 60% 50% at 85% 75%, rgba(59,130,246,0.18) 0%, transparent 60%),
          linear-gradient(135deg, #060912 0%, #0b1022 60%, #0a0e1a 100%);
      border: 1px solid rgba(255,255,255,0.08);
      box-shadow: 0 14px 40px rgba(0,0,0,0.4);
  }

  /* Animated grid overlay — subtle lat/lon vibe. */
  #hero-wrap::before {
      content: "";
      position: absolute;
      inset: 0;
      background-image:
          linear-gradient(rgba(16,185,129,0.04) 1px, transparent 1px),
          linear-gradient(90deg, rgba(16,185,129,0.04) 1px, transparent 1px);
      background-size: 40px 40px;
      mask-image: radial-gradient(ellipse 80% 80% at 50% 50%, black 30%, transparent 75%);
      -webkit-mask-image: radial-gradient(ellipse 80% 80% at 50% 50%, black 30%, transparent 75%);
      animation: gridDrift 40s linear infinite;
      pointer-events: none;
  }
  @keyframes gridDrift {
      from { background-position: 0 0, 0 0; }
      to   { background-position: 40px 40px, 40px 40px; }
  }

  /* Scan line that sweeps down the hero every 6s. */
  #hero-wrap::after {
      content: "";
      position: absolute;
      left: 0; right: 0; top: 0;
      height: 60%;
      background: linear-gradient(180deg, transparent 0%, rgba(16,185,129,0.06) 45%, rgba(52,211,153,0.15) 50%, rgba(16,185,129,0.06) 55%, transparent 100%);
      animation: scan 6s ease-in-out infinite;
      pointer-events: none;
      mix-blend-mode: screen;
  }
  @keyframes scan {
      0%   { transform: translateY(-60%); opacity: 0; }
      10%  { opacity: 1; }
      90%  { opacity: 1; }
      100% { transform: translateY(160%); opacity: 0; }
  }

  #hero-canvas {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      z-index: 1;
  }

  #hero-content {
      position: absolute;
      inset: 0;
      z-index: 2;
      display: grid;
      grid-template-columns: 1fr;
      align-items: center;
      padding: 1.4rem 1.75rem 1.4rem 1.75rem;
      pointer-events: none;
  }
  #hero-text { pointer-events: auto; max-width: 58%; }

  .pill {
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      background: rgba(16,185,129,0.12);
      color: #6ee7b7;
      border: 1px solid rgba(16,185,129,0.3);
      padding: 0.25rem 0.7rem;
      border-radius: 999px;
      font-size: 0.7rem;
      font-weight: 500;
      margin-bottom: 0.65rem;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      backdrop-filter: blur(6px);
  }
  .pill .dot {
      width: 6px; height: 6px;
      background: #10b981;
      border-radius: 50%;
      box-shadow: 0 0 10px #10b981;
      animation: pulse 1.8s ease-in-out infinite;
  }
  @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50%      { opacity: 0.4; transform: scale(1.3); }
  }

  h1 {
      font-family: 'Space Grotesk', sans-serif;
      font-weight: 700;
      font-size: 2.4rem;
      line-height: 1.05;
      letter-spacing: -0.02em;
      background: linear-gradient(120deg, #6ee7b7 0%, #60a5fa 45%, #a78bfa 90%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      filter: drop-shadow(0 4px 20px rgba(16,185,129,0.15));
  }
  h1 .glitch {
      display: inline-block;
      animation: float 6s ease-in-out infinite;
  }
  @keyframes float {
      0%, 100% { transform: translateY(0); }
      50%      { transform: translateY(-3px); }
  }

  #hero-text p.tagline {
      font-size: 0.92rem;
      color: #cbd5e1;
      margin-top: 0.55rem;
      max-width: 560px;
      line-height: 1.5;
      font-weight: 400;
  }

  @media (max-width: 760px) {
      #hero-text { max-width: 100%; }
      #hero-content { padding: 1.2rem; }
      h1 { font-size: 1.9rem; }
  }
</style>
</head>
<body>
<div id="hero-wrap">
  <canvas id="hero-canvas"></canvas>
  <div id="hero-content">
    <div id="hero-text">
      <span class="pill"><span class="dot"></span>Live satellite change detection</span>
      <h1><span class="glitch">Earth</span> <span class="glitch" style="animation-delay:-2s">Time</span> <span class="glitch" style="animation-delay:-4s">Machine</span></h1>
      <p class="tagline">Search or draw any region on Earth, pick two dates, and get a pixel-level change map with transition statistics and a downloadable report.</p>
    </div>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
(function() {
  const canvas = document.getElementById('hero-canvas');
  const wrap = document.getElementById('hero-wrap');

  const scene = new THREE.Scene();

  // Orthographic camera: no perspective distortion, so a sphere always
  // renders as a perfect circle regardless of where it sits in the view.
  // Crucial here because our hero is very wide relative to its height — a
  // perspective camera would stretch off-axis objects into ellipses.
  const VIEW_HEIGHT = 6;  // world-units tall the view shows
  let aspect = wrap.clientWidth / wrap.clientHeight;
  const camera = new THREE.OrthographicCamera(
      -VIEW_HEIGHT * aspect / 2,  VIEW_HEIGHT * aspect / 2,
       VIEW_HEIGHT / 2,          -VIEW_HEIGHT / 2,
      -200, 200
  );
  camera.position.set(0, 0, 10);
  camera.lookAt(0, 0, 0);

  const renderer = new THREE.WebGLRenderer({ canvas: canvas, alpha: true, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(wrap.clientWidth, wrap.clientHeight);

  // --- Starfield (3 layers at different depths for parallax) ---
  // With orthographic cameras `size` is in screen pixels, so we use larger
  // numeric values and set sizeAttenuation to false.
  function makeStarLayer(count, spread, size, color) {
      const geo = new THREE.BufferGeometry();
      const pos = new Float32Array(count * 3);
      for (let i = 0; i < count; i++) {
          pos[i*3]   = (Math.random() - 0.5) * spread;
          pos[i*3+1] = (Math.random() - 0.5) * spread;
          pos[i*3+2] = (Math.random() - 0.5) * spread;
      }
      geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      const mat = new THREE.PointsMaterial({
          color: color, size: size, transparent: true, opacity: 0.85,
          sizeAttenuation: false,
      });
      return new THREE.Points(geo, mat);
  }
  const stars1 = makeStarLayer(900, 300, 1.8, 0xffffff);
  const stars2 = makeStarLayer(500, 200, 1.3, 0xa7f3d0);
  const stars3 = makeStarLayer(250, 150, 1.0, 0x93c5fd);
  scene.add(stars1); scene.add(stars2); scene.add(stars3);

  // --- Globe: earth texture + emerald wireframe grid + atmospheric glow ---
  const globeGroup = new THREE.Group();
  // Position is recomputed from the orthographic frustum in positionGlobe()
  // so the globe always sits in the right-center regardless of aspect.
  globeGroup.position.set(3.0, -0.1, 0);
  scene.add(globeGroup);

  // Inner sphere: start white so the earth texture displays with true colors
  // when it loads. We start a texture load immediately and swap in the map
  // when ready; if the load fails, we fall back to a dark-blue ocean color.
  const innerGeo = new THREE.SphereGeometry(1.65, 64, 64);
  const innerMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
  const innerSphere = new THREE.Mesh(innerGeo, innerMat);
  globeGroup.add(innerSphere);

  const loader = new THREE.TextureLoader();
  loader.setCrossOrigin('anonymous');
  // jsdelivr mirrors the three.js repo with proper CORS headers, so the
  // texture loads cleanly from the user's browser.
  loader.load(
      'https://cdn.jsdelivr.net/gh/mrdoob/three.js@r128/examples/textures/planets/earth_atmos_2048.jpg',
      (tex) => {
          innerMat.map = tex;
          innerMat.needsUpdate = true;
      },
      undefined,
      (err) => {
          console.warn('Earth texture failed to load; falling back.', err);
          innerMat.color.set(0x0a1830);
      }
  );

  // Wireframe layer — subtle emerald latitude/longitude grid on top of the
  // earth texture. Lower opacity than before so the continents show through.
  const wireGeo = new THREE.SphereGeometry(1.665, 36, 24);
  const wireMat = new THREE.MeshBasicMaterial({
      color: 0x10b981, wireframe: true, transparent: true, opacity: 0.22,
  });
  const wireSphere = new THREE.Mesh(wireGeo, wireMat);
  globeGroup.add(wireSphere);

  // Atmosphere — a slightly larger sphere with a fake "fresnel" via additive blending.
  const atmGeo = new THREE.SphereGeometry(1.95, 48, 48);
  const atmMat = new THREE.ShaderMaterial({
      uniforms: {},
      vertexShader: `
          varying vec3 vNormal;
          void main() {
              vNormal = normalize(normalMatrix * normal);
              gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          }`,
      fragmentShader: `
          varying vec3 vNormal;
          void main() {
              float intensity = pow(0.65 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 2.2);
              gl_FragColor = vec4(0.22, 0.72, 0.95, 1.0) * intensity;
          }`,
      side: THREE.BackSide,
      blending: THREE.AdditiveBlending,
      transparent: true,
  });
  const atmosphere = new THREE.Mesh(atmGeo, atmMat);
  globeGroup.add(atmosphere);

  // --- Orbit frame ---
  // Both the orbit ring and the satellite live inside this group. Since
  // Three.js applies the same world matrix to every child, putting them
  // in a shared tilted parent guarantees the dot stays exactly on the
  // line at every angle. Previously the ring used Three.js's Euler tilt
  // (composed as Rx·Ry·Rz under the default 'XYZ' order, i.e. Z applied
  // first) while the satellite's position was hand-rolled with X applied
  // first, so they traced slightly different ellipses — visible as a
  // small offset between dot and line at the front of the orbit.
  const orbitFrame = new THREE.Group();
  orbitFrame.rotation.x = Math.PI / 2 - 0.45;
  orbitFrame.rotation.z = 0.25;
  globeGroup.add(orbitFrame);

  // --- Orbit ring (flat circle in XZ; tilt comes from orbitFrame) ---
  const orbitRadius = 2.5;
  const orbitCurve = new THREE.EllipseCurve(0, 0, orbitRadius, orbitRadius, 0, 2 * Math.PI, false, 0);
  const orbitPoints = orbitCurve.getPoints(128).map(p => new THREE.Vector3(p.x, 0, p.y));
  const orbitGeo = new THREE.BufferGeometry().setFromPoints(orbitPoints);
  const orbitMat = new THREE.LineBasicMaterial({ color: 0x3b82f6, transparent: true, opacity: 0.4 });
  const orbitLine = new THREE.Line(orbitGeo, orbitMat);
  orbitFrame.add(orbitLine);

  // --- Satellite (glowing point with trailing "signal" tail) ---
  const satGroup = new THREE.Group();
  orbitFrame.add(satGroup);

  const satGeo = new THREE.SphereGeometry(0.07, 12, 12);
  const satMat = new THREE.MeshBasicMaterial({ color: 0x34d399 });
  const sat = new THREE.Mesh(satGeo, satMat);
  satGroup.add(sat);

  // Satellite glow (sprite-ish halo).
  const haloGeo = new THREE.SphereGeometry(0.16, 12, 12);
  const haloMat = new THREE.MeshBasicMaterial({
      color: 0x10b981, transparent: true, opacity: 0.35, blending: THREE.AdditiveBlending,
  });
  const halo = new THREE.Mesh(haloGeo, haloMat);
  satGroup.add(halo);

  // Trailing signal cone down to earth.
  const coneGeo = new THREE.ConeGeometry(0.35, 1.4, 20, 1, true);
  const coneMat = new THREE.MeshBasicMaterial({
      color: 0x10b981, transparent: true, opacity: 0.12, side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending, depthWrite: false,
  });
  const cone = new THREE.Mesh(coneGeo, coneMat);
  cone.position.y = -0.8;
  satGroup.add(cone);

  // --- Mouse parallax ---
  let targetX = 0, targetY = 0;
  wrap.addEventListener('mousemove', (e) => {
      const rect = wrap.getBoundingClientRect();
      targetX = ((e.clientX - rect.left) / rect.width - 0.5) * 0.3;
      targetY = ((e.clientY - rect.top)  / rect.height - 0.5) * 0.2;
  });

  // --- Animation loop ---
  let t = 0;
  function animate() {
      requestAnimationFrame(animate);
      t += 0.01;

      // Globe rotates on its Y axis; the Earth texture on the inner sphere
      // spins with it so the continents drift across the face.
      globeGroup.rotation.y += 0.0035;

      // Parallax stars.
      stars1.rotation.y += 0.0002;
      stars2.rotation.y += 0.0004;
      stars3.rotation.y += 0.0006;
      stars1.rotation.x += 0.0001;

      // Satellite position: a flat circle in orbitFrame's local XZ plane.
      // The tilt is handled by orbitFrame's rotation, so we don't apply
      // any tilt math here — Three.js composes the parent matrices for us
      // and the dot lands exactly on the orbit line at every angle.
      const a = t * 0.7;
      satGroup.position.set(orbitRadius * Math.cos(a), 0, orbitRadius * Math.sin(a));
      // Point cone toward the globe center (world coords).
      satGroup.lookAt(globeGroup.position);
      satGroup.rotateX(Math.PI / 2);

      // Camera parallax.
      camera.position.x += (targetX * 1.5 - camera.position.x) * 0.04;
      camera.position.y += (-targetY * 1.2 - camera.position.y) * 0.04;
      camera.lookAt(0, 0, 0);

      renderer.render(scene, camera);
  }
  animate();

  // Keep the globe anchored on the right side of the hero but close enough
  // to center that the atmosphere halo fully fits. Scale relative to the
  // view height so the globe fills ~90% of it vertically.
  function positionGlobe() {
      const viewHalfW = VIEW_HEIGHT * aspect / 2;
      // Sit the globe center near the midpoint between the right edge of the
      // tagline copy (~58% from left of hero) and the right edge of the box.
      // In ortho world coords this is ~0.58 of the right half-width.
      globeGroup.position.x = viewHalfW * 0.58;
      globeGroup.position.y = -0.05;
      // Target atmosphere diameter = 90% of view height.
      const atmosphereDiameter = 1.95 * 2;
      globeGroup.scale.setScalar((VIEW_HEIGHT * 0.9) / atmosphereDiameter);
  }
  positionGlobe();

  // Handle resize — reset ortho frustum + renderer + globe placement.
  const ro = new ResizeObserver(() => {
      const w = wrap.clientWidth, h = wrap.clientHeight;
      renderer.setSize(w, h);
      aspect = w / h;
      camera.left   = -VIEW_HEIGHT * aspect / 2;
      camera.right  =  VIEW_HEIGHT * aspect / 2;
      camera.top    =  VIEW_HEIGHT / 2;
      camera.bottom = -VIEW_HEIGHT / 2;
      camera.updateProjectionMatrix();
      positionGlobe();
  });
  ro.observe(wrap);
})();
</script>
</body>
</html>
"""

# `st.components.v1.html` is slated for removal (2026-06-01). The
# successor `st.iframe` doesn't yet accept `srcdoc` in all Streamlit
# versions, so we encode the Three.js hero as a base64 data URL and hand
# that to `st.iframe` as `src`. Same sandbox / isolation semantics —
# scripts execute normally inside the iframe. The iframe has an opaque
# origin (can't touch window.parent), which is fine here because the
# hero is purely self-contained.
_hero_data_url = (
    "data:text/html;charset=utf-8;base64,"
    + base64.b64encode(HERO_HTML.encode("utf-8")).decode("ascii")
)
st.iframe(src=_hero_data_url, height=285)

# Compact stats strip — carries the info that used to live inside the hero
# (resolution, coverage, sources, status) but occupies one slim horizontal
# line instead of stealing ~200px of vertical space from the map below.
st.markdown(
    """
    <div class="stats-strip">
      <div class="stat-chip"><span class="k">Resolution</span><span class="v">10&nbsp;m&nbsp;/&nbsp;pixel</span></div>
      <div class="stat-chip"><span class="k">Coverage</span><span class="v">Global</span></div>
      <div class="stat-chip"><span class="k">Sources</span><span class="v">IO-LULC&nbsp;·&nbsp;Dynamic&nbsp;World</span></div>
      <div class="stat-chip"><span class="k">Status</span><span class="v"><span class="live-dot"></span>Online</span></div>
    </div>
    """,
    unsafe_allow_html=True,
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

# Base layers: Satellite is the DEFAULT visible basemap (matches the
# space/Earth aesthetic of the hero). Streets is available as a toggle in the
# top-right LayerControl. In folium, the first TileLayer added with
# overlay=False becomes the active base on load — so satellite goes first.
folium.TileLayer(
    tiles=(
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}"
    ),
    attr="Esri, Maxar, Earthstar Geographics, and the GIS User Community",
    name="Satellite (Esri World Imagery)",
    overlay=False,
    control=True,
    show=True,
).add_to(m)

folium.TileLayer(
    tiles="OpenStreetMap",
    name="Streets (OpenStreetMap)",
    overlay=False,
    control=True,
    show=False,
).add_to(m)

# Labels overlay sits on top of the satellite imagery so place names remain
# readable. It's an overlay (not a base), so it can stay enabled regardless
# of which base layer the user picks.
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
    show=True,
).add_to(m)

# Place search (OpenStreetMap Nominatim). Adds a 🔍 button in the top-left
# of the map; users type a place name, pick a result, and the map pans/zooms
# to it. Then they can draw the rectangle with the Draw tool below.
Geocoder(
    collapsed=True,
    position="topleft",
    add_marker=False,        # no persistent pin; just fly-to
    placeholder="Search a place...",
    # Live-suggestion options (passed through to Leaflet-Control-Geocoder).
    # As-you-type dropdown kicks in after `suggestMinLength` chars, debounced
    # by `suggestTimeout` ms. `defaultMarkGeocode=False` stops the control
    # from dropping a marker when a suggestion is picked.
    suggestMinLength=2,
    suggestTimeout=200,
    defaultMarkGeocode=False,
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
        - **🔍 Search** (top-left): type a place name (e.g. "Amazon", "Dubai",
          "Bengaluru") to fly the map to it, then draw the rectangle.
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
