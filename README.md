# 🌎 Earth Time Machine

Detect land-cover change anywhere on Earth between two years, using 10 m
Sentinel-derived land cover maps. Built as a reproducible pipeline plus a live
Streamlit UI where the user draws any region on a world map, picks two dates,
and gets back a change map, statistics, and an auto-generated report.

**No training required.** The pipeline uses a pre-computed global land-cover
product (Impact Observatory / Esri 10m Annual LULC, hosted on Microsoft
Planetary Computer) and adds the change-detection + visualization layer on top.

---

## What it does

1. **Data**: pulls 10m annual land cover from the `io-lulc-annual-v02`
   collection on Microsoft Planetary Computer — 9 classes (Water, Trees, Crops,
   Built Area, Bare Ground, Rangeland, …), global, 2017–2023.
2. **Classification**: already done. We operate on the pre-classified rasters.
3. **Change detection**: pixel-wise class comparison between two years,
   transition matrix, per-class area change, notable-transition summaries
   (forest loss, urban sprawl, agricultural expansion, etc.).
4. **Context**:
   - Optional Sentinel-2 L2A RGB previews (median cloud-free composites).
   - Optional NASA FIRMS active-fire detections overlaid for the analysis period.
5. **Outputs**: maps, transition bar-charts, a markdown report, and an
   interactive Streamlit web app.

## Featured case studies (ship with the app as presets)

| Preset | Period | Expected pattern |
|-------|--------|-------------------|
| Rondônia deforestation (Brazil) | 2018 → 2023 | Forest → Cropland / Rangeland |
| Dubai urban growth | 2017 → 2023 | Bare / Water → Built Area |
| Bengaluru sprawl (India) | 2017 → 2023 | Crops / Rangeland → Built Area |
| California Camp Fire area | 2018 → 2022 | Forest → Bare Ground / Rangeland |
| Borneo peatland (Indonesia) | 2017 → 2023 | Forest / Flooded Veg → Crops |

---

## Quick start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### Command-line (no UI)

Run a single case study end-to-end and write PNGs + a markdown report to
`outputs/amazon/`:

```bash
python examples/amazon_case_study.py
```

Or run on any custom bounding box:

```bash
python examples/cli.py \
    --bbox -63.2,-10.7,-62.5,-10.1 \
    --before 2018 --after 2023 \
    --name "Rondônia" \
    --out outputs/rondonia
```

### Notebook walkthrough

```bash
jupyter notebook notebooks/demo.ipynb
```

---

## Optional: fire overlay (NASA FIRMS)

Enable "Fetch NASA FIRMS fire detections" in the sidebar.

You'll need a free FIRMS MAP_KEY (takes ~30 seconds):

1. Go to https://firms.modaps.eosdis.nasa.gov/api/area/
2. Request a MAP_KEY with your email
3. Export it:

   ```bash
   export FIRMS_MAP_KEY="your-key-here"
   ```

Then re-run the Streamlit app. The "🔥 Fires" tab will show fire detections
within the AOI for the analysis period. Useful for attributing detected
forest losses to specific fire events.

---

## Optional: Dynamic World (near-real-time land cover)

The app supports Google Dynamic World as a second data source, giving
~weekly-resolution land cover from 2017 to today (vs. IO-LULC's yearly
product with a ~12-month publish lag).

Access requires a free Google Cloud project with the Earth Engine API
enabled, and a Noncommercial / Community registration on Earth Engine.
Set the project ID in the sidebar (or export `EARTHENGINE_PROJECT`).
For headless / CI / deployed use, set `GOOGLE_APPLICATION_CREDENTIALS`
to a GCP service-account JSON key (the service account needs the
`Earth Engine Resource Viewer` and `Service Usage Consumer` IAM roles).

---

## Deploy to Streamlit Community Cloud (free)

1. Push the repo to GitHub (the `.gitignore` already blocks service-account
   JSONs and `.streamlit/secrets.toml` — double-check with
   `git status` before committing).
2. Go to https://share.streamlit.io → **New app** → select your repo,
   branch, and `app.py`.
3. Under **Advanced settings → Secrets**, paste the contents of
   `.streamlit/secrets.toml.example` with real values filled in:
   - `EARTHENGINE_PROJECT` — your GCP project ID
   - `FIRMS_MAP_KEY` — NASA FIRMS API key (optional; only needed for fires)
   - `[GCP_SERVICE_ACCOUNT_JSON]` — paste every field from your
     service-account JSON file. Keep the literal `\n` inside
     `private_key` as-is.
4. Click Deploy. First build takes 2–4 minutes.

On boot, `app.py` loads those secrets and writes the service-account JSON
to a tempfile, so the rest of the code behaves identically to running
locally with `export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json`.

---

## Project structure

```
satellite-change-detection/
├── app.py                        # Streamlit app (entry point)
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── data.py                   # Planetary Computer queries (LULC + Sentinel-2)
│   ├── change_detection.py       # Core change-detection logic
│   ├── overlays.py               # NASA FIRMS fire data
│   └── viz.py                    # Plotting + folium overlays
├── examples/
│   ├── amazon_case_study.py      # Featured Rondônia case study
│   └── cli.py                    # Generic CLI over any bbox + year pair
└── notebooks/
    └── demo.ipynb                # Step-by-step walkthrough
```

---

## How it works, in more detail

### Land cover source

Impact Observatory's IO-LULC is a 10 m, 9-class global land-cover map produced
annually from Sentinel-2 imagery via a deep-learning classifier. It's the
output of a trained model, just not one we trained. Each pixel carries an
integer class code:

| Code | Class |
|-----:|-------|
| 1 | Water |
| 2 | Trees |
| 4 | Flooded Vegetation |
| 5 | Crops |
| 7 | Built Area |
| 8 | Bare Ground |
| 9 | Snow/Ice |
| 10 | Clouds |
| 11 | Rangeland |

### Change detection

Given two class maps for the same AOI at years `y1 < y2`:

1. Reproject / align to a common 10 m grid (EPSG:4326, same pixel centers).
2. Compute a pixel-wise change mask: `(before != after) & (before != 0) & (after != 0)`.
3. Build a transition matrix by counting all `(from_class, to_class)` pairs
   across the AOI; multiply counts by per-pixel hectares to get area values.
4. Aggregate into:
   - per-class totals before / after
   - top-N largest transitions
   - a fixed list of "notable" transitions (forest loss, urban sprawl, etc.)
5. Highlight forest-loss transitions (`Trees → anything`) separately on the
   change map for deforestation use cases.

Per-pixel area is computed using the mid-latitude approximation in EPSG:4326
(`dlat_m = dlat * 111,320`, `dlon_m = dlon * 111,320 * cos(lat)`), which is
accurate to < 0.3% for AOIs < ~5° of latitude.

### Why not a custom-trained model?

Three reasons — the same reasons this is a nice resume project:

- **Dataset bottleneck**: high-quality labeled change-detection datasets are
  rare and small (OSCD, LEVIR-CD are the standard ones). Operating on a
  globally pre-computed product sidesteps this entirely.
- **Compute bottleneck**: training on the required resolution globally costs
  thousands of GPU-hours. Inference over a user-selected AOI is seconds.
- **Real-world practice**: most production geospatial ML teams don't train
  their own land-cover models either — they orchestrate pre-trained products.

A natural extension is to layer a geospatial foundation model (IBM / NASA
**Prithvi**, on HuggingFace) for AOIs where the baseline IO-LULC disagrees
with your eye — useful for a capstone upgrade.

---

## Limitations

- **10 m resolution**: detects changes of ≥ ~500 m² reliably; individual
  buildings may or may not show up depending on alignment.
- **Seasonal variation**: the IO-LULC product is annual, so intra-year changes
  (e.g., harvested cropland mid-season) aren't captured.
- **Coverage gaps**: a handful of cloud-locked regions may have lower-quality
  classifications in specific years.
- **No sub-class changes**: the 9-class scheme can't distinguish, e.g.,
  different forest types. For finer thematic detail, use ESA WorldCover
  (11 classes) or Dynamic World (via Earth Engine).

---

## Extensions (worth talking about in interviews)

1. **Multi-temporal trajectories**. Fetch N ≥ 5 years and plot class-area
   time-series, animated GIFs of the change map, Sankey diagrams of
   transitions.
2. **Causal attribution**. Already wired for FIRMS fires. Add Global Forest
   Watch mining/palm-oil concessions, OpenStreetMap road-network diffs, rainfall
   anomalies, etc., to explain *why* changes occurred.
3. **Foundation-model classification**. Run NASA / IBM Prithvi on Sentinel-2
   imagery directly for specific AOIs and compare with IO-LULC. See
   https://huggingface.co/ibm-nasa-geospatial
4. **Uncertainty**. Compute bootstrap-style confidence bands by running the
   pipeline on several nearby dates and reporting the agreement.
5. **Natural-language query**. Wrap the analyze function in a small LLM
   prompt that parses "show me deforestation in Borneo between 2018 and 2023"
   into `(bbox, y1, y2)`. Tiny token cost, massive demo appeal.

---

## Attribution

- Impact Observatory / Esri LULC — © Impact Observatory, Microsoft, and Esri,
  under CC BY 4.0. Hosted by Microsoft Planetary Computer.
- Sentinel-2 — Copernicus data, European Union / ESA.
- NASA FIRMS — MODIS/VIIRS active fires, public domain.
- Base maps — OpenStreetMap contributors, © CartoDB.

## License

MIT.
