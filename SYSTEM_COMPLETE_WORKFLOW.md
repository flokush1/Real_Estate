# NCR Real Estate AI System — Complete Workflow & Architecture

**Version:** May 2026  
**Geography:** Delhi NCR — Delhi · Noida · Gurgaon · Faridabad · Ghaziabad · Greater Noida  
**Segments:** Builder Floor (BF) · Apartment (APT) · Plot/Land

---

## Table of Contents

1. [System Overview — What We Have Built](#1-system-overview)
2. [Codebase Map](#2-codebase-map)
3. [Data Sources and What Each Contains](#3-data-sources)
4. [Pipeline 1 — BF/APT Training Pipeline](#4-pipeline-1--bfapt-training-pipeline)
5. [Pipeline 2 — Plot Training Pipeline](#5-pipeline-2--plot-training-pipeline)
6. [Pipeline 3 — Forecast Intelligence Pipeline](#6-pipeline-3--forecast-intelligence-pipeline)
7. [API Layer — All Endpoints](#7-api-layer)
8. [Frontend — React Application Tabs](#8-frontend)
9. [Complete Data Flow Diagram](#9-complete-data-flow-diagram)
10. [Every Calculation and Estimation Explained](#10-every-calculation-and-estimation-explained)
11. [Artifacts and What They Store](#11-artifacts-and-what-they-store)
12. [What Has Been Done — Chronological Summary](#12-what-has-been-done)

---

## 1. System Overview

The system is a **full-stack real estate intelligence platform** for Delhi NCR. It has three distinct purposes:

| Purpose | What it produces |
|---------|-----------------|
| **Point-in-time valuation** | Estimated fair price (₹/sqft + total) for any property right now |
| **Forecast intelligence** | 20-quarter (5-year) price trajectory per property and locality |
| **Investment analytics** | Buy/Avoid score, ROI%, CAGR%, rental yield, YoY growth chain |

It is deployed as:
- A **FastAPI backend** (`api/main.py`) with 25+ endpoints
- A **React frontend** (`frontend/`) with 5 tabs
- An **offline Python ML package** (`real_estate/`) with ingestion → transformation → training components
- A **Streamlit forecast dashboard** (`fore_app.py`) for standalone analytics

---

## 2. Codebase Map

```
real_estate/                        ← root workspace
│
├── api/
│   └── main.py                     ← ALL FastAPI endpoints + serving logic
│
├── frontend/
│   └── src/
│       ├── App.jsx                 ← React app: 5 tabs, forms, API calls, charts
│       └── App.css                 ← Full design system and component styles
│
├── real_estate/                    ← installable Python package
│   ├── components/
│   │   ├── data_ingestion.py       ← BF/APT: merge raw CSVs
│   │   ├── data_transformation.py  ← BF/APT: feature engineering
│   │   ├── plot_data_ingestion.py  ← Plot: merge raw CSVs
│   │   ├── plot_data_transformation.py  ← Plot: feature engineering
│   │   └── plot_model_trainer.py   ← Plot: train + evaluate model
│   ├── pipeline/
│   │   └── training_pipeline.py    ← Orchestrates all stages
│   ├── utils/
│   │   ├── forecast_intelligence.py  ← Core forecast service (loads, resolves, computes)
│   │   ├── description_parser.py     ← NLP extractors for free-text description
│   │   ├── circle_rate_matcher.py    ← Fuzzy circle-rate lookup
│   │   ├── locality_matcher.py       ← Locality name normalization and fuzzy matching
│   │   └── market_intelligence.py    ← (placeholder)
│   ├── constant/__init__.py         ← All file paths, column names, artifact dirs
│   ├── entity/__init__.py           ← Config + Artifact dataclasses
│   ├── exception/exception.py       ← Custom exception with traceback
│   └── logging/logger.py            ← Structured logger
│
├── inputs/                          ← Forecast input data
│   ├── apartment_with_pi.csv
│   ├── builder_floor_with_pi.csv
│   ├── plot_with_pi.csv
│   ├── rbi_repo_rates.csv
│   ├── gsdp_repo_quarterly.csv
│   ├── repo_homeloan_quarterly.csv
│   ├── builder_mapped_pricing_final.csv
│   └── interpolated_growth_trend_builder/
│       ├── apt/{city}/{locality}.csv
│       ├── builder/{city}/{locality}.csv
│       └── plot/{city}/{locality}.csv
│
├── opt/                             ← Forecast model output artifacts
│   ├── apt/
│   ├── builder_floor/
│   └── plot/
│
├── artifact/                        ← Training pipeline artifacts
│   ├── data_ingestion/
│   ├── data_transformation/
│   ├── plot_ingestion/
│   ├── plot_transformation/
│   └── plot_model_trainer/
│
├── real_estate_data/
│   └── real_estate_data/
│       ├── ho_raw_data.csv           ← HousingOnline raw listings (BF + APT)
│       ├── mb_raw_data.csv           ← MagicBricks raw listings (BF + APT)
│       ├── ho_rent.csv               ← HousingOnline rent listings
│       ├── mb_rent.csv               ← MagicBricks rent listings
│       ├── ncr_colonies.json         ← All city→locality reference index
│       ├── NCR Roads.geojson         ← Road network (NH / SH / MDR geometries)
│       └── circle_rates/             ← Government floor prices by city (JSON)
│
├── notebooks/notebooks/
│   ├── sell/bf/best_bf_random_forest.pkl
│   ├── sell/apt/best_apt_random_forest.pkl
│   ├── sell/plot/best_plot_random_forest.pkl
│   └── rent/bf/, rent/apt/           ← Rent model pkl files
│
├── fore_app.py                       ← Streamlit forecast dashboard
├── app_plot_train.py                 ← Script to trigger plot training pipeline
└── setup.py                         ← Package install
```

---

## 3. Data Sources

### 3.1 Raw Listing Data (BF + APT)

| File | Source | Content |
|------|--------|---------|
| `real_estate_data/ho_raw_data.csv` | HousingOnline | Builder Floor + Apartment sell listings |
| `real_estate_data/mb_raw_data.csv` | MagicBricks | Builder Floor + Apartment sell listings |
| `real_estate_data/ho_rent.csv` | HousingOnline | Rental listings |
| `real_estate_data/mb_rent.csv` | MagicBricks | Rental listings |

**Key raw columns available:**
`id`, `property_type`, `city`, `locality`, `covered_area_value`, `covered_area_unit`, `price_numeric`, `latitude`, `longitude`, `description`, `amenities`, `bhk`, `bathrooms`, `balconies`, `age_of_property`, `furnishing_type`, `facing_direction`, `corner_property`, `gated_plot`, `developer_id`/`developer_uuid`, `agent_type`

Two sources have schema differences that are normalized at ingestion:
- HousingOnline uses `developer_uuid` → renamed to `developer_id`
- MagicBricks uses `user_type` → renamed to `agent_type`

### 3.2 Raw Plot Listing Data

Separate raw CSVs for plot/land listings, merged into `artifact/plot_ingestion/merged/combined_plot.csv`.

### 3.3 Reference Data

| File | Purpose |
|------|---------|
| `real_estate_data/ncr_colonies.json` | Maps each city to its known locality names — used for fuzzy locality matching and imputation |
| `real_estate_data/circle_rates/*.json` | Government minimum floor prices (₹/sqft) by locality and property type for each NCR city |
| `real_estate_data/NCR Roads.geojson` | Road network — MultiLineString/LineString geometries classified as NH / SH / MDR |

### 3.4 Forecast Input Files

| File | Content |
|------|---------|
| `inputs/apartment_with_pi.csv` | ~67,726 apartments — listing price + model (PI) price per sqft + amenity flags |
| `inputs/builder_floor_with_pi.csv` | ~42,098 builder floors — same schema |
| `inputs/plot_with_pi.csv` | Plots — listing + model price, geospatial columns |
| `inputs/interpolated_growth_trend_builder/` | Quarterly locality price panels, smoothed and gap-filled |
| `inputs/rbi_repo_rates.csv` | RBI repo rate by quarter |
| `inputs/gsdp_repo_quarterly.csv` | Delhi NCR state GDP growth (quarterly) |
| `inputs/repo_homeloan_quarterly.csv` | Home loan rate panel |
| `inputs/builder_mapped_pricing_final.csv` | Builder-level pricing benchmarks |

---

## 4. Pipeline 1 — BF/APT Training Pipeline

This is the **offline ML pipeline** for Builder Floor and Apartment models. Triggered via `TrainingPipeline.run_pipeline()`.

### Stage 1: Data Ingestion (`data_ingestion.py`)

**Input:** `ho_raw_data.csv` + `mb_raw_data.csv` (from `real_estate_data/`)

**What it does:**
1. Loads each source CSV independently
2. Applies source-specific column renames (`_COLUMN_RENAME_MAP`)
3. Drops columns that shouldn't exist in merged data (`possession_breakdown`, `listing_url`)
4. Forces known numeric columns to proper dtype — nulls out invalid text values
5. Aligns column union (preserves first-seen column order)
6. Concatenates all DataFrames into one merged CSV

**Output:** `artifact/data_ingestion/merged/merged.csv`

---

### Stage 2: Data Transformation (`data_transformation.py`)

**Input:** `artifact/data_ingestion/merged/merged.csv`

**What it does (in order):**

#### 2a. City exclusion filter
Drops rows where city matches a hardcoded exclusion list — mostly rural/out-of-NCR localities (Hapur, Mewat, Neemrana, etc.) to keep the model focused on core NCR.

#### 2b. Free-text extraction from `description` + `amenities`
Using `description_parser.py`, the following are extracted from raw text via regex:

| Extracted field | Method | Example text → value |
|----------------|--------|----------------------|
| `bhk` | `extract_bhk()` | "3 BHK" → 3 |
| `bathrooms` | `extract_bathrooms()` | "Bathrooms 2" → 2 |
| `balconies` | `extract_balconies()` | "2 balconies" → 2 |
| `price` | `extract_price()` | "₹3.57 Cr" → 35,700,000 |
| `covered_area` | `extract_area_sqft()` | "1200 sq.ft" → 1200.0 |
| `age_of_property` | `extract_age_of_property()` | "5 to 10 years" → bucket |
| `furnishing` | `extract_furnishing()` | "Semi-Furnished" → encoded |
| `facing` | `extract_facing()` | "North" → "North" |
| `road_width_ft` | `extract_road_width_ft()` | "30 ft road" → 30.0 |
| `is_parking` | `has_parking()` | "Car parking available" → 1 |
| `is_pool` | `has_pool()` | "Swimming pool" → 1 |
| `is_main_road` | `has_main_road()` | "Main road" → 1 |
| `is_garden_park` | `has_garden_park()` | "Park facing" → 1 |
| `is_corner` | `is_corner()` | "corner property" → 1 |
| `is_gated` | `is_gated()` | "gated community" → 1 |

Extracted values fill in where the structured columns were missing. Priority: structured column first, description fallback second.

#### 2c. Area unit standardization
All area values converted to sqft using `_SQFT_CONVERSION` map (40+ unit variants handled):

```
area_sqft = area_value × unit_factor
```

| Unit | Factor |
|------|--------|
| sq-ft / sqft / sft | 1.0 |
| sq-yrd / sqyd | 9.0 |
| sq-m / sqm | 10.7639 |
| acre | 43,560 |
| bigha | 27,000 |
| marla | 272.25 |
| hectare | 107,639 |

#### 2d. Age of property bucketing
Standardized into 5 categories: "Less than 5 years", "5 to 10 years", "10 to 20 years", "Above 20 years", "New Construction"

#### 2e. price_per_sqft derivation
```
price_per_sqft = price_numeric / covered_area_sqft
```

#### 2f. Locality imputation and normalization
`LocalityMatcher` loads `ncr_colonies.json` and uses fuzzy matching to fill missing or misspelled locality names against the known city-locality index.

#### 2g. Circle rate enrichment
`CircleRateMatcher` loads all files from `circle_rates/` directory. For each row, looks up government floor price for the (city, locality, property_type) tuple. Falls back through:
1. Exact locality match
2. Fuzzy token-sort match (RapidFuzz, threshold 82)
3. City-level default

If coordinates are available and circle rate still missing: **Haversine BallTree nearest-neighbour fallback** — finds the nearest donor row by (lat, lon) and copies its circle rate.

#### 2h. Target variable
```
target = log(1 + price_per_sqft)   [log1p transform for stability]
```

**Output:** `artifact/data_transformation/cleaned.csv`

---

### Stage 3: Model Training (BF/APT)

Trained separately (notebooks, not yet in the main `TrainingPipeline`):

**BF model:** `notebooks/notebooks/sell/bf/best_bf_random_forest.pkl`  
**APT model:** `notebooks/notebooks/sell/apt/best_apt_random_forest.pkl`  
**BF rent model:** `notebooks/notebooks/rent/bf/best_bf_random_forest.pkl`  
**APT rent model:** `notebooks/notebooks/rent/apt/best_apt_random_forest.pkl`

Each model predicts `log_ratio = log(1 + price_per_sqft / circle_rate)`.

Additionally, a **Voronoi KMeans** model is saved alongside each:
- `bf_vor_kmeans.pkl` — partition of Delhi NCR into spatial cells for BF
- `apt_vor_kmeans.pkl` — same for APT

---

## 5. Pipeline 2 — Plot Training Pipeline

Fully productionized end-to-end. Triggered via `app_plot_train.py` → `PlotDataIngestion` → `PlotDataTransformation` → `PlotModelTrainer`.

### Stage 1: Plot Data Ingestion (`plot_data_ingestion.py`)

**Input:** Raw plot CSV files in `real_estate_data/`  
**Output:** `artifact/plot_ingestion/merged/combined_plot.csv`

Merges all plot-type raw CSVs into a single file.

---

### Stage 2: Plot Data Transformation (`plot_data_transformation.py`)

**Input:** `artifact/plot_ingestion/merged/combined_plot.csv`

#### 2a. Description parsing
Same `description_parser.py` utilities, plus a richer **amenity rule set** with 25+ amenity categories detected via regex:

`park`, `school`, `hospital`, `market`, `metro`, `airport`, `highway`, `water_supply`, `electricity`, `security`, `gymnasium`, `swimming_pool`, `club_house`, `boundary_wall`, `sewerage`, `college`, `bank_atm`, `restaurant`, `temple`, `vastu`, `street_light`, `wide_road`, `jogging_track`, `yoga_meditation`, `maintenance`

And a **token map** for structured amenity columns (e.g. "gated community" → `is_gated`, "boundary wall" → `has_boundary_wall`).

#### 2b. Area standardization
Same conversion table as BF/APT, but focused on sqft → for `plot_area` column.

#### 2c. Merge fields
For each property, takes the best available value across source columns and description-parsed values using priority merge:
- Numeric: source column first, description fallback
- Categorical: source column first, description fallback

#### 2d. Quality gate filters

```
price_per_sqft > 0
plot_area > 0
circle_rate > 0
0.7 < (price_per_sqft / circle_rate) < 25
```

Rows outside these bounds are removed as outliers.

#### 2e. KNN imputation for missing `facing_direction` and `road_width_m`

Preprocessing pipeline per KNN model:
- Numeric features: median imputation → z-score scaling
- Categorical features: mode imputation → one-hot encoding

KNN uses **inverse-distance weighting**:

```
ŷ(x) = Σ wⱼyⱼ / Σ wⱼ    where wⱼ ∝ 1/d(x, xⱼ)
```

For facing (categorical), KNN classifier predicts the mode under weighted neighborhood.

#### 2f. Circle rate enrichment (with Haversine fallback)
Same `CircleRateMatcher` as BF/APT. When circle rate is missing after fuzzy match, **BallTree (Haversine metric)** nearest-neighbour fallback:

```
circle_rate_missing_row ← circle_rate_nearest_donor_by_lat_lon
```

Donor pool priority:
1. Same city + same mapped property type
2. Same city
3. Same property type
4. Any known circle-rate row

#### 2g. Road distance features
Loads `NCR Roads.geojson` — parses features into three arrays (`MDR`, `SH`, `NH`) of line-segment coordinates `(lat1, lon1, lat2, lon2)`.

For each property at `(lat, lon)`, computes minimum perpendicular distance to each road class in km:

**Local projection:**
```
K_lat = 110.574
K_lon = 111.320 × cos(φ)
```

**Projection parameter onto segment:**
```
t = clip(-(x₁·dx + y₁·dy) / (dx² + dy²), 0, 1)
```

**Distance:**
```
d = sqrt((x₁ + t·dx)² + (y₁ + t·dy)²)  [km]
```

**Output features:** `closest_distance_MDR_km`, `closest_distance_SH_km`, `closest_distance_NH_km`

#### 2h. Target transformation

```
y = log(1 + price_per_sqft)    [log1p]
```

Also stored: `log_plot_area = log(1 + plot_area)` used during evaluation.

**Output:** `artifact/plot_transformation/cleaning_plot_datav1.csv`

---

### Stage 3: Plot Model Training (`plot_model_trainer.py`)

#### 3a. Clean and validate
Re-applies quality gates (NaN drops, positive values, ratio window).

#### 3b. Train/test split
Standard 80/20 split, stratified by city/zone where possible.

#### 3c. Spatial feature engineering (`add_spatial_features`)

**KMeans spatial clustering:**
1. Scale `(latitude, longitude)` with `StandardScaler`
2. Fit KMeans with `n_clusters` (config-driven, typically 15–25)
3. For each row: cluster assignment `c_i = argmin_k ||x_i - μ_k||₂`
4. Distance to assigned center: `dist_to_center = ||x_i - μ_{c_i}||₂`
5. One-hot features: `c_0, c_1, ..., c_K`

#### 3d. Outlier removal (IsolationForest)
Applied to training set only. Features used: `latitude`, `longitude`, `log_plot_area`, `circle_rate`, `log_target`.

```python
IsolationForest(contamination=config.contamination, n_estimators=100)
```

Inliers (`predict == 1`) are kept; outliers are discarded before fitting.

#### 3e. Model search
Tests multiple algorithms: `RandomForestRegressor`, `XGBRegressor` (if available), `LGBMRegressor` (if available).

Uses `RandomizedSearchCV` with custom scorer:

```
price_num_mae_scorer: predicts log_ppsf → back-transforms to total price → computes MAE in INR
price_num_r2_scorer: same → computes R² on total price in INR
```

This means the hyperparameter search directly optimizes **real-money MAE**, not log-space error.

#### 3f. Evaluation metrics

| Metric | Formula |
|--------|---------|
| MAE | `mean(|totalPrice_actual - totalPrice_pred|)` |
| MAPE | `mean(|actual - pred| / actual)` |
| R² | `1 - SS_res / SS_tot` |

Current plot model snapshot: **MAE ~1.03 Cr, MAPE ~24.72%, R² ~0.68**

#### 3g. Artifacts saved

| File | Content |
|------|---------|
| `artifact/plot_model_trainer/plot_v3_production_model.pkl` | Dict with keys: `model`, `features`, `coord_scaler`, `kmeans` |
| `artifact/plot_model_trainer/plot_feature_columns.pkl` | List of feature column names |
| `artifact/plot_model_trainer/plot_actual_vs_predicted_total_price.html` | Plotly scatter chart |
| `artifact/plot_model_trainer/plot_residual_analysis.html` | Residual analysis chart |

---

## 6. Pipeline 3 — Forecast Intelligence Pipeline

This pipeline is separate from the training pipelines above. It was trained **offline in the notebooks** and its outputs are already stored in `opt/`. The serving layer reads those artifacts at runtime.

### 6.1 What Was Trained Offline

For each segment (APT, Builder Floor, Plot), a **FloData locality trend model** was trained:

**Input:** Historical quarterly price panels in `inputs/interpolated_growth_trend_builder/`  
Each file is `{city}/{locality}.csv` containing:
```
date, price_per_sqft  (quarterly, interpolated, smoothed)
```

**Segment scale:**
- Apartment: 609 trend localities, 67,726 properties, 18,059 training rows
- Builder Floor: 575 trend localities, 42,098 properties, 11,716 training rows

### 6.2 Feature Engineering (per locality-quarter row)

| Feature | Description |
|---------|-------------|
| `lag_log_1` | log(ppsf) 1 quarter ago — ~50% of model importance |
| `lag_log_2` | log(ppsf) 2 quarters ago |
| `lag_price_1` | Raw ppsf 1 quarter ago |
| `lag_price_2` | Raw ppsf 2 quarters ago |
| `growth_lag_1` | QoQ growth rate, 1q lag |
| `growth_lag_2` | QoQ growth rate, 2q lag |
| `yoy_growth` | Year-over-year growth rate |
| `momentum` | Change of growth rate (acceleration) |
| `growth_roll_mean_4` | 4-quarter rolling mean growth |
| `growth_roll_std_4` | 4-quarter rolling volatility |
| `price_vs_city_avg` | Locality / city median ratio |
| `price_vs_zone_avg` | Locality / zone median ratio |
| `zone_price_rank` | Rank within zone |
| `time_index` | Quarter number (linear trend) |
| `gsdp_grow_pct` | State GDP growth (from `gsdp_repo_quarterly.csv`) |
| `loan_repo_spread` | Home loan rate − RBI repo rate (from rate CSVs) |

### 6.3 Model

RandomForest ensemble predicting `log(price_per_sqft)` per quarter. Back-transform: `ŷ = exp(ŷ_log)`.

Time-series cross-validation (3 splits, no future leakage).

**APT holdout:** MAE 556.6 ₹/sqft, MAPE 5.10%, R² 0.9546  
**BF holdout:** MAE 581.6 ₹/sqft, MAPE 5.79%, R² 0.9821

### 6.4 Cumulative Growth Index

After training, the model is rolled forward **20 quarters (5 years)** per locality:

```
I_l,t = ŷ_l,t / ŷ_l,0
```

`ŷ_l,0` = predicted price at forecast start date (dynamically set to current quarter, e.g. 2026-Q3).  
`I_l,t` = how much the locality is expected to grow by quarter t relative to today.

This index is stored in `opt/{segment}/future_forecasts.csv`.

### 6.5 Rho — Per-property Blending Weight

For each property, three signals are computed and combined into $\rho_0$:

| Signal | Symbol | Meaning |
|--------|--------|---------|
| Comparable support | $C_i$ | Is the listing price backed by nearby comparable transactions? |
| Uniqueness | $U_i$ | How far is this property from its k-nearest neighbours? (k-NN distance percentile) |
| Model confidence | $M_i$ | Composite of model coverage, CV fit score, stability |

**Formula:**
```
ρ₀ = clip(0.25 + 0.30·Cᵢ + 0.20·Uᵢ − 0.25·Mᵢ, 0.10, 0.70)
```

**Interpretation:**
- `+0.25` base — always start with 25% listing influence
- `+0.30·Cᵢ` — strong comps → trust listing more
- `+0.20·Uᵢ` — unusual property → model may not represent it, lean on listing
- `−0.25·Mᵢ` — high model confidence → trust model more, reduce listing weight

**Rho decay over forecast horizon:**
```
ρₜ = ρ₀ × 0.90ᵗ    (γ = 0.90 per quarter)
```

By quarter 20 (year 5), ρ has decayed to ~3.8% of its original value — the forecast is nearly entirely model-driven.

**Stored in:** `opt/{segment}/rho_details.csv` (one row per property with `C_i`, `U_i`, `M_i`, `rho_0`)

### 6.6 FloData Forecast Formula

The **core equation** — applied per property, per quarter:

**Log-geometric form (primary):**
```
log V_{i,t} = log(Pᵢ) + log(I_{l,t}) + ρₜ × (log(Lᵢ) − log(Pᵢ))

Equivalent: V_{i,t} = Pᵢ × I_{l,t} × (Lᵢ/Pᵢ)^ρₜ
```

**Linear form (equivalent, used for validation):**
```
F_{i,t} = [ρₜ·Lᵢ + (1−ρₜ)·Pᵢ] × I_{l,t}
```

Where:
- `Pᵢ` = model price (pi_price_per_sqft) — the ML-estimated fair value
- `Lᵢ` = listing price (price_per_sqft) — what the seller is asking
- `I_{l,t}` = locality growth index at quarter t
- `ρₜ` = decayed blending weight

**Total price at horizon:**
```
TotalPrice_{i,t} = V_{i,t} × Aᵢ    [Aᵢ = area in sqft]
```

**Stored in:** `opt/{segment}/property_forecasts_flodata.csv` (one row per property per quarter — 1.35M rows for APT alone)

---

## 7. API Layer

All endpoints served from `api/main.py` via FastAPI + Uvicorn.

### 7.1 Startup (loads into memory once)

```python
bf_model          = load_bf_model()             # pkl
bf_rent_model     = load_bf_rent_model()         # pkl
apt_model         = load_apt_model()             # pkl
apt_rent_model    = load_apt_rent_model()        # pkl
plot_model_bundle = load_plot_model_bundle()     # dict {model, features, coord_scaler, kmeans}
bf_voronoi        = load_bf_voronoi()            # KMeans pkl
apt_voronoi       = load_apt_voronoi()           # KMeans pkl
road_segments     = load_road_segments()         # dict {MDR: np.ndarray, SH:..., NH:...}
circle_rates_by_city = load_all_circle_rates()   # dict {city: {locality: rate}}
ncr_stats         = _compute_ncr_stats()         # segment median ppsf/cr/area
forecast_service  = ForecastIntelligenceService(PROJECT_ROOT)
```

---

### 7.2 Valuation Endpoints

#### `POST /predict/builder-floor`
**Input form fields:**
`bhk`, `area_sqft`, `bathrooms`, `balconies`, `age`, `furnishing`, `facing`, `circle_rate`, `is_parking`, `is_pool`, `is_main_road`, `is_garden_park`, `is_gated`, `is_corner`, `lat`, `lon`

**Pipeline:**
1. Lookup circle rate from `circle_rates_by_city` if not provided
2. Compute Voronoi features: `vor_cell_X` (one-hot), `voronoi_dist_to_seed`
3. One-hot encode: `age_of_property`, `furnishing_type`, `facing_direction`
4. Assemble feature vector, call `bf_model.predict(X)`
5. Back-transform: `pred_ratio = exp(ŷ) − 1`
6. `ppsf = pred_ratio × circle_rate`
7. `total_price = ppsf × area_sqft`
8. Build XAI driver cards (`_build_xai_explanation`)

**Returns:** `predictedPpsf`, `totalPrice`, `predRatio`, `circleRate`, `xaiExplanation`

#### `POST /predict/apartment`
Same pipeline as BF. Additional features: `floor_level` (Low/Medium/High), `property_segment` (Base/Mid/High/Luxury), `is_ground`, `is_top`.

#### `POST /predict/plot`
**Pipeline:**
1. Lookup circle rate
2. Compute spatial cluster features using `get_spatial_cluster_features()` (KMeans from model bundle)
3. Compute road distances using `nearest_distance_km()` for MDR/SH/NH (or use provided values)
4. One-hot encode: `usage_type`, `facing_direction`, `road_width` bucket
5. Assemble features aligned to `plot_model_bundle["features"]`
6. Call `plot_model.predict(X)`
7. Back-transform: `ppsf = exp(ŷ) − 1` (log-ppsf model)
8. `total_price = ppsf × area_sqft`
9. Build XAI driver cards

**Returns:** `predictedPpsf`, `totalPrice`, `xaiExplanation`, road distances, cluster info

---

### 7.3 Rent Endpoints

#### `POST /predict/builder-floor-rent` and `POST /predict/apartment-rent`
Same feature pipeline as sell models. Rent models predict log monthly rent:
```
monthlyRent = exp(ŷ_rent) − 1
annualRent = monthlyRent × 12
rentalYieldPct = annualRent / totalPrice × 100
```

---

### 7.4 What-If Endpoint

#### `POST /what-if/builder-floor` | `/what-if/apartment` | `/what-if/plot`
Accepts `baseInputs` + `variations` list (each is a modified input dict).

Runs prediction for base + all variations, returns comparison table:
`basePrediction`, `variations[{label, predictedPpsf, totalPrice, deltaPct}]`

---

### 7.5 Circle Rate Lookup Endpoint

#### `GET /circle-rate`
Parameters: `locality`, `city`, `property_type` (optional)

Uses `lookup_circle_rate()` → `_locality_match_score()` (composite: token_set_ratio + token_sort_ratio + partial_ratio + sector token match + overlap bonus).

Returns: matched rate, matched locality name, city.

#### `GET /locality-suggestions`
Parameters: `query`, `city`, `n`

Returns top-N fuzzy-matched locality names from circle rate data.

---

### 7.6 Forecast Endpoints

#### `GET /forecast/segments`
Lists available segments with their availability status.

#### `GET /forecast/cities?segment=...`
Returns all cities present in the property master file for the segment.

#### `GET /forecast/localities?segment=...&city=...&query=...&limit=...`
Searches locality names using `_locality_score()` composite scorer.

#### `GET /forecast/property-ids?segment=...&city=...&locality=...&limit=...`
Returns property IDs for a locality. Uses `get_close_matches()` for fuzzy locality key match.

#### `GET /forecast/overview?segment=...`
Returns: `metrics.json` contents + first 24 lines of `model_summary.md` + formula references + file paths.

#### `GET /forecast/context?segment=...&city=...&locality=...&property_id=...&years=...`

The **main forecast deep-dive endpoint**. Returns a full intelligence payload:

```json
{
  "segment": ...,
  "city": ...,
  "kpis": {
    "listingPricePpsf": ...,
    "modelPricePpsf": ...,
    "deltaPct": ...,
    "propertyForecastRows": ...,
    "localityForecastRows": ...,
    "localityDistributionProperties": ...
  },
  "property": { ...all property attributes... },
  "rho": { "comp_support", "uniqueness", "model_confidence", "rho_0_file", "rho_0_formula_raw", "rho_weights" },
  "mathValidation": {
    "linearFormula": ...,
    "logFormula": ...,
    "linearMeanAbsError": ...,
    "logMeanAbsError": ...,
    "linearMaxAbsError": ...,
    "logMaxAbsError": ...
  },
  "series": {
    "historicalTrend": [...],
    "localityForecast": [...],
    "propertyForecast": [...],
    "distribution": [...]
  },
  "yoy": {
    "property": [...YoY chain Y+1..Y+5...],
    "locality": { yoy_pct, base_price, target_price, ... }
  },
  "quarterTable": [{ quarter, date, P_i, L_i, rho_0, rho_t, I_lt, forecast_price_per_sqft }, ...]
}
```

#### `GET /forecast/locality-intelligence?segment=...&city=...&locality=...`
Returns a locality profile:
`medianPpsf`, `listingCount`, `medianCircleRate`, `affordabilityScore`, `rentalYieldPct`, `volatilityPct`, `forecastedAppreciation`, `priceTrend` (historical + forecast), `topCompetingLocalities`

---

### 7.7 Buy Decision Endpoint

#### `POST /buy-decision`
**Input:** forecast context + `holdYears`

Computes four components:

| Component | Formula |
|-----------|---------|
| Valuation score | `clip(50 − Δ% × 2, 0, 100)` where `Δ% = (L−P)/P×100` |
| Growth score | `clip(50 + 3 × avgYoY%, 0, 100)` |
| Upside score | `clip(50 + 1.5 × upside%, 0, 100)` where `upside% = (V_horizon/L−1)×100` |
| Risk penalty | `clip(2 × σ_qoq, 0, 30)` |

```
overall = clip(0.40·val + 0.35·growth + 0.25·upside − riskPenalty, 0, 100)
```

Verdicts: `≥68 → Buy`, `50–67 → Watch`, `<50 → Avoid`

---

### 7.8 ROI Projection Endpoint

#### `POST /roi-projection`
**Input:** `holdYears`, `purchaseCostPct`, `exitCostPct`, `annualHoldingCostPct`, optional `rentYieldPct`

**Cashflow model:**

```
buyPrice           = L_i × A
purchaseCosts      = buyPrice × purchaseCostPct/100
totalInvested      = buyPrice + purchaseCosts

grossSalePrice     = V_horizon × A
exitCosts          = grossSalePrice × exitCostPct/100
netSaleProceeds    = grossSalePrice − exitCosts

annualRent         = buyPrice × rentYieldPct/100   (or from rent model)
rentalIncomeTotal  = annualRent × years
holdingCostsTotal  = buyPrice × annualHoldingCostPct/100 × years

netProfit          = netSaleProceeds + rentalIncomeTotal − holdingCostsTotal − totalInvested
ROI%               = netProfit / totalInvested × 100
payoffMultiple     = (netSaleProceeds + rentalIncomeTotal − holdingCostsTotal) / totalInvested
CAGR%              = (payoffMultiple^(1/years) − 1) × 100
```

---

### 7.9 Meta Endpoint

#### `GET /meta/options`
Returns all dropdown options for the UI: cities, age categories, furnishing types, facing options, floor levels, APT segments, plot options, road widths, forecast segments.

---

## 8. Frontend

React SPA built with Vite. Served at `http://127.0.0.1:5173`. Connects to API at `VITE_API_BASE` (default: `http://127.0.0.1:8001`).

### Tab 1: Builder Floor
- Form inputs: BHK, area, bathrooms, balconies, age, furnishing, facing, circle rate, amenity toggles, lat/lon
- On submit: calls `POST /predict/builder-floor`
- Displays: predicted ppsf, total price, pred ratio, XAI driver cards (scored +2/+1/0/−1/−2)
- What-if panel: modify any field and compare side-by-side with base prediction

### Tab 2: Apartment
- Same as BF + floor level, property segment selectors
- Calls `POST /predict/apartment`
- Same XAI + what-if panel

### Tab 3: Plot
- Form: area, usage type, facing, road width, binary flags, lat/lon, road distance overrides
- Calls `POST /predict/plot`
- XAI for plots: location, road connectivity, area, circle-rate alignment

### Tab 4: Forecast Intelligence
- Segment selector (Builder Floor / Apartment / Plot)
- City selector (loaded from `GET /forecast/cities`)
- Locality search with autocomplete (from `GET /forecast/localities`)
- Property ID selector (from `GET /forecast/property-ids`)
- Deep-dive fetch → `GET /forecast/context`
- Displays:
  - KPI cards: listing ppsf, model ppsf, delta%, forecast rows
  - Rho decomposition panel (C_i, U_i, M_i, ρ₀)
  - YoY chain cards (Y+1 to Y+5 with anchor/target prices)
  - Quarter table (all 20 quarters: date, P_i, L_i, ρₜ, I_lt, forecast)
  - Formula validation errors (linear vs log)
  - Sparkline charts for historical trend, locality forecast, property forecast
  - Locality distribution fan chart

### Tab 5: Locality Intelligence
- Segment + city + locality inputs
- Fetches `GET /forecast/locality-intelligence`
- Displays: median ppsf, listing count, circle rate, affordability score, rental yield, volatility, 1-year appreciation, combined historical + forecast price trend chart, top 5 competing localities

---

## 9. Complete Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         RAW DATA SOURCES                                  │
│                                                                            │
│  real_estate_data/                          inputs/                        │
│  ├── ho_raw_data.csv (sell BF/APT)          ├── apartment_with_pi.csv     │
│  ├── mb_raw_data.csv (sell BF/APT)          ├── builder_floor_with_pi.csv │
│  ├── ho_rent.csv                            ├── plot_with_pi.csv          │
│  ├── mb_rent.csv                            ├── rbi_repo_rates.csv        │
│  ├── ncr_colonies.json                      ├── gsdp_repo_quarterly.csv   │
│  ├── circle_rates/*.json                    ├── repo_homeloan_quarterly.csv│
│  └── NCR Roads.geojson                      └── interpolated_growth_trend/│
└──────────────────────────────────────────────────────────────────────────┘
         │                                           │
         ▼                                           ▼
┌─────────────────────┐              ┌──────────────────────────────┐
│  PIPELINE 1 (BF/APT)│              │  PIPELINE 3 (FORECAST)       │
│  DataIngestion      │              │  (ran offline in notebooks)   │
│  → merged.csv       │              │                              │
│  DataTransformation │              │  Feature engineering:         │
│  → cleaned.csv      │              │  lag_log_1/2, growth, yoy,   │
│  (NLP parsing,      │              │  macro features              │
│  area conversion,   │              │                              │
│  circle rate,       │              │  RandomForest ensemble        │
│  locality fill)     │              │  → predicts log(ppsf) per    │
│                     │              │    locality per quarter        │
│  → Notebooks:       │              │                              │
│  RF model for BF   │              │  Roll forward 20q:            │
│  RF model for APT   │              │  I_l,t = ŷ_l,t / ŷ_l,0      │
│  Voronoi KMeans     │              │                              │
└─────────────────────┘              │  Compute ρ per property       │
         │                           │  FloData: V_i,t = P_i·I_l,t  │
         ▼                           │         ·(L_i/P_i)^ρₜ        │
┌─────────────────────┐              │                              │
│  PIPELINE 2 (PLOT)  │              │  → opt/{segment}/             │
│  PlotDataIngestion  │              │     future_forecasts.csv      │
│  → combined_plot    │              │     property_forecasts.csv    │
│  PlotDataTransform  │              │     rho_details.csv           │
│  → cleaning_plot_v1 │              │     metrics.json              │
│  (KNN imputation,   │              └──────────────────────────────┘
│  road distances,    │                          │
│  IsolationForest,   │                          │
│  KMeans spatial)    │                          │
│  PlotModelTrainer   │                          │
│  → pkl model bundle │                          │
└─────────────────────┘                          │
         │                                        │
         ▼                                        ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      API LAYER  (api/main.py)                         │
│                                                                        │
│  On startup: loads all pkl models, road segments, circle rates,       │
│  NCR stats, forecast service into memory                              │
│                                                                        │
│  POST /predict/builder-floor   → circle rate → voronoi → RF → XAI   │
│  POST /predict/apartment       → circle rate → voronoi → RF → XAI   │
│  POST /predict/plot            → circle rate → KMeans → RF → XAI    │
│  POST /predict/*/rent          → rent model → monthly/annual rent    │
│  POST /what-if/*               → base + N variations                 │
│  GET  /circle-rate             → fuzzy lookup                        │
│  GET  /forecast/context        → FloData deep-dive context           │
│  GET  /forecast/locality-intel → locality profile                    │
│  POST /buy-decision            → 4-factor scoring → Buy/Watch/Avoid │
│  POST /roi-projection          → cashflow model → ROI / CAGR        │
│  GET  /meta/options            → dropdown options for UI             │
└──────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    REACT FRONTEND  (frontend/)                         │
│                                                                        │
│  Tab 1: Builder Floor  → valuate → XAI → what-if                    │
│  Tab 2: Apartment      → valuate → XAI → what-if                    │
│  Tab 3: Plot           → valuate → XAI → what-if                    │
│  Tab 4: Forecast Intel → locality search → property deep-dive        │
│                          rho panel → YoY chain → quarter table       │
│                          buy score → ROI projection                  │
│  Tab 5: Locality Intel → locality profile → price trend chart        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 10. Every Calculation and Estimation Explained

### 10.1 Area Unit Conversion
```
area_sqft = raw_area_value × unit_factor
```
Over 40 unit string variants handled (sqft, sqyrd, sqm, acre, bigha, marla, biswa, rood, ground, hectare).

### 10.2 Price Per Sqft
```
price_per_sqft = price_numeric / area_sqft
```

### 10.3 Quality Filter Ratio
```
ratio = price_per_sqft / circle_rate
keep if 0.7 < ratio < 25
```
Removes sub-floor-value and wildly overpriced listings.

### 10.4 Circle Rate Lookup Score
Composite fuzzy score for locality matching:
```
score = max(token_set_ratio, token_sort_ratio, partial_ratio)
      + overlap_ratio × 28
      + subset bonus (10)
      + sector token match bonus (35) or penalty (−25)
      + prefix bonus (12)
```
Threshold 72 for circle rate, 40 for suggestions.

### 10.5 Haversine Nearest-Neighbour (Circle Rate Fallback)
BallTree with Haversine distance metric. Finds nearest donor by (lat, lon) and copies its circle rate.

### 10.6 KNN Imputation (Missing Fields)
```
ŷ(x) = Σⱼ wⱼ·yⱼ / Σⱼ wⱼ    where wⱼ = 1/d(x, xⱼ)
```
Used for: `facing_direction`, `road_width_m` in plot transformation.

### 10.7 KMeans Spatial Clustering (Plot)
```
cᵢ = argmin_k ||xᵢ − μₖ||₂
dist_to_center = ||xᵢ − μ_{cᵢ}||₂
One-hot: c_0, c_1, ..., c_K
```

### 10.8 Voronoi Cells (BF/APT)
```
cell_id = KMeans.predict([lat, lon])
voronoi_dist_to_seed = ||[lat,lon] − center[cell_id]||₂
```

### 10.9 Road Distance (MDR/SH/NH)
```
K_lat = 110.574,  K_lon = 111.320 × cos(φ_rad)

For each segment (lat1,lon1)→(lat2,lon2):
  x1=(lon1−lon)·K_lon,  y1=(lat1−lat)·K_lat
  x2=(lon2−lon)·K_lon,  y2=(lat2−lat)·K_lat
  dx=x2−x1, dy=y2−y1
  t = clip(−(x1·dx + y1·dy)/(dx²+dy²), 0, 1)
  d = sqrt((x1+t·dx)² + (y1+t·dy)²)  [km]

closest_distance_X_km = min(d) over all segments in class X
```

### 10.10 BF/APT Model Prediction
```
ŷ = model.predict(feature_vector)         [log-ratio]
pred_ratio = exp(ŷ) − 1
ppsf = pred_ratio × circle_rate
total_price = ppsf × area_sqft
```

### 10.11 Plot Model Prediction
```
ŷ = model.predict(feature_vector)         [log ppsf]
ppsf = exp(ŷ) − 1
total_price = ppsf × area_sqft
```

### 10.12 Isolation Forest (Outlier Removal in Training)
Applied on `[latitude, longitude, log_plot_area, circle_rate, log_target]`.  
Contamination rate configured. Predictions: `+1` = inlier (keep), `-1` = outlier (drop).

### 10.13 Locality Trend Model Target
```
y = log(ppsf)       trained per locality-quarter row
ŷ = exp(ŷ_log)      back-transform
```

### 10.14 Growth Index
```
I_l,t = predicted_ppsf_at_quarter_t / predicted_ppsf_at_quarter_0
```

### 10.15 Rho Computation
```
ρ₀ = clip(0.25 + 0.30·Cᵢ + 0.20·Uᵢ − 0.25·Mᵢ, 0.10, 0.70)
ρₜ = ρ₀ × 0.90ᵗ
```

### 10.16 FloData Forecast (log form)
```
V_{i,t} = Pᵢ × I_{l,t} × (Lᵢ/Pᵢ)^ρₜ
```

### 10.17 FloData Forecast (linear form)
```
F_{i,t} = [ρₜ·Lᵢ + (1−ρₜ)·Pᵢ] × I_{l,t}
```

### 10.18 Property YoY Chain
```
YoY%_{Y+k} = (V_{Y+k} / V_{Y+(k−1)} − 1) × 100
```
Chain: Y+1 relative to Y+0, Y+2 relative to Y+1, etc.

### 10.19 Locality YoY
```
LocalityYoY% = (median_ppsf_{t+4q} / median_ppsf_t − 1) × 100
```

### 10.20 Distribution Statistics (per forecast date)
`median`, `mean`, `min`, `max`, `p25 = 25th percentile`, `p75 = 75th percentile` across all properties in locality.

### 10.21 Valuation Gap
```
Δ% = (Lᵢ − Pᵢ) / Pᵢ × 100
```
Positive = seller asking above fair value. Negative = potential bargain.

### 10.22 Upside %
```
upside% = (V_horizon / Lᵢ − 1) × 100
```
Forecast exit price vs current entry price.

### 10.23 QoQ Volatility
```
QoQ%_t = (V_t − V_{t−1}) / V_{t−1} × 100
σ_qoq = std(QoQ%_t)
```

### 10.24 Buy Score
```
val_score    = clip(50 − Δ%×2,        0, 100)
growth_score = clip(50 + 3×avgYoY%,   0, 100)
upside_score = clip(50 + 1.5×upside%, 0, 100)
risk_penalty = clip(2×σ_qoq,          0, 30)
overall      = clip(0.40·val + 0.35·growth + 0.25·upside − risk_penalty, 0, 100)
```

### 10.25 ROI & CAGR
```
buyPrice          = Lᵢ × A
totalInvested     = buyPrice × (1 + purchaseCostPct/100)
netSaleProceeds   = V_horizon × A × (1 − exitCostPct/100)
annualRent        = buyPrice × rentYieldPct/100
rentalTotal       = annualRent × years
holdingTotal      = buyPrice × annualHoldingCostPct/100 × years
netProfit         = netSaleProceeds + rentalTotal − holdingTotal − totalInvested
ROI%              = netProfit / totalInvested × 100
payoffMultiple    = (netSaleProceeds + rentalTotal − holdingTotal) / totalInvested
CAGR%             = (payoffMultiple^(1/years) − 1) × 100
```

### 10.26 Affordability Score (Locality)
```
affordability = clip((circle_rate / median_ppsf) × 50, 0, 100)
```
100 = below floor (very cheap). 50 = at floor. 0 = far above floor (premium).

### 10.27 Rental Yield Estimate (Locality)
```
base_yield = {BF: 3.2%, APT: 2.8%, Plot: 1.5%}
adjustment = −0.5 × clip(ppsf/circle_rate − 1, −1, 2)
rental_yield = base_yield + adjustment
```

### 10.28 Formula Validation Errors
```
linear_abs_err = |[ρₜ·Lᵢ + (1−ρₜ)·Pᵢ]·I_lt − stored_forecast|
geometric_abs_err = |Pᵢ·I_lt·(Lᵢ/Pᵢ)^ρₜ − stored_forecast|
```
Mean and max of each are reported in the `mathValidation` block.

### 10.29 XAI Driver Scores
Five dimensions, each scored −2 to +2:

| Driver | Key comparison |
|--------|----------------|
| Location | circle_rate / NCR median circle_rate |
| Circle-rate alignment | pred_ratio or ppsf/circle_rate |
| Area | property area / NCR median area |
| Connectivity | distance to nearest highway (plot) or main road flag (BF/APT) |
| Property quality | amenity flags, segment, floor level, age, shape |
| Growth potential | locality 1-year YoY% |

### 10.30 Model Evaluation Metrics
```
MAE  = (1/n) Σ |yᵢ − ŷᵢ|
MAPE = (1/n) Σ |yᵢ − ŷᵢ| / yᵢ
R²   = 1 − Σ(yᵢ−ŷᵢ)² / Σ(yᵢ−ȳ)²
```
For plot model, `y` and `ŷ` are total prices in INR (reconstructed from log-ppsf).

---

## 11. Artifacts and What They Store

### Training Pipeline Artifacts

| Path | Content |
|------|---------|
| `artifact/data_ingestion/merged/merged.csv` | Raw BF+APT data merged from HO + MB |
| `artifact/data_transformation/cleaned.csv` | Feature-engineered BF+APT dataset |
| `artifact/data_transformation/locality_filled.csv` | After locality imputation step |
| `artifact/plot_ingestion/merged/combined_plot.csv` | Raw plot data merged |
| `artifact/plot_transformation/cleaning_plot_datav1.csv` | Feature-engineered plot dataset |
| `artifact/plot_model_trainer/plot_v3_production_model.pkl` | Plot model bundle: {model, features, coord_scaler, kmeans} |
| `artifact/plot_model_trainer/plot_feature_columns.pkl` | Ordered feature list |
| `artifact/plot_model_trainer/plot_actual_vs_predicted_total_price.html` | Plotly evaluation chart |
| `artifact/plot_model_trainer/plot_residual_analysis.html` | Residual analysis chart |

### Forecast Output Artifacts (per segment)

| Path | Content |
|------|---------|
| `opt/{seg}/metrics.json` | Model type, training/test split sizes, overall MAE/RMSE/MAPE/R², CV results, rho stats, forecast params |
| `opt/{seg}/model_summary.md` | Human-readable training report — method, coverage, performance, top features |
| `opt/{seg}/rho_details.csv` | One row per property: `property_id, locality, price_per_sqft, pi_price_per_sqft, C_i, U_i, M_i, rho_0` |
| `opt/{seg}/future_forecasts.csv` | Locality-level forecasts: `locality, date, quarter, pred_price_per_sqft, I_lt, ...` |
| `opt/{seg}/property_forecasts_flodata.csv` | Per-property per-quarter: `property_id, locality, trend_locality, date, quarter, P_i, L_i, rho_0, rho_t, I_lt, forecast_price_per_sqft` |
| `opt/apt/property_forecasts_flodata_new.csv` | Latest APT run (preferred over old) |
| `opt/{seg}/test_predictions.csv` | Holdout set: actual vs predicted ppsf per locality-quarter |
| `opt/{seg}/horizon_metrics.csv` | MAE/MAPE/R² broken out per forecast quarter (1–20) |
| `opt/{seg}/locality_metrics.csv` | Per-locality holdout performance |
| `opt/{seg}/feature_importance.csv` | Feature name + importance score from ensemble |
| `opt/{seg}/cv_results.csv` | Per-fold CV metrics |

---

## 12. What Has Been Done — Chronological Summary

### Phase 1: Core Data Engineering
- Ingested and merged raw property listings from two sources (HousingOnline + MagicBricks)
- Built `description_parser.py` — NLP regex extractor for BHK, area, price, furnishing, facing, amenities from free-text
- Built comprehensive area unit standardization (40+ unit variants → sqft)
- Implemented `CircleRateMatcher` — loads government circle rates from JSON, fuzzy-matches localities using RapidFuzz
- Implemented `LocalityMatcher` — normalizes and fuzzy-matches locality names against `ncr_colonies.json`
- Built Haversine BallTree nearest-neighbour fallback for missing circle rates

### Phase 2: BF/APT Model Development (notebooks)
- Engineered feature vectors including circle-rate ratio target, Voronoi spatial cells, one-hot amenities
- Trained RandomForest models for BF sell, APT sell, BF rent, APT rent
- Saved model PKLs + Voronoi KMeans PKLs

### Phase 3: Plot Pipeline (productionized)
- Built `PlotDataTransformation` with 25+ amenity detection rules, KNN imputation for missing fields
- Built `PlotModelTrainer` with IsolationForest outlier removal, KMeans spatial clustering, RandomizedSearchCV with money-MAE scorer
- Wired up full `TrainingPipeline` orchestrator
- Produced `plot_v3_production_model.pkl` with spatial cluster + road distance features

### Phase 4: Road Distance Integration
- Loaded `NCR Roads.geojson` — parsed NH/SH/MDR geometries into segment arrays
- Implemented `nearest_distance_km()` — vectorized point-to-segment min distance in km (locally projected coordinates)
- Integrated as serving-time features in plot endpoint

### Phase 5: Forecast Intelligence Layer
- Trained locality trend models (RandomForest ensemble on quarterly panel data) offline in notebooks
- Engineered 15 lag/growth/macro features per locality-quarter
- Rolled forward 20 quarters → cumulative growth index per locality
- Computed per-property rho from comparables, uniqueness, model confidence
- Applied FloData formula to produce 1.35M+ property-quarter forecasts (APT alone)
- Saved all artifacts to `opt/`

### Phase 6: Backend Forecast Integration
- Built `ForecastIntelligenceService` — chunked CSV reads (memory-safe), locality resolution, YoY computation, formula validation
- Added 8 forecast endpoints to `api/main.py`
- Built `/buy-decision` endpoint with 4-factor weighted scoring
- Built `/roi-projection` endpoint with full cashflow model
- Built `/forecast/locality-intelligence` endpoint with affordability, yield, volatility, trend

### Phase 7: XAI Layer
- Built `_build_xai_explanation()` — 5–6 driver cards scored −2 to +2
- Computed NCR-wide reference stats at startup (`_compute_ncr_stats`)
- Integrated XAI output into all three prediction endpoints

### Phase 8: React Frontend
- Built 5-tab React SPA (Builder Floor, Apartment, Plot, Forecast Intelligence, Locality Intelligence)
- Implemented all form inputs with default values pre-populated
- Added sparkline chart builder for forecast series
- Integrated buy score and ROI projection into Forecast tab
- Added what-if comparison panel

### Phase 9: Production Hardening
- Deduplicated property ID responses to prevent React key warnings
- Normalized `fore_app.py.py` filename
- Added `PROJECT_ROOT` / `os.chdir()` fix so API imports work from any directory
- Verified all three forecast segments via API acceptance tests
- Documented in `PRODUCTION_READINESS_CHECKLIST.md`

---

*Document generated May 2026. For math-only reference see `SYSTEM_MATH_AND_CONTEXT.md`. For forecast-only deep dive see `FORECAST_INTELLIGENCE_DEEP_DIVE.md`.*
