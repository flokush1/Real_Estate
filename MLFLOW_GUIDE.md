# MLflow & Versioned Pipeline – Complete Guide
## All 3 Property Types: Apartment · Builder Floor · Plot/Land

---

## 1. System Architecture

```
real_estate/
├── real_estate_data/          ← raw source CSVs (ho_raw_data.csv, mb_raw_data.csv, plot CSVs)
├── real_estate/
│   ├── components/
│   │   ├── data_ingestion.py          ← fetches & merges apt/bf CSVs
│   │   ├── data_transformation.py     ← cleans all types → cleaned.csv
│   │   ├── apt_model_trainer.py       ← trains apartment model  [NEW]
│   │   ├── bf_model_trainer.py        ← trains builder floor model  [NEW]
│   │   ├── plot_data_ingestion.py     ← fetches plot data from PostgreSQL
│   │   ├── plot_data_transformation.py← cleans plot data
│   │   └── plot_model_trainer.py      ← trains plot/land model
│   ├── pipeline/
│   │   └── training_pipeline.py       ← orchestrates all 3 pipelines
│   ├── entity/__init__.py             ← all versioned config dataclasses
│   ├── constant/__init__.py           ← all path constants
│   └── utils/__init__.py              ← versioning helpers (registry)
├── app_apt_train.py           ← entry point: train apartment model
├── app_bf_train.py            ← entry point: train builder floor model
├── app_plot_train.py          ← entry point: train plot/land model
├── app.py                     ← entry point: ingest + clean only (no model)
├── api/main.py                ← FastAPI serving all predictions
└── artifact/                  ← ALL versioned outputs land here
    ├── model_registry.json
    ├── mlflow.db
    ├── data_ingestion/
    ├── data_transformation/
    ├── apt_model_trainer/
    ├── bf_model_trainer/
    ├── plot_ingestion/
    ├── plot_transformation/
    └── plot_model_trainer/
```

---

## 2. Versioned Directory Structure (after 2 training runs)

```
artifact/
│
├── model_registry.json                    ← tracks all versions (auto-written)
├── mlflow.db                              ← SQLite MLflow tracking store
│
├── data_ingestion/
│   ├── merged/merged.csv                  ← canonical (latest) merged raw data
│   ├── v1/
│   │   ├── raw/
│   │   │   ├── ho_raw_data.csv
│   │   │   └── mb_raw_data.csv
│   │   └── merged/merged.csv             ← v1 raw merge snapshot
│   └── v2/
│       ├── raw/
│       └── merged/merged.csv             ← v2 raw merge snapshot
│
├── data_transformation/
│   ├── cleaned.csv                        ← canonical cleaned data
│   ├── v1/
│   │   └── cleaned.csv                   ← v1 cleaned CSV
│   └── v2/
│       └── cleaned.csv                   ← v2 cleaned CSV
│
├── apt_model_trainer/
│   ├── best_apt_random_forest.pkl         ← canonical model (API loads this)
│   ├── apt_feature_columns.pkl            ← canonical features
│   ├── apt_vor_kmeans.pkl                 ← canonical Voronoi KMeans
│   ├── v1/
│   │   ├── best_apt_random_forest.pkl
│   │   ├── apt_feature_columns.pkl
│   │   └── apt_vor_kmeans.pkl
│   └── v2/
│       ├── best_apt_random_forest.pkl
│       ├── apt_feature_columns.pkl
│       └── apt_vor_kmeans.pkl
│
├── bf_model_trainer/
│   ├── best_bf_random_forest.pkl          ← canonical model (API loads this)
│   ├── bf_feature_columns.pkl
│   ├── bf_vor_kmeans.pkl
│   ├── v1/ ...
│   └── v2/ ...
│
├── plot_ingestion/
│   ├── merged/combined_plot.csv           ← canonical
│   ├── v1/merged/combined_plot.csv
│   └── v2/merged/combined_plot.csv
│
├── plot_transformation/
│   ├── cleaning_plot_datav1.csv           ← canonical
│   ├── v1/cleaning_plot_datav1.csv
│   └── v2/cleaning_plot_datav1.csv
│
└── plot_model_trainer/
    ├── plot_v3_production_model.pkl       ← canonical (API loads this)
    ├── plot_feature_columns.pkl
    ├── v1/ ...
    └── v2/ ...
```

> **Rule**: `v{N}/` = snapshot for training run N. Root = always latest (API never breaks).

---

## 3. One-Time Setup

```bash
# 1. Create & activate virtual environment (already done)
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows PowerShell

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Copy .env.example to .env and fill in your PostgreSQL credentials
copy .env.example .env
# Edit .env:
#   DB_HOST=localhost
#   DB_PORT=5432
#   DB_NAME=your_db
#   DB_USER=your_user
#   DB_PASSWORD=your_password
```

---

## 4. Training Commands

### 4.1 Apartment Model

```bash
# From project root (c:\...\real_estate)
python app_apt_train.py
```

**What happens:**
1. Reads `real_estate_data/` CSVs (ho_raw_data.csv, mb_raw_data.csv)
2. Saves raw+merged to `artifact/data_ingestion/v{N}/`
3. Cleans + engineers features → `artifact/data_transformation/v{N}/cleaned.csv`
4. Filters rows where `property_type_grouped == apartment`
5. Trains RandomForest / XGBoost / LightGBM (best selected by CV R²)
6. Saves model to `artifact/apt_model_trainer/v{N}/best_apt_random_forest.pkl`
7. Promotes to canonical `artifact/apt_model_trainer/best_apt_random_forest.pkl`
8. Writes to `artifact/model_registry.json`
9. Logs params + metrics to `artifact/mlflow.db`

---

### 4.2 Builder Floor Model

```bash
python app_bf_train.py
```

Same stages as apartment but filters `property_type_grouped == builder_floor`.  
Saves to `artifact/bf_model_trainer/v{N}/`.

---

### 4.3 Plot / Land Model

```bash
python app_plot_train.py
```

**What happens (plot uses PostgreSQL, not CSV):**
1. Fetches plot/land listings directly from PostgreSQL → `artifact/plot_ingestion/v{N}/`
2. Cleans + engineers plot features → `artifact/plot_transformation/v{N}/`
3. Trains XGBoost / LightGBM / RandomForest with spatial Voronoi + KMeans clusters
4. Saves to `artifact/plot_model_trainer/v{N}/`
5. Promotes to canonical path + registry + MLflow

---

### 4.4 Run ALL three (sequential)

```bash
python app_apt_train.py ; python app_bf_train.py ; python app_plot_train.py
```

---

## 5. What Each Training Run Produces

| File | Description |
|---|---|
| `artifact/data_ingestion/v{N}/raw/*.csv` | Snapshot of source CSVs at time of training |
| `artifact/data_ingestion/v{N}/merged/merged.csv` | Merged raw data (apt + bf) |
| `artifact/data_transformation/v{N}/cleaned.csv` | Fully cleaned + feature-engineered data |
| `artifact/apt_model_trainer/v{N}/best_apt_random_forest.pkl` | Trained apt model bundle |
| `artifact/apt_model_trainer/v{N}/apt_feature_columns.pkl` | Feature column list |
| `artifact/apt_model_trainer/v{N}/apt_vor_kmeans.pkl` | Voronoi KMeans (60 clusters) |
| `artifact/bf_model_trainer/v{N}/best_bf_random_forest.pkl` | Trained bf model bundle |
| `artifact/plot_ingestion/v{N}/merged/combined_plot.csv` | Merged plot raw data |
| `artifact/plot_transformation/v{N}/cleaning_plot_datav1.csv` | Cleaned plot data |
| `artifact/plot_model_trainer/v{N}/plot_v3_production_model.pkl` | Trained plot model bundle |
| `artifact/model_registry.json` | Version registry (all types) |
| `artifact/mlflow.db` | MLflow tracking store |

---

## 6. Model Registry (model_registry.json)

Auto-written after each training run. Example after 2 apt + 1 bf + 1 plot run:

```json
{
  "apt": {
    "latest": 2,
    "versions": {
      "1": {
        "timestamp": "2026-05-10T10:30:00",
        "model_path": "artifact/apt_model_trainer/v1/best_apt_random_forest.pkl",
        "metrics": {
          "mae": 4200.0,
          "mape": 0.14,
          "r2": 0.87,
          "best_model": "LightGBM",
          "data_version": 1,
          "cleaned_csv": "artifact/data_transformation/v1/cleaned.csv",
          "merged_csv":  "artifact/data_ingestion/v1/merged/merged.csv"
        }
      },
      "2": {
        "timestamp": "2026-05-15T09:00:00",
        "model_path": "artifact/apt_model_trainer/v2/best_apt_random_forest.pkl",
        "metrics": { "mae": 3900.0, "mape": 0.13, "r2": 0.89, "best_model": "XGBoost" }
      }
    }
  },
  "bf": {
    "latest": 1,
    "versions": { "1": { ... } }
  },
  "plot": {
    "latest": 1,
    "versions": { "1": { ... } }
  }
}
```

---

## 7. MLflow UI

```bash
# Start the UI (from project root)
mlflow ui --backend-store-uri sqlite:///artifact/mlflow.db --port 5000

# Open in browser
# http://localhost:5000
```

You will see 3 experiments:
- `apartment-price-model`
- `builder-floor-price-model`
- `plot-price-model`

Each run shows:
- Params: model_type, version, n_features, n_training_rows, n_voronoi_clusters
- Metrics: mae, mape, r2
- Artifacts: .pkl files for download

**Comparing v1 vs v2:**
1. Open experiment → check both runs → click **Compare**
2. Parallel coordinates chart + metric table shows side-by-side diff

---

## 8. API

```bash
# Start the API (from project root)
uvicorn api.main:app --reload --port 8000
```

### Endpoints

| Method | URL | Description |
|---|---|---|
| `GET` | `/health` | API health check |
| `GET` | `/meta/model-registry` | View all versioned model history |
| `GET` | `/meta/model-status` | Which model files are currently loaded |
| `POST` | `/predict/apartment` | Apartment price prediction |
| `POST` | `/predict/builder-floor` | Builder floor price prediction |
| `POST` | `/predict/plot` | Plot/land price prediction |
| `GET` | `/forecast/overview` | Market forecast |
| `GET` | `/localities` | Available localities |

### Check model registry via API
```bash
curl http://localhost:8000/meta/model-registry
```

---

## 9. Model Features Summary

### Apartment & Builder Floor (shared features)
| Group | Features |
|---|---|
| Core | bhk, bathrooms, balconies, covered_area_sqft |
| Amenities | is_parking, is_pool, is_main_road, is_garden_park, is_gated, is_corner |
| Price signal | circle_rate |
| Floor | is_ground_floor, is_top_floor, floor_low, floor_medium, floor_high |
| Age (OHE) | age_Less than 5 years, age_5 to 10 years, age_10 to 20 years, age_Above 20 years, age_New Construction |
| Furnishing (OHE) | furn_Furnished, furn_Semi-Furnished, furn_Unfurnished |
| Facing (OHE) | facing_East/West/North/South + 4 diagonals |
| Spatial | voronoi_dist_to_seed, vor_cell_0 … vor_cell_59 (60 Voronoi cells) |
| Apt only | locality_tier_budget, locality_tier_mid, locality_tier_high, locality_tier_luxury |

### Plot / Land
| Group | Features |
|---|---|
| Core | log_plot_area, latitude, longitude |
| Usage | usage_type (Residential/Commercial OHE) |
| Road | road_width (3 buckets OHE), nh_dist_km, sh_dist_km, mdr_dist_km |
| Price signal | circle_rate, locality_target_encoding |
| Spatial | KMeans cluster centroid distances (60 clusters) |

---

## 10. Re-Training Workflow (Production Cycle)

```
Every N days (new data available):
                     ┌──────────────────────────────────────┐
                     │                                      │
  New DB / CSV data  │  python app_apt_train.py             │
        ↓            │  python app_bf_train.py              │
  Ingest (v{N})      │  python app_plot_train.py            │
        ↓            │                                      │
  Clean (v{N})       │  Each script:                        │
        ↓            │    • writes artifact/.../v{N}/       │
  Train (v{N})       │    • updates model_registry.json     │
        ↓            │    • logs to mlflow.db               │
  Register + Promote │    • promotes to canonical path      │
        ↓            │                                      │
  API auto-serves ───┘  (no restart needed)                 │
  latest model                                              │
                                                            │
  Compare v1 vs v2 in MLflow UI ──────────────────────────  │
  Rollback: copy v{N-1}/*.pkl back to canonical dir         │
```

### To roll back to a previous version:

```bash
# Example: roll back apartment to v1
copy artifact\apt_model_trainer\v1\best_apt_random_forest.pkl artifact\apt_model_trainer\best_apt_random_forest.pkl
copy artifact\apt_model_trainer\v1\apt_feature_columns.pkl    artifact\apt_model_trainer\apt_feature_columns.pkl
copy artifact\apt_model_trainer\v1\apt_vor_kmeans.pkl         artifact\apt_model_trainer\apt_vor_kmeans.pkl
# Restart API (or it will hot-reload)
```

---

## 11. Code Files Changed / Created

| File | Status | Purpose |
|---|---|---|
| `real_estate/constant/__init__.py` | Modified | Added APT_*, BF_* path constants |
| `real_estate/entity/__init__.py` | Modified | Versioned configs for all 4 ingestion/transform stages + AptModelTrainerConfig + BfModelTrainerConfig + artifacts |
| `real_estate/utils/__init__.py` | Modified | `get_next_model_version`, `register_model_version` |
| `real_estate/components/apt_model_trainer.py` | **New** | Full apartment model trainer |
| `real_estate/components/bf_model_trainer.py` | **New** | Full builder floor model trainer |
| `real_estate/components/plot_model_trainer.py` | Modified | Added MLflow logging block |
| `real_estate/pipeline/training_pipeline.py` | Modified | Added `run_apt_training_pipeline`, `run_bf_training_pipeline`; MLflow experiment setup |
| `app_apt_train.py` | **New** | Entry point for apartment training |
| `app_bf_train.py` | **New** | Entry point for builder floor training |
| `app_plot_train.py` | Existing | Entry point for plot training |
| `api/main.py` | Modified | Added `GET /meta/model-registry` endpoint |
| `requirements.txt` | Modified | Added `mlflow` |


## Overview

MLflow is integrated as an **experiment tracking layer** on top of the existing versioned model pipeline.
Every training run is automatically logged to a local SQLite database (`artifact/mlflow.db`) and can be
explored visually in the MLflow UI.

---

## Architecture

```
Training Run
    │
    ├─ Stage 1: PlotDataIngestion        (fetch fresh data from DB)
    ├─ Stage 2: PlotDataTransformation   (clean + feature engineering)
    ├─ Stage 3: PlotModelTrainer         (train model)
    │       │
    │       └─ MLflow run logged ─────────────────────────────────────────────┐
    │               • params: model_type, version, n_clusters, test_size ...  │
    │               • metrics: mae, mape, r2, baseline_mae, baseline_r2       │
    │               • artifacts: plot_v3_production_model.pkl,                │
    │                            plot_feature_columns.pkl                     │
    │                                                                         │
    ├─ Stage 4: register_model_version   (writes artifact/model_registry.json)│
    └─ Stage 5: promote to canonical path (API always loads latest)           │
                                                                              │
                        sqlite:///artifact/mlflow.db  ◄────────────────────────
```

---

## What is Tracked per Run

| Category  | Key                  | Example value                  |
|-----------|----------------------|-------------------------------|
| **Params** | `model_type`         | `LGBMRegressor`               |
| **Params** | `version`            | `3`                           |
| **Params** | `n_clusters`         | `60`                          |
| **Params** | `contamination`      | `0.02`                        |
| **Params** | `test_size`          | `0.2`                         |
| **Params** | `n_training_rows`    | `18420`                       |
| **Params** | `n_features`         | `47`                          |
| **Metrics** | `mae`               | `1240000.0` (₹)               |
| **Metrics** | `mape`              | `0.138`                       |
| **Metrics** | `r2`                | `0.912`                       |
| **Metrics** | `baseline_mae`      | `1580000.0` (baseline RF)     |
| **Metrics** | `baseline_r2`       | `0.881`                       |
| **Artifacts** | `model/plot_v3_production_model.pkl` | trained model bundle |
| **Artifacts** | `model/plot_feature_columns.pkl`    | feature list         |

---

## Setup

### 1. Install MLflow

```bash
pip install mlflow
# or with requirements:
pip install -r requirements.txt
```

### 2. Run the training pipeline (MLflow tracking is automatic)

```bash
python app_plot_train.py
```

After the run completes:
- `artifact/mlflow.db` is created/updated
- `artifact/model_registry.json` is updated
- `artifact/plot_model_trainer/v{N}/` contains the versioned model
- `artifact/plot_model_trainer/plot_v3_production_model.pkl` (canonical) is promoted

---

## Viewing the MLflow UI

```bash
# From the project root
mlflow ui --backend-store-uri sqlite:///artifact/mlflow.db --port 5000
```

Open **http://localhost:5000** in your browser.

You will see:
- **Experiment**: `plot-price-model`
- All runs listed with their metrics
- Click any run to compare params, metrics, and download artifacts
- Use the **Compare** button to put two versions side-by-side

---

## Comparing Runs (v1 vs v2)

In the MLflow UI:
1. Check the boxes next to both runs
2. Click **Compare**
3. See a parallel coordinates chart, metric table, and param diff

Or via Python:

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///artifact/mlflow.db")

client = mlflow.tracking.MlflowClient()
runs = client.search_runs(
    experiment_ids=["1"],
    order_by=["metrics.r2 DESC"]
)
for r in runs:
    print(r.info.run_name, r.data.metrics)
```

---

## Promoting a Model to Production (optional)

MLflow has a staging workflow (None → Staging → Production → Archived).
Currently this project uses `artifact/model_registry.json` for versioning
and the canonical file path for the API. To also use MLflow's registry:

```python
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///artifact/mlflow.db")
client = MlflowClient()

# List registered model versions
for mv in client.search_model_versions("name='PlotPriceModel'"):
    print(mv.version, mv.current_stage, mv.run_id)

# Promote version 3 to Production
client.transition_model_version_stage(
    name="PlotPriceModel",
    version="3",
    stage="Production"
)
```

> Note: `mlflow.register_model` is not called in the current pipeline because
> the model artifact is a custom dict bundle (model + kmeans + scaler + iso_forest).
> The MLflow artifact store holds the `.pkl` files for reference. If you want
> full registry support, wrap the bundle in a `mlflow.pyfunc.PythonModel`.

---

## File Locations

| File | Purpose |
|---|---|
| `artifact/mlflow.db` | SQLite tracking database (auto-created on first run) |
| `artifact/model_registry.json` | Lightweight version registry (always written) |
| `artifact/plot_model_trainer/v{N}/` | Versioned model files |
| `artifact/plot_model_trainer/plot_v3_production_model.pkl` | Canonical model (API uses this) |

---

## API Endpoint

After starting the FastAPI server (`uvicorn api.main:app`), the model registry
is available at:

```
GET /meta/model-registry
```

Example response:
```json
{
  "registry": {
    "plot": {
      "latest": 2,
      "versions": {
        "1": {
          "timestamp": "2026-05-10T10:30:00",
          "model_path": "artifact/plot_model_trainer/v1/plot_v3_production_model.pkl",
          "metrics": { "mae": 1580000, "mape": 0.162, "r2": 0.881, "best_model": "RandomForest" }
        },
        "2": {
          "timestamp": "2026-05-12T14:00:00",
          "model_path": "artifact/plot_model_trainer/v2/plot_v3_production_model.pkl",
          "metrics": { "mae": 1240000, "mape": 0.138, "r2": 0.912, "best_model": "LGBMRegressor" }
        }
      }
    }
  }
}
```

---

## Code Changes Summary

| File | Change |
|---|---|
| `requirements.txt` | Added `mlflow` |
| `real_estate/components/plot_model_trainer.py` | Added `HAS_MLFLOW` guard + `mlflow.start_run` block after `joblib.dump` |
| `real_estate/pipeline/training_pipeline.py` | Added `mlflow.set_tracking_uri` + `mlflow.set_experiment` at pipeline start |
| `api/main.py` | Added `GET /meta/model-registry` endpoint |
| `real_estate/utils/__init__.py` | Added `get_next_model_version`, `register_model_version` (previous session) |
| `real_estate/entity/__init__.py` | Made `PlotModelTrainerConfig` version-aware (previous session) |

MLflow is **non-blocking** — if it is not installed or the tracking server is unreachable,
training continues normally and only emits a warning.
