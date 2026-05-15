# NCR Real Estate AI System

End-to-end real estate intelligence stack for Delhi NCR with:

- Point-in-time valuation for Builder Floor, Apartment, and Plot/Land
- Circle-rate-aware feature engineering and geospatial enrichment
- What-if simulation and comparison in React dashboard
- New forecast intelligence integration from `inputs/` + `opt/` artifacts

## 1) What This System Does Today

### Online Prediction (FastAPI + React)
- Predicts fair-value `INR/sqft` and total price for:
	- Builder Floor
	- Apartment
	- Plot / Land
- Supports locality-aware circle rate lookup, geocoding, and nearest road distance features (MDR/SH/NH).
- Includes what-if analysis for numeric and categorical variables.

### Offline Training Pipeline (Python package)
- Ingestion -> transformation -> model training for plot workflow.
- Uses Haversine fallback for circle-rate imputation to maximize row retention.
- Produces production artifacts under `artifact/plot_model_trainer/`.

### Forecast Intelligence (newly integrated)
- Integrates your new forecast subsystem from:
	- `inputs/`
	- `opt/`
	- `fore_app.py` logic and mathematics
- Serves deep-dive forecast context through new backend endpoints.
- Exposes the new module inside React under the `Forecast Intelligence` tab.

## 2) Architecture

### Frontend
- `frontend/src/App.jsx`: main React app, tabs, forms, what-if, forecast tab
- `frontend/src/App.css`: layout + components + forecast tab styling

### Backend
- `api/main.py`: FastAPI service, prediction endpoints, feature builders, forecast endpoints
- `real_estate/utils/forecast_intelligence.py`: newly added forecast data service

### Core ML Package
- `real_estate/components/plot_data_ingestion.py`
- `real_estate/components/plot_data_transformation.py`
- `real_estate/components/plot_model_trainer.py`
- `real_estate/pipeline/training_pipeline.py`
- `real_estate/utils/circle_rate_matcher.py`

### Data and Artifacts
- `real_estate_data/real_estate_data/`: raw and engineered datasets + circle rates + roads geojson
- `artifact/`: transformed tables + trained models + feature lists + diagnostics
- `inputs/`: new forecast input datasets and trend panels
- `opt/`: new forecast output artifacts (metrics, rho, forecasts, model summaries)

## 3) Mathematical Foundations

## 3.1 Plot valuation model (serving)
- Model predicts `log(price_per_sqft)` for plot.
- Final serving conversion:
	- `pred_ppsf = expm1(pred_log_ppsf)`
	- `total_price = pred_ppsf * area_sqft`

Spatial + geospatial features include:
- Latitude/longitude
- Circle rate
- Distances to nearest MDR/SH/NH
- KMeans cluster one-hot (`c_*`) and distance to cluster center
- Road width one-hots, usage/facing one-hots, binary land flags

## 3.2 Forecast intelligence (FloData method)
Integrated from your new artifacts and app logic.

### Rho base and decay
- `rho_0 = clip(0.25 + 0.30*C_i + 0.20*U_i - 0.25*M_i, 0.10, 0.70)`
- `rho_t = rho_0 * 0.90^t`

Where:
- `C_i`: comparable support
- `U_i`: uniqueness
- `M_i`: model confidence

### Forecast blending form used by output artifacts
- `F_i,t = [rho_t * L_i + (1 - rho_t) * P_i] * I_l,t`

Where:
- `L_i`: listing price per sqft
- `P_i`: model/fair price per sqft
- `I_l,t`: locality growth multiplier at quarter `t`

### Also tracked for consistency
The log-space form from report text is also evaluated for diagnostics:
- `log(V_i,t) = log(P_i) + log(I_l,t) + rho_t * (log(L_i) - log(P_i))`

The backend now computes validation errors for both formulas against the stored forecast values.

## 4) Deep Dive: New Folders and Files

## 4.1 `inputs/`

### Core property-with-PI files
- `inputs/apartment_with_pi.csv`
- `inputs/builder_floor_with_pi.csv`
- `inputs/plot_with_pi.csv`

These carry property-level base fields (`property_id`, locality/city, geometry, amenities) plus:
- `price_per_sqft` (listed)
- `pi_price_per_sqft` (model/fair anchor)

### Macro and rates (auxiliary to forecast modeling lineage)
- `inputs/gsdp_repo_quarterly.csv`
- `inputs/rbi_repo_rates.csv`
- `inputs/repo_homeloan_quarterly.csv`
- `inputs/builder_mapped_pricing_final.csv`

### Trend panels
- `inputs/interpolated_growth_trend_builder/apt/**`
- `inputs/interpolated_growth_trend_builder/builder/**`
- `inputs/interpolated_growth_trend_builder/plot/**`

These are locality-level time series used for historical trend context and locality growth indexing.

## 4.2 `opt/`

Per-segment output folders:
- `opt/apt/`
- `opt/builder_floor/`
- `opt/plot/`

Important files in each segment:
- `metrics.json`: holdout and CV summary, horizon, rho stats
- `model_summary.md`: training + method report
- `rho_details.csv`: property-level `C_i`, `U_i`, `M_i`, `rho_0`
- `future_forecasts.csv`: locality-level future trajectories
- `property_forecasts_flodata.csv` (or apt new variant): property-quarter forecasts
- `test_predictions.csv`, `horizon_metrics.csv`, `locality_metrics.csv`, `feature_importance.csv`, `cv_results.csv`

## 4.3 `fore_app.py`
This is a Streamlit forecast dashboard with polished UI and FloData decomposition. It is now functionally represented in the main system through:
- backend forecast endpoints
- React forecast tab

Note: this file has been standardized to a single `.py` extension for clarity.

## 5) What Was Implemented in This Integration

### New backend module
- Added `real_estate/utils/forecast_intelligence.py`.
- Handles:
	- Segment config and file discovery
	- Locality suggestion and property ID lookup
	- Chunked reads for very large forecast files (memory-safe)
	- Trend-locality resolution
	- YoY calculations from current date
	- Formula validation (linear vs log form)

### New backend API endpoints
Added in `api/main.py`:
- `GET /forecast/segments`
- `GET /forecast/cities?segment=...`
- `GET /forecast/localities?segment=...&city=...&query=...&limit=...`
- `GET /forecast/property-ids?segment=...&city=...&locality=...&limit=...`
- `GET /forecast/overview?segment=...`
- `GET /forecast/context?segment=...&city=...&locality=...&property_id=...&years=...`

Also exposed forecast segment metadata in:
- `GET /meta/options` as `forecastSegments`

### Frontend integration
Updated React app to include `Forecast Intelligence` tab:
- Segment selection
- Forecast locality search + suggestions
- Property ID selection
- Deep-dive fetch and KPI rendering
- Rho decomposition display
- YoY cards
- Quarter table display

Files updated:
- `frontend/src/App.jsx`
- `frontend/src/App.css`

## 6) Active vs Legacy (high-level)

### Active production path
- `api/main.py`
- `frontend/`
- `real_estate/components/plot_*`
- `app_plot_train.py`
- `artifact/plot_model_trainer/*`

### Legacy or optional utilities
- Streamlit variants: `str_app.py`, `str_app1.py`, `real_estate_streamlit_app.py`
- One-time/optional scripts: `clean_plot.py`, `export_pg_to_csv.py`

Reference docs created during cleanup:
- `UNUSED_FILES_TO_REMOVE.md`
- `PROJECT_ANALYSIS_AND_MLOPS_ROADMAP.md`

## 7) Run Instructions

## 7.1 Backend
From project root:

```bash
.venv\Scripts\python.exe -m uvicorn api.main:app --reload
```

## 7.2 Frontend
From project root:

```bash
cd frontend
npm install
npm run dev
```

Default URLs:
- API: `http://127.0.0.1:8000`
- UI: `http://127.0.0.1:5173`

## 8) Current Model Snapshot

Plot training artifacts currently report:
- MAE around `1.03 Cr`
- MAPE around `24.72%`
- R2 around `0.68`

Forecast subsystem (segment-wise) metrics are available in each `opt/<segment>/metrics.json` and surfaced through `GET /forecast/overview`.

## 9) Next Technical Steps

1. Add pagination/streaming for large quarter tables if needed in UI.
2. Add chart rendering for forecast series in React (property vs locality vs historical).
3. Add automated validation tests for forecast endpoint math consistency.
4. Add MLflow tracking and model registry per roadmap in `PROJECT_ANALYSIS_AND_MLOPS_ROADMAP.md`.

## 10) Production Readiness Snapshot (May 2026)

- API acceptance: passed for all three forecast segments (`builder-floor`, `apartment`, `plot`).
- Forecast deep-dive context: verified with non-empty quarter tables and formula validation metrics.
- Frontend flow: verified from tab load through locality search and deep-dive rendering.
- Stability hardening applied: deduplicated property ID responses to avoid duplicate-key rendering warnings.
- Naming cleanup applied: `fore_app.py.py` renamed to `fore_app.py`.

For operational checklist and deployment guardrails, see `PRODUCTION_READINESS_CHECKLIST.md`.


How to Run 

tensorboard --logdir artifact/tensorboard

# Train each model (creates v1, then v2, etc.)
python app_apt_train.py
python app_bf_train.py
python app_plot_train.py

# View MLflow experiments
mlflow ui --backend-store-uri sqlite:///artifact/mlflow.db --port 5000

# Start API
uvicorn api.main:app --reload --port 8000

cd C:\Users\kushp\OneDrive\Desktop\geoai_ml\real_estate
python -m uvicorn api.main:app --reload