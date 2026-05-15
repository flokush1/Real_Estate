# Commands Reference

All commands are run from the **project root** (`real_estate/`) with the venv activated.

## 0. Activate Virtual Environment

```powershell
# Windows PowerShell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

---

## 1. Fetch Data from PostgreSQL

Exports 4 tables (`ho_raw_data`, `mb_raw_data`, `ho_rent`, `mb_rent`) to `real_estate_data/real_estate_data/*.csv`.

**Prerequisite:** Create a `.env` file in the project root:
```
PG_HOST=<your_host>
PG_PORT=5432
PG_DB=indian_real_estate
PG_USER=<your_user>
PG_PASSWORD=<your_password>
```

```powershell
python export_pg_to_csv.py
```

---

## 2. Data Ingestion + Cleaning (Sell Data)

Runs ingestion (merges `ho_raw_data.csv` + `mb_raw_data.csv`) and full feature-engineering transformation.
Produces `artifact/data_transformation/cleaned.csv`.

```powershell
python app.py
```

> This also runs model training for **both apt and bf** (full pipeline).
> To run only ingestion + transformation without training, use the pipeline steps individually.

---

## 3. Model Training

### Apartment model only
```powershell
python app_apt_train.py
```
Output: `artifact/apt_model_trainer/v{N}/best_apt_random_forest.pkl`

### Builder Floor model only
```powershell
python app_bf_train.py
```
Output: `artifact/bf_model_trainer/v{N}/best_bf_random_forest.pkl`

### Plot model (ingestion + transformation + training)
```powershell
python app_plot_train.py
```
Output: `artifact/plot_model_trainer/v{N}/best_plot_random_forest.pkl`

### Full pipeline (apt + bf, end-to-end)
```powershell
python app.py
```

---

## 4. API Server

### Start (production-style, no reload)
```powershell
python -m uvicorn api.main:app --port 8080
```

### Start (development, with auto-reload on source changes)
```powershell
python -m uvicorn api.main:app --port 8080 --reload --reload-dir api --reload-dir real_estate
```

> Use port **8080** (port 8000 may be reserved by Windows). If 8080 is also blocked, try 8001 or 9000.

Verify API is running:
```powershell
curl http://127.0.0.1:8080/meta/options
```

---

## 5. Frontend Dev Server

```powershell
cd frontend
npm install        # first time only
npm run dev
```

Frontend runs at `http://localhost:5173` and proxies `/api/*` to the backend.

### Production build
```powershell
cd frontend
npm run build
```

---

## 6. MLflow UI

View experiment runs, metrics, and parameters logged during training.

```powershell
mlflow ui --backend-store-uri sqlite:///artifact/mlflow.db --port 5000
```

Open: [http://127.0.0.1:5000](http://127.0.0.1:5000)

Experiments tracked:
- `apartment-price-model`
- `builder-floor-price-model`
- `plot-price-model`

---

## 7. TensorBoard

View training charts (loss curves, feature importances, residuals, etc.).

```powershell
tensorboard --logdir artifact/tensorboard
```

Open: [http://localhost:6006](http://localhost:6006)

Per-segment log dirs:
```
artifact/tensorboard/apt/v{N}/
artifact/tensorboard/bf/v{N}/
artifact/tensorboard/plot/v{N}/
```

---

## 8. Syntax / Compile Checks

```powershell
python -m py_compile api/main.py real_estate/utils/forecast_intelligence.py
```

---

## 9. Full Workflow (end-to-end fresh run)

```powershell
# 1. Fetch latest data from DB
python export_pg_to_csv.py

# 2. Run full sell pipeline (ingestion → transformation → apt+bf training)
python app.py

# 3. Run plot pipeline
python app_plot_train.py

# 4. Start API
python -m uvicorn api.main:app --port 8080 --reload --reload-dir api --reload-dir real_estate

# 5. (separate terminals) View MLflow + TensorBoard
mlflow ui --backend-store-uri sqlite:///artifact/mlflow.db --port 5000
tensorboard --logdir artifact/tensorboard

# 6. Start frontend
cd frontend ; npm run dev
```
