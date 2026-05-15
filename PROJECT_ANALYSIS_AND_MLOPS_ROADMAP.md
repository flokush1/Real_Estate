# NCR Real Estate Project - Structure Analysis & MLOps Roadmap

**Generated Date:** May 6, 2026  
**Project:** NCR Real Estate Price Prediction System

---

## 📊 CURRENT PROJECT STRUCTURE

```
real_estate/
├── 🚀 PRODUCTION SYSTEM (ACTIVELY USED)
│   ├── api/
│   │   └── main.py                          # FastAPI backend server (ACTIVE)
│   │
│   ├── frontend/                             # React + Vite UI (ACTIVE)
│   │   ├── src/
│   │   │   ├── App.jsx                      # Main React component
│   │   │   ├── App.css                      # Styling
│   │   │   └── index.css                    # Global styles
│   │   ├── index.html
│   │   ├── package.json
│   │   └── vite.config.js
│   │
│   ├── real_estate/                          # Core Python package (ACTIVE)
│   │   ├── components/
│   │   │   ├── data_ingestion.py           # Apartment/BF ingestion
│   │   │   ├── data_transformation.py      # Apartment/BF transformation
│   │   │   ├── plot_data_ingestion.py      # Plot ingestion (ACTIVE)
│   │   │   ├── plot_data_transformation.py # Plot transformation (ACTIVE)
│   │   │   └── plot_model_trainer.py       # Plot model training (ACTIVE)
│   │   │
│   │   ├── constant/
│   │   │   └── __init__.py                  # Project constants
│   │   │
│   │   ├── entity/
│   │   │   └── __init__.py                  # Data classes & configs
│   │   │
│   │   ├── exception/
│   │   │   └── exception.py                 # Custom exceptions
│   │   │
│   │   ├── logging/
│   │   │   └── logger.py                    # Logging config
│   │   │
│   │   ├── pipeline/
│   │   │   └── training_pipeline.py         # Orchestrates training
│   │   │
│   │   └── utils/
│   │       ├── circle_rate_matcher.py       # Circle rate lookup (ACTIVE)
│   │       ├── description_parser.py        # Parse descriptions
│   │       ├── locality_matcher.py          # Locality matching
│   │       └── push_cleaned_to_pg.py        # Upload to PostgreSQL
│   │
│   ├── artifact/                             # Model artifacts & data (ACTIVE)
│   │   ├── plot_model_trainer/
│   │   │   ├── plot_v3_production_model.pkl # Trained Random Forest
│   │   │   ├── plot_feature_columns.pkl     # Feature list
│   │   │   ├── plot_actual_vs_predicted_total_price.html
│   │   │   └── plot_residual_analysis.html
│   │   │
│   │   ├── plot_transformation/
│   │   │   └── cleaning_plot_datav1.csv    # Transformed training data
│   │   │
│   │   └── plot_ingestion/
│   │       └── merged/
│   │           └── combined_plot.csv       # Merged raw data
│   │
│   ├── real_estate_data/                    # Training datasets
│   │   ├── ho_raw_data.csv
│   │   ├── mb_raw_data.csv
│   │   ├── circle_rates/                   # Circle rate JSON files (ACTIVE)
│   │   └── NCR Roads.geojson              # Road network data (ACTIVE)
│   │
│   ├── logs/                                # Application logs
│   │
│   ├── 🔧 TRAINING SCRIPTS (ACTIVE)
│   ├── app_plot_train.py                   # Train plot model
│   ├── app_plot.py                         # Plot pipeline runner
│   ├── show_model_features.py              # Feature analysis tool
│   │
│   └── 📦 CONFIGURATION (ACTIVE)
│       ├── .env                            # Environment variables
│       ├── requirements.txt                # Python dependencies
│       ├── setup.py                        # Package setup
│       └── README.md                       # Documentation
│
├── ❌ LEGACY/UNUSED FILES (CAN BE REMOVED)
│   ├── app.py                              # Old apartment/BF training (UNUSED)
│   ├── str_app.py                          # Old Streamlit app v1 (UNUSED)
│   ├── str_app1.py                         # Old Streamlit app v2 (UNUSED)
│   ├── real_estate_streamlit_app.py        # Standalone Streamlit (UNUSED - replaced by React)
│   ├── clean_plot.py                       # Standalone cleaning script (UNUSED - logic moved to components)
│   ├── export_pg_to_csv.py                 # One-time data export utility (UNUSED)
│   ├── model_without_metro3.ipynb          # Old notebook (UNUSED)
│   ├── delhi_ncr_dashboard.html            # Old static dashboard (UNUSED)
│   ├── apt_locality_centroids.json         # Old apartment centroids (UNUSED)
│   │
│   └── notebooks/                          # Jupyter notebooks (RESEARCH ONLY)
│       └── notebooks/
│           ├── rent/                       # Rent model experiments
│           └── sell/                       # Sale model experiments
│               ├── apt/                    # Old apartment models
│               ├── bf/                     # Old builder floor models
│               └── plot/                   # Old plot models
│
└── 🏗️ INFRASTRUCTURE
    ├── .venv/                              # Virtual environment
    ├── venv/                               # Duplicate venv (can remove)
    ├── dist/                               # Build artifacts (gitignored)
    ├── .github/                            # CI/CD configs
    ├── Dockerfile                          # Docker config
    └── .gitignore
```

---

## ❌ FILES TO REMOVE OR ARCHIVE

### **Category 1: Superseded Streamlit Apps**
These were replaced by the React + FastAPI architecture:

1. **`str_app.py`** - Old Streamlit version 1 (720 lines)
2. **`str_app1.py`** - Old Streamlit version 2 (similar to v1)
3. **`real_estate_streamlit_app.py`** - Standalone Streamlit app (355 lines)

**Action:** Move to `archive/old_streamlit_apps/` folder

---

### **Category 2: Standalone Script Replaced by Components**

4. **`clean_plot.py`** - Standalone data cleaning script  
   - **Replaced by:** `real_estate/components/plot_data_transformation.py`
   - All logic has been integrated into the pipeline

**Action:** Move to `archive/standalone_scripts/`

---

### **Category 3: One-Time Utility Scripts**

5. **`export_pg_to_csv.py`** - Export PostgreSQL tables to CSV  
   - Used once for initial data extraction
   - No longer needed in production

**Action:** Move to `archive/utilities/`

---

### **Category 4: Legacy Training Script**

6. **`app.py`** - Old apartment/builder-floor training pipeline  
   - Only 8 lines, wrapper around old pipeline
   - Not used in current production (only plot model is trained)

**Action:** Keep if you plan to train Apartment/BF models, otherwise move to `archive/`

---

### **Category 5: Research Notebooks**

7. **`model_without_metro3.ipynb`** - Old experimentation notebook
8. **`notebooks/notebooks/`** - Jupyter notebooks for research
   - Contains old model experiments (apartment, builder floor, plot)
   - Some have pre-trained `.pkl` files that may conflict with production

**Action:** Keep for reference but clearly mark as "RESEARCH ONLY - NOT IN PRODUCTION"

---

### **Category 6: Duplicate Virtual Environment**

9. **`venv/`** folder - Duplicate of `.venv/`
   - Project uses `.venv/` for Python environment
   - `venv/` is redundant

**Action:** Delete `venv/` folder

---

### **Category 7: Old Static Files**

10. **`delhi_ncr_dashboard.html`** - Old static visualization
11. **`apt_locality_centroids.json`** - Old apartment centroid data

**Action:** Move to `archive/old_ui/`

---

## ✅ CORE PRODUCTION SYSTEM COMPONENTS

### **Currently Active & Used:**

1. **`api/main.py`** - FastAPI server (1074 lines)
   - Serves predictions for BF, Apartment, Plot
   - Circle rate lookup
   - Geocoding
   - Road distance calculations

2. **`frontend/`** - React + Vite UI
   - Modern dashboard with What-If analysis
   - Real-time predictions

3. **`real_estate/components/`** - Training pipeline modules
   - `plot_data_ingestion.py` - Fetch & merge plot data from PostgreSQL
   - `plot_data_transformation.py` - Feature engineering (1180 lines, includes Haversine circle-rate fill)
   - `plot_model_trainer.py` - Random Forest training with KMeans spatial clusters

4. **`real_estate/utils/circle_rate_matcher.py`** - Circle rate lookup system

5. **`app_plot_train.py`** - Execute full plot training pipeline

6. **`artifact/plot_model_trainer/`** - Production model artifacts
   - Model: Random Forest (MAE=1.03 Cr, MAPE=24.72%, R²=0.68)
   - 87 features including spatial clusters

---

## 🚀 MLOPS ROADMAP - PHASE-BY-PHASE IMPLEMENTATION

---

## **PHASE 1: Experiment Tracking with MLflow (Week 1-2)**

### **Objective:** Track all training experiments, parameters, and metrics

### **1.1 Setup MLflow Tracking Server**

```bash
# Install MLflow
pip install mlflow

# Start MLflow tracking server
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow_artifacts
```

**Location:** Access at `http://localhost:5000`

---

### **1.2 Integrate MLflow into Plot Model Trainer**

**File to modify:** `real_estate/components/plot_model_trainer.py`

**Changes needed:**

```python
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

class PlotModelTrainer:
    def initiate_model_training(self) -> PlotModelTrainerArtifact:
        # Start MLflow run
        with mlflow.start_run(run_name="plot_model_training"):
            
            # Log parameters
            mlflow.log_params({
                "contamination": self.config.contamination,
                "n_clusters": self.config.n_clusters,
                "test_size": self.config.test_size,
                "random_state": self.config.random_state,
            })
            
            # Log data metrics
            mlflow.log_metrics({
                "total_rows": len(df),
                "rows_after_cleaning": len(df_clean),
                "training_rows": len(X_train),
                "test_rows": len(X_test),
            })
            
            # Log model performance
            mlflow.log_metrics({
                "test_mae_cr": final_mae / 1e7,
                "test_mape": final_mape,
                "test_r2": final_r2,
            })
            
            # Log model
            signature = infer_signature(X_test, test_preds_log)
            mlflow.sklearn.log_model(
                final_model,
                "random_forest_model",
                signature=signature,
            )
            
            # Log feature importance plot
            mlflow.log_artifact(self.config.actual_vs_predicted_file_path)
            mlflow.log_artifact(self.config.residual_analysis_file_path)
            
            # Log feature columns
            mlflow.log_dict(
                {"features": X_train.columns.tolist()},
                "feature_columns.json"
            )
```

**Benefits:**
- Track all training runs in one place
- Compare different hyperparameters
- See model performance trends over time
- Easy rollback to previous model versions

---

## **PHASE 2: Model Registry & Versioning (Week 3)**

### **Objective:** Manage model lifecycle and promote models to production

### **2.1 Register Model in MLflow Model Registry**

```python
# In plot_model_trainer.py after training
with mlflow.start_run() as run:
    # ... training code ...
    
    # Register model
    model_uri = f"runs:/{run.info.run_id}/random_forest_model"
    model_details = mlflow.register_model(
        model_uri=model_uri,
        name="plot_price_predictor"
    )
    
    # Transition to Staging
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name="plot_price_predictor",
        version=model_details.version,
        stage="Staging"
    )
```

### **2.2 Update API to Load from MLflow Registry**

**File to modify:** `api/main.py`

```python
import mlflow.pyfunc

# Load production model from registry
def load_production_model():
    model_name = "plot_price_predictor"
    stage = "Production"  # or "Staging"
    
    model_uri = f"models:/{model_name}/{stage}"
    model = mlflow.pyfunc.load_model(model_uri)
    return model

plot_model = load_production_model()
```

**Benefits:**
- Clear model versioning (v1, v2, v3...)
- Staging → Production promotion workflow
- Easy A/B testing between model versions
- Rollback capability if new model underperforms

---

## **PHASE 3: Data Validation & Monitoring (Week 4-5)**

### **Objective:** Ensure data quality and detect data drift

### **3.1 Integrate Great Expectations for Data Validation**

```bash
pip install great-expectations
```

**Create:** `real_estate/validation/plot_data_validator.py`

```python
import great_expectations as gx

class PlotDataValidator:
    def validate_ingested_data(self, df: pd.DataFrame) -> bool:
        # Define expectations
        expectations = [
            {"expectation_type": "expect_column_values_to_not_be_null",
             "kwargs": {"column": "latitude"}},
            {"expectation_type": "expect_column_values_to_be_between",
             "kwargs": {"column": "plot_area", "min_value": 300, "max_value": 100000}},
            {"expectation_type": "expect_column_values_to_be_between",
             "kwargs": {"column": "price_per_sqft", "min_value": 1000, "max_value": 350000}},
        ]
        
        # Validate
        context = gx.get_context()
        suite = context.create_expectation_suite("plot_data_suite")
        
        for exp in expectations:
            suite.add_expectation(**exp)
        
        results = context.run_validation_operator(
            "action_list_operator",
            assets_to_validate=[df],
            expectation_suite_name="plot_data_suite"
        )
        
        return results["success"]
```

**Integrate into:** `plot_data_ingestion.py`

---

### **3.2 Add Evidently AI for Data Drift Detection**

```bash
pip install evidently
```

**Create:** `real_estate/monitoring/drift_detector.py`

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

class DriftMonitor:
    def detect_drift(self, reference_data: pd.DataFrame, current_data: pd.DataFrame):
        report = Report(metrics=[DataDriftPreset()])
        
        report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=None
        )
        
        report.save_html("artifact/monitoring/drift_report.html")
        
        # Log drift to MLflow
        drift_detected = report.as_dict()["metrics"][0]["result"]["dataset_drift"]
        
        mlflow.log_metric("data_drift_detected", int(drift_detected))
        mlflow.log_artifact("artifact/monitoring/drift_report.html")
        
        return drift_detected
```

**Benefits:**
- Catch data quality issues before training
- Detect distribution shifts in production data
- Alert when model retraining is needed

---

## **PHASE 4: Automated Retraining Pipeline (Week 6)**

### **Objective:** Automate model retraining when data drifts or performance degrades

### **4.1 Create Automated Training Workflow**

**Create:** `real_estate/automation/auto_retrain.py`

```python
import schedule
import time
from real_estate.monitoring.drift_detector import DriftMonitor
from real_estate.pipeline.training_pipeline import TrainingPipeline

class AutoRetrainer:
    def __init__(self):
        self.drift_monitor = DriftMonitor()
        self.pipeline = TrainingPipeline()
        
    def check_and_retrain(self):
        # Load reference data (last training data)
        reference_data = pd.read_csv("artifact/plot_transformation/cleaning_plot_datav1.csv")
        
        # Load current data from PostgreSQL
        current_data = self.fetch_latest_data()
        
        # Check for drift
        drift_detected = self.drift_monitor.detect_drift(reference_data, current_data)
        
        if drift_detected:
            logging.warning("Data drift detected! Initiating retraining...")
            
            # Run training pipeline
            artifact = self.pipeline.run_plot_training_pipeline()
            
            # Evaluate new model
            new_model_metrics = self.evaluate_model(artifact)
            
            # If new model is better, promote to production
            if new_model_metrics["mae"] < self.get_production_mae():
                self.promote_to_production(artifact)
            
    def schedule_checks(self):
        # Check daily at 2 AM
        schedule.every().day.at("02:00").do(self.check_and_retrain)
        
        while True:
            schedule.run_pending()
            time.sleep(3600)  # Check every hour
```

---

## **PHASE 5: CI/CD Pipeline with GitHub Actions (Week 7)**

### **Objective:** Automate testing, validation, and deployment

### **5.1 Create GitHub Actions Workflow**

**Create:** `.github/workflows/ml_pipeline.yml`

```yaml
name: ML Pipeline CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test-and-validate:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Run data validation tests
      run: |
        pytest tests/test_data_validation.py
    
    - name: Run model tests
      run: |
        pytest tests/test_model_trainer.py
    
    - name: Check code quality
      run: |
        flake8 real_estate/ --max-line-length=120
        black --check real_estate/
  
  train-model:
    needs: test-and-validate
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: pip install -r requirements.txt
    
    - name: Train model
      env:
        MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
      run: |
        python app_plot_train.py
    
    - name: Upload artifacts
      uses: actions/upload-artifact@v3
      with:
        name: model-artifacts
        path: artifact/plot_model_trainer/
```

---

## **PHASE 6: Model Serving with Docker & FastAPI Optimization (Week 8)**

### **Objective:** Containerize application and optimize serving

### **6.1 Update Dockerfile for Production**

**Create/Update:** `Dockerfile`

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose ports
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=http://mlflow-server:5000

# Run FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### **6.2 Create Docker Compose for Full Stack**

**Create:** `docker-compose.yml`

```yaml
version: '3.8'

services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    volumes:
      - ./mlflow_artifacts:/mlflow/artifacts
      - ./mlflow.db:/mlflow/mlflow.db
    command: >
      mlflow server
      --backend-store-uri sqlite:///mlflow/mlflow.db
      --default-artifact-root /mlflow/artifacts
      --host 0.0.0.0
  
  api:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - mlflow
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    volumes:
      - ./artifact:/app/artifact
      - ./real_estate_data:/app/real_estate_data
  
  frontend:
    image: node:18-alpine
    working_dir: /app
    volumes:
      - ./frontend:/app
    ports:
      - "5173:5173"
    command: npm run dev -- --host
```

---

## **PHASE 7: Monitoring & Alerting (Week 9-10)**

### **Objective:** Monitor model performance in production and set up alerts

### **7.1 Add Prometheus Metrics to FastAPI**

```bash
pip install prometheus-fastapi-instrumentator
```

**Update:** `api/main.py`

```python
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()

# Add Prometheus metrics
Instrumentator().instrument(app).expose(app)

# Custom metrics
from prometheus_client import Counter, Histogram

prediction_counter = Counter(
    'plot_predictions_total',
    'Total number of plot predictions'
)

prediction_latency = Histogram(
    'plot_prediction_latency_seconds',
    'Plot prediction latency'
)

@app.post("/predict/plot")
def predict_plot(req: PlotRequest):
    with prediction_latency.time():
        prediction_counter.inc()
        # ... prediction logic ...
        return result
```

### **7.2 Set Up Grafana Dashboard**

**Create:** `docker-compose.yml` (add services)

```yaml
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
  
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
```

**Key Metrics to Monitor:**
- Prediction requests per minute
- Average prediction latency
- Model MAE/MAPE trend over time
- Data drift alerts
- API error rates

---

## **PHASE 8: Feature Store (Optional - Week 11+)**

### **Objective:** Centralize feature engineering for consistency

### **8.1 Implement Feast Feature Store**

```bash
pip install feast
```

**Create:** `feature_store/feature_definitions.py`

```python
from feast import Entity, Feature, FeatureView, FileSource
from feast.value_type import ValueType

plot_entity = Entity(name="plot_id", value_type=ValueType.STRING)

plot_features = FeatureView(
    name="plot_features",
    entities=["plot_id"],
    features=[
        Feature(name="latitude", dtype=ValueType.DOUBLE),
        Feature(name="longitude", dtype=ValueType.DOUBLE),
        Feature(name="circle_rate", dtype=ValueType.DOUBLE),
        Feature(name="plot_area", dtype=ValueType.DOUBLE),
        Feature(name="is_corner", dtype=ValueType.INT64),
        # ... more features
    ],
    batch_source=FileSource(
        path="real_estate_data/plot_features.parquet",
        event_timestamp_column="timestamp",
    ),
)
```

---

## 📋 IMPLEMENTATION CHECKLIST

### **Week 1-2: MLflow Setup**
- [ ] Install MLflow and start tracking server
- [ ] Integrate MLflow logging in `plot_model_trainer.py`
- [ ] Create MLflow experiment for plot models
- [ ] Log hyperparameters, metrics, and artifacts
- [ ] Test by running 3-5 training experiments

### **Week 3: Model Registry**
- [ ] Register models in MLflow Model Registry
- [ ] Create staging/production promotion workflow
- [ ] Update API to load model from registry
- [ ] Document model versioning process

### **Week 4-5: Data Validation**
- [ ] Install Great Expectations
- [ ] Create data validation suite for plot data
- [ ] Add Evidently AI drift detection
- [ ] Integrate validation into ingestion pipeline
- [ ] Create drift monitoring dashboard

### **Week 6: Automated Retraining**
- [ ] Create auto-retraining script
- [ ] Set up scheduled drift checks
- [ ] Implement model promotion logic
- [ ] Test end-to-end retraining flow

### **Week 7: CI/CD**
- [ ] Create GitHub Actions workflow
- [ ] Add unit tests for data validation
- [ ] Add unit tests for model training
- [ ] Set up automated model training on push
- [ ] Configure artifact storage

### **Week 8: Docker & Deployment**
- [ ] Update Dockerfile for production
- [ ] Create docker-compose for full stack
- [ ] Test containerized deployment locally
- [ ] Document deployment process
- [ ] Set up cloud deployment (AWS/GCP/Azure)

### **Week 9-10: Monitoring**
- [ ] Add Prometheus metrics to API
- [ ] Set up Prometheus + Grafana
- [ ] Create monitoring dashboards
- [ ] Configure alerting rules
- [ ] Test alert notifications

### **Week 11+ (Optional): Feature Store**
- [ ] Research Feast setup
- [ ] Define feature schemas
- [ ] Migrate feature engineering to Feast
- [ ] Update training & serving pipelines

---

## 🎯 SUCCESS METRICS

### **MLOps Maturity KPIs:**

1. **Experiment Tracking**
   - ✅ All training runs logged in MLflow
   - ✅ 100% parameter and metric tracking coverage

2. **Model Registry**
   - ✅ All models versioned in registry
   - ✅ Clear staging → production promotion workflow

3. **Automation**
   - ✅ Automated retraining on data drift
   - ✅ CI/CD pipeline for model training
   - ✅ < 10 min deployment time

4. **Monitoring**
   - ✅ Real-time model performance tracking
   - ✅ Data drift detection active
   - ✅ Alert system for model degradation

5. **Reproducibility**
   - ✅ Any model version can be reproduced
   - ✅ Full lineage tracking (data → model → deployment)

---

## 💰 ESTIMATED COSTS (Cloud Deployment)

### **AWS Infrastructure (Monthly):**
- EC2 t3.medium (API): ~$30
- EC2 t3.small (MLflow): ~$15
- RDS PostgreSQL: ~$25
- S3 Storage (artifacts): ~$10
- CloudWatch Monitoring: ~$10

**Total:** ~$90/month

### **Alternative (Free Tier):**
- Heroku (API): Free
- Render (MLflow): Free
- Supabase (PostgreSQL): Free
- GitHub Actions: 2000 mins/month free

---

## 📚 RECOMMENDED TOOLS & TECHNOLOGIES

| Component | Tool | Why? |
|-----------|------|------|
| Experiment Tracking | MLflow | Industry standard, free, self-hosted |
| Model Registry | MLflow Registry | Integrated with tracking |
| Data Validation | Great Expectations | Comprehensive validation rules |
| Drift Detection | Evidently AI | Easy-to-use, great visualizations |
| Orchestration | Apache Airflow / Prefect | Schedule & monitor pipelines |
| CI/CD | GitHub Actions | Free, integrated with repo |
| Containerization | Docker + Docker Compose | Standard for deployment |
| Monitoring | Prometheus + Grafana | Open-source, powerful |
| Feature Store | Feast (optional) | Online/offline consistency |

---

## 🚨 CRITICAL NOTES

1. **Current Production System is Working**  
   - Don't break what's working while adding MLOps
   - Implement incrementally, phase by phase
   - Keep old artifacts as backup during transition

2. **Database Credentials**  
   - Ensure `.env` is properly configured for PostgreSQL
   - Never commit `.env` to git
   - Use environment variables in production

3. **Model Versioning**  
   - Current model: `plot_v3_production_model.pkl`
   - After MLflow: Will be `plot_price_predictor/v4`, `v5`, etc.

4. **Testing is Critical**  
   - Add unit tests before CI/CD
   - Test data validation before production
   - Always test retraining on staging environment first

---

## 📞 NEXT STEPS

1. **Review this roadmap** with your team
2. **Prioritize phases** based on business needs
3. **Set up MLflow** as the foundation (Week 1-2)
4. **Start experiment tracking** immediately
5. **Schedule weekly progress reviews**

---

**Document Prepared By:** GitHub Copilot  
**For:** NCR Real Estate Price Prediction System  
**Version:** 1.0  
**Last Updated:** May 6, 2026
