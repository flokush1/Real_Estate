# Unused Files - Quick Reference

## Files to Archive or Remove

### ❌ **DEFINITELY UNUSED - Can Archive/Remove**

1. **`str_app.py`** (720+ lines)
   - Old Streamlit app version 1
   - **Replaced by:** `frontend/` (React app)
   - **Action:** Move to `archive/old_streamlit_apps/`

2. **`str_app1.py`** (700+ lines)
   - Old Streamlit app version 2
   - **Replaced by:** `frontend/` (React app)
   - **Action:** Move to `archive/old_streamlit_apps/`

3. **`real_estate_streamlit_app.py`** (355 lines)
   - Standalone Streamlit prediction app
   - **Replaced by:** `frontend/` + `api/main.py`
   - **Action:** Move to `archive/old_streamlit_apps/`

4. **`clean_plot.py`** (400+ lines)
   - Standalone data cleaning script
   - **Replaced by:** `real_estate/components/plot_data_transformation.py`
   - **Action:** Move to `archive/standalone_scripts/`

5. **`model_without_metro3.ipynb`**
   - Old Jupyter notebook for experimentation
   - **Status:** Research/exploration only, not in production
   - **Action:** Move to `notebooks/archive/`

6. **`delhi_ncr_dashboard.html`**
   - Old static HTML dashboard
   - **Replaced by:** React frontend
   - **Action:** Move to `archive/old_ui/`

7. **`apt_locality_centroids.json`**
   - Old apartment locality centroids
   - **Status:** Not used in current system
   - **Action:** Move to `archive/old_data/`

---

### ⚠️ **UTILITY SCRIPTS - Keep or Archive Based on Need**

8. **`export_pg_to_csv.py`** (60 lines)
   - One-time PostgreSQL table export utility
   - **Used:** Only when you need to export raw data from PostgreSQL
   - **Action:** 
     - If you never need to export again: Move to `archive/utilities/`
     - If occasionally needed: Keep in root

9. **`real_estate/utils/push_cleaned_to_pg.py`** (230 lines)
   - Upload cleaned data back to PostgreSQL
   - **Used:** Only if you want to push cleaned data to database
   - **Action:** 
     - If not used in production workflow: Move to `archive/utilities/`
     - If part of data pipeline: Keep

---

### 📓 **RESEARCH NOTEBOOKS - Keep for Reference**

10. **`notebooks/notebooks/rent/`**
    - Rent prediction model experiments
    - **Status:** Research only, not in production
    - **Action:** Keep but mark as "RESEARCH ONLY"

11. **`notebooks/notebooks/sell/apt/`**
    - Old apartment sale model experiments
    - **Status:** Research only
    - **Action:** Keep for reference

12. **`notebooks/notebooks/sell/bf/`**
    - Old builder floor model experiments
    - **Status:** Research only
    - **Action:** Keep for reference

13. **`notebooks/notebooks/sell/plot/`**
    - Old plot model experiments
    - **Status:** Research only (production uses `real_estate/components/`)
    - **Action:** Keep for reference

---

### 🗑️ **DUPLICATE FOLDER - Can Delete**

14. **`venv/`** folder
    - Duplicate Python virtual environment
    - **Active environment:** `.venv/`
    - **Action:** Delete `venv/` folder entirely

---

### ✅ **ACTIVELY USED - DO NOT REMOVE**

These files are **currently in production** and should NOT be removed:

- ✅ `api/main.py` - FastAPI backend
- ✅ `frontend/` - React UI
- ✅ `real_estate/` - Core Python package
- ✅ `app_plot_train.py` - Training entry point
- ✅ `app_plot.py` - Pipeline runner
- ✅ `show_model_features.py` - Feature analysis
- ✅ `artifact/` - Model artifacts and data
- ✅ `real_estate_data/` - Training datasets
- ✅ `requirements.txt` - Dependencies
- ✅ `setup.py` - Package setup
- ✅ `.env` - Environment variables

---

### 🤔 **UNCERTAIN - Review Before Removing**

15. **`app.py`** (8 lines)
    - Old apartment/builder floor training pipeline
    - **Currently:** Only used for training apt/bf models
    - **Action:** 
      - If you plan to train apt/bf models: Keep
      - If only using plot model: Archive

---

## Recommended Archive Structure

Create this folder structure to preserve old code:

```
archive/
├── old_streamlit_apps/
│   ├── str_app.py
│   ├── str_app1.py
│   └── real_estate_streamlit_app.py
│
├── standalone_scripts/
│   └── clean_plot.py
│
├── old_ui/
│   ├── delhi_ncr_dashboard.html
│   └── apt_locality_centroids.json
│
├── utilities/
│   ├── export_pg_to_csv.py
│   └── push_cleaned_to_pg.py (if not used)
│
└── old_notebooks/
    └── model_without_metro3.ipynb
```

---

## Commands to Archive Files

```bash
# Create archive folder
mkdir archive
mkdir archive/old_streamlit_apps
mkdir archive/standalone_scripts
mkdir archive/old_ui
mkdir archive/utilities
mkdir archive/old_notebooks

# Move Streamlit apps
mv str_app.py archive/old_streamlit_apps/
mv str_app1.py archive/old_streamlit_apps/
mv real_estate_streamlit_app.py archive/old_streamlit_apps/

# Move standalone scripts
mv clean_plot.py archive/standalone_scripts/

# Move old UI files
mv delhi_ncr_dashboard.html archive/old_ui/
mv apt_locality_centroids.json archive/old_ui/

# Move utilities (if not needed)
mv export_pg_to_csv.py archive/utilities/

# Move old notebook
mv model_without_metro3.ipynb archive/old_notebooks/

# Delete duplicate venv (CAREFUL!)
# First verify that .venv/ is the active one
rm -rf venv/
```

---

## Summary

| Category | Count | Action |
|----------|-------|--------|
| Streamlit apps | 3 files | Archive |
| Standalone scripts | 1 file | Archive |
| Old UI files | 2 files | Archive |
| Utilities | 2 files | Keep or Archive |
| Notebooks | 1 file + folders | Keep for reference |
| Duplicate venv | 1 folder | Delete |
| **Total to Archive** | **9-11 files** | - |

---

**After archiving, your project will be cleaner and easier to maintain!**

**Date:** May 6, 2026  
**Prepared By:** GitHub Copilot
