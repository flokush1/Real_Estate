"""
Rent Model Trainers (Apartment + Builder Floor)
================================================
Target: log1p(price_numeric) where price_numeric = monthly rent in INR.
At inference: monthly_rent = expm1(model.predict(X)[0])
Annual rent yield = (monthly_rent * 12) / buy_price * 100

Feature structure mirrors the sell models so the same build_features()
helper in api/main.py works unchanged.

Apartment rent:
  - Filters apartment rows from rent_cleaned.csv
  - Floor features: floor_low / floor_medium / floor_high (quantile buckets)
  - Locality tier OHE (fit on train only)

Builder Floor rent:
  - Filters builder_floor rows from rent_cleaned.csv
  - Floor features: current_floor, total_floors, is_ground_floor, is_top_floor, is_basement
"""

import os
import re
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from real_estate.entity import (
    AptRentModelTrainerArtifact,
    AptRentModelTrainerConfig,
    BfRentModelTrainerArtifact,
    BfRentModelTrainerConfig,
    DataTransformationArtifact,
)
from real_estate.exception.exception import RealEstateException
from real_estate.logging.logger import logging

HAS_XGB = True
HAS_LGBM = True

try:
    from xgboost import XGBRegressor
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
except ImportError:
    HAS_LGBM = False

warnings.filterwarnings("ignore")

# ── Floor regex patterns ──────────────────────────────────────
_PAT_NUM  = re.compile(r"^(\d+)\s+of\s+(\d+)$")
_PAT_GND  = re.compile(r"^Ground(?:\s+Floor)?\s+of\s+(\d+)$", re.IGNORECASE)
_PAT_UBAS = re.compile(r"^Upper\s+Basement\s+of\s+(\d+)$", re.IGNORECASE)
_PAT_LBAS = re.compile(r"^Lower\s+Basement\s+of\s+(\d+)$", re.IGNORECASE)

_LOCALITY_TIERS = ["budget", "mid", "high", "luxury"]

# Base numeric features — same as sell models so build_features() works
_BASE_NUMERIC_COLS = [
    "bhk", "bathrooms", "balconies", "covered_area_sqft",
    "is_parking", "is_pool", "is_main_road", "is_garden_park",
    "is_gated", "is_corner", "circle_rate",
    "is_ground_floor", "is_top_floor",
]


# ─────────────────────────────────────────────────────────────
#  Shared preprocessing helpers
# ─────────────────────────────────────────────────────────────

def _parse_floors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "floors" not in df.columns:
        for col in ["current_floor", "total_floors"]:
            df[col] = np.nan
        for col in ["is_ground_floor", "is_top_floor", "is_basement"]:
            df[col] = 0
        return df

    current, total = [], []
    for val in df["floors"]:
        if pd.isna(val):
            current.append(np.nan); total.append(np.nan); continue
        s = str(val).strip()
        m = _PAT_NUM.match(s)
        if m:
            current.append(int(m.group(1))); total.append(int(m.group(2))); continue
        m = _PAT_GND.match(s)
        if m:
            current.append(0); total.append(int(m.group(1))); continue
        m = _PAT_UBAS.match(s)
        if m:
            current.append(-1); total.append(int(m.group(1))); continue
        m = _PAT_LBAS.match(s)
        if m:
            current.append(-2); total.append(int(m.group(1))); continue
        current.append(np.nan); total.append(np.nan)

    df["current_floor"] = current
    df["total_floors"]  = total
    df["is_ground_floor"] = (df["current_floor"] == 0).astype(int)
    df["is_top_floor"] = (
        (df["current_floor"] == df["total_floors"]) & df["current_floor"].notna()
    ).astype(int)
    df["is_basement"] = (df["current_floor"] < 0).astype(int)
    return df


def _kmeans_impute(df: pd.DataFrame, cols: list, n_clusters: int = 10, seed: int = 42) -> pd.DataFrame:
    df = df.copy()
    to_impute = [c for c in cols if c in df.columns and df[c].isna().sum() > 0]
    if not to_impute:
        return df

    leakage = {"price_numeric", "price_per_sqft", "sqft_price"}
    feat_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in set(to_impute) | leakage and df[c].isna().sum() == 0
    ]
    if not feat_cols:
        for c in to_impute:
            df[c] = df[c].fillna(df[c].median())
        return df

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feat_cols])
    km = KMeans(n_clusters=min(n_clusters, len(df)), random_state=seed, n_init=10)
    df["_cluster"] = km.fit_predict(X_scaled)

    for c in to_impute:
        global_median = df[c].median()
        cluster_medians = df.groupby("_cluster")[c].median().fillna(global_median)
        null_mask = df[c].isna()
        df.loc[null_mask, c] = df.loc[null_mask, "_cluster"].map(cluster_medians)

    df.drop(columns=["_cluster"], inplace=True)
    return df


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, prefix in [
        ("age_of_property",  "age"),
        ("furnishing_type",  "furn"),
        ("facing_direction", "facing"),
    ]:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=prefix, dtype=int)
            df = pd.concat([df, dummies], axis=1)
            df.drop(columns=[col], inplace=True)
    return df


def _add_voronoi(X_train, X_test, n_cells, seed):
    latlon_tr = X_train[["latitude", "longitude"]].values
    latlon_te = X_test[["latitude", "longitude"]].values

    kmeans = KMeans(n_clusters=n_cells, random_state=seed, n_init=10)
    kmeans.fit(latlon_tr)
    centers = kmeans.cluster_centers_

    def _apply(X, latlon):
        X = X.copy()
        cell_ids = kmeans.predict(latlon)
        dists = np.sqrt(((latlon - centers[cell_ids]) ** 2).sum(axis=1))
        X["voronoi_dist_to_seed"] = dists
        cell_cat = pd.Categorical(cell_ids, categories=list(range(n_cells)))
        ohe = pd.get_dummies(pd.Series(cell_cat, index=X.index), prefix="vor_cell", dtype=int)
        X = pd.concat([X, ohe], axis=1)
        X.drop(columns=["latitude", "longitude"], errors="ignore", inplace=True)
        return X

    return _apply(X_train, latlon_tr), _apply(X_test, latlon_te), kmeans


def _fit_locality_tier_map(X_train, locality_col="locality"):
    if locality_col not in X_train.columns or "circle_rate" not in X_train.columns:
        return {}
    loc_cr = X_train.groupby(locality_col)["circle_rate"].median().dropna()
    if len(loc_cr) < 4:
        return {}
    q25, q50, q75 = loc_cr.quantile([0.25, 0.5, 0.75])

    def _tier(cr):
        if cr >= q75: return "luxury"
        if cr >= q50: return "high"
        if cr >= q25: return "mid"
        return "budget"

    return {loc: _tier(cr) for loc, cr in loc_cr.items()}


def _add_locality_tier_features(X, tier_map, locality_col="locality", default_tier="budget"):
    X = X.copy()
    if locality_col not in X.columns or not tier_map:
        for tier in _LOCALITY_TIERS:
            X[f"locality_tier_{tier}"] = 0
        if locality_col in X.columns:
            X.drop(columns=[locality_col], inplace=True)
        return X
    mapped = X[locality_col].map(tier_map).fillna(default_tier)
    for tier in _LOCALITY_TIERS:
        X[f"locality_tier_{tier}"] = (mapped == tier).astype(int)
    X.drop(columns=[locality_col], inplace=True)
    return X


def _get_models(random_state):
    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300, n_jobs=-1, random_state=random_state,
        )
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300, learning_rate=0.05,
            random_state=random_state, n_jobs=-1, verbosity=0,
        )
    if HAS_LGBM:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=300, learning_rate=0.05,
            random_state=random_state, n_jobs=-1, verbose=-1,
        )
    return models


# ─────────────────────────────────────────────────────────────
#  Apartment Rent Model Trainer
# ─────────────────────────────────────────────────────────────

class AptRentModelTrainer:
    """
    Trains a monthly-rent model for apartments.
    Target: log1p(price_numeric)  [monthly rent in INR]
    Features: same structure as AptModelTrainer so build_features() works at inference.
    """

    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        config: AptRentModelTrainerConfig = AptRentModelTrainerConfig(),
    ):
        self.data_transformation_artifact = data_transformation_artifact
        self.config = config

    @staticmethod
    def _info(msg): print(msg); logging.info(msg)
    @staticmethod
    def _warn(msg): print(msg); logging.warning(msg)

    @staticmethod
    def _add_floor_level_features(df: pd.DataFrame) -> pd.DataFrame:
        """Quantile-based floor buckets: floor_low / floor_medium / floor_high."""
        df = df.copy()
        if "current_floor" not in df.columns or df["current_floor"].isna().all():
            df["floor_low"] = 0; df["floor_medium"] = 1; df["floor_high"] = 0
        else:
            q25 = df["current_floor"].quantile(0.25)
            q75 = df["current_floor"].quantile(0.75)
            df["floor_level"] = np.where(
                df["current_floor"] <= q25, "low",
                np.where(df["current_floor"] >= q75, "high", "medium"),
            )
            for lvl in ["low", "medium", "high"]:
                df[f"floor_{lvl}"] = (df["floor_level"] == lvl).astype(int)
            df.drop(columns=["floor_level"], inplace=True)

        drop_cols = [c for c in ["current_floor", "total_floors", "is_basement"] if c in df.columns]
        df.drop(columns=drop_cols, inplace=True)
        return df

    def initiate_model_training(self) -> AptRentModelTrainerArtifact:
        try:
            self._info("============ Apt Rent Model Training Started ============")
            os.makedirs(self.config.model_dir, exist_ok=True)

            # 1. Load apartment rent rows
            df_all = pd.read_csv(
                self.data_transformation_artifact.transformed_file_path,
                low_memory=False,
            )
            type_col = "property_type_grouped" if "property_type_grouped" in df_all.columns else "property_type"
            df = df_all[
                df_all[type_col].str.lower().str.contains("apartment", na=False)
            ].copy()
            self._info(f"Apartment rent rows: {len(df)}")
            if len(df) < 50:
                raise ValueError(f"Too few apartment rent rows ({len(df)}).")

            # 2. Numeric coercion
            for col in ["bhk", "bathrooms", "balconies", "covered_area_sqft",
                        "is_parking", "is_pool", "is_main_road", "is_garden_park",
                        "is_gated", "is_corner", "circle_rate", "price_numeric",
                        "latitude", "longitude"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df["latitude"]  = df["latitude"].fillna(28.6)
            df["longitude"] = df["longitude"].fillna(77.2)

            # 3. Outlier filter: monthly rent 2,000–500,000; area 300–5000; bhk 1–6
            before = len(df)
            df = df[
                df["price_numeric"].between(2000, 500000) &
                df["covered_area_sqft"].between(300, 5000) &
                df["bhk"].between(1, 6)
            ].copy()
            self._info(f"Rows after outlier filter: {len(df)} (removed {before - len(df)})")
            if len(df) < 50:
                raise ValueError(f"Too few rows after outlier filter ({len(df)}).")

            # 4. Drop null circle_rate / price_numeric
            before = len(df)
            df = df.dropna(subset=["circle_rate", "price_numeric"])
            self._info(f"Rows after null drop: {len(df)} (dropped {before - len(df)})")

            # 5. Parse floors
            df = _parse_floors(df)
            before = len(df)
            df = df.dropna(subset=["current_floor", "total_floors"])
            df = df[df["total_floors"] <= 100].copy()
            self._info(f"Rows after floor parse: {len(df)} (dropped {before - len(df)} unparsable)")

            # 6. KMeans impute bathrooms/balconies
            df = _kmeans_impute(df, ["bathrooms", "balconies"])

            # 7. OHE categoricals
            df = _encode_categoricals(df)

            # 8. Floor level features
            df = self._add_floor_level_features(df)

            # 9. Build feature matrix
            ohe_cols  = [c for c in df.columns if c.startswith(("age_", "furn_", "facing_"))]
            base_cols = [c for c in (_BASE_NUMERIC_COLS + ["floor_low", "floor_medium", "floor_high"]) if c in df.columns]
            lat_lon   = [c for c in ["latitude", "longitude"] if c in df.columns]
            loc_col   = ["locality"] if "locality" in df.columns else []

            X = df[base_cols + ohe_cols + lat_lon + loc_col].copy()
            X = X.replace([np.inf, -np.inf], np.nan)
            num_cols = X.select_dtypes(include="number").columns
            X[num_cols] = X[num_cols].fillna(0)

            y_log = np.log1p(df["price_numeric"])
            self._info(f"Feature matrix (pre-Voronoi): {X.shape}")

            # 10. Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_log, test_size=self.config.test_size, random_state=self.config.random_state,
            )

            # 11. Voronoi features
            X_train, X_test, kmeans = _add_voronoi(
                X_train, X_test, n_cells=self.config.n_voronoi_clusters, seed=self.config.random_state,
            )
            self._info(f"After Voronoi: X_train={X_train.shape}  X_test={X_test.shape}")

            # 12. Locality tier
            locality_tier_map = _fit_locality_tier_map(X_train)
            X_train = _add_locality_tier_features(X_train, locality_tier_map)
            X_test  = _add_locality_tier_features(X_test,  locality_tier_map)

            # 13. Align columns
            all_cols = X_train.columns.tolist()
            X_test   = X_test.reindex(columns=all_cols, fill_value=0)

            # 14. CV model selection
            models = _get_models(self.config.random_state)
            best_name, best_score = None, -np.inf
            for name, model in models.items():
                scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
                self._info(f"  {name} CV R2: {scores.mean():.4f} ± {scores.std():.4f}")
                if scores.mean() > best_score:
                    best_score, best_name = scores.mean(), name

            self._info(f"Best model: {best_name}")
            final_model = models[best_name]
            final_model.fit(X_train, y_train)

            # 15. Evaluate
            y_pred_log = final_model.predict(X_test)
            pred_rent  = np.expm1(y_pred_log)
            actual_rent = np.expm1(np.asarray(y_test, dtype=float))

            mae  = mean_absolute_error(actual_rent, pred_rent)
            mape = mean_absolute_percentage_error(actual_rent, pred_rent)
            r2   = r2_score(actual_rent, pred_rent)
            self._info(f"Monthly rent  MAE={mae:.0f} | MAPE={mape*100:.1f}% | R2={r2:.4f}")

            # 16. Save bundle (same format as sell models)
            bundle = {
                "model": final_model,
                "kmeans": kmeans,
                "locality_tier_map": locality_tier_map,
                "features": X_train.columns.tolist(),
                "target": "log1p_monthly_rent",
                "prediction_formula": "monthly_rent = expm1(model.predict(X))",
                "best_model_name": best_name,
            }
            joblib.dump(bundle,                    self.config.model_file_path)
            joblib.dump(X_train.columns.tolist(),  self.config.feature_columns_file_path)
            joblib.dump(kmeans,                    self.config.voronoi_file_path)
            self._info(f"Saved apt rent model → {self.config.model_file_path}")
            self._info("============ Apt Rent Model Training Finished ============")

            return AptRentModelTrainerArtifact(
                model_file_path=self.config.model_file_path,
                feature_columns_file_path=self.config.feature_columns_file_path,
                voronoi_file_path=self.config.voronoi_file_path,
                best_model_name=best_name,
                mae=float(mae),
                mape=float(mape),
                r2=float(r2),
            )

        except Exception as e:
            raise RealEstateException(e, sys)


# ─────────────────────────────────────────────────────────────
#  Builder Floor Rent Model Trainer
# ─────────────────────────────────────────────────────────────

class BfRentModelTrainer:
    """
    Trains a monthly-rent model for builder floors (independent floors).
    Target: log1p(price_numeric)  [monthly rent in INR]
    Features: same structure as BfModelTrainer (uses current_floor/total_floors
    directly rather than quantile buckets, matching the BF sell model API call).
    """

    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        config: BfRentModelTrainerConfig = BfRentModelTrainerConfig(),
    ):
        self.data_transformation_artifact = data_transformation_artifact
        self.config = config

    @staticmethod
    def _info(msg): print(msg); logging.info(msg)
    @staticmethod
    def _warn(msg): print(msg); logging.warning(msg)

    def initiate_model_training(self) -> BfRentModelTrainerArtifact:
        try:
            self._info("============ BF Rent Model Training Started ============")
            os.makedirs(self.config.model_dir, exist_ok=True)

            # 1. Load builder_floor rent rows
            df_all = pd.read_csv(
                self.data_transformation_artifact.transformed_file_path,
                low_memory=False,
            )
            type_col = "property_type_grouped" if "property_type_grouped" in df_all.columns else "property_type"
            df = df_all[
                df_all[type_col].str.lower().str.contains("builder_floor", na=False)
            ].copy()
            self._info(f"Builder floor rent rows: {len(df)}")
            if len(df) < 50:
                raise ValueError(f"Too few builder_floor rent rows ({len(df)}).")

            # 2. Numeric coercion
            for col in ["bhk", "bathrooms", "balconies", "covered_area_sqft",
                        "is_parking", "is_pool", "is_main_road", "is_garden_park",
                        "is_gated", "is_corner", "circle_rate", "price_numeric",
                        "latitude", "longitude"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df["latitude"]  = df["latitude"].fillna(28.6)
            df["longitude"] = df["longitude"].fillna(77.2)

            # 3. Outlier filter: monthly rent 2,000–400,000; area 300–8000; bhk 1–8
            before = len(df)
            df = df[
                df["price_numeric"].between(2000, 400000) &
                df["covered_area_sqft"].between(300, 8000) &
                df["bhk"].between(1, 8)
            ].copy()
            self._info(f"Rows after outlier filter: {len(df)} (removed {before - len(df)})")
            if len(df) < 50:
                raise ValueError(f"Too few rows after outlier filter ({len(df)}).")

            # 4. Drop null circle_rate / price_numeric
            before = len(df)
            df = df.dropna(subset=["circle_rate", "price_numeric"])
            self._info(f"Rows after null drop: {len(df)} (dropped {before - len(df)})")

            # 5. Parse floors
            df = _parse_floors(df)
            before = len(df)
            df = df.dropna(subset=["current_floor", "total_floors"])
            df = df[df["total_floors"] <= 100].copy()
            self._info(f"Rows after floor parse: {len(df)} (dropped {before - len(df)} unparsable)")

            # 6. KMeans impute
            df = _kmeans_impute(df, ["bathrooms", "balconies"])

            # 7. OHE categoricals
            df = _encode_categoricals(df)

            # 8. Build feature matrix (BF uses current_floor/total_floors directly)
            ohe_cols  = [c for c in df.columns if c.startswith(("age_", "furn_", "facing_"))]
            floor_cols = [c for c in ["current_floor", "total_floors", "is_ground_floor", "is_top_floor", "is_basement"] if c in df.columns]
            base_cols = [c for c in _BASE_NUMERIC_COLS if c in df.columns] + floor_cols
            # Remove duplicates preserving order
            seen = set(); base_cols = [c for c in base_cols if not (c in seen or seen.add(c))]
            lat_lon   = [c for c in ["latitude", "longitude"] if c in df.columns]
            loc_col   = ["locality"] if "locality" in df.columns else []

            X = df[base_cols + ohe_cols + lat_lon + loc_col].copy()
            X = X.replace([np.inf, -np.inf], np.nan)
            num_cols = X.select_dtypes(include="number").columns
            X[num_cols] = X[num_cols].fillna(0)

            y_log = np.log1p(df["price_numeric"])
            self._info(f"Feature matrix (pre-Voronoi): {X.shape}")

            # 9. Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_log, test_size=self.config.test_size, random_state=self.config.random_state,
            )

            # 10. Voronoi features
            X_train, X_test, kmeans = _add_voronoi(
                X_train, X_test, n_cells=self.config.n_voronoi_clusters, seed=self.config.random_state,
            )
            self._info(f"After Voronoi: X_train={X_train.shape}  X_test={X_test.shape}")

            # 11. Locality tier (for BF too — helps with price differentiation)
            locality_tier_map = _fit_locality_tier_map(X_train)
            X_train = _add_locality_tier_features(X_train, locality_tier_map)
            X_test  = _add_locality_tier_features(X_test,  locality_tier_map)

            # 12. Align columns
            all_cols = X_train.columns.tolist()
            X_test   = X_test.reindex(columns=all_cols, fill_value=0)

            # 13. CV model selection
            models = _get_models(self.config.random_state)
            best_name, best_score = None, -np.inf
            for name, model in models.items():
                scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
                self._info(f"  {name} CV R2: {scores.mean():.4f} ± {scores.std():.4f}")
                if scores.mean() > best_score:
                    best_score, best_name = scores.mean(), name

            self._info(f"Best model: {best_name}")
            final_model = models[best_name]
            final_model.fit(X_train, y_train)

            # 14. Evaluate
            y_pred_log = final_model.predict(X_test)
            pred_rent   = np.expm1(y_pred_log)
            actual_rent = np.expm1(np.asarray(y_test, dtype=float))

            mae  = mean_absolute_error(actual_rent, pred_rent)
            mape = mean_absolute_percentage_error(actual_rent, pred_rent)
            r2   = r2_score(actual_rent, pred_rent)
            self._info(f"Monthly rent  MAE={mae:.0f} | MAPE={mape*100:.1f}% | R2={r2:.4f}")

            # 15. Save bundle
            bundle = {
                "model": final_model,
                "kmeans": kmeans,
                "locality_tier_map": locality_tier_map,
                "features": X_train.columns.tolist(),
                "target": "log1p_monthly_rent",
                "prediction_formula": "monthly_rent = expm1(model.predict(X))",
                "best_model_name": best_name,
            }
            joblib.dump(bundle,                    self.config.model_file_path)
            joblib.dump(X_train.columns.tolist(),  self.config.feature_columns_file_path)
            joblib.dump(kmeans,                    self.config.voronoi_file_path)
            self._info(f"Saved bf rent model → {self.config.model_file_path}")
            self._info("============ BF Rent Model Training Finished ============")

            return BfRentModelTrainerArtifact(
                model_file_path=self.config.model_file_path,
                feature_columns_file_path=self.config.feature_columns_file_path,
                voronoi_file_path=self.config.voronoi_file_path,
                best_model_name=best_name,
                mae=float(mae),
                mape=float(mape),
                r2=float(r2),
            )

        except Exception as e:
            raise RealEstateException(e, sys)
