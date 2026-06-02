"""
Builder Floor Rent Model Trainer
==================================
Mirrors bf_model_trainer.py but trained on monthly rent data.

Target  : log1p(price_numeric) = log1p(monthly_rent_INR)
Back-calc at inference: monthly_rent = expm1(model.predict(X))

Feature set is identical to the BF sell model so that the API's
build_features() helper can populate inference rows without changes.
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

_PAT_NUM  = re.compile(r"^(\d+)\s+of\s+(\d+)$")
_PAT_GND  = re.compile(r"^Ground(?:\s+Floor)?\s+of\s+(\d+)$", re.IGNORECASE)
_PAT_UBAS = re.compile(r"^Upper\s+Basement\s+of\s+(\d+)$", re.IGNORECASE)
_PAT_LBAS = re.compile(r"^Lower\s+Basement\s+of\s+(\d+)$", re.IGNORECASE)

_BASE_NUMERIC_COLS = [
    "bhk", "bathrooms", "balconies", "covered_area_sqft",
    "is_parking", "is_pool", "is_main_road", "is_garden_park",
    "is_gated", "is_corner", "circle_rate",
    "is_ground_floor", "is_top_floor",
    "floor_low", "floor_medium", "floor_high",
]

_MONTHLY_RENT_MIN = 3_000
_MONTHLY_RENT_MAX = 500_000
_RENT_PPSF_MIN    = 3
_RENT_PPSF_MAX    = 500
_AREA_MIN         = 150
_AREA_MAX         = 8_000


class BfRentModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        config: BfRentModelTrainerConfig = BfRentModelTrainerConfig(),
    ):
        self.data_transformation_artifact = data_transformation_artifact
        self.config = config

    @staticmethod
    def _info(msg: str) -> None:
        print(msg)
        logging.info(msg)

    @staticmethod
    def _warn(msg: str) -> None:
        print(msg)
        logging.warning(msg)

    @staticmethod
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

    @staticmethod
    def _kmeans_impute(df: pd.DataFrame, cols: list, n_clusters: int = 10, seed: int = 42) -> pd.DataFrame:
        df = df.copy()
        to_impute = [c for c in cols if c in df.columns and df[c].isna().sum() > 0]
        if not to_impute:
            return df
        leakage = {"price_per_sqft", "price_numeric"}
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

    @staticmethod
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

    @staticmethod
    def _filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
        mask = (
            df["bhk"].between(1, 6)
            & df["covered_area_sqft"].between(_AREA_MIN, _AREA_MAX)
            & df["price_numeric"].between(_MONTHLY_RENT_MIN, _MONTHLY_RENT_MAX)
            & df["price_per_sqft"].between(_RENT_PPSF_MIN, _RENT_PPSF_MAX)
            & (df.get("total_floors", pd.Series(0, index=df.index)).fillna(0) <= 50)
        )
        return df[mask].copy()

    @staticmethod
    def _add_floor_level_features(df: pd.DataFrame) -> pd.DataFrame:
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

    @staticmethod
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

    def _get_models(self) -> dict:
        models: dict = {
            "RandomForest": RandomForestRegressor(
                n_estimators=300, n_jobs=-1, random_state=self.config.random_state,
            )
        }
        if HAS_XGB:
            models["XGBoost"] = XGBRegressor(
                n_estimators=300, learning_rate=0.05,
                random_state=self.config.random_state, n_jobs=-1, verbosity=0,
            )
        if HAS_LGBM:
            models["LightGBM"] = LGBMRegressor(
                n_estimators=300, learning_rate=0.05,
                random_state=self.config.random_state, n_jobs=-1, verbose=-1,
            )
        return models

    def initiate_model_training(self) -> BfRentModelTrainerArtifact:
        try:
            self._info("============ BF Rent Model Training Started ============")
            os.makedirs(self.config.model_dir, exist_ok=True)

            # 1. Load builder_floor rent rows
            df_all = pd.read_csv(
                self.data_transformation_artifact.transformed_file_path, low_memory=False,
            )
            type_col = "property_type_grouped" if "property_type_grouped" in df_all.columns else "property_type"
            df = df_all[
                df_all[type_col].str.lower().str.contains("builder_floor|builder floor", na=False)
            ].copy()
            self._info(f"Builder floor rent rows: {len(df)}")
            if len(df) < 50:
                raise ValueError(f"Too few builder floor rent rows ({len(df)}) to train model.")

            # 2. Ensure numeric types
            for col in [
                "bhk", "bathrooms", "balconies", "covered_area_sqft",
                "is_parking", "is_pool", "is_main_road", "is_garden_park",
                "is_gated", "is_corner", "circle_rate", "price_numeric",
                "price_per_sqft", "latitude", "longitude",
            ]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df["latitude"]  = df["latitude"].fillna(28.6)
            df["longitude"] = df["longitude"].fillna(77.2)

            if "price_per_sqft" not in df.columns or df["price_per_sqft"].isna().all():
                df["price_per_sqft"] = df["price_numeric"] / df["covered_area_sqft"].replace(0, np.nan)

            # 3. Parse floors
            df = self._parse_floors(df)
            before = len(df)
            df = df.dropna(subset=["current_floor", "total_floors"])
            self._info(f"Rows after floor parse: {len(df)} (dropped {before - len(df)} unparsable)")
            df = df[df["total_floors"] <= 100].copy()

            # 4. Drop nulls
            before = len(df)
            df = df.dropna(subset=["circle_rate", "price_numeric"])
            self._info(f"Rows after null drop: {len(df)} (dropped {before - len(df)})")

            # 5. KMeans impute
            df = self._kmeans_impute(df, cols=["bathrooms", "balconies"])

            # 6. OHE categoricals
            df = self._encode_categoricals(df)

            # 7. Outlier filter
            before = len(df)
            df = self._filter_outliers(df)
            self._info(f"Rows after outlier filter: {len(df)} (removed {before - len(df)})")
            if len(df) < 50:
                raise ValueError(f"Too few rows remaining after outlier filter ({len(df)}).")

            # 8. Floor-level features
            df = self._add_floor_level_features(df)

            # 9. Build feature matrix
            ohe_cols  = [c for c in df.columns if c.startswith(("age_", "furn_", "facing_"))]
            base_cols = [c for c in _BASE_NUMERIC_COLS if c in df.columns]
            lat_lon   = [c for c in ["latitude", "longitude"] if c in df.columns]

            X     = df[base_cols + ohe_cols + lat_lon].copy()
            X     = X.replace([np.inf, -np.inf], np.nan).fillna(0)

            # Target: log1p(monthly_rent)
            y_log = np.log1p(df["price_numeric"])

            self._info(f"Feature matrix (pre-Voronoi): {X.shape}")

            # 10. Train / test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_log, test_size=self.config.test_size, random_state=self.config.random_state,
            )

            # 11. Voronoi
            X_train, X_test, kmeans = self._add_voronoi(
                X_train, X_test, n_cells=self.config.n_voronoi_clusters, seed=self.config.random_state,
            )

            # 12. Align columns
            all_cols = X_train.columns.tolist()
            X_test   = X_test.reindex(columns=all_cols, fill_value=0)

            # 13. CV model selection
            models = self._get_models()
            best_name, best_score = None, -np.inf
            cv_results: dict = {}
            for name, model in models.items():
                scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
                cv_results[name] = scores
                self._info(f"  {name} CV R2: {scores.mean():.4f} ± {scores.std():.4f}")
                if scores.mean() > best_score:
                    best_score, best_name = scores.mean(), name

            self._info(f"Best model: {best_name}")
            final_model = models[best_name]
            final_model.fit(X_train, y_train)

            # 14. Evaluate
            y_pred_log  = final_model.predict(X_test)
            pred_rent   = np.expm1(y_pred_log)
            actual_rent = np.expm1(np.asarray(y_test, dtype=float))

            mae  = mean_absolute_error(actual_rent, pred_rent)
            mape = mean_absolute_percentage_error(actual_rent, pred_rent)
            r2   = r2_score(actual_rent, pred_rent)
            self._info(f"Monthly rent  MAE={mae:.0f} | MAPE={mape*100:.2f}% | R2={r2:.4f}")

            # 15. Save bundle
            bundle = {
                "model": final_model,
                "kmeans": kmeans,
                "features": X_train.columns.tolist(),
                "target": "log1p_monthly_rent",
                "prediction_formula": "monthly_rent = expm1(model.predict(X))",
                "best_model_name": best_name,
            }
            joblib.dump(bundle,                     self.config.model_file_path)
            joblib.dump(X_train.columns.tolist(),   self.config.feature_columns_file_path)
            joblib.dump(kmeans,                     self.config.voronoi_file_path)

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
