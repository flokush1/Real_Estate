"""
Apartment Rent Model Trainer
=============================
Mirrors apt_model_trainer.py but trained on monthly rent data.

Target  : log1p(price_numeric) = log1p(monthly_rent_INR)
Back-calc at inference: monthly_rent = expm1(model.predict(X))

Feature set is identical to the apartment sell model so that the API's
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
    AptRentModelTrainerArtifact,
    AptRentModelTrainerConfig,
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

# ── Floor regex patterns ─────────────────────────────────────
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

_LOCALITY_TIERS = ["budget", "mid", "high", "luxury"]

# ── Rent-specific price filters ──────────────────────────────
_MONTHLY_RENT_MIN = 3_000      # INR / month
_MONTHLY_RENT_MAX = 500_000    # INR / month
_RENT_PPSF_MIN    = 3          # INR / sqft / month
_RENT_PPSF_MAX    = 500        # INR / sqft / month
_AREA_MIN         = 150        # sqft
_AREA_MAX         = 8_000      # sqft


class AptRentModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        config: AptRentModelTrainerConfig = AptRentModelTrainerConfig(),
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
    def _add_floor_level_features(
        df: pd.DataFrame, q25=None, q75=None, return_thresholds=False
    ) -> pd.DataFrame:
        """
        Convert current_floor into floor bands using train-based thresholds.
        If q25/q75 not provided, compute from df (train mode).
        If provided, use them (test mode - no leakage).
        """
        df = df.copy()
        if "current_floor" not in df.columns or df["current_floor"].isna().all():
            df["floor_low"] = 0; df["floor_medium"] = 1; df["floor_high"] = 0
            drop_cols = [c for c in ["current_floor", "total_floors", "is_basement"] if c in df.columns]
            df.drop(columns=drop_cols, inplace=True)
            if return_thresholds:
                return df, 0.0, 0.0
            return df
        
        # Compute thresholds from train if not provided
        if q25 is None:
            q25 = df["current_floor"].quantile(0.25)
        if q75 is None:
            q75 = df["current_floor"].quantile(0.75)
        if q25 > q75:
            q25, q75 = q75, q25
        
        df["floor_level"] = np.where(
            df["current_floor"] <= q25, "low",
            np.where(df["current_floor"] >= q75, "high", "medium"),
        )
        for lvl in ["low", "medium", "high"]:
            df[f"floor_{lvl}"] = (df["floor_level"] == lvl).astype(int)
        df.drop(columns=["floor_level"], inplace=True)
        
        drop_cols = [c for c in ["current_floor", "total_floors", "is_basement"] if c in df.columns]
        df.drop(columns=drop_cols, inplace=True)
        
        if return_thresholds:
            return df, float(q25), float(q75)
        return df

    @staticmethod
    def _fit_locality_tier_map(
        train_data: pd.DataFrame, 
        locality_col: str = "locality",
        price_col: str = "price_per_sqft"
    ) -> dict:
        """
        Build locality -> tier mapping from locality median price_per_sqft (not circle_rate).
        Tiers based on quartiles of locality medians:
          budget <= Q25 < mid <= Q50 < high <= Q75 < luxury
        """
        if locality_col not in train_data.columns or price_col not in train_data.columns:
            return {}
        
        # Compute locality medians from train price_per_sqft
        loc_medians = train_data.groupby(locality_col)[price_col].median().dropna()
        if len(loc_medians) < 4:
            return {}
        
        q25 = loc_medians.quantile(0.25)
        q50 = loc_medians.quantile(0.50)
        q75 = loc_medians.quantile(0.75)

        def _tier(price: float) -> str:
            if price <= q25: return "budget"
            if price <= q50: return "mid"
            if price <= q75: return "high"
            return "luxury"

        return {loc: _tier(median_price) for loc, median_price in loc_medians.items()}

    @staticmethod
    def _add_locality_tier_features(
        X: pd.DataFrame, tier_map: dict, locality_col: str = "locality", default_tier: str = "budget"
    ) -> pd.DataFrame:
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

    def initiate_model_training(self) -> AptRentModelTrainerArtifact:
        try:
            self._info("============ Apt Rent Model Training Started ============")
            os.makedirs(self.config.model_dir, exist_ok=True)

            # 1. Load apartment rent rows
            df_all = pd.read_csv(
                self.data_transformation_artifact.transformed_file_path, low_memory=False,
            )
            type_col = "property_type_grouped" if "property_type_grouped" in df_all.columns else "property_type"
            df = df_all[
                df_all[type_col].str.lower().str.contains("apartment", na=False)
            ].copy()
            self._info(f"Apartment rent rows: {len(df)}")
            if len(df) < 50:
                raise ValueError(f"Too few apartment rent rows ({len(df)}) to train model.")

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

            # Derive price_per_sqft from price_numeric if needed
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

            # 7. Domain outlier filter (rent-specific ranges)
            before = len(df)
            df = self._filter_outliers(df)
            self._info(f"Rows after outlier filter: {len(df)} (removed {before - len(df)})")
            if len(df) < 50:
                raise ValueError(f"Too few rows remaining after outlier filter ({len(df)}).")

            # 8. Build feature matrix BEFORE floor-level features
            # Keep locality for tier fitting, lat/lon for Voronoi, current_floor for bands
            ohe_cols  = [c for c in df.columns if c.startswith(("age_", "furn_", "facing_"))]
            base_cols = [c for c in _BASE_NUMERIC_COLS if c in df.columns and c not in ["floor_low", "floor_medium", "floor_high"]]
            lat_lon   = [c for c in ["latitude", "longitude"] if c in df.columns]
            loc_col   = ["locality"] if "locality" in df.columns else []
            floor_cols = [c for c in ["current_floor", "total_floors", "is_basement"] if c in df.columns]

            X     = df[base_cols + ohe_cols + lat_lon + loc_col + floor_cols].copy()
            X     = X.replace([np.inf, -np.inf], np.nan)
            num_cols = X.select_dtypes(include="number").columns
            X[num_cols] = X[num_cols].fillna(0)

            # Target: log1p(monthly_rent)
            y_log = np.log1p(df["price_numeric"])

            self._info(f"Feature matrix (pre-split): {X.shape}")

            # 9. Train / test split (BEFORE floor bands and locality tiers to prevent leakage)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_log, test_size=self.config.test_size, random_state=self.config.random_state,
            )

            # 10. Floor-level features using TRAIN-based thresholds (no leakage)
            X_train, floor_q25, floor_q75 = self._add_floor_level_features(
                X_train, return_thresholds=True
            )
            X_test = self._add_floor_level_features(X_test, q25=floor_q25, q75=floor_q75)
            self._info(f"Floor bands: Q25={floor_q25:.2f}, Q75={floor_q75:.2f}")

            # 11. Locality tier map from TRAIN locality median price_per_sqft (no leakage)
            # Need to map train indices back to original df to get price_per_sqft
            train_indices = X_train.index
            train_data_for_tiers = df.loc[train_indices, ["locality", "price_per_sqft"]].copy()
            locality_tier_map = self._fit_locality_tier_map(
                train_data_for_tiers, locality_col="locality", price_col="price_per_sqft"
            )
            X_train = self._add_locality_tier_features(X_train, locality_tier_map)
            X_test  = self._add_locality_tier_features(X_test,  locality_tier_map)
            self._info(f"Locality tier map: {len(locality_tier_map)} localities")

            # 12. Voronoi features (already correct - fits on train only)
            X_train, X_test, kmeans = self._add_voronoi(
                X_train, X_test, n_cells=self.config.n_voronoi_clusters, seed=self.config.random_state,
            )

            # 13. Align columns
            all_cols = X_train.columns.tolist()
            X_test   = X_test.reindex(columns=all_cols, fill_value=0)

            # 14. CV model selection
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

            # 15. Evaluate on monthly rent
            y_pred_log = final_model.predict(X_test)
            pred_rent  = np.expm1(y_pred_log)
            actual_rent = np.expm1(np.asarray(y_test, dtype=float))

            mae  = mean_absolute_error(actual_rent, pred_rent)
            mape = mean_absolute_percentage_error(actual_rent, pred_rent)
            r2   = r2_score(actual_rent, pred_rent)
            self._info(f"Monthly rent  MAE={mae:.0f} | MAPE={mape*100:.2f}% | R2={r2:.4f}")

            # 16. Save bundle (same format as apt sell model so _unwrap_model_bundle works)
            bundle = {
                "model": final_model,
                "kmeans": kmeans,
                "locality_tier_map": locality_tier_map,
                "features": X_train.columns.tolist(),
                "target": "log1p_monthly_rent",
                "prediction_formula": "monthly_rent = expm1(model.predict(X))",
                "best_model_name": best_name,
            }
            joblib.dump(bundle,                     self.config.model_file_path)
            joblib.dump(X_train.columns.tolist(),   self.config.feature_columns_file_path)
            joblib.dump(kmeans,                     self.config.voronoi_file_path)

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
