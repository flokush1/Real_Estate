"""
Builder Floor Model Trainer
============================
Matches notebooks/notebooks/sell/bf/builder_floor_model.ipynb exactly.

Target: log1p(price_by_circle) = log1p(price_per_sqft / circle_rate)
Back-calculation at inference: price_per_sqft = expm1(pred) × circle_rate

Key improvements over previous version:
  - Target is log1p(price_by_circle) instead of price_per_sqft directly
  - Floors parsed via regex "X of Y" → current_floor/total_floors
  - Floor-level buckets computed from quantiles (not hardcoded thresholds)
  - Outliers removed via domain hard rules (not IsolationForest)
  - bathrooms/balconies imputed via KMeans cluster-median (not fillna 0)
  - Voronoi KMeans fitted on X_train only (no data leakage)
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
    BfModelTrainerArtifact,
    BfModelTrainerConfig,
    DataTransformationArtifact,
)
from real_estate.exception.exception import RealEstateException
from real_estate.logging.logger import logging

HAS_XGB = True
HAS_LGBM = True
HAS_MLFLOW = True

try:
    from xgboost import XGBRegressor
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
except ImportError:
    HAS_LGBM = False

try:
    import mlflow
    import mlflow.sklearn
except ImportError:
    HAS_MLFLOW = False

try:
    from real_estate.utils.tb_analysis import log_tb_analysis
    HAS_TB_ANALYSIS = True
except Exception:
    HAS_TB_ANALYSIS = False

warnings.filterwarnings("ignore")

# ── Floor regex patterns (same as notebook) ─────────────────
_PAT_NUM  = re.compile(r"^(\d+)\s+of\s+(\d+)$")
_PAT_GND  = re.compile(r"^Ground(?:\s+Floor)?\s+of\s+(\d+)$", re.IGNORECASE)
_PAT_UBAS = re.compile(r"^Upper\s+Basement\s+of\s+(\d+)$", re.IGNORECASE)
_PAT_LBAS = re.compile(r"^Lower\s+Basement\s+of\s+(\d+)$", re.IGNORECASE)

# ── Base numeric feature columns (before Voronoi) ───────────
_BASE_NUMERIC_COLS = [
    "bhk", "bathrooms", "balconies", "covered_area_sqft",
    "is_parking", "is_pool", "is_main_road", "is_garden_park",
    "is_gated", "is_corner", "circle_rate",
    "is_ground_floor", "is_top_floor",
    "floor_low", "floor_medium", "floor_high",
]


class BfModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        config: BfModelTrainerConfig = BfModelTrainerConfig(),
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

    # ── Preprocessing helpers (match notebook functions) ────

    @staticmethod
    def _parse_floors(df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse 'floors' column ("X of Y" format) into:
          current_floor, total_floors, is_ground_floor, is_top_floor, is_basement
        Unparsable rows get NaN and are dropped downstream.
        """
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
    def _kmeans_impute(
        df: pd.DataFrame,
        cols: list,
        n_clusters: int = 10,
        seed: int = 42,
    ) -> pd.DataFrame:
        """KMeans cluster-median imputation for null values in `cols`."""
        df = df.copy()
        to_impute = [c for c in cols if c in df.columns and df[c].isna().sum() > 0]
        if not to_impute:
            return df

        leakage = {"price_per_sqft", "price_by_circle", "price_numeric"}
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
        """OHE for age_of_property, furnishing_type, facing_direction."""
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
        """Domain hard-rule outlier removal matching the BF notebook."""
        mask = (
            df["bhk"].between(1, 6)
            & df["covered_area_sqft"].between(350, 5000)
            & df["price_per_sqft"].between(2000, 40000)
            & (df.get("total_floors", pd.Series(0, index=df.index)).fillna(0) <= 50)
        )
        return df[mask].copy()

    @staticmethod
    def _add_floor_level_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert current_floor into quantile-based buckets (on the full df passed):
          floor_low  : current_floor <= Q25
          floor_med  : Q25 < current_floor < Q75
          floor_high : current_floor >= Q75
        Drops current_floor, total_floors, is_basement afterwards.
        """
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
    def _add_voronoi(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        n_cells: int,
        seed: int,
    ) -> tuple:
        """
        Fit KMeans on X_train lat/lon ONLY (no leakage).
        Adds voronoi_dist_to_seed + vor_cell_0..N-1 to both.
        Drops latitude, longitude from both.
        Returns X_train_aug, X_test_aug, fitted_kmeans.
        """
        latlon_tr = X_train[["latitude", "longitude"]].values
        latlon_te = X_test[["latitude", "longitude"]].values

        kmeans = KMeans(n_clusters=n_cells, random_state=seed, n_init=10)
        kmeans.fit(latlon_tr)
        centers = kmeans.cluster_centers_

        def _apply(X: pd.DataFrame, latlon: np.ndarray) -> pd.DataFrame:
            X = X.copy()
            cell_ids = kmeans.predict(latlon)
            dists = np.sqrt(((latlon - centers[cell_ids]) ** 2).sum(axis=1))
            X["voronoi_dist_to_seed"] = dists
            cell_cat = pd.Categorical(cell_ids, categories=list(range(n_cells)))
            ohe = pd.get_dummies(
                pd.Series(cell_cat, index=X.index), prefix="vor_cell", dtype=int
            )
            X = pd.concat([X, ohe], axis=1)
            X.drop(columns=["latitude", "longitude"], errors="ignore", inplace=True)
            return X

        return _apply(X_train, latlon_tr), _apply(X_test, latlon_te), kmeans

    def _get_models(self) -> dict:
        models: dict = {
            "RandomForest": RandomForestRegressor(
                n_estimators=300, n_jobs=-1,
                random_state=self.config.random_state,
            )
        }
        if HAS_XGB:
            models["XGBoost"] = XGBRegressor(
                n_estimators=300, learning_rate=0.05,
                random_state=self.config.random_state, n_jobs=-1,
                verbosity=0,
            )
        if HAS_LGBM:
            models["LightGBM"] = LGBMRegressor(
                n_estimators=300, learning_rate=0.05,
                random_state=self.config.random_state, n_jobs=-1,
                verbose=-1,
            )
        return models

    def initiate_model_training(self) -> BfModelTrainerArtifact:
        try:
            self._info("============ BF Model Training Started ============")
            os.makedirs(self.config.model_dir, exist_ok=True)

            # ── 1. Load & filter builder floor rows ───────────
            df_all = pd.read_csv(
                self.data_transformation_artifact.transformed_file_path,
                low_memory=False,
            )
            type_col = (
                "property_type_grouped"
                if "property_type_grouped" in df_all.columns
                else "property_type"
            )
            df = df_all[
                df_all[type_col].str.lower().str.contains(
                    "builder_floor|builder floor", na=False
                )
            ].copy()
            self._info(f"Builder floor rows: {len(df)}")
            if len(df) < 50:
                raise ValueError(f"Too few builder floor rows ({len(df)}) to train model.")

            # ── 2. Ensure numeric types ────────────────────────
            for col in [
                "bhk", "bathrooms", "balconies", "covered_area_sqft",
                "is_parking", "is_pool", "is_main_road", "is_garden_park",
                "is_gated", "is_corner", "circle_rate", "price_per_sqft",
                "latitude", "longitude",
            ]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df["latitude"]  = df["latitude"].fillna(28.6)
            df["longitude"] = df["longitude"].fillna(77.2)

            # ── 3. Parse floors (regex) ────────────────────────
            df = self._parse_floors(df)
            before = len(df)
            df = df.dropna(subset=["current_floor", "total_floors"])
            self._info(
                f"Rows after floor parse: {len(df)} (dropped {before - len(df)} unparsable)"
            )
            # Remove suspicious total_floors (looks like a year)
            df = df[df["total_floors"] <= 100].copy()

            # ── 4. Drop nulls in price / circle_rate ──────────
            before = len(df)
            df = df.dropna(subset=["circle_rate", "price_per_sqft"])
            self._info(
                f"Rows after price/circle_rate null drop: {len(df)} "
                f"(dropped {before - len(df)})"
            )

            # ── 5. KMeans impute bathrooms/balconies ──────────
            df = self._kmeans_impute(df, cols=["bathrooms", "balconies"])

            # ── 6. OHE: age / furnishing / facing ─────────────
            df = self._encode_categoricals(df)

            # ── 7. Domain outlier filter ──────────────────────
            before = len(df)
            df = self._filter_outliers(df)
            self._info(
                f"Rows after outlier filter: {len(df)} (removed {before - len(df)})"
            )
            if len(df) < 50:
                raise ValueError(
                    f"Too few rows remaining after outlier filter ({len(df)})."
                )

            # ── 8. Floor-level features (quantile-based) ──────
            df = self._add_floor_level_features(df)

            # ── 9. Compute price_by_circle (target) ───────────
            safe_cr = df["circle_rate"].replace(0, np.nan)
            df["price_by_circle"] = df["price_per_sqft"] / safe_cr
            before = len(df)
            df = df[np.isfinite(df["price_by_circle"])].copy()
            self._info(
                f"Rows after price_by_circle filter: {len(df)} "
                f"(dropped {before - len(df)} inf/NaN)"
            )

            # ── 10. Build base feature matrix (no Voronoi yet) ─
            ohe_cols  = [c for c in df.columns if c.startswith(("age_", "furn_", "facing_"))]
            base_cols = [c for c in _BASE_NUMERIC_COLS if c in df.columns]
            lat_lon   = [c for c in ["latitude", "longitude"] if c in df.columns]

            X     = df[base_cols + ohe_cols + lat_lon].copy()
            X     = X.replace([np.inf, -np.inf], np.nan).fillna(0)
            y_log = np.log1p(df["price_by_circle"])
            meta  = df[["circle_rate", "covered_area_sqft", "price_per_sqft"]].copy()

            self._info(f"Feature matrix (pre-Voronoi): {X.shape}")

            # ── 11. Train / test split ────────────────────────
            X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
                X, y_log, meta,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
            )

            # ── 12. Voronoi features (train-only KMeans) ──────
            X_train, X_test, kmeans = self._add_voronoi(
                X_train, X_test,
                n_cells=self.config.n_voronoi_clusters,
                seed=self.config.random_state,
            )
            self._info(f"After Voronoi: X_train={X_train.shape}  X_test={X_test.shape}")

            # Align columns (test may miss rare OHE categories)
            all_cols = X_train.columns.tolist()
            X_test   = X_test.reindex(columns=all_cols, fill_value=0)

            # ── 13. CV model selection ────────────────────────
            models = self._get_models()
            best_name, best_score = None, -np.inf
            cv_results: dict = {}
            for name, model in models.items():
                scores = cross_val_score(
                    model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1
                )
                cv_results[name] = scores
                self._info(f"  {name} CV R2: {scores.mean():.4f} ± {scores.std():.4f}")
                if scores.mean() > best_score:
                    best_score, best_name = scores.mean(), name

            self._info(f"Best model: {best_name}")
            final_model = models[best_name]
            final_model.fit(X_train, y_train)

            # ── 14. Evaluate — back-calculate to price_per_sqft
            y_pred_log  = final_model.predict(X_test)
            pred_pbc    = np.expm1(y_pred_log)
            actual_pbc  = np.expm1(np.asarray(y_test, dtype=float))
            cr_test     = meta_test["circle_rate"].values
            pred_ppsf   = pred_pbc * cr_test
            actual_ppsf = meta_test["price_per_sqft"].values

            mae    = mean_absolute_error(actual_ppsf, pred_ppsf)
            mape   = mean_absolute_percentage_error(actual_ppsf, pred_ppsf)
            r2     = r2_score(actual_ppsf, pred_ppsf)
            r2_pbc = r2_score(actual_pbc, pred_pbc)

            self._info(
                f"price_per_sqft  MAE={mae:.2f} | MAPE={mape*100:.2f}% | R2={r2:.4f}"
            )
            self._info(f"price_by_circle R2={r2_pbc:.4f}")

            # ── 15. Save artifacts ────────────────────────────
            bundle = {
                "model": final_model,
                "kmeans": kmeans,
                "features": X_train.columns.tolist(),
                "target": "log1p_price_by_circle",
                "prediction_formula": (
                    "price_per_sqft = expm1(model.predict(X)) * circle_rate"
                ),
                "best_model_name": best_name,
            }
            joblib.dump(bundle,                             self.config.model_file_path)
            joblib.dump(X_train.columns.tolist(),           self.config.feature_columns_file_path)
            joblib.dump(kmeans,                             self.config.voronoi_file_path)

            # ── 16. MLflow logging (non-blocking) ─────────────
            if HAS_MLFLOW:
                try:
                    run_name = (
                        f"bf_v{self.config.version}" if self.config.version > 0 else "bf"
                    )
                    # Experiment is set by the training pipeline ("builder-floor-price-model").
                    # Do NOT call set_experiment here — it would override to wrong name.
                    with mlflow.start_run(run_name=run_name):
                        # ── Tags ──────────────────────────────────────────
                        mlflow.set_tags({
                            "property_type": "builder_floor",
                            "model_version": str(self.config.version),
                        })
                        # ── Params: pipeline + best model hyperparams ─────
                        _hp_keys = (
                            "n_estimators", "learning_rate", "max_depth",
                            "min_samples_split", "min_samples_leaf",
                            "subsample", "colsample_bytree", "num_leaves",
                        )
                        best_hp = {
                            f"hp_{k}": str(v)
                            for k, v in final_model.get_params().items()
                            if k in _hp_keys
                        }
                        mlflow.log_params({
                            "model_type":         best_name,
                            "version":            self.config.version,
                            "n_voronoi_clusters": self.config.n_voronoi_clusters,
                            "test_size":          self.config.test_size,
                            "n_training_rows":    len(X_train),
                            "n_test_rows":        len(X_test),
                            "n_features":         X_train.shape[1],
                            "target":             "log1p_price_by_circle",
                            **best_hp,
                        })
                        # ── Test metrics ──────────────────────────────────
                        _rmse_ppsf      = float(np.sqrt(np.mean((actual_ppsf - pred_ppsf) ** 2)))
                        _median_ae_ppsf = float(np.median(np.abs(actual_ppsf - pred_ppsf)))
                        mlflow.log_metrics({
                            "mae_ppsf":       float(mae),
                            "mape_ppsf":      float(mape),
                            "rmse_ppsf":      _rmse_ppsf,
                            "median_ae_ppsf": _median_ae_ppsf,
                            "r2_ppsf":        float(r2),
                            "r2_pbc":         float(r2_pbc),
                        })
                        # ── CV summary (best model) ───────────────────────
                        _best_cv = cv_results[best_name]
                        mlflow.log_metrics({
                            "cv_r2_best_mean": float(_best_cv.mean()),
                            "cv_r2_best_std":  float(_best_cv.std()),
                        })
                        # ── Per-fold CV steps (best model) ────────────────
                        for fold_idx, fold_r2 in enumerate(_best_cv):
                            mlflow.log_metric("cv_r2", float(fold_r2), step=fold_idx)
                        # ── Per-candidate CV mean + std ───────────────────
                        for cand_name, cand_scores in cv_results.items():
                            _key = cand_name.lower().replace(" ", "_")
                            mlflow.log_metric(f"cv_r2_{_key}_mean", float(cand_scores.mean()))
                            mlflow.log_metric(f"cv_r2_{_key}_std",  float(cand_scores.std()))
                        # ── Artifacts ─────────────────────────────────────
                        mlflow.log_artifact(
                            self.config.model_file_path, artifact_path="model"
                        )
                        mlflow.log_artifact(
                            self.config.feature_columns_file_path, artifact_path="model"
                        )
                        try:
                            mlflow.sklearn.log_model(
                                final_model, artifact_path="sklearn_model"
                            )
                        except Exception:
                            pass
                        self._info(
                            f"MLflow run logged: {mlflow.active_run().info.run_id}"
                        )
                except Exception as _e:
                    self._warn(f"MLflow logging skipped (non-fatal): {_e}")

            # ── 17. TensorBoard deep-dive analysis ──────────
            if HAS_TB_ANALYSIS:
                try:
                    import os as _os
                    tb_dir = _os.path.join(
                        "artifact", "tensorboard", "bf",
                        f"v{self.config.version}" if self.config.version > 0 else "v0",
                    )
                    log_tb_analysis(
                        log_dir       = tb_dir,
                        property_type = "bf",
                        version       = self.config.version,
                        final_model   = final_model,
                        X_train       = X_train,
                        X_test        = X_test,
                        y_train       = y_train,
                        y_test        = y_test,
                        y_pred_log    = y_pred_log,
                        actual_ppsf   = actual_ppsf,
                        pred_ppsf     = pred_ppsf,
                        cv_results    = cv_results,
                        best_model_name = best_name,
                        mae           = mae,
                        mape          = mape,
                        r2            = r2,
                        r2_target     = r2_pbc,
                    )
                except Exception as _tb_e:
                    self._warn(f"TensorBoard analysis skipped (non-fatal): {_tb_e}")

            self._info(f"Saved bf model → {self.config.model_file_path}")
            self._info("============ BF Model Training Finished ============")

            return BfModelTrainerArtifact(
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
