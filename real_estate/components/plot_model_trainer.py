import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import (
    ParameterGrid,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler

from real_estate.entity import (
    PlotDataTransformationArtifact,
    PlotModelTrainerArtifact,
    PlotModelTrainerConfig,
)
from real_estate.exception.exception import RealEstateException
from real_estate.logging.logger import logging

HAS_XGB = True
HAS_LGBM = True
HAS_PLOTLY = True
HAS_MLFLOW = True

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

try:
    from xgboost import XGBRegressor
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
except ImportError:
    HAS_LGBM = False

try:
    import plotly.express as px  # type: ignore[import-not-found]
except ImportError:
    HAS_PLOTLY = False


warnings.filterwarnings("ignore")


class PlotModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: PlotDataTransformationArtifact,
        config: PlotModelTrainerConfig = PlotModelTrainerConfig(),
    ):
        self.data_transformation_artifact = data_transformation_artifact
        self.config = config

    @staticmethod
    def _emit_info(message: str) -> None:
        print(message)
        logging.info(message)

    @staticmethod
    def _emit_warning(message: str) -> None:
        print(message)
        logging.warning(message)

    @staticmethod
    def to_price(log_val):
        return np.expm1(log_val)

    @classmethod
    def get_total_price(cls, pred_log_sqft, log_plot_area):
        return cls.to_price(pred_log_sqft) * cls.to_price(log_plot_area)

    @classmethod
    def price_num_mae_scorer(cls, estimator, X, y):
        pred_log_sqft = estimator.predict(X)
        pred_total = cls.get_total_price(pred_log_sqft, X["log_plot_area"])
        actual_total = cls.get_total_price(y, X["log_plot_area"])
        return -mean_absolute_error(actual_total, pred_total)

    @classmethod
    def price_num_r2_scorer(cls, estimator, X, y):
        pred_log_sqft = estimator.predict(X)
        pred_total = cls.get_total_price(pred_log_sqft, X["log_plot_area"])
        actual_total = cls.get_total_price(y, X["log_plot_area"])
        return r2_score(actual_total, pred_total)

    @classmethod
    def evaluate_model_on_price_num(cls, model, X, y):
        pred_log_sqft = model.predict(X)
        actual_total = cls.get_total_price(y, X["log_plot_area"])
        pred_total = cls.get_total_price(pred_log_sqft, X["log_plot_area"])

        mae = mean_absolute_error(actual_total, pred_total)
        mape = mean_absolute_percentage_error(actual_total, pred_total)
        r2 = r2_score(actual_total, pred_total)
        return mae, mape, r2, actual_total, pred_total

    @staticmethod
    def clean_and_validate(df: pd.DataFrame) -> pd.DataFrame:
        required = [
            "latitude",
            "longitude",
            "plot_area",
            "price_per_sqft",
            "circle_rate",
        ]

        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        df = df.copy()

        for col in ["usage_type", "facing_direction"]:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.strip()
                    .str.replace("-", " ", regex=False)
                    .str.title()
                )

        df = df.replace([np.inf, -np.inf], np.nan)

        df = df.dropna(
            subset=[
                "latitude",
                "longitude",
                "plot_area",
                "price_per_sqft",
                "circle_rate",
            ]
        )

        df = df[
            (df["price_per_sqft"] > 0)
            & (df["plot_area"] > 0)
            & (df["circle_rate"] > 0)
        ].copy()

        df["ratio"] = df["price_per_sqft"] / df["circle_rate"].replace(0, np.nan)
        df = df[(df["ratio"] > 0.3) & (df["ratio"] < 25)].copy()
        df = df.drop(columns=["ratio"], errors="ignore")
        return df

    def remove_outliers_split(self, X_train, y_train):
        iso_features = [
            "latitude",
            "longitude",
            "log_plot_area",
            "circle_rate",
        ]

        missing_features = [col for col in iso_features if col not in X_train.columns]
        if missing_features:
            raise ValueError(f"Missing features for outlier removal: {missing_features}")

        data_for_iso = X_train[iso_features].copy()
        data_for_iso["log_target"] = y_train.values
        data_for_iso = data_for_iso.replace([np.inf, -np.inf], np.nan)

        valid_mask = data_for_iso.notna().all(axis=1)

        X_valid = X_train.loc[valid_mask].copy()
        y_valid = y_train.loc[valid_mask].copy()
        data_for_iso = data_for_iso.loc[valid_mask].copy()

        iso = IsolationForest(
            contamination=self.config.contamination,
            random_state=self.config.random_state,
            n_estimators=100,
        )

        preds = iso.fit_predict(data_for_iso)
        keep_mask = preds == 1

        X_clean = X_valid.loc[keep_mask].copy()
        y_clean = y_valid.loc[keep_mask].copy()
        return X_clean, y_clean, iso

    def add_spatial_features(self, X_train, X_test):
        X_train = X_train.copy()
        X_test = X_test.copy()

        scaler = StandardScaler()

        coords_train = X_train[["latitude", "longitude"]]
        coords_test = X_test[["latitude", "longitude"]]

        scaled_train = scaler.fit_transform(coords_train)
        scaled_test = scaler.transform(coords_test)

        kmeans = KMeans(
            n_clusters=self.config.n_clusters,
            random_state=self.config.random_state,
            n_init=10,
        )

        X_train["cluster"] = kmeans.fit_predict(scaled_train)
        X_test["cluster"] = kmeans.predict(scaled_test)

        centers = kmeans.cluster_centers_

        X_train["dist_to_center"] = np.linalg.norm(
            scaled_train - centers[X_train["cluster"]],
            axis=1,
        )

        X_test["dist_to_center"] = np.linalg.norm(
            scaled_test - centers[X_test["cluster"]],
            axis=1,
        )

        X_train = pd.get_dummies(
            X_train,
            columns=["cluster"],
            prefix="c",
            drop_first=False,
        )

        X_test = pd.get_dummies(
            X_test,
            columns=["cluster"],
            prefix="c",
            drop_first=False,
        )

        X_train, X_test = X_train.align(
            X_test,
            join="left",
            axis=1,
            fill_value=0,
        )

        return X_train, X_test, kmeans, scaler

    def plot_performance_dashboard(self, y_test, preds, X_test):
        if not HAS_PLOTLY:
            self._emit_warning("plotly not installed. Skipping dashboard generation.")
            return

        actual_total = self.get_total_price(y_test, X_test["log_plot_area"])
        pred_total = self.get_total_price(preds, X_test["log_plot_area"])

        results = pd.DataFrame(
            {
                "Actual (Cr)": actual_total / 1e7,
                "Predicted (Cr)": pred_total / 1e7,
                "Error %": (np.abs(actual_total - pred_total) / actual_total) * 100,
            }
        )

        fig = px.scatter(
            results,
            x="Actual (Cr)",
            y="Predicted (Cr)",
            color="Error %",
            title="Model Accuracy: Actual vs Predicted Total Price",
            color_continuous_scale="Portland",
            hover_data=results.columns,
        )

        max_value = max(results["Actual (Cr)"].max(), results["Predicted (Cr)"].max())

        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=max_value,
            y1=max_value,
            line={"color": "Black", "dash": "dash"},
        )

        fig.write_html(self.config.actual_vs_predicted_file_path)

        results["Residuals"] = results["Actual (Cr)"] - results["Predicted (Cr)"]

        fig_res = px.scatter(
            results,
            x="Predicted (Cr)",
            y="Residuals",
            color="Error %",
            title="Residual Analysis",
            color_continuous_scale="RdBu_r",
            hover_data=results.columns,
        )

        fig_res.add_hline(y=0, line_dash="dash")
        fig_res.write_html(self.config.residual_analysis_file_path)
        self._emit_info(
            f"Saved dashboards -> {self.config.actual_vs_predicted_file_path}, {self.config.residual_analysis_file_path}"
        )

    def get_models(self):
        models = {
            "Random Forest": RandomForestRegressor(
                n_estimators=100,
                n_jobs=-1,
                random_state=self.config.random_state,
            )
        }

        if HAS_XGB:
            models["XGBoost"] = XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=7,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                random_state=self.config.random_state,
                objective="reg:squarederror",
            )
        else:
            self._emit_warning("xgboost not found. Skipping XGBoost.")

        if HAS_LGBM:
            models["LightGBM"] = LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                random_state=self.config.random_state,
                verbose=-1,
            )
        else:
            self._emit_warning("lightgbm not found. Skipping LightGBM.")

        return models

    @staticmethod
    def get_param_grid(model_name):
        if model_name == "Random Forest":
            return {
                "n_estimators": [200, 400, 600],
                "max_depth": [10, 20, 30, None],
                "max_features": [0.6, 0.8, "sqrt"],
                "min_samples_leaf": [1, 2, 4],
            }

        if model_name == "XGBoost":
            return {
                "n_estimators": [300, 600, 900],
                "learning_rate": [0.01, 0.03, 0.05],
                "max_depth": [5, 7, 9],
                "subsample": [0.7, 0.8, 0.9],
                "colsample_bytree": [0.7, 0.8, 1.0],
            }

        if model_name == "LightGBM":
            return {
                "n_estimators": [300, 600, 900],
                "learning_rate": [0.01, 0.03, 0.05],
                "num_leaves": [31, 63, 127],
                "subsample": [0.7, 0.8, 0.9],
                "colsample_bytree": [0.7, 0.8, 1.0],
            }

        raise ValueError(f"No parameter grid defined for model: {model_name}")

    def tune_model(self, model_name, model, X_train, y_train):
        param_grid = self.get_param_grid(model_name)
        total_combinations = len(list(ParameterGrid(param_grid)))
        n_iter = min(100, total_combinations)

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=5,
            scoring=self.price_num_mae_scorer,
            random_state=self.config.random_state,
            n_jobs=-1,
            verbose=1,
        )

        search.fit(X_train, y_train)

        self._emit_info(
            f"Best CV MAE on total price: {-search.best_score_ / 1e7:.4f} Cr, params: {search.best_params_}"
        )
        return search.best_estimator_

    def initiate_model_training(self) -> PlotModelTrainerArtifact:
        try:
            self._emit_info("============ Plot Model Training Started ============")

            os.makedirs(self.config.model_dir, exist_ok=True)

            data_path = self.data_transformation_artifact.transformed_file_path
            df_raw = pd.read_csv(data_path, low_memory=False)
            df = self.clean_and_validate(df_raw)

            self._emit_info(f"Rows after basic cleaning: {len(df)}")

            y = np.log1p(df["price_per_sqft"])

            leakage_columns = [
                "price_per_sqft",
                "total_price",
                "price_num",
                "price_by_circle_ratio",
                "locality",
            ]

            X = df.drop(columns=leakage_columns, errors="ignore")

            if "plot_area" not in X.columns:
                raise ValueError("Missing plot_area after transformation output load")

            X["log_plot_area"] = np.log1p(X["plot_area"])
            X = X.drop(columns=["plot_area"], errors="ignore")

            categorical_columns = [
                col for col in ["usage_type", "facing_direction"] if col in X.columns
            ]
            if categorical_columns:
                X = pd.get_dummies(X, columns=categorical_columns, drop_first=False)

            X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
            )

            X_train, X_test, kmeans_model, coord_scaler = self.add_spatial_features(
                X_train, X_test
            )

            before_rows = len(X_train)
            X_train, y_train, iso_forest = self.remove_outliers_split(X_train, y_train)
            after_rows = len(X_train)

            self._emit_info(
                f"Training rows before outlier removal={before_rows}, after={after_rows}, removed={before_rows - after_rows}"
            )

            baseline_model = RandomForestRegressor(
                n_estimators=100,
                n_jobs=-1,
                random_state=self.config.random_state,
            )
            baseline_model.fit(X_train, y_train)

            baseline_mae, baseline_mape, baseline_r2, _, _ = self.evaluate_model_on_price_num(
                baseline_model,
                X_test,
                y_test,
            )
            self._emit_info(
                f"Baseline RF MAE={baseline_mae / 1e7:.4f} Cr, MAPE={baseline_mape * 100:.2f}%, R2={baseline_r2:.4f}"
            )

            self._emit_info("--- Model comparison started ---")

            models = self.get_models()
            best_name = None
            best_score = -np.inf
            cv_results: dict = {}

            for name, model in models.items():
                scores = cross_val_score(
                    model,
                    X_train,
                    y_train,
                    cv=5,
                    scoring=self.price_num_r2_scorer,
                    n_jobs=-1,
                )

                cv_results[name] = scores
                mean_score = scores.mean()
                std_score = scores.std()

                self._emit_info(f"{name} CV R2={mean_score:.4f} +/- {std_score:.4f}")

                if mean_score > best_score:
                    best_score = mean_score
                    best_name = name

            if best_name is None:
                raise ValueError("No model available for tuning")

            self._emit_info(f"Winner for tuning: {best_name}")

            final_model = self.tune_model(best_name, models[best_name], X_train, y_train)

            final_mae, final_mape, final_r2, _, _ = self.evaluate_model_on_price_num(
                final_model,
                X_test,
                y_test,
            )

            # Safeguard: if the tuned model underperforms the vanilla baseline RF,
            # fall back to the baseline to avoid deploying a regressed model.
            if final_r2 < baseline_r2:
                self._emit_warning(
                    f"Tuned {best_name} (R2={final_r2:.4f}) underperforms "
                    f"baseline RF (R2={baseline_r2:.4f}). "
                    f"Falling back to baseline RandomForest model."
                )
                final_model = baseline_model
                best_name = "Random Forest"
                final_mae, final_mape, final_r2 = baseline_mae, baseline_mape, baseline_r2

            self._emit_info("=" * 50)
            self._emit_info("FINAL TEST PERFORMANCE")
            self._emit_info(f"Model: {best_name}")
            self._emit_info(f"MAE  : ₹{final_mae / 1e7:.4f} Cr")
            self._emit_info(f"MAPE : {final_mape * 100:.2f}%")
            self._emit_info(f"R2   : {final_r2:.4f}")
            self._emit_info("=" * 50)

            test_preds_log = final_model.predict(X_test)
            self.plot_performance_dashboard(y_test, test_preds_log, X_test)

            artifacts = {
                "model": final_model,
                "kmeans": kmeans_model,
                "coord_scaler": coord_scaler,
                "iso_forest": iso_forest,
                "features": X_train.columns.tolist(),
                "target": "log_price_per_sqft",
                "evaluation_metric": "price_num_total_price",
                "best_model_name": best_name,
            }

            joblib.dump(artifacts, self.config.model_file_path)
            joblib.dump(X_train.columns.tolist(), self.config.feature_columns_file_path)

            if HAS_MLFLOW:
                try:
                    run_name = f"plot_v{self.config.version}" if self.config.version > 0 else "plot"
                    with mlflow.start_run(run_name=run_name):
                        mlflow.log_params({
                            "model_type": best_name,
                            "version": self.config.version,
                            "n_clusters": self.config.n_clusters,
                            "contamination": self.config.contamination,
                            "test_size": self.config.test_size,
                            "n_training_rows": after_rows,
                            "n_features": X_train.shape[1],
                        })
                        mlflow.log_metrics({
                            "mae": float(final_mae),
                            "mape": float(final_mape),
                            "r2": float(final_r2),
                            "baseline_mae": float(baseline_mae),
                            "baseline_r2": float(baseline_r2),
                        })
                        for fold_idx, fold_r2 in enumerate(cv_results[best_name]):
                            mlflow.log_metric("cv_r2", float(fold_r2), step=fold_idx)
                        for cand_name, cand_scores in cv_results.items():
                            mlflow.log_metric(
                                f"cv_r2_{cand_name.lower()}", float(cand_scores.mean())
                            )
                        mlflow.log_artifact(self.config.model_file_path, artifact_path="model")
                        mlflow.log_artifact(self.config.feature_columns_file_path, artifact_path="model")
                        self._emit_info(f"MLflow run logged: {mlflow.active_run().info.run_id}")
                except Exception as _mlflow_err:
                    self._emit_warning(f"MLflow logging skipped (non-fatal): {_mlflow_err}")

            self._emit_info(f"Saved plot model artifact -> {self.config.model_file_path}")
            self._emit_info(
                f"Saved plot feature columns artifact -> {self.config.feature_columns_file_path}"
            )

            # ── TensorBoard deep-dive analysis ─────────────────
            if HAS_TB_ANALYSIS:
                try:
                    import os as _os
                    tb_dir = _os.path.join(
                        "artifact", "tensorboard", "plot",
                        f"v{self.config.version}" if self.config.version > 0 else "v0",
                    )
                    # For plot: target is log_price_per_sqft; back-calc is total price
                    # We pass log-scale arrays and ppsf-equivalent arrays (actual ppsf)
                    plot_test_preds_log = final_model.predict(X_test)
                    actual_ppsf_plot = np.expm1(y_test.values)
                    pred_ppsf_plot   = np.expm1(plot_test_preds_log)
                    actual_total_plot, pred_total_plot = (
                        self.get_total_price(y_test, X_test["log_plot_area"]),
                        self.get_total_price(plot_test_preds_log, X_test["log_plot_area"]),
                    )
                    from sklearn.metrics import (
                        mean_absolute_error as _mae_fn,
                        mean_absolute_percentage_error as _mape_fn,
                        r2_score as _r2_fn,
                    )
                    log_tb_analysis(
                        log_dir       = tb_dir,
                        property_type = "plot",
                        version       = self.config.version,
                        final_model   = final_model,
                        X_train       = X_train,
                        X_test        = X_test,
                        y_train       = y_train,
                        y_test        = y_test.values,
                        y_pred_log    = plot_test_preds_log,
                        actual_ppsf   = actual_ppsf_plot,
                        pred_ppsf     = pred_ppsf_plot,
                        cv_results    = cv_results,
                        best_model_name = best_name,
                        mae     = float(_mae_fn(actual_ppsf_plot, pred_ppsf_plot)),
                        mape    = float(_mape_fn(actual_ppsf_plot, pred_ppsf_plot)),
                        r2      = float(_r2_fn(actual_ppsf_plot, pred_ppsf_plot)),
                        r2_target = float(_r2_fn(y_test.values, plot_test_preds_log)),
                    )
                except Exception as _tb_e:
                    self._emit_warning(f"TensorBoard analysis skipped (non-fatal): {_tb_e}")

            self._emit_info("============ Plot Model Training Finished ============")

            return PlotModelTrainerArtifact(
                model_file_path=self.config.model_file_path,
                feature_columns_file_path=self.config.feature_columns_file_path,
                best_model_name=best_name,
                mae=float(final_mae),
                mape=float(final_mape),
                r2=float(final_r2),
            )

        except Exception as e:
            raise RealEstateException(e, sys)
