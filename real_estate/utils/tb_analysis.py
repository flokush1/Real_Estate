"""
Comprehensive TensorBoard Analysis for ML Training Runs
========================================================
Covers every analysis panel available in TensorBoard for tree-based / ensemble
models (RandomForest, XGBoost, LightGBM).

TensorBoard tabs generated
--------------------------
SCALARS
  cv/fold_r2               — CV R² per fold (best model)   → shows fold stability
  cv/{model}_mean_r2       — Mean CV R² for every candidate model
  metrics/mae              — Final test MAE (price_per_sqft domain)
  metrics/mape_pct         — Final test MAPE %
  metrics/r2               — Final test R²
  metrics/r2_target        — R² on log-scale target (log1p_price_by_circle)

HISTOGRAMS
  distributions/actual_ppsf        — True price per sqft
  distributions/pred_ppsf          — Predicted price per sqft
  distributions/abs_error          — |pred − actual| in ₹/sqft
  distributions/pct_error          — 100×|pred−actual|/actual (per sample)
  distributions/target_log         — log-scale target (y_test)
  distributions/pred_log           — log-scale predictions
  feature_importance/top20         — Top-20 feature importance values

IMAGES (matplotlib figures embedded as images)
  plots/actual_vs_predicted        — Scatter: true vs predicted (₹/sqft)
  plots/residuals_vs_predicted     — Residuals scatter (heteroscedasticity)
  plots/residuals_distribution     — Histogram + KDE + Normal overlay
  plots/feature_importance         — Horizontal bar: top-20 features
  plots/cv_model_comparison        — Bar: mean ± std CV R² per candidate model
  plots/error_percentiles          — MAPE across error-bucket ranges (p10…p95)
  plots/price_distribution         — KDE: actual vs predicted ₹/sqft overlay
  plots/learning_curve             — Train/CV score vs training-set size
  plots/error_heatmap              — BHK × floor_level mean absolute error grid
  plots/correlation_top_features   — Correlation heatmap of top-15 features

TEXT
  model/summary            — Model name + hyperparameter dump
  model/features           — Full feature list
  training/summary         — Dataset stats, version, metric table

Usage
-----
from real_estate.utils.tb_analysis import log_tb_analysis

log_tb_analysis(
    log_dir       = "artifact/tensorboard/apt/v1",
    property_type = "apt",
    version       = 1,
    final_model   = model,
    X_train       = X_train,
    X_test        = X_test,
    y_train       = y_train,           # log-scale np.ndarray
    y_test        = y_test,            # log-scale np.ndarray
    y_pred_log    = y_pred_log,        # log-scale predictions
    actual_ppsf   = actual_ppsf,       # ₹/sqft np.ndarray
    pred_ppsf     = pred_ppsf,         # ₹/sqft np.ndarray
    cv_results    = cv_results,        # {model_name: np.ndarray of fold R²}
    best_model_name = best_name,
    mae    = mae,
    mape   = mape,
    r2     = r2,
    r2_target = r2_pbc,
)
"""

import io
import traceback

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be before pyplot import
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.model_selection import learning_curve

HAS_TB = False
SummaryWriter = None

try:
    from tensorboardX import SummaryWriter  # type: ignore[import-not-found]
    HAS_TB = True
except ImportError:
    try:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore[import-not-found]
        HAS_TB = True
    except ImportError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fig_to_arr(fig: plt.Figure) -> np.ndarray:
    """Convert a matplotlib Figure to a (H, W, 4) uint8 numpy array (RGBA)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    import PIL.Image  # type: ignore[import-not-found]
    img = PIL.Image.open(buf).convert("RGBA")
    arr = np.asarray(img)
    plt.close(fig)
    return arr  # (H, W, 4)


def _has_feature_importance(model) -> bool:
    return hasattr(model, "feature_importances_")


def _get_importances(model, feature_names: list) -> pd.Series:
    imp = pd.Series(model.feature_importances_, index=feature_names)
    return imp.sort_values(ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# Individual chart builders
# ─────────────────────────────────────────────────────────────────────────────

def _fig_actual_vs_predicted(
    actual: np.ndarray,
    predicted: np.ndarray,
    property_type: str,
) -> plt.Figure:
    """Scatter: actual vs predicted ₹/sqft with perfect-fit diagonal."""
    fig, ax = plt.subplots(figsize=(7, 6))
    lo = min(actual.min(), predicted.min()) * 0.95
    hi = max(actual.max(), predicted.max()) * 1.05

    # colour by absolute % error
    pct_err = np.abs(actual - predicted) / np.clip(actual, 1, None) * 100
    sc = ax.scatter(actual, predicted, c=pct_err, cmap="RdYlGn_r",
                    alpha=0.45, s=12, vmin=0, vmax=40)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="Perfect fit")
    plt.colorbar(sc, ax=ax, label="Abs % Error")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual ₹/sqft"); ax.set_ylabel("Predicted ₹/sqft")
    ax.set_title(f"[{property_type.upper()}] Actual vs Predicted")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _fig_residuals_vs_predicted(
    actual: np.ndarray,
    predicted: np.ndarray,
    property_type: str,
) -> plt.Figure:
    """Residuals vs predicted — reveals heteroscedasticity / bias."""
    residuals = predicted - actual
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(predicted, residuals, alpha=0.35, s=10, c="steelblue")
    ax.axhline(0, color="red", linestyle="--", lw=1.5)
    # Lowess smoother trend
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess  # type: ignore
        sorted_idx = np.argsort(predicted)
        sm = lowess(residuals[sorted_idx], predicted[sorted_idx], frac=0.15)
        ax.plot(sm[:, 0], sm[:, 1], color="orange", lw=2, label="Trend (LOWESS)")
        ax.legend(fontsize=8)
    except Exception:
        pass
    ax.set_xlabel("Predicted ₹/sqft"); ax.set_ylabel("Residual (pred − actual)")
    ax.set_title(f"[{property_type.upper()}] Residuals vs Predicted")
    fig.tight_layout()
    return fig


def _fig_residuals_distribution(
    actual: np.ndarray,
    predicted: np.ndarray,
    property_type: str,
) -> plt.Figure:
    """Histogram + KDE of residuals with fitted Normal overlay."""
    residuals = predicted - actual
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Absolute residuals
    ax = axes[0]
    ax.hist(residuals, bins=60, density=True, alpha=0.6, color="steelblue", label="Residuals")
    xr = np.linspace(residuals.min(), residuals.max(), 300)
    kde = scipy_stats.gaussian_kde(residuals)
    ax.plot(xr, kde(xr), "b-", lw=2, label="KDE")
    mu, sigma = residuals.mean(), residuals.std()
    ax.plot(xr, scipy_stats.norm.pdf(xr, mu, sigma), "r--", lw=1.5, label="Normal fit")
    ax.axvline(0, color="black", lw=1)
    ax.set_title("Residuals Distribution"); ax.set_xlabel("₹/sqft"); ax.legend(fontsize=8)

    # Percentage error distribution
    ax2 = axes[1]
    pct = (predicted - actual) / np.clip(actual, 1, None) * 100
    ax2.hist(pct, bins=60, density=True, alpha=0.6, color="orange", label="% Error")
    xp = np.linspace(pct.min(), pct.max(), 300)
    kde2 = scipy_stats.gaussian_kde(pct)
    ax2.plot(xp, kde2(xp), "-", color="darkorange", lw=2, label="KDE")
    ax2.axvline(0, color="black", lw=1)
    ax2.set_title("Percentage Error Distribution"); ax2.set_xlabel("%"); ax2.legend(fontsize=8)

    fig.suptitle(f"[{property_type.upper()}] Error Distributions", fontsize=13)
    fig.tight_layout()
    return fig


def _fig_feature_importance(
    model,
    feature_names: list,
    property_type: str,
    top_n: int = 25,
) -> plt.Figure:
    """Horizontal bar chart — top-N feature importances."""
    imp = _get_importances(model, feature_names).head(top_n)
    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.32)))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.85, len(imp)))[::-1]
    ax.barh(imp.index[::-1], imp.values[::-1], color=colors[::-1])
    ax.set_xlabel("Importance Score")
    ax.set_title(f"[{property_type.upper()}] Feature Importance — Top {top_n}")
    fig.tight_layout()
    return fig


def _fig_cv_model_comparison(
    cv_results: dict,
    property_type: str,
) -> plt.Figure:
    """Bar chart — mean ± std CV R² for each candidate model."""
    names  = list(cv_results.keys())
    means  = [cv_results[n].mean() for n in names]
    stds   = [cv_results[n].std()  for n in names]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"][:len(names)]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, means, yerr=stds, capsize=6,
                  color=colors, alpha=0.85, error_kw={"elinewidth": 2})
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.005,
                f"{m:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("CV R²"); ax.set_ylim(max(0, min(means) - 0.05), min(1, max(means) + 0.05))
    ax.set_title(f"[{property_type.upper()}] CV Model Comparison")
    fig.tight_layout()
    return fig


def _fig_error_percentiles(
    actual: np.ndarray,
    predicted: np.ndarray,
    property_type: str,
) -> plt.Figure:
    """Bar chart of MAPE at different abs-error percentile ranges."""
    abs_pct = np.abs(predicted - actual) / np.clip(actual, 1, None) * 100
    percentile_labels = ["0-10%", "10-25%", "25-50%", "50-75%", "75-90%", "90-95%", "95-100%"]
    boundaries = [0, 10, 25, 50, 75, 90, 95, 100]
    counts = []
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        p_lo = np.percentile(abs_pct, lo)
        p_hi = np.percentile(abs_pct, hi)
        mask = (abs_pct >= p_lo) & (abs_pct <= p_hi)
        counts.append(int(mask.sum()))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # ── Left: count per bucket ──────────────────────────────
    ax = axes[0]
    cmap = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(counts)))
    ax.bar(percentile_labels, counts, color=cmap)
    ax.set_title("Sample Count per Error Bucket")
    ax.set_xlabel("Error Percentile Range")
    ax.set_ylabel("# Samples")
    ax.tick_params(axis="x", rotation=25)

    # ── Right: cumulative coverage ──────────────────────────
    ax2 = axes[1]
    thresholds = [5, 10, 15, 20, 25, 30, 40, 50]
    coverages  = [float((abs_pct <= t).mean()) * 100 for t in thresholds]
    ax2.plot(thresholds, coverages, "o-", color="steelblue", lw=2)
    ax2.fill_between(thresholds, coverages, alpha=0.2, color="steelblue")
    ax2.set_title("Cumulative % Samples within Error Threshold")
    ax2.set_xlabel("Max Allowed Error %"); ax2.set_ylabel("% Samples")
    ax2.set_ylim(0, 105); ax2.grid(True, alpha=0.4)
    for t, c in zip(thresholds, coverages):
        ax2.text(t, c + 1.5, f"{c:.1f}%", ha="center", fontsize=7)

    fig.suptitle(f"[{property_type.upper()}] Error Percentile Analysis", fontsize=13)
    fig.tight_layout()
    return fig


def _fig_price_distribution(
    actual: np.ndarray,
    predicted: np.ndarray,
    property_type: str,
) -> plt.Figure:
    """Overlaid KDE: actual vs predicted ₹/sqft density."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # KDE overlay
    ax = axes[0]
    clip_lo = np.percentile(actual, 1)
    clip_hi = np.percentile(actual, 99)
    act_clip = np.clip(actual,    clip_lo, clip_hi)
    pre_clip = np.clip(predicted, clip_lo, clip_hi)
    xr = np.linspace(clip_lo, clip_hi, 400)
    ax.plot(xr, scipy_stats.gaussian_kde(act_clip)(xr), lw=2, label="Actual", color="steelblue")
    ax.plot(xr, scipy_stats.gaussian_kde(pre_clip)(xr), lw=2, label="Predicted", color="tomato",
            linestyle="--")
    ax.set_title("Price Density (₹/sqft)")
    ax.set_xlabel("₹/sqft"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Q-Q plot actual vs predicted
    ax2 = axes[1]
    q_act = np.percentile(actual,    np.linspace(5, 95, 50))
    q_pre = np.percentile(predicted, np.linspace(5, 95, 50))
    ax2.scatter(q_act, q_pre, s=20, alpha=0.7, color="purple")
    diag_lo = min(q_act.min(), q_pre.min())
    diag_hi = max(q_act.max(), q_pre.max())
    ax2.plot([diag_lo, diag_hi], [diag_lo, diag_hi], "k--", lw=1.5)
    ax2.set_title("Q-Q Plot: Actual vs Predicted ₹/sqft")
    ax2.set_xlabel("Actual Quantile"); ax2.set_ylabel("Predicted Quantile")
    ax2.grid(alpha=0.3)

    fig.suptitle(f"[{property_type.upper()}] Price Distribution Comparison", fontsize=13)
    fig.tight_layout()
    return fig


def _fig_learning_curve(
    model,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    property_type: str,
    cv: int = 5,
    n_points: int = 6,
) -> plt.Figure:
    """Train/CV score vs training-set size (learning curve)."""
    n_samples = len(X_train)
    train_sizes_rel = np.linspace(0.1, 1.0, n_points)
    train_sizes_abs = np.maximum(50, (train_sizes_rel * n_samples).astype(int))
    train_sizes_abs = np.unique(train_sizes_abs)

    try:
        ts, tr_sc, val_sc = learning_curve(
            model, X_train, y_train,
            train_sizes=train_sizes_abs,
            cv=cv, scoring="r2", n_jobs=-1, shuffle=False,
        )
    except Exception:
        # Fallback: return a placeholder
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "Learning curve unavailable", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="grey")
        ax.set_title(f"[{property_type.upper()}] Learning Curve")
        return fig

    tr_mean, tr_std   = tr_sc.mean(axis=1),  tr_sc.std(axis=1)
    val_mean, val_std = val_sc.mean(axis=1), val_sc.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ts, tr_mean,  "o-", color="steelblue",  label="Train R²")
    ax.fill_between(ts, tr_mean - tr_std, tr_mean + tr_std, alpha=0.15, color="steelblue")
    ax.plot(ts, val_mean, "s-", color="tomato",     label="CV R²")
    ax.fill_between(ts, val_mean - val_std, val_mean + val_std, alpha=0.15, color="tomato")
    ax.axhline(val_mean[-1], color="grey", linestyle=":", lw=1)
    ax.set_xlabel("Training Samples"); ax.set_ylabel("R²")
    ax.set_title(f"[{property_type.upper()}] Learning Curve")
    ax.legend(fontsize=9); ax.grid(alpha=0.35)
    fig.tight_layout()
    return fig


def _fig_error_heatmap(
    X_test: pd.DataFrame,
    actual: np.ndarray,
    predicted: np.ndarray,
    property_type: str,
) -> plt.Figure:
    """Mean absolute error heatmap: BHK × floor_level (if columns exist)."""
    cols_needed = ["bhk"]
    floor_col   = next(
        (c for c in ["floor_medium", "floor_high", "floor_low"] if c in X_test.columns),
        None,
    )

    if "bhk" not in X_test.columns or floor_col is None:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(0.5, 0.5, "bhk / floor_level columns not found",
                ha="center", va="center", transform=ax.transAxes, color="grey")
        ax.set_title(f"[{property_type.upper()}] Error Heatmap (unavailable)")
        return fig

    tmp = X_test[["bhk"]].copy()
    tmp["floor_level"] = (
        X_test.get("floor_high", 0).astype(int) * 2 +
        X_test.get("floor_medium", 0).astype(int) * 1
    )
    tmp["floor_label"] = tmp["floor_level"].map({0: "Low", 1: "Medium", 2: "High"})
    tmp["abs_err"] = np.abs(actual - predicted)
    tmp["bhk_int"] = pd.to_numeric(tmp["bhk"], errors="coerce").round(0).astype("Int64")

    pivot = tmp.groupby(["bhk_int", "floor_label"])["abs_err"].mean().unstack(fill_value=0)
    pivot = pivot.reindex(columns=["Low", "Medium", "High"], fill_value=0)

    fig, ax = plt.subplots(figsize=(7, max(3, len(pivot) * 0.5 + 1)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="Mean |Error| ₹/sqft")
    ax.set_xticks(range(len(pivot.columns)));  ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)));    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Floor Level"); ax.set_ylabel("BHK")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i, j]:.0f}", ha="center", va="center",
                    fontsize=8, color="black")

    ax.set_title(f"[{property_type.upper()}] Mean Abs Error by BHK × Floor Level")
    fig.tight_layout()
    return fig


def _fig_correlation_heatmap(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    property_type: str,
    top_n: int = 15,
) -> plt.Figure:
    """Pearson correlation heatmap among top-N features + target."""
    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(0.5, 0.5, "No numeric features", ha="center", va="center",
                transform=ax.transAxes, color="grey")
        ax.set_title(f"[{property_type.upper()}] Correlation Heatmap (unavailable)")
        return fig

    tmp = X_train[num_cols].copy()
    tmp["_target_"] = np.asarray(y_train)

    # Keep top-N columns most correlated with target
    corr_with_target = tmp.corr()["_target_"].drop("_target_").abs().sort_values(ascending=False)
    selected = corr_with_target.head(top_n).index.tolist() + ["_target_"]
    corr_mat = tmp[selected].corr()

    fig, ax = plt.subplots(figsize=(min(14, len(selected)), min(12, len(selected))))
    im = ax.imshow(corr_mat.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_xticks(range(len(selected))); ax.set_xticklabels(selected, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(selected))); ax.set_yticklabels(selected, fontsize=7)

    for i in range(len(selected)):
        for j in range(len(selected)):
            ax.text(j, i, f"{corr_mat.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="black" if abs(corr_mat.values[i, j]) < 0.7 else "white")

    ax.set_title(f"[{property_type.upper()}] Feature Correlation (top-{top_n})")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main public entry point
# ─────────────────────────────────────────────────────────────────────────────

def log_tb_analysis(
    log_dir: str,
    property_type: str,
    version: int,
    final_model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_pred_log: np.ndarray,
    actual_ppsf: np.ndarray,
    pred_ppsf: np.ndarray,
    cv_results: dict,
    best_model_name: str,
    mae: float,
    mape: float,
    r2: float,
    r2_target: float,
) -> None:
    """
    Write a full TensorBoard event file to `log_dir`.
    All failures are non-fatal — a warning is printed and training continues.

    View results:
        tensorboard --logdir artifact/tensorboard
    """
    if not HAS_TB:
        print(
            "[TensorBoard] tensorboardX / torch not installed — skipping TB analysis. "
            "Install with:  pip install tensorboardX"
        )
        return

    try:
        import os
        os.makedirs(log_dir, exist_ok=True)

        y_test_arr   = np.asarray(y_test,    dtype=float)
        y_pred_arr   = np.asarray(y_pred_log, dtype=float)
        actual_arr   = np.asarray(actual_ppsf, dtype=float)
        pred_arr     = np.asarray(pred_ppsf,   dtype=float)
        y_train_arr  = np.asarray(y_train,    dtype=float)
        feature_names = X_train.columns.tolist()

        writer = SummaryWriter(log_dir=log_dir)

        # ── SCALARS ──────────────────────────────────────────
        # CV fold scores for best model (step = fold index)
        for fold_idx, fold_r2 in enumerate(cv_results[best_model_name]):
            writer.add_scalar("cv/fold_r2", float(fold_r2), global_step=fold_idx)

        # Mean CV R² per candidate model (step = model index)
        for model_idx, (name, scores) in enumerate(cv_results.items()):
            writer.add_scalar(
                f"cv/{name.lower().replace(' ', '_')}_mean_r2",
                float(scores.mean()),
                global_step=model_idx,
            )

        # Final evaluation metrics
        writer.add_scalar("metrics/mae",       float(mae),       global_step=version)
        writer.add_scalar("metrics/mape_pct",  float(mape * 100), global_step=version)
        writer.add_scalar("metrics/r2",        float(r2),        global_step=version)
        writer.add_scalar("metrics/r2_target", float(r2_target), global_step=version)

        # ── HISTOGRAMS ───────────────────────────────────────
        writer.add_histogram("distributions/actual_ppsf", actual_arr)
        writer.add_histogram("distributions/pred_ppsf",   pred_arr)
        writer.add_histogram("distributions/abs_error",   np.abs(actual_arr - pred_arr))
        pct_err = np.abs(actual_arr - pred_arr) / np.clip(actual_arr, 1, None) * 100
        writer.add_histogram("distributions/pct_error",   pct_err)
        writer.add_histogram("distributions/target_log",  y_test_arr)
        writer.add_histogram("distributions/pred_log",    y_pred_arr)

        if _has_feature_importance(final_model):
            imp = _get_importances(final_model, feature_names)
            top20 = imp.head(20).values.astype(float)
            writer.add_histogram("feature_importance/top20", top20)

        # ── IMAGES ───────────────────────────────────────────
        def _safe_add_image(tag: str, fig_fn, *args, **kwargs):
            try:
                fig = fig_fn(*args, **kwargs)
                arr = _fig_to_arr(fig)          # (H, W, 4) uint8
                # TB expects (C, H, W) float [0,1]
                tb_arr = (arr[:, :, :3].transpose(2, 0, 1).astype(np.float32) / 255.0)
                writer.add_image(tag, tb_arr, global_step=version)
            except Exception as _e:
                print(f"[TensorBoard] Skipping '{tag}': {_e}")

        _safe_add_image(
            "plots/actual_vs_predicted", _fig_actual_vs_predicted,
            actual_arr, pred_arr, property_type,
        )
        _safe_add_image(
            "plots/residuals_vs_predicted", _fig_residuals_vs_predicted,
            actual_arr, pred_arr, property_type,
        )
        _safe_add_image(
            "plots/residuals_distribution", _fig_residuals_distribution,
            actual_arr, pred_arr, property_type,
        )
        if _has_feature_importance(final_model):
            _safe_add_image(
                "plots/feature_importance", _fig_feature_importance,
                final_model, feature_names, property_type,
            )
        _safe_add_image(
            "plots/cv_model_comparison", _fig_cv_model_comparison,
            cv_results, property_type,
        )
        _safe_add_image(
            "plots/error_percentiles", _fig_error_percentiles,
            actual_arr, pred_arr, property_type,
        )
        _safe_add_image(
            "plots/price_distribution", _fig_price_distribution,
            actual_arr, pred_arr, property_type,
        )
        _safe_add_image(
            "plots/learning_curve", _fig_learning_curve,
            final_model, X_train, y_train_arr, property_type,
        )
        _safe_add_image(
            "plots/error_heatmap", _fig_error_heatmap,
            X_test, actual_arr, pred_arr, property_type,
        )
        if _has_feature_importance(final_model):
            _safe_add_image(
                "plots/correlation_top_features", _fig_correlation_heatmap,
                X_train, y_train_arr, property_type,
            )

        # ── TEXT ─────────────────────────────────────────────
        # Model hyperparameters
        try:
            params_str = str(final_model.get_params())
        except Exception:
            params_str = repr(final_model)

        writer.add_text(
            "model/summary",
            (
                f"**Property type**: {property_type}  \n"
                f"**Model**: {best_model_name}  \n"
                f"**Version**: {version}  \n\n"
                f"**Hyperparameters**:\n```\n{params_str}\n```"
            ),
            global_step=version,
        )

        writer.add_text(
            "model/features",
            "**Features used**:\n\n" + "\n".join(f"- {f}" for f in feature_names),
            global_step=version,
        )

        writer.add_text(
            "training/summary",
            (
                f"| Metric | Value |\n"
                f"|--------|-------|\n"
                f"| Property Type | {property_type} |\n"
                f"| Model | {best_model_name} |\n"
                f"| Version | {version} |\n"
                f"| Train rows | {len(X_train)} |\n"
                f"| Test rows  | {len(X_test)} |\n"
                f"| Features   | {len(feature_names)} |\n"
                f"| MAE ₹/sqft | {mae:.2f} |\n"
                f"| MAPE %     | {mape * 100:.2f} |\n"
                f"| R² (₹/sqft)| {r2:.4f} |\n"
                f"| R² (target)| {r2_target:.4f} |\n"
                + "\n\n**CV Results** (best model folds):\n"
                + "\n".join(
                    f"- Fold {i+1}: {s:.4f}"
                    for i, s in enumerate(cv_results[best_model_name])
                )
            ),
            global_step=version,
        )

        writer.close()
        print(f"[TensorBoard] Analysis written → {log_dir}")
        print(f"[TensorBoard] View with:  tensorboard --logdir artifact/tensorboard")

    except Exception as _outer:
        print(f"[TensorBoard] Non-fatal error in TensorBoard analysis: {_outer}")
        traceback.print_exc()
