"""
MarketIntelligenceService — demand-supply diagnostics and liquidity analysis
for the NCR Real Estate AI platform.

Reads raw HO/MB listing data, builds per-(segment, city, locality, scrape_date)
metric artifacts, and serves structured context to the API.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from real_estate.utils.circle_rate_matcher import CircleRateMatcher

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SQFT_FACTORS: dict[str, float] = {
    "sqft": 1.0, "sq-ft": 1.0, "sq ft": 1.0, "sft": 1.0,
    "sqyd": 9.0, "sq-yrd": 9.0, "sq-yd": 9.0, "sqyrd": 9.0, "sq yrd": 9.0,
    "sqm": 10.7639, "sq-m": 10.7639, "sq m": 10.7639,
    "acre": 43560.0,
    "bigha": 27000.0,
    "marla": 272.25,
    "hectare": 107639.0,
}

PROPERTY_TYPE_TO_SEGMENT: dict[str, str] = {
    "apartment": "apt",
    "studio apartment": "apt",
    "penthouse": "apt",
    "builder floor": "builder_floor",
    "independent floor": "builder_floor",
    "builder floor apartment": "builder_floor",
    "plot": "plot",
    "residential plot": "plot",
    "land": "plot",
    "residential land": "plot",
    "independent house": "house_villa",
    "residential house": "house_villa",
    "villa": "house_villa",
}

SUPPORTED_SEGMENTS: set[str] = {"apt", "builder_floor", "plot", "house_villa"}

ACTIVE_EVENT_TYPES: set[str] = {"new", "unchanged", "updated"}

CITY_NORMALIZE: dict[str, str] = {
    "new delhi": "Delhi",
    "delhi": "Delhi",
    "gurugram": "Gurgaon",
    "gurgaon": "Gurgaon",
    "noida": "Noida",
    "greater noida": "Greater Noida",
    "greater noida west": "Greater Noida",
    "gr noida": "Greater Noida",
    "ghaziabad": "Ghaziabad",
    "faridabad": "Faridabad",
    "jaipur": "Jaipur",
}

# Localities whose administrative city is misclassified by one or both sources.
# Key = canonical locality title-case; Value = correct normalized city.
# Applied after city normalization in _normalize_market_data.
LOCALITY_CITY_OVERRIDES: dict[str, str] = {
    "Noida Extension": "Greater Noida",
    "Greater Noida West": "Greater Noida",
}

_READY_KEYWORDS = ("ready", "immediate", "ready to move")
_UC_KEYWORDS = ("under construction", "possession by", "new launch", "upcoming", "under-construction")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_sqft(value, unit: str) -> float | None:
    try:
        v = float(value)
        if not (v > 0):
            return None
    except (TypeError, ValueError):
        return None
    unit_key = re.sub(r"\s+", " ", str(unit).strip().lower())
    factor = SQFT_FACTORS.get(unit_key, 1.0)
    result = v * factor
    return result if result > 0 else None


def _norm_city(city_raw) -> str | None:
    if not isinstance(city_raw, str):
        return None
    key = re.sub(r"\s+", " ", city_raw.strip().lower())
    return CITY_NORMALIZE.get(key, city_raw.strip().title() if city_raw.strip() else None)


def _map_segment(property_type_raw) -> str | None:
    if not isinstance(property_type_raw, str):
        return None
    key = re.sub(r"\s+", " ", property_type_raw.strip().lower())
    return PROPERTY_TYPE_TO_SEGMENT.get(key)


def _safe_div(num: float, denom: float, default: float = 0.0) -> float:
    try:
        if denom == 0:
            return default
        return num / denom
    except Exception:
        return default


def percentile_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Scale values to 0–100 by percentile rank within the series."""
    pct = series.rank(pct=True) * 100
    return pct if higher_is_better else 100.0 - pct


def market_label(score: float) -> str:
    if score >= 75:
        return "Hot demand-led market"
    if score >= 60:
        return "Positive market"
    if score >= 45:
        return "Balanced market"
    if score >= 30:
        return "Supply-heavy market"
    return "Weak / stale market"


# ─────────────────────────────────────────────────────────────────────────────
# Service
# ─────────────────────────────────────────────────────────────────────────────

class MarketIntelligenceService:
    """
    Loads raw listing CSVs, normalizes them into a market-intelligence schema,
    computes per-(segment, city, locality, scrape_date) metrics and index scores,
    writes artifacts to opt/{segment}/market_intelligence.csv, and serves
    structured context via get_market_context().
    """

    def __init__(self, project_root: str) -> None:
        self.project_root = Path(project_root)
        self._artifacts: dict[str, pd.DataFrame] = {}
        # circle_rates/ lives at the project root, not nested inside real_estate_data/
        cr_dir = str(self.project_root / "circle_rates")
        try:
            self._cr_matcher: CircleRateMatcher | None = CircleRateMatcher(cr_dir)
        except Exception as exc:
            logger.warning("MarketIntelligenceService: circle rate matcher init failed: %s", exc)
            self._cr_matcher = None

        # Prefer the already-cleaned transformation output (accurate locality names,
        # validated prices/areas, correct property_type_grouped segments) over raw
        # HO/MB files.  Fall back to raw files if no cleaned artifact exists yet.
        cleaned = self._find_latest_cleaned_csv()
        if cleaned:
            logger.info("MarketIntelligenceService: using cleaned CSV %s", cleaned)
            self.raw_paths = [cleaned]
        else:
            logger.warning(
                "MarketIntelligenceService: no cleaned.csv found; falling back to raw HO/MB files"
            )
            self.raw_paths = [
                self.project_root / "real_estate_data" / "ho_raw_data.csv",
                self.project_root / "real_estate_data" / "mb_raw_data.csv",
            ]

    def _find_latest_cleaned_csv(self) -> Path | None:
        """Return the cleaned.csv from the highest-versioned data_transformation artifact."""
        import glob as _glob
        pattern = str(self.project_root / "artifact" / "data_transformation" / "*" / "cleaned.csv")
        candidates = sorted(_glob.glob(pattern))  # v4 < v5 < v6 … sorts correctly
        return Path(candidates[-1]) if candidates else None

    # ─────────────────────────────────────────────────────────────
    # 1. Load + normalize raw data
    # ─────────────────────────────────────────────────────────────

    # Only the columns actually consumed by _normalize_market_data
    _NEEDED_COLS = {
        "property_id", "property_type", "property_type_grouped", "city", "locality",
        "event_type", "price_numeric", "covered_area_value", "covered_area_unit",
        "sqft_price", "possession_status", "posting_date", "scrape_date",
        "agent_type", "user_type", "developer_id", "developer_uuid",
    }

    def load_market_data(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for path in self.raw_paths:
            if not path.exists():
                logger.warning("MarketIntelligenceService: raw file not found: %s", path)
                continue
            try:
                # Peek at available columns to build a tight usecols list,
                # then re-read only those columns — cuts memory & parse time.
                available = pd.read_csv(path, nrows=0).columns.tolist()
                use = [c for c in available if c in self._NEEDED_COLS]
                df = pd.read_csv(
                    path,
                    usecols=use,
                    low_memory=False,
                    dtype={
                        "property_id": str,
                        "property_type": str,
                        "city": str,
                        "locality": str,
                        "event_type": str,
                        "possession_status": str,
                        "agent_type": str,
                        "user_type": str,
                        "developer_id": str,
                        "developer_uuid": str,
                    },
                )
            except Exception as exc:
                logger.warning("MarketIntelligenceService: failed to read %s: %s", path, exc)
                continue
            df["_source_file"] = (
                "cleaned" if "cleaned" in path.name
                else "housing" if "ho_" in path.name
                else "magicbricks"
            )
            frames.append(df)

        if not frames:
            return pd.DataFrame()

        raw = pd.concat(frames, ignore_index=True)
        return self._normalize_market_data(raw)

    def _normalize_market_data(self, df: pd.DataFrame) -> pd.DataFrame:  # noqa: PLR0912
        n = len(df)

        def col(name: str, dtype=None) -> pd.Series:
            s = df.get(name, pd.Series([None] * n))
            if dtype:
                s = pd.to_numeric(s, errors="coerce") if dtype == "numeric" else s
            return s.reset_index(drop=True)

        out = pd.DataFrame()

        # IDs / source
        out["property_id"] = col("property_id").astype(str)
        out["source"] = col("_source_file").fillna("unknown").astype(str)

        # Segment: use property_type_grouped when available (cleaned CSV — already
        # normalised and validated). Fall back to raw property_type string mapping.
        pt_grouped = col("property_type_grouped").astype(str)
        has_grouped = pt_grouped.ne("nan") & pt_grouped.ne("None") & pt_grouped.ne("")
        _GROUPED_TO_SEG = {
            "apartment": "apt",
            "builder_floor": "builder_floor",
            "plot": "plot",
            "res_house": "house_villa",
            "villa": "house_villa",
        }
        pt_raw = col("property_type").astype(str)
        out["property_type_raw"] = pt_raw
        out["segment"] = np.where(
            has_grouped,
            pt_grouped.map(_GROUPED_TO_SEG),
            pt_raw.apply(lambda v: _map_segment(v) if v != "nan" else None),
        )

        # City (initial normalization)
        out["city"] = col("city").apply(lambda v: _norm_city(v) if pd.notna(v) else None)

        # Locality
        locality_raw = col("locality").fillna("").astype(str).str.strip()
        out["locality"] = locality_raw
        out["canonical_locality"] = locality_raw.str.title()

        # Locality-based city override: some localities are mis-tagged by source
        # (e.g. MB tags Noida Extension as "Noida" when it's geographically Greater Noida)
        for loc_title, correct_city in LOCALITY_CITY_OVERRIDES.items():
            mask = out["canonical_locality"] == loc_title
            out.loc[mask, "city"] = correct_city

        # Event type
        ev = col("event_type").astype(str).str.strip().str.lower()
        out["event_type"] = ev
        out["is_active_event"] = ev.isin(ACTIVE_EVENT_TYPES).astype(int)
        out["is_new_event"] = (ev == "new").astype(int)
        out["is_updated_event"] = (ev == "updated").astype(int)

        # Price
        out["price"] = pd.to_numeric(col("price_numeric"), errors="coerce")

        # Area → sqft
        area_val = pd.to_numeric(col("covered_area_value"), errors="coerce")
        area_unit = col("covered_area_unit").fillna("sqft").astype(str)
        out["area_sqft"] = pd.Series(
            [_to_sqft(v, u) for v, u in zip(area_val, area_unit)],
            dtype="Float64",
        )

        # PPSF: recompute then fall back to sqft_price
        sqft_price_raw = pd.to_numeric(col("sqft_price"), errors="coerce")
        computed_ppsf = out["price"] / out["area_sqft"].astype("float64")
        out["price_per_sqft"] = computed_ppsf.where(
            computed_ppsf.notna() & (computed_ppsf > 0), other=sqft_price_raw
        )

        # Possession
        poss = col("possession_status").fillna("").astype(str).str.lower().str.strip()
        out["possession_status"] = col("possession_status").fillna("").astype(str)
        out["is_ready"] = poss.apply(lambda v: int(any(kw in v for kw in _READY_KEYWORDS)))
        out["is_under_construction"] = poss.apply(lambda v: int(any(kw in v for kw in _UC_KEYWORDS)))

        # Dates
        out["posting_date"] = pd.to_datetime(col("posting_date"), errors="coerce")
        out["scrape_date"] = pd.to_datetime(col("scrape_date"), errors="coerce")

        # Agent / developer (handle column name differences between HO and MB)
        agent_raw = col("agent_type").fillna(col("user_type")).astype(str)
        out["agent_type"] = agent_raw
        dev_raw = col("developer_id").fillna(col("developer_uuid")).astype(str)
        out["developer_id"] = dev_raw

        # Days on market
        effective_start = out["posting_date"].fillna(out["scrape_date"])
        dom = (out["scrape_date"] - effective_start).dt.days.clip(lower=0)
        # When posting_date == scrape_date the scraper set the date artificially
        # (common in HO full-refresh uploads). Treat DOM as unknown so the
        # first_seen_scrape_date fallback in _compute_segment_artifact applies.
        same_day_mask = out["posting_date"].dt.normalize() == out["scrape_date"].dt.normalize()
        dom = dom.where(~same_day_mask, other=np.nan)
        out["days_on_market"] = dom

        return out

    # ─────────────────────────────────────────────────────────────
    # 2. Build and save artifacts
    # ─────────────────────────────────────────────────────────────

    def build_market_intelligence_artifacts(self) -> None:
        df = self.load_market_data()
        if df.empty:
            logger.warning("MarketIntelligenceService: no raw data loaded; skipping artifact build")
            return

        df = df[df["segment"].isin(SUPPORTED_SEGMENTS)].copy()
        if df.empty:
            logger.warning("MarketIntelligenceService: no rows for supported segments")
            return

        active_df = df[df["is_active_event"] == 1].copy()

        for seg in sorted(active_df["segment"].dropna().unique()):
            seg_active = active_df[active_df["segment"] == seg].copy()
            seg_full = df[df["segment"] == seg].copy()
            artifact = self._compute_segment_artifact(seg_active, seg_full)
            if artifact is None or artifact.empty:
                logger.warning("MarketIntelligenceService: no artifact produced for segment=%s", seg)
                continue

            out_dir = self.project_root / "opt" / seg
            out_dir.mkdir(parents=True, exist_ok=True)
            artifact.to_csv(out_dir / "market_intelligence.csv", index=False)
            logger.info(
                "MarketIntelligenceService: wrote market_intelligence.csv for segment=%s (%d rows)", seg, len(artifact)
            )

            latest = (
                artifact.sort_values("scrape_date", ascending=False)
                .drop_duplicates(["city", "canonical_locality"])
            )
            summary = {
                "segment": seg,
                "locality_count": int(latest["canonical_locality"].nunique()),
                "city_count": int(latest["city"].nunique()),
                "total_active_supply": int(latest["active_supply_stock"].sum()),
                "median_market_heat": float(latest["market_heat_index"].median()),
                "cities": sorted(latest["city"].dropna().unique().tolist()),
            }
            with open(out_dir / "market_summary.json", "w", encoding="utf-8") as fp:
                json.dump(summary, fp, indent=2)

        # Invalidate cache so next read picks up fresh files
        self._artifacts = {}

    def _compute_segment_artifact(
        self, active_df: pd.DataFrame, full_df: pd.DataFrame
    ) -> pd.DataFrame | None:
        if active_df.empty:
            return None

        # First-seen scrape date per property (for DOM fallback when posting_date missing)
        first_seen = (
            full_df.groupby("property_id")["scrape_date"].min().rename("first_seen_scrape_date")
        )
        active_df = active_df.join(first_seen, on="property_id", how="left")
        dom_fallback = (active_df["scrape_date"] - active_df["first_seen_scrape_date"]).dt.days.clip(lower=0)
        active_df["days_on_market"] = active_df["days_on_market"].fillna(dom_fallback).clip(lower=0)

        all_dates = sorted(active_df["scrape_date"].dropna().unique())

        grp_cols = ["segment", "city", "canonical_locality", "scrape_date"]
        rows: list[dict] = []

        for keys, grp in active_df.groupby(grp_cols, dropna=False):
            seg, city, locality, scrape_dt = keys
            if pd.isna(city) or not str(city).strip() or str(city) == "nan":
                continue
            if pd.isna(locality) or not str(locality).strip() or str(locality) == "nan":
                continue

            active_ids = set(grp["property_id"].unique())
            active_supply = len(active_ids)
            if active_supply == 0:
                continue

            # Scrape date index for absorption / price revision lookups
            try:
                dt_idx = all_dates.index(scrape_dt)
            except ValueError:
                dt_idx = -1

            # New supply
            new_count = int(grp["is_new_event"].sum())
            new_velocity = _safe_div(new_count, active_supply)

            # Proxy absorption (listings that vanished to next scrape run)
            # Only meaningful when the NEXT GLOBAL date also has data for this
            # specific locality. If the locality is absent on the next date it
            # simply means that scrape run didn't cover it — not that all
            # listings were sold, so we skip to avoid a false 100 % signal.
            absorbed_count = 0
            absorption_computable = False
            if dt_idx >= 0 and dt_idx + 1 < len(all_dates):
                next_dt = all_dates[dt_idx + 1]
                next_ids = set(
                    active_df[
                        (active_df["segment"] == seg)
                        & (active_df["city"] == city)
                        & (active_df["canonical_locality"] == locality)
                        & (active_df["scrape_date"] == next_dt)
                    ]["property_id"].unique()
                )
                if len(next_ids) > 0:
                    # Next scrape visited this locality — delta is meaningful
                    absorbed_count = len(active_ids - next_ids)
                    absorption_computable = True
                # else: locality absent on next date → skip (keep 0, not computable)
            absorption_rate = _safe_div(absorbed_count, active_supply) if absorption_computable else 0.0

            # Price revision from updated listings vs previous scrape
            updated_grp = grp[grp["is_updated_event"] == 1]
            updated_count = int(len(updated_grp))
            price_cut_count = price_hike_count = 0
            price_cut_pcts: list[float] = []
            price_hike_pcts: list[float] = []

            if dt_idx > 0 and updated_count > 0:
                # Use MOST RECENT prior price from ANY earlier scrape date.
                # An UPDATED event's original NEW listing can be several scrape dates back,
                # so looking only at prev_dt (immediately previous date) always misses it.
                prior_loc_df = active_df[
                    (active_df["segment"] == seg)
                    & (active_df["city"] == city)
                    & (active_df["canonical_locality"] == locality)
                    & (active_df["scrape_date"].isin(all_dates[:dt_idx]))
                ]
                prior_prices = (
                    prior_loc_df.sort_values("scrape_date")
                    .drop_duplicates("property_id", keep="last")
                    .set_index("property_id")["price"]
                    .to_dict()
                )
                for _, row_u in updated_grp.iterrows():
                    pid = row_u["property_id"]
                    curr_p = row_u["price"]
                    prev_p = prior_prices.get(pid)
                    if prev_p and pd.notna(prev_p) and pd.notna(curr_p) and prev_p > 0:
                        pct = (curr_p - prev_p) / prev_p * 100.0
                        if curr_p < prev_p:
                            price_cut_count += 1
                            price_cut_pcts.append(pct)
                        elif curr_p > prev_p:
                            price_hike_count += 1
                            price_hike_pcts.append(pct)

            price_cut_freq = _safe_div(price_cut_count, updated_count)
            price_hike_freq = _safe_div(price_hike_count, updated_count)
            price_cut_median_pct = float(np.median(price_cut_pcts)) if price_cut_pcts else 0.0
            price_hike_median_pct = float(np.median(price_hike_pcts)) if price_hike_pcts else 0.0

            # Days on market and stale inventory
            dom_vals = pd.to_numeric(grp["days_on_market"], errors="coerce").dropna()
            median_dom = float(dom_vals.median()) if len(dom_vals) > 0 else 0.0
            stale_count = int((dom_vals > 90).sum())
            stale_share = _safe_div(stale_count, active_supply)

            # Possession split
            ready_count = int(grp["is_ready"].sum())
            uc_count_loc = int(grp["is_under_construction"].sum())
            ready_share = _safe_div(ready_count, active_supply)
            uc_share = _safe_div(uc_count_loc, active_supply)

            # Price statistics
            ppsf_vals = pd.to_numeric(grp["price_per_sqft"], errors="coerce")
            ppsf_vals = ppsf_vals[ppsf_vals.notna() & (ppsf_vals > 0)]
            price_vals = pd.to_numeric(grp["price"], errors="coerce")
            price_vals = price_vals[price_vals.notna() & (price_vals > 0)]
            median_ppsf = float(ppsf_vals.median()) if len(ppsf_vals) > 0 else 0.0
            p25_ppsf = float(ppsf_vals.quantile(0.25)) if len(ppsf_vals) > 0 else 0.0
            p75_ppsf = float(ppsf_vals.quantile(0.75)) if len(ppsf_vals) > 0 else 0.0
            median_price = float(price_vals.median()) if len(price_vals) > 0 else 0.0

            # Supply participants
            agent_count = int(grp["agent_type"].replace("nan", pd.NA).dropna().nunique())
            dev_count = int(grp["developer_id"].replace("nan", pd.NA).dropna().nunique())

            # Circle rate lookup for this locality
            circle_rate: float = 0.0
            if self._cr_matcher is not None:
                cr_val = self._cr_matcher.get_rate(str(city), str(locality), "Residential")
                if cr_val is not None and cr_val > 0:
                    circle_rate = float(cr_val)
            price_to_circle = round(median_ppsf / circle_rate, 3) if circle_rate > 0 else 0.0

            rows.append({
                "segment": seg,
                "city": city,
                "locality": locality,
                "canonical_locality": locality,
                "scrape_date": scrape_dt,
                "active_supply_stock": active_supply,
                "new_supply_count": new_count,
                "new_supply_velocity": round(new_velocity, 4),
                "updated_count": updated_count,
                "absorbed_count": absorbed_count,
                "absorption_rate": round(absorption_rate, 4),
                "price_cut_count": price_cut_count,
                "price_cut_frequency": round(price_cut_freq, 4),
                "price_hike_count": price_hike_count,
                "price_hike_frequency": round(price_hike_freq, 4),
                "price_hike_median_pct": round(price_hike_median_pct, 2),
                "price_cut_median_pct": round(price_cut_median_pct, 2),
                "median_days_on_market": round(median_dom, 1),
                "stale_inventory_count": stale_count,
                "stale_inventory_share": round(stale_share, 4),
                "median_price": round(median_price, 0),
                "median_ppsf": round(median_ppsf, 0),
                "p25_ppsf": round(p25_ppsf, 0),
                "p75_ppsf": round(p75_ppsf, 0),
                "ready_inventory_share": round(ready_share, 4),
                "under_construction_share": round(uc_share, 4),
                "agent_count": agent_count,
                "developer_count": dev_count,
                "median_circle_rate": round(circle_rate, 0),
                "price_to_circle_ratio": price_to_circle,
                # Indices computed after full table is built
                "supply_pressure_index": 50.0,
                "demand_strength_index": 50.0,
                "liquidity_index": 50.0,
                "price_momentum_score": 50.0,
                "circle_premium_score": 50.0,
                "market_heat_index": 50.0,
                "mhi_rolling4": 50.0,
                "market_label": "Balanced market",
            })

        if not rows:
            return None

        artifact = pd.DataFrame(rows)
        artifact = self._compute_indices(artifact)
        return artifact

    def _compute_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill supply_pressure_index, liquidity_index, demand_strength_index,
        market_heat_index using percentile scoring within each city group."""
        df = df.copy()

        for city_val, idx in df.groupby("city").groups.items():
            g = df.loc[idx]

            def pct(col: str, higher_is_better: bool = True) -> pd.Series:
                s = pd.to_numeric(g[col], errors="coerce").fillna(0.0)
                return percentile_score(s, higher_is_better=higher_is_better)

            S_active = pct("active_supply_stock")
            S_new_vel = pct("new_supply_velocity")
            S_stale_hi = pct("stale_inventory_share")
            S_uc = pct("under_construction_share")
            supply_pressure = (
                0.40 * S_active + 0.25 * S_new_vel + 0.20 * S_stale_hi + 0.15 * S_uc
            ).clip(0, 100)
            df.loc[idx, "supply_pressure_index"] = supply_pressure.round(1)

            S_absorption = pct("absorption_rate")
            S_fast_dom = pct("median_days_on_market", higher_is_better=False)
            S_low_stale = pct("stale_inventory_share", higher_is_better=False)
            liquidity = (
                0.50 * S_absorption + 0.30 * S_fast_dom + 0.20 * S_low_stale
            ).clip(0, 100)
            df.loc[idx, "liquidity_index"] = liquidity.round(1)

            pc_freq = pd.to_numeric(g["price_cut_frequency"], errors="coerce").fillna(0.0)
            ph_freq = pd.to_numeric(g["price_hike_frequency"], errors="coerce").fillna(0.0)
            uc_cnt = pd.to_numeric(g["updated_count"], errors="coerce").fillna(0.0)
            momentum = (50.0 + 40.0 * ph_freq - 40.0 * pc_freq).clip(0, 100)
            momentum = momentum.where(uc_cnt > 0, other=50.0)
            df.loc[idx, "price_momentum_score"] = momentum.round(1)

            median_ppsf = pd.to_numeric(g["median_ppsf"], errors="coerce").fillna(0.0)
            cr_col = pd.to_numeric(g["median_circle_rate"], errors="coerce").fillna(0.0)
            # Use circle rate when available; fall back to city-median ppsf for unmatched localities
            city_ppsf_med = median_ppsf.median() if median_ppsf.median() > 0 else 1.0
            denom = cr_col.where(cr_col > 0, other=city_ppsf_med)
            relative = median_ppsf / denom.replace(0, np.nan).fillna(1.0)
            circle_premium = (50.0 + 50.0 * (relative - 1.0)).clip(0, 100)
            df.loc[idx, "circle_premium_score"] = circle_premium.round(1)
            # Update price_to_circle_ratio for rows that didn't have a circle rate at row-build time
            ratio = (median_ppsf / cr_col.replace(0, np.nan)).fillna(df.loc[idx, "price_to_circle_ratio"]).round(3)
            df.loc[idx, "price_to_circle_ratio"] = ratio

            demand = (
                0.35 * S_absorption + 0.25 * liquidity + 0.20 * momentum + 0.20 * circle_premium
            ).clip(0, 100)
            df.loc[idx, "demand_strength_index"] = demand.round(1)

            heat = (50.0 + 0.5 * demand - 0.5 * supply_pressure).clip(0, 100)
            df.loc[idx, "market_heat_index"] = heat.round(1)

        df["market_label"] = df["market_heat_index"].apply(market_label)

        # Rolling 4-scrape MHI smoothing per (city, canonical_locality)
        # Sorted by scrape_date within each group so the window is chronological.
        df = df.sort_values(["segment", "city", "canonical_locality", "scrape_date"]).copy()
        df["mhi_rolling4"] = (
            df.groupby(["segment", "city", "canonical_locality"])["market_heat_index"]
            .transform(lambda s: s.rolling(window=4, min_periods=1).mean())
            .round(1)
        )
        return df

    # ─────────────────────────────────────────────────────────────
    # 3. Load cached artifact
    # ─────────────────────────────────────────────────────────────

    def _load_artifact(self, segment: str) -> pd.DataFrame | None:
        if segment in self._artifacts:
            return self._artifacts[segment]
        path = self.project_root / "opt" / segment / "market_intelligence.csv"
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path, low_memory=False, parse_dates=["scrape_date"])
            self._artifacts[segment] = df
            return df
        except Exception as exc:
            logger.warning("MarketIntelligenceService: failed to load artifact for %s: %s", segment, exc)
            return None

    # ─────────────────────────────────────────────────────────────
    # 4. Public API methods
    # ─────────────────────────────────────────────────────────────

    def get_available_cities(self, segment: str) -> list[str]:
        df = self._load_artifact(segment)
        if df is None or df.empty:
            return []
        return sorted(str(c) for c in df["city"].dropna().unique() if str(c) != "nan")

    def get_available_localities(
        self, segment: str, city: str, query: str = "", limit: int = 30
    ) -> list[str]:
        df = self._load_artifact(segment)
        if df is None or df.empty:
            return []
        city_df = df[df["city"].str.lower() == city.strip().lower()]
        localities = [
            str(l) for l in city_df["canonical_locality"].dropna().unique()
            if str(l) != "nan" and str(l).strip()
        ]
        if query:
            q = query.strip().lower()
            localities = [l for l in localities if q in l.lower()]
        return sorted(localities)[:limit]

    def get_market_context(self, segment: str, city: str, locality: str) -> dict:
        df = self._load_artifact(segment)
        if df is None or df.empty:
            return self._empty_context(segment, city, locality, reason="No artifact available. Run build_market_intelligence_artifacts() first.")

        city_df = df[df["city"].str.lower() == city.strip().lower()]
        if city_df.empty:
            return self._empty_context(segment, city, locality, reason=f"City '{city}' not found in artifact.")

        canonical = self._resolve_locality(locality, city_df)
        if canonical is None:
            return self._empty_context(segment, city, locality, reason=f"Locality '{locality}' not found.")

        loc_df = city_df[city_df["canonical_locality"] == canonical].sort_values("scrape_date")
        if loc_df.empty:
            return self._empty_context(segment, city, locality, reason="No data rows for locality.")

        latest = loc_df.iloc[-1]
        latest_date = (
            str(pd.Timestamp(latest["scrape_date"]).date())
            if pd.notna(latest.get("scrape_date"))
            else ""
        )

        def _flt(col: str, default: float = 0.0) -> float:
            v = latest.get(col, default)
            try:
                f = float(v)
                return default if (f != f) else f  # nan check
            except (TypeError, ValueError):
                return default

        def _int(col: str, default: int = 0) -> int:
            return int(_flt(col, float(default)))

        kpis = {
            "activeSupplyStock": _int("active_supply_stock"),
            "newSupplyVelocity": _flt("new_supply_velocity"),
            "absorptionRate": _flt("absorption_rate"),
            "priceCutFrequency": _flt("price_cut_frequency"),
            "priceHikeFrequency": _flt("price_hike_frequency"),
            "medianDaysOnMarket": _flt("median_days_on_market"),
            "staleInventoryShare": _flt("stale_inventory_share"),
            "medianPpsf": _flt("median_ppsf"),
            "medianPrice": _flt("median_price"),
            "p25Ppsf": _flt("p25_ppsf"),
            "p75Ppsf": _flt("p75_ppsf"),
            "medianCircleRate": _flt("median_circle_rate"),
            "priceToCircleRatio": _flt("price_to_circle_ratio"),
            "readyInventoryShare": _flt("ready_inventory_share"),
            "underConstructionShare": _flt("under_construction_share"),
            "agentCount": _int("agent_count"),
            "developerCount": _int("developer_count"),
        }

        indices = {
            "supplyPressureIndex": _flt("supply_pressure_index", 50.0),
            "demandStrengthIndex": _flt("demand_strength_index", 50.0),
            "liquidityIndex": _flt("liquidity_index", 50.0),
            "priceMomentumScore": _flt("price_momentum_score", 50.0),
            "circlePremiumScore": _flt("circle_premium_score", 50.0),
            "marketHeatIndex": _flt("market_heat_index", 50.0),
            "marketHeatSmoothed": _flt("mhi_rolling4", 50.0),
            "marketLabel": str(latest.get("market_label", "Balanced market")),
        }

        def _series(col: str) -> list[dict]:
            return [
                {"date": str(pd.Timestamp(r["scrape_date"]).date()), "value": float(r[col])}
                for _, r in loc_df.iterrows()
                if pd.notna(r.get(col)) and pd.notna(r.get("scrape_date"))
            ]

        series = {
            "supplyTrend": _series("active_supply_stock"),
            "newSupplyTrend": _series("new_supply_count"),
            "absorptionTrend": _series("absorption_rate"),
            "priceCutTrend": _series("price_cut_frequency"),
            "priceHikeTrend": _series("price_hike_frequency"),
            "medianPpsfTrend": _series("median_ppsf"),
            "marketHeatTrend": _series("market_heat_index"),
            "marketHeatSmoothedTrend": _series("mhi_rolling4"),
        }

        drivers = self._build_market_drivers(kpis, indices)

        return {
            "segment": segment,
            "city": city,
            "locality": locality,
            "canonicalLocality": canonical,
            "latestScrapeDate": latest_date,
            "kpis": kpis,
            "indices": indices,
            "series": series,
            "drivers": drivers,
        }

    # ─────────────────────────────────────────────────────────────
    # 5. Helpers
    # ─────────────────────────────────────────────────────────────

    def _resolve_locality(self, locality: str, city_df: pd.DataFrame) -> str | None:
        if city_df.empty:
            return None
        query = locality.strip().lower()
        candidates = [
            str(c) for c in city_df["canonical_locality"].dropna().unique()
            if str(c) != "nan"
        ]
        if not candidates:
            return None

        # Exact
        for c in candidates:
            if c.lower() == query:
                return c
        # Substring
        subs = [c for c in candidates if query in c.lower() or c.lower() in query]
        if subs:
            return sorted(subs, key=lambda x: abs(len(x) - len(query)))[0]
        # Token overlap
        q_tokens = set(query.split())
        scored = sorted(
            candidates,
            key=lambda c: len(q_tokens & set(c.lower().split())) / max(len(q_tokens), 1),
            reverse=True,
        )
        best = scored[0]
        best_overlap = len(q_tokens & set(best.lower().split())) / max(len(q_tokens), 1)
        return best if best_overlap > 0.3 else None

    def _build_market_drivers(self, kpis: dict, indices: dict) -> list[dict]:
        drivers: list[dict] = []

        # 1. Inventory Pressure
        spi = indices.get("supplyPressureIndex", 50.0)
        if spi >= 70:
            score, label = -2, "Very high competing supply"
        elif spi >= 60:
            score, label = -1, "High competing supply"
        elif spi <= 35:
            score, label = 1, "Low inventory pressure"
        else:
            score, label = 0, "Moderate supply levels"
        drivers.append({
            "title": "Inventory Pressure",
            "score": score,
            "label": label,
            "explanation": (
                f"Supply pressure index: {spi:.0f}/100. "
                f"Active inventory is {'elevated' if score < 0 else 'low' if score > 0 else 'moderate'} "
                f"compared with similar localities in this city and segment."
            ),
        })

        # 2. Demand Momentum
        dsi = indices.get("demandStrengthIndex", 50.0)
        if dsi >= 75:
            score, label = 2, "Strong demand momentum"
        elif dsi >= 60:
            score, label = 1, "Healthy absorption"
        elif dsi < 35:
            score, label = -2, "Weak demand signals"
        elif dsi < 45:
            score, label = -1, "Below-average demand"
        else:
            score, label = 0, "Balanced demand"
        drivers.append({
            "title": "Demand Momentum",
            "score": score,
            "label": label,
            "explanation": (
                f"Demand strength index: {dsi:.0f}/100. "
                f"Absorption, liquidity, and pricing signals indicate "
                f"{'strong' if score > 0 else 'weak' if score < 0 else 'balanced'} buyer demand."
            ),
        })

        # 3. Liquidity
        li = indices.get("liquidityIndex", 50.0)
        if li >= 75:
            score, label = 2, "High exit liquidity"
        elif li >= 60:
            score, label = 1, "Moderate exit liquidity"
        elif li < 35:
            score, label = -2, "Low liquidity"
        elif li < 45:
            score, label = -1, "Below-average liquidity"
        else:
            score, label = 0, "Average resale liquidity"
        drivers.append({
            "title": "Liquidity",
            "score": score,
            "label": label,
            "explanation": (
                f"Liquidity index: {li:.0f}/100. "
                f"Days on market and absorption indicate "
                f"{'easy' if score > 0 else 'difficult' if score < 0 else 'moderate'} resale conditions."
            ),
        })

        # 4. Price Revision Behavior
        pc = kpis.get("priceCutFrequency", 0.0)
        ph = kpis.get("priceHikeFrequency", 0.0)
        if pc > 0.30:
            score, label = -2, "Heavy discounting visible"
        elif pc > 0.18:
            score, label = -1, "Some discounting visible"
        elif ph > 0.15:
            score, label = 1, "Price hikes outnumber cuts"
        else:
            score, label = 0, "Stable asking prices"
        drivers.append({
            "title": "Price Revision Behavior",
            "score": score,
            "label": label,
            "explanation": (
                f"Price cuts: {pc*100:.1f}% of updated listings. "
                f"Price hikes: {ph*100:.1f}%. "
                f"{'Sellers are reducing asking prices.' if score < 0 else 'Sellers are holding or raising prices.' if score > 0 else 'Asking prices are stable.'}"
            ),
        })

        # 5. Construction Pipeline
        uc = kpis.get("underConstructionShare", 0.0)
        ar = kpis.get("absorptionRate", 0.0)
        ready = kpis.get("readyInventoryShare", 0.0)
        li_val = indices.get("liquidityIndex", 50.0)
        if uc > 0.60 and ar < 0.05:
            score, label = -2, "Heavy unabsorbed pipeline"
        elif uc > 0.40:
            score, label = -1, "Significant under-construction stock"
        elif ready > 0.70 and li_val > 60:
            score, label = 1, "Ready inventory dominates"
        else:
            score, label = 0, "Mixed ready/under-construction"
        drivers.append({
            "title": "Construction Pipeline",
            "score": score,
            "label": label,
            "explanation": (
                f"Under-construction share: {uc*100:.1f}%. Ready to move: {ready*100:.1f}%. "
                f"{'High pipeline risk with low absorption.' if score == -2 else 'Meaningful pipeline.' if score == -1 else 'Strong ready inventory.' if score > 0 else 'Balanced supply mix.'}"
            ),
        })

        return drivers

    def _empty_context(self, segment: str, city: str, locality: str, reason: str = "") -> dict:
        return {
            "segment": segment,
            "city": city,
            "locality": locality,
            "canonicalLocality": locality,
            "latestScrapeDate": "",
            "error": reason,
            "kpis": {},
            "indices": {
                "supplyPressureIndex": 50.0,
                "demandStrengthIndex": 50.0,
                "liquidityIndex": 50.0,
                "marketHeatIndex": 50.0,
                "marketLabel": "Balanced market",
            },
            "series": {},
            "drivers": [],
        }
