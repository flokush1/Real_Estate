from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForecastSegmentConfig:
    key: str
    label: str
    out_dir: str
    trend_root: str
    property_file: str


SEGMENTS: dict[str, ForecastSegmentConfig] = {
    "builder-floor": ForecastSegmentConfig(
        key="builder-floor",
        label="Builder Floor",
        out_dir="opt/builder_floor",
        trend_root="inputs/interpolated_growth_trend_builder/builder",
        property_file="inputs/builder_floor_with_pi.csv",
    ),
    "apartment": ForecastSegmentConfig(
        key="apartment",
        label="Apartment / Flat",
        out_dir="opt/apt",
        trend_root="inputs/interpolated_growth_trend_builder/apt",
        property_file="inputs/apartment_with_pi.csv",
    ),
    "plot": ForecastSegmentConfig(
        key="plot",
        label="Plot",
        out_dir="opt/plot",
        trend_root="inputs/interpolated_growth_trend_builder/plot",
        property_file="inputs/plot_with_pi.csv",
    ),
}


def _norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def normalize_city_name(name: Any) -> str:
    text = _norm_text(name)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", "_", text).strip("_")
    aliases = {
        "new_delhi": "delhi",
        "north_delhi": "delhi",
        "south_delhi": "delhi",
        "east_delhi": "delhi",
        "west_delhi": "delhi",
        "central_delhi": "delhi",
        "ncr_delhi": "delhi",
        "dwarka": "delhi",
        "greater_noida": "noida",
        "greaternoida": "noida",
        "greater_noida_west": "noida",
        "gurugram": "gurgaon",
    }
    return aliases.get(text, text)


def normalize_locality_key(name: Any) -> str:
    text = _norm_text(name)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"([a-z])([0-9])", r"\1 \2", text)
    text = re.sub(r"([0-9])([a-z])", r"\1 \2", text)
    return re.sub(r"\s+", "_", text).strip("_")


def _locality_score(query: str, candidate: str) -> float:
    q = normalize_locality_key(query)
    c = normalize_locality_key(candidate)
    if not q or not c:
        return 0.0
    if q == c:
        return 100.0

    ratio = SequenceMatcher(None, q, c).ratio() * 100.0
    if q in c:
        ratio += 12.0
    if c.startswith(q):
        ratio += 8.0

    q_tokens = set(q.split("_"))
    c_tokens = set(c.split("_"))
    if q_tokens:
        overlap = len(q_tokens & c_tokens) / len(q_tokens)
        ratio += overlap * 20.0
    return ratio


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(out) or np.isinf(out):
        return default
    return out


class ForecastIntelligenceService:
    def __init__(self, project_root: str | Path) -> None:
        self.root = Path(project_root)
        self._small_csv_cache: dict[tuple[str, str], pd.DataFrame] = {}
        self._json_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._trend_index_cache: dict[str, dict[str, Path]] = {}

    def _cfg(self, segment: str) -> ForecastSegmentConfig:
        key = _norm_text(segment).replace("_", "-")
        if key not in SEGMENTS:
            raise ValueError(f"Unsupported forecast segment: {segment}")
        return SEGMENTS[key]

    def _property_forecast_path(self, cfg: ForecastSegmentConfig) -> Path:
        out_dir = self.root / cfg.out_dir
        if cfg.key == "apartment":
            preferred = out_dir / "property_forecasts_flodata_new.csv"
            if preferred.exists():
                return preferred
        return out_dir / "property_forecasts_flodata.csv"

    def _load_json(self, cfg: ForecastSegmentConfig, filename: str) -> dict[str, Any]:
        cache_key = (cfg.key, filename)
        if cache_key in self._json_cache:
            return self._json_cache[cache_key]

        fpath = self.root / cfg.out_dir / filename
        if not fpath.exists():
            self._json_cache[cache_key] = {}
            return {}

        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
        except Exception:
            data = {}

        self._json_cache[cache_key] = data
        return data

    def _load_small_csv(self, cfg: ForecastSegmentConfig, filename: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
        cache_key = (cfg.key, filename)
        if cache_key in self._small_csv_cache:
            return self._small_csv_cache[cache_key].copy()

        fpath = self.root / cfg.out_dir / filename
        if not fpath.exists():
            self._small_csv_cache[cache_key] = pd.DataFrame()
            return pd.DataFrame()

        df = pd.read_csv(fpath)
        for col in parse_dates or []:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        self._small_csv_cache[cache_key] = df
        return df.copy()

    def _load_property_master(self, cfg: ForecastSegmentConfig) -> pd.DataFrame:
        cache_key = (cfg.key, "property_master")
        if cache_key in self._small_csv_cache:
            return self._small_csv_cache[cache_key].copy()

        fpath = self.root / cfg.property_file
        if not fpath.exists():
            self._small_csv_cache[cache_key] = pd.DataFrame()
            return pd.DataFrame()

        df = pd.read_csv(fpath)
        if "property_id" not in df.columns:
            df["property_id"] = [f"prop_{i}" for i in range(len(df))]
        if "pi_price_per_sqft" not in df.columns and "predicted_price_per_sqft" in df.columns:
            df["pi_price_per_sqft"] = df["predicted_price_per_sqft"]
        if "covered_area_sqft" not in df.columns and "plot_area" in df.columns:
            df["covered_area_sqft"] = df["plot_area"]
        if "city" not in df.columns:
            df["city"] = ""

        df["property_id"] = df["property_id"].astype(str)
        df["locality"] = df.get("locality", "").astype(str).str.strip()
        df["city"] = df.get("city", "").astype(str).str.strip()
        df["city_key"] = df["city"].apply(normalize_city_name)
        df["locality_key"] = df["locality"].apply(normalize_locality_key)

        self._small_csv_cache[cache_key] = df
        return df.copy()

    def _trend_index(self, cfg: ForecastSegmentConfig) -> dict[str, Path]:
        if cfg.key in self._trend_index_cache:
            return self._trend_index_cache[cfg.key]

        root = self.root / cfg.trend_root
        idx: dict[str, Path] = {}
        if root.exists():
            for fp in root.rglob("*.csv"):
                key = f"{fp.parent.name}/{fp.stem}".strip().casefold()
                idx[key] = fp

        self._trend_index_cache[cfg.key] = idx
        return idx

    def _load_property_forecast_filtered(
        self,
        cfg: ForecastSegmentConfig,
        *,
        property_id: str | None = None,
        locality: str | None = None,
    ) -> pd.DataFrame:
        path = self._property_forecast_path(cfg)
        if not path.exists():
            return pd.DataFrame()

        usecols = [
            "property_id",
            "locality",
            "trend_locality",
            "date",
            "quarter",
            "decay_t",
            "P_i",
            "L_i",
            "rho_0",
            "rho_t",
            "I_lt",
            "forecast_price_per_sqft",
            "forecast_price_numeric",
        ]

        pid_key = str(property_id).strip() if property_id else None
        loc_key = _norm_text(locality) if locality else None
        frames: list[pd.DataFrame] = []

        for chunk in pd.read_csv(path, usecols=usecols, chunksize=200_000):
            mask = pd.Series(True, index=chunk.index)
            if pid_key:
                mask &= chunk["property_id"].astype(str).str.strip() == pid_key
            if loc_key:
                mask &= chunk["locality"].astype(str).str.strip().str.casefold() == loc_key

            filtered = chunk.loc[mask]
            if not filtered.empty:
                frames.append(filtered)

        if not frames:
            return pd.DataFrame(columns=usecols)

        out = pd.concat(frames, ignore_index=True)
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date"]).sort_values("date")
        return out

    def _resolve_trend_locality(
        self,
        selected_locality: str,
        selected_city: str,
        locality_forecasts: pd.DataFrame,
    ) -> str | None:
        if locality_forecasts.empty or "locality" not in locality_forecasts.columns:
            return None

        pools = locality_forecasts["locality"].dropna().astype(str).tolist()
        if not pools:
            return None

        city_key = normalize_city_name(selected_city)
        if city_key:
            city_scoped = [
                loc for loc in pools if normalize_city_name(str(loc).split("/")[0]) == city_key
            ]
            candidates = city_scoped or pools
        else:
            candidates = pools

        target = normalize_locality_key(selected_locality)
        by_slug: dict[str, list[str]] = {}
        for full in candidates:
            slug = str(full).split("/")[-1]
            by_slug.setdefault(normalize_locality_key(slug), []).append(full)

        if target in by_slug:
            return by_slug[target][0]

        if by_slug:
            best = max(by_slug.keys(), key=lambda k: _locality_score(target, k))
            if _locality_score(target, best) >= 70.0:
                return by_slug[best][0]

        return None

    def _load_hist_trend(self, cfg: ForecastSegmentConfig, trend_locality: str) -> pd.DataFrame:
        idx = self._trend_index(cfg)
        fp = idx.get(str(trend_locality).strip().casefold())
        if fp is None or not fp.exists():
            return pd.DataFrame(columns=["date", "price_per_sqft"])

        tmp = pd.read_csv(fp)
        if "date" not in tmp.columns or "price_per_sqft" not in tmp.columns:
            return pd.DataFrame(columns=["date", "price_per_sqft"])

        out = tmp[["date", "price_per_sqft"]].copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["price_per_sqft"] = pd.to_numeric(out["price_per_sqft"], errors="coerce")
        out = out.dropna(subset=["date", "price_per_sqft"]).sort_values("date")
        return out

    @staticmethod
    def _compute_property_yoy_series(
        property_forecast_df: pd.DataFrame,
        as_of: pd.Timestamp | None = None,
        years: int = 5,
    ) -> list[dict[str, Any]]:
        if property_forecast_df.empty:
            return []

        as_of = (as_of or pd.Timestamp.today()).normalize()
        s = property_forecast_df[["date", "forecast_price_per_sqft"]].copy()
        s = s.dropna(subset=["date", "forecast_price_per_sqft"]).sort_values("date")
        if s.empty:
            return []

        start_candidates = s[s["date"] >= as_of]
        start_row = start_candidates.iloc[0] if not start_candidates.empty else s.iloc[-1]
        start_date = pd.Timestamp(start_row["date"])
        start_price = _safe_float(start_row["forecast_price_per_sqft"], default=0.0)
        if start_price <= 0:
            return []

        anchor_date = start_date
        anchor_price = start_price
        rows: list[dict[str, Any]] = []

        for year_idx in range(1, years + 1):
            target_cutoff = start_date + pd.DateOffset(years=year_idx)
            target_candidates = s[s["date"] >= target_cutoff]
            if target_candidates.empty:
                break

            target_row = target_candidates.iloc[0]
            target_price = _safe_float(target_row["forecast_price_per_sqft"], default=0.0)
            if anchor_price <= 0:
                break

            yoy_pct = (target_price / anchor_price - 1.0) * 100.0
            rows.append(
                {
                    "label": f"Y+{year_idx}",
                    "yoy_pct": yoy_pct,
                    "anchor_date": pd.Timestamp(anchor_date).date().isoformat(),
                    "anchor_price": anchor_price,
                    "target_date": pd.Timestamp(target_row["date"]).date().isoformat(),
                    "target_price": target_price,
                }
            )

            anchor_date = pd.Timestamp(target_row["date"])
            anchor_price = target_price

        return rows

    @staticmethod
    def _compute_locality_yoy_from_today(
        forecast_df: pd.DataFrame,
        trend_locality: str,
        as_of: pd.Timestamp | None = None,
    ) -> dict[str, Any] | None:
        if forecast_df.empty or not trend_locality:
            return None

        as_of = (as_of or pd.Timestamp.today()).normalize()
        locality_norm = str(trend_locality).strip().casefold()

        tmp = forecast_df.copy()
        tmp["locality_norm"] = tmp["locality"].astype(str).str.strip().str.casefold()
        tmp = tmp[tmp["locality_norm"] == locality_norm]
        if tmp.empty:
            return None

        agg = (
            tmp.groupby("date", as_index=False)["pred_price_per_sqft"]
            .median()
            .sort_values("date")
        )
        if agg.empty:
            return None

        base_candidates = agg[agg["date"] >= as_of]
        base_row = base_candidates.iloc[0] if not base_candidates.empty else agg.iloc[-1]

        target_cutoff = pd.Timestamp(base_row["date"]) + pd.DateOffset(years=1)
        target_candidates = agg[agg["date"] >= target_cutoff]
        target_row = target_candidates.iloc[0] if not target_candidates.empty else agg.iloc[-1]

        base_price = _safe_float(base_row["pred_price_per_sqft"], default=0.0)
        target_price = _safe_float(target_row["pred_price_per_sqft"], default=0.0)
        if base_price <= 0:
            return None

        yoy_pct = (target_price / base_price - 1.0) * 100.0
        return {
            "yoy_pct": yoy_pct,
            "base_price": base_price,
            "target_price": target_price,
            "base_date": pd.Timestamp(base_row["date"]).date().isoformat(),
            "target_date": pd.Timestamp(target_row["date"]).date().isoformat(),
        }

    @staticmethod
    def _to_date_records(df: pd.DataFrame, value_col: str) -> list[dict[str, Any]]:
        if df.empty or "date" not in df.columns or value_col not in df.columns:
            return []
        out = df[["date", value_col]].dropna(subset=["date", value_col]).copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date.astype(str)
        out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
        out = out.dropna(subset=[value_col])
        return out.to_dict(orient="records")

    def available_segments(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for cfg in SEGMENTS.values():
            out_dir = self.root / cfg.out_dir
            rows.append(
                {
                    "key": cfg.key,
                    "label": cfg.label,
                    "available": str(out_dir.exists() and out_dir.is_dir()).lower(),
                }
            )
        return rows

    def list_cities(self, segment: str) -> list[str]:
        cfg = self._cfg(segment)
        props = self._load_property_master(cfg)
        if props.empty:
            return []

        cities = sorted(
            {
                str(c).strip()
                for c in props["city"].dropna().astype(str).tolist()
                if str(c).strip() and str(c).strip().lower() not in {"nan", "none"}
            }
        )

        if cities:
            return cities

        loc_fc = self._load_small_csv(cfg, "future_forecasts.csv", parse_dates=["date"])
        if "zone" in loc_fc.columns:
            return sorted({str(z).strip() for z in loc_fc["zone"].dropna().astype(str).tolist() if str(z).strip()})
        return []

    def suggest_localities(self, segment: str, city: str = "", query: str = "", limit: int = 8) -> list[str]:
        cfg = self._cfg(segment)
        props = self._load_property_master(cfg)
        if props.empty:
            return []

        view = props.copy()
        city_key = normalize_city_name(city)
        if city_key and "city_key" in view.columns:
            scoped = view[view["city_key"] == city_key]
            if not scoped.empty:
                view = scoped

        all_localities = sorted(
            {
                str(loc).strip()
                for loc in view["locality"].dropna().astype(str).tolist()
                if str(loc).strip()
            },
            key=lambda x: x.casefold(),
        )
        if not all_localities:
            return []

        q = query.strip()
        if not q:
            return all_localities[:limit]

        ranked = sorted(
            ((loc, _locality_score(q, loc)) for loc in all_localities),
            key=lambda t: t[1],
            reverse=True,
        )
        return [loc for loc, score in ranked[:limit] if score >= 35.0]

    def list_property_ids(self, segment: str, locality: str, city: str = "", limit: int = 250) -> list[str]:
        cfg = self._cfg(segment)
        props = self._load_property_master(cfg)
        if props.empty:
            return []

        view = props.copy()
        city_key = normalize_city_name(city)
        if city_key and "city_key" in view.columns:
            scoped = view[view["city_key"] == city_key]
            if not scoped.empty:
                view = scoped

        loc_key = normalize_locality_key(locality)
        if loc_key:
            exact = view[view["locality_key"] == loc_key]
            if exact.empty:
                choices = view["locality_key"].dropna().astype(str).unique().tolist()
                hit = get_close_matches(loc_key, choices, n=1, cutoff=0.72)
                if hit:
                    exact = view[view["locality_key"] == hit[0]]
            view = exact if not exact.empty else view.iloc[0:0]

        raw_ids = view["property_id"].dropna().astype(str).tolist()

        # Preserve original ordering but remove duplicate property IDs.
        seen: set[str] = set()
        deduped: list[str] = []
        for pid in raw_ids:
            if pid in seen:
                continue
            seen.add(pid)
            deduped.append(pid)
            if len(deduped) >= limit:
                break

        return deduped

    def overview(self, segment: str) -> dict[str, Any]:
        cfg = self._cfg(segment)
        metrics = self._load_json(cfg, "metrics.json")

        summary_path = self.root / cfg.out_dir / "model_summary.md"
        summary_head = ""
        if summary_path.exists():
            try:
                summary_head = "\n".join(summary_path.read_text(encoding="utf-8").splitlines()[:24])
            except Exception:
                summary_head = ""

        return {
            "segment": cfg.key,
            "label": cfg.label,
            "metrics": metrics,
            "summaryHead": summary_head,
            "paths": {
                "outDir": cfg.out_dir,
                "propertyFile": cfg.property_file,
                "trendRoot": cfg.trend_root,
                "propertyForecastFile": str(self._property_forecast_path(cfg).as_posix()),
            },
            "formula": {
                "rho_0": "clip(0.25 + 0.30*C_i + 0.20*U_i - 0.25*M_i, 0.10, 0.70)",
                "rho_t": "rho_t = rho_0 * 0.90^t",
                "forecast": "F_i,t = [rho_t*L_i + (1-rho_t)*P_i] * I_l,t",
            },
        }

    def context(
        self,
        segment: str,
        city: str,
        locality: str,
        property_id: str | None = None,
        years: int = 5,
    ) -> dict[str, Any]:
        cfg = self._cfg(segment)
        props = self._load_property_master(cfg)
        loc_fc = self._load_small_csv(cfg, "future_forecasts.csv", parse_dates=["date"])
        rho_df = self._load_small_csv(cfg, "rho_details.csv")

        if props.empty:
            raise ValueError(f"Property file not available for segment: {segment}")

        city_key = normalize_city_name(city)
        locality_key = normalize_locality_key(locality)

        scoped = props.copy()
        if city_key:
            city_slice = scoped[scoped["city_key"] == city_key]
            if not city_slice.empty:
                scoped = city_slice

        loc_slice = scoped[scoped["locality_key"] == locality_key]
        if loc_slice.empty:
            all_loc_keys = scoped["locality_key"].dropna().astype(str).unique().tolist()
            hit = get_close_matches(locality_key, all_loc_keys, n=1, cutoff=0.70)
            if hit:
                loc_slice = scoped[scoped["locality_key"] == hit[0]]

        if loc_slice.empty:
            raise ValueError(f"No properties found for locality: {locality}")

        if property_id:
            selected = loc_slice[loc_slice["property_id"].astype(str) == str(property_id)]
            if selected.empty:
                selected = loc_slice.iloc[[0]]
        else:
            selected = loc_slice.iloc[[0]]

        prop_row = selected.iloc[0]
        selected_id = str(prop_row["property_id"])
        selected_locality = str(prop_row.get("locality", locality)).strip()

        prop_fc = self._load_property_forecast_filtered(cfg, property_id=selected_id)
        if prop_fc.empty:
            prop_fc = self._load_property_forecast_filtered(cfg, locality=selected_locality)
            if not prop_fc.empty:
                selected_id = str(prop_fc["property_id"].astype(str).iloc[0])
                prop_fc = prop_fc[prop_fc["property_id"].astype(str) == selected_id].copy()

        if prop_fc.empty:
            raise ValueError(f"No forecast rows found for property/locality in segment: {segment}")

        trend_locality = str(prop_fc.get("trend_locality", pd.Series([""])).iloc[0]).strip()
        trend_is_city_fallback = trend_locality.lower().startswith("city_avg")
        if (not trend_locality) or trend_is_city_fallback:
            resolved = self._resolve_trend_locality(selected_locality, city, loc_fc)
            if resolved:
                trend_locality = resolved
                trend_is_city_fallback = False

        hist_trend = (
            self._load_hist_trend(cfg, trend_locality)
            if trend_locality and not trend_is_city_fallback
            else pd.DataFrame(columns=["date", "price_per_sqft"])
        )

        loc_forecast = (
            loc_fc[loc_fc["locality"].astype(str).str.strip().str.casefold() == trend_locality.casefold()].copy()
            if (not loc_fc.empty and trend_locality and not trend_is_city_fallback and "locality" in loc_fc.columns)
            else pd.DataFrame()
        )

        listing_price = _safe_float(prop_row.get("price_per_sqft"), default=0.0)
        model_price = _safe_float(prop_row.get("pi_price_per_sqft"), default=0.0)
        delta_pct = ((listing_price - model_price) / model_price * 100.0) if model_price > 0 else 0.0

        rho_row = pd.DataFrame()
        if not rho_df.empty:
            if "property_id" in rho_df.columns:
                rho_row = rho_df[rho_df["property_id"].astype(str) == selected_id]
            if rho_row.empty and "locality" in rho_df.columns:
                rho_row = rho_df[
                    rho_df["locality"].astype(str).str.strip().str.casefold() == selected_locality.casefold()
                ]

        as_of = pd.Timestamp.today().normalize()
        yoy_series = self._compute_property_yoy_series(prop_fc, as_of=as_of, years=max(1, min(years, 10)))
        locality_yoy = self._compute_locality_yoy_from_today(loc_fc, trend_locality, as_of=as_of)

        pf_num = prop_fc.copy()
        for col in ["P_i", "L_i", "rho_t", "I_lt", "forecast_price_per_sqft"]:
            if col in pf_num.columns:
                pf_num[col] = pd.to_numeric(pf_num[col], errors="coerce")
        pf_num = pf_num.dropna(subset=["P_i", "L_i", "rho_t", "I_lt", "forecast_price_per_sqft"])

        linear_pred = (pf_num["rho_t"] * pf_num["L_i"] + (1.0 - pf_num["rho_t"]) * pf_num["P_i"]) * pf_num["I_lt"]

        safe_ratio = np.clip(pf_num["L_i"] / np.clip(pf_num["P_i"], 1e-9, None), 1e-9, None)
        geometric_pred = pf_num["P_i"] * pf_num["I_lt"] * np.power(safe_ratio, pf_num["rho_t"])

        actual = pf_num["forecast_price_per_sqft"]
        linear_abs_err = np.abs(linear_pred - actual)
        geometric_abs_err = np.abs(geometric_pred - actual)

        all_loc_fc = self._load_property_forecast_filtered(cfg, locality=selected_locality)
        distribution_rows: list[dict[str, Any]] = []
        if not all_loc_fc.empty:
            agg = (
                all_loc_fc.groupby("date")["forecast_price_per_sqft"]
                .agg(
                    median="median",
                    mean="mean",
                    min="min",
                    max="max",
                    p25=lambda x: x.quantile(0.25),
                    p75=lambda x: x.quantile(0.75),
                )
                .reset_index()
                .sort_values("date")
            )
            agg["date"] = pd.to_datetime(agg["date"], errors="coerce").dt.date.astype(str)
            distribution_rows = agg.to_dict(orient="records")

        property_attrs = {
            "property_id": selected_id,
            "locality": selected_locality,
            "city": str(prop_row.get("city", "")).strip(),
            "bhk": _safe_float(prop_row.get("bhk"), default=np.nan),
            "covered_area_sqft": _safe_float(prop_row.get("covered_area_sqft"), default=np.nan),
            "price_per_sqft": listing_price,
            "pi_price_per_sqft": model_price,
            "circle_rate": _safe_float(prop_row.get("circle_rate"), default=np.nan),
            "is_parking": _safe_float(prop_row.get("is_parking"), default=np.nan),
            "is_corner": _safe_float(prop_row.get("is_corner"), default=np.nan),
            "is_gated": _safe_float(prop_row.get("is_gated"), default=np.nan),
            "road_width_ft": _safe_float(prop_row.get("road_width_ft"), default=np.nan),
        }

        rho_payload: dict[str, Any] = {}
        if not rho_row.empty:
            rr = rho_row.iloc[0]
            c_i = _safe_float(rr.get("comp_support"), default=0.0)
            u_i = _safe_float(rr.get("uniqueness"), default=0.0)
            m_i = _safe_float(rr.get("model_confidence"), default=0.0)
            rho_0_raw = 0.25 + 0.30 * c_i + 0.20 * u_i - 0.25 * m_i
            rho_0_clip = min(0.70, max(0.10, rho_0_raw))
            rho_payload = {
                "comp_support": c_i,
                "uniqueness": u_i,
                "model_confidence": m_i,
                "rho_0_file": _safe_float(rr.get("rho_0"), default=0.0),
                "rho_0_formula_raw": rho_0_raw,
                "rho_0_formula_clipped": rho_0_clip,
                "rho_weights": {
                    "base": 0.25,
                    "a_comp": 0.30,
                    "b_unique": 0.20,
                    "c_model": 0.25,
                    "gamma": 0.90,
                    "min": 0.10,
                    "max": 0.70,
                },
            }

        quarter_table = prop_fc[
            [
                "date",
                "quarter",
                "P_i",
                "L_i",
                "rho_0",
                "rho_t",
                "I_lt",
                "forecast_price_per_sqft",
            ]
        ].copy()
        quarter_table["date"] = pd.to_datetime(quarter_table["date"], errors="coerce").dt.date.astype(str)
        quarter_rows = quarter_table.to_dict(orient="records")

        return {
            "segment": cfg.key,
            "city": city,
            "requestedLocality": locality,
            "selectedPropertyId": selected_id,
            "trendLocality": trend_locality,
            "trendIsCityFallback": trend_is_city_fallback,
            "asOfDate": date.today().isoformat(),
            "kpis": {

                "listingPricePpsf": listing_price,
                "modelPricePpsf": model_price,
                "deltaPct": delta_pct,
                "propertyForecastRows": int(len(prop_fc)),
                "localityForecastRows": int(len(loc_forecast)),
                "localityDistributionProperties": int(all_loc_fc["property_id"].nunique()) if not all_loc_fc.empty else 0,
            },
            "property": property_attrs,
            "rho": rho_payload,
            "mathValidation": {
                "linearFormula": "F_i,t = [rho_t*L_i + (1-rho_t)*P_i] * I_l,t",
                "logFormula": "log(V_i,t)=log(P_i)+log(I_l,t)+rho_t*(log(L_i)-log(P_i))",
                "linearMeanAbsError": float(linear_abs_err.mean()) if len(linear_abs_err) else None,
                "linearMaxAbsError": float(linear_abs_err.max()) if len(linear_abs_err) else None,
                "logMeanAbsError": float(geometric_abs_err.mean()) if len(geometric_abs_err) else None,
                "logMaxAbsError": float(geometric_abs_err.max()) if len(geometric_abs_err) else None,
            },
            "series": {
                "historicalTrend": self._to_date_records(hist_trend, "price_per_sqft"),
                "localityForecast": self._to_date_records(loc_forecast, "pred_price_per_sqft"),
                "propertyForecast": self._to_date_records(prop_fc, "forecast_price_per_sqft"),
                "distribution": distribution_rows,
            },
            "yoy": {
                "property": yoy_series,
                "locality": locality_yoy,
            },
            "quarterTable": quarter_rows,
        }

    # ------------------------------------------------------------------
    # Locality Intelligence
    # ------------------------------------------------------------------

    def locality_intelligence(
        self,
        segment: str,
        city: str,
        locality: str,
    ) -> dict[str, Any]:
        """Return a comprehensive intelligence profile for a single locality."""
        cfg = self._cfg(segment)
        props = self._load_property_master(cfg)
        loc_fc = self._load_small_csv(cfg, "future_forecasts.csv", parse_dates=["date"])

        if props.empty:
            raise ValueError(f"Property file not available for segment: {segment}")

        city_key = normalize_city_name(city)
        locality_key = normalize_locality_key(locality)

        # Scope to city
        scoped = props.copy()
        if city_key:
            city_slice = scoped[scoped["city_key"] == city_key]
            if not city_slice.empty:
                scoped = city_slice

        # Find matching locality
        loc_slice = scoped[scoped["locality_key"] == locality_key]
        if loc_slice.empty:
            all_loc_keys = scoped["locality_key"].dropna().astype(str).unique().tolist()
            hit = get_close_matches(locality_key, all_loc_keys, n=1, cutoff=0.70)
            if hit:
                loc_slice = scoped[scoped["locality_key"] == hit[0]]

        if loc_slice.empty:
            raise ValueError(f"No properties found for locality: {locality}")

        matched_locality = str(loc_slice.iloc[0].get("locality", locality)).strip()
        matched_locality_key = str(loc_slice.iloc[0].get("locality_key", locality_key)).strip()

        # --- Median price per sqft ---
        ppsf_vals = pd.to_numeric(loc_slice["price_per_sqft"], errors="coerce").dropna()
        median_ppsf = float(ppsf_vals.median()) if not ppsf_vals.empty else None

        # --- Listing count (demand / supply proxy) ---
        listing_count = int(len(loc_slice))

        # --- Circle rate ---
        cr_col = loc_slice["circle_rate"] if "circle_rate" in loc_slice.columns else pd.Series(dtype=float)
        cr_vals = pd.to_numeric(cr_col, errors="coerce").dropna()
        median_circle_rate = float(cr_vals.median()) if not cr_vals.empty else None

        # --- Affordability score (0–100) ---
        # 50 when ppsf == circle_rate; 100 when ppsf == 0.5×cr; 0 when ppsf >> cr
        affordability_score: float | None = None
        if median_ppsf and median_ppsf > 0 and median_circle_rate and median_circle_rate > 0:
            raw_ratio = (median_circle_rate / median_ppsf) * 50.0
            affordability_score = float(min(100.0, max(0.0, raw_ratio)))

        # --- Resolve trend locality key ---
        trend_locality = self._resolve_trend_locality(matched_locality, city, loc_fc)

        # --- Historical trend ---
        hist_trend = pd.DataFrame(columns=["date", "price_per_sqft"])
        if trend_locality:
            hist_trend = self._load_hist_trend(cfg, trend_locality)

        # --- Future forecast series ---
        loc_forecast = pd.DataFrame()
        if not loc_fc.empty and trend_locality and "locality" in loc_fc.columns:
            loc_forecast = loc_fc[
                loc_fc["locality"].astype(str).str.strip().str.casefold() == trend_locality.casefold()
            ].copy()

        # --- Forecasted appreciation (YoY from today) ---
        as_of = pd.Timestamp.today().normalize()
        locality_yoy = self._compute_locality_yoy_from_today(loc_fc, trend_locality or "", as_of=as_of)

        # --- Volatility: QoQ stdev of forecast prices ---
        volatility_pct: float | None = None
        if not loc_forecast.empty and "pred_price_per_sqft" in loc_forecast.columns:
            series = loc_forecast.sort_values("date")["pred_price_per_sqft"].dropna()
            if len(series) >= 2:
                pct_changes = series.pct_change().dropna() * 100.0
                if len(pct_changes) > 1:
                    volatility_pct = float(pct_changes.std())

        # --- Combined price trend (historical + forecast) ---
        price_trend: list[dict[str, Any]] = []
        for _, row in hist_trend.iterrows():
            price_trend.append(
                {
                    "date": str(pd.Timestamp(row["date"]).date()),
                    "value": round(float(row["price_per_sqft"]), 2),
                    "type": "historical",
                }
            )
        if not loc_forecast.empty and "pred_price_per_sqft" in loc_forecast.columns:
            agg = (
                loc_forecast.groupby("date", as_index=False)["pred_price_per_sqft"]
                .median()
                .sort_values("date")
            )
            for _, row in agg.iterrows():
                price_trend.append(
                    {
                        "date": str(pd.Timestamp(row["date"]).date()),
                        "value": round(float(row["pred_price_per_sqft"]), 2),
                        "type": "forecast",
                    }
                )

        # --- Rental yield estimate ---
        # Base yields are typical NCR ranges; adjusted slightly by ppsf-to-circle-rate premium
        rental_yield_pct: float | None = None
        if median_ppsf and median_ppsf > 0:
            base_yield = {"builder-floor": 3.2, "apartment": 2.8, "plot": 1.5}.get(cfg.key, 3.0)
            if median_circle_rate and median_circle_rate > 0:
                ratio = median_ppsf / median_circle_rate
                adjustment = -0.5 * min(max(ratio - 1.0, -1.0), 2.0)
                rental_yield_pct = round(base_yield + adjustment, 2)
            else:
                rental_yield_pct = base_yield

        # --- Top competing localities (same city, closest median ppsf) ---
        competing: list[dict[str, Any]] = []
        if median_ppsf is not None and not scoped.empty and "price_per_sqft" in scoped.columns:
            city_ppsf = (
                scoped.groupby("locality_key")["price_per_sqft"]
                .apply(lambda x: pd.to_numeric(x, errors="coerce").median())
                .dropna()
                .reset_index()
            )
            city_ppsf.columns = ["locality_key", "median_ppsf"]
            city_ppsf["abs_diff"] = (city_ppsf["median_ppsf"] - median_ppsf).abs()
            city_ppsf = city_ppsf.sort_values("abs_diff")
            loc_name_map: dict[str, str] = (
                scoped.groupby("locality_key")["locality"].first().to_dict()
            )
            seen_keys: set[str] = {locality_key, matched_locality_key}
            for _, row in city_ppsf.iterrows():
                k = str(row["locality_key"])
                if k in seen_keys:
                    continue
                seen_keys.add(k)
                competing.append(
                    {
                        "locality": loc_name_map.get(k, k),
                        "medianPpsf": round(float(row["median_ppsf"]), 2),
                        "diffPpsf": round(float(row["abs_diff"]), 2),
                    }
                )
                if len(competing) >= 5:
                    break

        return {
            "segment": cfg.key,
            "city": city,
            "locality": matched_locality,
            "medianPpsf": round(median_ppsf, 2) if median_ppsf is not None else None,
            "listingCount": listing_count,
            "medianCircleRate": round(median_circle_rate, 2) if median_circle_rate is not None else None,
            "affordabilityScore": round(affordability_score, 1) if affordability_score is not None else None,
            "rentalYieldPct": rental_yield_pct,
            "volatilityPct": round(volatility_pct, 2) if volatility_pct is not None else None,
            "forecastedAppreciation": locality_yoy,
            "priceTrend": price_trend,
            "topCompetingLocalities": competing,
        }
