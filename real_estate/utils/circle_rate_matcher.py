"""
Smart circle-rate matcher – maps ``(city, locality)`` → ₹ per sq-ft.

Multi-level matching strategy (applied in order):
  1  Exact  (after normalisation)
  2  Alias expansion
  3  Comma-split parts
  4  Strip trailing sector suffix
  5  Bare sector extraction
  6  Fuzzy match
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

from rapidfuzz import fuzz, process

from real_estate.logging.logger import logging


# ──────────────────────── CONSTANTS ─────────────────────────

_CITY_KEY: Dict[str, str] = {
    "new delhi": "delhi",
    "delhi": "delhi",

    "noida": "noida",

    "greater noida": "greater_noida",
    "greater noida west": "greater_noida",
    "gr noida": "greater_noida",

    "gurgaon": "gurgaon",
    "gurugram": "gurgaon",

    "ghaziabad": "ghaziabad",
    "faridabad": "faridabad",

    "jaipur": "jaipur",
    "aligarh": "aligarh",
    "alwar": "alwar",
    "dehradun": "dehradun",
    "dehra dun": "dehradun",
    "pune": "pune",
    "pimpri chinchwad": "pune",
    "pcmc": "pune",

    # Gautam Buddha Nagar district → use Greater Noida rates (same admin area)
    "gautam buddha nagar": "greater_noida",
    "gb nagar": "greater_noida",

    # Nimka is a village in Faridabad district → use Faridabad rates
    "nimka": "faridabad",
}

_CITY_FALLBACKS: Dict[str, List[str]] = {
    # First try Greater Noida specific rates, then Noida sectors.
    "greater_noida": ["greater_noida", "noida"],
}

_CITY_ALIASES_SORTED: List[str] = sorted(_CITY_KEY.keys(), key=len, reverse=True)

# Post-normalisation aliases.
# Both key and value should be normalised strings.
_ALIASES: Dict[str, str] = {
    # ── Delhi abbreviations ──────────────────────────
    "gk 1": "greater kailash 1",
    "gk 2": "greater kailash 2",
    "gk 3": "greater kailash 3",
    "gk": "greater kailash",
    "cr park": "chittaranjan park",
    "c r park": "chittaranjan park",
    "mg road": "mehrauli gurgaon road",

    # ── Noida Extension → official circle-rate key ───
    "noida extension": "sector noida phase 2",
    "noida extn": "sector noida phase 2",
    "greater noida west": "sector noida phase 2",

    # ── Jaipur common spelling variants ──────────────
    "awadhpuri": "avadhpuri",
    "brahmpuri khurra": "brahampuri khura",
    "chandragupta nagar": "chandragupt nagar",
    "dahar ka balaji": "dehar ke balaji",
    "engineer colony": "engineers colony",
    "fatehtibba": "fateh tiba",
    "guru nanakpura": "guru nanak pura",
    "himmatnagar": "himmat nagar",
    "kanwarnagar": "kanwar nagar",
    "mahavir nagar": "mahaveer nagar",
    "milaap nagar": "milap nagar",
    "sarna duggar": "sarna dungar",
    "takht e shahi road": "takhte shahi road",
    "vivekanand colony": "viveka nand colony",
}

_ROMAN = {
    "i": "1",
    "ii": "2",
    "iii": "3",
    "iv": "4",
    "v": "5",
}

_FUZZY_CUTOFF = 82


# ──────────────────────── HELPERS ───────────────────────────

def _normalize(s: str) -> str:
    """
    Normalize locality/city keys.

    - lowercase
    - collapse whitespace
    - standardise Sec/Sector prefix
    - convert trailing Roman numerals to Arabic numbers
    - remove repeated punctuation-like separators lightly
    """
    if not isinstance(s, str):
        return ""

    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)

    # "Sec-2", "Sector-7", "SECTOR.6" → "sector 2"
    s = re.sub(r"\bsec(?:tor)?[.\-\s]*(?=\d)", "sector ", s)

    # Normalize common separators.
    s = re.sub(r"\s*-\s*", " - ", s)
    s = re.sub(r"\s*,\s*", ", ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # Trailing Roman numeral → Arabic.
    # \b avoids matching words like "rohini".
    m = re.search(r"\b(i{1,3}|iv|v)\s*$", s)
    if m and m.group(1) in _ROMAN:
        s = s[: m.start()].rstrip() + " " + _ROMAN[m.group(1)]

    return s.strip()


def _resolve_city(city: str) -> Optional[str]:
    """
    Resolve noisy city strings to supported city key.

    Examples:
      "Ghaziabad" -> "ghaziabad"
      "Ghaziabad, Delhi NCR" -> "ghaziabad"
      "Gurugram" -> "gurgaon"
    """
    if not isinstance(city, str):
        return None

    raw = city.lower().strip()
    if not raw:
        return None

    raw = re.sub(r"\s+", " ", raw)

    if raw in _CITY_KEY:
        return _CITY_KEY[raw]

    cleaned = re.sub(r"[^a-z0-9]+", " ", raw)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if cleaned in _CITY_KEY:
        return _CITY_KEY[cleaned]

    # Prefer longest alias first.
    for alias in _CITY_ALIASES_SORTED:
        pattern = rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])"
        if re.search(pattern, cleaned):
            return _CITY_KEY[alias]

    return None


def _safe_float(value: object) -> Optional[float]:
    """Parse numeric value safely. Returns None for invalid, null, '-', or non-positive values."""
    if value is None:
        return None

    if isinstance(value, (int, float)):
        rate = float(value)
        return rate if rate > 0 else None

    if isinstance(value, str):
        value = value.strip()
        if value in {"", "-", "None", "none", "null", "NULL"}:
            return None
        try:
            rate = float(value)
            return rate if rate > 0 else None
        except Exception:
            return None

    return None


# ──────────────────────── CLASS ─────────────────────────────

class CircleRateMatcher:
    """Load circle-rate JSONs and expose ``get_rate(city, locality)``."""

    def __init__(self, circle_rates_dir: str):
        self._lookup: Dict[str, Dict[str, object]] = {}
        self._choices: Dict[str, List[str]] = {}
        self._cache: Dict[Tuple[str, str, str], Optional[float]] = {}
        self._load(circle_rates_dir)

    @property
    def total_entries(self) -> int:
        return sum(len(v) for v in self._lookup.values())

    # ─── generic insert helpers ──────────────────────────────

    def _put(self, city: str, raw_key: str, rate: float):
        """Normalise raw_key and insert plain float into city lookup."""
        if not raw_key:
            return
        parsed = _safe_float(rate)
        if parsed is None:
            return

        norm_key = _normalize(str(raw_key))
        if not norm_key:
            return

        self._lookup.setdefault(city, {})[norm_key] = parsed

    def _put_typed(
        self,
        city: str,
        raw_key: str,
        prop_type: str,
        rate: float,
    ):
        """
        Insert property-type aware circle rate.

        Storage:
            lookup[city][norm_key] = {
                "Residential": 123,
                "Commercial": 456,
                "_default": 123
            }
        """
        if not raw_key:
            return

        parsed = _safe_float(rate)
        if parsed is None:
            return

        norm_key = _normalize(str(raw_key))
        if not norm_key:
            return

        prop_type = str(prop_type or "Residential").strip().title()
        if not prop_type:
            prop_type = "Residential"

        lkp = self._lookup.setdefault(city, {})
        entry = lkp.setdefault(norm_key, {})

        if not isinstance(entry, dict):
            # If this key was previously a float, upgrade to typed dict.
            entry = {"_default": float(entry)}
            lkp[norm_key] = entry

        if prop_type not in entry or parsed > float(entry[prop_type]):
            entry[prop_type] = parsed

        if "_default" not in entry:
            entry["_default"] = parsed
        elif prop_type == "Residential":
            entry["_default"] = parsed

    # ─── loaders ──────────────────────────────────────────────

    def _load(self, cr_dir: str):
        self._load_delhi(cr_dir)
        self._load_noida(cr_dir)
        self._load_greater_noida(cr_dir)
        self._load_gurgaon(cr_dir)
        self._load_ghaziabad(cr_dir)
        self._load_faridabad(cr_dir)
        self._load_jaipur(cr_dir)
        self._load_aligarh(cr_dir)
        self._load_alwar(cr_dir)
        self._load_dehradun(cr_dir)
        self._load_pune(cr_dir)
        self._load_missing_overrides(cr_dir)
        self._load_canonical_json_files(cr_dir)

        # Pre-compute choice lists for fuzzy matching.
        for city, lkp in self._lookup.items():
            self._choices[city] = list(lkp.keys())

        total = self.total_entries
        for city, lkp in self._lookup.items():
            logging.info(f"  CircleRate [{city}]: {len(lkp)} normalised entries")
        logging.info(f"  CircleRate total across cities: {total}")

    # Canonical JSON format for new cities:
    # [
    #   {
    #     "city": "CityName",          <- required when city can't be inferred from filename
    #     "locality": "Locality Name", <- required
    #     "property_type": "Residential", <- optional, defaults to "Residential"
    #     "circle_rate_sqft": 5000.0   <- required (₹/sqft)
    #   },
    #   ...
    # ]
    # Files already handled by the specific loaders above are skipped so there
    # is no double-loading. Any new city just needs a JSON file with this schema.
    _KNOWN_CANONICAL_FILES = {
        "merged_delhi_localities_cr.json",
        "noida_circle_rate (1).json",
        "greater-noida_circlerate (1).json",
        "gurgaon circle rate.json",
        "ghaziabad_circle_rate.json",
        "faridabad_circle.json",
        "jaipur_circle_rates_urban (1).json",
        "aligarh_circle.json",
        "alwar_circle.json",
        "missing_circle_rates.json",
        "dehradun_circle.json",   # handled by _load_dehradun
        "pune_circle.json",       # handled by _load_pune
        # Note: "delhi circle rat3e (1).json" intentionally NOT listed here
        # so the generic loader picks it up and merges it into delhi.
    }

    def _load_canonical_json_files(self, cr_dir: str) -> None:
        """
        Generic loader for any JSON file not already handled by a specific loader.

        Supports two list formats:
          1. Each record has a ``"city"`` field → city resolved from the record.
          2. No ``"city"`` field → city resolved from the filename (same logic
             as :func:`load_all_circle_rates` in the API layer).

        Flat-dict ``{locality: rate}`` files without a known city mapping are
        skipped with a warning.
        """
        import glob as _glob

        for fpath in sorted(_glob.glob(os.path.join(cr_dir, "*.json"))):
            fname = os.path.basename(fpath).lower()
            if fname in self._KNOWN_CANONICAL_FILES:
                continue  # already handled by a specific loader

            try:
                with open(fpath, encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception as exc:
                logging.warning(f"  Could not read circle-rate file {fpath}: {exc}")
                continue

            if not isinstance(data, list):
                # Flat-dict format: try to infer city from filename.
                file_city = _resolve_city(fname.replace("_circle", "").replace("_rate", "").replace(".json", ""))
                if file_city is None:
                    logging.warning(
                        f"  Skipping {fname}: flat-dict format requires city inferrable from filename."
                    )
                    continue
                loaded = 0
                for loc_key, raw_rate in data.items():
                    rate = _safe_float(raw_rate)
                    if loc_key and rate is not None:
                        self._put(file_city, str(loc_key), rate)
                        loaded += 1
                logging.info(f"  Canonical loader [{file_city}] from {fname}: {loaded} entries")
                continue

            # List format — read city from each record or from filename.
            file_city = _resolve_city(fname.replace("_circle", "").replace("_rate", "").replace(".json", ""))
            loaded = 0
            for item in data:
                if not isinstance(item, dict):
                    continue
                # Try _resolve_city first (handles known aliases like "Gurugram" → "gurgaon").
                # Fall back to a plain normalised string so any new city name works
                # without needing to be added to _CITY_KEY.
                raw_city = str(item.get("city", "")).strip()
                city_key = file_city or _resolve_city(raw_city) or _normalize(raw_city).lower() or None
                if not city_key:
                    continue
                loc = str(item.get("locality", "")).strip()
                prop_type = str(item.get("property_type") or "Residential").strip().title() or "Residential"
                rate = _safe_float(
                    item.get("circle_rate_sqft")
                    or item.get("circle_land_cost_inr_per_sqft")
                    or item.get("rate_2025_per_sqft")
                    or item.get("rate")
                )
                if loc and rate is not None:
                    self._put_typed(city_key, loc, prop_type, rate)
                    loaded += 1

            if loaded:
                logging.info(f"  Canonical loader from {fname}: {loaded} entries")

    def _load_delhi(self, cr_dir: str):
        path = os.path.join(cr_dir, "Merged_Delhi_Localities_cr.json")
        if not os.path.exists(path):
            logging.warning(f"  Delhi circle-rate file not found: {path}")
            return

        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)

        loaded = 0
        for entry in data:
            if not isinstance(entry, dict):
                continue
            loc = (entry.get("locality") or "").strip()
            rate = _safe_float(entry.get("circle_land_cost_inr_per_sqft"))
            if loc and rate is not None:
                self._put("delhi", loc, rate)
                loaded += 1

        logging.info(f"  Loaded {loaded} Delhi circle-rate entries")

    def _load_noida(self, cr_dir: str):
        path = os.path.join(cr_dir, "Noida_Circle_Rate (1).json")
        if not os.path.exists(path):
            logging.warning(f"  Noida circle-rate file not found: {path}")
            return

        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)

        loaded = 0
        if isinstance(data, dict):
            for key, raw_rate in data.items():
                rate = _safe_float(raw_rate)
                if key and rate is not None:
                    self._put("noida", key, rate)
                    loaded += 1

        logging.info(f"  Loaded {loaded} Noida circle-rate entries")

    def _load_greater_noida(self, cr_dir: str):
        path = os.path.join(cr_dir, "Greater-Noida_CircleRate (1).json")
        if not os.path.exists(path):
            logging.warning(f"  Greater Noida circle-rate file not found: {path}")
            return

        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)

        loaded = 0
        if not isinstance(data, dict):
            logging.warning(f"  Greater Noida circle-rate file has unexpected format: {path}")
            return

        for key, raw_rate in data.items():
            rate: Optional[float] = None

            if isinstance(raw_rate, list):
                numeric_vals = []
                for v in raw_rate:
                    parsed = _safe_float(v)
                    if parsed is not None:
                        numeric_vals.append(parsed)
                if numeric_vals:
                    rate = max(numeric_vals)
            else:
                rate = _safe_float(raw_rate)

            if key and rate is not None:
                self._put("greater_noida", key, rate)
                loaded += 1

        logging.info(f"  Loaded {loaded} Greater Noida circle-rate entries")

    def _load_gurgaon(self, cr_dir: str):
        """
        Load Gurgaon circle rates from:
            Gurgaon Circle rate.json

        Expected:
            [
                {
                    "locality": str,
                    "property_type": str,
                    "rate_2025_per_sqft": float
                }
            ]
        """
        path = os.path.join(cr_dir, "Gurgaon Circle rate.json")
        if not os.path.exists(path):
            logging.warning(f"  Gurgaon circle-rate file not found: {path}")
            return

        with open(path, encoding="utf-8") as fh:
            records = json.load(fh)

        if not isinstance(records, list):
            logging.warning(f"  Gurgaon circle-rate file has unexpected format: {path}")
            return

        def _set(norm_key: str, prop_type: str, rate: float):
            self._put_typed("gurgaon", norm_key, prop_type, rate)

        def _expand_and_set(raw_key: str, prop_type: str, rate: float):
            norm = _normalize(raw_key)
            _set(norm, prop_type, rate)

            # "99 TO 110"
            for s, e in re.findall(r"(\d+)\s+to\s+(\d+)", raw_key, re.I):
                for n in range(int(s), int(e) + 1):
                    _set(f"sector {n}", prop_type, rate)

            # "Sector 33,34,35,36,37,37A"
            m = re.match(r"^sector[s]?\s+([\d,\s]+[a-z\d,\s]*)$", norm)
            if m:
                for p in re.split(r"[\s,]+", m.group(1).strip()):
                    if p:
                        _set(f"sector {p}", prop_type, rate)

            # "Sector-16-17"
            m = re.match(r"^sector\s+(\d+)-(\d+)$", norm)
            if m:
                _set(f"sector {m.group(1)}", prop_type, rate)
                _set(f"sector {m.group(2)}", prop_type, rate)

            # "Sector-23-23A"
            m = re.match(r"^sector\s+(\d+)-(\d+[a-z]+)$", norm)
            if m:
                _set(f"sector {m.group(1)}", prop_type, rate)
                _set(f"sector {m.group(2)}", prop_type, rate)

            # "Sector-17 (P)"
            m = re.match(r"^sector\s+(\d+[a-z]*)\s*\(", norm)
            if m:
                _set(f"sector {m.group(1)}", prop_type, rate)

            # "Sector-10, 10A"
            m = re.match(
                r"^sector[s]?\s+(\d+[a-z]*)[,\s]+(\d+[a-z]*)(?:\s+of\s+.*)?$",
                norm,
            )
            if m:
                _set(f"sector {m.group(1)}", prop_type, rate)
                _set(f"sector {m.group(2)}", prop_type, rate)

        loaded = 0
        for rec in records:
            if not isinstance(rec, dict):
                continue

            locality = rec.get("locality")
            prop_type = rec.get("property_type", "Residential")
            rate = _safe_float(rec.get("rate_2025_per_sqft"))

            if not locality or rate is None:
                continue

            _expand_and_set(str(locality), str(prop_type), rate)
            loaded += 1

        logging.info(f"  Loaded {loaded} Gurgaon records from Gurgaon Circle rate.json")

    def _load_ghaziabad(self, cr_dir: str):
        path = os.path.join(cr_dir, "ghaziabad_circle_rate.json")
        if not os.path.exists(path):
            logging.warning(f"  Ghaziabad circle-rate file not found: {path}")
            return

        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)

        loaded = 0
        if isinstance(data, dict):
            for key, raw_rate in data.items():
                rate = _safe_float(raw_rate)
                if key and rate is not None:
                    self._put("ghaziabad", key, rate)
                    loaded += 1

        logging.info(f"  Loaded {loaded} Ghaziabad circle-rate entries")

    def _load_faridabad(self, cr_dir: str):
        """
        Load Faridabad circle rates from faridabad_circle.json.

        Supports:
          1. list[dict]:
                {"locality": str, "property_type": str, "rate_2025_per_sqft": float}
          2. legacy dict:
                {"Locality": 2222.22}
        """
        path = os.path.join(cr_dir, "faridabad_circle.json")
        if not os.path.exists(path):
            logging.warning(f"  Faridabad circle-rate file not found: {path}")
            return

        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)

        records: List[Dict[str, object]] = []

        if isinstance(data, list):
            records = [r for r in data if isinstance(r, dict)]
        elif isinstance(data, dict):
            records = [
                {
                    "locality": locality,
                    "property_type": "Residential",
                    "rate_2025_per_sqft": rate,
                }
                for locality, rate in data.items()
            ]
        else:
            logging.warning(f"  Faridabad circle-rate file has unexpected format: {path}")
            return

        prop_type_map = {
            "residential": "Residential",
            "commercial": "Commercial",
            "institutional": "Institutional",
            "krishi": "Agricultural",
            "agricultural": "Agricultural",
            "other": "Other",
        }

        def _set(norm_key: str, rate: float, prop_type: str = "Residential"):
            self._put_typed("faridabad", norm_key, prop_type, rate)

        def _set_sector(num: str, rate: float, prop_type: str = "Residential"):
            m = re.match(r"\d+", num)
            if not m:
                return
            n = int(m.group())
            if 1 <= n <= 200:
                _set(f"sector {num.lower()}", rate, prop_type)

        def _expand_and_set(raw_key: str, rate: float, prop_type: str = "Residential"):
            norm = _normalize(raw_key)
            _set(norm, rate, prop_type)

            # "Locality, Sector NN"
            b = re.match(r"^(.+?),\s*sector\s+(\d+[a-z]*)$", norm)
            if b:
                parent = b.group(1).strip()
                sec_num = b.group(2).strip()
                if parent:
                    _set(parent, rate, prop_type)
                _set_sector(sec_num, rate, prop_type)
                return

            if not re.search(r"\bsec", raw_key, re.I):
                return

            # Multi-number list after sec/sector.
            list_m = re.search(
                r"\bsec(?:tor)?\b[^,\d]*(\d+(?:\s*,\s*\d+)+)",
                raw_key,
                re.I,
            )
            if list_m:
                for n in re.findall(r"\d+", list_m.group(1)):
                    _set_sector(n, rate, prop_type)
                return

            # Two sectors separated by and/comma/hyphen.
            two_m = re.search(
                r"\bsec(?:tor)?[^0-9]*(\d+[a-z]?)(?:\s*(?:and|,|-)\s*)(\d+[a-z]?)",
                raw_key,
                re.I,
            )
            if two_m:
                _set_sector(two_m.group(1), rate, prop_type)
                _set_sector(two_m.group(2), rate, prop_type)
                return

            # Single sector.
            clean = re.sub(
                r"(?i)\b(above|upto|below|more|than|sq|yds?|yards?|plot)\b.*$",
                "",
                raw_key,
            ).strip()

            single_m = re.search(r"\bsec(?:tor)?[.\-\s]*(\d+[a-z]?)", clean, re.I)
            if single_m:
                _set_sector(single_m.group(1), rate, prop_type)

        loaded = 0
        for rec in records:
            locality = rec.get("locality")
            raw_rate = rec.get("rate_2025_per_sqft")

            if not locality:
                continue

            rate = _safe_float(raw_rate)
            if rate is None:
                continue

            prop_raw = str(rec.get("property_type", "Residential") or "Residential")
            prop_type = prop_type_map.get(prop_raw.strip().lower(), "Other")

            _expand_and_set(str(locality), rate, prop_type)
            loaded += 1

        logging.info(
            f"  Loaded {loaded} Faridabad records → "
            f"{len(self._lookup.get('faridabad', {}))} normalised entries in lookup"
        )

    def _load_jaipur(self, cr_dir: str):
        """
        Load Jaipur circle rates from jaipur_circle_rate.json.

        Strategy:
          - For each Zone, choose highest residential Exterior rate.
          - If no residential rate exists, fallback to highest commercial Exterior rate.
          - Store Zone name.
          - Also store useful Colony names from Rows.
          - Skip agricultural-only/generic slab labels.
        """
        path = os.path.join(cr_dir, "jaipur_circle_rates_urban (1).json")
        if not os.path.exists(path):
            logging.warning(f"  Jaipur circle-rate file not found: {path}")
            return

        try:
            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh)
        except Exception as e:
            logging.warning(f"  Could not parse Jaipur circle-rate JSON: {e}")
            return

        records = doc if isinstance(doc, list) else doc.get("data", [])
        if not isinstance(records, list):
            logging.warning("  Jaipur circle-rate JSON has unexpected top-level format")
            return

        generic_colonies = {
            "residential",
            "commercial",
            "industrial",
            "agriculture",
            "agricultural",
            "irrigated",
            "non-irrigated",
            "non irrigated",
        }

        bad_tokens = [
            "irrigated",
            "non-irrigated",
            "non irrigated",
            "nh/sh/mh",
            "other 0 to 100 meter",
            "above 101",
            "above 201",
            "far away",
            "near from other road",
            "near from other roads",
        ]

        loaded_zones = 0
        loaded_colonies = 0

        for sro in records:
            if not isinstance(sro, dict):
                continue

            zones = sro.get("Zones") or []
            if not isinstance(zones, list):
                continue

            for zone_block in zones:
                if not isinstance(zone_block, dict):
                    continue

                zone_name = (zone_block.get("Zone") or "").strip()
                if not zone_name:
                    continue

                rows = zone_block.get("Rows") or []
                if not isinstance(rows, list):
                    continue

                best_res: Optional[float] = None
                best_com: Optional[float] = None

                for row in rows:
                    if not isinstance(row, dict):
                        continue

                    land_type = str(row.get("Type Of Land") or "").strip().lower()
                    rate = _safe_float(row.get("Exterior"))

                    if rate is None:
                        continue

                    if land_type == "residential":
                        if best_res is None or rate > best_res:
                            best_res = rate
                    elif land_type == "commercial":
                        if best_com is None or rate > best_com:
                            best_com = rate

                chosen = best_res if best_res is not None else best_com
                if chosen is None:
                    # Skip agriculture-only zones.
                    continue

                self._put("jaipur", zone_name, chosen)
                loaded_zones += 1

                # Store useful colony names as aliases to zone's chosen rate.
                for row in rows:
                    if not isinstance(row, dict):
                        continue

                    colony = str(row.get("Colony") or "").strip()
                    if not colony:
                        continue

                    colony_norm = _normalize(colony)
                    if not colony_norm:
                        continue

                    if colony_norm in generic_colonies:
                        continue

                    if any(token in colony_norm for token in bad_tokens):
                        continue

                    self._put("jaipur", colony, chosen)
                    loaded_colonies += 1

        logging.info(
            f"  Loaded Jaipur records → "
            f"{loaded_zones} zone entries, {loaded_colonies} useful colony aliases"
        )

    def _load_dehradun(self, cr_dir: str) -> None:
        """
        Load Dehradun circle rates from dehradun_circle.json.

        The file uses a nested road/zone format:
            {
                "circle_rates_uttarakhand_2025_per_sqft": {
                    "Dehradun": {
                        "non_agricultural_land": [
                            {
                                "zone": "A",
                                "locality": "Rajpur Road (Clocktower to RTO)",
                                "rate_per_sqft_area_upto_50_sqm_plot": 5760.03,
                                "rate_per_sqft_area_50_350_sqm_plot": 4645.15
                            },
                            ...
                        ]
                    }
                }
            }

        Entries are stored keyed by the full locality description AND by the
        abbreviated road name (text before the first parenthesis) so that
        locality names containing e.g. "Rajpur Road" can match via fuzzy.
        """
        path = os.path.join(cr_dir, "dehradun_circle.json")
        if not os.path.exists(path):
            return

        try:
            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh)
        except Exception as exc:
            logging.warning(f"  Could not parse Dehradun circle-rate JSON: {exc}")
            return

        outer = doc.get("circle_rates_uttarakhand_2025_per_sqft", {})
        deh_data = outer.get("Dehradun", {})
        entries = deh_data.get("non_agricultural_land", [])

        loaded = 0
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            locality = str(entry.get("locality", "")).strip()
            if not locality:
                continue
            # Use the smaller 50-350 sqm rate as the conservative residential rate
            rate = _safe_float(
                entry.get("rate_per_sqft_area_50_350_sqm_plot")
                or entry.get("rate_per_sqft_area_upto_50_sqm_plot")
            )
            if rate is None:
                continue

            self._put("dehradun", locality, rate)
            loaded += 1

            # Also index by the root road name before the parenthesis.
            # "Rajpur Road (Clocktower to RTO)" → "Rajpur Road"
            paren_idx = locality.find("(")
            if paren_idx > 0:
                root_name = locality[:paren_idx].strip()
                if root_name and root_name.lower() != locality.lower():
                    self._put("dehradun", root_name, rate)

        logging.info(f"  Loaded {loaded} Dehradun circle-rate entries from dehradun_circle.json")

    def _load_pune(self, cr_dir: str) -> None:
        """
        Load Pune circle rates from pune_circle.json.

        File format (flat dict, 844 localities):
            {
                "Kharadi": {
                    "Residential_Flats_Apts":  [5000, 8000],
                    "Commercial_Offices":      [6000, 9500],
                    "Commercial_Shops":        [7500, 12000],
                    "Residential_Land_Plots":  [2000, 4000]
                },
                ...
            }

        Loading strategy:
          - Residential → midpoint of Residential_Flats_Apts range
          - Commercial   → max midpoint of Commercial_Offices and Commercial_Shops
          - _default     → Residential rate (set by _put_typed)
        """
        path = os.path.join(cr_dir, "pune_circle.json")
        if not os.path.exists(path):
            logging.warning(f"  Pune circle-rate file not found: {path}")
            return

        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            logging.warning(f"  Could not parse Pune circle-rate JSON: {exc}")
            return

        if not isinstance(data, dict):
            logging.warning(f"  Pune circle-rate file has unexpected format: {path}")
            return

        def _midpoint(val: object) -> Optional[float]:
            """Return midpoint of [min, max] list, or safe-float of scalar."""
            if isinstance(val, list) and len(val) == 2:
                lo = _safe_float(val[0])
                hi = _safe_float(val[1])
                if lo is not None and hi is not None:
                    return (lo + hi) / 2.0
                return lo or hi
            return _safe_float(val)

        loaded = 0
        for locality, sub in data.items():
            if not locality or not isinstance(sub, dict):
                continue

            res_rate = _midpoint(sub.get("Residential_Flats_Apts"))
            land_rate = _midpoint(sub.get("Residential_Land_Plots"))
            com_office = _midpoint(sub.get("Commercial_Offices"))
            com_shop = _midpoint(sub.get("Commercial_Shops"))

            # Residential: prefer flat/apt rate, fall back to land plot rate
            res = res_rate if res_rate is not None else land_rate
            # Commercial: take the higher of the two commercial categories
            com_vals = [v for v in (com_office, com_shop) if v is not None]
            com = max(com_vals) if com_vals else None

            if res is not None:
                self._put_typed("pune", locality, "Residential", res)
                loaded += 1
            if land_rate is not None:
                self._put_typed("pune", locality, "Residential_Land", land_rate)
            if com is not None:
                self._put_typed("pune", locality, "Commercial", com)

        logging.info(f"  Loaded {loaded} Pune circle-rate entries from pune_circle.json")

    def _load_aligarh(self, cr_dir: str):
        """
        Load Aligarh circle rates from aligarh_circle.json.

        Expected format:
            {
                "Aligarh": {
                    "Hamidpur": 1393.55,
                    "Tappal": 3251.58,
                    ...
                }
            }

        Also supports flat:
            {"Hamidpur": 1393.55, ...}
        """
        path = os.path.join(cr_dir, "aligarh_circle.json")
        if not os.path.exists(path):
            logging.warning(f"  Aligarh circle-rate file not found: {path}")
            return

        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:
            logging.warning(f"  Could not parse Aligarh circle-rate JSON: {e}")
            return

        if not isinstance(data, dict):
            logging.warning(f"  Aligarh circle-rate file has unexpected format: {path}")
            return

        loc_map = data.get("Aligarh", data)
        if not isinstance(loc_map, dict):
            logging.warning(f"  Aligarh circle-rate file has unexpected nested format: {path}")
            return

        loaded = 0
        for locality, raw_rate in loc_map.items():
            rate = _safe_float(raw_rate)
            if not locality or rate is None:
                continue
            self._put("aligarh", str(locality), rate)
            loaded += 1

        logging.info(f"  Loaded {loaded} Aligarh circle-rate entries")

    def _load_alwar(self, cr_dir: str):
        """
        Load Alwar circle rates from alwar_circle.json.

        Expected format:
            [
                {
                    "locality": "...",
                    "property_type": "Industrial",
                    "circle_rate_sqft": 418.06,
                    "zone": "ALWAR-II"
                }
            ]
        """
        path = os.path.join(cr_dir, "alwar_circle.json")
        if not os.path.exists(path):
            logging.warning(f"  Alwar circle-rate file not found: {path}")
            return

        try:
            with open(path, encoding="utf-8") as fh:
                records = json.load(fh)
        except Exception as e:
            logging.warning(f"  Could not parse Alwar circle-rate JSON: {e}")
            return

        if not isinstance(records, list):
            logging.warning(f"  Alwar circle-rate file has unexpected format: {path}")
            return

        prop_type_map = {
            "residential": "Residential",
            "commercial": "Commercial",
            "industrial": "Industrial",
            "institutional": "Institutional",
            "agricultural": "Agricultural",
            "krishi": "Agricultural",
            "other": "Other",
        }

        loaded = 0
        for rec in records:
            if not isinstance(rec, dict):
                continue

            locality = rec.get("locality")
            raw_rate = rec.get("circle_rate_sqft")
            raw_prop_type = rec.get("property_type", "Residential")

            if not locality:
                continue

            rate = _safe_float(raw_rate)
            if rate is None:
                continue

            prop_key = str(raw_prop_type or "Residential").strip().lower()
            prop_type = prop_type_map.get(prop_key, str(raw_prop_type).strip().title())

            self._put_typed("alwar", str(locality), prop_type, rate)

            # Also store zone as backup key if present.
            zone = rec.get("zone")
            if zone:
                self._put_typed("alwar", str(zone), prop_type, rate)

            loaded += 1

        logging.info(f"  Loaded {loaded} Alwar circle-rate entries")

    def _load_missing_overrides(self, cr_dir: str):
        """
        Load manually updated rates from missing_circle_rates.json.

        Expected shape:
            {
              "City": {"Locality": 12345.0, "Unknown": null},
              "_metadata": ...
            }

        Null values are ignored.
        """
        path = os.path.join(cr_dir, "missing_circle_rates.json")
        if not os.path.exists(path):
            return

        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:
            logging.warning(f"  Could not parse missing overrides JSON: {path} ({e})")
            return

        if not isinstance(data, dict):
            logging.warning(f"  Invalid missing overrides JSON format: {path}")
            return

        loaded = 0

        for city, loc_map in data.items():
            if str(city).startswith("_") or not isinstance(loc_map, dict):
                continue

            city_norm = str(city).strip().lower()
            if not city_norm:
                continue

            city_key = _CITY_KEY.get(city_norm, city_norm)

            for locality, raw_rate in loc_map.items():
                if locality is None:
                    continue

                parsed_rate = _safe_float(raw_rate)
                locality_s = str(locality).strip()

                if parsed_rate is None or not locality_s:
                    continue

                self._put(city_key, locality_s, parsed_rate)
                loaded += 1

        if loaded > 0:
            logging.info(f"  Loaded {loaded} manual circle-rate overrides from {path}")

    # ─── matching ─────────────────────────────────────────────

    @staticmethod
    def _unwrap(val: object, prop_type: str) -> Optional[float]:
        """Typed entries are dicts; plain entries are floats."""
        if isinstance(val, dict):
            return (
                val.get(prop_type)
                or val.get(str(prop_type).title())
                or val.get("Residential")
                or val.get("_default")
            )
        if isinstance(val, (int, float)):
            return float(val)
        return None

    def _try_in(
        self,
        norm: str,
        lookup: Dict[str, object],
        choices: List[str],
        prop_type: str = "Residential",
    ) -> Optional[float]:
        """Run the full multi-level pipeline against one city dict."""

        # 1 — exact
        if norm in lookup:
            return self._unwrap(lookup[norm], prop_type)

        # 2 — alias expansion → exact
        aliased = _ALIASES.get(norm)
        if aliased and aliased in lookup:
            return self._unwrap(lookup[aliased], prop_type)

        # 3 — comma-split parts
        parts = [p.strip() for p in norm.split(",") if p.strip()]
        if len(parts) > 1:
            for part in parts:
                if part in lookup:
                    return self._unwrap(lookup[part], prop_type)

                a = _ALIASES.get(part)
                if a and a in lookup:
                    return self._unwrap(lookup[a], prop_type)

        # 4 — strip trailing ", sector NN" or " sector NN"
        stripped = re.sub(r",?\s*sector\s+\d+[a-z]*\s*$", "", norm).strip()
        if stripped and stripped != norm:
            if stripped in lookup:
                return self._unwrap(lookup[stripped], prop_type)

            a = _ALIASES.get(stripped)
            if a and a in lookup:
                return self._unwrap(lookup[a], prop_type)

        # 5 — bare sector extraction
        sec = re.search(r"\bsector\s+(\d+[a-z]*)\b", norm)
        if sec:
            skey = f"sector {sec.group(1)}"
            if skey in lookup:
                return self._unwrap(lookup[skey], prop_type)

        # 5b — remove sector and try remaining non-sector portion.
        without_sec = re.sub(r"\bsector\s+\d+[a-z]*\b", "", norm)
        without_sec = without_sec.strip().strip(",").strip()
        if without_sec and without_sec != norm:
            if without_sec in lookup:
                return self._unwrap(lookup[without_sec], prop_type)

            a = _ALIASES.get(without_sec)
            if a and a in lookup:
                return self._unwrap(lookup[a], prop_type)

        # 6 — fuzzy match.
        # Skip bare sector to avoid false positives like sector 73 ≈ sector 7.
        is_bare_sector = bool(re.fullmatch(r"sector\s+\d+[a-z]*", norm))
        if not is_bare_sector and choices:
            hit = process.extractOne(
                norm,
                choices,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=_FUZZY_CUTOFF,
            )
            if hit:
                return self._unwrap(lookup[hit[0]], prop_type)

        return None

    def get_rate(
        self,
        city: str,
        locality: str,
        prop_type: str = "Residential",
    ) -> Optional[float]:
        """
        Return the circle rate in ₹ per sqft, or None.

        Args:
            city: City name, e.g. "Gurgaon", "Delhi", "Jaipur".
            locality: Locality / sector string.
            prop_type: Property type for typed city lookups.
        """
        if not isinstance(city, str) or not isinstance(locality, str):
            return None

        city_c = city.lower().strip()
        loc_c = locality.strip()

        if not city_c or not loc_c:
            return None

        city_key = _resolve_city(city)
        if city_key is None:
            # For cities not in _CITY_KEY (e.g. newly added cities like Pune),
            # fall back to the plain normalised city string used during loading.
            city_key = _normalize(city_c) or None

        if not city_key or city_key not in self._lookup:
            return None

        norm = _normalize(loc_c)
        cache_key = (city_key, norm, prop_type)

        if cache_key in self._cache:
            return self._cache[cache_key]

        search_order = _CITY_FALLBACKS.get(city_key, [city_key])

        result: Optional[float] = None
        for search_city in search_order:
            if search_city not in self._lookup:
                continue

            result = self._try_in(
                norm,
                self._lookup[search_city],
                self._choices.get(search_city, []),
                prop_type,
            )

            if result is not None:
                break

        self._cache[cache_key] = result
        return result