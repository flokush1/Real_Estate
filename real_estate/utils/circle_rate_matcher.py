"""
Smart circle-rate matcher – maps ``(city, locality)`` → ₹ per sq-ft.

Multi-level matching strategy (applied in order):
  1  Exact  (after normalisation)
  2  Alias expansion  (GK I → Greater Kailash 1 …)
  3  Comma-split parts  (parent locality first, then sector)
  4  Strip trailing sector suffix  ("Sushant Lok Phase 1, Sector 27" → "Sushant Lok Phase 1")
  5  Bare sector extraction  ("Noida Extension Sector 1" → "Sector 1")
  6  Fuzzy match  (rapidfuzz ``token_sort_ratio`` ≥ threshold)
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
}

_CITY_FALLBACKS: Dict[str, List[str]] = {
    # First try Greater Noida specific rates, then Noida sectors.
    "greater_noida": ["greater_noida", "noida"],
}

_CITY_ALIASES_SORTED: List[str] = sorted(_CITY_KEY.keys(), key=len, reverse=True)

# Post-normalisation aliases  (both sides are normalised strings)
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
}

_ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5"}

_FUZZY_CUTOFF = 82  # rapidfuzz token_sort_ratio threshold

# ──────────────────────── HELPERS ───────────────────────────


def _normalize(s: str) -> str:
    """Lowercase → collapse whitespace → standardise Sector prefix →
    trailing Roman numerals → Arabic numbers."""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    # "Sec-2", "Sector-7", "SECTOR.6" → "sector 2", "sector 7" …
    s = re.sub(r"\bsec(?:tor)?[.\-\s]*(?=\d)", "sector ", s)
    # Trailing Roman numeral → Arabic   (\b avoids "rohini")
    m = re.search(r"\b(i{1,3}|iv|v)\s*$", s)
    if m and m.group(1) in _ROMAN:
        s = s[: m.start()].rstrip() + " " + _ROMAN[m.group(1)]
    return s.strip()


def _resolve_city(city: str) -> Optional[str]:
    """Resolve noisy city strings to one supported city key.

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

    # Replace punctuation with spaces to handle values like
    # "Ghaziabad, Delhi NCR" / "Gurugram-Delhi NCR".
    cleaned = re.sub(r"[^a-z0-9]+", " ", raw)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if cleaned in _CITY_KEY:
        return _CITY_KEY[cleaned]

    # Prefer the longest alias first (e.g. "greater noida west" before "noida").
    for alias in _CITY_ALIASES_SORTED:
        pattern = rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])"
        if re.search(pattern, cleaned):
            return _CITY_KEY[alias]

    return None


# ──────────────────────── CLASS ─────────────────────────────


class CircleRateMatcher:
    """Load all NCR circle-rate JSONs; expose ``get_rate(city, locality)``."""

    def __init__(self, circle_rates_dir: str):
        self._lookup: Dict[str, Dict[str, float]] = {}
        self._choices: Dict[str, List[str]] = {}
        self._cache: Dict[Tuple[str, str, str], Optional[float]] = {}
        self._load(circle_rates_dir)

    @property
    def total_entries(self) -> int:
        return sum(len(v) for v in self._lookup.values())

    # ─── loaders ──────────────────────────────────────────────

    def _put(self, city: str, raw_key: str, rate: float):
        """Normalise *raw_key* and insert into the city lookup."""
        self._lookup.setdefault(city, {})[_normalize(raw_key)] = rate

    def _load(self, cr_dir: str):
        self._load_delhi(cr_dir)
        self._load_noida(cr_dir)
        self._load_greater_noida(cr_dir)
        self._load_gurgaon(cr_dir)
        self._load_ghaziabad(cr_dir)
        self._load_faridabad(cr_dir)
        self._load_missing_overrides(cr_dir)
        # pre-compute choice lists for fuzzy matching
        for city, lkp in self._lookup.items():
            self._choices[city] = list(lkp.keys())
        total = self.total_entries
        for city, lkp in self._lookup.items():
            logging.info(f"  CircleRate [{city}]: {len(lkp)} normalised entries")
        logging.info(f"  CircleRate total across cities: {total}")

    # .............................................................

    def _load_delhi(self, cr_dir: str):
        path = os.path.join(cr_dir, "Merged_Delhi_Localities_cr.json")
        if not os.path.exists(path):
            logging.warning(f"  Delhi circle-rate file not found: {path}")
            return
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        for entry in data:
            loc = (entry.get("locality") or "").strip()
            rate = entry.get("circle_land_cost_inr_per_sqft")
            if loc and rate is not None:
                self._put("delhi", loc, rate)

    def _load_noida(self, cr_dir: str):
        path = os.path.join(cr_dir, "Noida_Circle_Rate (1).json")
        if not os.path.exists(path):
            logging.warning(f"  Noida circle-rate file not found: {path}")
            return
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        for key, rate in data.items():
            if rate is not None:
                self._put("noida", key, rate)

    def _load_greater_noida(self, cr_dir: str):
        path = os.path.join(cr_dir, "Greater-Noida_CircleRate (1).json")
        if not os.path.exists(path):
            logging.warning(f"  Greater Noida circle-rate file not found: {path}")
            return
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)

        for key, raw_rate in data.items():
            rate: Optional[float] = None

            if isinstance(raw_rate, (int, float)):
                rate = float(raw_rate)
            elif isinstance(raw_rate, list):
                numeric_vals = [
                    float(v) for v in raw_rate if isinstance(v, (int, float))
                ]
                if numeric_vals:
                    # Some entries have multiple slabs; keep the highest rate.
                    rate = max(numeric_vals)

            if rate is not None:
                self._put("greater_noida", key, rate)

    def _load_gurgaon(self, cr_dir: str):
        """
        Load Gurgaon circle rates from:
            Gurgaon Circle rate.json – flat list of records:
                {"locality": str, "property_type": str, "rate_2025_per_sqft": float}

        Lookup structure per normalised key:
            lkp[norm] = {
                "Residential":   <float>,
                "Commercial":    <float>,
                "Agricultural":  <float>,
                "Institutional": <float>,
                "Other":         <float>,
                "_default":      <float>   # Residential if present, else first seen
            }
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

        lkp = self._lookup.setdefault("gurgaon", {})

        # ── helpers ────────────────────────────────────────────────────────────

        def _set(norm_key: str, prop_type: str, rate: float):
            """Insert rate; on collision keep higher value. _default = Residential ?? first seen."""
            entry = lkp.setdefault(norm_key, {})
            if prop_type not in entry or rate > entry[prop_type]:
                entry[prop_type] = rate
            if "_default" not in entry:
                entry["_default"] = rate
            elif prop_type == "Residential":
                entry["_default"] = rate

        def _expand_and_set(raw_key: str, prop_type: str, rate: float):
            """Normalise key, store it, then expand all sector-range patterns."""
            norm = _normalize(raw_key)
            _set(norm, prop_type, rate)

            # ── "99 TO 110" / "104 TO 106"  (range in key) ──────────────────
            for s, e in re.findall(r"(\d+)\s+to\s+(\d+)", raw_key, re.I):
                for n in range(int(s), int(e) + 1):
                    _set(f"sector {n}", prop_type, rate)

            # ── comma-separated  "Sector 33,34,35,36,37,37A" ─────────────────
            m = re.match(r"^sector[s]?\s+([\d,\s]+[a-z\d,\s]*)$", norm)
            if m:
                for p in re.split(r"[\s,]+", m.group(1).strip()):
                    if p:
                        _set(f"sector {p}", prop_type, rate)

            # ── "Sector-16-17" → sector 16, sector 17 ────────────────────────
            m = re.match(r"^sector\s+(\d+)-(\d+)$", norm)
            if m:
                _set(f"sector {m.group(1)}", prop_type, rate)
                _set(f"sector {m.group(2)}", prop_type, rate)

            # ── "Sector-23-23A" → sector 23, sector 23a ──────────────────────
            m = re.match(r"^sector\s+(\d+)-(\d+[a-z]+)$", norm)
            if m:
                _set(f"sector {m.group(1)}", prop_type, rate)
                _set(f"sector {m.group(2)}", prop_type, rate)

            # ── "Sector-17 (P)" → sector 17 ──────────────────────────────────
            m = re.match(r"^sector\s+(\d+[a-z]*)\s*\(", norm)
            if m:
                _set(f"sector {m.group(1)}", prop_type, rate)

            # ── "Sector-10, 10A" / "Sector 9,9A" / "Sector-10,10A of HB" ─────
            m = re.match(
                r"^sector[s]?\s+(\d+[a-z]*)[,\s]+(\d+[a-z]*)(?:\s+of\s+.*)?$", norm
            )
            if m:
                _set(f"sector {m.group(1)}", prop_type, rate)
                _set(f"sector {m.group(2)}", prop_type, rate)

        for rec in records:
            locality = rec.get("locality")
            prop_type = rec.get("property_type", "Residential")
            rate = rec.get("rate_2025_per_sqft")
            if not locality or rate is None:
                continue
            _expand_and_set(locality, prop_type, float(rate))

        logging.info(
            f"  Loaded {len(records)} Gurgaon records from Gurgaon Circle rate.json"
        )

    def _load_ghaziabad(self, cr_dir: str):
        path = os.path.join(cr_dir, "ghaziabad_circle_rate.json")
        if not os.path.exists(path):
            logging.warning(f"  Ghaziabad circle-rate file not found: {path}")
            return
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        for key, rate in data.items():
            if rate is not None:
                self._put("ghaziabad", key, rate)

    def _load_faridabad(self, cr_dir: str):
        """
        Load Faridabad circle rates from faridabad_circle.json.

                Supported file shapes:
                    1) list[dict] records
                                {"locality": str, "property_type": str, "rate_2025_per_sqft": float}
                    2) legacy flat dict (treated as Residential)
                                {"Locality": 2222.22, ...}

        Patterns handled (in order of prevalence):
          A  Plain locality name            "Agwanpur"
          B  Locality + Sector suffix        "Kaushambi, Sector 14"
                → store full key + parent "kaushambi" + "sector 14"
          C  Sec/Sector descriptor keys      "Sec-14 Above 500 Sq.yds."
                → extract all sector numbers after stripping noise words
                → "Neherpar Sec-97, 98"   → sector 97, sector 98
                → "Sec 30 and 31 …"       → sector 30, sector 31
          D  Sector + multi-number list      "Nehar par sec Plot 79,80,81,82,83"
                → expand to sector 79 … sector 83

        Storage format (mirrors Gurgaon multi-type dict so _unwrap works):
            lkp[norm_key] = {"Residential": <float>, "_default": <float>}
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
            # Backward compatibility for old "{locality: rate}" payloads.
            records = [
                {
                    "locality": locality,
                    "property_type": "Residential",
                    "rate_2025_per_sqft": rate,
                }
                for locality, rate in data.items()
            ]
        else:
            logging.warning(
                f"  Faridabad circle-rate file has unexpected format: {path}"
            )
            return

        lkp = self._lookup.setdefault("faridabad", {})

        # ── helpers ────────────────────────────────────────────────────────────

        def _set(norm_key: str, rate: float, prop_type: str = "Residential"):
            """
            Insert/update one (key, prop_type) → rate entry.
            On collision keep the higher rate (conservative approach).
            _default always points to Residential if present, else first seen.
            """
            entry = lkp.setdefault(norm_key, {})
            if prop_type not in entry or rate > entry[prop_type]:
                entry[prop_type] = rate
            # _default = Residential when available, otherwise first value seen
            if "_default" not in entry:
                entry["_default"] = rate
            elif prop_type == "Residential":
                entry["_default"] = rate

        def _set_sector(num: str, rate: float, prop_type: str = "Residential"):
            """Helper: register a bare sector key only when number is plausible."""
            n = int(re.match(r"\d+", num).group())
            if 1 <= n <= 200:
                _set(f"sector {num.lower()}", rate, prop_type)

        def _expand_and_set(
            raw_key: str, rate: float, prop_type: str = "Residential"
        ):
            """
            Normalise raw_key, store it, then fire all sector-expansion rules.
            """
            norm = _normalize(raw_key)
            _set(norm, rate, prop_type)

            # ── Pattern B: "Locality, Sector NN[X]" ──────────────────────────
            # e.g. "Kaushambi, Sector 14" → "kaushambi" + "sector 14"
            # e.g. "Siddhartha Vihar, Sector 10" → "siddhartha vihar" + "sector 10"
            b = re.match(r"^(.+?),\s*sector\s+(\d+[a-z]*)$", norm)
            if b:
                parent = b.group(1).strip()
                sec_num = b.group(2).strip()
                if parent:
                    _set(parent, rate, prop_type)  # store parent locality
                _set_sector(sec_num, rate, prop_type)  # store bare sector
                return  # done – no further expansion needed

            # ── Only continue if there is a sec/sector token in the key ───────
            if not re.search(r"\bsec", raw_key, re.I):
                return

            # ── Pattern D: multi-number list after "sec/sector" ───────────────
            # "Nehar par sec Plot 79,80,81,82,83"
            # "Neher par sec Plot 84,85,86,87,88,89,90"
            list_m = re.search(
                r"\bsec(?:tor)?\b[^,\d]*(\d+(?:\s*,\s*\d+)+)", raw_key, re.I
            )
            if list_m:
                for n in re.findall(r"\d+", list_m.group(1)):
                    _set_sector(n, rate, prop_type)
                return

            # ── Pattern C-i: "Sec 30 and 31 …" / "Neherpar Sec-97, 98" ──────
            # Two numbers separated by "and", comma, or hyphen
            two_m = re.search(
                r"\bsec(?:tor)?[^0-9]*(\d+[a-z]?)(?:\s*(?:and|,|-)\s*)(\d+[a-z]?)",
                raw_key,
                re.I,
            )
            if two_m:
                _set_sector(two_m.group(1), rate, prop_type)
                _set_sector(two_m.group(2), rate, prop_type)
                return

            # ── Pattern C-ii: single sector number ───────────────────────────
            # Strip noise words before extracting the number
            # "Sec-14 Above 500 Sq.yds." → 14
            # "Sec -18A Above 500sq"     → 18a
            # "Surya Nagar - Sec.91"     → 91
            # "sec 19 Above 500sq"       → 19
            clean = re.sub(
                r"(?i)\b(above|upto|below|more|than|sq|yds?|yards?|plot)\b.*$",
                "",
                raw_key,
            ).strip()
            single_m = re.search(r"\bsec(?:tor)?[.\-\s]*(\d+[a-z]?)", clean, re.I)
            if single_m:
                _set_sector(single_m.group(1), rate, prop_type)

        # ── Main loop ─────────────────────────────────────────────────────────
        prop_type_map = {
            "residential": "Residential",
            "commercial": "Commercial",
            "institutional": "Institutional",
            "krishi": "Agricultural",
            "agricultural": "Agricultural",
            "other": "Other",
        }

        for rec in records:
            locality = rec.get("locality")
            raw_rate = rec.get("rate_2025_per_sqft")
            if not locality or raw_rate is None:
                continue

            try:
                rate = float(raw_rate)
            except Exception:
                continue

            prop_raw = str(rec.get("property_type", "Residential") or "Residential")
            prop_type = prop_type_map.get(prop_raw.strip().lower(), "Other")
            _expand_and_set(str(locality), rate, prop_type)

        logging.info(
            f"  Loaded {len(records)} Faridabad records → "
            f"{len(lkp)} normalised entries in lookup"
        )

    def _load_missing_overrides(self, cr_dir: str):
        """
        Load user-updated rates from missing_circle_rates.json.

        Expected shape:
            {
              "City": {"Locality": 12345.0, "Unknown": null},
              "_metadata": ...
            }
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

            for locality, rate in loc_map.items():
                if locality is None or rate is None:
                    continue

                parsed_rate = None
                if isinstance(rate, (int, float)):
                    parsed_rate = float(rate)
                elif isinstance(rate, str):
                    try:
                        parsed_rate = float(rate.strip())
                    except Exception:
                        parsed_rate = None

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
        """Gurgaon entries are dicts; all other cities store plain floats."""
        if isinstance(val, dict):
            return val.get(prop_type) or val.get("_default")
        if isinstance(val, (int, float)):
            return float(val)
        return None

    def _try_in(
        self,
        norm: str,
        lookup: Dict[str, float],
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

        # 3 — comma-split parts (parent locality first)
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

        # 5 — bare sector extraction  ("noida extension sector 1" → "sector 1")
        sec = re.search(r"\bsector\s+(\d+[a-z]*)\b", norm)
        if sec:
            skey = f"sector {sec.group(1)}"
            if skey in lookup:
                return self._unwrap(lookup[skey], prop_type)

        # 5b — extract *non-sector* portion  ("sector 36 sohna" → "sohna")
        without_sec = re.sub(r"\bsector\s+\d+[a-z]*\b", "", norm)
        without_sec = without_sec.strip().strip(",").strip()
        if without_sec and without_sec != norm:
            if without_sec in lookup:
                return self._unwrap(lookup[without_sec], prop_type)
            a = _ALIASES.get(without_sec)
            if a and a in lookup:
                return self._unwrap(lookup[a], prop_type)

        # 6 — fuzzy match (skip for bare "sector NN" to avoid false
        #     positives like "sector 73" ≈ "sector 7")
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

    # .............................................................

    def get_rate(
        self, city: str, locality: str, prop_type: str = "Residential"
    ) -> Optional[float]:
        """Return the circle rate (₹ per sqft) or ``None``.

        Args:
            city:      City name (e.g. "Gurgaon", "Delhi").
            locality:  Locality / sector string.
            prop_type: Property type for Gurgaon multi-rate lookup.
                       One of "Residential", "Commercial", "Agricultural",
                       "Institutional", "Other". Defaults to "Residential".
                       Ignored for all other cities (plain-float lookup).
        """
        if not isinstance(city, str) or not isinstance(locality, str):
            return None
        city_c = city.lower().strip()
        loc_c = locality.strip()
        if not city_c or not loc_c:
            return None

        city_key = _resolve_city(city)
        if city_key is None:
            return None

        if city_key not in self._lookup:
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
