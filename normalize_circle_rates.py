"""
normalize_circle_rates.py
=========================
Converts the Uttarakhand circle-rate JSON (which has a deeply nested
structure that load_all_circle_rates() cannot parse) into the canonical
flat-list format it already understands perfectly.

Run once after adding or updating the raw file:

    python normalize_circle_rates.py

Output is written to the same circle_rates folder:
    uttarakhand_circle_normalized.json

The loader picks it up automatically on the next API restart because
it glob-scans every *.json in that folder.  Each record in the output
carries a "city" field, so no changes to _city_from_filename() or
_normalize_city_key() are needed.

─── Source file structure ───────────────────────────────────────────────
{
  "circle_rates_uttarakhand_2025_per_sqft": {

    "Dehradun": {                  ← dict with named sub-sections
      "non_agricultural_land":              [...],   ← plot/land rates
      "residential_apartments_and_commercial_spaces": [...],  ← flat+commercial
      "combined_rates":                     [...]    ← construction (skipped)
    },

    "Almora": [...],               ← flat arrays for other districts
    "Haridwar": [...],
    "Tehri": [...],
    ...
  }
}

─── Rate field mapping ───────────────────────────────────────────────────
Dehradun / residential_apartments_and_commercial_spaces:
  residential_flat_rate_per_sqft        → property_type = "Residential"
  other_commercial_rate_per_sqft        → property_type = "Commercial"

Dehradun / non_agricultural_land (land/plot rate used as Residential
  fallback for localities not already covered by the apartments section):
  rate_per_sqft_area_50_350_sqm_plot    → property_type = "Residential"

Flat-array districts (Almora, Haridwar, Tehri, Pithoragarh, Chamoli):
  residential_flat_rate_per_sqft        → property_type = "Residential"
  commercial_shop_hotel_office_rate_per_sqft
  or other_commercial_rate_per_sqft     → property_type = "Commercial"
"""

import json
import re
from pathlib import Path

CR_FOLDER = Path("real_estate_data/real_estate_data/circle_rates")
RAW_FILE  = CR_FOLDER / "dehradun_circle.json"
OUT_FILE  = CR_FOLDER / "uttarakhand_circle_normalized.json"


# ── locality splitting ─────────────────────────────────────────────────────

def _split_localities(text: str) -> list[str]:
    """
    Split a locality string on  /  but ONLY at the top level (i.e. not inside
    parentheses).  This handles both patterns that appear in the data:

    1. Single locality with parenthetical description – no split:
         "Rajpur Road (Clocktower to RTO)"

    2. Multiple localities separated by  /  some with their own parens:
         "Rishikul Road (From rishikul tirahe to shankar ashram)/
          Bheemgoda marg(from bheemgoda .../Birla Road/Joghamal Road"
       → 4 separate locality records

    3. Plain multi-locality without parens:
         "Jakhandevi/Chausar/Tilakpur/Talla Joshikhola"
       → 4 records
    """
    parts: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in text:
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth = max(depth - 1, 0)
            buf.append(ch)
        elif ch == "/" and depth == 0:
            part = "".join(buf).strip().strip("/,")
            if len(part) >= 4:
                parts.append(part)
            buf = []
        else:
            buf.append(ch)
    last = "".join(buf).strip().strip("/,")
    if len(last) >= 4:
        parts.append(last)
    return parts or [text.strip()]


# ── record emitter ─────────────────────────────────────────────────────────

def _emit(
    records: list,
    city: str,
    locality_text: str,
    res_rate: float | None,
    com_rate: float | None,
) -> None:
    for loc in _split_localities(locality_text):
        if res_rate is not None:
            records.append({
                "city": city,
                "locality": loc,
                "property_type": "Residential",
                "circle_rate_sqft": round(float(res_rate), 2),
            })
        if com_rate is not None:
            records.append({
                "city": city,
                "locality": loc,
                "property_type": "Commercial",
                "circle_rate_sqft": round(float(com_rate), 2),
            })


# ── city-specific processors ───────────────────────────────────────────────

def _process_dehradun(city: str, city_data: dict, records: list) -> None:
    """
    Dehradun has three named sub-sections.  We use two of them:

    1. residential_apartments_and_commercial_spaces
       → emits Residential + Commercial rows for each locality.

    2. non_agricultural_land (plot / land rates)
       → emits a Residential fallback ONLY for localities that are NOT
         already covered by section 1.  This means a locality-rate lookup
         for a plot in Dehradun will find the land circle rate when no
         flat rate exists for that specific road.
    """
    seen: set[str] = set()

    # ── section 1: flats & commercial ─────────────────────────────────────
    for entry in city_data.get("residential_apartments_and_commercial_spaces", []):
        loc_text = entry.get("locality", "").strip()
        if not loc_text:
            continue
        res = entry.get("residential_flat_rate_per_sqft")
        com = entry.get("other_commercial_rate_per_sqft")
        _emit(records, city, loc_text, res, com)
        for loc in _split_localities(loc_text):
            seen.add(loc.lower())

    # ── section 2: non-agricultural land (plot/land rate) ─────────────────
    for entry in city_data.get("non_agricultural_land", []):
        loc_text = entry.get("locality", "").strip()
        if not loc_text:
            continue
        # Use the standard plot-size band (50–350 sqm) as the representative rate
        land_rate = entry.get("rate_per_sqft_area_50_350_sqm_plot")
        if land_rate is None:
            continue
        for loc in _split_localities(loc_text):
            if loc.lower() not in seen:
                records.append({
                    "city": city,
                    "locality": loc,
                    "property_type": "Residential",
                    "circle_rate_sqft": round(float(land_rate), 2),
                })
                seen.add(loc.lower())


def _process_flat_array_city(city: str, entries: list, records: list) -> None:
    """
    Almora / Haridwar / Tehri / Pithoragarh / Chamoli are stored as a flat
    list of zone-dicts.  Each entry covers one or more localities separated
    by  /  and carries residential + commercial rates directly.
    """
    for entry in entries:
        loc_text = entry.get("locality", "").strip()
        if not loc_text:
            continue
        res = entry.get("residential_flat_rate_per_sqft")
        com = (
            entry.get("commercial_shop_hotel_office_rate_per_sqft")
            or entry.get("other_commercial_rate_per_sqft")
        )
        _emit(records, city, loc_text, res, com)


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    with open(RAW_FILE, encoding="utf-8") as f:
        raw = json.load(f)

    # Unwrap the top-level metadata key
    root: dict = raw.get("circle_rates_uttarakhand_2025_per_sqft", raw)

    records: list[dict] = []
    for city, city_data in root.items():
        if isinstance(city_data, dict):
            _process_dehradun(city, city_data, records)
        elif isinstance(city_data, list):
            _process_flat_array_city(city, city_data, records)

    OUT_FILE.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Summary
    from collections import Counter
    by_city = Counter(r["city"] for r in records)
    print(f"\nWrote {len(records)} total records → {OUT_FILE}\n")
    print("Records per city:")
    for c, n in sorted(by_city.items()):
        print(f"  {c:<20} {n:>4} records")


if __name__ == "__main__":
    main()
