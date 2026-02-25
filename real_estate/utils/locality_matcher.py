"""
City-aware locality matcher using JSON colony files + fuzzy matching.

Strategy:
    1. Load per-city locality lists from a JSON file (e.g. ncr_colonies.json).
    2. For each row with a missing locality, restrict search to that row's city.
    3. Try exact / substring match on *description* first, then *address*.
    4. Fall back to fuzzy (token_set_ratio) matching on description, then address.
    5. If both sources yield a hit, pick the one with the higher score.
"""

import json
import re
from typing import Dict, List, Optional, Tuple

from rapidfuzz import fuzz, process


# ── Minimum fuzzy score to accept a match ──────────────────────
_FUZZY_THRESHOLD = 80          # token_sort_ratio  0-100
_EXACT_BONUS     = 200         # boost for exact/substring hits so they always win
_MIN_CHUNK_LEN   = 3           # ignore text chunks shorter than this

# ── Sector regex: matches "Sector 10", "Sector 4A", "Sector B8", "Sector C2A" ──
_SECTOR_RE = re.compile(
    r"\bsector[\s\-]*(?:no\.?\s*)?([A-Za-z]?\d{1,3}[A-Za-z]?(?:[\-/]\d{1,3}[A-Za-z]?)?)",
    re.IGNORECASE,
)


class LocalityMatcher:
    """Match localities from address / description against per-city JSON lists."""

    # Map human-readable city → JSON key
    _CITY_KEY_MAP: Dict[str, str] = {
        "new delhi":          "new_delhi",
        "delhi":              "new_delhi",
        "gurgaon":            "gurgaon",
        "gurugram":           "gurgaon",
        "faridabad":          "faridabad",
        "ghaziabad":          "ghaziabad",
        "noida":              "noida",
        "greater noida":      "greater_noida",
        "meerut":             "meerut",
        "bhiwadi":            "bhiwadi",
        "alwar":              "alwar",
        "baghpat":            "baghpat",
        "bharatpur":          "bharatpur",
        "bulandshahr":        "bulandshahr",
        "dadri":              "dadri",
        "gautam buddha nagar":"gautam_buddha_nagar",
        "hapur":              "hapur",
        "jhajjar":            "jhajjar",
        "karnal":             "karnal",
        "mewat":              "mewat",
        "muzaffarnagar":      "muzaffarnagar",
        "neemrana":           "neemrana",
        "palwal":             "palwal",
        "panipat":            "panipat",
        "rewari":             "rewari",
        "rohtak":             "rohtak",
        "shahjahanpur":       "shahjahanpur",
        "sonipat":            "sonipat",
    }

    # ──────────────────────────────────────────────────────────
    def __init__(self, json_path: str):
        """
        Args:
            json_path: Path to the city → [localities…] JSON file.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            self._raw: Dict[str, List[str]] = json.load(f)

        # Pre-compute lowercase lookup structures per city
        # city_key → { "lowercase locality" : "Original Cased Locality" }
        self._city_localities: Dict[str, Dict[str, str]] = {}
        # city_key → sorted list of original-cased localities (for rapidfuzz)
        self._city_choices: Dict[str, List[str]] = {}

        for city_key, locs in self._raw.items():
            lower_map = {}
            for loc in locs:
                loc = loc.strip()
                if loc:
                    lower_map[loc.lower()] = loc
            self._city_localities[city_key] = lower_map
            self._city_choices[city_key] = list(lower_map.values())

        self.total_localities = sum(len(v) for v in self._city_choices.values())

    # ──────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────
    def extract_locality(
        self,
        city: str,
        description: str,
        address: str,
        current_locality: str,
    ) -> str:
        """
        Main entry point — called once per DataFrame row.

        Args:
            city:             Value from the 'city' column.
            description:      Free-text property description.
            address:          Full address string.
            current_locality: Existing locality value (may be NaN / empty).

        Returns:
            Best matched locality string (with sector appended when found),
            or the original value if nothing found.
        """
        # If locality already present, still try to enrich with sector
        existing = current_locality if isinstance(current_locality, str) else ""
        existing = existing.strip()

        if existing:
            # Already has a value — just try to append sector if not already there
            if "sector" not in existing.lower():
                sector = self._extract_sector(address) or self._extract_sector(description)
                if sector:
                    return f"{existing}, {sector}"
            return existing

        city_key = self._resolve_city_key(city)

        # ── Sector from address (preferred) then description ──
        sector = self._extract_sector(address) or self._extract_sector(description)

        if city_key is None:
            # Unknown city — can't do locality match, but still return sector
            return sector if sector else current_locality

        # --- Score from description ---
        desc_loc, desc_score = self._best_match(city_key, description)

        # --- Score from address ---
        addr_loc, addr_score = self._best_match(city_key, address)

        # Pick the winning locality
        if desc_loc and addr_loc:
            locality = desc_loc if desc_score >= addr_score else addr_loc
        elif desc_loc:
            locality = desc_loc
        elif addr_loc:
            locality = addr_loc
        else:
            locality = None

        # ── Combine locality + sector ─────────────────────────
        if locality and sector:
            # Don't duplicate if sector already inside the locality name
            if "sector" not in locality.lower():
                return f"{locality}, {sector}"
            return locality
        if locality:
            return locality
        if sector:
            return sector          # sector alone is better than nothing

        return current_locality    # nothing found

    # ──────────────────────────────────────────────────────────
    #  Internals
    # ──────────────────────────────────────────────────────────
    @staticmethod
    def _extract_sector(text: str) -> Optional[str]:
        """
        Extract the first 'Sector X' pattern from text.
        Returns a normalised string like 'Sector 10' or None.
        Address is checked before description by the caller.
        """
        if not isinstance(text, str):
            return None
        m = _SECTOR_RE.search(text)
        if m:
            return f"Sector {m.group(1).upper()}"
        return None

    def _resolve_city_key(self, city: str) -> Optional[str]:
        """Map a human-readable city name to the JSON key."""
        if not isinstance(city, str):
            return None
        key = self._CITY_KEY_MAP.get(city.strip().lower())
        if key and key in self._city_localities:
            return key
        # Fallback: try slugifying the city name directly
        slug = city.strip().lower().replace(" ", "_")
        if slug in self._city_localities:
            return slug
        return None

    # ──────────────────────────────────────────────────────────
    def _best_match(
        self, city_key: str, text: str
    ) -> Tuple[Optional[str], float]:
        """
        Return (matched_locality, score) from *text* against the city's
        locality list.  Score > 100 means exact/substring hit.

        Strategy:
            1. Exact substring scan  (fast, O(n) over localities)
            2. Fuzzy token_set_ratio (covers abbreviations / typos)
        """
        if not isinstance(text, str) or not text.strip():
            return (None, 0.0)

        text_lower = text.lower()
        lower_map = self._city_localities[city_key]

        # ── 1. Exact / substring match ─────────────────────────
        best_exact: Optional[str] = None
        best_exact_len = 0
        for loc_lower, loc_original in lower_map.items():
            if loc_lower in text_lower:
                # prefer the *longest* matching locality (more specific)
                if len(loc_lower) > best_exact_len:
                    # optional: verify word-boundary so "Sector 1" doesn't
                    # falsely match inside "Sector 10"
                    pat = re.compile(
                        r"(?<![a-zA-Z0-9])"
                        + re.escape(loc_lower)
                        + r"(?![a-zA-Z0-9])",
                        re.IGNORECASE,
                    )
                    if pat.search(text):
                        best_exact = loc_original
                        best_exact_len = len(loc_lower)

        if best_exact is not None:
            return (best_exact, _EXACT_BONUS + best_exact_len)

        # ── 2. Fuzzy matching on text chunks ──────────────────
        #    Split the text into smaller parts (comma, period, pipe,
        #    newline) so that fuzzy scoring isn't diluted by long text.
        choices = self._city_choices[city_key]
        if not choices:
            return (None, 0.0)

        chunks = re.split(r"[,.\|\n;]+", text)
        best_fuzzy: Optional[str] = None
        best_fuzzy_score: float = 0.0

        for chunk in chunks:
            chunk = chunk.strip()
            if len(chunk) < _MIN_CHUNK_LEN:
                continue
            result = process.extractOne(
                chunk,
                choices,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=_FUZZY_THRESHOLD,
            )
            if result is not None:
                matched_str, score, _idx = result
                if score > best_fuzzy_score:
                    best_fuzzy = matched_str
                    best_fuzzy_score = score

        if best_fuzzy is not None:
            return (best_fuzzy, best_fuzzy_score)

        return (None, 0.0)
