"""
Utility helpers to extract structured information from the free-text
`description` and `amenities` columns.
"""

import re
import numpy as np
import pandas as pd


# ────────────────────────────────────────────────────────────────
#  NUMERIC EXTRACTORS
# ────────────────────────────────────────────────────────────────

def extract_bhk(text: str) -> float | None:
    """Extract BHK count from text like '3 BHK' or 'Bedrooms 3'."""
    if not isinstance(text, str):
        return None
    m = re.search(r'(\d+)\s*bhk', text, re.I)
    if m:
        val = float(m.group(1))
        return val if val <= 12 else None
    m = re.search(r'bedrooms?\s*(\d+)', text, re.I)
    if m:
        val = float(m.group(1))
        return val if val <= 12 else None
    return None


def extract_bathrooms(text: str) -> float | None:
    """Extract bathroom count from text like 'Bathrooms 3' or '3 bathrooms'."""
    if not isinstance(text, str):
        return None
    m = re.search(r'bathrooms?\s*(\d+)', text, re.I)
    if m:
        val = float(m.group(1))
        return val if val <= 12 else None
    m = re.search(r'(\d+)\s*bathrooms?', text, re.I)
    if m:
        val = float(m.group(1))
        return val if val <= 12 else None
    return None


def extract_balconies(text: str) -> float | None:
    """Extract balcony count from text like '4 balconies' or 'balconies 4'."""
    if not isinstance(text, str):
        return None
    m = re.search(r'(\d+)\s*balcon', text, re.I)
    if m:
        val = float(m.group(1))
        return val if val <= 12 else None
    m = re.search(r'balcon\w*\s*(\d+)', text, re.I)
    if m:
        val = float(m.group(1))
        return val if val <= 12 else None
    return None


def extract_price(text: str) -> float | None:
    """
    Extract price from text like '₹3.57 Cr', '36 Lac', 'Sale Price 3600000'.
    Returns numeric value (e.g. 3600000.0).
    """
    if not isinstance(text, str):
        return None
    # Direct "Sale Price 1234567"
    m = re.search(r'sale\s*price\s*(\d[\d,]*)', text, re.I)
    if m:
        val = float(m.group(1).replace(',', ''))
        if val > 0:
            return val
    # "₹3.57 Cr" / "3.57 Crore"
    m = re.search(r'(\d+\.?\d*)\s*(?:cr|crore)', text, re.I)
    if m:
        return float(m.group(1)) * 1e7
    # "36 Lac" / "41 Lakh"
    m = re.search(r'(\d+\.?\d*)\s*(?:lac|lakh)', text, re.I)
    if m:
        return float(m.group(1)) * 1e5
    return None


import re

# ────────────────────────────────────────────────────────────────
#  AREA CONVERSION FACTORS  →  everything to Sq-ft
# ────────────────────────────────────────────────────────────────

_SQFT_RE = r"(?:sq(?:uare)?[\s.\-_]{0,4}(?:ft|feet|foot)\.?|\bsft\b)"
_SQYRD_RE = r"(?:sq(?:uare)?[\s.\-_]{0,4}(?:y(?:a?r?ds?|d)\.?s?\.?)|\bsqyrd\.?)"
_SQM_RE = r"(?:sq(?:uare)?[\s.\-_]{0,4}(?:m(?:e?t(?:er|re)?r?s?)?\.?)|\bsqm(?:tr?)?\.?)"

_AREA_UNIT_PATTERNS = [
    (_SQFT_RE, 1.0),
    (_SQYRD_RE, 9.0),
    (_SQM_RE, 10.7639),
    (r"acres?", 43560.0),
    (r"bigha", 27000.0),
    (r"hectares?", 107639.0),
    (r"marla", 272.25),
    (r"grounds?", 2400.0),
    (r"roods?", 10890.0),
    (r"biswa\d?", 1350.0),
]


def extract_area_sqft(text: str) -> float | None:
    """
    Extract area from text in ANY unit (sqft, sq-yrd, sq-m, acre, bigha,
    hectare, marla, ground, rood, biswa) and convert to sqft.
    """
    if not isinstance(text, str):
        return None
    for unit_pattern, factor in _AREA_UNIT_PATTERNS:
        # Updated regex: \d[\d,]* allows for commas within the number
        m = re.search(r"(\d[\d,]*\.?\d*)\s*" + unit_pattern, text, re.I)
        if m:
            # Strip commas out before converting to float
            clean_number = m.group(1).replace(",", "")
            sqft = float(clean_number) * factor
            return sqft if sqft >= 225 else None
    return None


def extract_area_value_and_unit(text: str) -> tuple[float | None, str | None]:
    """
    Extract area value and unit from text.
    Returns (value, unit) as a tuple.
    Unit is normalized (e.g., 'Sq-ft', 'Sq-yrd', 'Sq-m', 'Acre', etc.)
    """
    if not isinstance(text, str):
        return (None, None)

    unit_map = {
        _SQFT_RE: "Sq-ft",
        _SQYRD_RE: "Sq-yrd",
        _SQM_RE: "Sq-m",
        r"acres?": "Acre",
        r"bigha": "Bigha",
        r"hectares?": "Hectare",
        r"marla": "Marla",
        r"grounds?": "Ground",
        r"roods?": "Rood",
        r"biswa\d?": "Biswa",
    }

    for unit_pattern, unit_name in unit_map.items():
        # Updated regex here as well
        m = re.search(r"(\d[\d,]*\.?\d*)\s*" + unit_pattern, text, re.I)
        if m:
            clean_number = m.group(1).replace(",", "")
            return (float(clean_number), unit_name)
    return (None, None)


# ────────────────────────────────────────────────────────────────
#  AGE EXTRACTOR
# ────────────────────────────────────────────────────────────────

def extract_age_of_property(text: str) -> str | None:
    """Extract age category from description text."""
    if not isinstance(text, str):
        return None
    text_lower = text.lower()
    if 'new construction' in text_lower or 'new property' in text_lower:
        return 'New Construction'
    m = re.search(r'(\d+)\s*(?:to\s*)?(\d+)?\s*years?\s*old', text, re.I)
    if m:
        age = int(m.group(1))
        if age < 1:
            return 'New Construction'
        elif age < 5:
            return 'Less than 5 years'
        elif age <= 10:
            return '5 to 10 years'
        elif age <= 20:
            return '10 to 20 years'
        else:
            return 'Above 20 years'
    # patterns like "5 to 10 years"
    m = re.search(r'(\d+)\s*to\s*(\d+)\s*years?', text, re.I)
    if m:
        low, high = int(m.group(1)), int(m.group(2))
        mid = (low + high) / 2
        if mid < 5:
            return 'Less than 5 years'
        elif mid <= 10:
            return '5 to 10 years'
        elif mid <= 20:
            return '10 to 20 years'
        else:
            return 'Above 20 years'
    if 'less than 5' in text_lower:
        return 'Less than 5 years'
    if '5 to 10' in text_lower:
        return '5 to 10 years'
    if '10 to 15' in text_lower or '10 to 20' in text_lower or '15 to 20' in text_lower:
        return '10 to 20 years'
    if 'above 20' in text_lower or 'more than 20' in text_lower:
        return 'Above 20 years'
    return None


# ────────────────────────────────────────────────────────────────
#  FURNISHING EXTRACTOR
# ────────────────────────────────────────────────────────────────

def extract_furnishing(text: str) -> str | None:
    """Extract furnishing type from text."""
    if not isinstance(text, str):
        return None
    text_lower = text.lower()
    if 'fully furnished' in text_lower or 'fully-furnished' in text_lower:
        return 'Furnished'
    if 'semi-furnished' in text_lower or 'semi furnished' in text_lower:
        return 'Semi-Furnished'
    if 'unfurnished' in text_lower or 'un-furnished' in text_lower:
        return 'Unfurnished'
    if 'furnished' in text_lower:
        return 'Furnished'
    return None


# ────────────────────────────────────────────────────────────────
#  FACING EXTRACTOR
# ────────────────────────────────────────────────────────────────

_FACING_PATTERNS = [
    (r'north\s*-?\s*east\s*facing', 'North - East'),
    (r'north\s*-?\s*west\s*facing', 'North - West'),
    (r'south\s*-?\s*east\s*facing', 'South - East'),
    (r'south\s*-?\s*west\s*facing', 'South -West'),
    (r'north\s*facing', 'North'),
    (r'south\s*facing', 'South'),
    (r'east\s*facing', 'East'),
    (r'west\s*facing', 'West'),
    # without "facing" keyword
    (r'north\s*-?\s*east', 'North - East'),
    (r'north\s*-?\s*west', 'North - West'),
    (r'south\s*-?\s*east', 'South - East'),
    (r'south\s*-?\s*west', 'South -West'),
]


def extract_facing(text: str) -> str | None:
    """Extract facing direction from text."""
    if not isinstance(text, str):
        return None
    for pattern, direction in _FACING_PATTERNS:
        if re.search(pattern, text, re.I):
            return direction
    return None


# ────────────────────────────────────────────────────────────────
#  BOOLEAN FEATURE EXTRACTORS  (description + amenities)
# ────────────────────────────────────────────────────────────────

def has_parking(text: str) -> bool:
    """
    True if the property has dedicated car parking.
    Covers amenity tags ("Parking", "Private Garage", "Visitor Parking"),
    description phrases ("1 covered car parking", "open parking space",
    "reserved parking", "carport"), and numeric parking mentions.
    Deliberately excludes "park facing" (a garden/view mention).
    """
    if not isinstance(text, str):
        return False
    return bool(re.search(
        r'\bparking\b'                     # Parking / car parking
        r'|\bgarage\b'                     # Garage / Private Garage
        r'|\bcarport\b'                    # Carport
        r'|\bcar\s*(?:space|park)\b'       # car space / car park
        r'|\bcovered\s*(?:car\s*)?park'    # covered parking / covered car park
        r'|\bopen\s*(?:car\s*)?park'       # open parking / open car park
        r'|\breserved\s*parking\b'         # reserved parking
        r'|\bvisitor\s*parking\b'          # visitor parking
        r'|\bprivate\s*garage\b'           # private garage
        r'|\bstilt\s*parking\b'            # stilt parking
        r'|\bbasement\s*parking\b',        # basement parking
        text, re.I,
    ))


def has_pool(text: str) -> bool:
    """
    True if the property has a swimming pool or water recreation feature.
    Covers amenity tag ("Pool"), and description variants:
    infinity pool, kids pool, recreation pool, splash pool, jacuzzi, etc.
    """
    if not isinstance(text, str):
        return False
    return bool(re.search(
        r'\bpool\b'                        # Pool (amenity tag / general)
        r'|\bswimming\b'                   # swimming (pool)
        r'|\binfinity\s*pool\b'            # infinity pool
        r'|\bsplash\s*pool\b'              # splash pool
        r'|\bkids?\s*pool\b'              # kids pool
        r'|\badult\s*pool\b'               # adult pool
        r'|\bprivate\s*pool\b'             # private pool
        r'|\bjacuzzi\b'                    # jacuzzi
        r'|\baqua\s*(?:park|zone)\b',      # aqua park / aqua zone
        text, re.I,
    ))


def has_main_road(text: str) -> bool:
    """
    True if the property is on / close to / facing a main / wide road.

    Signals picked from real data:
      Amenity tag  : 'Main Road'
      Explicit     : 'main road', 'main-road', 'on main road',
                     'main road facing', 'faces main road'
      Road-facing  : 'road facing', 'road-facing', 'road front',
                     'facing road', 'front road', 'on road'
      Wide road    : '<N> meter/metre/feet/ft wide road',
                     '<N>m road', '<N>ft road', 'wide road',
                     '60/75/80/100/120/130 metre road'
      Highway/ring : 'highway', 'ring road', 'bypass road',
                     'expressway', 'national highway', 'nh-', 'sh-'
      Access       : 'main road access', 'main road connectivity',
                     'main road proximity'

    Intentionally EXCLUDED (plain internal-road connectivity mentions):
      'road connectivity', 'good road', 'road network',
      'road number', 'road view' alone (ambiguous)
    """
    if not isinstance(text, str):
        return False
    return bool(re.search(
        # ── explicit "main road" ────────────────────────────────
        r'\bmain[\s\-]road\b'
        # ── road-facing / road-front ────────────────────────────
        r'|\broad[\s\-]facing\b'
        r'|\broad[\s\-]front\b'
        r'|\bfacing\s+(?:the\s+)?road\b'
        r'|\bfront\s+(?:of\s+)?road\b'
        r'|\bon\s+(?:the\s+)?road\b'
        # ── wide road with measurement ──────────────────────────
        r'|\b\d+\s*(?:meter|metre|mtr|feet|ft|m)\s*(?:wide\s*)?road\b'
        r'|\broad\s*(?:of\s*)?\d+\s*(?:meter|metre|mtr|feet|ft|m)\b'
        r'|\b\d+m\s*road[\s\-]facing\b'
        r'|\bwide\s*road\b'
        # ── highway / bypass / expressway ───────────────────────
        r'|\bexpressway\b'
        r'|\bnational\s*highway\b'
        r'|\bnh[\s\-]\d+'
        r'|\bsh[\s\-]\d+'
        r'|\bring\s*road\b'
        r'|\bbypass\s*road\b'
        r'|\bbypass\b'
        # ── broad road / arterial ───────────────────────────────
        r'|\bbroad\s*road\b'
        r'|\barterial\s*road\b'
        r'|\b(?:60|75|80|100|120|130)\s*(?:meter|metre|mtr|feet|ft|m)?\s*road\b',
        text, re.I,
    ))


def has_garden_park(text: str) -> bool:
    """
    True if the property has / overlooks a garden or park area.
    Covers amenity tags ("Garden/Park", "Park"), description phrases
    ("garden", "park facing", "green area", "landscaped", jogging track).
    Uses negative look-ahead to avoid matching 'parking'.
    """
    if not isinstance(text, str):
        return False
    return bool(re.search(
        r'\bgarden\b'                            # garden (any context)
        r'|\bpark(?!ing\b)(?:\s*/\s*park)?\b'   # park / Garden/Park  but NOT parking
        r'|\bpark[\s\-]facing\b'                 # park facing (view)
        r'|\bgreen\s*(?:area|zone|space|belt)\b' # green area / zone / belt
        r'|\blandscaped?\b'                      # landscaped / landscape
        r'|\bopen\s*space\b'                     # open space
        r'|\bjogging\b'                          # jogging track
        r'|\bstrolling\s*track\b'                # strolling track
        r'|\blawn\b'                             # lawn
        r'|\bamphitheatre\b'                     # amphitheatre (outdoor)
        r'|\bchildren.?s?\s*play\b',             # children's play area
        text, re.I,
    ))


# ────────────────────────────────────────────────────────────────
#  GATED / CORNER / ROAD-WIDTH EXTRACTORS
# ────────────────────────────────────────────────────────────────

def is_gated(text: str) -> bool:
    """
    True if the property is in a gated community / society / complex.
    Covers:
      - gated society / gated community / gated complex / gated colony
      - gated township / gated enclave / gated compound
      - boundary wall / enclosed society / secure compound
      - gated (standalone, preceded/followed by whitespace)
    """
    if not isinstance(text, str):
        return False
    return bool(re.search(
        r'\bgated\b'                               # gated (any context)
        r'|\bboundary\s*wall\b'                    # boundary wall
        r'|\benclosed\s*(?:communit|societ|colony|complex)\b'
        r'|\bsecure[d]?\s*(?:compound|complex|societ|communit)\b'
        r'|\bguarded\s*(?:complex|societ|communit|colony)\b',
        text, re.I,
    ))


def is_corner(text: str) -> bool:
    """
    True if the property is a corner plot / corner property / corner house.
    Covers:
      - corner plot / corner property / corner flat / corner house
      - corner floor / corner apartment / corner unit
      - 3 side corner / three side corner / two side corner
      - L-type corner / L type corner
      - corner facing / corner location
    """
    if not isinstance(text, str):
        return False
    return bool(re.search(
        r'\bcorner\s*(?:plot|propert|flat|house|floor|apartment'
        r'|unit|villa|bungalow|build)\b'            # corner + property noun
        r'|(?:3|three|two|2|4|four)\s*side\s*corner\b'  # N-side corner
        r'|\bl[\s\-]?type\s*corner\b'              # L-type corner
        r'|\bcorner\s*(?:facing|location|side)\b'   # corner facing/location
        r'|\bcorner\s*(?:1st|2nd|3rd|first|second|third|ground)\b'  # corner + floor
        r'|\b(?:plot|propert|flat|house|floor|apartment)\s*(?:on|at|in)?\s*corner\b',  # noun + corner
        text, re.I,
    ))


# Conversion from common units to feet
_ROAD_UNIT_TO_FT = {
    'm':      3.28084,
    'meter':  3.28084,
    'metre':  3.28084,
    'mtr':    3.28084,
    'ft':     1.0,
    'feet':   1.0,
    'foot':   1.0,
}

# Ordered patterns – first match wins
_ROAD_WIDTH_PATTERNS = [
    # "30 ft wide road", "9 mtr wide road", "100 meter road"
    re.compile(
        r'(\d+\.?\d*)\s*(meter|metre|mtr|feet|ft|foot|m)\s*(?:wide\s*)?road',
        re.I,
    ),
    # "road width 20m", "road width: 30 ft"
    re.compile(
        r'road\s*width[:\s]*(\d+\.?\d*)\s*(meter|metre|mtr|feet|ft|foot|m)?',
        re.I,
    ),
    # "width of 20 m road"
    re.compile(
        r'width\s*(?:of\s*)?\s*(\d+\.?\d*)\s*(meter|metre|mtr|feet|ft|foot|m)',
        re.I,
    ),
    # compact: "9m road", "30ft road"
    re.compile(
        r'(\d+\.?\d*)\s*(m|ft|mtr)\s*road',
        re.I,
    ),
    # "frontage of 70 metres", "frontage is 100 feet"
    re.compile(
        r'frontage\s*(?:of|is)?\s*(\d+\.?\d*)\s*(meter|metre|mtr|feet|ft|foot|m)',
        re.I,
    ),
]


def extract_road_width_ft(text: str) -> float | None:
    """
    Extract road / frontage width from text and return it in **feet**.
    Returns None when no numeric width is found.

    Examples
    --------
    >>> extract_road_width_ft('9m road facing north')
    29.53  (≈ 9 × 3.28084)
    >>> extract_road_width_ft('30 ft wide road')
    30.0
    """
    if not isinstance(text, str):
        return None
    for pat in _ROAD_WIDTH_PATTERNS:
        m = pat.search(text)
        if m:
            value = float(m.group(1))
            unit_str = (m.group(2) or 'm').lower().strip()
            factor = _ROAD_UNIT_TO_FT.get(unit_str, 1.0)
            return round(value * factor, 2)
    return None


# ────────────────────────────────────────────────────────────────
#  COMBINED TEXT HELPER
# ────────────────────────────────────────────────────────────────

def combine_text(row: pd.Series, cols: list[str] = ["description", "amenities"]) -> str:
    """Concatenate multiple text columns into one search string."""
    parts = []
    for c in cols:
        val = row.get(c)
        if isinstance(val, str) and val.strip():
            parts.append(val)
    return " ".join(parts)
