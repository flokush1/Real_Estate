# Market Intelligence — Mathematics & Calculations

**Source file:** `real_estate/utils/market_intelligence.py`
**Artifacts:** `opt/{segment}/market_intelligence.csv`
**Segments:** `apt` (Apartment), `builder_floor` (Builder Floor), `plot` (Plot)

---

## 1. Data Ingestion & Normalization

### Raw Sources

| File | Source |
|------|--------|
| `real_estate_data/real_estate_data/ho_raw_data.csv` | Housing.com |
| `real_estate_data/real_estate_data/mb_raw_data.csv` | MagicBricks |

Both sources are concatenated into a single normalized frame. Only `event_type ∈ {new, unchanged, updated}` (active listings) are used in metric computation.

### Area Normalization → Square Feet

All area values are converted to sqft using fixed unit factors:

| Unit | Multiplier to sqft |
|------|--------------------|
| sqft / sq-ft / sft | 1.0 |
| sqyd / sq-yrd | 9.0 |
| sqm / sq-m | 10.7639 |
| acre | 43,560.0 |
| bigha | 27,000.0 |
| marla | 272.25 |
| hectare | 107,639.0 |

$$\text{area\_sqft} = \text{covered\_area\_value} \times \text{unit\_factor}$$

### Price Per Square Foot (PPSF)

$$\text{ppsf} = \frac{\text{price\_numeric}}{\text{area\_sqft}}$$

If the computed value is invalid (NaN or ≤ 0), the source field `sqft_price` is used instead.

### Days on Market (DOM)

$$\text{DOM} = \text{scrape\_date} - \text{posting\_date} \quad (\text{clipped at } 0)$$

**Exception:** When `posting_date == scrape_date` (HO full-refresh uploads), DOM is treated as unknown (`NaN`). The fallback uses the **first-seen scrape date** per property:

$$\text{DOM\_fallback} = \text{scrape\_date} - \min(\text{scrape\_date})\bigg|_{\text{property\_id}}$$

### City Normalization & Locality Overrides

Raw city strings are normalized to canonical names. Certain localities also override the source-assigned city to correct geographic mis-tagging:

| Locality | Forced City |
|----------|-------------|
| Noida Extension | Greater Noida |
| Greater Noida West | Greater Noida |

---

## 2. Per-Locality Metrics (computed per `segment × city × locality × scrape_date` group)

### 2.1 Supply Metrics

| Column | Formula |
|--------|---------|
| `active_supply_stock` | Count of unique `property_id` with active event |
| `new_supply_count` | Count of rows where `event_type = new` |
| `new_supply_velocity` | $\dfrac{\text{new\_supply\_count}}{\text{active\_supply\_stock}}$ |
| `updated_count` | Count of rows where `event_type = updated` |

### 2.2 Absorption Rate

Measures what fraction of active listings disappeared by the next scrape date, treating removal as a sale/lease:

$$\text{absorbed\_count} = \left|\,\text{active\_ids}_t \setminus \text{active\_ids}_{t+1}\,\right|$$

$$\text{absorption\_rate} = \frac{\text{absorbed\_count}}{\text{active\_supply\_stock}_t}$$

**Guard condition:** If the locality has **zero listings** on date $t+1$, absorption is set to 0 (not computed), preventing a false 100% signal when a scrape run simply did not cover the locality.

### 2.3 Price Revision (on `updated` listings only)

For each updated listing, the price change versus the previous scrape date:

$$\Delta_{\%} = \frac{p_t - p_{t-1}}{p_{t-1}} \times 100$$

| Column | Formula |
|--------|---------|
| `price_cut_count` | Count of listings where $p_t < p_{t-1}$ |
| `price_hike_count` | Count of listings where $p_t > p_{t-1}$ |
| `price_cut_frequency` | $\dfrac{\text{price\_cut\_count}}{\text{updated\_count}}$ |
| `price_hike_frequency` | $\dfrac{\text{price\_hike\_count}}{\text{updated\_count}}$ |
| `price_cut_median_pct` | Median of all $\Delta_{\%}$ where $\Delta < 0$ |
| `price_hike_median_pct` | Median of all $\Delta_{\%}$ where $\Delta > 0$ |

### 2.4 Days on Market & Stale Inventory

| Column | Formula |
|--------|---------|
| `median_days_on_market` | $\text{median}(\text{DOM})$ across all active listings |
| `stale_inventory_count` | Count of listings with $\text{DOM} > 90$ days |
| `stale_inventory_share` | $\dfrac{\text{stale\_inventory\_count}}{\text{active\_supply\_stock}}$ |

### 2.5 Price Statistics

| Column | Formula |
|--------|---------|
| `median_ppsf` | $\text{median}(\text{ppsf})$ |
| `p25_ppsf` | 25th percentile of ppsf |
| `p75_ppsf` | 75th percentile of ppsf |
| `median_price` | $\text{median}(\text{price\_numeric})$ |

### 2.6 Circle Rate & Price-to-Circle Ratio

`median_circle_rate` is the government-mandated floor price (₹/sqft) for the locality, sourced from city-specific JSON files in `real_estate_data/real_estate_data/circle_rates/` via `CircleRateMatcher`.

$$\text{price\_to\_circle\_ratio} = \frac{\text{median\_ppsf}}{\text{median\_circle\_rate}}$$

A ratio > 1 means the market is trading above the government circle rate (premium locality). A ratio < 1 signals potential distress or valuation risk.

### 2.7 Possession Split

| Column | Formula |
|--------|---------|
| `ready_inventory_share` | Share of listings with possession status = ready / immediate |
| `under_construction_share` | Share of listings with possession status = under construction / new launch |

### 2.8 Participant Counts

| Column | Formula |
|--------|---------|
| `agent_count` | Unique non-null `agent_type` values |
| `developer_count` | Unique non-null `developer_id` values |

---

## 3. Composite Index Scores

All indices (except PMS and CPS) are computed **within each city group** using percentile ranking, so scores are relative to other localities in the same city.

### Percentile Scoring Function

$$P(x_i) = \frac{\text{rank}(x_i)}{N} \times 100$$

Where $N$ is the number of localities in the city. `higher_is_better=False` inverts this: $P'(x_i) = 100 - P(x_i)$.

---

### 3.1 Supply Pressure Index (0–100, higher = more supply pressure)

$$\text{SPI} = 0.40 \cdot P(\text{active\_supply\_stock}) + 0.25 \cdot P(\text{new\_supply\_velocity}) + 0.20 \cdot P(\text{stale\_inventory\_share}) + 0.15 \cdot P(\text{under\_construction\_share})$$

| Component | Weight | Direction |
|-----------|--------|-----------|
| Active supply stock | 40% | Higher stock → higher pressure |
| New supply velocity | 25% | Higher velocity → higher pressure |
| Stale inventory share | 20% | More stale stock → higher pressure |
| Under-construction share | 15% | More UC supply → higher pressure |

---

### 3.2 Liquidity Index (0–100, higher = more liquid)

$$\text{LI} = 0.50 \cdot P(\text{absorption\_rate}) + 0.30 \cdot P^{-}(\text{median\_dom}) + 0.20 \cdot P^{-}(\text{stale\_inventory\_share})$$

Where $P^{-}$ denotes `higher_is_better=False`.

| Component | Weight | Direction |
|-----------|--------|-----------|
| Absorption rate | 50% | Higher absorption → more liquid |
| Median days on market | 30% | Lower DOM → more liquid |
| Stale inventory share | 20% | Less stale → more liquid |

---

### 3.3 Price Momentum Score (0–100, 50 = neutral)

$$\text{PMS} = \text{clip}\bigl(50 + 40 \cdot f_{\text{hike}} - 40 \cdot f_{\text{cut}},\; 0,\; 100\bigr)$$

If `updated_count = 0`, PMS = 50 (neutral).

| Value | Interpretation |
|-------|---------------|
| > 50 | Price hikes dominating |
| = 50 | Neutral / no revision data |
| < 50 | Price cuts dominating |

---

### 3.4 Circle Premium Score (0–100)

Measures how expensive the locality is **relative to its government circle rate**:

$$\text{relative} = \frac{\text{median\_ppsf}}{\text{median\_circle\_rate}}$$

$$\text{CPS} = \text{clip}\bigl(50 + 50 \cdot (\text{relative} - 1),\; 0,\; 100\bigr)$$

| Score | Meaning |
|-------|---------|
| 100 | Market PPSF is twice the circle rate (strong premium) |
| 50 | Market PPSF equals the circle rate |
| 0 | Market PPSF is at or below the circle rate |

**Fallback:** If no circle rate is available for the locality, the city-median PPSF is used as the denominator instead, preserving a relative intra-city comparison.

---

### 3.5 Demand Strength Index (0–100)

$$\text{DSI} = 0.35 \cdot P(\text{absorption\_rate}) + 0.25 \cdot \text{LI} + 0.20 \cdot \text{PMS} + 0.20 \cdot \text{CPS}$$

| Component | Weight |
|-----------|--------|
| Absorption rate (percentile) | 35% |
| Liquidity Index | 25% |
| Price Momentum Score | 20% |
| Circle Premium Score | 20% |

---

### 3.6 Market Heat Index (0–100)

The headline score combining demand and supply dynamics:

$$\text{MHI}_t = \text{clip}\bigl(50 + 0.5 \cdot \text{DSI} - 0.5 \cdot \text{SPI},\; 0,\; 100\bigr)$$

| Value | Interpretation |
|-------|---------------|
| High DSI | Pushes MHI up (strong demand) |
| High SPI | Pushes MHI down (oversupplied) |
| 50 | Perfectly balanced |

### 3.7 Market Label (from MHI)

| MHI Range | Label |
|-----------|-------|
| ≥ 75 | Hot demand-led market |
| 60 – 74 | Positive market |
| 45 – 59 | Balanced market |
| 30 – 44 | Supply-heavy market |
| < 30 | Weak / stale market |

---

## 4. Temporal Smoothing

Individual scrape dates introduce noise: a single data collection run may miss some listings, inflate absorption counts, or capture an anomalous pricing day. Smoothing over multiple scrape dates removes this noise.

### 4.1 Rolling 4-Scrape MHI (implemented)

After all per-date MHI values are computed, a rolling mean is applied over the **last 4 scrape dates** within each `(segment, city, locality)` group:

$$\text{MHI\_rolling4}_t = \frac{1}{\min(4,\, t)} \sum_{k=\max(1,\, t-3)}^{t} \text{MHI}_k$$

- Window size: 4 scrape dates (not calendar days)
- `min_periods=1` so the first 1–3 dates still produce a value (partial window)
- Stored as column `mhi_rolling4`; exposed in the API as `marketHeatSmoothed`

**Why 4?** With typically 1–2 scrapes per month, a window of 4 covers roughly 2–3 months — long enough to damp week-to-week noise but short enough to still track turning points quickly.

### 4.2 Exponential Smoothing (reference formula)

An alternative to the rolling mean is exponential weighted smoothing (EMA), which down-weights older observations:

$$\text{MHI\_smoothed}_t = \alpha \cdot \text{MHI}_t + (1 - \alpha) \cdot \text{MHI\_smoothed}_{t-1}$$

where $\alpha \in (0, 1)$ is the smoothing factor:

| $\alpha$ | Behaviour |
|----------|-----------|
| 0.8 | Strongly reactive — follows recent changes quickly |
| 0.5 | Balanced — equal weight to recent vs. historical |
| 0.2 | Slow-moving — long memory, resistant to spikes |

A recommended starting point is $\alpha = 0.4$ (roughly equivalent to a 4-period rolling mean in responsiveness).

---

## 5. Summary Flow

```
Raw CSV (HO + MB)
        │
        ▼
  Normalize & clean
  (area → sqft, city overrides, DOM fix)
        │
        ▼
  Filter: active events only
  (event_type ∈ {new, unchanged, updated})
        │
        ▼
  Group by: segment × city × locality × scrape_date
        │
        ▼
  Compute per-group metrics
  (supply, absorption, price revisions, DOM, PPSF)
        │
        ├── Circle rate lookup (CircleRateMatcher)
        │   → median_circle_rate, price_to_circle_ratio
        │
        ▼
  _compute_indices()
  ── within each city:
     SPI → LI → PMS
     CPS = ppsf / circle_rate (or city-median fallback)
     DSI → MHI → label
        │
        ▼
  Rolling smoothing
  mhi_rolling4 = rolling_mean(MHI, last 4 scrape dates)
        │
        ▼
  Write opt/{segment}/market_intelligence.csv
  Write opt/{segment}/market_summary.json
```

---

## 6. Key Data Quality Notes

| Issue | Handling |
|-------|---------|
| HO full-refresh uploads (posting_date = scrape_date) | DOM marked NaN; first-seen date fallback applied |
| Locality absent from next scrape date | Absorption not computed (set to 0, not 100%) |
| MagicBricks mis-tags "Noida Extension" as city="Noida" | Overridden to "Greater Noida" via `LOCALITY_CITY_OVERRIDES` |
| Locality has no circle rate data | CPS falls back to city-median PPSF comparison |
| Single noisy scrape date skewing MHI | Smoothed via `mhi_rolling4` (4-scrape rolling mean) |
