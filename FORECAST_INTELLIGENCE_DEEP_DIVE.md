# FloData Forecast Intelligence — Complete Technical Reference

**System:** NCR Real Estate AI Platform  
**Method:** FloData Future Price Method  
**Segments:** Builder Floor · Apartment / Flat · Plot  
**Geography:** Delhi NCR (Delhi, Noida, Gurgaon, Faridabad, Ghaziabad, Greater Noida)

---

## Table of Contents

1. [What This System Does](#1-what-this-system-does)
2. [Data Inputs](#2-data-inputs)
3. [Stage 1 — Historical Trend Construction](#3-stage-1--historical-trend-construction)
4. [Stage 2 — Locality Growth Index (ML Trend Model)](#4-stage-2--locality-growth-index-ml-trend-model)
5. [Stage 3 — Rho (Blending Weight)](#5-stage-3--rho-blending-weight)
6. [Stage 4 — FloData Forecast Formula](#6-stage-4--flodata-forecast-formula)
7. [Stage 5 — Output Artifacts](#7-stage-5--output-artifacts)
8. [YoY Analytics](#8-yoy-analytics)
9. [Locality Distribution Statistics](#9-locality-distribution-statistics)
10. [Buy Decision Scoring](#10-buy-decision-scoring)
11. [ROI Projection Model](#11-roi-projection-model)
12. [Locality Intelligence Profile](#12-locality-intelligence-profile)
13. [XAI — Explainable Prediction Drivers](#13-xai--explainable-prediction-drivers)
14. [Formula Consistency Validation](#14-formula-consistency-validation)
15. [Serving-Time Valuation (API)](#15-serving-time-valuation-api)
16. [Model Performance Benchmarks](#16-model-performance-benchmarks)
17. [Key Files and Code Locations](#17-key-files-and-code-locations)
18. [Symbol Glossary](#18-symbol-glossary)

---

## 1. What This System Does

The forecast intelligence layer converts two raw numbers — **a listing price** and **a model-estimated fair value** — into a full multi-quarter price trajectory for every property. It then translates that trajectory into investor-grade analytics: Year-over-Year growth, buy/avoid decisions, and full cashflow ROI projections.

The three-layer architecture:

```
RAW DATA
  └── Historical locality price panels (quarterly, per sqft)
  └── Property master files (listing price + ML-predicted price)
  └── Government circle rates (floor valuation)

LAYER 1 — Locality Growth Index  (ML: RandomForest ensemble)
  └── Learns how each locality grows quarter-over-quarter
  └── Projects 20 future quarters (5 years) per locality

LAYER 2 — Property Forecast       (FloData blending formula)
  └── Blends listing price and model price via decaying rho weight
  └── Multiplies by locality growth index
  └── Produces per-property, per-quarter forecast

LAYER 3 — Investment Analytics    (deterministic scoring)
  └── YoY series, buy score, ROI, CAGR, volatility
```

---

## 2. Data Inputs

### 2.1 Property Master Files

Three CSV files, one per segment:

| File | Segment | Properties (approx.) |
|------|---------|----------------------|
| `inputs/apartment_with_pi.csv` | Apartment | ~67,726 |
| `inputs/builder_floor_with_pi.csv` | Builder Floor | ~42,098 |
| `inputs/plot_with_pi.csv` | Plot | ~varies |

**Critical columns in each file:**

| Column | Meaning |
|--------|---------|
| `property_id` | Unique listing identifier (e.g. `HO_19312180`) |
| `locality` | Locality name as listed (e.g. `Sector 19 Dwarka`) |
| `city` | City name (e.g. `Delhi`) |
| `price_per_sqft` | **Listing price** $L_i$ — what the seller is asking |
| `pi_price_per_sqft` | **Model price** $P_i$ — ML-estimated fair value |
| `covered_area_sqft` | Built-up area (BF / APT) |
| `plot_area` | Land area in sqft (Plot) |
| `circle_rate` | Government floor price per sqft |
| `bhk` | Number of bedrooms (BF / APT) |
| `is_parking`, `is_corner`, `is_gated` | Binary property features |
| `road_width_ft` | Road width in feet |

### 2.2 Historical Trend Files (Interpolated Growth Panels)

Located in `inputs/interpolated_growth_trend_builder/{segment}/`.

Structure: `{city}/{locality}.csv`

Each file contains a **quarterly time series** of median price per sqft, already smoothed and interpolated to fill gaps:

```
date,price_per_sqft
2021-01-01,8200.0
2021-04-01,8450.0
2021-07-01,8700.0
...
```

Coverage per segment:
- Apartment: 609 localities with trend data
- Builder Floor: 575 localities with trend data

### 2.3 Government Circle Rate Data

Located in `real_estate_data/real_estate_data/circle_rates/` as JSON files.

Circle rate is the **minimum government-declared floor value** for a locality. It is used to:
1. Anchor the ML valuation model (BF/APT models predict a ratio over circle rate).
2. Compute affordability metrics and XAI explanations.

### 2.4 Macroeconomic Inputs

| File | Content |
|------|---------|
| `inputs/rbi_repo_rates.csv` | RBI repo rate history |
| `inputs/gsdp_repo_quarterly.csv` | State GDP quarterly growth |
| `inputs/repo_homeloan_quarterly.csv` | Home loan rate panel |

These feed into the locality trend model as auxiliary features (`loan_repo_spread`, `gsdp_grow_pct`).

---

## 3. Stage 1 — Historical Trend Construction

Before any ML, the raw listing data is **aggregated into quarterly locality price panels**:

### 3.1 Unit standardisation

All area values are converted to sqft using multiplicative factors:

$$
\text{area\_sqft} = \text{area\_value} \times \text{unit\_factor}
$$

| Unit | Factor |
|------|--------|
| sq-ft | 1.0 |
| sq-yrd | 9.0 |
| sq-m | 10.7639 |
| acre | 43,560 |

### 3.2 Price per sqft

$$
\text{ppsf} = \frac{\text{total\_price}}{\text{area\_sqft}}
$$

### 3.3 Quality gates (outlier removal)

Only rows passing all three tests are kept:

$$
\text{ppsf} > 0, \quad \text{area} > 0, \quad \text{circle\_rate} > 0
$$

$$
0.7 < \frac{\text{ppsf}}{\text{circle\_rate}} < 25
$$

This ratio sanity window removes both suspiciously cheap and wildly overpriced listings before aggregation.

### 3.4 Quarterly aggregation

For each `(locality, quarter)` pair, the median ppsf is taken. Gaps are filled by interpolation to produce the smooth quarterly panels stored in `inputs/interpolated_growth_trend_builder/`.

---

## 4. Stage 2 — Locality Growth Index (ML Trend Model)

This is the **engine** that predicts how prices will move in each locality over the next 20 quarters.

### 4.1 Features engineered from the quarterly panel

For each locality-quarter row, the model receives:

| Feature | Description |
|---------|-------------|
| `lag_log_1` | log(ppsf) one quarter ago — captures immediate momentum (≈ 50% importance) |
| `lag_log_2` | log(ppsf) two quarters ago — captures medium-term momentum |
| `lag_price_1` | Raw price one quarter ago |
| `lag_price_2` | Raw price two quarters ago |
| `growth_lag_1` | QoQ growth rate lagged one quarter |
| `growth_lag_2` | QoQ growth rate lagged two quarters |
| `yoy_growth` | Year-over-year growth rate |
| `momentum` | Short-term acceleration (change of change) |
| `growth_roll_mean_4` | Rolling 4-quarter mean of growth rates |
| `growth_roll_std_4` | Rolling 4-quarter volatility of growth rates |
| `price_vs_city_avg` | Locality price / city median price ratio |
| `price_vs_zone_avg` | Locality price / zone median price ratio |
| `zone_price_rank` | Rank of locality price within its zone |
| `time_index` | Quarter number (trend proxy) |
| `gsdp_grow_pct` | State GDP growth rate |
| `loan_repo_spread` | Home loan rate minus RBI repo rate |

### 4.2 Model architecture

An **ensemble** of decision trees (typically RandomForest or gradient-boosted variant) is trained to predict `log(price_per_sqft)` for the next quarter. The training uses time-series cross-validation (no future leakage).

**Log-space target:**

$$
y = \log(\text{ppsf})
$$

**Back-transform for predictions:**

$$
\widehat{\text{ppsf}} = \exp(\hat{y})
$$

### 4.3 Cumulative growth index construction

After fitting the model, it is rolled forward iteratively for 20 future quarters. Each quarter the predicted ppsf is fed back as a lag feature for the next quarter.

The cumulative **locality growth index** at quarter $t$ relative to a base period is:

$$
I_{l,t} = \frac{\hat{p}_{l,t}}{\hat{p}_{l,0}}
$$

where $\hat{p}_{l,0}$ is the locality price at the forecast start date. This $I_{l,t}$ is stored in `opt/{segment}/future_forecasts.csv`.

### 4.4 Holdout performance (Apartment segment)

| Metric | Holdout | Cross-Val (mean) |
|--------|---------|------------------|
| MAE (₹/sqft) | 556.6 | 554.6 ± 12.6 |
| RMSE (₹/sqft) | 1,275.5 | 1,141.1 ± 23.9 |
| MAPE | 5.10% | 5.42% ± 0.22% |
| R² | 0.9546 | 0.9604 ± 0.003 |

### 4.5 Holdout performance (Builder Floor segment)

| Metric | Holdout | Cross-Val (mean) |
|--------|---------|------------------|
| MAE (₹/sqft) | 581.6 | 614.3 ± 47.1 |
| RMSE (₹/sqft) | 1,012.5 | 1,017.9 ± 85.4 |
| MAPE | 5.79% | 5.48% ± 0.33% |
| R² | 0.9821 | 0.9835 ± 0.002 |

---

## 5. Stage 3 — Rho (Blending Weight)

$\rho$ is the core innovation of the FloData method. It controls **how much weight is placed on the seller's listing price vs the ML model's fair-value estimate**. High $\rho$ means the forecast stays close to the listing price; low $\rho$ means it converges toward the model.

### 5.1 Component inputs

| Symbol | Name | What it measures |
|--------|------|-----------------|
| $C_i$ | Comparable support | How well the listing price is backed by similar nearby transactions (0–1) |
| $U_i$ | Uniqueness | How unusual the property is compared to its local peer group — via k-NN distance percentile (0–1) |
| $M_i$ | Model confidence | Composite of model coverage, cross-val fit score, and prediction stability (0–1) |

### 5.2 Initial rho (per property)

$$
\rho_0 = \text{clip}\!\left(0.25 + 0.30 \cdot C_i + 0.20 \cdot U_i - 0.25 \cdot M_i,\; 0.10,\; 0.70\right)
$$

**Interpretation of each term:**

- `+0.25` — base weight anchoring at 25% listing influence
- `+0.30 × C_i` — if many comparable sales confirm the listing price, trust it more
- `+0.20 × U_i` — if the property is unusual, the ML model may not represent it well, so weight the listing more
- `−0.25 × M_i` — if model confidence is high, reduce listing influence and trust the model

**Clipping:** $\rho_0$ is bounded to $[0.10, 0.70]$ to prevent degenerate fully-listing or fully-model forecasts.

### 5.3 Rho across the forecast population (Apartment)

| Statistic | Value |
|-----------|-------|
| Mean $\rho_0$ | 0.324 |
| Median $\rho_0$ | 0.322 |
| Min $\rho_0$ | 0.128 |
| Max $\rho_0$ | 0.632 |

### 5.4 Rho decay over forecast horizon

As the forecast moves further into the future, listing price information becomes less relevant. The weight decays exponentially:

$$
\rho_t = \rho_0 \cdot \gamma^t, \quad \gamma = 0.90
$$

where $t$ is the quarter index (0, 1, 2, …, 19).

**Example with $\rho_0 = 0.32$:**

| Quarter | $\rho_t$ |
|---------|----------|
| 0 (current) | 0.320 |
| 4 (Y+1) | 0.209 |
| 8 (Y+2) | 0.137 |
| 12 (Y+3) | 0.089 |
| 16 (Y+4) | 0.058 |
| 20 (Y+5) | 0.038 |

By year 5, the forecast is almost entirely determined by the ML model's trajectory. The listing price contribution has decayed to under 4%.

---

## 6. Stage 4 — FloData Forecast Formula

### 6.1 Log-space formulation (primary)

The canonical FloData formula operates in log-space:

$$
\log V_{i,t} = \log(P_i) + \log(I_{l,t}) + \rho_t \cdot \bigl(\log(L_i) - \log(P_i)\bigr)
$$

Equivalently in geometric form:

$$
V_{i,t} = P_i \cdot I_{l,t} \cdot \left(\frac{L_i}{P_i}\right)^{\rho_t}
$$

**Where:**
- $P_i$ = model price (fair value estimate, ₹/sqft)
- $L_i$ = listing price (asking price, ₹/sqft)
- $I_{l,t}$ = locality growth index at quarter $t$
- $\rho_t$ = decayed blending weight
- $V_{i,t}$ = forecast price per sqft at quarter $t$

**Intuition:**  
When $\rho_t = 0$: $V_{i,t} = P_i \cdot I_{l,t}$ — pure model trajectory, listing price ignored.  
When $\rho_t = 1$: $V_{i,t} = L_i \cdot I_{l,t}$ — listing price scaled by locality trend.  
At intermediate $\rho_t$: geometric blend, weighted toward model as time increases.

### 6.2 Linear-space formulation (equivalent, used for validation)

$$
F_{i,t} = \bigl[\rho_t \cdot L_i + (1 - \rho_t) \cdot P_i\bigr] \cdot I_{l,t}
$$

This is mathematically related but not exactly identical to the log form. Both are computed at serving time to verify consistency (`mathValidation` block in the API response).

### 6.3 Total price at horizon

$$
\widehat{\text{TotalPrice}}_{i,t} = V_{i,t} \cdot A_i
$$

where $A_i$ is the property area in sqft.

### 6.4 What a typical forecast looks like

For an apartment at **₹15,000/sqft listing** in Sector 4 Dwarka with:
- $P_i = 15,936$ (model slightly above listing)
- $\rho_0 = 0.327$, $\gamma = 0.90$
- $I_{l,t=4} = 1.062$ (locality grew 6.2% over 4 quarters)

Quarter 4 forecast:

$$\rho_4 = 0.327 \times 0.90^4 = 0.214$$

$$V_{i,4} = 15{,}936 \times 1.062 \times \left(\frac{15{,}000}{15{,}936}\right)^{0.214} \approx 16{,}778 \text{ ₹/sqft}$$

---

## 7. Stage 5 — Output Artifacts

For each segment, the pipeline writes these files to `opt/{segment}/`:

| File | Rows | Description |
|------|------|-------------|
| `future_forecasts.csv` | ~localities × 20 quarters | Locality-level quarterly price forecasts |
| `property_forecasts_flodata.csv` | ~properties × 20 quarters | Per-property quarterly forecast (all FloData columns) |
| `property_forecasts_flodata_new.csv` | Same, latest run | APT segment: newer model version preferred |
| `rho_details.csv` | One row per property | $C_i$, $U_i$, $M_i$, $\rho_0$ per property |
| `metrics.json` | — | Model-level performance stats, hyperparams |
| `model_summary.md` | — | Human-readable training report |
| `feature_importance.csv` | One row per feature | Feature importances from the ensemble |
| `test_predictions.csv` | Test set rows | Holdout predictions vs actuals |
| `horizon_metrics.csv` | One row per horizon | MAE/MAPE broken out by forecast quarter |
| `locality_metrics.csv` | One row per locality | Per-locality holdout metrics |
| `cv_results.csv` | Cross-val folds | Fold-level metric details |

### Property forecast file columns

| Column | Type | Description |
|--------|------|-------------|
| `property_id` | str | Unique property identifier |
| `locality` | str | Locality name |
| `trend_locality` | str | Matched trend locality key |
| `date` | date | Quarter start date |
| `quarter` | int | Quarter index (1–20) |
| `P_i` | float | Model price (fixed for property) |
| `L_i` | float | Listing price (fixed for property) |
| `rho_0` | float | Initial rho |
| `rho_t` | float | Decayed rho for this quarter |
| `I_lt` | float | Locality growth index for this quarter |
| `forecast_price_per_sqft` | float | $V_{i,t}$ — the final forecast value |
| `forecast_price_numeric` | float | Alias for forecast_price_per_sqft |

---

## 8. YoY Analytics

### 8.1 Property Year-over-Year series

Starting from today's forecast date, the system walks the quarterly forecast forward in yearly steps:

$$
\text{YoY\%}_{Y+k} = \left(\frac{V_{i,\,Y+k}}{V_{i,\,Y+(k-1)}} - 1\right) \times 100
$$

This produces a chain: Y+1, Y+2, Y+3, Y+4, Y+5 — each showing the growth relative to the prior year's price (not always relative to today).

**Output fields per YoY row:**
```json
{
  "label": "Y+1",
  "yoy_pct": 8.3,
  "anchor_date": "2026-07-01",
  "anchor_price": 15800,
  "target_date": "2027-07-01",
  "target_price": 17112
}
```

### 8.2 Locality YoY (from today)

The locality-level series is aggregated by taking the **median** across all properties in the locality at each date. The YoY is then:

$$
\text{LocalityYoY\%} = \left(\frac{\text{median}_{t+4q}}{\text{median}_{t}} - 1\right) \times 100
$$

This is a single number (not a chain) representing the expected 1-year price appreciation for the locality.

---

## 9. Locality Distribution Statistics

At every forecast date, the system computes the **distribution of prices across all properties** within the locality:

| Statistic | Symbol |
|-----------|--------|
| Median | $p_{50}$ |
| Mean | $\bar{V}_t$ |
| 25th percentile | $p_{25}$ |
| 75th percentile | $p_{75}$ |
| Min | $V_{min,t}$ |
| Max | $V_{max,t}$ |

These power the **distribution fan chart** in the UI, which shows the spread of property prices within a locality over time — helping investors see both the trend direction and the valuation uncertainty band.

---

## 10. Buy Decision Scoring

The buy-decision endpoint converts the forecast context into a single 0–100 investment score with a three-tier verdict.

### 10.1 Input signals

**Valuation gap:**

$$
\Delta\% = \left(\frac{L_i - P_i}{P_i}\right) \times 100
$$

Positive = overpriced relative to model; Negative = undervalued.

**Expected upside at chosen horizon:**

$$
\text{upside\%} = \left(\frac{V_{i,\text{horizon}}}{L_i} - 1\right) \times 100
$$

**Average YoY growth:** mean of the YoY chain described in Section 8.1.

**QoQ Volatility:**

$$
\text{QoQ\%}_t = \frac{V_{i,t} - V_{i,t-1}}{V_{i,t-1}} \times 100
$$

$$
\sigma_{\text{qoq}} = \text{std}(\text{QoQ\%}_t)
$$

### 10.2 Score components

| Component | Formula | Weight |
|-----------|---------|--------|
| Valuation score | $\text{clip}(50 - \Delta\% \times 2,\; 0, 100)$ | 40% |
| Growth score | $\text{clip}(50 + 3 \times \overline{\text{YoY\%}},\; 0, 100)$ | 35% |
| Upside score | $\text{clip}(50 + 1.5 \times \text{upside\%},\; 0, 100)$ | 25% |

**Risk penalty:**

$$
\text{riskPenalty} = \text{clip}(2 \times \sigma_{\text{qoq}},\; 0, 30)
$$

### 10.3 Overall score

$$
\text{overall} = \text{clip}\!\left(0.40 \cdot s_{val} + 0.35 \cdot s_{growth} + 0.25 \cdot s_{upside} - \text{riskPenalty},\; 0, 100\right)
$$

### 10.4 Verdict bands

| Score Range | Verdict |
|-------------|---------|
| ≥ 68 | **Buy** |
| 50 – 67 | **Watch** |
| < 50 | **Avoid** |

### 10.5 Confidence

Confidence is computed from data depth (how many forecast rows exist for the property), whether the trend locality is a city-average fallback, and the $\rho_0$ spread. Lower confidence is reported when a city-average trend is used instead of a locality-specific trend.

---

## 11. ROI Projection Model

The ROI endpoint builds a simplified cashflow model over a user-selected holding horizon.

### 11.1 Price inputs

$$
\text{buyPrice} = L_i \times A_i
$$

$$
\text{purchaseCosts} = \text{buyPrice} \times \frac{\text{purchaseCostPct}}{100}
$$

$$
\text{totalInvested} = \text{buyPrice} + \text{purchaseCosts}
$$

At the end of the holding period:

$$
\text{grossSalePrice} = V_{i,\text{horizon}} \times A_i
$$

$$
\text{exitCosts} = \text{grossSalePrice} \times \frac{\text{exitCostPct}}{100}
$$

$$
\text{netSaleProceeds} = \text{grossSalePrice} - \text{exitCosts}
$$

### 11.2 Rental income

If the user supplies a rent yield:

$$
\text{annualRent} = \text{buyPrice} \times \frac{\text{rentYieldPct}}{100}
$$

For apartments and builder floors where no yield is supplied, a rent model (trained separately in `notebooks/notebooks/rent/`) estimates monthly rent in log-space:

$$
\text{monthlyRent} = \exp(\hat{y}_{\text{rent}}) - 1
$$

$$
\text{annualRent} = \text{monthlyRent} \times 12
$$

Total rental income over holding period:

$$
\text{rentalIncomeTotal} = \text{annualRent} \times \text{years}
$$

### 11.3 Holding costs

$$
\text{holdingCostsTotal} = \text{buyPrice} \times \frac{\text{annualHoldingCostPct}}{100} \times \text{years}
$$

### 11.4 Profit and returns

$$
\text{netProfit} = \text{netSaleProceeds} + \text{rentalIncomeTotal} - \text{holdingCostsTotal} - \text{totalInvested}
$$

$$
\text{ROI\%} = \frac{\text{netProfit}}{\text{totalInvested}} \times 100
$$

$$
\text{payoffMultiple} = \frac{\text{netSaleProceeds} + \text{rentalIncomeTotal} - \text{holdingCostsTotal}}{\text{totalInvested}}
$$

$$
\text{CAGR\%} = \left(\text{payoffMultiple}^{1/\text{years}} - 1\right) \times 100
$$

---

## 12. Locality Intelligence Profile

In addition to property-level forecasts, the system produces a full intelligence brief for any locality:

### 12.1 Affordability score

$$
\text{affordability} = \text{clip}\!\left(\frac{\text{circleRate}}{\text{medianPpsf}} \times 50,\; 0,\; 100\right)
$$

Score of 50 = ppsf equals circle rate (fairly priced by government floor). Score of 100 = ppsf is half the circle rate (below floor, very cheap). Score approaching 0 = ppsf far exceeds circle rate (premium zone).

### 12.2 Rental yield estimate

Base yield by segment:

| Segment | Base Yield |
|---------|-----------|
| Builder Floor | 3.2% |
| Apartment | 2.8% |
| Plot | 1.5% |

Adjusted for market premium:

$$
\text{rentYield} = \text{baseYield} - 0.5 \times \text{clip}\!\left(\frac{\text{ppsf}}{\text{circleRate}} - 1,\; -1,\; 2\right)
$$

Higher-premium localities have their yield trimmed (investors chase capital gains, rental yield compresses).

### 12.3 QoQ volatility

Standard deviation of quarter-over-quarter returns from the locality forecast series:

$$
\sigma_{\text{qoq}} = \text{std}\!\left(\frac{p_t - p_{t-1}}{p_{t-1}} \times 100\right)
$$

### 12.4 Competing localities

The five most similar localities (by closest median ppsf within the same city) are surfaced as alternatives, helping investors compare options at the same price point.

---

## 13. XAI — Explainable Prediction Drivers

Every valuation is accompanied by 5–6 scored driver cards that explain what is pushing the price up or down.

### 13.1 Driver scoring scale

Each driver is scored on a 5-point integer scale:

| Score | Meaning |
|-------|---------|
| +2 | Strong positive driver |
| +1 | Mild positive driver |
| 0 | Neutral |
| -1 | Mild negative driver |
| -2 | Strong negative driver |

### 13.2 Location / Market Zone driver

Based on circle rate relative to segment median:

$$
\text{crRatio} = \frac{\text{circleRate}}{\text{ncr\_median\_cr}}
$$

| crRatio | Score |
|---------|-------|
| ≥ 2.0 | +2 (premium zone) |
| ≥ 1.3 | +1 (above-average) |
| ≥ 0.7 | 0 (typical) |
| ≥ 0.4 | -1 (budget/emerging) |
| < 0.4 | -2 (peripheral) |

### 13.3 Circle-Rate Alignment driver

Based on `pred_ratio` = predicted price / circle rate (BF/APT) or `ppsf / circle_rate` (fallback):

| predRatio | Score |
|-----------|-------|
| ≥ 1.5 | +2 (strong demand above floor) |
| ≥ 1.2 | +1 (healthy premium) |
| ≥ 0.85 | 0 (aligned with floor) |
| ≥ 0.65 | -1 (slight discount) |
| < 0.65 | -2 (deep discount) |

### 13.4 Area Impact driver

Based on area relative to segment median:

$$
\text{areaRatio} = \frac{A_i}{\text{ncr\_median\_area}}
$$

| areaRatio | Score |
|-----------|-------|
| ≥ 2.0 | +2 (very large) |
| ≥ 1.4 | +1 (above average) |
| ≥ 0.7 | 0 (typical) |
| ≥ 0.4 | -1 (compact) |
| < 0.4 | -2 (very small) |

### 13.5 Road Connectivity driver

For **plots**: based on nearest distance to NH/SH/MDR highway:

$$
d_{\min} = \min(\text{NH km},\; \text{SH km},\; \text{MDR km})
$$

| $d_{\min}$ | Score |
|-----------|-------|
| < 1.5 km | +2 |
| < 5.0 km | +1 |
| < 12.0 km | 0 |
| < 20.0 km | -1 |
| ≥ 20.0 km | -2 |

Road width modifies the score ±1.

For **BF/APT**: based on main-road frontage flag and Voronoi cluster distance.

### 13.6 Property Quality driver

Aggregates individual binary amenity flags:
- Luxury/high segment: +1 or +2
- Parking, gated community, pool, park-facing: +1 each
- New construction vs above 20 years: ±1
- High floor level (8th+): +1 (APT only)
- Irregular plot shape (Plot): -1

### 13.7 Growth Potential driver (where available)

Based on the locality's 1-year YoY forecast:

$$
\text{yoy\%} = \text{LocalityYoY from Section 8.2}
$$

| yoy% | Score |
|------|-------|
| > 15% | +2 |
| > 8% | +1 |
| > 2% | 0 |
| > -2% | -1 |
| ≤ -2% | -2 |

---

## 14. Formula Consistency Validation

At every API call, the backend recomputes both formula variants against the stored forecast and reports the error:

```
linearFormula : F_i,t = [rho_t*L_i + (1-rho_t)*P_i] * I_l,t
logFormula    : V_i,t = P_i * I_l,t * (L_i/P_i)^rho_t
```

| Metric | Description |
|--------|-------------|
| `linearMeanAbsError` | MAE between linear formula and stored forecast |
| `linearMaxAbsError` | Max absolute error (linear) |
| `logMeanAbsError` | MAE between geometric formula and stored forecast |
| `logMaxAbsError` | Max absolute error (geometric) |

Both errors should be near zero for a clean system state. Large values indicate a formula version mismatch between the offline trainer and the serving code.

---

## 15. Serving-Time Valuation (API)

Before a property enters the forecast pipeline, the API first estimates its current fair value using the segment-specific ML model.

### 15.1 Builder Floor and Apartment

The models predict a **log ratio** over circle rate:

$$
\hat{y} = \text{model}(\mathbf{x})
$$

$$
\text{predRatio} = \exp(\hat{y}) - 1
$$

$$
\widehat{\text{ppsf}} = \text{predRatio} \times \text{circleRate}
$$

$$
\widehat{\text{totalPrice}} = \widehat{\text{ppsf}} \times A
$$

Geospatial features fed to these models:
- **Voronoi cell** (KMeans-partitioned map cells): `vor_cell_0`, …, `vor_cell_K`
- **Distance to Voronoi seed**: `voronoi_dist_to_seed` (Euclidean km)
- **Target-encoded locality**: `locality_target_encoding`

### 15.2 Plot

The plot model predicts log-price-per-sqft directly:

$$
\widehat{\text{ppsf}} = \exp(\hat{y}_{log\_ppsf}) - 1
$$

Geospatial features fed to the plot model:
- **Spatial cluster** (KMeans): `c_0`, …, `c_K` (one-hot) + `dist_to_center`
- **Road distances**: `closest_distance_NH_km`, `closest_distance_SH_km`, `closest_distance_MDR_km`

### 15.3 Road distance computation

For each road class (NH, SH, MDR), road geometries are loaded from GeoJSON. For each line segment:

$$
K_{lat} = 110.574, \quad K_{lon} = 111.320 \times \cos(\phi)
$$

$$
t = \text{clip}\!\left(-\frac{x_1 dx + y_1 dy}{dx^2 + dy^2},\; 0,\; 1\right)
$$

$$
d = \sqrt{(x_1 + t \cdot dx)^2 + (y_1 + t \cdot dy)^2}
$$

The minimum $d$ across all segments of the road class is the feature value, converted to km.

---

## 16. Model Performance Benchmarks

### Apartment locality trend model
- **Training rows:** 18,059 | **Test rows:** 2,024
- **Localities with trend data:** 609 (out of 1,936 property localities, 1,054 mapped)
- **Total property-quarter forecasts generated:** 1,354,520 across 66,770 properties

### Builder Floor locality trend model
- **Training rows:** 11,716 | **Test rows:** 1,960
- **Localities with trend data:** 575 (out of 1,722 property localities, 790 mapped)
- **Total property-quarter forecasts generated:** 841,960 across 41,430 properties

### Rho distribution (Apartment)
- Mean $\rho_0$: **0.324** — indicating that on average, the listing price contributes about 32% of forecast influence at $t=0$, decaying to ~3.8% by quarter 20.
- The narrow mean/median spread (0.324 vs 0.322) indicates a symmetrically distributed population with few extreme outliers.

---

## 17. Key Files and Code Locations

| Purpose | File |
|---------|------|
| Core forecast service (loading, context, yoy) | `real_estate/utils/forecast_intelligence.py` |
| FastAPI endpoints (buy score, ROI, XAI, valuation) | `api/main.py` |
| Streamlit forecast dashboard | `fore_app.py` |
| Plot data transformation and cleaning | `real_estate/components/plot_data_transformation.py` |
| Plot model training pipeline | `real_estate/components/plot_model_trainer.py` |
| Apartment/BF data transformation | `real_estate/components/data_transformation.py` |
| Circle rate matching | `real_estate/utils/circle_rate_matcher.py` |
| Apartment model summary | `opt/apt/model_summary.md` |
| Builder Floor model summary | `opt/builder_floor/model_summary.md` |
| Apartment metrics | `opt/apt/metrics.json` |
| Apartment rho details | `opt/apt/rho_details.csv` |
| Apartment locality forecasts | `opt/apt/future_forecasts.csv` |
| Apartment property forecasts | `opt/apt/property_forecasts_flodata_new.csv` |

---

## 18. Symbol Glossary

| Symbol | Meaning |
|--------|---------|
| $L_i$ | Listing price per sqft for property $i$ |
| $P_i$ | Model/fair price per sqft for property $i$ |
| $A_i$ | Area of property $i$ in sqft |
| $C_i$ | Comparable support score (0–1) |
| $U_i$ | Uniqueness score (0–1) |
| $M_i$ | Model confidence score (0–1) |
| $\rho_0$ | Initial blending weight (listing trust) |
| $\rho_t$ | Decayed blending weight at quarter $t$ |
| $\gamma$ | Decay factor = 0.90 per quarter |
| $I_{l,t}$ | Cumulative locality growth index at quarter $t$ |
| $V_{i,t}$ | Forecast price per sqft for property $i$ at quarter $t$ |
| $F_{i,t}$ | Linear formula forecast (equivalent to $V_{i,t}$) |
| $\hat{p}_{l,t}$ | Predicted locality median ppsf at quarter $t$ |
| $\Delta\%$ | Valuation gap = $(L_i - P_i)/P_i \times 100$ |
| $\sigma_{\text{qoq}}$ | Standard deviation of quarter-over-quarter % returns |
| $t$ | Quarter index (0 = current, 1 = next quarter, …, 19 = 5 years out) |

---

## Appendix: End-to-End Data Flow Summary

```
inputs/
  apartment_with_pi.csv          ← L_i, P_i per property
  interpolated_growth_trend/     ← quarterly ppsf history per locality

OFFLINE TRAINING (FloData trainer notebook)
  ├── Engineer lag features from historical panels
  ├── Train RandomForest ensemble → locality trend model
  ├── Roll forward 20 quarters → I_{l,t} for all localities
  ├── Compute C_i, U_i, M_i → ρ_0 per property
  ├── Apply: V_{i,t} = P_i × I_{l,t} × (L_i/P_i)^{ρ_t}
  └── Write → opt/{segment}/
        future_forecasts.csv
        property_forecasts_flodata.csv
        rho_details.csv
        metrics.json

API (api/main.py) at SERVING TIME
  ├── GET /forecast/context
  │     Load property forecast rows
  │     Compute YoY chain, locality YoY
  │     Validate linear vs log formula
  │     Return: series, kpis, rho, distribution, quarter table
  │
  ├── POST /buy-decision
  │     valuation gap + upside + yoy + volatility
  │     → weighted score → Buy / Watch / Avoid
  │
  ├── POST /roi-projection
  │     buy price + exit price + rent + holding costs
  │     → netProfit, ROI%, CAGR%
  │
  ├── GET /forecast/locality-intelligence
  │     affordability, rental yield, volatility, trend
  │     → locality profile
  │
  └── POST /predict (BF / APT / Plot)
        ML model → predRatio → ppsf → total price
        + XAI driver cards
```
