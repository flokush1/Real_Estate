# NCR Real Estate AI System: Math and System Context

## 1) What this system is about

This system is an end-to-end real estate intelligence platform for Delhi NCR. It has three major functions:

1. Online valuation API for Builder Floor, Apartment, and Plot.
2. Offline training pipeline (especially productionized for Plot) with geospatial features.
3. Forecast intelligence and investment analytics (buy decision and ROI projections).

In plain terms, it estimates fair value per sqft, computes total property value, projects future price paths, and converts those signals into actionable investment metrics.

---

## 2) Scope of this math document

This document inventories the math currently used in active production code paths, plus closely integrated forecast logic:

- API serving math and geospatial calculations.
- Plot transformation and trainer math.
- Forecast intelligence formulas.
- Buy decision and ROI scoring formulas.
- Supporting similarity and matching scores used in localization and circle-rate logic.

Research notebooks contain additional experimental math, but the equations below are focused on the deployed code paths.

---

## 3) Core symbols used below

- $L_i$: listing price per sqft for property $i$.
- $P_i$: model/fair price per sqft for property $i$.
- $I_{l,t}$: locality growth multiplier for locality $l$ at forecast quarter $t$.
- $\rho_0$: initial blend weight.
- $\rho_t$: decayed blend weight at quarter $t$.
- $F_{i,t}$: forecast price per sqft at quarter $t$.
- $A$: area in sqft.

---

## 4) Data normalization and unit conversion math

### 4.1 Area unit conversion to sqft

All area values are standardized using multiplicative conversion factors, e.g.:

- sq-ft: $\times 1.0$
- sq-yrd: $\times 9.0$
- sq-m: $\times 10.7639$
- acre: $\times 43560$

Generic formula:

$$
\text{area\_sqft} = \text{area\_value} \times \text{unit\_factor}
$$

This appears in both generic and plot-focused transformations.

### 4.2 Derived unit price

$$
\text{price\_per\_sqft} = \frac{\text{total\_price}}{\text{plot\_area\_sqft}}
$$

---

## 5) Rule-based filters and quality bounds

The system applies explicit numeric constraints to remove noisy or implausible points.

Examples from production plot path:

- Keep rows with positive values:
  - $\text{price\_per\_sqft} > 0$
  - $\text{plot\_area} > 0$
  - $\text{circle\_rate} > 0$
- Ratio sanity window:

$$
0.7 < \frac{\text{price\_per\_sqft}}{\text{circle\_rate}} < 25
$$

- Additional range checks in transformation:
  - plot area bounded window.
  - price per sqft bounded window.

These steps are mathematical gates that reduce outlier-driven instability before model fitting.

---

## 6) Missing-value imputation math

### 6.1 KNN imputation (weighted by distance)

For fields like road width and facing direction, KNN models are trained on available rows and used to infer missing rows.

- Numeric preprocessing: median imputation + z-score scaling.
- Categorical preprocessing: mode imputation + one-hot encoding.
- KNN uses distance weighting:

$$
\hat{y}(x) = \frac{\sum_{j \in N_k(x)} w_j y_j}{\sum_{j \in N_k(x)} w_j}, \quad w_j \propto \frac{1}{d(x,x_j)}
$$

Classification variant predicts category mode under weighted neighborhood influence.

### 6.2 Nearest-neighbor Haversine fallback for missing circle rate

When circle rate is missing but coordinates are known, donor rows are selected by nearest geodesic distance (BallTree with Haversine metric).

Conceptually:

$$
\text{circle\_rate}_{\text{missing row}} \leftarrow \text{circle\_rate}_{\text{nearest donor in lat/lon space}}
$$

Donor pool priority:

1. Same city + same mapped property type.
2. Same city.
3. Same property type.
4. Any known circle-rate row.

---

## 7) Geospatial feature math

### 7.1 KMeans spatial clustering

Coordinates are standardized first, then clustered with KMeans.

- Cluster assignment:

$$
c_i = \arg\min_k \|x_i - \mu_k\|_2
$$

- Distance to assigned center:

$$
\text{dist\_to\_center}_i = \|x_i - \mu_{c_i}\|_2
$$

- One-hot cluster features are generated: $c_0, c_1, ..., c_K$.

### 7.2 Voronoi-like cell features at serving time

For BF/Apartment and some notebook experiments, a KMeans partition is used as Voronoi cells.

- Predicted cell ID from $(lat, lon)$.
- Euclidean distance to seed center.
- One-hot features for all cells.

### 7.3 Nearest road distance (MDR/SH/NH)

Road geometries are represented as line segments. For each segment endpoint pair, point-to-segment distance is computed in locally projected km coordinates.

Local projection constants:

$$
K_{lat} = 110.574, \quad K_{lon} = 111.320 \cdot \cos(\text{lat in radians})
$$

Projection parameter per segment:

$$
t = \text{clip}\left(-\frac{x_1 dx + y_1 dy}{dx^2 + dy^2}, 0, 1\right)
$$

Distance to nearest point on segment is then minimized over all segments in each road class.

Output features:

- closest_distance_MDR_km
- closest_distance_SH_km
- closest_distance_NH_km

---

## 8) Target transformation and value reconstruction

Plot model is trained on log-transformed target:

$$
y = \log(1 + \text{price\_per\_sqft})
$$

Model prediction back-transform:

$$
\widehat{\text{ppsf}} = \exp(\hat{y}) - 1
$$

Total price:

$$
\widehat{\text{total\_price}} = \widehat{\text{ppsf}} \cdot A
$$

Trainer also computes total price from log variables using $\exp(\cdot)-1$ on both predicted sqft price and log area where relevant.

---

## 9) Model evaluation metrics

The production trainer uses standard regression metrics on reconstructed total price:

### 9.1 MAE

$$
\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
$$

### 9.2 MAPE

$$
\text{MAPE} = \frac{1}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right|
$$

### 9.3 R2

$$
R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}
$$

Cross-validation and hyperparameter search are done with these reconstructed-price scorers.

---

## 10) Forecast intelligence math

### 10.1 Rho construction

Base blend weight:

$$
\rho_0 = \text{clip}(0.25 + 0.30C_i + 0.20U_i - 0.25M_i,\; 0.10,\; 0.70)
$$

Where:

- $C_i$: comparable support.
- $U_i$: uniqueness.
- $M_i$: model confidence.

### 10.2 Decay over forecast horizon

$$
\rho_t = \rho_0 \cdot 0.90^t
$$

### 10.3 Linear blend forecast (artifact-aligned)

$$
F_{i,t} = [\rho_t L_i + (1-\rho_t)P_i] \cdot I_{l,t}
$$

### 10.4 Log/geometric equivalent checked for validation

$$
\log(V_{i,t}) = \log(P_i) + \log(I_{l,t}) + \rho_t(\log(L_i)-\log(P_i))
$$

Equivalent geometric expression used in validation code:

$$
V_{i,t} = P_i \cdot I_{l,t} \cdot \left(\frac{L_i}{P_i}\right)^{\rho_t}
$$

### 10.5 Formula-consistency error tracking

The backend computes absolute errors against stored forecast values for both formulas:

- linearMeanAbsError, linearMaxAbsError
- logMeanAbsError, logMaxAbsError

---

## 11) Trend and YoY analytics math

### 11.1 Property YoY sequence

From current anchor date to yearly checkpoints:

$$
\text{YoY\%} = \left(\frac{P_{target}}{P_{anchor}} - 1\right) \cdot 100
$$

### 11.2 Locality YoY from forecast medians

Locality forecast is aggregated by date median and then compared year-over-year using same ratio form.

### 11.3 Locality distribution statistics

For each date, across locality properties:

- median
- mean
- min/max
- quantiles $p25$, $p75$

---

## 12) Buy decision scoring math

The buy-decision endpoint converts forecast context into factor scores.

### 12.1 Inputs

- valuation gap percent:

$$
\Delta\% = \left(\frac{L_i - P_i}{P_i}\right) \cdot 100
$$

- expected upside to selected horizon:

$$
\text{upside\%} = \left(\frac{F_{horizon}}{L_i} - 1\right) \cdot 100
$$

- average YoY percent from property series.
- volatility as standard deviation of quarter-over-quarter returns:

$$
\text{QoQ\%}_t = \left(\frac{F_t - F_{t-1}}{F_{t-1}}\right) \cdot 100,
\quad \sigma_{qoq} = \text{std}(\text{QoQ\%})
$$

### 12.2 Score components and final score

- valuation score: $\text{clip}(50 + (-\Delta\%\cdot 2), 0, 100)$
- growth score: $\text{clip}(50 + 3\cdot\text{avgYoY\%}, 0, 100)$
- upside score: $\text{clip}(50 + 1.5\cdot\text{upside\%}, 0, 100)$
- risk penalty: $\text{clip}(2\cdot\sigma_{qoq}, 0, 30)$

Overall:

$$
\text{overall} = \text{clip}(0.40\cdot val + 0.35\cdot growth + 0.25\cdot upside - riskPenalty, 0, 100)
$$

Recommendation bands:

- Buy if overall >= 68
- Watch if 50 <= overall < 68
- Avoid otherwise

Confidence is also computed from data depth and fallback flags and then clipped.

---

## 13) ROI projection math

The ROI endpoint builds a cashflow model from buy price, holding period, forecast exit price, and cost assumptions.

### 13.1 Price components

$$
\text{buyPrice} = L_i \cdot A
$$

$$
\text{purchaseCosts} = \text{buyPrice} \cdot \frac{purchaseCostPct}{100}
$$

$$
\text{grossSalePrice} = F_{horizon} \cdot A
$$

$$
\text{exitCosts} = \text{grossSalePrice} \cdot \frac{exitCostPct}{100}
$$

$$
\text{netSaleProceeds} = \text{grossSalePrice} - \text{exitCosts}
$$

### 13.2 Rental and holding cost

If manual rent yield is provided:

$$
\text{annualRent} = \text{buyPrice} \cdot \frac{rentYieldPct}{100}
$$

Else for apartment and builder-floor, a rent model estimates monthly rent in log-space then back-transforms with $\exp(\cdot)-1$.

Total rental over holding years:

$$
\text{rentalIncomeTotal} = \text{annualRent} \cdot years
$$

Holding costs:

$$
\text{holdingCostsTotal} = \text{buyPrice} \cdot \frac{annualHoldingCostPct}{100} \cdot years
$$

### 13.3 Net profit, ROI, payoff, CAGR

$$
\text{totalInvested} = \text{buyPrice} + \text{purchaseCosts}
$$

$$
\text{netProfit} = \text{netSaleProceeds} + \text{rentalIncomeTotal} - \text{holdingCostsTotal} - \text{totalInvested}
$$

$$
\text{ROI\%} = \frac{\text{netProfit}}{\text{totalInvested}} \cdot 100
$$

$$
\text{payoffMultiple} = \frac{\text{netSaleProceeds} + \text{rentalIncomeTotal} - \text{holdingCostsTotal}}{\text{totalInvested}}
$$

$$
\text{CAGR\%} = \left(\text{payoffMultiple}^{1/years} - 1\right) \cdot 100
$$

Verdict bands are then assigned from ROI percentage.

---

## 14) Serving-time valuation formulas by segment

### 14.1 Builder Floor and Apartment serving

Models output a log ratio; API transforms and scales by circle rate:

$$
\text{predRatio} = \exp(\hat{y}) - 1
$$

$$
\widehat{\text{ppsf}} = \text{predRatio} \cdot \text{circleRate}
$$

$$
\widehat{\text{total}} = \widehat{\text{ppsf}} \cdot A
$$

### 14.2 Plot serving

$$
\widehat{\text{ppsf}} = \exp(\hat{y}_{log\_ppsf}) - 1,
\quad
\widehat{\text{total}} = \widehat{\text{ppsf}} \cdot A
$$

---

## 15) Similarity and matching score math

### 15.1 Locality suggestion score

A composite score combines:

- Sequence similarity ratio.
- substring and prefix bonuses.
- token overlap bonus.

This is used to rank locality suggestions in forecast endpoints.

### 15.2 Circle-rate fuzzy matcher threshold

Circle-rate locality matching uses RapidFuzz token_sort_ratio with cutoff 82 after exact and alias fallback passes.

---

## 16) Important implementation files

Primary references for the formulas above:

- README.md
- api/main.py
- real_estate/components/plot_data_transformation.py
- real_estate/components/plot_model_trainer.py
- real_estate/components/data_transformation.py
- real_estate/utils/forecast_intelligence.py
- real_estate/utils/circle_rate_matcher.py

---

## 17) Practical interpretation

The system combines:

1. Statistical ML prediction (log-scale regression + geospatial features).
2. Deterministic economic blending (rho-decayed listing/model fusion).
3. Investment math (cashflows, ROI, CAGR, volatility penalties).

So it is not only a price predictor; it is a valuation + forecast + decision-support engine.
