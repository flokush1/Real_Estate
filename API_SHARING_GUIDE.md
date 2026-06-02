# NCR Real Estate API — Sharing Guide

> **Who to share this with**: clients, integration partners, frontend developers,
> LLM-agent builders, third-party analysts.
>
> **Who NOT to share with**: keep the internal/admin section private.

---

## Base URL

```
http://<your-host>:8000
```

Interactive docs (Swagger UI): `http://<your-host>:8000/docs`

---

## 1. Meta / Discovery (share freely)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness check — returns `{"status":"ok"}` |
| `GET` | `/meta/options` | All valid enum values (age, furnishing, facing, floor-level, etc.) |
| `GET` | `/meta/model-status` | Which models are loaded and ready |
| `GET` | `/localities?query=&city=` | Fuzzy-search locality names |
| `GET` | `/circle-rate?locality=&city=` | Circle rate (₹/sqft) for a locality |
| `GET` | `/geocode?locality=&city=` | Lat/lon for a locality name |

---

## 2. Price Prediction (share freely)

### Builder Floor

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict/builder-floor` | Sell price prediction + XAI explanation |
| `POST` | `/predict/builder-floor/all` | **Sell + Rent + Rental Yield** in one call |

**Minimal request body (`/predict/builder-floor`):**

```json
{
  "locality": "Rohini",
  "city": "Delhi",
  "bhk": 3,
  "area_sqft": 1200,
  "age": "0-5 years",
  "furnishing": "Semi-Furnished",
  "facing": "East",
  "floor_level": "Ground",
  "circle_rate": 85000
}
```

**Response includes**: `ppsf`, `total`, `predRatio`, `explanation` (7-driver XAI).

`/predict/builder-floor/all` additionally returns `rent_monthly`, `rent_annual`, `rental_yield_pct`.

---

### Apartment

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict/apartment` | Sell price prediction + XAI explanation |
| `POST` | `/predict/apartment/all` | **Sell + Rent + Rental Yield** in one call |

Same request body shape as builder floor (see `/meta/options` for valid enum values).

---

### Plot

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict/plot` | Plot price prediction + XAI explanation |

**Minimal request body:**

```json
{
  "locality": "Sector 15",
  "city": "Noida",
  "lat": 28.5850,
  "lon": 77.3300,
  "area_sqft": 200,
  "circle_rate": 55000,
  "usage_type": "Residential",
  "facing_direction": "East"
}
```

---

## 3. Property Intelligence — Fair Value Analysis (share freely)

These endpoints compare a **listed property** against the AI model price and return
a full investment-grade analysis: valuation, location quality, connectivity, safety,
and unit desirability scores.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/property-intelligence/analyze` | Builder floor fair-value analysis |
| `POST` | `/property-intelligence/analyze-apartment` | Apartment fair-value analysis |
| `POST` | `/property-intelligence/analyze-plot` | Plot fair-value analysis |

**Extra required field**: `listing_price` (the price the seller is asking).

**Response includes**: valuation delta, PI score (0–100), investment signal
(Underpriced / Fair Value / Overpriced), and 7 sub-scores.

---

## 4. Forecast & Trend Intelligence (share freely)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/forecast/segments` | Available segments |
| `GET` | `/forecast/cities?segment=` | Cities with forecast data |
| `GET` | `/forecast/localities?segment=&city=` | Localities in a city |
| `GET` | `/forecast/property-ids?segment=&locality=` | Property IDs in a locality |
| `GET` | `/forecast/overview?segment=&city=&locality=` | Locality-level price trend |
| `GET` | `/forecast/context?segment=&city=&locality=&property_id=` | Full forecast context for a property |

---

## 5. Property Listing & Analytics (share freely)

> New in this release. Lets partners explore the property database and retrieve
> per-property forecast and blending metadata.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/properties/list?segment=&city=&locality=&limit=` | List properties with listed vs model ppsf |
| `GET` | `/properties/summary?segment=&property_id=` | Full property row for a specific ID |
| `GET` | `/properties/rho?segment=&property_id=` | Rho blending weights (BF/APT) |
| `GET` | `/properties/rho?segment=plot&locality=` | Rho weights (Plot, locality-level) |
| `GET` | `/properties/yoy?segment=&property_id=&years=5` | Annual YoY forecast for a property |

**`/properties/list` example response item:**

```json
{
  "property_id": "HO_19623346",
  "locality": "Nawada",
  "city": "Delhi",
  "area_sqft": 1080,
  "listed_ppsf": 10185,
  "model_ppsf": 9126,
  "delta_pct": 11.6,
  "bhk": 3
}
```

**`/properties/yoy` example response:**

```json
{
  "segment": "builder-floor",
  "property_id": "HO_19623346",
  "locality": "Nawada",
  "yoy": [
    {"label":"Y+1","start_date":"2026-07-01","end_date":"2027-04-01","anchor_ppsf":9057,"target_ppsf":9659,"yoy_pct":6.64},
    {"label":"Y+2","start_date":"2027-04-01","end_date":"2028-04-01","anchor_ppsf":9659,"target_ppsf":11056,"yoy_pct":14.46},
    ...
  ]
}
```

---

## 6. Insights (share freely)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/insights/buy-decision?segment=&city=&locality=&property_id=` | Buy / Hold / Wait recommendation |
| `GET` | `/insights/roi?segment=&city=&locality=&property_id=` | ROI projection |

---

## 7. Market Intelligence (share freely)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/market-intelligence/segments` | Available segments |
| `GET` | `/market-intelligence/cities?segment=` | Cities with demand/supply data |
| `GET` | `/market-intelligence/localities?segment=&city=` | Localities with market data |
| `GET` | `/market-intelligence/context?segment=&city=&locality=` | Demand/supply diagnostics |

---

## 8. MCP Server (for LLM / AI agents)

The MCP server exposes **all** of the above endpoints as tools for Claude Desktop,
Cursor, chat.py, and any MCP-compatible LLM client.

**STDIO transport (Claude Desktop / Cursor):**

```json
// claude_desktop_config.json
{
  "mcpServers": {
    "ncr-real-estate": {
      "command": "python",
      "args": ["C:/path/to/real_estate/mcp_server.py"],
      "cwd":  "C:/path/to/real_estate"
    }
  }
}
```

**HTTP transport (MCP Inspector / testing):**

```bash
python mcp_server.py --transport http --port 9000
# MCP endpoint: http://127.0.0.1:9000/mcp/
```

---

## ── Internal / Admin (DO NOT share externally) ──

| Endpoint | Reason |
|----------|--------|
| `GET /meta/model-registry` | Exposes internal versioning paths |
| `GET /locality-encoding` | Internal model feature encoding |
| `GET /road-distances` | Raw geospatial computation endpoint |

---

## Authentication

> The current API has **no authentication**. For production sharing, deploy
> behind a reverse proxy (nginx / API Gateway) with at minimum:
> - API key header (`X-API-Key`)
> - Rate limiting
> - HTTPS termination

---

## Common Query Parameters

| Parameter | Type | Notes |
|-----------|------|-------|
| `segment` | string | `builder-floor` \| `apartment` \| `plot` |
| `city` | string | `Delhi` \| `Gurgaon` \| `Noida` \| `Faridabad` \| `Ghaziabad` |
| `locality` | string | Fuzzy-matched — use `/localities?query=` to find exact names |
| `property_id` | string | From `/properties/list` (format: `HO_XXXXXXXX` for BF/APT) |

---

## Quick Start (curl)

```bash
# 1. Health check
curl http://localhost:8000/health

# 2. Find a locality
curl "http://localhost:8000/localities?query=rohini&city=Delhi"

# 3. Predict builder floor sell price
curl -X POST http://localhost:8000/predict/builder-floor \
  -H "Content-Type: application/json" \
  -d '{"locality":"Rohini","city":"Delhi","bhk":3,"area_sqft":1200,
       "age":"0-5 years","furnishing":"Semi-Furnished","facing":"East",
       "floor_level":"Ground","circle_rate":85000}'

# 4. Get sell + rent + yield in one call
curl -X POST http://localhost:8000/predict/builder-floor/all \
  -H "Content-Type: application/json" \
  -d '{"locality":"Rohini","city":"Delhi","bhk":3,"area_sqft":1200,
       "age":"0-5 years","furnishing":"Semi-Furnished","facing":"East",
       "floor_level":"Ground","circle_rate":85000}'

# 5. List properties in a locality
curl "http://localhost:8000/properties/list?segment=builder-floor&city=Delhi&locality=Rohini&limit=10"

# 6. Get 5-year YoY forecast for a property
curl "http://localhost:8000/properties/yoy?segment=builder-floor&property_id=HO_19623346&years=5"
```
