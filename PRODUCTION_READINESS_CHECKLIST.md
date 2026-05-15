# Production Readiness Checklist

Date: 2026-05-06
Scope: Forecast Intelligence integration plus core prediction UX.

## 1) Service Health
- Backend API reachable at /meta/options and /forecast/* endpoints.
- Frontend can load metadata and render all tabs.
- Forecast segment availability confirmed for:
  - builder-floor
  - apartment
  - plot

## 2) Forecast Acceptance Coverage
- /forecast/overview returns metrics and summary content for each segment.
- /forecast/localities returns suggestions for city-scoped queries.
- /forecast/property-ids returns deduplicated property IDs (order preserved).
- /forecast/context returns:
  - selectedPropertyId
  - quarterTable with forecast horizon rows
  - mathValidation payload (linear + log diagnostics)
  - series blocks for charts

## 3) Frontend UX Validation
- Forecast tab can:
  - choose segment
  - search and select locality
  - choose property ID
  - load deep-dive payload
- KPI cards render correctly.
- Quarter table renders correctly.
- Forecast trend charts render for:
  - selected property forecast
  - locality forecast
  - historical locality trend
  - locality median distribution
- Formula fit signal renders from API mathValidation values.

## 4) Data and Naming Hygiene
- Legacy filename standardized:
  - fore_app.py.py -> fore_app.py
- README references updated to standardized name.

## 5) Regression Checks to Run Before Deploy
- Python syntax compile:
  - python -m py_compile api/main.py real_estate/utils/forecast_intelligence.py
- Frontend production build:
  - cd frontend
  - npm run build
- API smoke script:
  - verify /forecast/segments, /forecast/overview, /forecast/context

## 6) Deployment Guardrails
- Keep opt/* artifacts versioned and immutable per release.
- Keep inputs/* schema stable or add explicit versioning.
- Monitor /forecast/context latency for high-cardinality localities.
- Log request parameters (segment/city/locality/property_id) for traceability.
- Add alerting for endpoint 5xx rate spikes.

## 7) Decision Intelligence APIs
- Buy decision endpoint:
  - GET /insights/buy-decision?segment=...&city=...&locality=...&property_id=...&hold_years=...
- ROI endpoint:
  - GET /insights/roi?segment=...&city=...&locality=...&property_id=...&hold_years=...&purchase_cost_pct=...&annual_holding_cost_pct=...&exit_cost_pct=...&rent_yield_pct=...
- UI integration:
  - Forecast tab now includes a Decision Intelligence panel with:
    - Should I Buy This?
    - Estimate ROI
    - Editable ROI assumptions
