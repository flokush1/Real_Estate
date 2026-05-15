import React, { useEffect, useMemo, useState } from 'react'
import './App.css'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'

const TAB_KEYS = {
  BF: 'builder-floor',
  APT: 'apartment',
  PLOT: 'plot',
  FORECAST: 'forecast',
  MI: 'market-intelligence',
}

const defaultOptions = {
  cities: ['Delhi', 'Noida', 'Gurgaon', 'Faridabad', 'Ghaziabad', 'Greater Noida', 'Jaipur'],
  ageCategories: ['10 to 20 years', '5 to 10 years', 'Above 20 years', 'Less than 5 years', 'New Construction'],
  furnishingCategories: ['Furnished', 'Semi-Furnished', 'Unfurnished'],
  facingCategories: ['East', 'North', 'North-East', 'North-West', 'South', 'South-East', 'South-West', 'West'],
  floorLevels: ['Low (Ground - 1st)', 'Medium (2nd - 7th)', 'High (8th+)'],
  aptPropertySegments: ['Base', 'Mid', 'High', 'Luxury'],
  plotUsageOptions: ['Residential', 'Commercial'],
  plotFacingOptions: ['North', 'South', 'East', 'West', 'North East', 'North West', 'South East', 'South West', 'Central'],
  plotRoadWidthOptions: ['Upto 9m', '9m to 18m', '18m+'],
  forecastSegments: [
    { key: 'builder-floor', label: 'Builder Floor', available: 'true' },
    { key: 'apartment', label: 'Apartment / Flat', available: 'true' },
    { key: 'plot', label: 'Plot', available: 'true' },
  ],
}

const defaultBfForm = {
  bhk: 3,
  area_sqft: 1200,
  bathrooms: 2,
  balconies: 1,
  age: defaultOptions.ageCategories[1],
  furnishing: defaultOptions.furnishingCategories[1],
  facing: defaultOptions.facingCategories[1],
  circle_rate: 11891.59,
  is_parking: 1,
  is_pool: 0,
  is_main_road: 0,
  is_garden_park: 1,
  is_gated: 1,
  is_corner: 0,
  lat: 28.6139,
  lon: 77.209,
}

const defaultAptForm = {
  ...defaultBfForm,
  floor_level: defaultOptions.floorLevels[1],
  is_ground: 0,
  is_top: 0,
  property_segment: defaultOptions.aptPropertySegments[1],
}

const defaultPlotForm = {
  area_sqft: 1800,
  usage_type: defaultOptions.plotUsageOptions[0],
  facing_direction: defaultOptions.plotFacingOptions[0],
  circle_rate: 11891.59,
  is_park_facing: 0,
  is_corner: 0,
  is_rectangular: 1,
  is_gated: 1,
  has_boundary_wall: 1,
  road_width_upto_9m: 0,
  road_width_9_to_18m: 1,
  road_width_18_plus: 0,
  lat: 28.6139,
  lon: 77.209,
  closest_distance_MDR_km: 0,
  closest_distance_SH_km: 0,
  closest_distance_NH_km: 0,
}

function fmtPrice(value) {
  if (value == null || Number.isNaN(value)) return '--'
  if (value >= 1e7) return `INR ${(value / 1e7).toFixed(2)} Cr`
  if (value >= 1e5) return `INR ${(value / 1e5).toFixed(2)} L`
  return `INR ${Math.round(value).toLocaleString('en-IN')}`
}

function fmtNumber(value, digits = 2) {
  if (value == null || Number.isNaN(value)) return '--'
  return Number(value).toLocaleString('en-IN', {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  })
}

function fmtPct(value, digits = 2) {
  if (value == null || Number.isNaN(value)) return '--'
  return `${value >= 0 ? '+' : ''}${fmtNumber(value, digits)}%`
}

function buildSparklinePath(points, width = 360, height = 120, padding = 12) {
  if (!Array.isArray(points) || points.length < 2) return ''

  const min = Math.min(...points)
  const max = Math.max(...points)
  const span = Math.max(max - min, 1e-9)
  const innerW = width - padding * 2
  const innerH = height - padding * 2

  return points
    .map((y, idx) => {
      const xPos = padding + (idx / (points.length - 1)) * innerW
      const yPos = padding + (1 - (y - min) / span) * innerH
      return `${idx === 0 ? 'M' : 'L'} ${xPos.toFixed(2)} ${yPos.toFixed(2)}`
    })
    .join(' ')
}

function TrendCard({ title, points, color, suffix = 'INR/sqft', digits = 2 }) {
  const valid = Array.isArray(points) ? points.filter((v) => Number.isFinite(v)) : []
  const current = valid.length ? valid[valid.length - 1] : null
  const start = valid.length ? valid[0] : null
  const delta = current != null && start != null ? current - start : null
  const deltaPct = current != null && start != null && start !== 0 ? ((current - start) / start) * 100 : null
  const path = buildSparklinePath(valid)

  return (
    <article className="trend-card">
      <div className="trend-head">
        <h4>{title}</h4>
        <div className="trend-current">{current == null ? '--' : `${fmtNumber(current, digits)} ${suffix}`}</div>
      </div>

      <svg className="trend-chart" viewBox="0 0 360 120" role="img" aria-label={title}>
        <line x1="12" y1="108" x2="348" y2="108" className="trend-axis" />
        {path ? <path d={path} className="trend-line" style={{ stroke: color }} /> : null}
      </svg>

      <div className={`trend-delta ${delta != null && delta < 0 ? 'neg' : 'pos'}`}>
        {delta == null ? '--' : `${delta >= 0 ? '+' : ''}${fmtNumber(delta, digits)} (${fmtPct(deltaPct, 2)})`}
      </div>
    </article>
  )
}

function Field({ label, children, hint }) {
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      {children}
      {hint ? <span className="field-hint">{hint}</span> : null}
    </label>
  )
}

function MetricCard({ label, value, delta }) {
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      {delta ? <div className={`metric-delta ${delta.startsWith('-') ? 'neg' : 'pos'}`}>{delta}</div> : null}
    </div>
  )
}

// ─── Score bar: -2 (red) … 0 (grey) … +2 (green) ─────────────────────────
const SCORE_COLORS = {
  2:  { bar: '#2a7d4f', bg: '#dff5eb', label: 'Strong +' },
  1:  { bar: '#5aae78', bg: '#edf8f2', label: 'Positive' },
  0:  { bar: '#9e9e9e', bg: '#f5f5f5', label: 'Neutral'  },
  '-1': { bar: '#e08c4a', bg: '#fdf3e7', label: 'Negative' },
  '-2': { bar: '#c0392b', bg: '#fde8e3', label: 'Strong −' },
}

function ExplainPanel({ explanation }) {
  const [open, setOpen] = React.useState(false)
  if (!explanation) return null

  const { drivers = [], narrative, ppsfVsMedianPct, ncrReference } = explanation

  return (
    <section className="xai-panel">
      <button className="xai-toggle" onClick={() => setOpen((p) => !p)}>
        <span>AI Explanation</span>
        <span className="xai-toggle-icon">{open ? '▲' : '▼'}</span>
      </button>

      {open ? (
        <div className="xai-body">
          <p className="xai-narrative">{narrative}</p>

          {ncrReference ? (
            <div className="xai-ref-row">
              <span>Segment median: INR {fmtNumber(ncrReference.medianPpsf, 0)}/sqft</span>
              <span>Median circle rate: INR {fmtNumber(ncrReference.medianCr, 0)}/sqft</span>
              <span>Median area: {fmtNumber(ncrReference.medianArea, 0)} sqft</span>
              {ppsfVsMedianPct != null ? (
                <span className={ppsfVsMedianPct >= 0 ? 'xai-pos' : 'xai-neg'}>
                  {ppsfVsMedianPct >= 0 ? '+' : ''}{fmtNumber(ppsfVsMedianPct, 1)}% vs median
                </span>
              ) : null}
            </div>
          ) : null}

          <div className="xai-drivers">
            {drivers.map((d) => {
              const sc = String(d.score)
              const col = SCORE_COLORS[sc] || SCORE_COLORS[0]
              const barWidth = `${((d.score + 2) / 4) * 100}%`
              return (
                <div key={d.key} className="xai-driver" style={{ background: col.bg }}>
                  <div className="xai-driver-head">
                    <span className="xai-driver-label">{d.label}</span>
                    <span className="xai-driver-badge" style={{ background: col.bar }}>
                      {col.label}
                    </span>
                  </div>
                  <div className="xai-bar-track">
                    <div className="xai-bar-fill" style={{ width: barWidth, background: col.bar }} />
                    <div className="xai-bar-tick" />
                  </div>
                  <p className="xai-driver-detail">{d.detail}</p>
                </div>
              )
            })}
          </div>
        </div>
      ) : null}
    </section>
  )
}

function WhatIfPanel({ title, config, selectedFeature, onChangeFeature, value, onChangeValue, onRun, running, result }) {
  const selected = config.find((item) => item.key === selectedFeature) || config[0]
  const isCategoryComparison = selected.type === 'select' || selected.type === 'binary'

  return (
    <section className="whatif-block">
      <h4>{title}</h4>
      <div className="whatif-controls">
        <Field label="Feature">
          <select value={selectedFeature} onChange={(e) => onChangeFeature(e.target.value)}>
            {config.map((item) => (
              <option key={item.key} value={item.key}>
                {item.label}
              </option>
            ))}
          </select>
        </Field>

        {selected.type === 'number' ? (
          <Field label="What-If Value">
            <input
              type="number"
              min={selected.min}
              max={selected.max}
              step={selected.step || 1}
              value={value}
              onChange={(e) => onChangeValue(Number(e.target.value))}
            />
          </Field>
        ) : null}

        {isCategoryComparison ? <p className="whatif-note">All available values for this feature will be compared against the current baseline in one run.</p> : null}
      </div>

      <button className="btn ghost" onClick={onRun} disabled={running}>
        {running ? 'Running What-If...' : isCategoryComparison ? 'Compare All Values' : 'Run What-If'}
      </button>

      {result?.mode === 'single' ? (
        <div className="whatif-metrics">
          <MetricCard label="Baseline" value={fmtPrice(result.baselineTotal)} />
          <MetricCard
            label="What-If"
            value={fmtPrice(result.whatIfTotal)}
            delta={`${result.deltaTotal >= 0 ? '+' : ''}${fmtNumber(result.deltaTotal, 0)} (${result.deltaPct >= 0 ? '+' : ''}${fmtNumber(result.deltaPct)}%)`}
          />
          <MetricCard
            label="What-If INR/sqft"
            value={`INR ${fmtNumber(result.whatIfPpsf, 0)}`}
            delta={`${result.deltaPpsf >= 0 ? '+' : ''}${fmtNumber(result.deltaPpsf, 0)}`}
          />
        </div>
      ) : null}

      {result?.mode === 'categorical' ? (
        <>
          <div className="whatif-metrics whatif-metrics-single">
            <MetricCard label={`Baseline (${result.baselineLabel})`} value={fmtPrice(result.baselineTotal)} />
          </div>

          <div className="comparison-wrap">
            <table className="comparison-table">
              <thead>
                <tr>
                  <th>{result.featureLabel}</th>
                  <th>Total Price</th>
                  <th>Delta vs Baseline</th>
                  <th>Delta %</th>
                  <th>INR/sqft</th>
                  <th>Delta INR/sqft</th>
                </tr>
              </thead>
              <tbody>
                {result.comparisons.map((item) => (
                  <tr key={item.valueLabel} className={item.isBaseline ? 'baseline-row' : ''}>
                    <td>{item.valueLabel}{item.isBaseline ? ' (Baseline)' : ''}</td>
                    <td>{fmtPrice(item.total)}</td>
                    <td className={item.deltaTotal < 0 ? 'neg' : 'pos'}>{`${item.deltaTotal >= 0 ? '+' : ''}${fmtNumber(item.deltaTotal, 0)}`}</td>
                    <td className={item.deltaPct < 0 ? 'neg' : 'pos'}>{`${item.deltaPct >= 0 ? '+' : ''}${fmtNumber(item.deltaPct)}%`}</td>
                    <td>{`INR ${fmtNumber(item.ppsf, 0)}`}</td>
                    <td className={item.deltaPpsf < 0 ? 'neg' : 'pos'}>{`${item.deltaPpsf >= 0 ? '+' : ''}${fmtNumber(item.deltaPpsf, 0)}`}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : null}
    </section>
  )
}

async function apiGet(path) {
  const response = await fetch(`${API_BASE}${path}`)
  if (!response.ok) {
    throw new Error(`GET ${path} failed: ${response.status}`)
  }
  return response.json()
}

async function apiPost(path, payload) {
  const response = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!response.ok) {
    const msg = await response.text()
    throw new Error(msg || `POST ${path} failed: ${response.status}`)
  }
  return response.json()
}

function MarketIntelligenceTab({
  miSegment, setMiSegment,
  miCity, setMiCity,
  miLocalityQuery, setMiLocalityQuery,
  miLocalities,
  miSelectedLocality, setMiSelectedLocality,
  miContext, miLoading, miError,
  onLoad,
}) {
  const locality = (miSelectedLocality || miLocalityQuery).trim()
  const segmentOptions = [
    { id: 'apt', label: 'Apartment' },
    { id: 'builder_floor', label: 'Builder Floor' },
    { id: 'plot', label: 'Plot' },
  ]
  const cityOptions = ['Delhi', 'Noida', 'Gurgaon', 'Faridabad', 'Ghaziabad', 'Greater Noida', 'Jaipur']

  const d = miContext
  const kpis = d?.kpis || {}
  const idx = d?.indices || {}
  const series = d?.series || {}
  const drivers = d?.drivers || []

  const heatScore = idx.marketHeatIndex ?? 50
  const heatColor = heatScore >= 75 ? '#2a7d4f' : heatScore >= 60 ? '#5aae78' : heatScore >= 45 ? '#d9892b' : heatScore >= 30 ? '#e08c4a' : '#c0392b'

  const DRIVER_COLORS = {
    2:  { bar: '#2a7d4f', bg: '#dff5eb', label: 'Strong +' },
    1:  { bar: '#5aae78', bg: '#edf8f2', label: 'Positive' },
    0:  { bar: '#9e9e9e', bg: '#f5f5f5', label: 'Neutral'  },
    '-1': { bar: '#e08c4a', bg: '#fdf3e7', label: 'Negative' },
    '-2': { bar: '#c0392b', bg: '#fde8e3', label: 'Strong −' },
  }

  function MiniSparkline({ points, color = '#2a7d4f' }) {
    if (!Array.isArray(points) || points.length < 2) return null
    const vals = points.map((p) => (typeof p === 'object' ? p.value : p)).filter(Number.isFinite)
    if (vals.length < 2) return null
    const path = buildSparklinePath(vals, 200, 60, 6)
    return (
      <svg viewBox="0 0 200 60" className="market-sparkline" role="img">
        {path ? <path d={path} fill="none" stroke={color} strokeWidth="2" /> : null}
      </svg>
    )
  }

  function IndexGauge({ label, value, color }) {
    const pct = Math.max(0, Math.min(100, value ?? 50))
    return (
      <div className="market-index-card">
        <div className="market-index-label">{label}</div>
        <div className="market-index-value" style={{ color }}>{pct.toFixed(0)}<span className="market-index-unit"> / 100</span></div>
        <div className="market-gauge-track">
          <div className="market-gauge-fill" style={{ width: `${pct}%`, background: color }} />
        </div>
      </div>
    )
  }

  return (
    <>
      <h3>Market Intelligence</h3>
      <p className="li-desc">Demand-supply diagnostics, liquidity analysis, and price revision signals for any NCR locality.</p>

      {/* Controls */}
      <div className="forecast-controls">
        <Field label="Segment">
          <select value={miSegment} onChange={(e) => { setMiSegment(e.target.value); setMiSelectedLocality('') }}>
            {segmentOptions.map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}
          </select>
        </Field>
        <Field label="City">
          <select value={miCity} onChange={(e) => { setMiCity(e.target.value); setMiSelectedLocality(''); setMiLocalityQuery('') }}>
            {cityOptions.map((c) => <option key={c}>{c}</option>)}
          </select>
        </Field>
        <Field label="Locality" hint="Type to search">
          <input
            value={miLocalityQuery}
            onChange={(e) => { setMiLocalityQuery(e.target.value); setMiSelectedLocality('') }}
            placeholder="e.g. Sector 150, Dwarka"
          />
        </Field>
      </div>

      {miLocalities.length ? (
        <div className="suggestions forecast-suggestions">
          {miLocalities.map((item) => (
            <button
              key={item}
              className={`suggestion ${miSelectedLocality === item ? 'active' : ''}`}
              onClick={() => { setMiSelectedLocality(item); setMiLocalityQuery(item) }}
            >
              {item}
            </button>
          ))}
        </div>
      ) : null}

      <button className="btn" onClick={onLoad} disabled={miLoading || !locality}>
        {miLoading ? 'Analysing...' : 'Analyse Market'}
      </button>

      {miError ? <p className="error-msg" style={{ marginTop: 12 }}>{miError}</p> : null}

      {d ? (
        <section className="li-results">
          <p className="li-desc" style={{ marginBottom: 4 }}>
            Latest scrape: <strong>{d.latestScrapeDate || '—'}</strong> · {d.city} · {d.canonicalLocality}
          </p>

          {/* Section 2: Headline indices */}
          <h4 className="market-section-title">Market Pulse</h4>
          <div className="market-heat-banner" style={{ borderColor: heatColor }}>
            <div className="market-heat-score" style={{ color: heatColor }}>{heatScore.toFixed(0)}</div>
            <div className="market-heat-info">
              <div className="market-heat-label">Market Heat Index</div>
              <div className="market-heat-tag" style={{ background: heatColor }}>{idx.marketLabel || 'Balanced market'}</div>
              {idx.marketHeatSmoothed != null ? (
                <div className="market-heat-smoothed">
                  Smoothed (4-scrape): <strong>{(idx.marketHeatSmoothed).toFixed(0)}</strong>
                </div>
              ) : null}
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <MiniSparkline points={series.marketHeatTrend} color={heatColor} />
              {series.marketHeatSmoothedTrend?.length > 1 ? (
                <MiniSparkline points={series.marketHeatSmoothedTrend} color="#7f8c8d" />
              ) : null}
            </div>
          </div>

          <div className="market-grid market-index-grid">
            <IndexGauge label="Supply Pressure" value={idx.supplyPressureIndex} color="#c0392b" />
            <IndexGauge label="Demand Strength" value={idx.demandStrengthIndex} color="#2a7d4f" />
            <IndexGauge label="Liquidity" value={idx.liquidityIndex} color="#2980b9" />
            <IndexGauge label="Price Momentum" value={idx.priceMomentumScore} color="#8e44ad" />
            <IndexGauge label="Circle Premium" value={idx.circlePremiumScore} color="#d35400" />
          </div>

          {/* Section 3: Supply diagnostics */}
          <h4 className="market-section-title">Supply Diagnostics</h4>
          <div className="market-grid">
            <div className="market-card">
              <div className="market-card-label">Active Supply Stock</div>
              <div className="market-card-value">{(kpis.activeSupplyStock ?? 0).toLocaleString('en-IN')}</div>
              <div className="market-card-hint">Active listings in this locality &amp; segment</div>
              <MiniSparkline points={series.supplyTrend} color="#7f8c8d" />
            </div>
            <div className="market-card">
              <div className="market-card-label">New Supply Velocity</div>
              <div className="market-card-value">{((kpis.newSupplyVelocity ?? 0) * 100).toFixed(1)}%</div>
              <div className="market-card-hint">Share of active listings that newly appeared</div>
              <MiniSparkline points={series.newSupplyTrend} color="#7f8c8d" />
            </div>
            <div className="market-card">
              <div className="market-card-label">Ready Inventory Share</div>
              <div className="market-card-value">{((kpis.readyInventoryShare ?? 0) * 100).toFixed(1)}%</div>
              <div className="market-card-hint">Ready to move listings</div>
            </div>
            <div className="market-card">
              <div className="market-card-label">Under-Construction Share</div>
              <div className="market-card-value">{((kpis.underConstructionShare ?? 0) * 100).toFixed(1)}%</div>
              <div className="market-card-hint">Under-construction listings</div>
            </div>
            <div className="market-card">
              <div className="market-card-label">Stale Inventory Share</div>
              <div className="market-card-value">{((kpis.staleInventoryShare ?? 0) * 100).toFixed(1)}%</div>
              <div className="market-card-hint">Listings on market &gt; 90 days</div>
            </div>
          </div>

          {/* Section 4: Demand / Liquidity */}
          <h4 className="market-section-title">Demand &amp; Liquidity</h4>
          <div className="market-grid">
            <div className="market-card">
              <div className="market-card-label">Proxy Absorption Rate</div>
              <div className="market-card-value">{((kpis.absorptionRate ?? 0) * 100).toFixed(1)}%</div>
              <div className="market-card-hint">Listings that disappeared between scrape runs — proxy for sold/removed</div>
              <MiniSparkline points={series.absorptionTrend} color="#2980b9" />
            </div>
            <div className="market-card">
              <div className="market-card-label">Median Days on Market</div>
              <div className="market-card-value">{(kpis.medianDaysOnMarket ?? 0).toFixed(0)} days</div>
              <div className="market-card-hint">Median listing age at time of scrape</div>
            </div>
            <div className="market-card">
              <div className="market-card-label">Price Cut Frequency</div>
              <div className="market-card-value">{((kpis.priceCutFrequency ?? 0) * 100).toFixed(1)}%</div>
              <div className="market-card-hint">Share of updated listings with reduced asking price</div>
              <MiniSparkline points={series.priceCutTrend} color="#c0392b" />
            </div>
            <div className="market-card">
              <div className="market-card-label">Price Hike Frequency</div>
              <div className="market-card-value">{((kpis.priceHikeFrequency ?? 0) * 100).toFixed(1)}%</div>
              <div className="market-card-hint">Share of updated listings with increased asking price</div>
              <MiniSparkline points={series.priceHikeTrend} color="#2a7d4f" />
            </div>
            <div className="market-card">
              <div className="market-card-label">Agent Count</div>
              <div className="market-card-value">{kpis.agentCount ?? 0}</div>
              <div className="market-card-hint">Distinct agent types active in this locality</div>
            </div>
            <div className="market-card">
              <div className="market-card-label">Developer Count</div>
              <div className="market-card-value">{kpis.developerCount ?? 0}</div>
              <div className="market-card-hint">Distinct developers with active listings</div>
            </div>
          </div>

          {/* Section 5: Pricing intelligence */}
          <h4 className="market-section-title">Pricing Intelligence</h4>
          <div className="market-grid">
            <div className="market-card">
              <div className="market-card-label">Median Price</div>
              <div className="market-card-value">{fmtPrice(kpis.medianPrice)}</div>
            </div>
            <div className="market-card">
              <div className="market-card-label">Median INR/sqft</div>
              <div className="market-card-value">INR {fmtNumber(kpis.medianPpsf, 0)}</div>
              <MiniSparkline points={series.medianPpsfTrend} color="#8e44ad" />
            </div>
            <div className="market-card">
              <div className="market-card-label">P25 INR/sqft</div>
              <div className="market-card-value">INR {fmtNumber(kpis.p25Ppsf, 0)}</div>
              <div className="market-card-hint">25th percentile asking PPSF</div>
            </div>
            <div className="market-card">
              <div className="market-card-label">P75 INR/sqft</div>
              <div className="market-card-value">INR {fmtNumber(kpis.p75Ppsf, 0)}</div>
              <div className="market-card-hint">75th percentile asking PPSF</div>
            </div>
            {kpis.priceToCircleRatio > 0 ? (
              <div className="market-card">
                <div className="market-card-label">Circle Rate (Govt.)</div>
                <div className="market-card-value">INR {fmtNumber(kpis.medianCircleRate ?? 0, 0)}</div>
                <div className="market-card-hint">Government floor price / sqft</div>
              </div>
            ) : null}
            {kpis.priceToCircleRatio > 0 ? (
              <div className="market-card">
                <div className="market-card-label">Price-to-Circle Ratio</div>
                <div className="market-card-value" style={{ color: kpis.priceToCircleRatio >= 1 ? '#2a7d4f' : '#c0392b' }}>
                  {fmtNumber(kpis.priceToCircleRatio, 2)}×
                </div>
                <div className="market-card-hint">
                  {kpis.priceToCircleRatio >= 1
                    ? `Market trades ${fmtNumber((kpis.priceToCircleRatio - 1) * 100, 0)}% above circle rate`
                    : `Market trades below circle rate`}
                </div>
              </div>
            ) : null}
          </div>

          {/* Section 7: XAI Drivers */}
          {drivers.length ? (
            <>
              <h4 className="market-section-title">Market Intelligence Drivers</h4>
              <div className="xai-drivers market-drivers">
                {drivers.map((drv, i) => {
                  const sc = String(drv.score)
                  const col = DRIVER_COLORS[sc] || DRIVER_COLORS[0]
                  const barWidth = `${((drv.score + 2) / 4) * 100}%`
                  return (
                    <div key={i} className="xai-driver market-driver-card" style={{ background: col.bg }}>
                      <div className="xai-driver-head">
                        <span className="xai-driver-label">{drv.title}</span>
                        <span className="xai-driver-badge" style={{ background: col.bar }}>{drv.label}</span>
                      </div>
                      <div className="xai-bar-track">
                        <div className="xai-bar-fill" style={{ width: barWidth, background: col.bar }} />
                        <div className="xai-bar-tick" />
                      </div>
                      <p className="xai-driver-detail">{drv.explanation}</p>
                    </div>
                  )
                })}
              </div>
            </>
          ) : null}
        </section>
      ) : null}
    </>
  )
}

function App() {
  const [options, setOptions] = useState(defaultOptions)
  const [modelStatus, setModelStatus] = useState({})
  const [activeTab, setActiveTab] = useState(TAB_KEYS.BF)

  const [city, setCity] = useState(defaultOptions.cities[0])
  const [localityQuery, setLocalityQuery] = useState('')
  const [selectedLocality, setSelectedLocality] = useState('')
  const [suggestions, setSuggestions] = useState([])

  const [bfForm, setBfForm] = useState(defaultBfForm)
  const [aptForm, setAptForm] = useState(defaultAptForm)
  const [plotForm, setPlotForm] = useState(defaultPlotForm)

  const [bfResult, setBfResult] = useState(null)
  const [aptResult, setAptResult] = useState(null)
  const [plotResult, setPlotResult] = useState(null)

  const [loadingPredict, setLoadingPredict] = useState(false)
  const [loadingWhatIf, setLoadingWhatIf] = useState(false)
  const [error, setError] = useState('')
  const [info, setInfo] = useState('')

  const [bfWhatIf, setBfWhatIf] = useState({ feature: 'area_sqft', value: defaultBfForm.area_sqft, result: null })
  const [aptWhatIf, setAptWhatIf] = useState({ feature: 'area_sqft', value: defaultAptForm.area_sqft, result: null })
  const [plotWhatIf, setPlotWhatIf] = useState({ feature: 'area_sqft', value: defaultPlotForm.area_sqft, result: null })

  const [forecastSegment, setForecastSegment] = useState(defaultOptions.forecastSegments[0].key)
  const [forecastLocalityQuery, setForecastLocalityQuery] = useState('')
  const [forecastSelectedLocality, setForecastSelectedLocality] = useState('')
  const [forecastLocalityOptions, setForecastLocalityOptions] = useState([])
  const [forecastPropertyIds, setForecastPropertyIds] = useState([])
  const [forecastPropertyId, setForecastPropertyId] = useState('')
  const [forecastOverview, setForecastOverview] = useState(null)
  const [forecastContext, setForecastContext] = useState(null)
  const [loadingForecast, setLoadingForecast] = useState(false)
  const [buyDecision, setBuyDecision] = useState(null)
  const [loadingBuyDecision, setLoadingBuyDecision] = useState(false)
  const [roiInsight, setRoiInsight] = useState(null)
  const [loadingRoiInsight, setLoadingRoiInsight] = useState(false)
  const [roiAssumptions, setRoiAssumptions] = useState({
    holdYears: 5,
    areaSqft: '',
    purchaseCostPct: 7,
    annualHoldingCostPct: 1.5,
    exitCostPct: 2,
    rentYieldPct: '',
  })
  const [autoRentYieldPct, setAutoRentYieldPct] = useState('')

  // --- Market Intelligence tab state ---
  const [miSegment, setMiSegment] = useState('apt')
  const [miCity, setMiCity] = useState('Noida')
  const [miLocalityQuery, setMiLocalityQuery] = useState('')
  const [miLocalities, setMiLocalities] = useState([])
  const [miSelectedLocality, setMiSelectedLocality] = useState('')
  const [miContext, setMiContext] = useState(null)
  const [miLoading, setMiLoading] = useState(false)
  const [miError, setMiError] = useState('')

  const chosenLocality = (selectedLocality || localityQuery).trim()
  const forecastLocality = (forecastSelectedLocality || forecastLocalityQuery).trim()
  const activeInsightSegment =
    activeTab === TAB_KEYS.FORECAST
      ? forecastSegment
      : activeTab === TAB_KEYS.BF
        ? TAB_KEYS.BF
        : activeTab === TAB_KEYS.APT
          ? TAB_KEYS.APT
          : activeTab === TAB_KEYS.PLOT
            ? TAB_KEYS.PLOT
            : ''
  const activeInsightLocality = activeTab === TAB_KEYS.FORECAST ? forecastLocality : chosenLocality
  const activeInsightPropertyId = activeTab === TAB_KEYS.FORECAST ? forecastPropertyId : ''
  const activeInsightAreaSqft =
    activeTab === TAB_KEYS.BF
      ? Number(bfForm.area_sqft)
      : activeTab === TAB_KEYS.APT
        ? Number(aptForm.area_sqft)
        : activeTab === TAB_KEYS.PLOT
          ? Number(plotForm.area_sqft)
          : null

  useEffect(() => {
    async function bootstrap() {
      try {
        const [opts, status] = await Promise.all([apiGet('/meta/options'), apiGet('/meta/model-status')])
        setOptions(opts)
        setModelStatus(status)

        if (opts.forecastSegments?.length) {
          const firstForecast = opts.forecastSegments.find((item) => String(item.available) === 'true') || opts.forecastSegments[0]
          if (firstForecast?.key) setForecastSegment(firstForecast.key)
        }

        setBfForm((prev) => ({ ...prev, age: opts.ageCategories[1], furnishing: opts.furnishingCategories[1], facing: opts.facingCategories[1] }))
        setAptForm((prev) => ({ ...prev, age: opts.ageCategories[1], furnishing: opts.furnishingCategories[1], facing: opts.facingCategories[1], floor_level: opts.floorLevels[1], property_segment: opts.aptPropertySegments[1] }))
        setPlotForm((prev) => ({ ...prev, usage_type: opts.plotUsageOptions[0], facing_direction: opts.plotFacingOptions[0] }))
      } catch (e) {
        setError(`Failed to load app metadata: ${e.message}`)
      }
    }
    bootstrap()
  }, [])

  useEffect(() => {
    if (activeTab !== TAB_KEYS.FORECAST) return
    async function loadOverview() {
      try {
        const data = await apiGet(`/forecast/overview?segment=${encodeURIComponent(forecastSegment)}`)
        setForecastOverview(data)
      } catch {
        setForecastOverview(null)
      }
    }
    loadOverview()
  }, [activeTab, forecastSegment])

  useEffect(() => {
    if (activeTab !== TAB_KEYS.FORECAST) return
    const q = forecastLocalityQuery.trim()
    if (!q) {
      setForecastLocalityOptions([])
      return
    }

    const t = setTimeout(async () => {
      try {
        const data = await apiGet(
          `/forecast/localities?segment=${encodeURIComponent(forecastSegment)}&city=${encodeURIComponent(city)}&query=${encodeURIComponent(q)}&limit=8`,
        )
        setForecastLocalityOptions(data.items || [])
      } catch {
        setForecastLocalityOptions([])
      }
    }, 180)

    return () => clearTimeout(t)
  }, [activeTab, forecastSegment, city, forecastLocalityQuery])

  // Market intelligence locality suggestions
  useEffect(() => {
    if (activeTab !== TAB_KEYS.MI) return
    const q = miLocalityQuery.trim()
    if (!q) {
      setMiLocalities([])
      return
    }
    const t = setTimeout(async () => {
      try {
        const data = await apiGet(
          `/market-intelligence/localities?segment=${encodeURIComponent(miSegment)}&city=${encodeURIComponent(miCity)}&query=${encodeURIComponent(q)}&limit=30`,
        )
        setMiLocalities(data.localities || [])
      } catch {
        setMiLocalities([])
      }
    }, 180)
    return () => clearTimeout(t)
  }, [activeTab, miSegment, miCity, miLocalityQuery])

  useEffect(() => {
    if (activeTab !== TAB_KEYS.FORECAST) return
    const locality = (forecastSelectedLocality || forecastLocalityQuery).trim()
    if (!locality) {
      setForecastPropertyIds([])
      setForecastPropertyId('')
      return
    }

    async function loadPropertyIds() {
      try {
        const data = await apiGet(
          `/forecast/property-ids?segment=${encodeURIComponent(forecastSegment)}&city=${encodeURIComponent(city)}&locality=${encodeURIComponent(locality)}&limit=250`,
        )
        const ids = [...new Set(data.items || [])]
        setForecastPropertyIds(ids)
        setForecastPropertyId((prev) => (ids.includes(prev) ? prev : ids[0] || ''))
      } catch {
        setForecastPropertyIds([])
        setForecastPropertyId('')
      }
    }

    loadPropertyIds()
  }, [activeTab, forecastSegment, city, forecastSelectedLocality, forecastLocalityQuery])

  useEffect(() => {
    if (![TAB_KEYS.BF, TAB_KEYS.APT, TAB_KEYS.PLOT, TAB_KEYS.FORECAST].includes(activeTab)) return
    setBuyDecision(null)
    setRoiInsight(null)
    setAutoRentYieldPct('')
  }, [activeTab, activeInsightSegment, activeInsightLocality, activeInsightPropertyId])

  useEffect(() => {
    if (!localityQuery.trim()) {
      setSuggestions([])
      return
    }

    const t = setTimeout(async () => {
      try {
        const data = await apiGet(`/localities?city=${encodeURIComponent(city)}&query=${encodeURIComponent(localityQuery)}&limit=8`)
        setSuggestions(data.items || [])
      } catch {
        setSuggestions([])
      }
    }, 220)

    return () => clearTimeout(t)
  }, [city, localityQuery])

  useEffect(() => {
    async function hydrateCircleRate() {
      if (!chosenLocality) return
      try {
        const [bfCr, plotCr] = await Promise.all([
          apiGet(`/circle-rate?city=${encodeURIComponent(city)}&locality=${encodeURIComponent(chosenLocality)}`),
          apiGet(`/circle-rate?city=${encodeURIComponent(city)}&locality=${encodeURIComponent(chosenLocality)}&property_type=${encodeURIComponent(plotForm.usage_type)}`),
        ])

        if (bfCr.value != null) {
          setBfForm((prev) => ({ ...prev, circle_rate: Number(bfCr.value) }))
          setAptForm((prev) => ({ ...prev, circle_rate: Number(bfCr.value) }))
        }
        if (plotCr.value != null) {
          setPlotForm((prev) => ({ ...prev, circle_rate: Number(plotCr.value) }))
        }
      } catch {
        // Keep manual values if lookup fails.
      }
    }

    hydrateCircleRate()
  }, [city, chosenLocality, plotForm.usage_type])

  useEffect(() => {
    async function updateRoadDistances() {
      if (!plotForm.lat || !plotForm.lon) return
      try {
        const distances = await apiGet(`/road-distances?lat=${plotForm.lat}&lon=${plotForm.lon}`)
        setPlotForm((prev) => ({
          ...prev,
          closest_distance_MDR_km: distances.closest_distance_MDR_km || 0,
          closest_distance_SH_km: distances.closest_distance_SH_km || 0,
          closest_distance_NH_km: distances.closest_distance_NH_km || 0,
        }))
      } catch {
        // Keep existing values if lookup fails
      }
    }

    updateRoadDistances()
  }, [plotForm.lat, plotForm.lon])

  const syncLatLon = (lat, lon) => {
    setBfForm((prev) => ({ ...prev, lat, lon }))
    setAptForm((prev) => ({ ...prev, lat, lon }))
    setPlotForm((prev) => ({ ...prev, lat, lon }))
  }

  const geocodeLocality = async () => {
    if (!chosenLocality) return
    try {
      setInfo('Fetching coordinates...')
      const data = await apiGet(`/geocode?city=${encodeURIComponent(city)}&locality=${encodeURIComponent(chosenLocality)}`)
      if (data.latLon && data.latLon.length === 2) {
        syncLatLon(Number(data.latLon[0]), Number(data.latLon[1]))
        setInfo(`Coordinates updated: ${fmtNumber(data.latLon[0], 5)}, ${fmtNumber(data.latLon[1], 5)}`)
      } else {
        setInfo('No coordinates found for this locality.')
      }
    } catch (e) {
      setInfo(`Geocode failed: ${e.message}`)
    }
  }

  const predictBuilderFloor = async (payload = bfForm) => apiPost('/predict/builder-floor', payload)
  const predictApartment = async (payload = aptForm) => apiPost('/predict/apartment', payload)
  const predictPlot = async (payload = plotForm) => apiPost('/predict/plot', payload)

  const loadMarketIntelligence = async () => {
    const locality = (miSelectedLocality || miLocalityQuery).trim()
    if (!locality) {
      setMiError('Please select a locality first.')
      return
    }
    setMiError('')
    setMiLoading(true)
    try {
      const params = new URLSearchParams({ segment: miSegment, city: miCity, locality })
      const data = await apiGet(`/market-intelligence/context?${params.toString()}`)
      if (data.error) {
        setMiError(data.error)
        setMiContext(null)
      } else {
        setMiContext(data)
      }
    } catch (e) {
      setMiError(`Market intelligence failed: ${e.message}`)
      setMiContext(null)
    } finally {
      setMiLoading(false)
    }
  }

  const loadForecastContext = async () => {
    const locality = forecastLocality
    if (!locality) {
      setError('Please select a forecast locality first.')
      return
    }

    setError('')
    setLoadingForecast(true)
    try {
      const params = new URLSearchParams({
        segment: forecastSegment,
        city,
        locality,
        years: '5',
      })
      if (forecastPropertyId) params.set('property_id', forecastPropertyId)

      const data = await apiGet(`/forecast/context?${params.toString()}`)
      setForecastContext(data)
    } catch (e) {
      setError(`Forecast deep dive failed: ${e.message}`)
      setForecastContext(null)
    } finally {
      setLoadingForecast(false)
    }
  }

  const loadBuyDecision = async () => {
    const locality = activeInsightLocality
    if (!activeInsightSegment) {
      setError('Please select a valid segment first.')
      return
    }
    if (!locality) {
      setError('Please select a locality first.')
      return
    }

    setError('')
    setLoadingBuyDecision(true)
    try {
      const params = new URLSearchParams({
        segment: activeInsightSegment,
        city,
        locality,
        hold_years: String(Number(roiAssumptions.holdYears) || 5),
      })
      if (activeInsightPropertyId) params.set('property_id', activeInsightPropertyId)

      const data = await apiGet(`/insights/buy-decision?${params.toString()}`)
      setBuyDecision(data)
    } catch (e) {
      setError(`Buy decision failed: ${e.message}`)
      setBuyDecision(null)
    } finally {
      setLoadingBuyDecision(false)
    }
  }

  const loadRoiInsight = async () => {
    const locality = activeInsightLocality
    if (!activeInsightSegment) {
      setError('Please select a valid segment first.')
      return
    }
    if (!locality) {
      setError('Please select a locality first.')
      return
    }

    setError('')
    setLoadingRoiInsight(true)
    try {
      const params = new URLSearchParams({
        segment: activeInsightSegment,
        city,
        locality,
        hold_years: String(Number(roiAssumptions.holdYears) || 5),
        purchase_cost_pct: String(Number(roiAssumptions.purchaseCostPct) || 0),
        annual_holding_cost_pct: String(Number(roiAssumptions.annualHoldingCostPct) || 0),
        exit_cost_pct: String(Number(roiAssumptions.exitCostPct) || 0),
      })

      const rentYieldRaw = String(roiAssumptions.rentYieldPct ?? '').trim()
      const rentYieldVal = Number(rentYieldRaw)
      if (rentYieldRaw !== '' && Number.isFinite(rentYieldVal) && rentYieldVal >= 0) {
        params.set('rent_yield_pct', String(rentYieldVal))
      }

      const areaVal = Number(roiAssumptions.areaSqft)
      if (Number.isFinite(areaVal) && areaVal > 0) {
        params.set('area_sqft', String(areaVal))
      } else if (activeTab !== TAB_KEYS.FORECAST && Number.isFinite(activeInsightAreaSqft) && activeInsightAreaSqft > 0) {
        params.set('area_sqft', String(activeInsightAreaSqft))
      }
      if (activeInsightPropertyId) params.set('property_id', activeInsightPropertyId)

      const data = await apiGet(`/insights/roi?${params.toString()}`)
      setRoiInsight(data)

      if (rentYieldRaw === '' && data?.rentalYield?.source === 'rent_model') {
        setAutoRentYieldPct(String(data.rentalYield.effectivePct ?? ''))
      } else if (rentYieldRaw === '') {
        setAutoRentYieldPct('')
      }
    } catch (e) {
      setError(`ROI insight failed: ${e.message}`)
      setRoiInsight(null)
    } finally {
      setLoadingRoiInsight(false)
    }
  }

  const runPredict = async (tabKey) => {
    setError('')
    setLoadingPredict(true)
    try {
      if (tabKey === TAB_KEYS.BF) {
        const out = await predictBuilderFloor()
        setBfResult(out)
      }
      if (tabKey === TAB_KEYS.APT) {
        const out = await predictApartment()
        setAptResult(out)
      }
      if (tabKey === TAB_KEYS.PLOT) {
        const out = await predictPlot()
        setPlotResult(out)
      }
    } catch (e) {
      setError(`Prediction failed: ${e.message}`)
    } finally {
      setLoadingPredict(false)
    }
  }

  const bfWhatIfConfig = useMemo(
    () => [
      { key: 'area_sqft', label: 'Area (sqft)', type: 'number', min: 100, max: 20000, step: 50 },
      { key: 'circle_rate', label: 'Circle Rate', type: 'number', min: 100, max: 100000, step: 100 },
      { key: 'bhk', label: 'BHK', type: 'number', min: 1, max: 10, step: 1 },
      { key: 'bathrooms', label: 'Bathrooms', type: 'number', min: 1, max: 10, step: 1 },
      { key: 'balconies', label: 'Balconies', type: 'number', min: 0, max: 10, step: 1 },
      { key: 'age', label: 'Age', type: 'select', options: options.ageCategories },
      { key: 'furnishing', label: 'Furnishing', type: 'select', options: options.furnishingCategories },
      { key: 'facing', label: 'Facing', type: 'select', options: options.facingCategories },
      { key: 'is_parking', label: 'Parking Available', type: 'binary' },
      { key: 'is_pool', label: 'Swimming Pool', type: 'binary' },
      { key: 'is_main_road', label: 'Main Road Access', type: 'binary' },
      { key: 'is_garden_park', label: 'Garden / Park', type: 'binary' },
      { key: 'is_gated', label: 'Gated Community', type: 'binary' },
      { key: 'is_corner', label: 'Corner Property', type: 'binary' },
    ],
    [options],
  )

  const aptWhatIfConfig = useMemo(
    () => [
      ...bfWhatIfConfig,
      { key: 'floor_level', label: 'Floor Level', type: 'select', options: options.floorLevels },
      { key: 'is_ground', label: 'Ground Floor', type: 'binary' },
      { key: 'is_top', label: 'Top Floor', type: 'binary' },
      { key: 'property_segment', label: 'Property Segment', type: 'select', options: options.aptPropertySegments },
    ],
    [bfWhatIfConfig, options],
  )

  const plotWhatIfConfig = useMemo(
    () => [
      { key: 'area_sqft', label: 'Plot Area (sqft)', type: 'number', min: 100, max: 50000, step: 100 },
      { key: 'circle_rate', label: 'Circle Rate', type: 'number', min: 100, max: 100000, step: 100 },
      { key: 'usage_type', label: 'Usage Type', type: 'select', options: options.plotUsageOptions },
      { key: 'facing_direction', label: 'Facing Direction', type: 'select', options: options.plotFacingOptions },
      { key: 'road_width_bucket', label: 'Approach Road Width', type: 'select', options: options.plotRoadWidthOptions },
      { key: 'is_park_facing', label: 'Park Facing', type: 'binary' },
      { key: 'is_corner', label: 'Corner Plot', type: 'binary' },
      { key: 'is_rectangular', label: 'Rectangular Shape', type: 'binary' },
      { key: 'is_gated', label: 'Gated Plot', type: 'binary' },
      { key: 'has_boundary_wall', label: 'Boundary Wall', type: 'binary' },
    ],
    [options],
  )

  const applyRoadWidth = (payload, bucket) => ({
    ...payload,
    road_width_upto_9m: bucket === 'Upto 9m' ? 1 : 0,
    road_width_9_to_18m: bucket === '9m to 18m' ? 1 : 0,
    road_width_18_plus: bucket === '18m+' ? 1 : 0,
  })

  const getDefaultWhatIfValue = (config, featureKey, formState) => {
    const selected = config.find((item) => item.key === featureKey)
    if (!selected) return ''
    if (selected.type === 'number') {
      return formState[selected.key] ?? selected.min ?? 0
    }
    if (selected.type === 'binary') {
      return Number(formState[selected.key] ?? 0)
    }
    if (selected.type === 'select') {
      if (selected.key === 'road_width_bucket') {
        return formState.road_width_18_plus ? '18m+' : formState.road_width_9_to_18m ? '9m to 18m' : 'Upto 9m'
      }
      return formState[selected.key] ?? selected.options?.[0] ?? ''
    }
    return ''
  }

  const formatFeatureValueLabel = (type, value) => {
    if (type === 'binary') return Number(value) === 1 ? 'Yes' : 'No'
    return String(value)
  }

  const buildSingleWhatIfResult = (base, mod) => ({
    mode: 'single',
    baselineTotal: base.total,
    whatIfTotal: mod.total,
    baselinePpsf: base.ppsf,
    whatIfPpsf: mod.ppsf,
    deltaTotal: mod.total - base.total,
    deltaPct: base.total ? ((mod.total - base.total) / base.total) * 100 : 0,
    deltaPpsf: mod.ppsf - base.ppsf,
  })

  const buildCategoricalWhatIfResult = (featureLabel, featureType, baselineValue, baselineOut, rows) => ({
    mode: 'categorical',
    featureLabel,
    baselineLabel: formatFeatureValueLabel(featureType, baselineValue),
    baselineTotal: baselineOut.total,
    baselinePpsf: baselineOut.ppsf,
    comparisons: rows.map(({ value, out }) => ({
      valueLabel: formatFeatureValueLabel(featureType, value),
      isBaseline: String(value) === String(baselineValue),
      total: out.total,
      ppsf: out.ppsf,
      deltaTotal: out.total - baselineOut.total,
      deltaPct: baselineOut.total ? ((out.total - baselineOut.total) / baselineOut.total) * 100 : 0,
      deltaPpsf: out.ppsf - baselineOut.ppsf,
    })),
  })

  const runWhatIf = async (tabKey) => {
    setError('')
    setLoadingWhatIf(true)
    try {
      if (tabKey === TAB_KEYS.BF) {
        const selected = bfWhatIfConfig.find((item) => item.key === bfWhatIf.feature)
        const basePayload = { ...bfForm }
        if (!selected) return

        const base = await predictBuilderFloor(basePayload)

        if (selected.type === 'number') {
          const modPayload = { ...bfForm, [bfWhatIf.feature]: bfWhatIf.value }
          const mod = await predictBuilderFloor(modPayload)
          setBfWhatIf((prev) => ({ ...prev, result: buildSingleWhatIfResult(base, mod) }))
        } else {
          const optionsToCompare = selected.type === 'binary' ? [0, 1] : selected.options
          const comparisons = await Promise.all(
            optionsToCompare.map(async (optionValue) => {
              const payload = { ...bfForm, [bfWhatIf.feature]: optionValue }
              const out = await predictBuilderFloor(payload)
              return { value: optionValue, out }
            }),
          )

          setBfWhatIf((prev) => ({
            ...prev,
            result: buildCategoricalWhatIfResult(selected.label, selected.type, bfForm[bfWhatIf.feature], base, comparisons),
          }))
        }
      }

      if (tabKey === TAB_KEYS.APT) {
        const selected = aptWhatIfConfig.find((item) => item.key === aptWhatIf.feature)
        const basePayload = { ...aptForm }
        if (!selected) return

        const base = await predictApartment(basePayload)

        if (selected.type === 'number') {
          const modPayload = { ...aptForm, [aptWhatIf.feature]: aptWhatIf.value }
          const mod = await predictApartment(modPayload)
          setAptWhatIf((prev) => ({ ...prev, result: buildSingleWhatIfResult(base, mod) }))
        } else {
          const optionsToCompare = selected.type === 'binary' ? [0, 1] : selected.options
          const comparisons = await Promise.all(
            optionsToCompare.map(async (optionValue) => {
              const payload = { ...aptForm, [aptWhatIf.feature]: optionValue }
              const out = await predictApartment(payload)
              return { value: optionValue, out }
            }),
          )

          setAptWhatIf((prev) => ({
            ...prev,
            result: buildCategoricalWhatIfResult(selected.label, selected.type, aptForm[aptWhatIf.feature], base, comparisons),
          }))
        }
      }

      if (tabKey === TAB_KEYS.PLOT) {
        const selected = plotWhatIfConfig.find((item) => item.key === plotWhatIf.feature)
        if (!selected) return

        let basePayload = { ...plotForm }
        const baseFeatureValue = selected.key === 'road_width_bucket'
          ? plotForm.road_width_18_plus ? '18m+' : plotForm.road_width_9_to_18m ? '9m to 18m' : 'Upto 9m'
          : plotForm[selected.key]

        if (selected.key === 'road_width_bucket') {
          basePayload = applyRoadWidth(basePayload, baseFeatureValue)
        }
        const base = await predictPlot(basePayload)

        if (selected.type === 'number') {
          let modPayload = { ...plotForm }
          if (selected.key === 'road_width_bucket') {
            modPayload = applyRoadWidth(modPayload, plotWhatIf.value)
          } else {
            modPayload[selected.key] = plotWhatIf.value
          }

          const mod = await predictPlot(modPayload)
          setPlotWhatIf((prev) => ({ ...prev, result: buildSingleWhatIfResult(base, mod) }))
        } else {
          const optionsToCompare = selected.type === 'binary' ? [0, 1] : selected.options
          const comparisons = await Promise.all(
            optionsToCompare.map(async (optionValue) => {
              let payload = { ...plotForm }
              if (selected.key === 'road_width_bucket') {
                payload = applyRoadWidth(payload, optionValue)
              } else {
                payload[selected.key] = optionValue
              }
              const out = await predictPlot(payload)
              return { value: optionValue, out }
            }),
          )

          setPlotWhatIf((prev) => ({
            ...prev,
            result: buildCategoricalWhatIfResult(selected.label, selected.type, baseFeatureValue, base, comparisons),
          }))
        }
      }
    } catch (e) {
      setError(`What-If failed: ${e.message}`)
    } finally {
      setLoadingWhatIf(false)
    }
  }

  const activeResult = activeTab === TAB_KEYS.BF ? bfResult : activeTab === TAB_KEYS.APT ? aptResult : activeTab === TAB_KEYS.PLOT ? plotResult : null

  const decisionIntelligenceSection = (
    <>
      <section className="insights-panel">
        <h4>Decision Intelligence</h4>
        <div className="insights-actions">
          <button className="btn ghost" onClick={loadBuyDecision} disabled={loadingBuyDecision}>
            {loadingBuyDecision ? 'Analyzing Buy Decision...' : 'Should I Buy This?'}
          </button>
          <button className="btn ghost" onClick={loadRoiInsight} disabled={loadingRoiInsight}>
            {loadingRoiInsight ? 'Calculating ROI...' : 'Estimate ROI'}
          </button>
        </div>

        <div className="roi-assumptions-grid">
          <Field label="Hold Years">
            <input
              type="number"
              min="1"
              max="10"
              value={roiAssumptions.holdYears}
              onChange={(e) => setRoiAssumptions((prev) => ({ ...prev, holdYears: Number(e.target.value) }))}
            />
          </Field>
          <Field label="Area Override (sqft)">
            <input
              type="number"
              min="0"
              placeholder="Use property/tab area"
              value={roiAssumptions.areaSqft}
              onChange={(e) => setRoiAssumptions((prev) => ({ ...prev, areaSqft: e.target.value }))}
            />
          </Field>
          <Field label="Buy Cost %">
            <input
              type="number"
              min="0"
              max="20"
              step="0.1"
              value={roiAssumptions.purchaseCostPct}
              onChange={(e) => setRoiAssumptions((prev) => ({ ...prev, purchaseCostPct: Number(e.target.value) }))}
            />
          </Field>
          <Field label="Annual Holding Cost %">
            <input
              type="number"
              min="0"
              max="10"
              step="0.1"
              value={roiAssumptions.annualHoldingCostPct}
              onChange={(e) => setRoiAssumptions((prev) => ({ ...prev, annualHoldingCostPct: Number(e.target.value) }))}
            />
          </Field>
          <Field label="Exit Cost %">
            <input
              type="number"
              min="0"
              max="10"
              step="0.1"
              value={roiAssumptions.exitCostPct}
              onChange={(e) => setRoiAssumptions((prev) => ({ ...prev, exitCostPct: Number(e.target.value) }))}
            />
          </Field>
          <Field
            label="Rent Yield % (Manual Override)"
            hint={
              activeInsightSegment === TAB_KEYS.PLOT
                ? 'Optional for plot/land. Leave blank to assume no rental income.'
                : 'Leave blank to auto-calculate from rent model for apartment/builder floor.'
            }
          >
            <input
              type="number"
              min="0"
              max="10"
              step="0.1"
              value={roiAssumptions.rentYieldPct !== '' ? roiAssumptions.rentYieldPct : autoRentYieldPct}
              placeholder={activeInsightSegment === TAB_KEYS.PLOT ? 'Optional' : 'Auto from rent model'}
              onChange={(e) => setRoiAssumptions((prev) => ({ ...prev, rentYieldPct: e.target.value }))}
            />
          </Field>
        </div>
      </section>

      {buyDecision?.decision ? (
        <section className="insight-result">
          <div className="whatif-metrics">
            <MetricCard label="Buy Recommendation" value={buyDecision.decision.recommendation} />
            <MetricCard label="Decision Score" value={fmtNumber(buyDecision.decision.score, 2)} />
            <MetricCard label="Confidence" value={fmtPct((buyDecision.decision.confidence || 0) * 100, 0)} />
            <MetricCard label="Valuation Gap" value={fmtPct(buyDecision.decision.valuationGapPct, 2)} />
            <MetricCard label="Expected Upside" value={fmtPct(buyDecision.decision.expectedUpsidePct, 2)} />
            <MetricCard label="Avg YoY" value={fmtPct(buyDecision.decision.avgYoYPct, 2)} />
          </div>

          {buyDecision.decision.reasons?.length ? (
            <div className="insight-list">
              <h5>Why</h5>
              <ul>
                {buyDecision.decision.reasons.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </div>
          ) : null}

          {buyDecision.decision.risks?.length ? (
            <div className="insight-list risk">
              <h5>Risks</h5>
              <ul>
                {buyDecision.decision.risks.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </div>
          ) : null}
        </section>
      ) : null}

      {roiInsight?.returns ? (
        <section className="insight-result">
          <div className="whatif-metrics">
            <MetricCard label="ROI Verdict" value={roiInsight.verdict} />
            <MetricCard label="Total ROI" value={fmtPct(roiInsight.returns.roiPct, 2)} />
            <MetricCard label="Annualized CAGR" value={fmtPct(roiInsight.returns.annualizedCagrPct, 2)} />
            <MetricCard label="Total Invested" value={fmtPrice(roiInsight.cashflows.totalInvested)} />
            <MetricCard label="Net Sale Proceeds" value={fmtPrice(roiInsight.cashflows.netSaleProceeds)} />
            <MetricCard label="Net Profit" value={fmtPrice(roiInsight.cashflows.netProfit)} />
          </div>
          <p className="insight-note">
            Horizon quarter used: {roiInsight.valuation.horizonQuarterUsed}. Buy ppsf: INR {fmtNumber(roiInsight.valuation.buyPpsf, 2)}. Exit ppsf: INR {fmtNumber(roiInsight.valuation.forecastExitPpsf, 2)}.
          </p>
          {roiInsight.rentalYield ? (
            <p className="insight-note">
              Rental yield source: {roiInsight.rentalYield.source === 'manual_override'
                ? 'Manual override'
                : roiInsight.rentalYield.source === 'rent_model'
                  ? 'Rent model (derived)'
                  : 'No rental assumption'}
              . Effective yield: {fmtNumber(roiInsight.rentalYield.effectivePct, 2)}%
              {roiInsight.rentalYield.monthlyRentEstimate != null
                ? `. Estimated monthly rent: ${fmtPrice(roiInsight.rentalYield.monthlyRentEstimate)}`
                : ''}
            </p>
          ) : null}
        </section>
      ) : null}
    </>
  )

  const forecastSeries = useMemo(() => {
    if (!forecastContext?.series) return null

    const toPoints = (rows, valueKey) =>
      (rows || [])
        .map((row) => Number(row?.[valueKey]))
        .filter((v) => Number.isFinite(v))

    return {
      historical: toPoints(forecastContext.series.historicalTrend, 'price_per_sqft'),
      locality: toPoints(forecastContext.series.localityForecast, 'pred_price_per_sqft'),
      property: toPoints(forecastContext.series.propertyForecast, 'forecast_price_per_sqft'),
      median: toPoints(forecastContext.series.distribution, 'median'),
    }
  }, [forecastContext])

  const forecastFormulaSignal = useMemo(() => {
    const linearMae = forecastContext?.mathValidation?.linearMeanAbsError
    const logMae = forecastContext?.mathValidation?.logMeanAbsError
    if (linearMae == null || logMae == null) return null

    const better = linearMae <= logMae ? 'Linear' : 'Log'
    const worst = Math.max(linearMae, logMae)
    const best = Math.max(Math.min(linearMae, logMae), 1e-9)
    const ratio = worst / best

    return { better, ratio }
  }, [forecastContext])

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <h1>NCR Property Intelligence</h1>
          <p>React migration of your Streamlit estimator with improved UX, what-if simulations, and API-driven predictions.</p>
        </div>
        <div className="status-pills">
          <span className={`pill ${modelStatus.builderFloor ? 'ok' : 'bad'}`}>Builder Floor: {modelStatus.builderFloor ? 'Ready' : 'Missing'}</span>
          <span className={`pill ${modelStatus.apartment ? 'ok' : 'bad'}`}>Apartment: {modelStatus.apartment ? 'Ready' : 'Missing'}</span>
          <span className={`pill ${modelStatus.plot ? 'ok' : 'bad'}`}>Plot: {modelStatus.plot ? 'Ready' : 'Missing'}</span>
        </div>
      </header>

      <main className="layout">
        <aside className="panel sidebar">
          <h2>Location Intelligence</h2>
          <Field label="City / Region">
            <select value={city} onChange={(e) => setCity(e.target.value)}>
              {options.cities.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </Field>

          <Field label="Locality Search" hint="Type and pick from suggestions for better circle-rate matching.">
            <input value={localityQuery} onChange={(e) => { setLocalityQuery(e.target.value); setSelectedLocality('') }} placeholder="Sector 85, Preet Vihar..." />
          </Field>

          {suggestions.length ? (
            <div className="suggestions">
              {suggestions.map((item) => (
                <button
                  key={item}
                  className={`suggestion ${selectedLocality === item ? 'active' : ''}`}
                  onClick={() => {
                    setSelectedLocality(item)
                    setLocalityQuery(item)
                  }}
                >
                  {item}
                </button>
              ))}
            </div>
          ) : null}

          <div className="coords-grid">
            <Field label="Latitude">
              <input
                type="number"
                step="0.000001"
                value={bfForm.lat}
                onChange={(e) => syncLatLon(Number(e.target.value), bfForm.lon)}
              />
            </Field>
            <Field label="Longitude">
              <input
                type="number"
                step="0.000001"
                value={bfForm.lon}
                onChange={(e) => syncLatLon(bfForm.lat, Number(e.target.value))}
              />
            </Field>
          </div>

          <button className="btn" onClick={geocodeLocality}>Auto Geocode Locality</button>

          {info ? <div className="message info">{info}</div> : null}
          {error ? <div className="message error">{error}</div> : null}
        </aside>

        <section className="panel content">
          <div className="tabs">
            <button className={activeTab === TAB_KEYS.BF ? 'active' : ''} onClick={() => setActiveTab(TAB_KEYS.BF)}>Builder Floor</button>
            <button className={activeTab === TAB_KEYS.APT ? 'active' : ''} onClick={() => setActiveTab(TAB_KEYS.APT)}>Apartment</button>
            <button className={activeTab === TAB_KEYS.PLOT ? 'active' : ''} onClick={() => setActiveTab(TAB_KEYS.PLOT)}>Plot / Land</button>
            <button className={activeTab === TAB_KEYS.FORECAST ? 'active' : ''} onClick={() => setActiveTab(TAB_KEYS.FORECAST)}>Forecast Intelligence</button>
            <button className={activeTab === TAB_KEYS.MI ? 'active' : ''} onClick={() => setActiveTab(TAB_KEYS.MI)}>Market Intelligence</button>
          </div>

          {activeTab === TAB_KEYS.BF ? (
            <>
              <h3>Builder Floor Inputs</h3>
              <div className="form-grid">
                <Field label="BHK"><input type="number" min="1" max="10" value={bfForm.bhk} onChange={(e) => setBfForm((p) => ({ ...p, bhk: Number(e.target.value) }))} /></Field>
                <Field label="Area (sqft)"><input type="number" min="100" max="20000" value={bfForm.area_sqft} onChange={(e) => setBfForm((p) => ({ ...p, area_sqft: Number(e.target.value) }))} /></Field>
                <Field label="Bathrooms"><input type="number" min="1" max="10" value={bfForm.bathrooms} onChange={(e) => setBfForm((p) => ({ ...p, bathrooms: Number(e.target.value) }))} /></Field>
                <Field label="Balconies"><input type="number" min="0" max="10" value={bfForm.balconies} onChange={(e) => setBfForm((p) => ({ ...p, balconies: Number(e.target.value) }))} /></Field>
                <Field label="Age"><select value={bfForm.age} onChange={(e) => setBfForm((p) => ({ ...p, age: e.target.value }))}>{options.ageCategories.map((x) => <option key={x}>{x}</option>)}</select></Field>
                <Field label="Furnishing"><select value={bfForm.furnishing} onChange={(e) => setBfForm((p) => ({ ...p, furnishing: e.target.value }))}>{options.furnishingCategories.map((x) => <option key={x}>{x}</option>)}</select></Field>
                <Field label="Facing"><select value={bfForm.facing} onChange={(e) => setBfForm((p) => ({ ...p, facing: e.target.value }))}>{options.facingCategories.map((x) => <option key={x}>{x}</option>)}</select></Field>
                <Field label="Circle Rate (INR/sqft)"><input type="number" min="100" max="100000" value={bfForm.circle_rate} onChange={(e) => setBfForm((p) => ({ ...p, circle_rate: Number(e.target.value) }))} /></Field>
                <Field label="Parking"><select value={bfForm.is_parking} onChange={(e) => setBfForm((p) => ({ ...p, is_parking: Number(e.target.value) }))}><option value="0">No</option><option value="1">Yes</option></select></Field>
                <Field label="Pool"><select value={bfForm.is_pool} onChange={(e) => setBfForm((p) => ({ ...p, is_pool: Number(e.target.value) }))}><option value="0">No</option><option value="1">Yes</option></select></Field>
                <Field label="Main Road"><select value={bfForm.is_main_road} onChange={(e) => setBfForm((p) => ({ ...p, is_main_road: Number(e.target.value) }))}><option value="0">No</option><option value="1">Yes</option></select></Field>
                <Field label="Garden/Park"><select value={bfForm.is_garden_park} onChange={(e) => setBfForm((p) => ({ ...p, is_garden_park: Number(e.target.value) }))}><option value="0">No</option><option value="1">Yes</option></select></Field>
                <Field label="Gated"><select value={bfForm.is_gated} onChange={(e) => setBfForm((p) => ({ ...p, is_gated: Number(e.target.value) }))}><option value="0">No</option><option value="1">Yes</option></select></Field>
                <Field label="Corner"><select value={bfForm.is_corner} onChange={(e) => setBfForm((p) => ({ ...p, is_corner: Number(e.target.value) }))}><option value="0">No</option><option value="1">Yes</option></select></Field>
              </div>

              <button className="btn" onClick={() => runPredict(TAB_KEYS.BF)} disabled={loadingPredict}>{loadingPredict ? 'Predicting...' : 'Predict Builder Floor'}</button>

              <WhatIfPanel
                title="What-If Analysis"
                config={bfWhatIfConfig}
                selectedFeature={bfWhatIf.feature}
                onChangeFeature={(feature) =>
                  setBfWhatIf((p) => ({
                    ...p,
                    feature,
                    value: getDefaultWhatIfValue(bfWhatIfConfig, feature, bfForm),
                    result: null,
                  }))
                }
                value={bfWhatIf.value}
                onChangeValue={(value) => setBfWhatIf((p) => ({ ...p, value, result: null }))}
                onRun={() => runWhatIf(TAB_KEYS.BF)}
                running={loadingWhatIf}
                result={bfWhatIf.result}
              />

              {decisionIntelligenceSection}
            </>
          ) : null}

          {activeTab === TAB_KEYS.APT ? (
            <>
              <h3>Apartment Inputs</h3>
              <div className="form-grid">
                <Field label="BHK"><input type="number" min="1" max="10" value={aptForm.bhk} onChange={(e) => setAptForm((p) => ({ ...p, bhk: Number(e.target.value) }))} /></Field>
                <Field label="Area (sqft)"><input type="number" min="100" max="20000" value={aptForm.area_sqft} onChange={(e) => setAptForm((p) => ({ ...p, area_sqft: Number(e.target.value) }))} /></Field>
                <Field label="Bathrooms"><input type="number" min="1" max="10" value={aptForm.bathrooms} onChange={(e) => setAptForm((p) => ({ ...p, bathrooms: Number(e.target.value) }))} /></Field>
                <Field label="Balconies"><input type="number" min="0" max="10" value={aptForm.balconies} onChange={(e) => setAptForm((p) => ({ ...p, balconies: Number(e.target.value) }))} /></Field>
                <Field label="Age"><select value={aptForm.age} onChange={(e) => setAptForm((p) => ({ ...p, age: e.target.value }))}>{options.ageCategories.map((x) => <option key={x}>{x}</option>)}</select></Field>
                <Field label="Furnishing"><select value={aptForm.furnishing} onChange={(e) => setAptForm((p) => ({ ...p, furnishing: e.target.value }))}>{options.furnishingCategories.map((x) => <option key={x}>{x}</option>)}</select></Field>
                <Field label="Facing"><select value={aptForm.facing} onChange={(e) => setAptForm((p) => ({ ...p, facing: e.target.value }))}>{options.facingCategories.map((x) => <option key={x}>{x}</option>)}</select></Field>
                <Field label="Circle Rate (INR/sqft)"><input type="number" min="100" max="100000" value={aptForm.circle_rate} onChange={(e) => setAptForm((p) => ({ ...p, circle_rate: Number(e.target.value) }))} /></Field>
                <Field label="Floor Level"><select value={aptForm.floor_level} onChange={(e) => setAptForm((p) => ({ ...p, floor_level: e.target.value }))}>{options.floorLevels.map((x) => <option key={x}>{x}</option>)}</select></Field>
                <Field label="Ground Floor"><select value={aptForm.is_ground} onChange={(e) => setAptForm((p) => ({ ...p, is_ground: Number(e.target.value) }))}><option value="0">No</option><option value="1">Yes</option></select></Field>
                <Field label="Top Floor"><select value={aptForm.is_top} onChange={(e) => setAptForm((p) => ({ ...p, is_top: Number(e.target.value) }))}><option value="0">No</option><option value="1">Yes</option></select></Field>
                <Field label="Property Segment"><select value={aptForm.property_segment} onChange={(e) => setAptForm((p) => ({ ...p, property_segment: e.target.value }))}>{options.aptPropertySegments.map((x) => <option key={x}>{x}</option>)}</select></Field>
                <Field label="Parking"><select value={aptForm.is_parking} onChange={(e) => setAptForm((p) => ({ ...p, is_parking: Number(e.target.value) }))}><option value="0">No</option><option value="1">Yes</option></select></Field>
                <Field label="Pool"><select value={aptForm.is_pool} onChange={(e) => setAptForm((p) => ({ ...p, is_pool: Number(e.target.value) }))}><option value="0">No</option><option value="1">Yes</option></select></Field>
                <Field label="Main Road"><select value={aptForm.is_main_road} onChange={(e) => setAptForm((p) => ({ ...p, is_main_road: Number(e.target.value) }))}><option value="0">No</option><option value="1">Yes</option></select></Field>
                <Field label="Garden/Park"><select value={aptForm.is_garden_park} onChange={(e) => setAptForm((p) => ({ ...p, is_garden_park: Number(e.target.value) }))}><option value="0">No</option><option value="1">Yes</option></select></Field>
                <Field label="Gated"><select value={aptForm.is_gated} onChange={(e) => setAptForm((p) => ({ ...p, is_gated: Number(e.target.value) }))}><option value="0">No</option><option value="1">Yes</option></select></Field>
                <Field label="Corner"><select value={aptForm.is_corner} onChange={(e) => setAptForm((p) => ({ ...p, is_corner: Number(e.target.value) }))}><option value="0">No</option><option value="1">Yes</option></select></Field>
              </div>

              <button className="btn" onClick={() => runPredict(TAB_KEYS.APT)} disabled={loadingPredict}>{loadingPredict ? 'Predicting...' : 'Predict Apartment'}</button>

              <WhatIfPanel
                title="What-If Analysis"
                config={aptWhatIfConfig}
                selectedFeature={aptWhatIf.feature}
                onChangeFeature={(feature) =>
                  setAptWhatIf((p) => ({
                    ...p,
                    feature,
                    value: getDefaultWhatIfValue(aptWhatIfConfig, feature, aptForm),
                    result: null,
                  }))
                }
                value={aptWhatIf.value}
                onChangeValue={(value) => setAptWhatIf((p) => ({ ...p, value, result: null }))}
                onRun={() => runWhatIf(TAB_KEYS.APT)}
                running={loadingWhatIf}
                result={aptWhatIf.result}
              />

              {decisionIntelligenceSection}
            </>
          ) : null}

          {activeTab === TAB_KEYS.PLOT ? (
            <>
              <h3>Plot / Land Inputs</h3>
              <div className="form-grid">
                <Field label="Plot Area (sqft)"><input type="number" min="100" max="50000" value={plotForm.area_sqft} onChange={(e) => setPlotForm((p) => ({ ...p, area_sqft: Number(e.target.value) }))} /></Field>
                <Field label="Circle Rate (INR/sqft)"><input type="number" min="100" max="100000" value={plotForm.circle_rate} onChange={(e) => setPlotForm((p) => ({ ...p, circle_rate: Number(e.target.value) }))} /></Field>
                <Field label="Distance to MDR (km)" hint="Major District Road"><input type="number" value={plotForm.closest_distance_MDR_km} readOnly style={{ backgroundColor: '#f5f5f5' }} /></Field>
                <Field label="Distance to SH (km)" hint="State Highway"><input type="number" value={plotForm.closest_distance_SH_km} readOnly style={{ backgroundColor: '#f5f5f5' }} /></Field>
                <Field label="Distance to NH (km)" hint="National Highway"><input type="number" value={plotForm.closest_distance_NH_km} readOnly style={{ backgroundColor: '#f5f5f5' }} /></Field>
                <Field label="Usage Type"><select value={plotForm.usage_type} onChange={(e) => setPlotForm((p) => ({ ...p, usage_type: e.target.value }))}>{options.plotUsageOptions.map((x) => <option key={x}>{x}</option>)}</select></Field>
                <Field label="Facing Direction"><select value={plotForm.facing_direction} onChange={(e) => setPlotForm((p) => ({ ...p, facing_direction: e.target.value }))}>{options.plotFacingOptions.map((x) => <option key={x}>{x}</option>)}</select></Field>
                <Field label="Approach Road Width"><select value={plotForm.road_width_18_plus ? '18m+' : plotForm.road_width_9_to_18m ? '9m to 18m' : 'Upto 9m'} onChange={(e) => {
                  const bucket = e.target.value
                  setPlotForm((p) => ({
                    ...p,
                    road_width_upto_9m: bucket === 'Upto 9m' ? 1 : 0,
                    road_width_9_to_18m: bucket === '9m to 18m' ? 1 : 0,
                    road_width_18_plus: bucket === '18m+' ? 1 : 0,
                  }))
                }}>{options.plotRoadWidthOptions.map((x) => <option key={x}>{x}</option>)}</select></Field>
                <Field label="Park Facing"><select value={plotForm.is_park_facing} onChange={(e) => setPlotForm((p) => ({ ...p, is_park_facing: Number(e.target.value) }))}><option value="0">No</option><option value="1">Yes</option></select></Field>
                <Field label="Corner Plot"><select value={plotForm.is_corner} onChange={(e) => setPlotForm((p) => ({ ...p, is_corner: Number(e.target.value) }))}><option value="0">No</option><option value="1">Yes</option></select></Field>
                <Field label="Rectangular Shape"><select value={plotForm.is_rectangular} onChange={(e) => setPlotForm((p) => ({ ...p, is_rectangular: Number(e.target.value) }))}><option value="0">No</option><option value="1">Yes</option></select></Field>
                <Field label="Gated Plot"><select value={plotForm.is_gated} onChange={(e) => setPlotForm((p) => ({ ...p, is_gated: Number(e.target.value) }))}><option value="0">No</option><option value="1">Yes</option></select></Field>
                <Field label="Boundary Wall"><select value={plotForm.has_boundary_wall} onChange={(e) => setPlotForm((p) => ({ ...p, has_boundary_wall: Number(e.target.value) }))}><option value="0">No</option><option value="1">Yes</option></select></Field>
              </div>

              <button className="btn" onClick={() => runPredict(TAB_KEYS.PLOT)} disabled={loadingPredict}>{loadingPredict ? 'Predicting...' : 'Predict Plot / Land'}</button>

              <WhatIfPanel
                title="What-If Analysis"
                config={plotWhatIfConfig}
                selectedFeature={plotWhatIf.feature}
                onChangeFeature={(feature) =>
                  setPlotWhatIf((p) => ({
                    ...p,
                    feature,
                    value: getDefaultWhatIfValue(plotWhatIfConfig, feature, plotForm),
                    result: null,
                  }))
                }
                value={plotWhatIf.value}
                onChangeValue={(value) => setPlotWhatIf((p) => ({ ...p, value, result: null }))}
                onRun={() => runWhatIf(TAB_KEYS.PLOT)}
                running={loadingWhatIf}
                result={plotWhatIf.result}
              />

              {decisionIntelligenceSection}
            </>
          ) : null}

          {activeTab === TAB_KEYS.FORECAST ? (
            <>
              <h3>Forecast Intelligence (FloData)</h3>

              <div className="forecast-controls">
                <Field label="Segment">
                  <select
                    value={forecastSegment}
                    onChange={(e) => {
                      setForecastSegment(e.target.value)
                      setForecastContext(null)
                      setForecastLocalityQuery('')
                      setForecastSelectedLocality('')
                      setForecastPropertyId('')
                    }}
                  >
                    {(options.forecastSegments || []).map((item) => (
                      <option key={item.key} value={item.key}>
                        {item.label}
                      </option>
                    ))}
                  </select>
                </Field>

                <Field label="Forecast Locality" hint="Search localities from new forecast artifacts.">
                  <input
                    value={forecastLocalityQuery}
                    onChange={(e) => {
                      setForecastLocalityQuery(e.target.value)
                      setForecastSelectedLocality('')
                      setForecastContext(null)
                    }}
                    placeholder="Type locality, e.g., Sector 57"
                  />
                </Field>

                <Field label="Property ID" hint="Optional, uses first matching property if empty.">
                  <select value={forecastPropertyId} onChange={(e) => setForecastPropertyId(e.target.value)}>
                    <option value="">Auto Select</option>
                    {forecastPropertyIds.map((id) => (
                      <option key={id} value={id}>
                        {id}
                      </option>
                    ))}
                  </select>
                </Field>
              </div>

              {forecastLocalityOptions.length ? (
                <div className="suggestions forecast-suggestions">
                  {forecastLocalityOptions.map((item) => (
                    <button
                      key={item}
                      className={`suggestion ${forecastSelectedLocality === item ? 'active' : ''}`}
                      onClick={() => {
                        setForecastSelectedLocality(item)
                        setForecastLocalityQuery(item)
                        setForecastContext(null)
                      }}
                    >
                      {item}
                    </button>
                  ))}
                </div>
              ) : null}

              <button className="btn" onClick={loadForecastContext} disabled={loadingForecast}>
                {loadingForecast ? 'Loading Deep Dive...' : 'Load Forecast Deep Dive'}
              </button>

              {decisionIntelligenceSection}

              {forecastOverview ? (
                <section className="forecast-overview">
                  <div className="whatif-metrics">
                    <MetricCard label="Forecast Horizon" value={`${forecastOverview.metrics?.forecast_horizon ?? '--'} Q`} />
                    <MetricCard label="Forecast Rows" value={fmtNumber(forecastOverview.metrics?.n_property_forecasts ?? null, 0)} />
                    <MetricCard label="Rho Mean" value={fmtNumber(forecastOverview.metrics?.rho_stats?.mean ?? null, 4)} />
                  </div>
                </section>
              ) : null}

              {forecastContext ? (
                <section className="forecast-detail">
                  <div className="whatif-metrics">
                    <MetricCard label="Selected Property" value={forecastContext.selectedPropertyId || '--'} />
                    <MetricCard label="Listed INR/sqft" value={`INR ${fmtNumber(forecastContext.kpis?.listingPricePpsf, 0)}`} />
                    <MetricCard label="Model INR/sqft" value={`INR ${fmtNumber(forecastContext.kpis?.modelPricePpsf, 0)}`} />
                    <MetricCard label="Listed vs Model" value={fmtPct(forecastContext.kpis?.deltaPct)} />
                    <MetricCard label="Linear Formula MAE" value={fmtNumber(forecastContext.mathValidation?.linearMeanAbsError, 6)} />
                    <MetricCard label="Log Formula MAE" value={fmtNumber(forecastContext.mathValidation?.logMeanAbsError, 6)} />
                  </div>

                  {forecastContext.rho ? (
                    <div className="rho-box">
                      <h4>Rho Construction</h4>
                      <p>
                        C_i={fmtNumber(forecastContext.rho.comp_support, 4)} | U_i={fmtNumber(forecastContext.rho.uniqueness, 4)} | M_i={fmtNumber(forecastContext.rho.model_confidence, 4)}
                      </p>
                      <p>
                        rho_0(file)={fmtNumber(forecastContext.rho.rho_0_file, 4)} | rho_0(formula)={fmtNumber(forecastContext.rho.rho_0_formula_clipped, 4)}
                      </p>
                    </div>
                  ) : null}

                  {forecastContext.yoy?.property?.length ? (
                    <div className="whatif-metrics">
                      {forecastContext.yoy.property.map((row) => (
                        <MetricCard
                          key={row.label}
                          label={`${row.label} YoY`}
                          value={fmtPct(row.yoy_pct)}
                          delta={`${row.anchor_date} -> ${row.target_date}`}
                        />
                      ))}
                    </div>
                  ) : null}

                  {forecastSeries ? (
                    <section className="forecast-trends-grid">
                      <TrendCard title="Selected Property Forecast" points={forecastSeries.property} color="#178e9b" />
                      <TrendCard title="Locality Forecast" points={forecastSeries.locality} color="#d9892b" />
                      <TrendCard title="Historical Locality Trend" points={forecastSeries.historical} color="#6c54b3" />
                      <TrendCard title="Locality Median Distribution" points={forecastSeries.median} color="#2a7d4f" />
                    </section>
                  ) : null}

                  {forecastFormulaSignal ? (
                    <div className="formula-diagnostic">
                      <strong>Formula Fit Signal:</strong> {forecastFormulaSignal.better} form currently fits better. Error ratio (worse/better): {fmtNumber(forecastFormulaSignal.ratio, 2)}x
                    </div>
                  ) : null}

                  {forecastContext.quarterTable?.length ? (
                    <div className="comparison-wrap forecast-table-wrap">
                      <table className="comparison-table">
                        <thead>
                          <tr>
                            <th>Date</th>
                            <th>Quarter</th>
                            <th>P_i</th>
                            <th>L_i</th>
                            <th>rho_0</th>
                            <th>rho_t</th>
                            <th>I_l,t</th>
                            <th>Forecast INR/sqft</th>
                          </tr>
                        </thead>
                        <tbody>
                          {forecastContext.quarterTable.map((row) => (
                            <tr key={`${row.date}-${row.quarter}`}>
                              <td>{row.date}</td>
                              <td>{row.quarter}</td>
                              <td>{fmtNumber(row.P_i, 2)}</td>
                              <td>{fmtNumber(row.L_i, 2)}</td>
                              <td>{fmtNumber(row.rho_0, 4)}</td>
                              <td>{fmtNumber(row.rho_t, 4)}</td>
                              <td>{fmtNumber(row.I_lt, 4)}</td>
                              <td>{fmtNumber(row.forecast_price_per_sqft, 2)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : null}
                </section>
              ) : null}
            </>
          ) : null}

          {activeTab === TAB_KEYS.MI ? (
            <MarketIntelligenceTab
              miSegment={miSegment}
              setMiSegment={setMiSegment}
              miCity={miCity}
              setMiCity={setMiCity}
              miLocalityQuery={miLocalityQuery}
              setMiLocalityQuery={setMiLocalityQuery}
              miLocalities={miLocalities}
              miSelectedLocality={miSelectedLocality}
              setMiSelectedLocality={setMiSelectedLocality}
              miContext={miContext}
              miLoading={miLoading}
              miError={miError}
              onLoad={loadMarketIntelligence}
            />
          ) : null}

          {activeResult ? (
            <section className="results-band">
              <MetricCard label="Estimated Price" value={fmtPrice(activeResult.total)} />
              <MetricCard label="Estimated INR/sqft" value={`INR ${fmtNumber(activeResult.ppsf, 0)}`} />
              {'predRatio' in activeResult ? <MetricCard label="Pred/Circle Ratio" value={fmtNumber(activeResult.predRatio)} /> : <MetricCard label="Model Target" value="Price per sqft" />}
            </section>
          ) : null}

          {activeResult?.explanation ? <ExplainPanel explanation={activeResult.explanation} /> : null}
        </section>
      </main>
    </div>
  )
}

export default App
