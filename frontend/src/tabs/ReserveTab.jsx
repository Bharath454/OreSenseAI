import { useEffect, useState, useRef } from 'react'
import { MapContainer, TileLayer, Rectangle, Popup, LayersControl } from 'react-leaflet'
import { fetchSector } from '../api/client'
import { Info, Layers } from 'lucide-react'

const CLASS_COLORS = {
  Measured:  '#10b981',
  Indicated: '#3b82f6',
  Inferred:  '#8b5cf6',
  Below:     '#1f2937',
}

const FEATURE_LABELS = {
  hyperspectral_mn_idx: 'Hyperspectral Mn Index',
  ndvi_anomaly:         'NDVI Anomaly',
  soil_moisture:        'Soil Moisture',
  insar_deform_mm:      'InSAR Deformation (mm)',
  elevation_m:          'Elevation (m)',
  amt_conductivity:     'AMT Conductivity (mS/m)',
  ant_velocity_pct:     'ANT Velocity Anomaly (%)',
}

function DataBadge({ mode }) {
  return <span className={`badge badge-${mode?.toLowerCase()}`}><span className="dot" />{mode}</span>
}

function CellPopup({ cellId, onClose }) {
  const [data, setData] = useState(null)
  useEffect(() => {
    fetchSector(cellId)
      .then(setData)
      .catch(() => setData({ error: 'Failed to load' }))
  }, [cellId])

  if (!data) return <div className="spinner" />
  if (data.error) return <div style={{ color: 'var(--accent-red)' }}>{data.error}</div>

  const est = data.estimate || {}
  const features = est.feature_json ? JSON.parse(est.feature_json || '{}') : {}

  return (
    <div style={{ minWidth: 280, maxWidth: 340 }}>
      <div style={{ marginBottom: 12, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <strong style={{ fontSize: 14 }}>Sector {cellId}</strong>
        <DataBadge mode="SIMULATED" />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 12 }}>
        <div style={{ background: 'var(--bg-base)', borderRadius: 8, padding: '8px 12px' }}>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 2 }}>ORE PROBABILITY</div>
          <div style={{ fontSize: 22, fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--accent-ore)' }}>
            {((est.ore_prob || 0) * 100).toFixed(1)}%
          </div>
        </div>
        <div style={{ background: 'var(--bg-base)', borderRadius: 8, padding: '8px 12px' }}>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 2 }}>RESERVE CLASS</div>
          <div style={{ fontSize: 14, fontWeight: 700, color: CLASS_COLORS[est.reserve_class] || '#fff' }}>
            {est.reserve_class || '—'}
          </div>
        </div>
      </div>

      <div style={{ marginBottom: 12 }}>
        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
          Feature Breakdown
        </div>
        {Object.entries(features).slice(0, 6).map(([k, v]) => (
          <div key={k} className="driver-bar-row" style={{ padding: '4px 0' }}>
            <span className="driver-name" style={{ minWidth: 140, fontSize: 11 }}>{FEATURE_LABELS[k] || k}</span>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--accent-ore)', marginLeft: 'auto' }}>
              {typeof v === 'number' ? v.toFixed(3) : v}
            </span>
          </div>
        ))}
      </div>

      <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
        Uncertainty: ±{((est.ore_uncertainty || 0) * 100).toFixed(1)}% · Model: {est.model_version?.split('_')[1] || 'v1'}
      </div>
    </div>
  )
}

export default function ReserveTab({ gridData, role, addToast }) {
  const [selectedCell, setSelectedCell] = useState(null)
  const [layer, setLayer] = useState('reserve_class')  // reserve_class | ore_prob | uncertainty
  const center = [21.665, 80.645]

  // Build rectangles from grid data
  const cells = gridData || []

  const getColor = (cell) => {
    if (layer === 'reserve_class') return CLASS_COLORS[cell.reserve_class] || CLASS_COLORS.Below
    if (layer === 'ore_prob') {
      const p = cell.ore_prob || 0
      if (p > 0.7) return '#10b981'
      if (p > 0.5) return '#3b82f6'
      if (p > 0.3) return '#8b5cf6'
      return '#1f2937'
    }
    // uncertainty
    const u = cell.ore_uncertainty || 0
    const intensity = Math.min(1, u / 0.2)
    return `rgba(245,158,11,${0.1 + 0.9 * intensity})`
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Header stats */}
      <div className="grid-4">
        {[
          { label: 'Measured',  color: CLASS_COLORS.Measured,  count: cells.filter(c => c.reserve_class === 'Measured').length },
          { label: 'Indicated', color: CLASS_COLORS.Indicated, count: cells.filter(c => c.reserve_class === 'Indicated').length },
          { label: 'Inferred',  color: CLASS_COLORS.Inferred,  count: cells.filter(c => c.reserve_class === 'Inferred').length },
          { label: 'Below',     color: CLASS_COLORS.Below,     count: cells.filter(c => c.reserve_class === 'Below' || !c.reserve_class).length },
        ].map(({ label, color, count }) => (
          <div key={label} className="stat-card" style={{ borderTopColor: color }}>
            <div className="stat-value" style={{ color, fontSize: 28 }}>{count}</div>
            <div className="stat-label">{label} Cells</div>
          </div>
        ))}
      </div>

      {/* Map + Controls */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <div className="card-title-icon"><Layers size={14} /></div>
            Manganese Reserve Map – Balaghat AOI
          </div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <DataBadge mode="SIMULATED" />
            <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Grid: 20×13 cells</span>
            {/* Layer toggle */}
            <select
              value={layer}
              onChange={e => setLayer(e.target.value)}
              style={{
                background: 'var(--bg-elevated)', border: '1px solid var(--border)',
                borderRadius: 6, color: 'var(--text-primary)', padding: '4px 8px', fontSize: 12
              }}
            >
              <option value="reserve_class">Reserve Class</option>
              <option value="ore_prob">Ore Probability</option>
              <option value="uncertainty">Uncertainty</option>
            </select>
          </div>
        </div>

        <div className="map-container">
          <MapContainer
            center={center}
            zoom={12}
            style={{ height: '100%', width: '100%' }}
          >
            <TileLayer
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              attribution='© OpenStreetMap'
              opacity={0.35}
            />
            {cells.map(cell => {
              if (!cell.lat || !cell.lon) return null
              // Estimate cell bounds from lat/lon ±half-cell
              const dLat = 0.0065  // ~13km / 20 rows
              const dLon = 0.005   // ~13km / 13 cols
              const bounds = [
                [cell.lat - dLat / 2, cell.lon - dLon / 2],
                [cell.lat + dLat / 2, cell.lon + dLon / 2],
              ]
              return (
                <Rectangle
                  key={cell.cell_id}
                  bounds={bounds}
                  pathOptions={{
                    fillColor: getColor(cell),
                    fillOpacity: 0.7,
                    color: getColor(cell),
                    weight: 0.5,
                    opacity: 0.5,
                  }}
                  eventHandlers={{
                    click: () => setSelectedCell(cell.cell_id),
                  }}
                >
                  {selectedCell === cell.cell_id && (
                    <Popup onClose={() => setSelectedCell(null)} maxWidth={360}>
                      <CellPopup cellId={cell.cell_id} onClose={() => setSelectedCell(null)} />
                    </Popup>
                  )}
                </Rectangle>
              )
            })}
          </MapContainer>
        </div>

        {/* Legend */}
        <div style={{ marginTop: 12 }}>
          <div className="heatmap-legend">
            {Object.entries(CLASS_COLORS).map(([cls, color]) => (
              <div key={cls} className="legend-item">
                <div className="legend-dot" style={{ background: color }} />
                {cls}
              </div>
            ))}
            <div className="legend-item">
              <Info size={12} style={{ color: 'var(--text-muted)' }} />
              <span style={{ color: 'var(--text-muted)', fontSize: 11 }}>
                Click a cell for feature breakdown
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Data source info */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">Data Sources & Acquisition Dates</div>
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
          {[
            { name: 'Sentinel-2 NDVI',         mode: 'SIMULATED', note: 'Revisit: ~5 days', acq: 'Simulated scene' },
            { name: 'Sentinel-1 SAR',           mode: 'SIMULATED', note: 'Revisit: 12 days', acq: 'Simulated scene' },
            { name: 'InSAR Deformation',        mode: 'CACHED',    note: 'Pre-computed',    acq: 'Synthetic interferogram' },
            { name: 'Hyperspectral (EnMAP)',    mode: 'SIMULATED', note: 'No scene for AOI', acq: 'Simulated Mn index' },
            { name: 'AMT/CSAMT Conductivity',  mode: 'SIMULATED', note: 'Field survey req.', acq: 'Derived from ore model' },
            { name: 'ANT Tomography',          mode: 'SIMULATED', note: 'Seismic net req.',  acq: 'Derived from ore model' },
          ].map(src => (
            <div key={src.name} style={{
              flex: '1 1 260px',
              background: 'var(--bg-elevated)',
              border: '1px solid var(--border)',
              borderRadius: 10,
              padding: '12px 16px',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                <span style={{ fontSize: 13, fontWeight: 600 }}>{src.name}</span>
                <DataBadge mode={src.mode} />
              </div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{src.note}</div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2, fontFamily: 'var(--font-mono)' }}>{src.acq}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
