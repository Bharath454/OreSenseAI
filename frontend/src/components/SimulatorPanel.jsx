import { useState } from 'react'
import { postSimEvent } from '../api/client'
import { Sliders, CloudRain, Hammer, Users, Play, RefreshCw } from 'lucide-react'

const EVENTS = [
  {
    id: 'heavy_rain',
    icon: '🌧',
    title: 'Heavy Rain Event',
    description: 'Injects a rainfall spike that increases equipment downtime via the causal pathway (rainfall → downtime → shortfall). Watch the risk gauge rise.',
    color: 'var(--accent-blue)',
    params: [
      { key: 'rain_mm', label: 'Rainfall (mm/day)', min: 10, max: 80, default: 40, unit: 'mm' },
    ],
    defaultDuration: 120,
  },
  {
    id: 'equipment_breakdown',
    icon: '🔧',
    title: 'Equipment Breakdown',
    description: 'Forces 2–3 machines into BREAKDOWN state, simulating unexpected failure. Observe how the RL twin recommends maintenance actions.',
    color: 'var(--accent-red)',
    params: [
      { key: 'hours', label: 'Downtime (hours)', min: 1, max: 12, default: 6, unit: 'h' },
    ],
    defaultDuration: 60,
  },
  {
    id: 'crew_shortage',
    icon: '👷',
    title: 'Crew Shortage',
    description: 'Reduces available crew, driving shortfall via the crew pathway in the causal model. Simulates a strike or mass absence.',
    color: 'var(--accent-purple)',
    params: [
      { key: 'shortage', label: 'Workers Missing', min: 10, max: 60, default: 30, unit: 'workers' },
    ],
    defaultDuration: 180,
  },
]

function EventControl({ event, addToast }) {
  const [params, setParams] = useState(
    Object.fromEntries(event.params.map(p => [p.key, p.default]))
  )
  const [duration, setDuration] = useState(event.defaultDuration)
  const [loading, setLoading] = useState(false)
  const [active, setActive] = useState(false)

  const inject = async () => {
    setLoading(true)
    try {
      const result = await postSimEvent(event.id, params, duration)
      setActive(true)
      addToast(`${event.title} injected for ${duration} minutes`, 'warning')
      setTimeout(() => setActive(false), duration * 60 * 1000)
    } catch (e) {
      addToast(`Failed: ${e.message}`, 'error')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="sim-control" style={{ borderColor: active ? event.color + '55' : undefined }}>
      {active && (
        <div style={{
          position: 'absolute',
          top: 0, left: 0, right: 0, height: 2,
          background: event.color,
          borderRadius: '12px 12px 0 0',
          animation: 'pulse 1s infinite',
        }} />
      )}
      <div className="sim-control-header">
        <div className="sim-control-icon">{event.icon}</div>
        <div>
          <div className="sim-control-title">{event.title}</div>
          {active && (
            <div style={{ fontSize: 10, color: event.color, fontWeight: 700, marginTop: 2 }}>
              ● ACTIVE – watch dashboard react live
            </div>
          )}
        </div>
      </div>

      <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 16, lineHeight: 1.5 }}>
        {event.description}
      </div>

      {event.params.map(p => (
        <div key={p.key} className="slider-row">
          <div className="slider-label">{p.label}</div>
          <input
            type="range"
            className="slider-input"
            min={p.min}
            max={p.max}
            value={params[p.key]}
            onChange={e => setParams(pr => ({ ...pr, [p.key]: Number(e.target.value) }))}
          />
          <div className="slider-value">{params[p.key]}{p.unit}</div>
        </div>
      ))}

      <div className="slider-row">
        <div className="slider-label">Duration</div>
        <input
          type="range"
          className="slider-input"
          min={15}
          max={480}
          step={15}
          value={duration}
          onChange={e => setDuration(Number(e.target.value))}
        />
        <div className="slider-value">{duration}min</div>
      </div>

      <button
        className="btn btn-primary"
        style={{ marginTop: 12, width: '100%', justifyContent: 'center', background: event.color, color: '#fff' }}
        onClick={inject}
        disabled={loading}
      >
        {loading ? (
          <><div className="spinner" style={{ width: 14, height: 14, borderWidth: 2 }} /> Injecting…</>
        ) : (
          <><Play size={14} /> Inject Event</>
        )}
      </button>
    </div>
  )
}

export default function SimulatorPanel({ addToast }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Header */}
      <div className="card" style={{ borderColor: 'rgba(245,158,11,0.3)' }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: 16 }}>
          <div style={{ fontSize: 40 }}>⚡</div>
          <div>
            <h1 style={{ fontSize: 18, fontWeight: 800, marginBottom: 4 }}>Simulator Control Panel</h1>
            <p style={{ fontSize: 13, color: 'var(--text-muted)', lineHeight: 1.6, maxWidth: 600 }}>
              Inject synthetic events into the live pipeline and observe how predictions react
              in near-real-time. Events propagate through the causal model to the shortfall
              predictor and RL twin, and are pushed to the dashboard via WebSocket.
              <br />
              <span style={{ color: 'var(--accent-purple)', fontWeight: 600 }}>All data is SIMULATED.</span>
              {' '}This demonstrates the causal chain from external shocks to production shortfall.
            </p>
          </div>
        </div>
      </div>

      {/* Event controls */}
      <div className="grid-3">
        {EVENTS.map(event => (
          <div key={event.id} style={{ position: 'relative' }}>
            <EventControl event={event} addToast={addToast} />
          </div>
        ))}
      </div>

      {/* How it works */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">How Events Propagate Through the System</div>
        </div>
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          {[
            { step: '1', label: 'Event Injected',  desc: 'POST /api/simulate/event stored in DB' },
            { step: '2', label: 'Pipeline Trigger', desc: 'Next ingestion pass reads injected parameters' },
            { step: '3', label: 'Weather Adapter',  desc: 'Rainfall overridden with injected value' },
            { step: '4', label: 'Causal Path',      desc: 'rain→downtime→shortfall in causal graph' },
            { step: '5', label: 'Risk Update',      desc: 'Shortfall model outputs new risk_pct' },
            { step: '6', label: 'WS Broadcast',     desc: 'Dashboard receives update in <2s' },
          ].map(({ step, label, desc }) => (
            <div key={step} style={{
              flex: '1 1 160px',
              background: 'var(--bg-elevated)',
              borderRadius: 10,
              padding: '12px 14px',
            }}>
              <div style={{
                width: 24, height: 24, borderRadius: '50%',
                background: 'var(--accent-ore)',
                color: '#000', fontSize: 11, fontWeight: 800,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                marginBottom: 8,
              }}>{step}</div>
              <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 4 }}>{label}</div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{desc}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
