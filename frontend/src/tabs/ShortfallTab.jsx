import { useState, useEffect } from 'react'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, BarChart, Bar, Cell
} from 'recharts'
import { fetchCausal, fetchIoT } from '../api/client'
import { TrendingDown, Activity, AlertTriangle, Users, Cloud, Hammer } from 'lucide-react'

function RiskGauge({ risk, uncertainty }) {
  const r = 70
  const circumference = 2 * Math.PI * r
  const filled = circumference * ((risk || 0) / 100)
  const color = risk > 60 ? '#ef4444' : risk > 35 ? '#f59e0b' : '#10b981'

  return (
    <div className="gauge-container">
      <div className="gauge-ring">
        <svg className="gauge-svg" viewBox="0 0 180 180">
          <circle cx={90} cy={90} r={r} fill="none" stroke="var(--bg-elevated)" strokeWidth={14} />
          <circle
            cx={90} cy={90} r={r}
            fill="none"
            stroke={color}
            strokeWidth={14}
            strokeDasharray={`${filled} ${circumference - filled}`}
            strokeLinecap="round"
            style={{ filter: `drop-shadow(0 0 8px ${color})`, transition: 'stroke-dasharray 1s ease' }}
          />
          {/* Uncertainty band */}
          {uncertainty && (
            <circle
              cx={90} cy={90} r={r}
              fill="none"
              stroke={color}
              strokeWidth={14}
              strokeDasharray={`${circumference * uncertainty / 100} ${circumference * (1 - uncertainty / 100)}`}
              strokeDashoffset={-filled}
              strokeOpacity={0.15}
            />
          )}
        </svg>
        <div className="gauge-value">
          <div className="gauge-number" style={{ color }}>
            {risk?.toFixed(0) ?? '—'}
          </div>
          <div className="gauge-unit">% Risk</div>
        </div>
      </div>
      <div className="gauge-label">
        {risk > 60 ? '🔴 HIGH RISK' : risk > 35 ? '🟡 MODERATE' : '🟢 LOW RISK'}
      </div>
      {uncertainty && (
        <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
          Uncertainty: ±{uncertainty?.toFixed(1)}%
        </div>
      )}
    </div>
  )
}

function CausalSection({ causal }) {
  if (!causal) return (
    <div style={{ color: 'var(--text-muted)', textAlign: 'center', padding: 32 }}>
      Loading causal analysis…
    </div>
  )

  const mediation = causal.mediation_summary || {}
  const drivers = [
    { name: 'Equipment Downtime',  effect: causal.equipment_downtime?.effect, via: 'Direct', color: '#ef4444' },
    { name: 'Blasting Delays',    effect: causal.blasting_delays?.effect,    via: 'Direct', color: '#f59e0b' },
    { name: 'Crew Available',     effect: causal.crew?.effect,               via: 'Direct', color: '#3b82f6' },
    { name: 'Rainfall (direct)',  effect: causal.rainfall_direct?.effect,    via: 'Via Downtime', color: '#8b5cf6' },
  ].filter(d => d.effect !== undefined)

  const maxEffect = Math.max(...drivers.map(d => Math.abs(d.effect || 0))) || 1

  return (
    <div>
      <div style={{ marginBottom: 20 }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 4 }}>
          Causal Effect Estimates
        </div>
        <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
          Effect size = percentage-point change in shortfall per unit increase in driver
        </div>
      </div>
      {drivers.map(d => (
        <div key={d.name} className="causal-effect">
          <div className="causal-treatment">{d.name}</div>
          <div style={{ flex: 1 }}>
            <div className="progress-bar" style={{ height: 10 }}>
              <div
                className="progress-fill"
                style={{
                  width: `${Math.abs(d.effect) / maxEffect * 100}%`,
                  background: d.color,
                  opacity: 0.8,
                }}
              />
            </div>
          </div>
          <div className="causal-estimate" style={{ color: d.color }}>
            {d.effect > 0 ? '+' : ''}{d.effect?.toFixed(3)}
          </div>
        </div>
      ))}

      {/* Mediation box */}
      <div style={{
        marginTop: 20,
        background: 'var(--bg-elevated)',
        border: '1px solid rgba(139,92,246,0.3)',
        borderRadius: 10,
        padding: 16,
      }}>
        <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--accent-purple)', marginBottom: 8 }}>
          ⚡ Mediation Analysis: Rainfall → Downtime → Shortfall
        </div>
        <div className="grid-2" style={{ gap: 12 }}>
          {[
            { label: 'Path: Rainfall→Downtime', val: mediation.path_a_rainfall_to_downtime },
            { label: 'Path: Downtime→Shortfall', val: mediation.path_b_downtime_to_shortfall },
            { label: 'Direct Rainfall Effect',   val: mediation.path_c_prime_direct_effect },
            { label: '% Mediated by Downtime',  val: mediation.proportion_mediated ? `${(mediation.proportion_mediated * 100).toFixed(0)}%` : null },
          ].map(({ label, val }) => (
            <div key={label}>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>{label}</div>
              <div style={{ fontSize: 16, fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>
                {typeof val === 'number' ? val.toFixed(4) : val ?? '—'}
              </div>
            </div>
          ))}
        </div>
        <div style={{ marginTop: 8, fontSize: 11, color: 'var(--accent-green)' }}>
          ✓ {mediation.interpretation || 'Running interpretation…'}
        </div>
        {causal.equipment_downtime?.refutation_random_confounder && (
          <div style={{ marginTop: 8, fontSize: 11, color: 'var(--text-muted)' }}>
            Refutation tests: Random confounder{' '}
            <span className={`causal-pass ${causal.equipment_downtime.refutation_random_confounder.passed ? 'ok' : 'fail'}`}>
              {causal.equipment_downtime.refutation_random_confounder.passed ? 'PASSED' : 'FAILED'}
            </span>
            {' '}· Placebo treatment{' '}
            <span className={`causal-pass ${causal.equipment_downtime.refutation_placebo?.passed ? 'ok' : 'fail'}`}>
              {causal.equipment_downtime.refutation_placebo?.passed ? 'PASSED' : 'FAILED'}
            </span>
          </div>
        )}
      </div>
    </div>
  )
}

function IoTTable({ iot }) {
  const machines = iot?.machines || []
  return (
    <div style={{ overflowX: 'auto' }}>
      <table className="data-table">
        <thead>
          <tr>
            <th>Machine</th>
            <th>Type</th>
            <th>State</th>
            <th>Health</th>
            <th>Vibration</th>
            <th>Fuel (L/h)</th>
            <th>Last Update</th>
          </tr>
        </thead>
        <tbody>
          {machines.length === 0 ? (
            <tr><td colSpan={7} style={{ textAlign: 'center', color: 'var(--text-muted)' }}>Loading IoT data…</td></tr>
          ) : machines.map(m => (
            <tr key={m.machine_id}>
              <td style={{ fontWeight: 600, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)', fontSize: 12 }}>{m.machine_id}</td>
              <td>{m.machine_type || '—'}</td>
              <td><span className={`state-chip state-${m.state}`}>{m.state}</span></td>
              <td>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <div className="progress-bar" style={{ width: 60 }}>
                    <div
                      className="progress-fill"
                      style={{
                        width: `${(m.health_score || 0) * 100}%`,
                        background: m.health_score > 0.7 ? 'var(--accent-green)' : m.health_score > 0.4 ? 'var(--accent-ore)' : 'var(--accent-red)',
                      }}
                    />
                  </div>
                  <span className="mono" style={{ fontSize: 11 }}>{((m.health_score || 0) * 100).toFixed(0)}%</span>
                </div>
              </td>
              <td className="mono">{m.vibration_g?.toFixed(2) ?? '—'} g</td>
              <td className="mono">{m.fuel_rate_lph?.toFixed(1) ?? '—'}</td>
              <td style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                {m.time ? new Date(m.time).toLocaleTimeString() : '—'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div style={{ marginTop: 8, textAlign: 'right' }}>
        <span className="badge badge-simulated"><span className="dot" />SIMULATED – MQTT</span>
      </div>
    </div>
  )
}

export default function ShortfallTab({ shortfallData, role }) {
  const [causal, setCausal] = useState(null)
  const [iot, setIoT] = useState(null)

  useEffect(() => {
    fetchCausal().then(setCausal).catch(() => {})
    const loadIoT = () => fetchIoT().then(setIoT).catch(() => {})
    loadIoT()
    const interval = setInterval(loadIoT, 5_000)
    return () => clearInterval(interval)
  }, [])

  const risk = shortfallData?.risk_pct ?? 0
  const forecast = shortfallData?.forecast || []
  const drivers = shortfallData?.top_drivers || []

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* KPI row */}
      <div className="grid-4">
        <div className="stat-card">
          <div className="stat-value text-ore">{risk.toFixed(0)}%</div>
          <div className="stat-label">Current Risk</div>
        </div>
        <div className="stat-card">
          <div className="stat-value text-cyan">{shortfallData?.uncertainty?.toFixed(1) ?? '—'}%</div>
          <div className="stat-label">Uncertainty</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{forecast.length > 0 ? forecast[6]?.risk_pct?.toFixed(0) : '—'}%</div>
          <div className="stat-label">7-Day Forecast</div>
        </div>
        <div className="stat-card">
          <div className="stat-value text-purple">{forecast.length > 0 ? forecast[29]?.risk_pct?.toFixed(0) : '—'}%</div>
          <div className="stat-label">30-Day Outlook</div>
        </div>
      </div>

      <div className="grid-2">
        {/* Gauge */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <div className="card-title-icon"><TrendingDown size={14} /></div>
              Live Shortfall Risk
            </div>
            <span className="badge badge-simulated"><span className="dot" />SIMULATED</span>
          </div>
          <RiskGauge risk={risk} uncertainty={shortfallData?.uncertainty} />
        </div>

        {/* Drivers */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">
              <div className="card-title-icon"><AlertTriangle size={14} /></div>
              Top Risk Drivers
            </div>
          </div>
          {drivers.length === 0 ? (
            <div style={{ color: 'var(--text-muted)', textAlign: 'center', padding: 24 }}>Loading…</div>
          ) : drivers.map((d, i) => {
            const colors = ['#ef4444', '#f59e0b', '#8b5cf6', '#3b82f6', '#10b981']
            return (
              <div key={d.feature} className="driver-bar-row">
                <div className="driver-name">{d.feature.replace(/_/g, ' ')}</div>
                <div className="driver-bar-track">
                  <div
                    className="driver-bar-fill"
                    style={{
                      width: `${d.importance * 500}%`,
                      background: colors[i % colors.length],
                      maxWidth: '100%',
                    }}
                  />
                </div>
                <div className="driver-value">{(d.importance * 100).toFixed(1)}%</div>
              </div>
            )
          })}
        </div>
      </div>

      {/* 30-day forecast chart */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">30-Day Shortfall Risk Forecast</div>
          <span className="badge badge-simulated"><span className="dot" />SIMULATED</span>
        </div>
        <ResponsiveContainer width="100%" height={220}>
          <AreaChart data={forecast} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="riskGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="rgba(255,255,255,0.04)" />
            <XAxis dataKey="date" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} interval={4} />
            <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted)' }} domain={[0, 100]} unit="%" />
            <Tooltip
              contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 8 }}
              labelStyle={{ color: 'var(--text-primary)' }}
              itemStyle={{ color: '#ef4444' }}
              formatter={v => [`${v.toFixed(1)}%`, 'Risk']}
            />
            <Area type="monotone" dataKey="risk_pct" stroke="#ef4444" fill="url(#riskGrad)" strokeWidth={2} dot={false} />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Causal analysis */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <div className="card-title-icon"><Cloud size={14} /></div>
            Causal Shortfall Analysis (DoWhy + EconML)
          </div>
          <span className="badge badge-simulated"><span className="dot" />SIMULATED</span>
        </div>
        <CausalSection causal={causal} />
      </div>

      {/* IoT Table */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <div className="card-title-icon"><Activity size={14} /></div>
            Live Equipment Status (IoT – MQTT)
          </div>
          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Refreshes every 5s</span>
        </div>
        <IoTTable iot={iot} />
      </div>
    </div>
  )
}
