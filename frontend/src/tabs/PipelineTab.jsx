import { useEffect, useState } from 'react'
import { fetchSources, fetchFederated, fetchModels } from '../api/client'
import { Activity, Server, Database, Cpu, Wifi, CheckCircle, XCircle } from 'lucide-react'

const PIPELINE_STAGES = [
  { id: 'sentinel2',    label: 'Sentinel-2 (NDVI)',     icon: '🛰',  group: 'Input' },
  { id: 'sentinel1',    label: 'Sentinel-1 (SAR)',      icon: '📡',  group: 'Input' },
  { id: 'weather',      label: 'Weather (Open-Meteo)',  icon: '🌧',  group: 'Input' },
  { id: 'amt',          label: 'AMT/CSAMT',             icon: '⚡',  group: 'Input' },
  { id: 'ant',          label: 'ANT Tomography',        icon: '🌍',  group: 'Input' },
  { id: 'hyperspectral',label: 'Hyperspectral',         icon: '🔬',  group: 'Input' },
  { id: 'fused',        label: 'Feature Fusion',        icon: '🔀',  group: 'Process' },
  { id: 'reserve',      label: 'Reserve Model (RF)',    icon: '🏔',  group: 'Process' },
  { id: 'shortfall',    label: 'Shortfall Model (GB)',  icon: '📊',  group: 'Process' },
  { id: 'causal',       label: 'Causal Analysis',       icon: '🔗',  group: 'Process' },
  { id: 'rl',           label: 'RL Twin (PPO)',         icon: '🤖',  group: 'Process' },
  { id: 'api',          label: 'FastAPI + WS',          icon: '⚡',  group: 'Output' },
  { id: 'dashboard',    label: 'React Dashboard',       icon: '🖥',  group: 'Output' },
]

export default function PipelineTab() {
  const [sources, setSources] = useState(null)
  const [fl, setFL] = useState(null)
  const [models, setModels] = useState(null)

  useEffect(() => {
    fetchSources().then(setSources).catch(() => {})
    fetchFederated().then(setFL).catch(() => {})
    fetchModels().then(setModels).catch(() => {})
    const interval = setInterval(() => {
      fetchSources().then(setSources).catch(() => {})
    }, 15_000)
    return () => clearInterval(interval)
  }, [])

  const groupedStages = ['Input', 'Process', 'Output']
  const stagesByGroup = {}
  for (const s of PIPELINE_STAGES) {
    if (!stagesByGroup[s.group]) stagesByGroup[s.group] = []
    stagesByGroup[s.group].push(s)
  }

  const pipelineStatus = sources?.pipeline || {}
  const stageStatuses = pipelineStatus.stages || {}

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Pipeline Flow */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <div className="card-title-icon"><Activity size={14} /></div>
            Data Pipeline – Live Status
          </div>
          {pipelineStatus.last_run && (
            <span style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
              Last run: {new Date(pipelineStatus.last_run).toLocaleTimeString()} · {pipelineStatus.latency_ms}ms
            </span>
          )}
        </div>

        <div style={{ display: 'flex', gap: 20, alignItems: 'flex-start', flexWrap: 'wrap' }}>
          {groupedStages.map((group, gi) => (
            <div key={group} style={{ flex: '1 1 220px' }}>
              <div style={{
                fontSize: 10, fontWeight: 700, letterSpacing: '0.08em',
                color: 'var(--text-muted)', textTransform: 'uppercase',
                marginBottom: 10, paddingLeft: 8,
                borderLeft: `2px solid ${gi === 0 ? 'var(--accent-cyan)' : gi === 1 ? 'var(--accent-ore)' : 'var(--accent-green)'}`,
              }}>
                {group}
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {stagesByGroup[group].map(stage => {
                  const isOk = stageStatuses[stage.id] === 'OK' || !stageStatuses[stage.id]
                  const hasError = stageStatuses[stage.id] === 'ERROR'
                  return (
                    <div
                      key={stage.id}
                      className={`pipeline-stage ${isOk ? 'active' : hasError ? 'error' : ''}`}
                    >
                      <div className={`stage-indicator ${hasError ? 'error' : isOk ? 'ok' : 'idle'}`} />
                      <div style={{ fontSize: 16 }}>{stage.icon}</div>
                      <div>
                        <div className="stage-name">{stage.label}</div>
                        <div className="stage-detail">
                          {hasError ? 'Error' : isOk ? 'OK' : 'Waiting'}
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Data Sources Health */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <div className="card-title-icon"><Wifi size={14} /></div>
            Data Source Status
          </div>
          <span className="badge badge-simulated"><span className="dot" />MODE: {sources?.mode || 'SIMULATED'}</span>
        </div>
        <table className="data-table">
          <thead>
            <tr>
              <th>Source</th>
              <th>Data Mode</th>
              <th>Schedule</th>
              <th>Note</th>
            </tr>
          </thead>
          <tbody>
            {(sources?.sources || []).map(s => (
              <tr key={s.name}>
                <td style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{s.name}</td>
                <td>
                  <span className={`badge badge-${s.mode?.toLowerCase()}`}>
                    <span className="dot" />{s.mode}
                  </span>
                </td>
                <td style={{ fontFamily: 'var(--font-mono)', fontSize: 11 }}>
                  {s.revisit_days ? `${s.revisit_days}d` : s.revisit_hours ? `${s.revisit_hours}h` : 'Continuous'}
                </td>
                <td style={{ fontSize: 11, color: 'var(--text-muted)' }}>{s.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Model Registry */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <div className="card-title-icon"><Cpu size={14} /></div>
            Model Registry
          </div>
        </div>
        <table className="data-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Version</th>
              <th>Trained At</th>
              <th>Key Metric</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {(models?.models || []).map(m => (
              <tr key={m.version}>
                <td style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{m.name}</td>
                <td className="mono" style={{ fontSize: 11 }}>{m.version?.split('_')[1] || m.version}</td>
                <td style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                  {m.trained_at ? new Date(m.trained_at).toLocaleString() : '—'}
                </td>
                <td className="mono" style={{ fontSize: 11 }}>
                  {m.metrics?.r2 ? `R²: ${m.metrics.r2}` :
                   m.metrics?.r2_surface_plus_subsurface ? `R²: ${m.metrics.r2_surface_plus_subsurface}` : '—'}
                </td>
                <td>
                  {m.is_active
                    ? <span style={{ color: 'var(--accent-green)', fontSize: 12, display: 'flex', alignItems: 'center', gap: 4 }}><CheckCircle size={12} /> Active</span>
                    : <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>Archived</span>
                  }
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Federated Learning Panel */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <div className="card-title-icon"><Server size={14} /></div>
            Federated Learning – FedAvg (Flower)
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <span className="badge badge-simulated"><span className="dot" />3 Sites</span>
            <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>No raw data leaves site</span>
          </div>
        </div>

        <div className="grid-3" style={{ marginBottom: 16 }}>
          {['Balaghat', 'Nagpur', 'Gumgaon'].map(site => (
            <div key={site} style={{
              background: 'var(--bg-elevated)',
              border: '1px solid var(--border)',
              borderRadius: 10,
              padding: '12px 16px',
              textAlign: 'center',
            }}>
              <div style={{ fontSize: 22, marginBottom: 6 }}>🏭</div>
              <div style={{ fontSize: 13, fontWeight: 700 }}>{site}</div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Training locally</div>
              <div style={{ fontSize: 10, marginTop: 6 }}>
                <span className="badge badge-simulated"><span className="dot" />Weights only →</span>
              </div>
            </div>
          ))}
        </div>

        <div style={{ marginBottom: 8, fontSize: 12, color: 'var(--text-muted)' }}>
          Global accuracy per FL round:
        </div>
        {(fl?.rounds || []).length === 0 ? (
          <div style={{ color: 'var(--text-muted)', fontSize: 13, padding: '16px 0' }}>
            Federated learning rounds in progress… (may take a few minutes)
          </div>
        ) : (
          (fl?.rounds || []).map(round => (
            <div key={round.round} className="fl-round-row">
              <div className="fl-round-num">Round {round.round}</div>
              <div className="fl-round-bar">
                <div
                  className="fl-round-fill"
                  style={{ width: `${(round.global_accuracy || 0) * 100}%` }}
                />
              </div>
              <div className="fl-round-acc">
                {((round.global_accuracy || 0) * 100).toFixed(1)}%
              </div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', minWidth: 120 }}>
                {round.completed_at ? new Date(round.completed_at).toLocaleTimeString() : '—'}
              </div>
            </div>
          ))
        )}
        <div style={{ marginTop: 12, fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.6 }}>
          Each mine site trains a local neural network on its own production data. Only model weights
          (not raw data) are sent to the FL server for FedAvg aggregation. Global accuracy improves
          across rounds, demonstrating collaborative learning without data sharing.
        </div>
      </div>
    </div>
  )
}
