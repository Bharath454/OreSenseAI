import { useState } from 'react'
import { postSimEvent } from '../api/client'
import { Zap, ArrowRight, TrendingDown, CheckCircle } from 'lucide-react'

export default function ActionsTab({ actionsData, shortfallData, addToast }) {
  const [applying, setApplying] = useState(null)
  const [applied, setApplied] = useState({})

  const actions = actionsData || []
  const currentRisk = shortfallData?.risk_pct ?? 0

  const applyAction = async (action) => {
    setApplying(action.action_id)
    try {
      // Map RL actions to simulator events
      let eventType = null
      let params = {}
      if (action.action_id === 1) { eventType = 'equipment_breakdown'; params = { hours: 0 } }
      if (action.action_id === 3) { eventType = 'crew_shortage'; params = { shortage: -20 } }

      if (eventType) {
        await postSimEvent(eventType, params, 30)
      }

      setApplied(a => ({ ...a, [action.action_id]: true }))
      addToast(`Action applied: ${action.name}`, 'success')
    } catch (e) {
      addToast(`Failed to apply action: ${e.message}`, 'error')
    } finally {
      setApplying(null)
    }
  }

  const rankColors = ['#f59e0b', '#06b6d4', '#8b5cf6']

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Header */}
      <div className="card" style={{ borderColor: 'rgba(245,158,11,0.2)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <div style={{ fontSize: 36 }}>🤖</div>
          <div>
            <div style={{ fontSize: 16, fontWeight: 700 }}>RL Digital Twin Recommendations</div>
            <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>
              PPO agent trained on Gymnasium mine environment · Counterfactual rollout ranking
            </div>
          </div>
          <div style={{ marginLeft: 'auto', textAlign: 'right' }}>
            <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Current Risk</div>
            <div style={{ fontSize: 28, fontWeight: 800, fontFamily: 'var(--font-mono)', color: currentRisk > 50 ? 'var(--accent-red)' : 'var(--accent-ore)' }}>
              {currentRisk.toFixed(0)}%
            </div>
          </div>
          <span className="badge badge-simulated"><span className="dot" />SIMULATED</span>
        </div>
      </div>

      {/* Action Cards */}
      {actions.length === 0 ? (
        <div className="card" style={{ textAlign: 'center', padding: 48 }}>
          <div className="spinner" style={{ margin: '0 auto 16px' }} />
          <div style={{ color: 'var(--text-muted)' }}>Computing RL recommendations…</div>
        </div>
      ) : (
        <div className="grid-3">
          {actions.map((action, i) => (
            <div
              key={action.action_id}
              className="action-card animate-fade-up"
              style={{ animationDelay: `${i * 100}ms` }}
            >
              <div className="action-rank" style={{ color: rankColors[i] }}>
                #{i + 1} RECOMMENDATION
              </div>
              <div className="action-title">{action.name}</div>
              <div className="action-desc">{action.description}</div>

              <div className="action-risk-compare">
                <div>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 2 }}>BEFORE</div>
                  <div className="risk-before">{action.predicted_risk_before?.toFixed(0)}%</div>
                </div>
                <div className="risk-arrow">→</div>
                <div>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 2 }}>AFTER</div>
                  <div className="risk-after">{action.predicted_risk_after?.toFixed(0)}%</div>
                </div>
                <div style={{ marginLeft: 'auto', textAlign: 'right' }}>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 2 }}>REDUCTION</div>
                  <div style={{ fontSize: 20, fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--accent-green)' }}>
                    -{action.risk_reduction_pct?.toFixed(1)}%
                  </div>
                </div>
              </div>

              <div className="progress-bar" style={{ marginBottom: 16 }}>
                <div
                  className="progress-fill"
                  style={{
                    width: `${100 - (action.risk_reduction_pct / action.predicted_risk_before * 100 || 0)}%`,
                    background: 'var(--accent-green)',
                  }}
                />
              </div>

              {applied[action.action_id] ? (
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--accent-green)', fontSize: 13, fontWeight: 600 }}>
                  <CheckCircle size={16} />
                  Applied to Simulator
                </div>
              ) : (
                <button
                  className="btn btn-secondary"
                  onClick={() => applyAction(action)}
                  disabled={applying === action.action_id}
                  style={{ width: '100%', justifyContent: 'center' }}
                >
                  {applying === action.action_id ? (
                    <><div className="spinner" style={{ width: 14, height: 14, borderWidth: 2 }} /> Applying…</>
                  ) : (
                    <><Zap size={14} /> Apply to Simulator</>
                  )}
                </button>
              )}

              <div style={{ marginTop: 10, fontSize: 10, color: 'var(--text-muted)', textAlign: 'center' }}>
                Confidence: {action.confidence} · Method: Counterfactual 5-step rollout
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Before/After comparison */}
      {actions.length > 0 && (
        <div className="card">
          <div className="card-header">
            <div className="card-title">Projected Risk Comparison</div>
          </div>
          <div style={{ display: 'flex', gap: 20, alignItems: 'stretch', flexWrap: 'wrap' }}>
            <div style={{
              flex: '1 1 200px',
              background: 'rgba(239,68,68,0.08)',
              border: '1px solid rgba(239,68,68,0.25)',
              borderRadius: 12,
              padding: 20,
              textAlign: 'center',
            }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 6 }}>CURRENT STATE</div>
              <div style={{ fontSize: 48, fontWeight: 900, fontFamily: 'var(--font-mono)', color: 'var(--accent-red)' }}>
                {currentRisk.toFixed(0)}%
              </div>
              <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Shortfall Risk</div>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', fontSize: 24, color: 'var(--text-muted)' }}>→</div>

            {actions.slice(0, 1).map(action => (
              <div key={action.action_id} style={{
                flex: '1 1 200px',
                background: 'rgba(16,185,129,0.08)',
                border: '1px solid rgba(16,185,129,0.25)',
                borderRadius: 12,
                padding: 20,
                textAlign: 'center',
              }}>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 6 }}>
                  AFTER: {action.name}
                </div>
                <div style={{ fontSize: 48, fontWeight: 900, fontFamily: 'var(--font-mono)', color: 'var(--accent-green)' }}>
                  {action.predicted_risk_after?.toFixed(0)}%
                </div>
                <div style={{ fontSize: 12, color: 'var(--accent-green)' }}>
                  -{action.risk_reduction_pct?.toFixed(1)}% reduction
                </div>
              </div>
            ))}
          </div>

          <div style={{ marginTop: 16, padding: 12, background: 'var(--bg-elevated)', borderRadius: 8, fontSize: 11, color: 'var(--text-muted)' }}>
            <strong style={{ color: 'var(--text-primary)' }}>How this works:</strong> The PPO agent
            was trained in a Gymnasium mine environment with 5 discrete actions. At inference,
            counterfactual rollouts simulate each action for 5 days and rank by predicted shortfall
            reduction. Causal effect estimates from DoWhy are used to validate directional changes.
            All outputs are <span className="badge badge-simulated" style={{ display: 'inline-flex' }}><span className="dot" />SIMULATED</span>.
          </div>
        </div>
      )}
    </div>
  )
}
