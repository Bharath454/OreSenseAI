import { useState, useCallback, useEffect } from 'react'
import { useWebSocket } from './hooks/useWebSocket'
import { fetchGrid, fetchShortfall, fetchActions } from './api/client'
import ReserveTab from './tabs/ReserveTab'
import ShortfallTab from './tabs/ShortfallTab'
import ActionsTab from './tabs/ActionsTab'
import PipelineTab from './tabs/PipelineTab'
import SimulatorPanel from './components/SimulatorPanel'
import { Map, TrendingDown, Zap, Activity, Sliders, Gem } from 'lucide-react'

const TABS = [
  { id: 'reserve',   label: 'Reserve Map',     icon: Map },
  { id: 'shortfall', label: 'Shortfall',        icon: TrendingDown },
  { id: 'actions',   label: 'Actions',          icon: Zap },
  { id: 'pipeline',  label: 'Pipeline',         icon: Activity },
  { id: 'simulator', label: 'Simulator',        icon: Sliders },
]

const ROLES = ['Geologist', 'Mine Manager', 'Planner']

export default function App() {
  const [activeTab, setActiveTab] = useState('reserve')
  const [role, setRole] = useState('Mine Manager')
  const [gridData, setGridData] = useState(null)
  const [shortfallData, setShortfallData] = useState(null)
  const [actionsData, setActionsData] = useState(null)
  const [lastUpdate, setLastUpdate] = useState(null)
  const [toasts, setToasts] = useState([])

  const addToast = useCallback((msg, type = 'success') => {
    const id = Date.now()
    setToasts(t => [...t, { id, msg, type }])
    setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), 4000)
  }, [])

  // WebSocket handler
  const handleWsMessage = useCallback((data) => {
    if (data.grid)      setGridData(data.grid)
    if (data.shortfall) setShortfallData(data.shortfall)
    if (data.actions)   setActionsData(data.actions)
    if (data.timestamp) setLastUpdate(data.timestamp)
    if (data.type === 'model_updated') {
      addToast(`Model updated: ${data.version}`, 'success')
    }
    if (data.type === 'update') {
      // subtle refresh indicator
    }
  }, [addToast])

  const wsStatus = useWebSocket(handleWsMessage)

  // Initial REST load
  useEffect(() => {
    fetchGrid().then(d => { if (d?.cells) setGridData(d.cells) }).catch(() => {})
    fetchShortfall().then(d => { if (d?.risk_pct !== undefined) setShortfallData(d) }).catch(() => {})
    fetchActions().then(d => { if (d?.actions) setActionsData(d.actions) }).catch(() => {})
  }, [])

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="header">
        <div className="header-brand">
          <div className="header-logo">⛏</div>
          <div>
            <div className="header-title">OreSense AI</div>
            <div className="header-subtitle">MOIL Manganese Intelligence Platform</div>
          </div>
        </div>
        <div className="header-right">
          <div className="ws-status">
            <div className={`ws-dot ${wsStatus}`} />
            <span>{wsStatus === 'connected' ? 'Live' : wsStatus === 'connecting' ? 'Connecting…' : 'Offline'}</span>
          </div>
          {lastUpdate && (
            <span style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
              Updated {new Date(lastUpdate).toLocaleTimeString()}
            </span>
          )}
          <div className="role-selector">
            {ROLES.map(r => (
              <button
                key={r}
                className={`role-btn ${role === r ? 'active' : ''}`}
                onClick={() => setRole(r)}
              >
                {r}
              </button>
            ))}
          </div>
        </div>
      </header>

      {/* ── Nav Tabs ── */}
      <nav className="nav-tabs">
        {TABS.map(tab => {
          const Icon = tab.icon
          return (
            <button
              key={tab.id}
              className={`nav-tab ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              <Icon size={14} />
              {tab.label}
              {tab.id === 'shortfall' && <span className="status-dot" />}
            </button>
          )
        })}
      </nav>

      {/* ── Content ── */}
      <main className="main-content">
        {activeTab === 'reserve'   && <ReserveTab   gridData={gridData} role={role} addToast={addToast} />}
        {activeTab === 'shortfall' && <ShortfallTab shortfallData={shortfallData} role={role} />}
        {activeTab === 'actions'   && <ActionsTab   actionsData={actionsData} shortfallData={shortfallData} addToast={addToast} />}
        {activeTab === 'pipeline'  && <PipelineTab  />}
        {activeTab === 'simulator' && <SimulatorPanel addToast={addToast} />}
      </main>

      {/* ── Toast Notifications ── */}
      <div className="toast-container">
        {toasts.map(t => (
          <div key={t.id} className={`toast ${t.type}`}>
            <span style={{ fontSize: 16 }}>
              {t.type === 'success' ? '✓' : t.type === 'warning' ? '⚠' : '✕'}
            </span>
            <div>
              <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>{t.msg}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
