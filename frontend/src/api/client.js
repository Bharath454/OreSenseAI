import axios from 'axios'

const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({ baseURL: BASE, timeout: 30_000 })

export const fetchGrid        = ()          => api.get('/api/grid').then(r => r.data)
export const fetchSector      = (id)        => api.get(`/api/sector/${id}`).then(r => r.data)
export const fetchShortfall   = ()          => api.get('/api/shortfall').then(r => r.data)
export const fetchActions     = ()          => api.get('/api/actions').then(r => r.data)
export const fetchCausal      = ()          => api.get('/api/causal').then(r => r.data)
export const fetchFederated   = ()          => api.get('/api/federated/status').then(r => r.data)
export const fetchSources     = ()          => api.get('/api/sources/status').then(r => r.data)
export const fetchIoT         = ()          => api.get('/api/iot').then(r => r.data)
export const fetchModels      = ()          => api.get('/api/models').then(r => r.data)

export const postSimEvent = (event_type, parameters, duration_minutes = 60) =>
  api.post('/api/simulate/event', { event_type, parameters, duration_minutes }).then(r => r.data)

export const postGroundTruth = (cell_id, actual_grade, notes = '') =>
  api.post('/api/groundtruth', { cell_id, actual_grade, notes }).then(r => r.data)

export default api
