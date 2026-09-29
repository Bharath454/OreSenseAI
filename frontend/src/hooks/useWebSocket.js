import { useEffect, useRef, useState, useCallback } from 'react'

const WS_BASE = import.meta.env.VITE_WS_URL || 'ws://localhost:8000'

export function useWebSocket(onMessage) {
  const [status, setStatus] = useState('connecting') // connecting | connected | disconnected
  const ws = useRef(null)
  const reconnectTimer = useRef(null)
  const onMessageRef = useRef(onMessage)
  onMessageRef.current = onMessage

  const connect = useCallback(() => {
    setStatus('connecting')
    try {
      const socket = new WebSocket(`${WS_BASE}/ws`)
      ws.current = socket

      socket.onopen = () => {
        setStatus('connected')
        // Keep-alive ping every 25s
        socket._pingInterval = setInterval(() => {
          if (socket.readyState === WebSocket.OPEN) socket.send('ping')
        }, 25_000)
      }

      socket.onmessage = (evt) => {
        try {
          const data = JSON.parse(evt.data)
          if (data.type !== 'pong') onMessageRef.current(data)
        } catch (_) {}
      }

      socket.onclose = () => {
        setStatus('disconnected')
        clearInterval(socket._pingInterval)
        // Reconnect after 3 seconds
        reconnectTimer.current = setTimeout(connect, 3_000)
      }

      socket.onerror = () => {
        socket.close()
      }
    } catch (e) {
      setStatus('disconnected')
      reconnectTimer.current = setTimeout(connect, 3_000)
    }
  }, [])

  useEffect(() => {
    connect()
    return () => {
      clearTimeout(reconnectTimer.current)
      if (ws.current) {
        ws.current.onclose = null
        ws.current.close()
      }
    }
  }, [connect])

  return status
}
