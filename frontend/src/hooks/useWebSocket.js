/**
 * useWebSocket.js — Phase 3 Addition
 * Real-time score updates via WebSocket connection
 */
import { useState, useEffect, useRef, useCallback } from 'react'
import { WS_BASE_URL } from '../constants/config'

/**
 * Hook for live anomaly score updates via WebSocket
 * @param {string} asset - Asset ticker (e.g., 'SP500')
 * @returns {Object} { data, connected, error, reconnect }
 */
export function useLiveScores(asset) {
  const [data, setData] = useState(null)
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState(null)
  const ws = useRef(null)
  const reconnectTimeout = useRef(null)

  const connect = useCallback(() => {
    if (!asset) return

    // Clean up existing connection
    if (ws.current) {
      ws.current.close()
    }

    const url = `${WS_BASE_URL}/ws/scores/${asset}`
    ws.current = new WebSocket(url)

    ws.current.onopen = () => {
      setConnected(true)
      setError(null)
      console.log(`[WS] Connected to ${asset}`)
    }

    ws.current.onclose = (event) => {
      setConnected(false)
      console.log(`[WS] Disconnected from ${asset}`, event.code)

      // Auto-reconnect after 5 seconds if not a clean close
      if (event.code !== 1000) {
        reconnectTimeout.current = setTimeout(() => {
          console.log(`[WS] Attempting reconnect to ${asset}...`)
          connect()
        }, 5000)
      }
    }

    ws.current.onerror = (err) => {
      setError('WebSocket connection failed')
      console.error(`[WS] Error for ${asset}:`, err)
    }

    ws.current.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data)
        setData(payload)
      } catch (e) {
        console.error('[WS] Failed to parse message:', e)
      }
    }
  }, [asset])

  useEffect(() => {
    connect()

    return () => {
      if (ws.current) {
        ws.current.close(1000) // Clean close
      }
      if (reconnectTimeout.current) {
        clearTimeout(reconnectTimeout.current)
      }
    }
  }, [connect])

  const reconnect = useCallback(() => {
    if (reconnectTimeout.current) {
      clearTimeout(reconnectTimeout.current)
    }
    connect()
  }, [connect])

  return { data, connected, error, reconnect }
}

export default useLiveScores
