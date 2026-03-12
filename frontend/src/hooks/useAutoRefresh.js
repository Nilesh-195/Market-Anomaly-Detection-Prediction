import { useEffect, useRef } from 'react'
import { REFRESH_INTERVAL_MS } from '../constants/config'

export function useAutoRefresh(callback, interval = REFRESH_INTERVAL_MS) {
  const savedCallback = useRef(callback)

  useEffect(() => { savedCallback.current = callback }, [callback])

  useEffect(() => {
    const id = setInterval(() => savedCallback.current(), interval)
    return () => clearInterval(id)
  }, [interval])
}
