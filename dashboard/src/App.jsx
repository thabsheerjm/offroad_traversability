import { useState, useEffect } from 'react'
import { WS_URL } from './constants'
import PreSession  from './PreSession'
import LiveSession from './LiveSession'
import PostSession from './PostSession'

export default function App() {
  const [screen, setScreen]   = useState('pre')
  const [summary, setSummary] = useState(null)
  const [ws, setWs]           = useState(null)

  useEffect(() => {
    const sock = new WebSocket(WS_URL)
    sock.onopen  = () => setWs(sock)
    sock.onerror = () => console.error('WebSocket error')
    return () => sock.close()
  }, [])

  const handleStart   = () => setScreen('live')
  const handleEnd     = (s) => { setSummary(s); setScreen('post') }
  const handleRestart = () => { setSummary(null); setScreen('pre') }

  if (screen === 'pre')
    return <PreSession ws={ws} onStart={handleStart} />
  if (screen === 'live')
    return <LiveSession ws={ws} onEnd={handleEnd} />
  if (screen === 'post')
    return <PostSession summary={summary} onRestart={handleRestart} />
}
