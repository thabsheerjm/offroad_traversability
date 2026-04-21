import { useState, useEffect } from 'react'
import { G, mono, inter } from './constants'

export default function PreSession({ ws, onStart }) {
  const [videos, setVideos]     = useState([])
  const [selected, setSelected] = useState('')

  useEffect(() => {
    if (!ws) return
    ws.onmessage = (e) => {
      const data = JSON.parse(e.data)
      if (data.event === 'video_list') {
        setVideos(data.videos)
        setSelected(data.videos[0] || '')
      }
      if (data.event === 'session_started') onStart()
    }
    ws.send(JSON.stringify({ cmd: 'get_videos' }))
  }, [ws])

  const handleStart = () => {
    if (!selected) return
    ws?.send(JSON.stringify({ cmd: 'start_session', video: selected }))
  }

  return (
    <div style={{
      minHeight: '100vh', background: G.bg, fontFamily: inter,
      display: 'flex', flexDirection: 'column',
      alignItems: 'center', justifyContent: 'center', gap: 32,
    }}>
      <div style={{ textAlign: 'center' }}>
        <div style={{ fontSize: 11, color: G.muted, letterSpacing: '0.12em',
          textTransform: 'uppercase', marginBottom: 8 }}>
          Field Test Dashboard
        </div>
        <div style={{ fontSize: 28, fontWeight: 600, color: G.text }}>
          Offroad Traversability
        </div>
      </div>

      <div style={{
        background: G.surface, border: `1px solid ${G.border}`,
        borderRadius: 12, padding: '28px 32px', width: 380,
        display: 'flex', flexDirection: 'column', gap: 20,
      }}>
        <div>
          <div style={{ fontSize: 11, color: G.muted, letterSpacing: '0.06em',
            textTransform: 'uppercase', marginBottom: 8 }}>
            Model
          </div>
          <div style={{
            background: G.greenLight, border: `1px solid ${G.border}`,
            borderRadius: 6, padding: '10px 14px',
            fontFamily: mono, fontSize: 13, color: G.green,
          }}>
            deeplabv3_mnv3_finetuned
          </div>
        </div>

        <div>
          <div style={{ fontSize: 11, color: G.muted, letterSpacing: '0.06em',
            textTransform: 'uppercase', marginBottom: 8 }}>
            Video source
          </div>
          <select
            value={selected}
            onChange={e => setSelected(e.target.value)}
            style={{
              width: '100%', padding: '10px 14px', borderRadius: 6,
              border: `1px solid ${G.border}`, background: G.surface,
              fontFamily: mono, fontSize: 13, color: G.text,
              appearance: 'none', cursor: 'pointer',
            }}
          >
            {videos.map(v => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
        </div>

        <div>
          <div style={{ fontSize: 11, color: G.muted, letterSpacing: '0.06em',
            textTransform: 'uppercase', marginBottom: 8 }}>
            System status
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {[
              ['Bridge', !!ws,  ws ? 'Connected' : 'Connecting...'],
              ['Model',  true,  'Loaded'],
              ['Camera', false, 'Video mode'],
            ].map(([label, ok, note]) => (
              <div key={label} style={{ display: 'flex', alignItems: 'center',
                justifyContent: 'space-between', fontSize: 12 }}>
                <span style={{ color: G.muted }}>{label}</span>
                <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={{ width: 7, height: 7, borderRadius: '50%',
                    background: ok ? G.greenMid : G.amber,
                    display: 'inline-block' }}/>
                  <span style={{ fontFamily: mono, color: G.text }}>{note}</span>
                </span>
              </div>
            ))}
          </div>
        </div>

        <button
          onClick={handleStart}
          disabled={!selected || !ws}
          style={{
            background: G.green, color: '#fff', border: 'none',
            borderRadius: 8, padding: '13px 0', fontSize: 14,
            fontWeight: 600, fontFamily: inter, cursor: 'pointer',
            letterSpacing: '0.04em', marginTop: 4,
            opacity: selected && ws ? 1 : 0.5,
          }}
        >
          Start Session
        </button>
      </div>
    </div>
  )
}
