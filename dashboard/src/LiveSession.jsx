import { useState, useEffect, useRef } from 'react'
import { G, mono, inter, CONF_THRESHOLD, elapsed } from './constants'
import { MetricCard, MiniChart, ConfBadge } from './components'

export default function LiveSession({ ws, onEnd }) {
  const [frame, setFrame]     = useState(null)
  const [history, setHistory] = useState([])
  const [startTs]             = useState(Date.now())
  const [timer, setTimer]     = useState('00:00')
  const imgRef  = useRef(null)
  const histRef = useRef([])

  useEffect(() => {
    const interval = setInterval(() => setTimer(elapsed(startTs)), 1000)
    return () => clearInterval(interval)
  }, [startTs])

  useEffect(() => {
    if (!ws) return
    ws.onmessage = (e) => {
      const data = JSON.parse(e.data)
      if (data.event) return
      setFrame(data)
      const entry = {
        t:    histRef.current.length,
        fps:  +data.display_fps.toFixed(1),
        inf:  +data.inference_ms.toFixed(1),
        conf: +(data.confidence * 100).toFixed(1),
      }
      histRef.current = [...histRef.current, entry].slice(-120)
      setHistory([...histRef.current])
      if (imgRef.current && data.overlay_jpg_b64) {
        imgRef.current.src = 'data:image/jpeg;base64,' + data.overlay_jpg_b64
      }
    }
  }, [ws])

  const handleEnd = () => {
    ws?.send(JSON.stringify({ cmd: 'end_session' }))
    const h = histRef.current
    const mean = arr => arr.length
      ? (arr.reduce((a, b) => a + b, 0) / arr.length).toFixed(1) : '--'
    onEnd({
      duration: timer,
      frames:   h.length,
      meanConf: mean(h.map(x => x.conf)),
      meanFps:  mean(h.map(x => x.fps)),
      meanInf:  mean(h.map(x => x.inf)),
      model:    frame?.model_id ?? '--',
      history:  h,
    })
  }

  return (
    <div style={{ minHeight: '100vh', background: G.bg,
      fontFamily: inter, padding: '14px 18px' }}>

      <div style={{ display: 'flex', alignItems: 'center',
        justifyContent: 'space-between', marginBottom: 14 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div className="pulse-dot" />
          <span style={{ fontSize: 13, fontWeight: 600, color: G.text }}>
            Offroad Traversability
          </span>
          <span style={{ fontSize: 11, fontFamily: mono, color: G.greenMid,
            background: G.greenLight, padding: '2px 8px', borderRadius: 4 }}>
            {frame?.model_id ?? 'waiting...'}
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <span style={{ fontFamily: mono, fontSize: 13, color: G.muted }}>
            {timer}
          </span>
          <button onClick={handleEnd} style={{
            background: G.red, color: '#fff', border: 'none',
            borderRadius: 6, padding: '7px 16px', fontSize: 12,
            fontWeight: 600, fontFamily: inter, cursor: 'pointer',
          }}>
            End Session
          </button>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: 14 }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          <div style={{ display: 'flex', gap: 10 }}>
            <MetricCard label="FPS"
              value={frame ? frame.display_fps.toFixed(1) : '--'} unit="fps" />
            <MetricCard label="Inference"
              value={frame ? frame.inference_ms.toFixed(1) : '--'} unit="ms" />
            <MetricCard label="Preprocess"
              value={frame ? frame.preprocess_ms.toFixed(1) : '--'} unit="ms" />
          </div>

          <div style={{
            background: G.surface, border: `1px solid ${G.border}`,
            borderRadius: 10, overflow: 'hidden', position: 'relative',
            minHeight: 320,
          }}>
            <img ref={imgRef} alt="overlay"
              style={{ width: '100%', display: 'block' }} />
            <div style={{ position: 'absolute', top: 12, right: 12 }}>
              <ConfBadge value={frame ? frame.confidence : null} />
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {[
            { label: 'Confidence %',         key: 'conf', color: G.red,      domain: [0,100], threshold: CONF_THRESHOLD },
            { label: 'Inference latency ms',  key: 'inf',  color: G.green,    domain: [0,60]  },
            { label: 'FPS',                   key: 'fps',  color: G.greenMid, domain: [0,80]  },
          ].map(({ label, key, color, domain, threshold }) => (
            <div key={key} style={{
              background: G.surface, border: `1px solid ${G.border}`,
              borderRadius: 8, padding: '12px 14px',
            }}>
              <div style={{ fontSize: 10, color: G.muted, letterSpacing: '0.06em',
                textTransform: 'uppercase', marginBottom: 8 }}>
                {label}
              </div>
              <MiniChart data={history} dataKey={key} color={color}
                domain={domain} threshold={threshold} />
            </div>
          ))}

          <div style={{
            background: G.surface, border: `1px solid ${G.border}`,
            borderRadius: 8, padding: '12px 14px',
          }}>
            <div style={{ fontSize: 10, color: G.muted, letterSpacing: '0.06em',
              textTransform: 'uppercase', marginBottom: 10 }}>
              Frame info
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {[
                ['Model',     frame?.model_id ?? '--'],
                ['Timestamp', frame ? String(frame.timestamp_us).slice(-8) : '--'],
                ['Frames',    history.length],
              ].map(([k, v]) => (
                <div key={k} style={{ display: 'flex',
                  justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ color: G.muted, fontSize: 11 }}>{k}</span>
                  <span style={{ fontFamily: mono, fontSize: 11,
                    color: G.text }}>{v}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
