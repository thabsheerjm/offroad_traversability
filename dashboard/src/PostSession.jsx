import { G, mono, inter, CONF_THRESHOLD, BRIDGE_URL } from './constants'
import { MiniChart } from './components'

export default function PostSession({ summary, onRestart }) {
  const handleDownload = () => {
    window.open(`${BRIDGE_URL}/download_log`, '_blank')
  }

  return (
    <div style={{
      minHeight: '100vh', background: G.bg, fontFamily: inter,
      display: 'flex', flexDirection: 'column',
      alignItems: 'center', justifyContent: 'center', gap: 24,
    }}>
      <div style={{ fontSize: 11, color: G.muted, letterSpacing: '0.12em',
        textTransform: 'uppercase' }}>
        Session complete
      </div>
      <div style={{ fontSize: 24, fontWeight: 600, color: G.text }}>
        Session Report
      </div>

      <div style={{
        background: G.surface, border: `1px solid ${G.border}`,
        borderRadius: 12, padding: '28px 32px', width: 420,
        display: 'flex', flexDirection: 'column', gap: 16,
      }}>
        {[
          ['Duration',        summary.duration],
          ['Total frames',    summary.frames],
          ['Mean confidence', summary.meanConf + '%'],
          ['Mean FPS',        summary.meanFps],
          ['Mean inference',  summary.meanInf + 'ms'],
          ['Model',           summary.model],
        ].map(([k, v]) => (
          <div key={k} style={{ display: 'flex',
            justifyContent: 'space-between', alignItems: 'center',
            borderBottom: `0.5px solid ${G.border}`, paddingBottom: 12 }}>
            <span style={{ fontSize: 12, color: G.muted }}>{k}</span>
            <span style={{ fontFamily: mono, fontSize: 13, color: G.text }}>{v}</span>
          </div>
        ))}

        <div style={{ marginTop: 4 }}>
          <div style={{ fontSize: 10, color: G.muted, letterSpacing: '0.06em',
            textTransform: 'uppercase', marginBottom: 8 }}>
            Confidence over session
          </div>
          <MiniChart data={summary.history} dataKey="conf"
            color={G.green} domain={[0, 100]} threshold={CONF_THRESHOLD} />
        </div>

        <div style={{ display: 'flex', gap: 10, marginTop: 8 }}>
          <button onClick={handleDownload} style={{
            flex: 1, background: G.green, color: '#fff', border: 'none',
            borderRadius: 8, padding: '12px 0', fontSize: 13,
            fontWeight: 600, fontFamily: inter, cursor: 'pointer',
          }}>
            Download log
          </button>
          <button onClick={onRestart} style={{
            flex: 1, background: G.surface, color: G.green,
            border: `1px solid ${G.border}`, borderRadius: 8,
            padding: '12px 0', fontSize: 13,
            fontWeight: 600, fontFamily: inter, cursor: 'pointer',
          }}>
            New session
          </button>
        </div>
      </div>
    </div>
  )
}
