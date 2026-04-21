import { G, mono, inter, CONF_THRESHOLD } from './constants'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'

export function ConfBadge({ value }) {
  const pct   = value !== null ? (value * 100).toFixed(1) : null
  const above = pct !== null && parseFloat(pct) >= CONF_THRESHOLD
  return (
    <div style={{
      background: above ? 'rgba(15,110,86,0.88)' : 'rgba(139,26,26,0.88)',
      color: '#fff', fontFamily: mono, fontSize: 13, fontWeight: 600,
      padding: '5px 10px', borderRadius: 6, letterSpacing: '0.04em',
      transition: 'background 0.4s',
    }}>
      {pct !== null ? `${pct}% conf` : '-- conf'}
    </div>
  )
}

export function MetricCard({ label, value, unit }) {
  return (
    <div style={{
      background: G.surface, border: `1px solid ${G.border}`,
      borderTop: `3px solid ${G.greenMid}`, borderRadius: 8,
      padding: '10px 16px', flex: 1,
    }}>
      <div style={{ fontSize: 10, color: G.muted, fontFamily: inter,
        letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 6 }}>
        {label}
      </div>
      <div style={{ fontSize: 22, fontWeight: 600, fontFamily: mono, color: G.green }}>
        {value}
        <span style={{ fontSize: 11, color: G.muted, marginLeft: 4,
          fontFamily: inter, fontWeight: 400 }}>{unit}</span>
      </div>
    </div>
  )
}

export function MiniChart({ data, dataKey, color, domain, threshold }) {
  return (
    <ResponsiveContainer width="100%" height={90}>
      <LineChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
        <XAxis dataKey="t" hide />
        <YAxis
          domain={domain}
          width={28}
          tick={{ fontSize: 9, fontFamily: mono, fill: G.muted }}
          tickCount={3}
        />
        <Tooltip
          contentStyle={{ fontSize: 11, fontFamily: mono,
            background: G.surface, border: `1px solid ${G.border}`,
            borderRadius: 6 }}
          labelStyle={{ display: 'none' }}
        />
        {threshold !== undefined && (
          <ReferenceLine y={threshold} stroke={G.amber} strokeDasharray="4 3" />
        )}
        <Line type="monotone" dataKey={dataKey} stroke={color}
          dot={false} strokeWidth={1.5} isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  )
}
