export const BRIDGE_PORT = import.meta.env.VITE_BRIDGE_PORT ?? '8000'
export const WS_URL      = `ws://localhost:${BRIDGE_PORT}/ws`
export const BRIDGE_URL  = `http://localhost:${BRIDGE_PORT}`

export const CONF_THRESHOLD = 35

export const G = {
  bg:         '#f8faf8',
  surface:    '#ffffff',
  border:     '#d4e8d4',
  green:      '#0F6E56',
  greenLight: '#E1F5EE',
  greenMid:   '#1D9E75',
  text:       '#0a1a0a',
  muted:      '#5a7a5a',
  red:        '#8B1A1A',
  redLight:   '#FCEBEB',
  amber:      '#BA7517',
  amberLight: '#FAEEDA',
}

export const mono  = "'JetBrains Mono', 'Fira Mono', monospace"
export const inter = "'Inter', 'Segoe UI', system-ui, sans-serif"

export function elapsed(startTs) {
  const secs = Math.floor((Date.now() - startTs) / 1000)
  const m = String(Math.floor(secs / 60)).padStart(2, '0')
  const s = String(secs % 60).padStart(2, '0')
  return `${m}:${s}`
}
