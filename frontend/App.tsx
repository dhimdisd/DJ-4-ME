import React, { useEffect, useState } from 'react'

export default function App() {
  const [message, setMessage] = useState<string>('...')

  useEffect(() => {
    fetch('/api/greeting?name=You')
      .then(r => r.json())
      .then(d => setMessage(d.message))
      .catch(() => setMessage('API not reachable'))
  }, [])

  return (
    <main style={{ fontFamily: 'system-ui, sans-serif', padding: 24 }}>
      <h1>FastAPI + React SPA</h1>
      <p>{message}</p>
      <p>Try navigating to a client route like <code>/dashboard</code> once built.</p>
    </main>
  )
}
