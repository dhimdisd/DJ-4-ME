import React, { useEffect, useState } from 'react'

export default function App() {
  const [loggedIn, setLoggedIn] = useState<boolean | null>(null)
  const [message, setMessage] = useState<string>('...')

  useEffect(() => {
    fetch('/api/auth/spotify/status', { credentials: 'include' })
      .then(r => r.json())
      .then(d => setLoggedIn(!!d.logged_in))
      .catch(() => setLoggedIn(false))

    fetch('/api/greeting?name=You')
      .then(r => r.json())
      .then(d => setMessage(d.message))
      .catch(() => setMessage('API not reachable'))
  }, [])

  if (loggedIn === null) {
    return <main style={{ padding: 24 }}>Loading…</main>
  }

  return (
    <main style={{ fontFamily: 'system-ui, sans-serif', padding: 24 }}>
      <h1>FastAPI + React SPA</h1>
      <p>{message}</p>

      {loggedIn ? (
        <div>
          <p>You're logged in with Spotify ✅</p>
          <form method="post" action="/api/auth/spotify/logout">
            <button type="submit">Log out</button>
          </form>
        </div>
      ) : (
        <a href="/api/auth/spotify/login">
          <button>Log in with Spotify</button>
        </a>
      )}
    </main>
  )
}
