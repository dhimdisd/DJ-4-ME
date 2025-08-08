## GPT links

-- Project set up
https://chatgpt.com/canvas/shared/68957f4cda0c8191bdc12672ba2949c3

-- AI DJ
https://chatgpt.com/share/68957f9d-d524-800e-b39e-a0182d119e63

# Overview

A minimal, production-friendly setup where a **Python API (FastAPI)** serves data and—after you build the SPA—also serves your **React** front end. During development, you’ll run them separately with a dev proxy for `/api`.

---

## Project structure

```
fastapi-react-spa/
├─ backend/
│  ├─ main.py
│  ├─ requirements.txt
│  └─ frontend_dist/           # created after you build the React app (copied in)
├─ frontend/
│  ├─ index.html
│  ├─ package.json
│  ├─ vite.config.ts
│  └─ src/
│     ├─ main.tsx
│     └─ App.tsx
├─ .gitignore
└─ Dockerfile                  # optional: single image that builds SPA & runs API
```

---

## Backend (FastAPI)

**backend/requirements.txt**

```
fastapi==0.115.0
uvicorn[standard]==0.30.6
```

**backend/main.py**

```python
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="FastAPI + React SPA")

# CORS for local dev (Vite runs on 5173 by default)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.get("/api/greeting")
def greeting(name: str = "world"):
    return {"message": f"Hello, {name}!"}

# === Static file serving for production ===
# After building the React app, copy its dist/ into backend/frontend_dist/
BUILD_DIR = os.path.join(os.path.dirname(__file__), "frontend_dist")
if os.path.isdir(BUILD_DIR):
    app.mount("/", StaticFiles(directory=BUILD_DIR, html=True), name="spa")

    # Client-side routing fallback (e.g., /dashboard)
    @app.get("/{full_path:path}")
    async def spa_fallback(full_path: str):
        index = os.path.join(BUILD_DIR, "index.html")
        return FileResponse(index)
```

**Run the API (dev):**

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

---

## Frontend (React + Vite)

**frontend/package.json**

```json
{
  "name": "react-spa",
  "private": true,
  "version": "0.0.1",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview --port 5173"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  },
  "devDependencies": {
    "@types/react": "^18.3.5",
    "@types/react-dom": "^18.3.0",
    "@vitejs/plugin-react": "^4.3.1",
    "typescript": "^5.5.4",
    "vite": "^5.4.3"
  }
}
```

**frontend/vite.config.ts**

```ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000' // dev: pass API calls to FastAPI
    }
  },
  build: {
    outDir: 'dist'
  }
})
```

**frontend/index.html**

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>FastAPI + React</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

**frontend/src/main.tsx**

```tsx
import React from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'

createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
```

**frontend/src/App.tsx**

```tsx
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
```

**Run the SPA (dev):**

```bash
cd frontend
npm i
npm run dev
# open http://localhost:5173
```

---

## Build the SPA and let FastAPI serve it (prod-like)

```bash
# from project root
cd frontend
npm run build
# copy the build output into the backend
rm -rf ../backend/frontend_dist && mkdir -p ../backend/frontend_dist
cp -r dist/* ../backend/frontend_dist/

# run the API (now also serving the SPA)
cd ../backend
uvicorn main:app --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

---

## Optional: Single Docker image (builds SPA + runs API)

**Dockerfile** (at repo root)

```dockerfile
# --- Build the React app ---
FROM node:20-alpine AS web
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend .
RUN npm run build

# --- Python API ---
FROM python:3.12-slim AS api
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY backend ./backend
# bring in built SPA
RUN mkdir -p backend/frontend_dist
COPY --from=web /app/frontend/dist/ ./backend/frontend_dist/

EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build & run:**

```bash
docker build -t fastapi-react-spa .
docker run --rm -p 8000:8000 fastapi-react-spa
# open http://localhost:8000
```

---

## Notes & tips

* **Client routing**: The `spa_fallback` in `main.py` ensures `/some/client/route` serves `index.html`.
* **CORS**: Only needed in dev because Vite runs on 5173. In prod (SPA served by FastAPI), you can remove/lock down CORS.
* **Env config**: Add `.env` and read with `pydantic-settings` if you need secrets.
* **Testing**: Use `pytest` + `httpx` for API tests.
* **Security**: For auth, add `fastapi-users` or roll JWT with `python-jose` + `passlib`.

---

## What next?

Tell me if you want auth, a DB (Postgres + SQLModel/SQLAlchemy), or a sample CRUD module. I can drop those in next.
