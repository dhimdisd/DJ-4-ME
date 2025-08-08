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
