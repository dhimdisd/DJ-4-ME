import os, time, secrets, urllib.parse
import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import RedirectResponse, JSONResponse
from app.core.config import settings

router = APIRouter(tags=["Auth - Spotify"], prefix="/spotify")

AUTH_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"

def _auth_url(state: str) -> str:
    params = {
        "client_id": settings.SPOTIFY_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": settings.SPOTIFY_REDIRECT_URI,
        "scope": settings.SPOTIFY_SCOPES,
        "state": state,
        "show_dialog": "false",
    }
    return f"{AUTH_URL}?{urllib.parse.urlencode(params)}"

@router.get("/status")
async def status(req: Request):
    sess = req.session or {}
    access_token = sess.get("spotify_access_token")
    expires_at = sess.get("spotify_expires_at", 0)
    logged_in = bool(access_token and time.time() < expires_at)
    return {"logged_in": logged_in}

@router.get("/login")
async def login(req: Request):
    # CSRF protection via state
    state = secrets.token_urlsafe(24)
    req.session["spotify_oauth_state"] = state
    return RedirectResponse(url=_auth_url(state))

@router.get("/callback")
async def callback(req: Request, code: str | None = None, state: str | None = None, error: str | None = None):
    if error:
        return RedirectResponse(url="/?auth=error")

    saved_state = req.session.get("spotify_oauth_state")
    if not state or state != saved_state:
        return RedirectResponse(url="/?auth=state_mismatch")

    # Exchange code for tokens
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": settings.SPOTIFY_REDIRECT_URI,
        "client_id": settings.SPOTIFY_CLIENT_ID,
        "client_secret": settings.SPOTIFY_CLIENT_SECRET,
    }
    async with httpx.AsyncClient() as client:
        r = await client.post(TOKEN_URL, data=data, headers={"Content-Type": "application/x-www-form-urlencoded"})
        r.raise_for_status()
        tok = r.json()

    # Save into session cookie
    req.session.update({
        "spotify_access_token": tok["access_token"],
        "spotify_refresh_token": tok.get("refresh_token"),
        "spotify_expires_at": int(time.time()) + int(tok.get("expires_in", 3600)) - 30,  # small safety margin
    })
    # Optional: clean CSRF state
    req.session.pop("spotify_oauth_state", None)

    return RedirectResponse(url="/")

@router.post("/logout")
async def logout(req: Request):
    # Clear spotify keys from session
    for k in ["spotify_access_token", "spotify_refresh_token", "spotify_expires_at"]:
        req.session.pop(k, None)
    return JSONResponse({"ok": True})

@router.get("/me")
async def me(req: Request):
    token = req.session.get("spotify_access_token")
    if not token or time.time() >= req.session.get("spotify_expires_at", 0):
        return JSONResponse({"error": "not_authenticated"}, status_code=401)

    async with httpx.AsyncClient() as client:
        r = await client.get(
            "https://api.spotify.com/v1/me",
            headers={"Authorization": f"Bearer {token}"}
        )
    return JSONResponse(r.json(), status_code=r.status_code)
