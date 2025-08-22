#!/usr/bin/env python3
# downloader.py
from __future__ import annotations
import logging, shutil, traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import yt_dlp
    HAS_YT_DLP = True
except Exception:
    yt_dlp = None
    HAS_YT_DLP = False


def ffmpeg_ok() -> bool:
    ok = shutil.which("ffmpeg") is not None
    if not ok:
        logging.warning("FFmpeg not found on PATH. Install it for audio extraction/conversion.")
    return ok


def build_ydl_opts(
    outdir: Path,
    prefer_codec: str = "wav",
    prefer_quality: str = "0",
    cookies_from_browser: Optional[str] = None,
    cookiefile: Optional[str] = None,
    extract_flat: bool = False,
) -> Dict:
    ydl_opts: Dict = {
        "quiet": True,
        "outtmpl": str(outdir / "%(title)s.%(ext)s"),
    }
    if extract_flat:
        ydl_opts["extract_flat"] = True
        ydl_opts["skip_download"] = True
    else:
        ydl_opts["noplaylist"] = True
        ydl_opts["format"] = "bestaudio/best"
        ydl_opts["postprocessors"] = [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": prefer_codec,      # "wav" or "mp3"
            "preferredquality": prefer_quality,  # "0" for wav; "192" etc. for mp3
        }]
    if cookies_from_browser:
        ydl_opts["cookiesfrombrowser"] = (cookies_from_browser,)
        logging.info("Using cookies from browser: %s", cookies_from_browser)
    elif cookiefile:
        ydl_opts["cookiefile"] = cookiefile
        logging.info("Using cookiefile: %s", cookiefile)
    return ydl_opts


def _ensure_yt_dlp():
    if not HAS_YT_DLP:
        raise RuntimeError("yt_dlp is not installed. pip install yt-dlp")


def create_ydl(
    listing: bool,
    outdir: Path,
    codec: str,
    mp3_bitrate: str,
    cookies_from_browser: Optional[str],
    cookiefile: Optional[str],
):
    _ensure_yt_dlp()
    quality = "0" if codec == "wav" else mp3_bitrate
    opts = build_ydl_opts(
        outdir=outdir,
        prefer_codec=codec,
        prefer_quality=quality,
        cookies_from_browser=cookies_from_browser,
        cookiefile=cookiefile,
        extract_flat=listing,
    )
    return yt_dlp.YoutubeDL(opts)


def list_playlist(playlist_url: str, ydl_list) -> Tuple[List[Dict], str]:
    """Return (entries, playlist_title) with flat items: title, url, id, duration."""
    logging.info("Fetching playlist: %s", playlist_url)
    try:
        info = ydl_list.extract_info(playlist_url, download=False)
        entries = info.get("entries", []) or []
        playlist_title = info.get("title") or "playlist"
        logging.info("Found %d videos in playlist: %s", len(entries), playlist_title)
        flat = []
        for e in entries:
            if not e:
                continue
            vid = e.get("id")
            flat.append({
                "title": e.get("title"),
                "video_id": vid,
                "url": f"https://www.youtube.com/watch?v={vid}" if vid else None,
                "duration": e.get("duration"),
            })
        return [x for x in flat if x.get("url")], playlist_title
    except Exception as e:
        logging.error("Failed to fetch playlist: %s", e)
        traceback.print_exc()
        return [], "playlist"


def download_one(
    video_url: str,
    outdir: Path,
    codec: str = "wav",
    mp3_bitrate: str = "192",
    cookies_from_browser: Optional[str] = None,
    cookiefile: Optional[str] = None,
) -> Optional[Path]:
    """Download one track and convert to target codec. Returns final file path or None."""
    _ensure_yt_dlp()
    logging.info("Downloading %s from: %s", codec.upper(), video_url)
    outdir.mkdir(parents=True, exist_ok=True)
    if not ffmpeg_ok():
        logging.warning("FFmpeg is required to convert to %s. Attempting anyway; may fail.", codec.upper())

    ydl_dl = create_ydl(
        listing=False, outdir=outdir, codec=codec, mp3_bitrate=mp3_bitrate,
        cookies_from_browser=cookies_from_browser, cookiefile=cookiefile,
    )
    try:
        info = ydl_dl.extract_info(video_url, download=True)
        candidate = None
        try:
            reqs = info.get("requested_downloads") or []
            if reqs and "filepath" in reqs[0]:
                candidate = Path(reqs[0]["filepath"])
        except Exception:
            pass
        if not candidate:
            title = (info.get("title") or "track").replace("/", "-")
            candidate = next((p for p in outdir.glob(f"{title}.{codec}")), None)
        if candidate and candidate.exists():
            logging.info("Saved %s: %s", codec.upper(), candidate)
            return candidate
        logging.warning("Could not locate downloaded %s file for: %s", codec.upper(), video_url)
        return None
    except Exception as e:
        logging.error("%s download failed for %s: %s", codec.upper(), video_url, e)
        traceback.print_exc()
        return None


def download_playlist(
    playlist_url: str,
    outdir: Path,
    codec: str = "wav",
    mp3_bitrate: str = "192",
    cookies_from_browser: Optional[str] = None,
    cookiefile: Optional[str] = None,
) -> List[Path]:
    """Download all tracks in a playlist. Returns list of file paths that were saved."""
    _ensure_yt_dlp()
    ydl_list = create_ydl(
        listing=True, outdir=Path("."), codec=codec, mp3_bitrate=mp3_bitrate,
        cookies_from_browser=cookies_from_browser, cookiefile=cookiefile,
    )
    items, _ = list_playlist(playlist_url, ydl_list)
    if not items:
        return []

    saved: List[Path] = []
    for i, item in enumerate(items, 1):
        logging.info("[PLAYLIST %d/%d] %s", i, len(items), item.get("title"))
        p = download_one(
            item["url"], outdir=outdir, codec=codec, mp3_bitrate=mp3_bitrate,
            cookies_from_browser=cookies_from_browser, cookiefile=cookiefile,
        )
        if p:
            saved.append(p)
    return saved
