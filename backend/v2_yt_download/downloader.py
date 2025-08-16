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


def _ensure_yt_dlp():
    if not HAS_YT_DLP:
        raise RuntimeError("yt_dlp is not installed. pip install yt-dlp")


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
            "preferredquality": prefer_quality,  # "0" for wav; "192" for mp3, etc.
        }]
    if cookies_from_browser:
        ydl_opts["cookiesfrombrowser"] = (cookies_from_browser,)
        logging.info("Using cookies from browser: %s", cookies_from_browser)
    elif cookiefile:
        ydl_opts["cookiefile"] = cookiefile
        logging.info("Using cookiefile: %s", cookiefile)
    return ydl_opts


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


def _fetch_title(video_url: str) -> Optional[str]:
    """Lightweight info probe to get the title (for filename prediction)."""
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
            info = ydl.extract_info(video_url, download=False)
        title = (info.get("title") or "track").replace("/", "-")
        return title
    except Exception as e:
        logging.warning("Could not prefetch title for %s: %s", video_url, e)
        return None


def _expected_path(outdir: Path, title: Optional[str], codec: str) -> Optional[Path]:
    if not title:
        return None
    return outdir / f"{title}.{codec}"


def download_one(
    video_url: str,
    outdir: Path,
    codec: str = "wav",
    mp3_bitrate: str = "192",
    cookies_from_browser: Optional[str] = None,
    cookiefile: Optional[str] = None,
) -> Optional[Path]:
    """
    Download one track and convert to target codec.
    Skip if the expected file already exists.
    Returns final file path or None.
    """
    _ensure_yt_dlp()
    outdir.mkdir(parents=True, exist_ok=True)

    # Predict filename; if present, skip
    title = _fetch_title(video_url)
    expected = _expected_path(outdir, title, codec)
    if expected and expected.exists():
        logging.info("Already downloaded, skipping: %s", expected)
        return expected

    logging.info("Downloading %s from: %s", codec.upper(), video_url)
    if not ffmpeg_ok():
        logging.warning("FFmpeg is required to convert to %s. Attempting anyway; may fail.", codec.upper())

    ydl_dl = create_ydl(
        listing=False, outdir=outdir, codec=codec, mp3_bitrate=mp3_bitrate,
        cookies_from_browser=cookies_from_browser, cookiefile=cookiefile,
    )
    try:
        info = ydl_dl.extract_info(video_url, download=True)

        # Try requested_downloads (newer yt-dlp)
        candidate = None
        try:
            reqs = info.get("requested_downloads") or []
            if reqs and "filepath" in reqs[0]:
                candidate = Path(reqs[0]["filepath"])
        except Exception:
            pass

        # Fallback: inferred by title
        if not candidate:
            if not title:
                title = (info.get("title") or "track").replace("/", "-")
            candidate = outdir / f"{title}.{codec}"

        if candidate.exists():
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
    """
    Download all tracks in a playlist, skipping files already present.
    Returns list of file paths that were saved or already existed.
    """
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
        url = item["url"]
        title = (item.get("title") or "track").replace("/", "-")
        expected = _expected_path(outdir, title, codec)

        if expected and expected.exists():
            logging.info("[SKIP %d/%d] Already exists: %s", i, len(items), expected.name)
            saved.append(expected)
            continue

        logging.info("[DL %d/%d] %s", i, len(items), title)
        p = download_one(
            url, outdir=outdir, codec=codec, mp3_bitrate=mp3_bitrate,
            cookies_from_browser=cookies_from_browser, cookiefile=cookiefile,
        )
        if p:
            saved.append(p)
    return saved
