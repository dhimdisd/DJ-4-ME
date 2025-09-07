#!/usr/bin/env python3
# downloader.py
from __future__ import annotations
import logging, shutil, traceback, csv, datetime, os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os, json, datetime

try:
    import yt_dlp
    HAS_YT_DLP = True
except Exception:
    yt_dlp = None
    HAS_YT_DLP = False


LEDGER_NAME = "_download_ledger.csv"  # lives under downloads/ (parent of audio/)
# -------------------------------------------------------------

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
    force: bool = False,
) -> Dict:
    """Build yt-dlp options. Title-based filenames.
       If force=True, omit download_archive and enable overwrites.
    """
    ydl_opts: Dict[str, Any] = {
        "quiet": False,
        "noprogress": True,
        "outtmpl": str(outdir / "%(title)s.%(ext)s"),
        "continuedl": True,
    }

    if not force:
        ydl_opts["download_archive"] = str(outdir.parent / "_download_archive.txt")
        ydl_opts["overwrites"] = False
    else:
        # Equivalent to --no-download-archive + --force-overwrites
        ydl_opts["overwrites"] = True

    if extract_flat:
        ydl_opts["extract_flat"] = True
        ydl_opts["skip_download"] = True
    else:
        ydl_opts["noplaylist"] = True
        ydl_opts["format"] = "bestaudio/best"
        ydl_opts["postprocessors"] = [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": prefer_codec,      # "wav" or "mp3"
            "preferredquality": prefer_quality,  # "0" for wav; "192"/"320" etc. for mp3
        }]

    if cookies_from_browser:
        ydl_opts["cookiesfrombrowser"] = (cookies_from_browser,)
        logging.info("Using cookies from browser: %s", cookies_from_browser)
    elif cookiefile:
        ydl_opts["cookiefile"] = str(cookiefile)
        logging.info("Using cookiefile: %s", cookiefile)

    return ydl_opts


def _ensure_yt_dlp():
    if not HAS_YT_DLP:
        raise RuntimeError("yt_dlp is not installed. pip install yt-dlp")


# ----------------------------
# Helpers: filesystem + ledger
# ----------------------------
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _ledger_path_for_outdir(outdir: Path) -> Path:
    """Put the ledger at downloads/_download_ledger.csv (parent of audio/)."""
    # If outdir is .../downloads/audio/<something>, take its parent of "audio"
    if outdir.name == "audio":
        return outdir.parent / LEDGER_NAME
    # Else, if outdir is something like .../downloads/audio/afro..., put ledger in downloads/
    if outdir.parent.name == "audio":
        return outdir.parent.parent / LEDGER_NAME
    # Fallback: adjacent to the current outdir
    return outdir / LEDGER_NAME


def _append_ledger_row(outdir: Path, row: dict):
    """Append to a simple CSV ledger, creating header if missing."""
    path = _ledger_path_for_outdir(outdir)
    _ensure_dir(path.parent)
    fieldnames = [
        "id", "title", "song_url", "filepath", "codec",
        "downloaded_at", "status", "playlist_url", "playlist_title",
    ]
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        rec = {k: row.get(k, "") for k in fieldnames}
        w.writerow(rec)


def _row_from_info(
    info: Dict[str, Any],
    candidate_path: Optional[Path],
    codec: str,
    status: str,
    playlist_url: Optional[str],
    playlist_title: Optional[str] = None,
) -> dict:
    vid   = (info or {}).get("id") or ""
    title = (info or {}).get("title") or ""
    url   = (info or {}).get("webpage_url") or (f"https://www.youtube.com/watch?v={vid}" if vid else "")
    ts    = datetime.datetime.now().isoformat(timespec="seconds")
    return {
        "id": vid,
        "title": title,
        "song_url": url,
        "filepath": str(candidate_path) if candidate_path else "",
        "codec": codec,
        "downloaded_at": ts,
        "status": status,                       # "download" or "skip"
        "playlist_url": playlist_url or "",
        "playlist_title": (playlist_title or "").strip(),
    }


def _expected_output_path(info: Optional[Dict[str, Any]], outdir: Path, ext: str) -> Optional[Path]:
    if not info:
        return None
    title = (info.get("title") or "track").replace("/", "-").strip()
    return outdir / f"{title}.{ext}"


def _create_single_symlink(target: Path, playlists_root: Path):
    """
    Symlink single downloads to:
      <playlists_root>/singles/<filename>
    """
    singles_dir = playlists_root / "singles"
    _ensure_dir(singles_dir)
    link = singles_dir / target.name
    try:
        if link.exists() or link.is_symlink():
            try:
                if link.is_symlink():
                    current = Path(os.readlink(link))
                    if current.resolve() == target.resolve():
                        return link
            except Exception:
                pass
            link.unlink()
        rel_target = os.path.relpath(target, start=singles_dir)
        os.symlink(rel_target, link)
        return link
    except Exception as e:
        logging.warning("Could not create single symlink %s -> %s: %s", link, target, e)
        return None


def _create_playlist_symlink(target: Path, playlists_root: Path, playlist_title: str):
    """
    Symlink playlist downloads to:
      <playlists_root>/<playlist_title>/<filename>
    """
    safe_title = (playlist_title or "playlist").replace("/", "-").strip()
    playlist_dir = playlists_root / safe_title
    _ensure_dir(playlist_dir)
    link = playlist_dir / target.name
    try:
        if link.exists() or link.is_symlink():
            link.unlink()
        rel_target = os.path.relpath(target, start=playlist_dir)
        os.symlink(rel_target, link)
        return link
    except Exception as e:
        logging.warning("Could not create playlist symlink %s -> %s: %s", link, target, e)
        return None


# ----------------------------
# Shared YDL cache (reuse cookies)
# ----------------------------
_SHARED_YDL: dict = {}

def get_shared_ydl(
    listing: bool,
    outdir: Path,
    codec: str,
    mp3_bitrate: str,
    cookies_from_browser: Optional[str],
    cookiefile: Optional[str],
    force: bool = False,
):
    _ensure_yt_dlp()
    quality = "0" if codec == "wav" else mp3_bitrate
    opts = build_ydl_opts(
        outdir=outdir if not listing else Path("."),
        prefer_codec=codec,
        prefer_quality=quality,
        cookies_from_browser=cookies_from_browser,
        cookiefile=cookiefile,
        extract_flat=listing,
        force=force,
    )
    key = (
        "list" if listing else "dl",
        str(outdir.resolve()),
        codec,
        mp3_bitrate,
        cookies_from_browser or "",
        str(cookiefile) if cookiefile else "",
        f"force={force}",
    )
    if key not in _SHARED_YDL:
        _SHARED_YDL[key] = yt_dlp.YoutubeDL(opts)
    return _SHARED_YDL[key]


def create_ydl(
    listing: bool,
    outdir: Path,
    codec: str,
    mp3_bitrate: str,
    cookies_from_browser: Optional[str],
    cookiefile: Optional[str],
    force: bool = False,
):
    _ensure_yt_dlp()
    quality = "0" if codec == "wav" else mp3_bitrate
    opts = build_ydl_opts(
        outdir=outdir if not listing else Path("."),
        prefer_codec=codec,
        prefer_quality=quality,
        cookies_from_browser=cookies_from_browser,
        cookiefile=cookiefile,
        extract_flat=listing,
        force=force,
    )
    return yt_dlp.YoutubeDL(opts)


def list_playlist(playlist_url: str, ydl_list) -> Tuple[List[Dict], str, str]:
    """Return (entries, playlist_title, playlist_id) with flat items: title, url, id, duration."""
    logging.info("Fetching playlist: %s", playlist_url)
    try:
        info = ydl_list.extract_info(playlist_url, download=False)
        entries = info.get("entries", []) or []
        playlist_title = info.get("title") or "playlist"
        playlist_id = info.get("id") or ""   # <— add this
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
        return [x for x in flat if x.get("url")], playlist_title, playlist_id
    except Exception as e:
        logging.error("Failed to fetch playlist: %s", e)
        traceback.print_exc()
        return [], "playlist", ""


# ----------------------------
# Downloaders
# ----------------------------
def download_one_with_ydl(
    video_url: str,
    ydl_dl,
    outdir: Path,
    codec: str = "wav",
    playlist_url: Optional[str] = None,
    cookies_from_browser: Optional[str] = None,
    cookiefile: Optional[str] = None,
    force: bool = False,
    playlist_title: Optional[str] = None,
):
    outdir.mkdir(parents=True, exist_ok=True)
    if not ffmpeg_ok():
        logging.warning("FFmpeg is required to convert to %s. Attempting anyway; may fail.", codec.upper())

    # Probe metadata (same cookies)
    ydl_list = get_shared_ydl(
        listing=True, outdir=Path("."), codec=codec, mp3_bitrate="320",
        cookies_from_browser=cookies_from_browser, cookiefile=cookiefile, force=force,
    )
    try:
        info_probe = ydl_list.extract_info(video_url, download=False)
    except Exception as e:
        logging.error("Metadata probe failed: %s", e)
        traceback.print_exc()
        info_probe = None

    candidate = _expected_output_path(info_probe, outdir, codec)
    if candidate and candidate.exists() and not force:
        logging.info("[SKIP] File already exists: %s", candidate)
        _append_ledger_row(outdir, _row_from_info(info_probe or {}, candidate, codec, "skip", playlist_url, playlist_title))
        # Add playlist symlink even on skip
        playlists_root = outdir.parent / "dj_playlists"
        if playlist_title:
            _create_playlist_symlink(candidate, playlists_root, playlist_title)
        return candidate, info_probe

    logging.info("Downloading %s from: %s", codec.upper(), video_url)
    try:
        info = ydl_dl.extract_info(video_url, download=True)
        final = _expected_output_path(info, outdir, codec)
        if final and final.exists():
            logging.info("[DONE] Saved %s", final)
            _append_ledger_row(outdir, _row_from_info(info or {}, final, codec, "download", playlist_url, playlist_title))
            # Create playlist symlink
            playlists_root = outdir.parent / "dj_playlists"
            if playlist_title:
                _create_playlist_symlink(final, playlists_root, playlist_title)
            return final, info

        # Fallback: try by title or any newly created file
        guess = None
        if info and info.get("title"):
            t = (info["title"]).replace("/", "-").strip()
            guess = next(outdir.glob(f"{t}.*"), None)
        guess = guess or next(outdir.glob("*.*"), None)
        if guess and guess.exists():
            _append_ledger_row(outdir, _row_from_info(info or {}, guess, codec, "download", playlist_url, playlist_title))
            playlists_root = outdir.parent / "dj_playlists"
            if playlist_title:
                _create_playlist_symlink(guess, playlists_root, playlist_title)
            return guess, info

        logging.warning("Download completed but output file not found.")
        return None, info
    except Exception as e:
        logging.error("%s download failed for %s: %s", codec.upper(), video_url, e)
        traceback.print_exc()
        return None, None


def download_one(
    video_url: str,
    outdir: Path,
    codec: str = "wav",
    mp3_bitrate: str = "320",
    cookies_from_browser: Optional[str] = None,
    cookiefile: Optional[str] = None,
    force: bool = False,
) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
    """Single track download. Also symlinks into dj_playlists/singles and logs ledger."""
    _ensure_yt_dlp()
    outdir.mkdir(parents=True, exist_ok=True)
    if not ffmpeg_ok():
        logging.warning("FFmpeg is required to convert to %s. Attempting anyway; may fail.", codec.upper())

    ydl_probe = create_ydl(
        listing=True, outdir=outdir, codec=codec, mp3_bitrate=mp3_bitrate,
        cookies_from_browser=cookies_from_browser, cookiefile=cookiefile, force=force,
    )
    try:
        ydl_info = ydl_probe.extract_info(video_url, download=False)
    except Exception:
        ydl_info = None

    candidate = _expected_output_path(ydl_info, outdir, codec) if ydl_info else None
    if candidate and candidate.exists() and not force:
        logging.info("[SKIP] File already exists: %s", candidate)
        _append_ledger_row(outdir, _row_from_info(ydl_info or {}, candidate, codec, "skip", "singles", "singles"))
        playlists_root = outdir.parent / "dj_playlists"
        _create_single_symlink(candidate, playlists_root)
        return candidate, ydl_info

    ydl_dl = get_shared_ydl(
        listing=False, outdir=outdir, codec=codec, mp3_bitrate=mp3_bitrate,
        cookies_from_browser=cookies_from_browser, cookiefile=cookiefile, force=force,
    )
    try:
        info = ydl_dl.extract_info(video_url, download=True)
        candidate2 = _expected_output_path(info, outdir, codec)
        final_path = candidate2 if (candidate2 and candidate2.exists()) else candidate
        if final_path and final_path.exists():
            _append_ledger_row(outdir, _row_from_info(info or {}, final_path, codec, "download", "singles", "singles"))
            playlists_root = outdir.parent / "dj_playlists"
            _create_single_symlink(final_path, playlists_root)
            return final_path, info
        return None, info
    except Exception as e:
        logging.error("%s download failed for %s: %s", codec.upper(), video_url, e)
        traceback.print_exc()
        return None, None


def download_playlist(
    playlist_url: str,
    outdir: Path,
    codec: str = "wav",
    mp3_bitrate: str = "320",
    cookies_from_browser: Optional[str] = None,
    cookiefile: Optional[str] = None,
) -> List[Path]:
    _ensure_yt_dlp()
    ydl_list = get_shared_ydl(
        listing=True, outdir=Path("."), codec=codec, mp3_bitrate=mp3_bitrate,
        cookies_from_browser=cookies_from_browser, cookiefile=cookiefile,
    )
    items, playlist_title, playlist_id = list_playlist(playlist_url, ydl_list)
    if not items:
        return []

    # where symlinks and ledger/archive live
    downloads_dir   = outdir.parent.resolve()
    playlists_root  = downloads_dir / "dj_playlists"
    playlists_root.mkdir(parents=True, exist_ok=True)

    # playlist dir name: "{title} - {id}"
    safe_title = (playlist_title or "playlist").strip()
    safe_id    = (playlist_id or "").strip()
    pl_dirname = safe_title
    playlist_dir = playlists_root / pl_dirname
    playlist_dir.mkdir(parents=True, exist_ok=True)

    # write playlist.metadata
    meta_path = playlist_dir / "playlist.metadata.json"
    payload = {
        "title": safe_title,
        "id": safe_id,
        "source_url": playlist_url,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "total_items": len(items),
        "codec": codec,
        "audio_outdir": str(outdir.resolve()),
    }
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logging.info("Wrote %s", meta_path)
    except Exception as e:
        logging.warning("Failed to write playlist.metadata: %s", e)

    # shared downloader instance for actual downloads
    ydl_dl = get_shared_ydl(
        listing=False, outdir=outdir, codec=codec, mp3_bitrate=mp3_bitrate,
        cookies_from_browser=cookies_from_browser, cookiefile=cookiefile,
    )

    saved: List[Path] = []
    for i, item in enumerate(items, 1):
        logging.info("[PLAYLIST %d/%d] %s", i, len(items), item.get("title"))
        p, info = download_one_with_ydl(
            item["url"], ydl_dl, outdir=outdir, codec=codec,
            playlist_url=playlist_url,
            cookies_from_browser=cookies_from_browser,
            cookiefile=cookiefile,
            playlist_title=safe_title,
        )
        if p and p.exists():
            saved.append(p)
            # create/refresh symlink inside the playlist folder
            try:
                link = playlist_dir / p.name
                if link.exists() or link.is_symlink():
                    link.unlink()
                rel = os.path.relpath(p, start=playlist_dir)
                link.symlink_to(rel)
            except Exception as e:
                logging.warning("Symlink failed for %s -> %s: %s", link, p, e)

    return saved