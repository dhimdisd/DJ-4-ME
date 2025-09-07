#!/usr/bin/env python3
# dj_cli.py
from __future__ import annotations
import argparse, logging
from pathlib import Path
import pandas as pd

import re
from urllib.parse import urlparse, parse_qs

from downloader import (
    create_ydl, list_playlist, download_one, download_playlist
)
from dj_analyzer import analyze_audio, write_track_json

# --- optional tagging deps (for setting MP3 genre) ---
try:
    from mutagen.easyid3 import EasyID3  # type: ignore
    from mutagen.id3 import ID3, ID3NoHeaderError  # type: ignore
    HAS_MUTAGEN = True
except Exception:
    HAS_MUTAGEN = False


YT_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

def _looks_like_playlist_id(s: str) -> bool:
    # Common playlist ID prefixes (PL..., RD..., OLAK..., etc)
    return s.startswith(("PL", "RD", "OLAK", "VLPL", "UU", "LL", "FL"))

def _is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")

def _url_has_list_param(url: str) -> bool:
    try:
        q = parse_qs(urlparse(url).query)
        return "list" in q
    except Exception:
        return False

def _normalize_to_url(arg: str, site: str = "youtube", treat_as_playlist: bool | None = None) -> str:
    """Turn an ID or URL into a canonical URL. If treat_as_playlist is None, just build /watch?v=... for video IDs."""
    if _is_url(arg):
        return arg
    # ID
    if treat_as_playlist is True or (treat_as_playlist is None and _looks_like_playlist_id(arg)):
        return f"https://{'music.' if site=='music' else ''}youtube.com/playlist?list={arg}"
    # else assume video id
    return f"https://{'music.' if site=='music' else ''}youtube.com/watch?v={arg}"

def _probe_is_playlist(url: str, *, codec: str, browser: str | None, cookiefile: str | None) -> bool:
    """Ask yt-dlp (flat) whether this is a playlist (has entries) or a single."""
    ydl = create_ydl(
        listing=True, outdir=Path("."), codec=codec, mp3_bitrate="320",
        cookies_from_browser=browser, cookiefile=cookiefile
    )
    info = ydl.extract_info(url, download=False)
    return bool(info and info.get("entries"))

def _set_mp3_genre(path: Path, genre: str):
    if not HAS_MUTAGEN or not genre:
        return
    try:
        try:
            tags = EasyID3(str(path))
        except ID3NoHeaderError:
            tags = EasyID3()
        tags["genre"] = genre
        tags.save(str(path))
    except Exception as e:
        logging.warning("Failed to set MP3 genre for %s: %s", path, e)



def setup_logging(verbosity: int = 1):
    level = logging.INFO if verbosity == 1 else (logging.DEBUG if verbosity >= 2 else logging.WARNING)
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


# ---- commands ----
def cmd_download_one(args):
    out, _info = download_one(
        video_url=args.url,
        outdir=Path(args.outdir),
        codec=args.codec,
        mp3_bitrate=args.mp3_bitrate,
        cookies_from_browser=args.browser,
        cookiefile=args.cookiefile,
    )
    if out:
        print(out.resolve())


def cmd_download_playlist(args):
    saved = download_playlist(
        playlist_url=args.playlist,
        outdir=Path(args.outdir),
        codec=args.codec,
        mp3_bitrate=args.mp3_bitrate,
        cookies_from_browser=args.browser,
        cookiefile=args.cookiefile,
    )
    print(f"Saved {len(saved)} files in {Path(args.outdir).resolve()}")


def cmd_analyze_local(args):
    folder = Path(args.localdir)
    audio = [p for p in sorted(folder.glob("*")) if p.suffix.lower() in (".wav",".mp3",".flac",".aiff",".aif")]
    if not audio:
        logging.error("No audio files found in %s", folder); return
    rows = []
    for i, path in enumerate(audio, 1):
        logging.info("[LOCAL %d/%d] %s", i, len(audio), path.name)
        meta = analyze_audio(path)
        write_track_json(path, meta)
        rows.append({
            "title": meta.get("title") or path.stem,
            "file_path": str(path),
            "bpm": meta.get("bpm"),
            "camelot": meta.get("camelot"),
            "key_root": meta.get("key_root"),
            "mode": meta.get("mode"),
            "intro_end_s": meta.get("intro_end_s"),
            "outro_start_s": meta.get("outro_start_s"),
        })
    df = pd.DataFrame(rows)
    out_csv = Path(args.csv) if args.csv else (folder / "local_analysis.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} rows → {out_csv.resolve()}")


def cmd_analyze_playlist(args):
    # Option A: just analyze files you downloaded earlier (pass a folder)
    if args.download is False and args.downloaded_dir:
        folder = Path(args.downloaded_dir)
        args2 = argparse.Namespace(localdir=str(folder), csv=args.csv)
        cmd_analyze_local(args2)
        return

    # Option B: list playlist and (optionally) download+analyze
    outdir = Path(args.outdir)
    ydl_list = create_ydl(
        listing=True, outdir=Path("."), codec=args.codec, mp3_bitrate=args.mp3_bitrate,
        cookies_from_browser=args.browser, cookiefile=args.cookiefile,
    )
    items, pl_title = list_playlist(args.playlist, ydl_list)
    if not items:
        logging.error("No items found in playlist"); return

    rows = []
    for i, item in enumerate(items, 1):
        logging.info("[PLAYLIST %d/%d] %s", i, len(items), item.get("title"))
        audio_path = None
        if args.download:
            audio_path, info = download_one(
                item["url"], outdir=outdir, codec=args.codec, mp3_bitrate=args.mp3_bitrate,
                cookies_from_browser=args.browser, cookiefile=args.cookiefile,
            )
        if audio_path and audio_path.exists():
            meta = analyze_audio(audio_path)
            # --- attach source-derived genre/tags ---
            source_tags = (info or {}).get('tags') or []
            categories  = (info or {}).get('categories') or []
            genre_guess = (info or {}).get('genre') or (categories[0] if categories else None)
            meta.update({
                'source_tags': source_tags,
                'source_categories': categories,
                'genre_guess': genre_guess,
                'source_channel': (info or {}).get('channel'),
                'source_uploader': (info or {}).get('uploader'),
            })
            # set MP3 genre tag if applicable
            if genre_guess and str(audio_path).lower().endswith('.mp3'):
                _set_mp3_genre(audio_path, genre_guess)

            # (genre block cleaned)
            write_track_json(audio_path, meta)
        else:
            meta = {"title": item.get("title"), "bpm": None, "camelot": None, "key_root": None, "mode": None,
                    "intro_end_s": None, "outro_start_s": None}
        rows.append({
            "title": meta.get("title") or item.get("title"),
            "youtube_url": item.get("url"),
            "downloaded_path": str(audio_path) if audio_path else "",
            "bpm": meta.get("bpm"),
            "camelot": meta.get("camelot"),
            "key_root": meta.get("key_root"),
            "mode": meta.get("mode"),
            "intro_end_s": meta.get("intro_end_s"),
            "outro_start_s": meta.get("outro_start_s"),
        })

    safe = "".join(ch for ch in pl_title if ch.isalnum() or ch in " _-").strip() or "playlist"
    out_csv = Path(args.csv) if args.csv else (outdir / f"{safe}.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {len(rows)} rows → {out_csv.resolve()}")

def cmd_download_auto(args):
    # Decide playlist vs single based on the arg content / probe
    inp = args.input.strip()
    site = args.site  # "music" or "youtube"
    # Quick heuristics first:
    if _is_url(inp):
        if _url_has_list_param(inp):
            is_playlist = True
            url = inp
        else:
            # Probe (could be a video URL or a playlist URL without ?list)
            url = inp
            is_playlist = _probe_is_playlist(
                url, codec=args.codec,
                browser=args.browser, cookiefile=args.cookiefile
            )
    else:
        # It's an ID: guess from shape, otherwise probe
        if _looks_like_playlist_id(inp):
            is_playlist = True
            url = _normalize_to_url(inp, site=site, treat_as_playlist=True)
        elif YT_VIDEO_ID_RE.match(inp):
            is_playlist = False
            url = _normalize_to_url(inp, site=site, treat_as_playlist=False)
        else:
            # Fallback: assume video, then probe
            url_guess = _normalize_to_url(inp, site=site, treat_as_playlist=False)
            is_playlist = _probe_is_playlist(
                url_guess, codec=args.codec,
                browser=args.browser, cookiefile=args.cookiefile
            )
            url = url_guess

    # Dispatch to existing commands
    if is_playlist:
        # Reuse your existing playlist handler
        class P: pass
        p = P()
        p.playlist = url
        p.outdir = args.outdir
        p.codec = args.codec
        p.mp3_bitrate = args.mp3_bitrate
        p.browser = args.browser
        p.cookiefile = args.cookiefile
        p.force = args.force
        p.no_archive = args.no_archive
        cmd_download_playlist(p)
    else:
        class O: pass
        o = O()
        o.url = url
        o.outdir = args.outdir
        o.codec = args.codec
        o.mp3_bitrate = args.mp3_bitrate
        o.browser = args.browser
        o.cookiefile = args.cookiefile
        o.force = args.force
        o.no_archive = args.no_archive
        cmd_download_one(o)


# ---- CLI ----
def parse_args():
    ap = argparse.ArgumentParser(description="DJ downloader + analyzer CLI")
    ap.add_argument("-v","--verbose", action="count", default=1, help="-v (info), -vv (debug)")

    sub = ap.add_subparsers(dest="cmd", required=True)

    # download-one
    s = sub.add_parser("download-one", help="Download a single YouTube URL")
    s.add_argument("url", help="YouTube video URL")
    s.add_argument("--outdir", default="downloads")
    s.add_argument("--codec", choices=["wav","mp3"], default="wav")
    s.add_argument("--mp3-bitrate", default="320")
    s.add_argument("--browser", choices=["chrome","firefox","edge","brave","chromium"])
    s.add_argument("--cookiefile")
    s.set_defaults(func=cmd_download_one)

    # download-playlist
    s = sub.add_parser("download-playlist", help="Download all audio from a playlist")
    s.add_argument("playlist", help="YouTube playlist URL")
    s.add_argument("--outdir", default="downloads")
    s.add_argument("--codec", choices=["wav","mp3"], default="wav")
    s.add_argument("--mp3-bitrate", default="320")
    s.add_argument("--browser", choices=["chrome","firefox","edge","brave","chromium"])
    s.add_argument("--cookiefile")
    s.set_defaults(func=cmd_download_playlist)

    # download-playlist or song
    s = sub.add_parser("download", help="Auto-detect playlist or single from ID/URL and download")
    s.add_argument("input", help="Playlist ID/URL or Video ID/URL")
    s.add_argument("--site", choices=["music","youtube"], default="youtube", help="Interpret bare IDs for this site")
    s.add_argument("--outdir", required=True)
    s.add_argument("--codec", default="mp3")
    s.add_argument("--mp3-bitrate", default="320")
    s.add_argument("--browser")
    s.add_argument("--cookiefile")
    s.add_argument("--force", action="store_true")
    s.add_argument("--no-archive", action="store_true")
    s.set_defaults(func=cmd_download_auto)

    # analyze-local
    s = sub.add_parser("analyze-local", help="Analyze already-downloaded audio in a folder")
    s.add_argument("localdir", help="Folder with audio files")
    s.add_argument("--csv", help="Output CSV path (default: <localdir>/local_analysis.csv)")
    s.set_defaults(func=cmd_analyze_local)

    # analyze-playlist
    s = sub.add_parser("analyze-playlist", help="List/Download playlist and analyze")
    s.add_argument("playlist", help="YouTube playlist URL")
    s.add_argument("--outdir", default="downloads")
    s.add_argument("--csv", help="Output CSV path (default: <outdir>/<playlist-name>.csv)")
    s.add_argument("--download", action="store_true", help="Actually download audio before analysis")
    s.add_argument("--downloaded-dir", help="If not downloading now, analyze files from this folder")
    s.add_argument("--codec", choices=["wav","mp3"], default="wav")
    s.add_argument("--mp3-bitrate", default="320")
    s.add_argument("--browser", choices=["chrome","firefox","edge","brave","chromium"])
    s.add_argument("--cookiefile")
    s.set_defaults(func=cmd_analyze_playlist)



    return ap.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()