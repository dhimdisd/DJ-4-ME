#!/usr/bin/env python3
# dj_cli.py
from __future__ import annotations
import argparse, logging
from pathlib import Path
import pandas as pd

from downloader import (
    create_ydl, list_playlist, download_one, download_playlist
)
from dj_analyzer import analyze_audio, write_track_json


def setup_logging(verbosity: int = 1):
    level = logging.INFO if verbosity == 1 else (logging.DEBUG if verbosity >= 2 else logging.WARNING)
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


# ---- commands ----
def cmd_download_one(args):
    out = download_one(
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
            audio_path = download_one(
                item["url"], outdir=outdir, codec=args.codec, mp3_bitrate=args.mp3_bitrate,
                cookies_from_browser=args.browser, cookiefile=args.cookiefile,
            )
        if audio_path and audio_path.exists():
            meta = analyze_audio(audio_path)
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
    s.add_argument("--mp3-bitrate", default="192")
    s.add_argument("--browser", choices=["chrome","firefox","edge","brave","chromium"])
    s.add_argument("--cookiefile")
    s.set_defaults(func=cmd_download_one)

    # download-playlist
    s = sub.add_parser("download-playlist", help="Download all audio from a playlist")
    s.add_argument("playlist", help="YouTube playlist URL")
    s.add_argument("--outdir", default="downloads")
    s.add_argument("--codec", choices=["wav","mp3"], default="wav")
    s.add_argument("--mp3-bitrate", default="192")
    s.add_argument("--browser", choices=["chrome","firefox","edge","brave","chromium"])
    s.add_argument("--cookiefile")
    s.set_defaults(func=cmd_download_playlist)

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
    s.add_argument("--mp3-bitrate", default="192")
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
