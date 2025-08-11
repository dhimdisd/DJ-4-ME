import argparse
import logging
import shutil
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import yt_dlp
import librosa
import numpy as np
import pandas as pd

# -------------------------
# Logging setup
# -------------------------
LOG_PATH = Path("dj_pipeline.log")

def setup_logging(verbosity: int = 1):
    level = logging.INFO if verbosity == 1 else (logging.DEBUG if verbosity >= 2 else logging.WARNING)
    logger = logging.getLogger()
    logger.setLevel(level)

    fmt = logging.Formatter("[%(levelname)s] %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(level)

    fh = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(level)

    # Avoid duplicate handlers on reruns
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        logger.addHandler(ch)
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        logger.addHandler(fh)

    logging.info("Logging to console and %s", LOG_PATH.resolve())


# -------------------------
# yt-dlp helpers
# -------------------------
def ffmpeg_ok() -> bool:
    ok = shutil.which("ffmpeg") is not None
    if not ok:
        logging.warning("FFmpeg not found on PATH. Install FFmpeg for audio extraction/conversion.")
    return ok


def build_ydl_opts(
    outdir: Path,
    prefer_codec: str = "wav",
    prefer_quality: str = "0",
    cookies_from_browser: Optional[str] = None,
    cookiefile: Optional[str] = None,
    extract_flat: bool = False,
) -> Dict:
    """
    Build yt-dlp options for listing or downloading.
    prefer_codec: 'wav' (lossless, default) or 'mp3'
    prefer_quality: for mp3, e.g., '192'; for wav, '0' (ignored by ffmpeg)
    """
    ydl_opts: Dict = {
        "quiet": True,
        "outtmpl": str(outdir / "%(title)s.%(ext)s"),
    }

    if extract_flat:
        # For listing a playlist (no downloads)
        ydl_opts["extract_flat"] = True
        ydl_opts["skip_download"] = True
        # Let yt-dlp fetch the playlist index fully
    else:
        # For actual downloads
        ydl_opts["noplaylist"] = True  # we pass single video URLs to the downloader
        ydl_opts["format"] = "bestaudio/best"
        ydl_opts["postprocessors"] = [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": prefer_codec,
            "preferredquality": prefer_quality,
        }]

    # Cookies
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
) -> yt_dlp.YoutubeDL:
    """Create ONE YoutubeDL instance (so cookies are read once)."""
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


def list_playlist(
    playlist_url: str,
    ydl_list: yt_dlp.YoutubeDL
) -> List[Dict]:
    """Return flat list of entries (title, url, id, duration if available) without downloading."""
    logging.info("Fetching playlist: %s", playlist_url)
    try:
        info = ydl_list.extract_info(playlist_url, download=False)
        entries = info.get("entries", []) or []
        logging.info("Found %d videos in playlist", len(entries))

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
        return [x for x in flat if x.get("url")]
    except Exception as e:
        logging.error("Failed to fetch playlist: %s", e)
        traceback.print_exc()
        return []


def download_one(
    video_url: str,
    ydl_dl: yt_dlp.YoutubeDL,
    outdir: Path,
    codec: str
) -> Optional[Path]:
    """Download best audio and convert to target codec using a REUSED YoutubeDL instance."""
    logging.info("Downloading %s from: %s", codec.upper(), video_url)
    outdir.mkdir(parents=True, exist_ok=True)

    if not ffmpeg_ok():
        logging.warning("FFmpeg is required to convert to %s. Attempting anyway; may fail.", codec.upper())

    try:
        info = ydl_dl.extract_info(video_url, download=True)

        # Try to get final filepath from yt-dlp metadata first
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

    except yt_dlp.utils.DownloadError as e:
        logging.error("Download failed (DownloadError) for %s: %s", video_url, e)
        logging.error("If you saw a 'Sign in to confirm you’re not a bot' message, pass cookies via --browser or --cookiefile.")
        return None
    except Exception as e:
        logging.error("%s download failed for %s: %s", codec.upper(), video_url, e)
        traceback.print_exc()
        return None


# -------------------------
# Audio analysis
# -------------------------
def estimate_key_naive(y: np.ndarray, sr: int) -> str:
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    pitch_class = int(np.argmax(chroma.mean(axis=1)))
    pitch_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    return f"{pitch_names[pitch_class]} (approx)"


def analyze_audio(path: Path) -> Dict:
    logging.info("Analyzing audio: %s", path)
    try:
        y, sr = librosa.load(path, sr=None, mono=True)
        logging.debug("Audio loaded: %d samples @ %d Hz", len(y), sr)

        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        bpm = float(tempo)
        logging.info("Estimated BPM: %.2f (beats detected: %d)", bpm, len(beat_times))

        mix_in_idx = 32 if len(beat_times) > 32 else max(0, len(beat_times) // 4)
        mix_in_time = float(beat_times[mix_in_idx]) if len(beat_times) else 0.0

        if len(beat_times) >= 33:
            outro_start_idx = max(0, len(beat_times) - 32)
            mix_out_time = float(beat_times[outro_start_idx])
        else:
            mix_out_time = float(beat_times[0]) if len(beat_times) else 0.0

        key_est = estimate_key_naive(y, sr)
        logging.info("Approximate key: %s", key_est)

        return {
            "bpm": round(bpm, 2),
            "beats_count": int(len(beat_times)),
            "mix_in_time_s": round(mix_in_time, 2),
            "mix_out_time_s": round(mix_out_time, 2),
            "approx_key": key_est,
        }
    except Exception as e:
        logging.error("Analysis failed for %s: %s", path, e)
        traceback.print_exc()
        return {
            "bpm": None,
            "beats_count": None,
            "mix_in_time_s": None,
            "mix_out_time_s": None,
            "approx_key": None,
        }


# -------------------------
# Main pipeline
# -------------------------
def process_playlist(
    playlist_url: str,
    outdir: Path,
    csv_path: Path,
    download_audio: bool,
    codec: str,
    mp3_bitrate: str,
    cookies_from_browser: Optional[str],
    cookiefile: Optional[str],
) -> None:
    logging.info("Starting playlist processing...")

    # Create ONE YDL for listing (loads cookies once)
    ydl_list = create_ydl(
        listing=True,
        outdir=Path("."),  # not used for listing
        codec=codec,
        mp3_bitrate=mp3_bitrate,
        cookies_from_browser=cookies_from_browser,
        cookiefile=cookiefile,
    )
    items = list_playlist(playlist_url, ydl_list)
    if not items:
        logging.error("No playlist items found. Exiting.")
        return

    # Create ONE YDL for all downloads (loads cookies once)
    ydl_dl = None
    if download_audio:
        ydl_dl = create_ydl(
            listing=False,
            outdir=outdir,
            codec=codec,
            mp3_bitrate=mp3_bitrate,
            cookies_from_browser=cookies_from_browser,
            cookiefile=cookiefile,
        )

    rows = []
    for i, item in enumerate(items, 1):
        title = item["title"]
        url = item["url"]
        logging.info("[TRACK %d/%d] %s", i, len(items), title)

        audio_path = None
        if download_audio and ydl_dl is not None:
            audio_path = download_one(url, ydl_dl, outdir, codec)

        if audio_path and audio_path.exists():
            analysis = analyze_audio(audio_path)
        else:
            if download_audio:
                logging.warning("Skipping analysis (file missing or download failed): %s", title)
            analysis = {
                "bpm": None,
                "beats_count": None,
                "mix_in_time_s": None,
                "mix_out_time_s": None,
                "approx_key": None,
            }

        rows.append({
            "title": title,
            "youtube_url": url,
            "downloaded_path": str(audio_path) if audio_path else "",
            **analysis
        })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    logging.info("Saved %d rows to %s", len(df), csv_path.resolve())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YouTube playlist → audio downloads (WAV/MP3) + DJ metadata CSV")
    p.add_argument("playlist_url", help="YouTube playlist URL")
    p.add_argument("--outdir", default="downloads", help="Directory to store audio files (default: downloads)")
    p.add_argument("--csv", default="playlist_tracks.csv", help="CSV output path (default: playlist_tracks.csv)")
    p.add_argument("--download", action="store_true", help="Actually download audio. Omit to only list & CSV.")
    p.add_argument("--codec", choices=["wav", "mp3"], default="wav", help="Output audio codec (default: wav)")
    p.add_argument("--mp3-bitrate", default="192", help="MP3 bitrate kbps (only if --codec mp3). Default: 192")
    p.add_argument("--browser", choices=["chrome", "firefox", "edge", "brave", "chromium"],
                   help="Use cookies from this browser (must be logged in).")
    p.add_argument("--cookiefile", help="Path to cookies.txt exported from your browser.")
    p.add_argument("-v", "--verbose", action="count", default=1, help="Increase verbosity: -v (info, default), -vv (debug)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.verbose)

    outdir = Path(args.outdir)
    csv_path = Path(args.csv)

    process_playlist(
        playlist_url=args.playlist_url,
        outdir=outdir,
        csv_path=csv_path,
        download_audio=args.download,
        codec=args.codec,
        mp3_bitrate=args.mp3_bitrate,
        cookies_from_browser=args.browser,
        cookiefile=args.cookiefile,
    )
