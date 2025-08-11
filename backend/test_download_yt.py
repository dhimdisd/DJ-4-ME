#!/usr/bin/env python3
# yt_playlist_to_dj_metadata.py

import argparse
import json
import logging
import shutil
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yt_dlp
import librosa
import numpy as np
import pandas as pd
from scipy.signal import find_peaks  # modern replacement for deprecated peak_pick

# Optional loudness lib
try:
    import pyloudnorm as pyln
    HAS_LOUDNORM = True
except Exception:
    HAS_LOUDNORM = False

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

    # Ensure we actually see logs each run
    logger.handlers.clear()
    logger.addHandler(ch)
    logger.addHandler(fh)

    logging.info("Logging to console and %s", LOG_PATH.resolve())

# -------------------------
# NumPy-safe helpers
# -------------------------
def to_float(x) -> float:
    """Safely convert numpy scalars/arrays (0-d/1-d) to a Python float."""
    try:
        arr = np.asarray(x)
        if arr.ndim == 0:
            return float(arr.item())
        if arr.ndim >= 1:
            return float(arr.flat[0])
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return 0.0

def r2(x) -> float:
    """Round to 2 decimals with NumPy-safe conversion."""
    return round(to_float(x), 2)

def list_r2(seq, limit=None):
    """Round a sequence to 2 decimals (NumPy-safe), with optional length cap."""
    if seq is None:
        return []
    out = [r2(v) for v in seq]
    return out[:limit] if limit is not None else out

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
    prefer_quality: for mp3 (e.g., '192'); for wav, '0' is fine
    """
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
            "preferredcodec": prefer_codec,
            "preferredquality": prefer_quality,
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
) -> yt_dlp.YoutubeDL:
    """Create ONE YoutubeDL instance (so cookies/config are read once)."""
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
) -> Tuple[List[Dict], str]:
    """Return (entries, playlist_title). Each entry -> {title, url, id, duration}."""
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
    ydl_dl: yt_dlp.YoutubeDL,
    outdir: Path,
    codec: str
) -> Optional[Path]:
    """Download best audio and convert to target codec using the reused YDL instance."""
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
# DJ analysis helpers
# -------------------------
_PITCH_NAMES   = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
_CAMELOT_MAJOR = ["8B","3B","10B","5B","12B","7B","2B","9B","4B","11B","6B","1B"]
_CAMELOT_MINOR = ["5A","12A","7A","2A","9A","4A","11A","6A","1A","8A","3A","10A"]

def estimate_key_ks(y, sr):
    """Simple Krumhansl-Schmuckler key estimate → (root, mode, camelot, confidence)."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = librosa.util.normalize(chroma.mean(axis=1), norm=1)

    maj_prof = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88], float)
    min_prof = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17], float)
    maj_prof /= maj_prof.sum(); min_prof /= min_prof.sum()

    scores = []
    for root in range(12):
        scores.append(("maj", root, float(np.dot(np.roll(maj_prof, root), chroma_mean))))
        scores.append(("min", root, float(np.dot(np.roll(min_prof, root), chroma_mean))))

    mode, root, _ = max(scores, key=lambda x: x[2])
    sorted_scores = sorted(scores, key=lambda x: x[2], reverse=True)
    conf = sorted_scores[0][2] - sorted_scores[1][2]

    key_root = _PITCH_NAMES[root]
    camelot  = _CAMELOT_MAJOR[root] if mode == "maj" else _CAMELOT_MINOR[root]
    return key_root, ("major" if mode=="maj" else "minor"), camelot, r2(conf)

def camelot_distance(c1: str, c2: str) -> int:
    """Return harmonic distance between Camelot keys (0 = same, 1 = adjacent wheel, etc.)."""
    if not c1 or not c2:
        return 99
    def parse(c):
        return int(c[:-1]), c[-1].upper()
    n1, m1 = parse(c1); n2, m2 = parse(c2)
    if m1 == m2:
        d = min((n1-n2) % 12, (n2-n1) % 12)
        return int(d)
    if n1 == n2:
        return 1
    return 2

def energy_features(y, sr, include_curve: bool = False):
    """RMS dB summary and optional downsampled curve (times, values)."""
    hop = 1024
    frame_len = 2048
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop)[0]
    rms_db = 20*np.log10(np.maximum(rms, 1e-8))
    rms_db_mean = to_float(rms_db.mean())
    rms_db_std  = to_float(rms_db.std())

    if not include_curve:
        return rms_db_mean, rms_db_std, None, None

    target_rate = 10.0
    frames_per_sec = sr / hop
    step = max(1, int(frames_per_sec / target_rate))
    rms_db_ds = rms_db[::step]
    times_ds  = librosa.frames_to_time(np.arange(len(rms_db))[::step], sr=sr, hop_length=hop)
    return rms_db_mean, rms_db_std, times_ds.tolist(), rms_db_ds.tolist()

def estimate_lufs(y, sr) -> Optional[float]:
    """Integrated LUFS via pyloudnorm if available."""
    if not HAS_LOUDNORM:
        return None
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(y.astype(np.float64))
    return r2(loudness)

def structure_heuristics(y, sr, beat_times):
    """Heuristic intro/outro, breakdowns, drops from energy + onsets (NumPy-safe)."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_times = librosa.times_like(onset_env, sr=sr)

    # Peaks via SciPy
    if onset_env.size:
        hop_length = 512
        distance_frames = max(1, int(0.1 * sr / hop_length))  # ~0.1s spacing
        height = 0.8 * to_float(onset_env.max())
        peaks, _ = find_peaks(onset_env, height=height, distance=distance_frames)
        onset_peaks_s = onset_times[peaks].tolist()
    else:
        onset_peaks_s = []

    rms_db_mean, rms_db_std, times_ds, rms_db_ds = energy_features(y, sr, include_curve=True)

    # Intro end: sustained energy above (mean - 0.5*std) for >= 8s
    thresh = rms_db_mean - 0.5*rms_db_std
    intro_end_s = 0.0
    if times_ds:
        acc = 0.0
        last_t = to_float(times_ds[0])
        for t, e in zip(times_ds, rms_db_ds):
            t = to_float(t); e = to_float(e)
            dt = t - last_t
            last_t = t
            if e >= thresh:
                acc += dt
                if acc >= 8.0:
                    intro_end_s = t - 4.0
                    break
            else:
                acc = 0.0

    # Outro start: last 32 beats fallback
    if len(beat_times) >= 33:
        outro_start_s = to_float(beat_times[-32])
    else:
        outro_start_s = to_float(beat_times[0]) if len(beat_times) else 0.0

    # Breakdowns: local minima over ~6s windows
    breakdowns = []
    if times_ds and len(times_ds) > 2:
        dt = max(1e-6, to_float(times_ds[1]) - to_float(times_ds[0]))
        win = max(1, int(6.0 / dt))
        for i in range(win, len(rms_db_ds)-win):
            center = to_float(rms_db_ds[i])
            window_vals = [to_float(v) for v in rms_db_ds[i-win:i+win+1]]
            if center == min(window_vals) and center < (rms_db_mean - 0.3*rms_db_std):
                breakdowns.append(to_float(times_ds[i]))

    # Drops: strong onsets shortly after a breakdown
    drops = []
    if onset_peaks_s and breakdowns:
        b_iter = iter(breakdowns)
        b = next(b_iter, None)
        for p in onset_peaks_s:
            p = to_float(p)
            while b is not None and p - b > 20.0:
                b = next(b_iter, None)
            if b is None:
                break
            if 0 < p - b < 12.0:
                drops.append(p)

    return {
        "intro_end_s": r2(intro_end_s),
        "outro_start_s": r2(outro_start_s),
        "onset_peaks_s": list_r2(onset_peaks_s, limit=20),
        "breakdowns_s":  list_r2(breakdowns, limit=10),
        "drops_s":       list_r2(drops, limit=10),
        "rms_db_mean": r2(rms_db_mean),
        "rms_db_std":  r2(rms_db_std),
        "energy_curve": {
            "times_s": list_r2(times_ds or [], limit=600),
            "rms_db":  list_r2(rms_db_ds or [], limit=600),
        }
    }

def phrase_markers(beat_times, beats_per_bar=4, bars_per_phrase=8):
    """Return 32-beat (8-bar) phrase boundaries in seconds."""
    phrase = beats_per_bar * bars_per_phrase  # 32
    markers = []
    for i in range(0, len(beat_times), phrase):
        markers.append(to_float(beat_times[i]))
    return list_r2(markers)

def bar_starts(beat_times, beats_per_bar=4):
    """Return bar start times (every 4 beats) in seconds."""
    markers = []
    for i in range(0, len(beat_times), beats_per_bar):
        markers.append(to_float(beat_times[i]))
    return list_r2(markers, limit=200)

# -------------------------
# Analyze one track
# -------------------------
def analyze_audio(path: Path) -> Dict:
    logging.info("Analyzing audio: %s", path)
    try:
        y, sr = librosa.load(path, sr=None, mono=True)
        logging.debug("Audio loaded: %d samples @ %d Hz", len(y), sr)

        # Tempo & beats
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        bpm = to_float(tempo)  # NumPy-safe
        bars = int(len(beat_times) // 4)

        # Key (K-S) + Camelot
        key_root, mode, camelot, key_conf = estimate_key_ks(y, sr)

        # Structure heuristics
        struct = structure_heuristics(y, sr, beat_times)
        phrases = phrase_markers(beat_times)
        bars_s  = bar_starts(beat_times)

        # Mix cues (phrase after intro; safe fallbacks)
        if phrases:
            mix_in_time = next((t for t in phrases if t >= struct["intro_end_s"]),
                               phrases[min(1, len(phrases)-1)])
        else:
            mix_in_time = to_float(beat_times[32]) if len(beat_times) > 32 else 0.0
        mix_out_time = struct["outro_start_s"]

        # Loudness (optional)
        lufs = estimate_lufs(y, sr)

        result = {
            "bpm": r2(bpm),
            "beats_count": int(len(beat_times)),
            "bars_count": bars,
            "key_root": key_root,
            "mode": mode,
            "camelot": camelot,
            "key_confidence": key_conf,
            "mix_in_time_s": r2(mix_in_time),
            "mix_out_time_s": r2(mix_out_time),
            "intro_end_s": struct["intro_end_s"],
            "outro_start_s": struct["outro_start_s"],
            "phrase_boundaries_s": phrases[:50],
            "bar_starts_s": bars_s,
            "onset_peaks_s": struct["onset_peaks_s"],
            "breakdowns_s": struct["breakdowns_s"],
            "drops_s": struct["drops_s"],
            "rms_db_mean": struct["rms_db_mean"],
            "rms_db_std": struct["rms_db_std"],
            "lufs_integrated": lufs,
            "energy_curve_sample": struct["energy_curve"],
        }
        logging.info("BPM %.2f | Key %s %s (%s) | Phrases %d | Bars %d",
                     result["bpm"], result["key_root"], result["mode"], result["camelot"],
                     len(result["phrase_boundaries_s"]), result["bars_count"])
        return result

    except Exception as e:
        logging.error("Analysis failed for %s: %s", path, e)
        traceback.print_exc()
        return {
            "bpm": None, "beats_count": None, "bars_count": None,
            "key_root": None, "mode": None, "camelot": None, "key_confidence": None,
            "mix_in_time_s": None, "mix_out_time_s": None,
            "intro_end_s": None, "outro_start_s": None,
            "phrase_boundaries_s": [], "bar_starts_s": [],
            "onset_peaks_s": [], "breakdowns_s": [], "drops_s": [],
            "rms_db_mean": None, "rms_db_std": None, "lufs_integrated": None,
            "energy_curve_sample": {"times_s": [], "rms_db": []},
        }

def write_track_json(audio_path: Path, analysis: Dict):
    """Save detailed analysis next to the audio file as <stem>.djmeta.json"""
    if not audio_path:
        return
    meta_path = audio_path.with_suffix("").with_suffix(".djmeta.json")
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        logging.info("Wrote metadata JSON: %s", meta_path)
    except Exception as e:
        logging.warning("Failed to write track JSON for %s: %s", audio_path, e)

# -------------------------
# Main pipeline
# -------------------------
def process_playlist(
    playlist_url: str,
    outdir: Path,
    csv_path: Optional[Path],
    download_audio: bool,
    codec: str,
    mp3_bitrate: str,
    cookies_from_browser: Optional[str],
    cookiefile: Optional[str],
) -> None:
    logging.info("Starting playlist processing...")

    # YDL for listing
    ydl_list = create_ydl(
        listing=True,
        outdir=Path("."),  # not used for listing
        codec=codec,
        mp3_bitrate=mp3_bitrate,
        cookies_from_browser=cookies_from_browser,
        cookiefile=cookiefile,
    )
    items, playlist_title = list_playlist(playlist_url, ydl_list)
    if not items:
        logging.error("No playlist items found. Exiting.")
        return

    # Default CSV: in outdir, named after playlist
    if csv_path is None:
        safe_title = "".join(ch for ch in playlist_title if ch.isalnum() or ch in " _-").strip()
        csv_path = outdir / f"{safe_title or 'playlist'}.csv"

    # YDL for downloads (once)
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
            write_track_json(audio_path, analysis)
        else:
            if download_audio:
                logging.warning("Skipping analysis (file missing or download failed): %s", title)
            analysis = {
                "bpm": None, "beats_count": None, "bars_count": None,
                "key_root": None, "mode": None, "camelot": None, "key_confidence": None,
                "mix_in_time_s": None, "mix_out_time_s": None,
                "intro_end_s": None, "outro_start_s": None,
                "rms_db_mean": None, "rms_db_std": None, "lufs_integrated": None,
            }

        rows.append({
            "title": title,
            "youtube_url": url,
            "downloaded_path": str(audio_path) if audio_path else "",
            **analysis
        })

    df = pd.DataFrame(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logging.info("Saved %d rows to %s", len(df), csv_path.resolve())
    logging.info("Tip: choose next tracks where BPM within ±2–4 and camelot_distance(current, candidate) <= 1.")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YouTube playlist → audio downloads (WAV/MP3) + rich DJ metadata (CSV + JSON)")
    p.add_argument("playlist_url", help="YouTube playlist URL")
    p.add_argument("--outdir", default="downloads", help="Directory to store audio files (default: downloads)")
    p.add_argument("--csv", default=None, help="CSV output path. Default: <outdir>/<playlist-title>.csv")
    p.add_argument("--download", action="store_true", help="Actually download audio. Omit to only list & CSV of URLs.")
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
    process_playlist(
        playlist_url=args.playlist_url,
        outdir=outdir,
        csv_path=(Path(args.csv) if args.csv else None),
        download_audio=args.download,
        codec=args.codec,
        mp3_bitrate=args.mp3_bitrate,
        cookies_from_browser=args.browser,
        cookiefile=args.cookiefile,
    )
