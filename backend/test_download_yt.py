#!/usr/bin/env python3
# yt_playlist_to_dj_metadata.py
# - Lists a YouTube playlist (yt-dlp), optionally downloads audio (wav/mp3)
# - Analyzes local audio: BPM, beats, phrases, bars, K-S key, Camelot, loudness
# - Adds: downbeat (bar-1) estimation + vocal-likelihood curve (avoid vocals)
# - Exports: CSV (summary) + per-track JSON (.djmeta.json) with rich metadata

import argparse
import json
import logging
import math
import shutil
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import librosa
from scipy.signal import find_peaks

# Optional deps
try:
    import yt_dlp  # only needed if you use --playlist/--download
    HAS_YT_DLP = True
except Exception:
    HAS_YT_DLP = False

try:
    import pyloudnorm as pyln
    HAS_LOUDNORM = True
except Exception:
    HAS_LOUDNORM = False

try:
    import scipy.ndimage as ndi
    HAS_SNDIMAGE = True
except Exception:
    HAS_SNDIMAGE = False


# -------------------------
# Logging
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

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        logger.addHandler(ch)
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        logger.addHandler(fh)

    logging.info("Logging to console and %s", LOG_PATH.resolve())


# -------------------------
# Small utils
# -------------------------
def to_float(x) -> float:
    try:
        arr = np.asarray(x)
        if arr.ndim == 0:
            return float(arr.item())
        return float(arr.flat[0])
    except Exception:
        try:
            return float(x)
        except Exception:
            return 0.0

def r2(x) -> float:
    return round(to_float(x), 2)

def ffmpeg_ok() -> bool:
    ok = shutil.which("ffmpeg") is not None
    if not ok:
        logging.warning("FFmpeg not found on PATH. Install it for audio extraction/conversion.")
    return ok


# -------------------------
# yt-dlp (optional)
# -------------------------
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
):
    if not HAS_YT_DLP:
        raise RuntimeError("yt_dlp is not installed. pip install yt-dlp")
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
    ydl_list
) -> Tuple[List[Dict], str]:
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
    ydl_dl,
    outdir: Path,
    codec: str
) -> Optional[Path]:
    logging.info("Downloading %s from: %s", codec.upper(), video_url)
    outdir.mkdir(parents=True, exist_ok=True)
    if not ffmpeg_ok():
        logging.warning("FFmpeg is required to convert to %s. Attempting anyway; may fail.", codec.upper())
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


# -------------------------
# Music theory + analysis helpers
# -------------------------
_PITCH_NAMES   = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
_CAMELOT_MAJOR = ["8B","3B","10B","5B","12B","7B","2B","9B","4B","11B","6B","1B"]
_CAMELOT_MINOR = ["5A","12A","7A","2A","9A","4A","11A","6A","1A","8A","3A","10A"]

def estimate_key_ks(y, sr):
    """Krumhansl–Schmuckler key estimate: returns (root, mode, camelot, confidence)."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = librosa.util.normalize(chroma.mean(axis=1), norm=1)

    maj_prof = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88], float)
    min_prof = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17], float)
    maj_prof /= maj_prof.sum(); min_prof /= min_prof.sum()

    scores = []
    for root in range(12):
        scores.append(("maj", root, float(np.dot(np.roll(maj_prof, root), chroma_mean))))
        scores.append(("min", root, float(np.dot(np.roll(min_prof, root), chroma_mean))))

    mode, root, top = max(scores, key=lambda x: x[2])
    scores_sorted = sorted(scores, key=lambda x: x[2], reverse=True)
    conf = float(scores_sorted[0][2] - scores_sorted[1][2])

    key_root = _PITCH_NAMES[root]
    camelot  = _CAMELOT_MAJOR[root] if mode == "maj" else _CAMELOT_MINOR[root]
    return key_root, ("major" if mode=="maj" else "minor"), camelot, round(conf, 4)


def energy_features(y, sr, include_curve: bool = False):
    hop = 1024
    frame_len = 2048
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop)[0]
    rms_db = 20*np.log10(np.maximum(rms, 1e-8))
    rms_db_mean = float(rms_db.mean())
    rms_db_std  = float(rms_db.std())

    if not include_curve:
        return rms_db_mean, rms_db_std, None, None

    target_rate = 10.0
    frames_per_sec = sr / hop
    step = max(1, int(frames_per_sec / target_rate))
    rms_db_ds = rms_db[::step]
    times_ds  = librosa.frames_to_time(np.arange(len(rms_db))[::step], sr=sr, hop_length=hop)
    return rms_db_mean, rms_db_std, times_ds.tolist(), rms_db_ds.tolist()


def estimate_lufs(y, sr) -> Optional[float]:
    if not HAS_LOUDNORM:
        return None
    meter = pyln.Meter(sr)
    loudness = float(meter.integrated_loudness(y.astype(np.float64)))
    return round(loudness, 2)


# --- Downbeat & bars ---
def estimate_downbeat_offset(y, sr, beat_frames) -> int:
    """
    Heuristic downbeat (bar-1) offset in 4/4: pick the shift 0..3 with max onset strength on bar starts.
    """
    if beat_frames is None or len(beat_frames) < 8:
        return 0
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    scores = []
    for shift in range(4):
        idxs = beat_frames[shift::4]
        vals = []
        for fr in idxs:
            fr = int(fr)
            fr = max(0, min(fr, len(onset_env)-1))
            vals.append(onset_env[fr])
        scores.append(float(np.mean(vals)) if vals else 0.0)
    return int(np.argmax(scores))


def bars_from_beats_with_offset(beat_times: np.ndarray, offset: int = 0) -> List[float]:
    if beat_times is None or len(beat_times) == 0:
        return []
    out = []
    for i in range(offset, len(beat_times), 4):
        out.append(float(beat_times[i]))
    return [round(t, 2) for t in out]


# --- Vocal-likelihood ---
def vocal_likelihood_curve(y, sr, include_curve=True):
    """
    Crude vocal activity 0..1 from harmonic energy in 300–3400 Hz.
    (Requires SciPy for smoothing; falls back to unsmoothed if absent.)
    """
    y_h, _ = librosa.effects.hpss(y)
    S = librosa.feature.melspectrogram(y=y_h, sr=sr, n_fft=2048, hop_length=512, n_mels=64)
    freqs = librosa.mel_frequencies(n_mels=S.shape[0], fmin=0, fmax=sr//2)

    band = (freqs >= 300) & (freqs <= 3400)
    band_energy = S[band].sum(axis=0)
    total_energy = np.maximum(S.sum(axis=0), 1e-8)
    raw = band_energy / total_energy

    if HAS_SNDIMAGE:
        smoothed = ndi.gaussian_filter1d(raw.astype(np.float32), sigma=2)
    else:
        smoothed = raw  # no smoothing fallback

    vl = smoothed - smoothed.min()
    denom = float(vl.max()) or 1.0
    vl = vl / denom

    times = librosa.frames_to_time(np.arange(len(vl)), sr=sr, hop_length=512)
    if not include_curve:
        return float(np.mean(vl)), None, None

    target_rate = 10.0
    frames_per_sec = sr / 512.0
    step = max(1, int(frames_per_sec / target_rate))
    vl_ds = vl[::step]
    times_ds = times[::step]
    return float(np.mean(vl)), times_ds.tolist(), vl_ds.tolist()


def vocal_score_around(times: np.ndarray, values: np.ndarray, t: float, win: float = 4.0) -> float:
    if times is None or values is None or len(times) == 0:
        return 0.0
    lo, hi = t - win/2, t + win/2
    mask = (times >= lo) & (times <= hi)
    if not np.any(mask):
        idx = int(np.argmin(np.abs(times - t)))
        return float(values[idx])
    return float(np.mean(values[mask]))


# --- Structure heuristics (intro/outro/drops) ---
def structure_heuristics(y, sr, beat_times):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_times = librosa.times_like(onset_env, sr=sr)

    if onset_env.size:
        hop_length = 512
        distance_frames = max(1, int(0.1 * sr / hop_length))
        height = 0.8 * to_float(onset_env.max())
        peaks, _ = find_peaks(onset_env, height=height, distance=distance_frames)
        onset_peaks_s = onset_times[peaks].tolist()
    else:
        onset_peaks_s = []

    rms_db_mean, rms_db_std, times_ds, rms_db_ds = energy_features(y, sr, include_curve=True)

    # Intro end: energy above (mean - 0.5*std) for >= 8s
    thresh = rms_db_mean - 0.5*rms_db_std
    intro_end_s = 0.0
    if times_ds is not None and len(times_ds) > 0:
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
        beat_dur = (to_float(beat_times[-1]) - to_float(beat_times[0])) / max(1, len(beat_times)-1)
        out_idx = max(0, len(beat_times) - 32)
        outro_start_s = float(beat_times[out_idx])
    else:
        outro_start_s = float(beat_times[0]) if len(beat_times) else 0.0

    # Breakdowns: local minima over ~6s windows (on energy curve)
    breakdowns = []
    if times_ds is not None and len(times_ds) > 2:
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
        "onset_peaks_s": [r2(x) for x in onset_peaks_s[:20]],
        "breakdowns_s":  [r2(x) for x in breakdowns[:10]],
        "drops_s":       [r2(x) for x in drops[:10]],
        "rms_db_mean": r2(rms_db_mean),
        "rms_db_std":  r2(rms_db_std),
        "energy_curve": {
            "times_s": [r2(x) for x in (times_ds or [])[:600]],
            "rms_db":  [r2(x) for x in (rms_db_ds or [])[:600]],
        }
    }


def phrase_markers(beat_times, beats_per_bar=4, bars_per_phrase=8):
    phrase = beats_per_bar * bars_per_phrase  # 32
    markers = []
    for i in range(0, len(beat_times), phrase):
        markers.append(float(beat_times[i]))
    return [round(t,2) for t in markers]

def bar_starts_simple(beat_times, beats_per_bar=4):
    markers = []
    for i in range(0, len(beat_times), beats_per_bar):
        markers.append(float(beat_times[i]))
    return [round(t,2) for t in markers]


# -------------------------
# Analyze one audio file
# -------------------------
def analyze_audio(path: Path) -> Dict:
    logging.info("Analyzing audio: %s", path)
    try:
        y, sr = librosa.load(path, sr=None, mono=True)
        logging.debug("Audio loaded: %d samples @ %d Hz", len(y), sr)

        # Tempo & beats
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        bpm = float(tempo)
        bars = int(len(beat_times) // 4)

        # Key (K-S)
        key_root, mode, camelot, key_conf = estimate_key_ks(y, sr)

        # Structure
        struct = structure_heuristics(y, sr, beat_times)
        phrases = phrase_markers(beat_times)
        bars_simple = bar_starts_simple(beat_times)

        # Downbeat alignment
        downbeat_offset = estimate_downbeat_offset(y, sr, beat_frames)
        bars_downbeat = bars_from_beats_with_offset(beat_times, downbeat_offset)
        bar1_time_s   = bars_downbeat[0] if bars_downbeat else (float(beat_times[0]) if len(beat_times) else 0.0)

        # Vocal likelihood
        vocal_mean, vocal_t, vocal_v = vocal_likelihood_curve(y, sr, include_curve=True)
        LOW_VOCAL_THR = 0.4
        low_vocal_bar_starts = []
        for bt in bars_downbeat:
            vs = vocal_score_around(np.asarray(vocal_t), np.asarray(vocal_v), float(bt), win=4.0)
            if vs <= LOW_VOCAL_THR:
                low_vocal_bar_starts.append(round(float(bt), 2))

        # Mix cues (basic)
        mix_in_time = phrases[1] if len(phrases) > 1 else (beat_times[32] if len(beat_times) > 32 else 0.0)
        mix_out_time = struct["outro_start_s"]

        # Loudness (optional)
        lufs = estimate_lufs(y, sr)

        result = {
            "title": path.stem,
            "bpm": round(bpm, 2),
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
            "bar_starts_s": bars_simple[:200],
            "downbeat_offset": int(downbeat_offset),
            "bar1_time_s": r2(bar1_time_s),
            "downbeat_bar_starts_s": bars_downbeat[:200],
            "low_vocal_bar_starts_s": low_vocal_bar_starts[:200],
            "onset_peaks_s": struct["onset_peaks_s"],
            "breakdowns_s": struct["breakdowns_s"],
            "drops_s": struct["drops_s"],
            "rms_db_mean": struct["rms_db_mean"],
            "rms_db_std": struct["rms_db_std"],
            "lufs_integrated": lufs,
            "energy_curve_sample": struct["energy_curve"],
            "vocal_mean": round(float(vocal_mean), 3),
            "vocal_curve_sample": {
                "times_s": [r2(t) for t in (vocal_t or [])][:600],
                "values":  [round(float(v),3) for v in (vocal_v or [])][:600],
            }
        }
        logging.info("BPM %.2f | Key %s %s (%s) | Bars %d | Phrases %d | Downbeat offset %d",
                     result["bpm"], result["key_root"], result["mode"], result["camelot"],
                     result["bars_count"], len(result["phrase_boundaries_s"]), result["downbeat_offset"])
        return result

    except Exception as e:
        logging.error("Analysis failed for %s: %s", path, e)
        traceback.print_exc()
        return {
            "title": path.stem,
            "bpm": None, "beats_count": None, "bars_count": None,
            "key_root": None, "mode": None, "camelot": None, "key_confidence": None,
            "mix_in_time_s": None, "mix_out_time_s": None,
            "intro_end_s": None, "outro_start_s": None,
            "phrase_boundaries_s": [], "bar_starts_s": [], "downbeat_bar_starts_s": [],
            "downbeat_offset": None, "bar1_time_s": None, "low_vocal_bar_starts_s": [],
            "onset_peaks_s": [], "breakdowns_s": [], "drops_s": [],
            "rms_db_mean": None, "rms_db_std": None, "lufs_integrated": None,
            "energy_curve_sample": {"times_s": [], "rms_db": []},
            "vocal_mean": None, "vocal_curve_sample": {"times_s": [], "values": []},
        }


def write_track_json(audio_path: Path, analysis: Dict):
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
    if not HAS_YT_DLP:
        logging.error("yt-dlp not installed. Install or use --localdir to analyze local audio.")
        return

    logging.info("Starting playlist processing...")

    # Listing YDL
    ydl_list = create_ydl(
        listing=True,
        outdir=Path("."), codec=codec, mp3_bitrate=mp3_bitrate,
        cookies_from_browser=cookies_from_browser, cookiefile=cookiefile,
    )
    items, playlist_title = list_playlist(playlist_url, ydl_list)
    if not items:
        logging.error("No playlist items found. Exiting.")
        return

    # Default CSV: in outdir, named after playlist
    if csv_path is None:
        safe_title = "".join(ch for ch in playlist_title if ch.isalnum() or ch in " _-").strip()
        csv_path = outdir / f"{safe_title or 'playlist'}.csv"

    # Create one YDL for downloads
    ydl_dl = None
    if download_audio:
        ydl_dl = create_ydl(
            listing=False,
            outdir=outdir, codec=codec, mp3_bitrate=mp3_bitrate,
            cookies_from_browser=cookies_from_browser, cookiefile=cookiefile,
        )

    rows = []
    for i, item in enumerate(items, 1):
        title = item["title"]
        url = item["url"]
        logging.info("[TRACK %d/%d] %s", i, len(items), title)

        audio_path = None
        if download_audio and ydl_dl is not None:
            audio_path = download_one(url, ydl_dl, outdir, codec)

        # Analyze only if file exists
        if audio_path and audio_path.exists():
            analysis = analyze_audio(audio_path)
            write_track_json(audio_path, analysis)
        else:
            logging.info("Skipping analysis (file missing or --download not set): %s", title)
            analysis = {"title": title, "bpm": None, "key_root": None, "mode": None, "camelot": None}

        rows.append({
            "title": title,
            "youtube_url": url,
            "downloaded_path": str(audio_path) if audio_path else "",
            "bpm": analysis.get("bpm"),
            "camelot": analysis.get("camelot"),
            "key_root": analysis.get("key_root"),
            "mode": analysis.get("mode"),
            "intro_end_s": analysis.get("intro_end_s"),
            "outro_start_s": analysis.get("outro_start_s"),
        })

    df = pd.DataFrame(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logging.info("Saved %d rows to %s", len(df), csv_path.resolve())
    logging.info("Tip: export cues with export_hotcues_dual.py based on .djmeta.json files.")


def process_local_directory(
    localdir: Path,
    csv_path: Optional[Path],
    exts=(".wav", ".mp3", ".flac", ".aiff", ".aif"),
) -> None:
    logging.info("Scanning local directory: %s", localdir)
    files = [p for p in sorted(localdir.glob("*")) if p.suffix.lower() in exts]
    if not files:
        logging.error("No audio files found in %s", localdir)
        return

    # Default CSV: in folder, name summary
    if csv_path is None:
        csv_path = localdir / "local_analysis.csv"

    rows = []
    for i, path in enumerate(files, 1):
        logging.info("[LOCAL %d/%d] %s", i, len(files), path.name)
        analysis = analyze_audio(path)
        write_track_json(path, analysis)

        rows.append({
            "title": analysis.get("title") or path.stem,
            "file_path": str(path),
            "bpm": analysis.get("bpm"),
            "camelot": analysis.get("camelot"),
            "key_root": analysis.get("key_root"),
            "mode": analysis.get("mode"),
            "intro_end_s": analysis.get("intro_end_s"),
            "outro_start_s": analysis.get("outro_start_s"),
        })

    df = pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logging.info("Saved %d rows to %s", len(df), csv_path.resolve())


# -------------------------
# CLI
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YouTube playlist / local audio → rich DJ metadata (CSV + .djmeta.json)")
    p.add_argument("--playlist", help="YouTube playlist URL (requires yt-dlp)")
    p.add_argument("--outdir", default="downloads", help="Directory for downloads (and default CSV for playlist).")
    p.add_argument("--csv", default=None, help="CSV output path. Default: for playlist → <outdir>/<playlist-title>.csv; for localdir → <localdir>/local_analysis.csv")
    p.add_argument("--download", action="store_true", help="When used with --playlist, download audio (WAV/MP3).")
    p.add_argument("--codec", choices=["wav", "mp3"], default="wav", help="Download codec (default: wav)")
    p.add_argument("--mp3-bitrate", default="192", help="MP3 bitrate kbps (only if --codec mp3). Default: 192")
    p.add_argument("--browser", choices=["chrome", "firefox", "edge", "brave", "chromium"], help="Use browser cookies for yt-dlp.")
    p.add_argument("--cookiefile", help="Path to cookies.txt for yt-dlp.")
    p.add_argument("--localdir", help="Analyze already-downloaded audio in this folder (no YouTube).")
    p.add_argument("-v", "--verbose", action="count", default=1, help="Increase verbosity: -v (info), -vv (debug)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.verbose)

    if args.playlist:
        outdir = Path(args.outdir)
        process_playlist(
            playlist_url=args.playlist,
            outdir=outdir,
            csv_path=(Path(args.csv) if args.csv else None),
            download_audio=args.download,
            codec=args.codec,
            mp3_bitrate=args.mp3_bitrate,
            cookies_from_browser=args.browser,
            cookiefile=args.cookiefile,
        )
    elif args.localdir:
        process_local_directory(
            localdir=Path(args.localdir),
            csv_path=(Path(args.csv) if args.csv else None),
        )
    else:
        logging.error("Nothing to do. Use --playlist ... or --localdir ...")
