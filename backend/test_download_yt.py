#!/usr/bin/env python3
# yt_playlist_to_dj_metadata.py
# Adds --localdir mode to analyze existing audio files without downloading.

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

    logger.handlers.clear()
    logger.addHandler(ch)
    logger.addHandler(fh)
    logging.info("Logging to console and %s", LOG_PATH.resolve())

# -------------------------
# NumPy-safe helpers
# -------------------------
def to_float(x) -> float:
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
    return round(to_float(x), 2)

def list_r2(seq, limit=None):
    if seq is None:
        return []
    out = [r2(v) for v in seq]
    return out[:limit] if limit is not None else out

# -------------------------
# Serato-ish helpers
# -------------------------
def beat_bar_from_time(t: float, bpm: float) -> tuple[int, int]:
    beat_dur = 60.0 / max(1e-6, bpm)
    bar_dur  = 4 * beat_dur
    beat_idx = int(round(t / beat_dur)) + 1
    bar_idx  = int(round(t / bar_dur)) + 1
    return beat_idx, bar_idx

def phrase_candidates(beat_times: np.ndarray, phrase_beats: int = 32) -> list[float]:
    if beat_times is None or len(beat_times) == 0:
        return []
    idxs = range(0, len(beat_times), phrase_beats)
    return [float(beat_times[i]) for i in idxs]

# -------------------------
# yt-dlp helpers (playlist mode)
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
    if not HAS_LOUDNORM:
        return None
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(y.astype(np.float64))
    return r2(loudness)

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

    if len(beat_times) >= 33:
        outro_start_s = to_float(beat_times[-32])
    else:
        outro_start_s = to_float(beat_times[0]) if len(beat_times) else 0.0

    breakdowns = []
    if times_ds and len(times_ds) > 2:
        dt = max(1e-6, to_float(times_ds[1]) - to_float(times_ds[0]))
        win = max(1, int(6.0 / dt))
        for i in range(win, len(rms_db_ds)-win):
            center = to_float(rms_db_ds[i])
            window_vals = [to_float(v) for v in rms_db_ds[i-win:i+win+1]]
            if center == min(window_vals) and center < (rms_db_mean - 0.3*rms_db_std):
                breakdowns.append(to_float(times_ds[i]))

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
    phrase = beats_per_bar * bars_per_phrase  # 32
    markers = []
    for i in range(0, len(beat_times), phrase):
        markers.append(to_float(beat_times[i]))
    return list_r2(markers)

def bar_starts(beat_times, beats_per_bar=4):
    markers = []
    for i in range(0, len(beat_times), beats_per_bar):
        markers.append(to_float(beat_times[i]))
    return list_r2(markers, limit=200)

# -------------------------
# NEW: Mix-in/out suggestion logic
# -------------------------
def nearest_energy_rise_times(y: np.ndarray, sr: int, times: List[float]) -> List[tuple[float, float]]:
    hop = 1024
    frame_len = 2048
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop)[0]
    rms_db = 20*np.log10(np.maximum(rms, 1e-8))
    rms_t  = librosa.frames_to_time(np.arange(len(rms_db)), sr=sr, hop_length=hop)

    def local_slope(center_s: float, win_s: float = 2.0) -> float:
        lo, hi = center_s - win_s, center_s + win_s
        mask = (rms_t >= lo) & (rms_t <= hi)
        m = rms_db[mask]
        t = rms_t[mask]
        if len(m) < 3:
            return 0.0
        tn = t - t.mean()
        denom = float((tn**2).sum()) or 1.0
        slope = float((tn * (m - m.mean())).sum()) / denom
        return slope

    return [(t, local_slope(t)) for t in times]

def suggest_mix_points(
    y: np.ndarray,
    sr: int,
    bpm: float,
    beat_times: np.ndarray,
    intro_end_s: float | None
) -> dict:
    phrase_starts = phrase_candidates(beat_times, phrase_beats=32)
    if not phrase_starts:
        return {"mix_in_candidates": [], "mix_out_candidates": [], "best_mix_in": None, "best_mix_out": None}

    # MIX-IN: phrase starts AFTER intro_end_s, prefer rising energy
    min_start = float(intro_end_s or 0.0)
    mix_in_times = [t for t in phrase_starts if t >= min_start]
    slopes = dict(nearest_energy_rise_times(y, sr, mix_in_times))
    mix_in_scored = []
    for t in mix_in_times:
        beat_idx, bar_idx = beat_bar_from_time(t, bpm)
        score = slopes.get(t, 0.0)
        mix_in_scored.append({
            "time_s": round(t, 2),
            "beat_idx": beat_idx,
            "bar_idx": bar_idx,
            "score": round(float(score), 4)
        })
    mix_in_scored.sort(key=lambda x: x["score"], reverse=True)
    best_mix_in = mix_in_scored[0] if mix_in_scored else None

    # MIX-OUT: last 32–64 beats (one or two phrases from the end)
    last_beat_t = float(beat_times[-1]) if len(beat_times) else 0.0
    beat_dur = 60.0 / max(1e-6, bpm)
    last32_t  = last_beat_t - 32 * beat_dur
    last64_t  = last_beat_t - 64 * beat_dur
    mix_out_times = [t for t in phrase_starts if t >= last64_t]
    slopes_out = dict(nearest_energy_rise_times(y, sr, mix_out_times))
    mix_out_scored = []
    for t in mix_out_times:
        beat_idx, bar_idx = beat_bar_from_time(t, bpm)
        slope = slopes_out.get(t, 0.0)
        score = -abs(float(slope))
        if t >= last32_t:
            score += 0.1
        mix_out_scored.append({
            "time_s": round(t, 2),
            "beat_idx": beat_idx,
            "bar_idx": bar_idx,
            "score": round(score, 4)
        })
    mix_out_scored.sort(key=lambda x: x["score"], reverse=True)
    best_mix_out = mix_out_scored[0] if mix_out_scored else None

    return {
        "mix_in_candidates": mix_in_scored[:8],
        "mix_out_candidates": mix_out_scored[:8],
        "best_mix_in": best_mix_in,
        "best_mix_out": best_mix_out,
    }

# -------------------------
# Analyze one track
# -------------------------
def analyze_audio(path: Path) -> Dict:
    logging.info("Analyzing audio: %s", path)
    try:
        y, sr = librosa.load(path, sr=None, mono=True)
        logging.debug("Audio loaded: %d samples @ %d Hz", len(y), sr)

        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        bpm = to_float(tempo)
        bars = int(len(beat_times) // 4)

        key_root, mode, camelot, key_conf = estimate_key_ks(y, sr)

        struct = structure_heuristics(y, sr, beat_times)
        phrases = phrase_markers(beat_times)
        bars_s  = bar_starts(beat_times)

        mix_suggestions = suggest_mix_points(
            y=y, sr=sr, bpm=bpm, beat_times=beat_times, intro_end_s=struct.get("intro_end_s")
        )

        if phrases:
            mix_in_time = next((t for t in phrases if t >= struct["intro_end_s"]),
                               phrases[min(1, len(phrases)-1)])
        else:
            mix_in_time = to_float(beat_times[32]) if len(beat_times) > 32 else 0.0
        mix_out_time = struct["outro_start_s"]

        lufs = estimate_lufs(y, sr)

        result = {
            "title": path.stem,
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
            "mix_in_candidates": mix_suggestions["mix_in_candidates"],
            "mix_out_candidates": mix_suggestions["mix_out_candidates"],
            "best_mix_in": mix_suggestions["best_mix_in"],
            "best_mix_out": mix_suggestions["best_mix_out"],
        }
        logging.info("BPM %.2f | Key %s %s (%s) | Phrases %d | Bars %d",
                     result["bpm"], result["key_root"], result["mode"], result["camelot"],
                     len(result["phrase_boundaries_s"]), result["bars_count"])
        if result["best_mix_in"]:
            b = result["best_mix_in"]
            logging.info('[MIX-IN] t=%ss | Beat %d | Bar %d', b["time_s"], b["beat_idx"], b["bar_idx"])
        if result["best_mix_out"]:
            b = result["best_mix_out"]
            logging.info('[MIX-OUT] t=%ss | Beat %d | Bar %d', b["time_s"], b["beat_idx"], b["bar_idx"])
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
            "phrase_boundaries_s": [], "bar_starts_s": [],
            "onset_peaks_s": [], "breakdowns_s": [], "drops_s": [],
            "rms_db_mean": None, "rms_db_std": None, "lufs_integrated": None,
            "energy_curve_sample": {"times_s": [], "rms_db": []},
            "mix_in_candidates": [], "mix_out_candidates": [],
            "best_mix_in": None, "best_mix_out": None,
        }

def write_track_json(audio_path: Path, analysis: Dict):
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
# Local analysis mode
# -------------------------
SUPPORTED_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".aiff", ".aif", ".alac", ".ogg"}

def iter_audio_files(folder: Path, recursive: bool = True) -> List[Path]:
    if not folder.exists():
        return []
    patterns = ["**/*"] if recursive else ["*"]
    out = []
    for pat in patterns:
        for p in folder.glob(pat):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                out.append(p)
    return sorted(out)

def process_local_folder(
    localdir: Path,
    csv_path: Optional[Path],
) -> None:
    logging.info("Analyzing local folder: %s", localdir.resolve())
    files = iter_audio_files(localdir, recursive=True)
    if not files:
        logging.error("No audio files found in %s (exts: %s)", localdir, ", ".join(sorted(SUPPORTED_EXTS)))
        return

    if csv_path is None:
        csv_path = localdir / "local_analysis.csv"

    rows = []
    for i, audio_path in enumerate(files, 1):
        logging.info("[LOCAL %d/%d] %s", i, len(files), audio_path.name)
        analysis = analyze_audio(audio_path)
        write_track_json(audio_path, analysis)
        rows.append({
            "title": analysis.get("title") or audio_path.stem,
            "youtube_url": "",
            "downloaded_path": str(audio_path),
            "bpm": analysis.get("bpm"),
            "camelot": analysis.get("camelot"),
            "intro_end_s": analysis.get("intro_end_s"),
            "mix_in_time_s": analysis.get("mix_in_time_s"),
            "mix_out_time_s": analysis.get("mix_out_time_s"),
            "outro_start_s": analysis.get("outro_start_s"),
            "best_mix_in": json.dumps(analysis.get("best_mix_in"), ensure_ascii=False) if analysis.get("best_mix_in") else "",
            "best_mix_out": json.dumps(analysis.get("best_mix_out"), ensure_ascii=False) if analysis.get("best_mix_out") else "",
        })

    df = pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logging.info("Saved %d rows to %s", len(df), csv_path.resolve())

# -------------------------
# Playlist pipeline (unchanged)
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

    ydl_list = create_ydl(
        listing=True,
        outdir=Path("."),
        codec=codec,
        mp3_bitrate=mp3_bitrate,
        cookies_from_browser=cookies_from_browser,
        cookiefile=cookiefile,
    )
    items, playlist_title = list_playlist(playlist_url, ydl_list)
    if not items:
        logging.error("No playlist items found. Exiting.")
        return

    if csv_path is None:
        safe_title = "".join(ch for ch in playlist_title if ch.isalnum() or ch in " _-").strip()
        csv_path = outdir / f"{safe_title or 'playlist'}.csv"

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
            if title and not analysis.get("title"):
                analysis["title"] = title
            write_track_json(audio_path, analysis)
        else:
            if download_audio:
                logging.warning("Skipping analysis (file missing or download failed): %s", title)
            analysis = {
                "title": title,
                "bpm": None, "beats_count": None, "bars_count": None,
                "key_root": None, "mode": None, "camelot": None, "key_confidence": None,
                "mix_in_time_s": None, "mix_out_time_s": None,
                "intro_end_s": None, "outro_start_s": None,
                "best_mix_in": None, "best_mix_out": None,
            }

        rows.append({
            "title": analysis.get("title") or title,
            "youtube_url": url,
            "downloaded_path": str(audio_path) if audio_path else "",
            "bpm": analysis.get("bpm"),
            "camelot": analysis.get("camelot"),
            "intro_end_s": analysis.get("intro_end_s"),
            "mix_in_time_s": analysis.get("mix_in_time_s"),
            "mix_out_time_s": analysis.get("mix_out_time_s"),
            "outro_start_s": analysis.get("outro_start_s"),
            "best_mix_in": json.dumps(analysis.get("best_mix_in"), ensure_ascii=False) if analysis.get("best_mix_in") else "",
            "best_mix_out": json.dumps(analysis.get("best_mix_out"), ensure_ascii=False) if analysis.get("best_mix_out") else "",
        })

    df = pd.DataFrame(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logging.info("Saved %d rows to %s", len(df), csv_path.resolve())
    logging.info("Tip: choose next tracks where BPM within ±2–4 and camelot_distance(current, candidate) <= 1.")

# -------------------------
# CLI
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze YouTube playlist (download+analyze) OR a local folder of audio files, and output DJ metadata."
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--playlist-url", help="YouTube playlist URL (enables yt-dlp playlist mode)")
    src.add_argument("--localdir", help="Path to folder with existing audio files (analyzes WAV/MP3/… in place)")

    p.add_argument("--outdir", default="downloads", help="[playlist mode] Directory to store audio files (default: downloads)")
    p.add_argument("--csv", default=None, help="CSV output path. Default: <outdir>/<playlist-title>.csv (playlist) or <localdir>/local_analysis.csv (local)")
    p.add_argument("--download", action="store_true", help="[playlist mode] Actually download audio. Omit to only list & CSV of URLs.")
    p.add_argument("--codec", choices=["wav", "mp3"], default="wav", help="[playlist mode] Output audio codec (default: wav)")
    p.add_argument("--mp3-bitrate", default="192", help="[playlist mode] MP3 bitrate kbps (only if --codec mp3). Default: 192")
    p.add_argument("--browser", choices=["chrome", "firefox", "edge", "brave", "chromium"],
                   help="[playlist mode] Use cookies from this browser (must be logged in).")
    p.add_argument("--cookiefile", help="[playlist mode] Path to cookies.txt exported from your browser.")
    p.add_argument("-v", "--verbose", action="count", default=1, help="Increase verbosity: -v (info, default), -vv (debug)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.verbose)

    if args.localdir:
        process_local_folder(localdir=Path(args.localdir), csv_path=(Path(args.csv) if args.csv else None))
    else:
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
