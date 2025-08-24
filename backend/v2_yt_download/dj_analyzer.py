#!/usr/bin/env python3
# dj_analyzer.py
from __future__ import annotations
import logging, json, traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import librosa
from scipy.signal import find_peaks

def _time_to_index(t: float, starts: List[float]) -> Optional[int]:
    """Return 1-based index of the segment whose start <= t < next_start.
    If t is before the first start, return 1. If list is empty, None.
    """
    try:
        if not starts:
            return None
        # ensure sorted numeric
        xs = [float(x) for x in starts]
        idx = 1
        for i, s in enumerate(xs):
            if t >= s:
                idx = i + 1
            else:
                break
        return idx
    except Exception:
        return None

# optional
try:
    import pyloudnorm as pyln
    HAS_LOUDNORM = True
except Exception:
    HAS_LOUDNORM = False


# -------- utils --------
def to_float(x) -> float:
    try:
        arr = np.asarray(x)
        if arr.ndim == 0: return float(arr.item())
        return float(arr.flat[0])
    except Exception:
        try: return float(x)
        except Exception: return 0.0

def r2(x) -> float:
    return round(to_float(x), 2)


# -------- key / camelot --------
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
    conf = float(sorted_scores[0][2] - sorted_scores[1][2])

    key_root = _PITCH_NAMES[root]
    camelot  = _CAMELOT_MAJOR[root] if mode == "maj" else _CAMELOT_MINOR[root]
    return key_root, ("major" if mode=="maj" else "minor"), camelot, round(conf, 4)


# -------- energy + loudness --------
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


# -------- downbeat / bars --------
def estimate_downbeat_offset(y, sr, beat_frames) -> int:
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


# -------- vocal curve --------
def vocal_likelihood_curve(y, sr, include_curve=True):
    y_h, _ = librosa.effects.hpss(y)
    S = librosa.feature.melspectrogram(y=y_h, sr=sr, n_fft=2048, hop_length=512, n_mels=64)
    freqs = librosa.mel_frequencies(n_mels=S.shape[0], fmin=0, fmax=sr//2)
    band = (freqs >= 300) & (freqs <= 3400)
    band_energy = S[band].sum(axis=0)
    total_energy = np.maximum(S.sum(axis=0), 1e-8)
    raw = band_energy / total_energy

    # light smoothing by moving average (no SciPy dependency)
    win = 5
    if len(raw) >= win:
        kernel = np.ones(win) / win
        smoothed = np.convolve(raw, kernel, mode="same")
    else:
        smoothed = raw

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


# -------- structure heuristics --------
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
        out_idx = max(0, len(beat_times) - 32)
        outro_start_s = float(beat_times[out_idx])
    else:
        outro_start_s = float(beat_times[0]) if len(beat_times) else 0.0

    # Breakdowns from energy local minima (~6s window)
    breakdowns = []
    if times_ds is not None and len(times_ds) > 2:
        dt = max(1e-6, to_float(times_ds[1]) - to_float(times_ds[0]))
        win = max(1, int(6.0 / dt))
        for i in range(win, len(rms_db_ds)-win):
            center = to_float(rms_db_ds[i])
            window_vals = [to_float(v) for v in rms_db_ds[i-win:i+win+1]]
            if center == min(window_vals) and center < (rms_db_mean - 0.3*rms_db_std):
                breakdowns.append(to_float(times_ds[i]))

    # Drops: strong onsets shortly after breakdowns
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
    return [r2(float(beat_times[i])) for i in range(0, len(beat_times), phrase)]

def bar_starts_simple(beat_times, beats_per_bar=4):
    return [r2(float(beat_times[i])) for i in range(0, len(beat_times), beats_per_bar)]


# -------- analyze one file --------
def analyze_audio(path: Path) -> Dict[str, Any]:
    logging.info("Analyzing audio: %s", path)
    try:
        y, sr = librosa.load(path, sr=None, mono=True)

        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        bpm = float(tempo)
        bars = int(len(beat_times) // 4)

        key_root, mode, camelot, key_conf = estimate_key_ks(y, sr)

        struct = structure_heuristics(y, sr, beat_times)
        phrases = phrase_markers(beat_times)
        bars_simple = bar_starts_simple(beat_times)

        downbeat_offset = estimate_downbeat_offset(y, sr, beat_frames)
        bars_downbeat = bars_from_beats_with_offset(beat_times, downbeat_offset)
        bar1_time_s   = bars_downbeat[0] if bars_downbeat else (float(beat_times[0]) if len(beat_times) else 0.0)

        vocal_mean, vocal_t, vocal_v = vocal_likelihood_curve(y, sr, include_curve=True)
        LOW_VOCAL_THR = 0.4
        low_vocal_bar_starts = []
        for bt in bars_downbeat:
            # quick local mean around each bar
            if vocal_t is not None and vocal_v is not None and len(vocal_t):
                # nearest frame to bt
                idx = int(np.argmin(np.abs(np.asarray(vocal_t) - bt)))
                vs = float(vocal_v[idx])
            else:
                vs = float(vocal_mean)
            if vs <= LOW_VOCAL_THR:
                low_vocal_bar_starts.append(r2(bt))

        mix_in_time = phrases[1] if len(phrases) > 1 else (beat_times[32] if len(beat_times) > 32 else 0.0)
        mix_out_time = struct["outro_start_s"]
        # compute bar/phrase indices for key markers (1-based)
        mix_in_bar  = _time_to_index(float(mix_in_time), bars_downbeat) or _time_to_index(float(mix_in_time), bars_simple) or None
        mix_out_bar = _time_to_index(float(mix_out_time), bars_downbeat) or _time_to_index(float(mix_out_time), bars_simple) or None
        intro_end_bar   = _time_to_index(float(struct["intro_end_s"]), bars_downbeat) if struct.get("intro_end_s") is not None else None
        outro_start_bar = _time_to_index(float(struct["outro_start_s"]), bars_downbeat) if struct.get("outro_start_s") is not None else None
        mix_in_phrase_index  = _time_to_index(float(mix_in_time), phrases) if phrases else None
        mix_out_phrase_index = _time_to_index(float(mix_out_time), phrases) if phrases else None

        
        # --- Added: bar/phrase index arrays for compact, beat-based navigation ---
        bar_starts_s_out = bars_simple[:200]
        bar_starts_bars = list(range(1, len(bar_starts_s_out) + 1))

        phrase_s_out = phrases[:50] if phrases else []
        phrase_boundaries_bars = []
        if phrase_s_out:
            for p in phrase_s_out:
                idx = _time_to_index(float(p), bars_downbeat) or _time_to_index(float(p), bars_simple) or None
                phrase_boundaries_bars.append(idx)
        phrase_boundaries_phrases = list(range(1, len(phrase_s_out) + 1))

        lufs = estimate_lufs(y, sr)

        return {
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
            "mix_in_bar": mix_in_bar,
            "mix_out_bar": mix_out_bar,
            "intro_end_bar": intro_end_bar,
            "outro_start_bar": outro_start_bar,
            "mix_in_phrase_index": mix_in_phrase_index,
            "mix_out_phrase_index": mix_out_phrase_index,
            "intro_end_s": struct["intro_end_s"],
            "outro_start_s": struct["outro_start_s"],
            "phrase_boundaries_s": phrase_s_out,
            "phrase_boundaries_bars": phrase_boundaries_bars,
            "phrase_boundaries_phrases": phrase_boundaries_phrases,
            "bar_starts_s": bar_starts_s_out,
            "bar_starts_bars": bar_starts_bars,
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

    except Exception as e:
        logging.error("Analysis failed for %s: %s", path, e)
        traceback.print_exc()
        return {
            "title": path.stem,
            "bpm": None, "beats_count": None, "bars_count": None,
            "key_root": None, "mode": None, "camelot": None, "key_confidence": None,
            "mix_in_time_s": None, "mix_out_time_s": None, "mix_in_bar": None, "mix_out_bar": None, "intro_end_bar": None, "outro_start_bar": None, "mix_in_phrase_index": None, "mix_out_phrase_index": None,
            "intro_end_s": None, "outro_start_s": None,
            "phrase_boundaries_s": [], "bar_starts_s": [], "downbeat_bar_starts_s": [],
            "downbeat_offset": None, "bar1_time_s": None, "low_vocal_bar_starts_s": [],
            "onset_peaks_s": [], "breakdowns_s": [], "drops_s": [],
            "rms_db_mean": None, "rms_db_std": None, "lufs_integrated": None,
            "energy_curve_sample": {"times_s": [], "rms_db": []},
            "vocal_mean": None, "vocal_curve_sample": {"times_s": [], "values": []},
        }


def write_track_json(audio_path: Path, analysis: Dict[str, Any]):
    meta_path = audio_path.with_suffix("").with_suffix(".djmeta.json")
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2, separators=(",", ": "))
        logging.info("Wrote metadata JSON: %s", meta_path)
    except Exception as e:
        logging.warning("Failed to write track JSON for %s: %s", audio_path, e)
