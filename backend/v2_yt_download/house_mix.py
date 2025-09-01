#!/usr/bin/env python3
"""
house_mix.py — Auto‑mix two house tracks using analyzer .djmeta.json

Features
- Phrase-aligned planner (aligns B intro/mix-in to A outro/mix-out)
- Recipes: intro_to_outro (default), breakdown_to_drop, drop_swap
- Phase‑vocoder time‑stretch (librosa >= 0.10 compatible)
- 3‑band EQ‑style crossfade over N bars
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, filtfilt

# ---------------------------- JSON -----------------------------------------
def read_json(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------------------- Filters / bands -------------------------------
LOW_CUTOFF = 150.0   # Hz
HIGH_CUTOFF = 8000.0 # Hz

def _butter_band_filters(sr: int):
    ny = 0.5 * sr
    low = max(1e-6, LOW_CUTOFF / ny)
    high = min(0.9999, HIGH_CUTOFF / ny)
    bL, aL = butter(2, low, btype="low")
    bM, aM = butter(2, [low, high], btype="band")
    bH, aH = butter(2, high, btype="high")
    return (bL, aL), (bM, aM), (bH, aH)

def split_bands(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    (bL, aL), (bM, aM), (bH, aH) = _butter_band_filters(sr)
    if y.ndim == 1:
        low = filtfilt(bL, aL, y)
        mid = filtfilt(bM, aM, y)
        high = filtfilt(bH, aH, y)
    else:
        low  = np.vstack([filtfilt(bL, aL, y[c]) for c in range(y.shape[0])])
        mid  = np.vstack([filtfilt(bM, aM, y[c]) for c in range(y.shape[0])])
        high = np.vstack([filtfilt(bH, aH, y[c]) for c in range(y.shape[0])])
    return low.astype(np.float32), mid.astype(np.float32), high.astype(np.float32)

# ---------------------------- Helpers ---------------------------------------
def resample_to_match(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return y
    if y.ndim == 1:
        return librosa.resample(y, orig_sr=sr_in, target_sr=sr_out)
    chans = [librosa.resample(y[c], orig_sr=sr_in, target_sr=sr_out) for c in range(y.shape[0])]
    m = max(ch.shape[-1] for ch in chans)
    chans = [np.pad(ch, (0, m - ch.shape[-1])) for ch in chans]
    return np.vstack(chans)

def timestretch_waveform(y: np.ndarray, rate: float, sr: int, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    """Time‑stretch waveform by rate using STFT + phase vocoder (librosa>=0.10)."""
    rate = float(rate)
    if not np.isfinite(rate) or abs(rate - 1.0) < 1e-6:
        return y
    if y.ndim == 1:
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        Ds = librosa.phase_vocoder(D, rate=rate, hop_length=hop_length)
        out_len = int(round(len(y) / rate))
        y_out = librosa.istft(Ds, hop_length=hop_length, length=out_len)
        return y_out.astype(np.float32)
    else:
        chans = []
        for c in range(y.shape[0]):
            D = librosa.stft(y[c], n_fft=n_fft, hop_length=hop_length)
            Ds = librosa.phase_vocoder(D, rate=rate, hop_length=hop_length)
            out_len = int(round(y[c].shape[-1] / rate))
            y_out = librosa.istft(Ds, hop_length=hop_length, length=out_len)
            chans.append(y_out.astype(np.float32))
        m = max(len(ch) for ch in chans)
        chans = [np.pad(ch, (0, m - len(ch))) for ch in chans]
        return np.vstack(chans)

def timestretch_to_bpm(y: np.ndarray, bpm_src: float, bpm_tgt: float, sr: int) -> np.ndarray:
    if bpm_src <= 0 or bpm_tgt <= 0 or abs(bpm_src - bpm_tgt) < 1e-6:
        return y
    rate = float(bpm_tgt) / float(bpm_src)
    return timestretch_waveform(y, rate, sr)

def secs_to_samples(t: float, sr: int) -> int:
    return int(np.round(float(t) * sr))

def mk_env(length: int, start_gain: float, end_gain: float) -> np.ndarray:
    return np.linspace(start_gain, end_gain, num=length, dtype=np.float32)

# ---------------------------- Planner ---------------------------------------
def nearest_phrase_at_or_after(t_s: float, phrases_s: List[float]) -> float:
    if not phrases_s:
        return t_s
    for p in phrases_s:
        if p >= t_s:
            return p
    return phrases_s[-1]

def phrase_seconds(bpm: float, bars_per_phrase: int = 8) -> float:
    if bpm <= 0:
        bpm = 128.0
    return (4 * bars_per_phrase) * 60.0 / bpm

def build_candidates(recipe: str, A: Dict[str, Any], B: Dict[str, Any], tgt_bpm: float) -> List[Tuple[float, float, str]]:
    a_ph = A.get("phrase_boundaries_s") or []
    b_ph = B.get("phrase_boundaries_s") or []
    a_out = A.get("mix_out_time_s") or A.get("outro_start_s") or 0.0
    b_in  = B.get("mix_in_time_s")  or B.get("intro_end_s")  or 0.0
    a_out = nearest_phrase_at_or_after(a_out, a_ph)
    b_in  = nearest_phrase_at_or_after(b_in,  b_ph)
    ph_len = phrase_seconds(tgt_bpm, 8)
    cands: List[Tuple[float,float,str]] = []
    if recipe == "intro_to_outro":
        for delta in (-ph_len, 0.0, ph_len):
            cands.append((max(0.0, a_out + delta), b_in, f"A_out{delta:+.0f}s→B_in"))
    elif recipe == "breakdown_to_drop":
        drops = B.get("drops_s") or []
        b_drop = nearest_phrase_at_or_after((drops[0] if drops else b_in + ph_len), b_ph)
        for pre in (ph_len, 0.0):
            cands.append((max(0.0, a_out - pre), max(0.0, b_drop - ph_len), "A_out→B_drop"))
    elif recipe == "drop_swap":
        a_drop = nearest_phrase_at_or_after((A.get("drops_s") or [a_out])[0], a_ph)
        b_drop = nearest_phrase_at_or_after((B.get("drops_s") or [b_in])[0], b_ph)
        cands.append((a_drop, b_drop, "drop_swap"))
    else:
        for delta in (-ph_len, 0.0, ph_len):
            cands.append((max(0.0, a_out + delta), b_in, f"A_out{delta:+.0f}s→B_in"))
    return cands

def plan_alignment(A: Dict[str, Any], B: Dict[str, Any], bars: int, bpm_target: float | None, recipe: str) -> Tuple[float, float, float, str]:
    bpm_a = float(A.get("bpm") or 0)
    bpm_b = float(B.get("bpm") or 0)
    tgt = float(bpm_target) if (bpm_target and float(bpm_target) > 60) else (bpm_a or bpm_b or 128.0)
    cands = build_candidates(recipe, A, B, tgt)

    # simple scoring using vocals/energy (prefer low vocals; A lower energy, B ok energy)
    def score_window(meta: Dict[str, Any], t_s: float, mode: str) -> float:
        vc = meta.get("vocal_curve_sample", {})
        ec = meta.get("energy_curve_sample", {})
        vt = np.array(vc.get("times_s") or []); vv = np.array(vc.get("values") or [])
        et = np.array(ec.get("times_s") or []); ev = np.array(ec.get("rms_db") or [])
        def interp(times, vals, t, default=0.4):
            if len(times) == 0: return default
            idx = int(np.argmin(np.abs(times - t)))
            return float(vals[idx])
        v = interp(vt, vv, t_s, default=0.3)  # prefer low vocals
        e = interp(et, ev, t_s, default=0.0)
        if mode == "out":   # prefer low/declining energy
            return 0.6*v + 0.4*max(0.0, e)
        else:               # prefer low vocals; penalize ultra-low energy
            return 0.7*v + 0.3*max(0.0, -e)

    scored = []
    for a_anchor, b_anchor, label in cands:
        s = score_window(A, a_anchor, "out") + score_window(B, b_anchor, "in")
        scored.append((s, a_anchor, b_anchor, label))
    scored.sort(key=lambda x: x[0])
    _, a_best, b_best, chosen_label = scored[0]

    bar_dur = 4.0 * 60.0 / max(tgt, 1e-6)
    crossfade_time_s = bars * bar_dur

    # return anchors (seconds) so main can align B intro to A outro exactly
    return a_best, b_best, crossfade_time_s, chosen_label

# ---------------------------- Envelopes / render ----------------------------
def equal_power_fades(n: int) -> Tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return np.cos(0.5 * np.pi * t), np.sin(0.5 * np.pi * t)

def build_eq_envelopes(sr: int, crossfade_len_s: float) -> Dict[str, Any]:
    cf_len = secs_to_samples(crossfade_len_s, sr)
    a_amp_cf, b_amp_cf = equal_power_fades(cf_len)
    half = cf_len // 2
    a_low_cf = np.concatenate([mk_env(half, 1.0, 0.3), mk_env(cf_len - half, 0.3, 0.0)])
    b_low_cf = np.concatenate([mk_env(half, 0.0, 0.0), mk_env(cf_len - half, 0.0, 1.0)])
    a_mid_cf = a_amp_cf * 0.9
    b_mid_cf = b_amp_cf * 1.0
    a_high_cf = a_amp_cf * 0.9
    b_high_cf = b_amp_cf * 1.0
    return {
        "len_cf": cf_len,
        "a": {"amp": a_amp_cf, "low": a_low_cf, "mid": a_mid_cf, "high": a_high_cf},
        "b": {"amp": b_amp_cf, "low": b_low_cf, "mid": b_mid_cf, "high": b_high_cf},
    }

def apply_band_envs(y: np.ndarray, sr: int,
                    env_amp: np.ndarray, env_low: np.ndarray,
                    env_mid: np.ndarray, env_high: np.ndarray,
                    start_sample: int, total_len: int) -> np.ndarray:
    """Apply band envelopes safely, handling partial overlaps and clip lengths."""
    low, mid, high = split_bands(y, sr)
    out = np.zeros((y.shape[0], total_len), dtype=np.float32)

    cf_len = len(env_amp)
    seg_start = start_sample
    seg_end   = start_sample + cf_len

    def place(sig: np.ndarray, band_env: np.ndarray) -> np.ndarray:
        o = np.zeros_like(out)
        for c in range(sig.shape[0]):
            sig_len = sig.shape[1]
            # Pre region
            pre_end = min(seg_start, sig_len, total_len)
            if pre_end > 0:
                o[c, :pre_end] = sig[c, :pre_end] * band_env[0]

            # Mid (crossfade) region
            mid_start = min(seg_start, sig_len, total_len)
            mid_stop_raw = min(seg_end, sig_len, total_len)
            mid_len = max(0, mid_stop_raw - mid_start)
            if mid_len > 0:
                env_slice = band_env[:mid_len]
                o[c, mid_start:mid_start+mid_len] = sig[c, mid_start:mid_start+mid_len] * env_slice

            # Post region
            post_start_eff = min(mid_stop_raw, sig_len, total_len)
            if post_start_eff < min(sig_len, total_len):
                tail_len = min(sig_len, total_len) - post_start_eff
                o[c, post_start_eff:post_start_eff+tail_len] = sig[c, post_start_eff:post_start_eff+tail_len] * band_env[min(len(band_env)-1, max(0, mid_len-1))]
        return o

    band_amp = env_amp
    low_env  = band_amp * env_low
    mid_env  = band_amp * env_mid
    high_env = band_amp * env_high

    out += place(low,  low_env)
    out += place(mid,  mid_env)
    out += place(high, high_env)
    return out

# ---------------------------- Main ------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Auto-mix two house tracks using .djmeta.json")
    ap.add_argument('--track-a', required=True)
    ap.add_argument('--meta-a', required=True)
    ap.add_argument('--track-b', required=True)
    ap.add_argument('--meta-b', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--bars', type=int, default=32, help='Crossfade length in bars (default 32)')
    ap.add_argument('--recipe', choices=['intro_to_outro','breakdown_to_drop','drop_swap'], default='intro_to_outro')
    ap.add_argument('--target-bpm', default='auto', help="Target BPM ('auto'=use A's BPM or provide a number)")
    args = ap.parse_args()

    # load JSON
    A = read_json(Path(args.meta_a))
    B = read_json(Path(args.meta_b))

    # load audio (preserve stereo)
    yA, srA = librosa.load(args.track_a, sr=None, mono=False)
    yB, srB = librosa.load(args.track_b, sr=None, mono=False)
    if yA.ndim == 1: yA = yA[np.newaxis, :]
    if yB.ndim == 1: yB = yB[np.newaxis, :]

    # resample B to A's samplerate if needed
    if srB != srA:
        yB = resample_to_match(yB, srB, srA)
        srB = srA

    # choose target bpm & plan
    tgt = None if args.target_bpm == 'auto' else float(args.target_bpm)
    a_anchor_s, b_anchor_s, cf_len_s, label = plan_alignment(A, B, args.bars, tgt, args.recipe)

    # stretch B to target bpm
    bpmA = float(A.get('bpm') or 0.0)
    bpmB = float(B.get('bpm') or 0.0)
    bpm_target = bpmA if (tgt is None) else tgt
    if bpm_target <= 0:
        bpm_target = bpmA if bpmA > 0 else bpmB if bpmB > 0 else 128.0
    yB_ts = timestretch_to_bpm(yB, bpmB if bpmB > 0 else bpm_target, bpm_target, srA)

    # crossfade anchor at A's anchor time
    cf_start = secs_to_samples(a_anchor_s, srA)

    # place B so that its intro/mix-in anchor lands at cf_start
    b_start_at = cf_start - secs_to_samples(b_anchor_s, srA)
    b_start_at = max(0, b_start_at)

    # envelopes
    env = build_eq_envelopes(srA, cf_len_s)

    # compute total length to fit A, B placement, and crossfade window
    lenA = yA.shape[1]
    total_len = max(lenA, b_start_at + yB_ts.shape[1], cf_start + env['len_cf'])

    # apply envelopes to A (fade out across window)
    outA = apply_band_envs(yA, srA, env['a']['amp'], env['a']['low'], env['a']['mid'], env['a']['high'],
                           start_sample=cf_start, total_len=total_len)

    # build B timeline and apply envelopes (fade in across window)
    timelineB = np.zeros_like(outA)
    endB = min(timelineB.shape[-1], b_start_at + yB_ts.shape[1])
    seg_len = max(0, endB - b_start_at)
    if seg_len > 0:
        timelineB[..., b_start_at:endB] = yB_ts[..., :seg_len]
    outB = apply_band_envs(timelineB, srA, env['b']['amp'], env['b']['low'], env['b']['mid'], env['b']['high'],
                           start_sample=cf_start, total_len=total_len)

    # sum and normalize lightly
    mix = outA + outB
    peak = float(np.max(np.abs(mix)) + 1e-9)
    if peak > 1.0:
        mix = mix / peak * 0.98

    # write file (samples, channels)
    wav = mix.T if mix.ndim == 2 else mix
    sf.write(args.out, wav, srA)
    print(f"Recipe={args.recipe} | Align=A@{a_anchor_s:.2f}s to B@{b_anchor_s:.2f}s | Bars={args.bars} | TargetBPM={bpm_target:.2f}")
    print(f"Wrote: {args.out} | sr={srA} | length_s={wav.shape[0]/srA:.2f}")

if __name__ == "__main__":
    main()
