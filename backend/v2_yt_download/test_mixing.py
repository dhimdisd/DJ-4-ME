#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os, shutil, math, argparse, subprocess, tempfile
from pathlib import Path

# --- Optional deps (graceful fallbacks) ---
HAS_LIBROSA = False
HAS_SOUND_FILE = False
HAS_PYRUBBERBAND = False

try:
    import librosa
    HAS_LIBROSA = True
except Exception:
    HAS_LIBROSA = False

try:
    import soundfile as sf
    HAS_SOUND_FILE = True
except Exception:
    HAS_SOUND_FILE = False

try:
    import pyrubberband as pyrb
    HAS_PYRUBBERBAND = True
except Exception:
    HAS_PYRUBBERBAND = False

from pydub import AudioSegment

# ---------------- Utils ----------------

def segment_to_mono_array(seg: AudioSegment, target_sr=11025):
    """Convert AudioSegment to mono float32 array at target_sr (fallback analysis)."""
    if seg.frame_rate != target_sr:
        seg = seg.set_frame_rate(target_sr)
    seg = seg.set_channels(1)
    arr = seg.get_array_of_samples()
    import numpy as np
    x = np.array(arr).astype(np.float32)
    x /= max(1, (2 ** (8 * seg.sample_width - 1) - 1))  # normalize
    return x, target_sr

def energy_envelope(y, frame_size=1024, hop=512):
    import numpy as np
    if len(y) < frame_size:
        return np.array([float((y**2).sum())], dtype=np.float32)
    n_frames = 1 + (len(y) - frame_size) // hop
    env = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        s = i * hop
        env[i] = float((y[s:s+frame_size]**2).sum())
    env -= env.min()
    m = env.max()
    if m > 1e-12:
        env /= m
    return env

def estimate_bpm_fallback(seg: AudioSegment, min_bpm=80, max_bpm=140):
    """Simple autocorr BPM estimator (fallback if librosa not available)."""
    import numpy as np
    y, sr = segment_to_mono_array(seg, target_sr=11025)
    env = energy_envelope(y, frame_size=1024, hop=512)
    env = env - env.mean()
    ac = np.correlate(env, env, mode='full')[len(env)-1:]
    env_rate = sr / 512.0  # frames per second
    min_lag = max(1, int(env_rate * 60.0 / max_bpm))
    max_lag = max(min_lag+1, int(env_rate * 60.0 / min_bpm))
    search = ac[min_lag:max_lag]
    if len(search) == 0:
        return 120.0
    peak_lag = int(search.argmax()) + min_lag
    bpm = 60.0 * env_rate / peak_lag
    # snap near common house values
    for c in [118,120,122,124,126,128]:
        if abs(bpm - c) < 1.0:
            return float(c)
    return float(bpm)

def estimate_bpm_librosa(path: str):
    y, sr = librosa.load(path, sr=None, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(tempo if tempo > 0 else 120.0)

def estimate_bpm(path: str, seg: AudioSegment):
    """Try librosa first, fallback to internal method."""
    if HAS_LIBROSA:
        try:
            return estimate_bpm_librosa(path)
        except Exception:
            pass
    return estimate_bpm_fallback(seg)

def snap_length_to_beats(length_ms: int, bpm: float, beats: int = 4):
    beat_ms = 60000.0 / max(1e-6, bpm)
    block_ms = beat_ms * beats
    snapped = int((length_ms // int(block_ms)) * int(block_ms))
    return max(snapped, int(block_ms))

def bars_to_ms(bars: int, bpm: float):
    beat_ms = 60000.0 / max(1e-6, bpm)
    return int(bars * 4 * beat_ms)

def time_stretch_pitch_locked_wav(in_wav: str, out_wav: str, rate: float):
    """Stretch time with pitch preserved. Tries pyrubberband, else rubberband CLI, else raises."""
    if abs(rate - 1.0) < 1e-6:
        shutil.copyfile(in_wav, out_wav)
        return

    if HAS_PYRUBBERBAND and HAS_SOUND_FILE:
        y, sr = sf.read(in_wav)
        y_st = pyrb.time_stretch(y, sr, rate)
        sf.write(out_wav, y_st, sr)
        return

    # Try rubberband CLI
    if shutil.which("rubberband"):
        # rubberband --tempo RATE*100 (percent)
        tempo_percent = rate * 100.0
        cmd = ["rubberband", "--tempo", f"{tempo_percent:.6f}", "--pitch", "0", in_wav, out_wav]
        subprocess.run(cmd, check=True)
        return

    raise RuntimeError(
        "Pitch-locked stretching unavailable. Install pyrubberband (plus rubberband library) "
        "or the 'rubberband' CLI."
    )

def resample_like_platter(seg: AudioSegment, rate: float) -> AudioSegment:
    """Simple resample (tempo+pitch change) like nudging the platter."""
    if abs(rate - 1.0) < 1e-6:
        return seg
    new_rate = int(seg.frame_rate * rate)
    seg2 = seg._spawn(seg.raw_data, overrides={"frame_rate": new_rate})
    return seg2.set_frame_rate(seg.frame_rate)

# ------- EQ helpers (DJ isolator-ish) -------
def low_cut(seg: AudioSegment, cutoff=180):
    return seg.high_pass_filter(cutoff)

def high_cut(seg: AudioSegment, cutoff=10000):
    return seg.low_pass_filter(cutoff)

# -------------- Core mixer --------------

def make_transition(song1: AudioSegment,
                    song2: AudioSegment,
                    bpm1: float,
                    bpm2: float,
                    overlap_bars: int = 16,
                    style: str = "smooth",
                    key_lock: bool = True):
    """
    Returns final_mix, bpm1, bpm2, song2_synced
    style: 'smooth' | 'bass_swap'
    """
    # Tempo match song2 to bpm1
    rate = bpm1 / max(1e-6, bpm2)

    # Use key-lock if requested and available; otherwise resample
    if key_lock:
        # Write temp wavs and stretch with pitch lock
        with tempfile.TemporaryDirectory() as td:
            in_wav = str(Path(td) / "in.wav")
            out_wav = str(Path(td) / "out.wav")
            # export song2 to wav for highest quality
            song2.export(in_wav, format="wav")
            try:
                time_stretch_pitch_locked_wav(in_wav, out_wav, rate)
                song2_sync = AudioSegment.from_file(out_wav, format="wav")
            except Exception as e:
                print(f"[warn] Key-lock stretch not available ({e}). Falling back to resample.")
                song2_sync = resample_like_platter(song2, rate)
    else:
        song2_sync = resample_like_platter(song2, rate)

    # Compute overlap in ms and snap to bars
    overlap_ms = bars_to_ms(overlap_bars, bpm1)
    outro_ms = snap_length_to_beats(overlap_ms, bpm1, beats=4)
    intro_ms = outro_ms

    # Build parts aligned to bar boundaries (end of song1, start of song2)
    part1_main = song1[:-outro_ms] if len(song1) > outro_ms else AudioSegment.silent(duration=0)
    part1_outro = song1[-outro_ms:]
    part2_intro = song2_sync[:intro_ms]
    part2_rest = song2_sync[intro_ms:]

    if style == "smooth":
        # soften highs of outgoing, tame lows of incoming, long crossfade
        p1 = high_cut(part1_outro, cutoff=10000).fade_out(outro_ms)
        p2 = low_cut(part2_intro, cutoff=200).fade_in(intro_ms)
        mix = p1.overlay(p2)
    elif style == "bass_swap":
        # first half: cut lows from incoming; second half: cut lows from outgoing
        half = intro_ms // 2
        p1_first = part1_outro[:half]
        p2_first = low_cut(part2_intro[:half], cutoff=200).apply_gain(-1).fade_in(half)
        first_mix = p1_first.overlay(p2_first - 2)

        p1_second = low_cut(part1_outro[half:], cutoff=200).apply_gain(-2).fade_out(intro_ms - half)
        p2_second = high_cut(part2_intro[half:], cutoff=10000).fade_in(intro_ms - half)
        second_mix = p1_second.overlay(p2_second)

        mix = first_mix + second_mix
    else:
        # safe default crossfade
        mix = part1_outro.fade_out(outro_ms).overlay(part2_intro.fade_in(intro_ms))

    final_mix = part1_main + mix + part2_rest
    return final_mix, bpm1, bpm2, song2_sync


# -------------- CLI --------------

def main():
    ap = argparse.ArgumentParser(description="Beat-sync and mix two house tracks like a DJ.")
    ap.add_argument("--song1", required=True, help="Path to first track (plays first).")
    ap.add_argument("--song2", required=True, help="Path to second track (mixes in).")
    ap.add_argument("--out",   required=True, help="Output file path (e.g., mix.mp3).")
    ap.add_argument("--style", choices=["smooth","bass_swap"], default="bass_swap",
                    help="Transition style.")
    ap.add_argument("--overlap-bars", type=int, default=32, help="Overlap length in bars.")
    ap.add_argument("--key-lock", action="store_true", help="Preserve pitch when tempo-matching.")
    ap.add_argument("--no-key-lock", action="store_true", help="Force platter-style resampling.")
    ap.add_argument("--bpm1", type=float, default=0.0, help="Override detected BPM for song1.")
    ap.add_argument("--bpm2", type=float, default=0.0, help="Override detected BPM for song2.")
    ap.add_argument("--format", default=None, help="Output format (mp3/wav/flac). Default inferred.")
    args = ap.parse_args()

    # Load with pydub (so we can filter and mix)
    s1 = AudioSegment.from_file(args.song1)
    s2 = AudioSegment.from_file(args.song2)

    # BPM detection
    bpm1 = args.bpm1 if args.bpm1 > 0 else estimate_bpm(args.song1, s1)
    bpm2 = args.bpm2 if args.bpm2 > 0 else estimate_bpm(args.song2, s2)

    print(f"[info] Detected BPMs: song1={bpm1:.2f}, song2={bpm2:.2f}")

    key_lock = args.key_lock and not args.no_key_lock
    final_mix, _, _, _ = make_transition(
        s1, s2, bpm1, bpm2,
        overlap_bars=args.overlap_bars,
        style=args.style,
        key_lock=key_lock
    )

    # Export
    out_path = Path(args.out)
    fmt = args.format or out_path.suffix.lstrip(".").lower() or "mp3"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_mix.export(out_path.as_posix(), format=fmt)
    print(f"[done] Wrote {out_path} ({fmt})")

if __name__ == "__main__":
    main()
