#!/usr/bin/env python3
"""
serato_mixplan_and_hotcues.py

One-stop tool:
- Read .djmeta.json files (from yt_playlist_to_dj_metadata.py)
- Score pairwise compatibility (BPM proximity, Camelot distance, phrasing)
- Build a greedy set order
- Export a mix plan CSV (who → who, when to start/stop)
- Export dual hotcues CSV per track (IN + OUT, downbeat + vocal aware)

Usage:
  python serato_mixplan_and_hotcues.py metas/ \
      --plan-out mix_plan.csv \
      --hotcues-out hotcues_dual/ \
      --combined-cues all_hotcues_dual.csv \
      --bpm-tolerance 2.5 --max-harmonic-distance 1
"""

import argparse, json, csv, math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# -------------------------
# Small helpers
# -------------------------
def to_float(x) -> float:
    try:
        arr = np.asarray(x)
        if arr.ndim == 0: return float(arr.item())
        return float(arr.flat[0])
    except Exception:
        try: return float(x)
        except Exception: return 0.0

def r2(x): return round(to_float(x), 2)

def load_meta(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def collect_metas(inputs: List[str]) -> List[Path]:
    out: List[Path] = []
    for s in inputs:
        P = Path(s)
        if P.is_dir():
            out.extend(sorted(P.glob("*.djmeta.json")))
            # also accept audio files with sidecar metas
            for a in sorted(P.glob("*")):
                if a.suffix.lower() in (".wav", ".mp3", ".flac", ".aiff", ".aif"):
                    m = a.with_suffix("").with_suffix(".djmeta.json")
                    if m.exists():
                        out.append(m)
        elif P.suffix == ".json" and P.name.endswith(".djmeta.json"):
            out.append(P)
    # dedupe while preserving order
    seen = set(); uniq = []
    for p in out:
        if p.resolve() not in seen:
            uniq.append(p); seen.add(p.resolve())
    return uniq

# -------------------------
# Music compatibility
# -------------------------
def camelot_distance(c1: Optional[str], c2: Optional[str]) -> int:
    if not c1 or not c2: return 99
    # e.g., "8A" -> 8, "A"
    try:
        n1, m1 = int(c1[:-1]), c1[-1].upper()
        n2, m2 = int(c2[:-1]), c2[-1].upper()
    except Exception:
        return 99
    if m1 == m2:
        # same mode ring distance
        d = min((n1-n2) % 12, (n2-n1) % 12)
        return int(d)
    # cross-mode: same number is strong
    if n1 == n2: return 1
    # else treat as 2
    return 2

def compat_score(a: Dict[str, Any], b: Dict[str, Any],
                 bpm_tol: float = 2.5,
                 harm_max: int = 2) -> float:
    """Positive is better. Large negative = avoid."""
    bpm_a = to_float(a.get("bpm") or 0.0)
    bpm_b = to_float(b.get("bpm") or 0.0)
    if bpm_a <= 0 or bpm_b <= 0:
        return -10.0

    # BPM proximity (within tolerance gets big boost, then decays)
    bpm_diff = abs(bpm_a - bpm_b)
    bpm_term = max(0.0, (bpm_tol - bpm_diff) / max(bpm_tol, 1e-6))  # 0..1

    # Harmonic distance (Camelot)
    harm = camelot_distance(a.get("camelot"), b.get("camelot"))
    if harm > harm_max:  # too far harmonically
        harm_term = -1.0
    else:
        harm_term = 1.0 - (harm / max(harm_max, 1e-6))  # 1 → 0 over the range

    # Phrase timing hint: prefer if b’s intro_end exists (we can mix in early clean)
    has_intro_end = 1.0 if b.get("intro_end_s") is not None else 0.0

    # Vocal hint: prefer lower mean vocals on B
    vocal_b = to_float(b.get("vocal_mean") or 0.5)
    vocal_term = 1.0 - min(max(vocal_b, 0.0), 1.0)  # lower vocals → closer to 1

    # Simple weighted sum
    score = 2.0*bpm_term + 1.5*harm_term + 0.5*has_intro_end + 0.5*vocal_term
    return float(score)

# -------------------------
# Hotcues (IN + OUT, downbeat + vocal aware)
# -------------------------
HEADER_CUES = ["track","cue_time_s","label","hotcue","bar","beat","direction","comment"]

def beat_bar(t: float, bpm: float) -> Tuple[Optional[int], Optional[int]]:
    if bpm <= 0: return (None, None)
    beat = 60.0 / bpm
    bar  = 4 * beat
    return int(round(t/beat))+1, int(round(t/bar))+1

def time_minus_beats(t: float, bpm: float, beats: int) -> float:
    if bpm <= 0: return t
    return t - beats * (60.0 / bpm)

def pick_bar_after(t: float, bars: List[float], alt_bars: Optional[List[float]] = None) -> float:
    for b in (bars or []):
        if to_float(b) >= t - 1e-6:
            return to_float(b)
    if alt_bars:
        for b in alt_bars:
            if to_float(b) >= t - 1e-6:
                return to_float(b)
    return t

def pick_bar_before(t: float, bars: List[float], alt_bars: Optional[List[float]] = None) -> float:
    last = None
    for b in (bars or []):
        fb = to_float(b)
        if fb <= t + 1e-6: last = fb
        else: break
    if last is not None: return last
    if alt_bars:
        for b in alt_bars:
            fb = to_float(b)
            if fb <= t + 1e-6: last = fb
            else: break
        if last is not None: return last
    return t

def prefer_low_vocal(t: float, low_vocal_bars: List[float], bars_fallback: List[float]) -> float:
    if not low_vocal_bars:
        return pick_bar_after(t, bars_fallback)
    best = None
    best_abs = 9e9
    for b in low_vocal_bars:
        delta = abs(to_float(b) - t)
        if delta < best_abs and delta <= 8.0:
            best = to_float(b); best_abs = delta
    return best if best is not None else pick_bar_after(t, bars_fallback)

def build_dual_hotcues(meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    title = meta.get("title") or "track"
    bpm   = to_float(meta.get("bpm") or 0.0)

    bars_down   = [to_float(x) for x in (meta.get("downbeat_bar_starts_s") or [])]
    bars_legacy = [to_float(x) for x in (meta.get("bar_starts_s") or [])]
    phrases     = [to_float(x) for x in (meta.get("phrase_boundaries_s") or [])]
    low_vocal   = [to_float(x) for x in (meta.get("low_vocal_bar_starts_s") or [])]

    intro_end   = to_float(meta.get("intro_end_s") or (phrases[1] if len(phrases)>1 else (bars_down[1] if len(bars_down)>1 else 0.0)))
    drops       = [to_float(x) for x in (meta.get("drops_s") or [])]
    outro_start = to_float(meta.get("outro_start_s") or (phrases[-2] if len(phrases)>2 else (bars_down[-2] if len(bars_down)>2 else (bars_down[-1] if bars_down else 0.0))))

    rows: List[Dict[str, Any]] = []

    # IN: A/B/C/D
    A_t = bars_down[0] if bars_down else (bars_legacy[0] if bars_legacy else 0.0)
    A_b, A_bar = beat_bar(A_t, bpm)
    rows.append({"track": title,"cue_time_s": r2(A_t),"label":"INTRO START","hotcue":"A","bar":A_bar,"beat":A_b,"direction":"IN","comment":"Bar 1 Beat 1"})

    B_t = prefer_low_vocal(intro_end, low_vocal, bars_down or bars_legacy)
    B_b, B_bar = beat_bar(B_t, bpm)
    rows.append({"track": title,"cue_time_s": r2(B_t),"label":"GROOVE START","hotcue":"B","bar":B_bar,"beat":B_b,"direction":"IN","comment":"First bar after intro (low vocals)"})

    if drops:
        C_target = time_minus_beats(drops[0], bpm, 32)
        C_t = pick_bar_before(C_target, bars_down, bars_legacy)
        C_t = prefer_low_vocal(C_t, low_vocal, bars_down or bars_legacy)
        C_comment = "8 bars before first drop"
    else:
        if phrases:
            track_end = max((phrases[-1] if phrases else 0.0), (bars_down[-1] if bars_down else 0.0))
            idx = min(range(len(phrases)), key=lambda i: abs(phrases[i] - 0.30 * track_end))
            C_t = prefer_low_vocal(phrases[idx], low_vocal, bars_down or bars_legacy)
        else:
            C_t = B_t
        C_comment = "Early build (fallback)"
    C_b, C_bar = beat_bar(C_t, bpm)
    rows.append({"track": title,"cue_time_s": r2(C_t),"label":"PRE-DROP / BUILD","hotcue":"C","bar":C_bar,"beat":C_b,"direction":"IN","comment":C_comment})

    if bpm > 0:
        D_prefer = B_t + (8*4)*(60.0/bpm)
    else:
        D_prefer = B_t + 15.0
    D_t = prefer_low_vocal(D_prefer, low_vocal, bars_down or bars_legacy)
    D_b, D_bar = beat_bar(D_t, bpm)
    rows.append({"track": title,"cue_time_s": r2(D_t),"label":"LOOP DRUMS","hotcue":"D","bar":D_bar,"beat":D_b,"direction":"IN","comment":"Clean loop phrase"})

    # OUT: X/Y/Z
    if bpm > 0:
        X_target = time_minus_beats(outro_start, bpm, 64)
        Y_target = time_minus_beats(outro_start, bpm, 32)
    else:
        X_target = max(outro_start - 30.0, 0.0)
        Y_target = max(outro_start - 15.0, 0.0)
    X_t = pick_bar_before(X_target, bars_down, bars_legacy)
    Y_t = pick_bar_before(Y_target, bars_down, bars_legacy)
    Z_t = pick_bar_after(outro_start, bars_down, bars_legacy)

    X_b, X_bar = beat_bar(X_t, bpm)
    Y_b, Y_bar = beat_bar(Y_t, bpm)
    Z_b, Z_bar = beat_bar(Z_t, bpm)

    rows.append({"track": title,"cue_time_s": r2(X_t),"label":"OUT START","hotcue":"X","bar":X_bar,"beat":X_b,"direction":"OUT","comment":"Start 32–64 beat blend"})
    rows.append({"track": title,"cue_time_s": r2(Y_t),"label":"BASS SWAP","hotcue":"Y","bar":Y_bar,"beat":Y_b,"direction":"OUT","comment":"Swap lows"})
    rows.append({"track": title,"cue_time_s": r2(Z_t),"label":"OUT END","hotcue":"Z","bar":Z_bar,"beat":Z_b,"direction":"OUT","comment":"Close fader / phrase end"})

    return rows

def write_csv(path: Path, header: List[str], rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows: w.writerow(r)

# -------------------------
# Build a simple set order (greedy)
# -------------------------
def build_set_order(metas: List[Dict[str, Any]],
                    bpm_tol: float, harm_max: int) -> List[int]:
    """
    Greedy: pick a start track (lowest average vocal & has intro), then keep adding
    the best compatible next track not yet used.
    Returns a list of indices into metas.
    """
    if not metas: return []

    # choose start: low vocal, has intro_end, moderate BPM (~median)
    bpm_vals = [to_float(m.get("bpm") or 0.0) for m in metas if to_float(m.get("bpm") or 0.0) > 0]
    bpm_median = float(np.median(bpm_vals)) if bpm_vals else 120.0

    best_idx, best_score = 0, -1e9
    for i, m in enumerate(metas):
        has_intro = 1.0 if m.get("intro_end_s") is not None else 0.0
        vocal     = 1.0 - min(max(to_float(m.get("vocal_mean") or 0.5), 0.0), 1.0)
        bpm_pen   = -abs(to_float(m.get("bpm") or bpm_median) - bpm_median) / 10.0
        s = 1.0*has_intro + 1.0*vocal + 1.0*bpm_pen
        if s > best_score:
            best_idx, best_score = i, s

    order = [best_idx]
    used = {best_idx}

    while len(order) < len(metas):
        last = metas[order[-1]]
        best_next = None
        best_s = -1e9
        for j, cand in enumerate(metas):
            if j in used: continue
            s = compat_score(last, cand, bpm_tol=bpm_tol, harm_max=harm_max)
            if s > best_s:
                best_s, best_next = s, j
        if best_next is None:
            # append any remaining
            for j in range(len(metas)):
                if j not in used:
                    best_next = j; break
        order.append(best_next); used.add(best_next)

    return order

# -------------------------
# Mix plan builder
# -------------------------
HEADER_PLAN = [
    "order","from_track","to_track",
    "from_bpm","to_bpm","camelot_from","camelot_to",
    "mix_in_time_s (to)","mix_in_bar (to)","mix_out_time_s (from)","mix_out_bar (from)",
    "notes","compat_score"
]

def bar_index_at_time(t: float, bars: List[float]) -> Optional[int]:
    if not bars: return None
    idx = 0
    for i, b in enumerate(bars):
        if to_float(b) <= t + 1e-6:
            idx = i + 1
        else:
            break
    return idx

def build_mix_plan(metas: List[Dict[str, Any]], order: List[int],
                   bpm_tol: float, harm_max: int) -> List[Dict[str, Any]]:
    plan = []
    for k in range(len(order)-1):
        a = metas[order[k]]   # current
        b = metas[order[k+1]] # next/incoming

        score = compat_score(a, b, bpm_tol=bpm_tol, harm_max=harm_max)

        # Choose OUT from A: ~32 bars before outro_start (X or Y)
        bpm_a = to_float(a.get("bpm") or 0.0)
        bars_a = [to_float(x) for x in (a.get("downbeat_bar_starts_s") or a.get("bar_starts_s") or [])]
        outro_a = to_float(a.get("outro_start_s") or (bars_a[-2] if len(bars_a)>2 else (bars_a[-1] if bars_a else 0.0)))
        if bpm_a > 0:
            out_target = max(outro_a - (32* (60.0/bpm_a)), 0.0)  # ~8 bars before outro
        else:
            out_target = max(outro_a - 30.0, 0.0)
        out_time = pick_bar_before(out_target, bars_a, bars_a)
        out_bar_idx = bar_index_at_time(out_time, bars_a)

        # Choose IN for B: groove start (post-intro, low-vocal aware)
        bpm_b = to_float(b.get("bpm") or 0.0)
        bars_b_down = [to_float(x) for x in (b.get("downbeat_bar_starts_s") or [])]
        bars_b_legacy = [to_float(x) for x in (b.get("bar_starts_s") or [])]
        low_b = [to_float(x) for x in (b.get("low_vocal_bar_starts_s") or [])]
        intro_b = to_float(b.get("intro_end_s") or (bars_b_down[1] if len(bars_b_down)>1 else (bars_b_legacy[1] if len(bars_b_legacy)>1 else 0.0)))
        in_time = prefer_low_vocal(intro_b, low_b, bars_b_down or bars_b_legacy)
        in_bar_idx = bar_index_at_time(in_time, bars_b_down or bars_b_legacy)

        plan.append({
            "order": k+1,
            "from_track": a.get("title"),
            "to_track": b.get("title"),
            "from_bpm": r2(bpm_a),
            "to_bpm": r2(bpm_b),
            "camelot_from": a.get("camelot"),
            "camelot_to": b.get("camelot"),
            "mix_in_time_s (to)": r2(in_time),
            "mix_in_bar (to)": in_bar_idx,
            "mix_out_time_s (from)": r2(out_time),
            "mix_out_bar (from)": out_bar_idx,
            "notes": "Align phrases; swap bass ~Y; fade by Z",
            "compat_score": r2(score),
        })
    return plan

# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Build a set order + mix plan and export dual hotcues from .djmeta.json")
    ap.add_argument("inputs", nargs="+", help="Folder(s) or .djmeta.json file(s)")
    ap.add_argument("--plan-out", default="mix_plan.csv", help="Path to write the mix plan CSV")
    ap.add_argument("--hotcues-out", default="hotcues_dual", help="Directory to write per-track dual hotcues CSVs")
    ap.add_argument("--combined-cues", help="Optional combined cues CSV path")
    ap.add_argument("--bpm-tolerance", type=float, default=2.5, help="Max BPM difference (soft) for scoring")
    ap.add_argument("--max-harmonic-distance", type=int, default=1, help="Max Camelot distance for good scores (0..1 ideal)")
    return ap.parse_args()

def main():
    args = parse_args()
    meta_files = collect_metas(args.inputs)
    if not meta_files:
        print("No .djmeta.json files found."); return

    metas = [load_meta(p) for p in meta_files]

    # 1) Build set order
    order = build_set_order(metas, bpm_tol=args.bpm_tolerance, harm_max=args.max_harmonic_distance)

    # 2) Mix plan
    plan_rows = build_mix_plan(metas, order, bpm_tol=args.bpm_tolerance, harm_max=args.max_harmonic_distance)
    write_csv(Path(args.plan_out), HEADER_PLAN, plan_rows)
    print(f"Wrote mix plan: {Path(args.plan_out).resolve()}")

    # 3) Dual hotcues per track + combined
    outdir = Path(args.hotcues_out)
    combined: List[Dict[str, Any]] = []
    for idx in order:
        meta = metas[idx]
        cues = build_dual_hotcues(meta)
        name = (meta.get("title") or "track").strip()
        write_csv(outdir / f"{name}_hotcues_dual.csv", HEADER_CUES, cues)
        combined.extend(cues)
    if args.combined_cues:
        write_csv(Path(args.combined_cues), HEADER_CUES, combined)
        print(f"Wrote combined cues: {Path(args.combined_cues).resolve()}")

    # 4) Print friendly summary
    print("\nSet order:")
    for i, idx in enumerate(order, 1):
        print(f"  {i:>2}. {metas[idx].get('title')}")

if __name__ == "__main__":
    main()
