#!/usr/bin/env python3
# export_vdj_cues.py
from __future__ import annotations
import argparse, json, shutil, time
from pathlib import Path
import xml.etree.ElementTree as ET

SUPPORTED_AUDIO_EXTS = [".wav", ".mp3"]
PREFIX = "[DJ4ME] "  # so we can safely update/remove only our cues

def load_meta(meta_path: Path) -> dict | None:
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read {meta_path}: {e}")
        return None

def find_audio_for_meta(meta_path: Path, audio_dir: Path) -> Path | None:
    base = meta_path.stem  # removes .djmeta, leaves original title
    # if file is like "Track.djmeta", stem is "Track"
    # If your meta files are named "Some Title.djmeta.json", .stem returns "Some Title.djmeta"
    # so strip a trailing ".djmeta" if present:
    if base.endswith(".djmeta"):
        base = base[:-7]
    # Find matching audio file
    for ext in SUPPORTED_AUDIO_EXTS:
        cand = audio_dir / f"{base}{ext}"
        if cand.exists():
            return cand.resolve()
    # fallback: case-insensitive search
    lc = base.lower()
    for p in audio_dir.glob("*"):
        if p.suffix.lower() in SUPPORTED_AUDIO_EXTS and p.stem.lower() == lc:
            return p.resolve()
    return None

def cues_from_meta(meta: dict) -> list[tuple[str, float]]:
    """Return list of (label, time_s) hotcues (seconds)."""
    out = []
    # Prefer downbeat/bar starts for intro if present
    intro_t = meta.get("downbeat_time_s") \
              or (meta.get("bar_starts_s", [0.0])[0] if meta.get("bar_starts_s") else 0.0)
    out.append((f"{PREFIX}Intro Start", float(intro_t)))

    if meta.get("mix_in_time_s") is not None:
        out.append((f"{PREFIX}Mix In", float(meta["mix_in_time_s"])))
    if meta.get("outro_start_s") is not None:
        out.append((f"{PREFIX}Outro Start", float(meta["outro_start_s"])))
    if meta.get("mix_out_time_s") is not None:
        out.append((f"{PREFIX}Mix Out", float(meta["mix_out_time_s"])))

    drops = meta.get("drops_s") or []
    if drops:
        out.append((f"{PREFIX}Drop 1", float(drops[0])))
    bdowns = meta.get("breakdowns_s") or []
    if bdowns:
        out.append((f"{PREFIX}Breakdown 1", float(bdowns[0])))

    # Keep stable ordering, remove dupes by (label,time)
    seen = set()
    dedup = []
    for label, t in out:
        key = (label, round(t, 3))
        if key not in seen and t is not None:
            dedup.append((label, t))
            seen.add(key)
    return dedup

def phrase_cues(meta: dict, max_phrases: int) -> list[tuple[str, float]]:
    """Optional: add first N phrase start cues."""
    if max_phrases <= 0:
        return []
    times = meta.get("phrase_boundaries_s") or []
    out = []
    for i, t in enumerate(times[:max_phrases], start=1):
        try:
            out.append((f"{PREFIX}Phrase {i}", float(t)))
        except Exception:
            pass
    return out

def default_database_path() -> Path:
    # macOS default
    mac = Path.home() / "Library/Application Support/VirtualDJ/database.xml"
    if mac.exists():
        return mac
    # Windows default
    win = Path.home() / "Documents/VirtualDJ/database.xml"
    return win

def backup_database(db_path: Path) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    backup_path = db_path.with_name(db_path.stem + f".bak.{ts}" + db_path.suffix)
    shutil.copy2(db_path, backup_path)
    print(f"[INFO] Backed up database to {backup_path}")
    return backup_path

def get_or_create_song_node(root: ET.Element, file_path: Path) -> ET.Element:
    # VirtualDJ stores absolute path in FilePath attribute
    fp = str(file_path)
    for song in root.findall(".//Song"):
        if song.get("FilePath") == fp:
            return song
    # Create a new Song node
    song = ET.SubElement(root, "Song")
    song.set("FilePath", fp)
    return song

def clear_our_cues(song: ET.Element):
    pois = list(song.findall("./Poi"))
    for poi in pois:
        name = poi.get("Name") or ""
        if name.startswith(PREFIX):
            song.remove(poi)

def add_cue(song: ET.Element, label: str, time_s: float, num: int):
    """Add a VirtualDJ cue. Pos uses seconds (float)."""
    poi = ET.SubElement(song, "Poi")
    poi.set("Name", label)
    poi.set("Type", "cue")
    poi.set("Pos", f"{float(time_s):.3f}")
    poi.set("Num", str(num))
    # Color is optional; leaving it out lets VDJ color automatically.

def main():
    ap = argparse.ArgumentParser(description="Export DJ analyzer hotcues to VirtualDJ database.xml")
    ap.add_argument("--metadata-dir", required=True, help="Directory containing *.djmeta.json")
    ap.add_argument("--audio-dir", required=True, help="Directory containing audio files (wav/mp3)")
    ap.add_argument("--database", default=str(default_database_path()), help="Path to VirtualDJ database.xml")
    ap.add_argument("--phrase-cues", type=int, default=0, help="Also add first N phrase start cues (default 0)")
    args = ap.parse_args()

    metadata_dir = Path(args.metadata_dir)
    audio_dir = Path(args.audio_dir)
    db_path = Path(args.database)

    if not db_path.exists():
        print(f"[ERROR] VirtualDJ database not found at: {db_path}")
        print("       Open VirtualDJ once so it creates the file, or pass --database to point to it.")
        return

    # Load/parse database and backup
    backup_database(db_path)
    tree = ET.parse(db_path)
    root = tree.getroot()

    meta_files = sorted(metadata_dir.glob("*.djmeta.json"))
    if not meta_files:
        print(f"[WARN] No *.djmeta.json in {metadata_dir}")
        return

    updated = 0
    for meta_path in meta_files:
        meta = load_meta(meta_path)
        if not meta:
            continue
        audio_path = find_audio_for_meta(meta_path, audio_dir)
        if not audio_path:
            print(f"[WARN] No audio match for {meta_path.name} in {audio_dir}")
            continue

        # Build cues
        cues = cues_from_meta(meta)
        cues += phrase_cues(meta, args.phrase_cues)
        if not cues:
            print(f"[INFO] No cues to write for {audio_path.name}")
            continue

        song = get_or_create_song_node(root, audio_path)
        clear_our_cues(song)

        # Assign cue numbers sequentially after existing user cues
        existing_nums = {int(p.get("Num")) for p in song.findall("./Poi") if p.get("Num", "").isdigit()}
        next_num = (max(existing_nums) + 1) if existing_nums else 1
        for label, t in cues:
            add_cue(song, label, t, next_num)
            next_num += 1

        print(f"[OK] Wrote {len(cues)} cues for {audio_path.name}")
        updated += 1

    if updated:
        tree.write(db_path, encoding="utf-8", xml_declaration=True)
        print(f"[DONE] Updated {updated} track(s) in {db_path}")
    else:
        print("[INFO] Nothing to update.")

if __name__ == "__main__":
    main()