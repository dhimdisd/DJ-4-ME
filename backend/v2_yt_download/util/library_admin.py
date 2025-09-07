#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, sys, shutil
from pathlib import Path
from typing import List, Dict, Optional

# ---------- Paths ----------
def resolve_paths(outdir: Path):
    audio_dir = outdir.resolve()
    downloads_dir = audio_dir.parent
    ledger_path = downloads_dir / "_download_ledger.csv"
    archive_path = downloads_dir / "_download_archive.txt"
    playlists_root = downloads_dir / "dj_playlists"
    return audio_dir, downloads_dir, ledger_path, archive_path, playlists_root

# ---------- Ledger helpers ----------
def read_ledger(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return list(r)

def write_ledger(path: Path, rows: List[Dict[str, str]]):
    if not rows:
        # If empty, keep a minimal header
        fieldnames = ["id","title","song_url","filepath","codec","downloaded_at","status","playlist_url","playlist_title"]
        with open(path, "w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        return
    fieldnames = rows[0].keys()
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

def remove_from_archive(archive_path: Path, ids_to_remove: set[str], dry: bool):
    if not archive_path.exists() or not ids_to_remove:
        return
    with open(archive_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    kept = []
    removed = []
    for line in lines:
        if any(vid in line for vid in ids_to_remove):
            removed.append(line)
        else:
            kept.append(line)
    if dry:
        if removed:
            print(f"[DRY] Would remove {len(removed)} entries from archive: {archive_path}")
        return
    with open(archive_path, "w", encoding="utf-8") as f:
        f.write("\n".join(kept) + ("\n" if kept else ""))
    if removed:
        print(f"[OK] Removed {len(removed)} entries from archive: {archive_path}")

# ---------- Utilities ----------
def collect_broken_symlinks(root: Path) -> List[Path]:
    out = []
    if not root.exists():
        return out
    for p in root.rglob("*"):
        if p.is_symlink() and not p.exists():
            out.append(p)
    return out

def unlink_path(p: Path, dry: bool):
    if not p.exists() and not p.is_symlink():
        return
    if dry:
        print(f"[DRY] Unlink {p}")
        return
    p.unlink(missing_ok=True)

def rm_file(p: Path, dry: bool):
    if dry:
        print(f"[DRY] Delete file {p}")
        return
    if p.exists():
        p.unlink()
        print(f"[OK] Deleted {p}")

def rm_tree(p: Path, dry: bool):
    if not p.exists():
        return
    if dry:
        print(f"[DRY] Remove folder {p}")
        return
    shutil.rmtree(p)
    print(f"[OK] Removed folder {p}")

# ---------- Commands ----------
def cmd_delete_track(args):
    audio_dir, downloads_dir, ledger_path, archive_path, playlists_root = resolve_paths(Path(args.outdir))
    ledger = read_ledger(ledger_path)

    # Determine target by TITLE or PATH
    targets: List[Path] = []
    ids_to_remove: set[str] = set()

    if args.arg:
        # If arg looks like a file path
        candidate = Path(args.arg)
        if candidate.suffix:
            # Path mode
            targets = [candidate if candidate.is_absolute() else (Path.cwd() / candidate)]
        else:
            # Title mode — match ledger title
            title = args.arg
            for row in ledger:
                if (row.get("title") or "") == title:
                    fp = Path(row.get("filepath") or "")
                    if fp:
                        targets.append(fp)
                        if row.get("id"):
                            ids_to_remove.add(row["id"])
    else:
        print("delete-track requires a TITLE or PATH")
        sys.exit(2)

    # Delete / unlink
    removed_files = set()
    for t in targets:
        if t.exists():
            rm_file(t, args.dry_run)
            removed_files.add(str(t))

    # Clean ledger rows
    kept = []
    removed_rows = 0
    for row in ledger:
        fp = row.get("filepath") or ""
        if fp in removed_files or (args.arg and (row.get("title") == args.arg)):
            removed_rows += 1
            if row.get("id"):
                ids_to_remove.add(row["id"])
            continue
        kept.append(row)

    if args.dry_run:
        print(f"[DRY] Would remove {removed_rows} ledger row(s) at {ledger_path}")
    else:
        write_ledger(ledger_path, kept)
        if removed_rows:
            print(f"[OK] Removed {removed_rows} ledger row(s) at {ledger_path}")

    remove_from_archive(archive_path, ids_to_remove, args.dry_run)

def cmd_delete_playlist(args):
    audio_dir, downloads_dir, ledger_path, archive_path, playlists_root = resolve_paths(Path(args.outdir))
    playlist_title = args.title
    playlist_dir = playlists_root / playlist_title

    if not playlist_dir.exists():
        print(f"[WARN] Playlist folder not found: {playlist_dir}")
        return

    # Gather targets from symlinks in this playlist folder
    target_files: List[Path] = []
    for p in sorted(playlist_dir.rglob("*")):
        if p.is_symlink():
            try:
                target = p.resolve()
            except Exception:
                target = None
            if target:
                target_files.append(target)

    # Map to ledger rows
    ledger = read_ledger(ledger_path)
    ids_to_remove: set[str] = set()
    file_set = set(str(p) for p in target_files)

    # Optionally delete audio
    if not args.keep_audio:
        for f in target_files:
            rm_file(f, args.dry_run)

    # Remove symlink folder
    rm_tree(playlist_dir, args.dry_run)

    # Clean ledger rows whose filepath matches any of these
    kept = []
    removed_rows = 0
    for row in ledger:
        fp = row.get("filepath") or ""
        if fp in file_set:
            removed_rows += 1
            if row.get("id"):
                ids_to_remove.add(row["id"])
            continue
        kept.append(row)

    if args.dry_run:
        print(f"[DRY] Would remove {removed_rows} ledger row(s) belonging to playlist '{playlist_title}'")
    else:
        write_ledger(ledger_path, kept)
        if removed_rows:
            print(f"[OK] Removed {removed_rows} ledger row(s) for playlist '{playlist_title}'")

    # Remove archive entries for these IDs
    remove_from_archive(archive_path, ids_to_remove, args.dry_run)

def cmd_clean_orphans(args):
    audio_dir, downloads_dir, ledger_path, archive_path, playlists_root = resolve_paths(Path(args.outdir))
    # 1) Broken playlist symlinks
    broken = collect_broken_symlinks(playlists_root)
    for b in broken:
        unlink_path(b, args.dry_run)
    if broken:
        print(f"[OK] {'Would remove' if args.dry_run else 'Removed'} {len(broken)} broken symlink(s) under {playlists_root}")

    # 2) Ledger rows whose file is missing
    ledger = read_ledger(ledger_path)
    kept = []
    removed_rows = 0
    ids_to_remove: set[str] = set()
    for row in ledger:
        fp = Path(row.get("filepath") or "")
        if fp and not fp.exists():
            removed_rows += 1
            if row.get("id"):
                ids_to_remove.add(row["id"])
        else:
            kept.append(row)
    if args.dry_run:
        print(f"[DRY] Would remove {removed_rows} ledger row(s) for missing files")
    else:
        write_ledger(ledger_path, kept)
        if removed_rows:
            print(f"[OK] Removed {removed_rows} ledger row(s) for missing files")

    remove_from_archive(archive_path, ids_to_remove, args.dry_run)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Library admin: delete tracks/playlists and clean orphans.")
    ap.add_argument("--outdir", required=True, help="Path to audio folder (e.g., v2_yt_download/downloads/audio)")
    ap.add_argument("--dry-run", action="store_true", help="Preview actions without making changes")

    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("delete-track", help="Delete a single track by exact TITLE or by PATH")
    sp.add_argument("arg", help="Either exact track TITLE (matches ledger) or a file PATH")
    sp.set_defaults(func=cmd_delete_track)

    sp = sub.add_parser("delete-playlist", help="Delete a playlist by its folder/title name")
    sp.add_argument("title", help="Playlist title (folder name under downloads/dj_playlists)")
    sp.add_argument("--keep-audio", action="store_true", help="Do NOT delete audio files; only remove symlinks, ledger and archive entries")
    sp.set_defaults(func=cmd_delete_playlist)

    sp = sub.add_parser("clean-orphans", help="Remove broken symlinks and ledger rows pointing to missing files")
    sp.set_defaults(func=cmd_clean_orphans)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()