#!/usr/bin/env python3
# library_admin.py
from __future__ import annotations
import argparse, csv, os, shutil, sys
from pathlib import Path
from typing import List, Dict, Optional

LEDGER_NAME = "_download_ledger.csv"
ARCHIVE_NAME = "_download_archive.txt"

def ledger_path_for_outdir(outdir: Path) -> Path:
    # mirrors downloader.py logic: ledger lives under downloads/
    if outdir.name == "audio":
        return outdir.parent / LEDGER_NAME
    if outdir.parent.name == "audio":
        return outdir.parent.parent / LEDGER_NAME
    return outdir / LEDGER_NAME

def archive_path_for_outdir(outdir: Path) -> Path:
    if outdir.name == "audio":
        return outdir / ARCHIVE_NAME  # archive was stored next to audio in our build
    if outdir.parent.name == "audio":
        return outdir.parent / ARCHIVE_NAME
    return outdir / ARCHIVE_NAME

def read_ledger(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r)

def write_ledger(path: Path, rows: List[Dict[str, str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # write an empty file with header
        fieldnames = ["id","title","song_url","filepath","codec","downloaded_at","status","playlist_url","playlist_title"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

def remove_archive_ids(archive_path: Path, ids_to_remove: List[str]):
    if not archive_path.exists() or not ids_to_remove:
        return
    try:
        with open(archive_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        ids = set(ids_to_remove)
        def keep(line: str) -> bool:
            # typical line: "youtube y6iMvNkAQt4"
            parts = line.strip().split()
            return not (len(parts) >= 2 and parts[-1] in ids)
        new_lines = [ln for ln in lines if keep(ln)]
        with open(archive_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
    except Exception as e:
        print(f"[WARN] Could not prune archive {archive_path}: {e}")

def safe_unlink(p: Path):
    try:
        if p.exists() or p.is_symlink():
            p.unlink()
    except Exception as e:
        print(f"[WARN] Failed to remove {p}: {e}")

def find_all_symlinks_for_file(playlists_root: Path, filename: str) -> List[Path]:
    out: List[Path] = []
    if not playlists_root.exists():
        return out
    for sub in playlists_root.rglob(filename):
        # only count symlinks, leave regular files alone
        if sub.is_symlink():
            out.append(sub)
    return out

def cmd_delete_track(outdir: Path, title_or_path: str, dry_run: bool):
    """
    Delete one track by title *or* by exact audio filepath.
    Removes:
      - audio file in outdir
      - symlinks in downloads/dj_playlists/**/
      - ledger rows with matching filepath OR matching title
      - archive entry for its id (if present)
    """
    ledger_path = ledger_path_for_outdir(outdir)
    archive_path = archive_path_for_outdir(outdir)
    downloads_root = outdir.parent if outdir.name == "audio" else outdir
    playlists_root = downloads_root / "dj_playlists"

    rows = read_ledger(ledger_path)
    if not rows:
        print("[INFO] Ledger is empty; nothing to delete.")
    # resolve target(s)
    # prefer filepath match
    candidates = []
    title_key = title_or_path.strip()
    p = Path(title_or_path)
    if p.suffix:  # looks like a file path
        # normalize to absolute
        target = p if p.is_absolute() else (Path.cwd() / p)
        candidates = [r for r in rows if Path(r.get("filepath","")).resolve() == target.resolve()]
    if not candidates:
        # fallback: match by title against filename stem or ledger title
        candidates = [r for r in rows if (r.get("title","").strip() == title_key) or Path(r.get("filepath","")).stem == Path(title_key).stem]

    if not candidates:
        print(f"[WARN] No matching ledger rows for {title_or_path}. Nothing done.")
        return

    # Collect removals
    filepaths = []
    ids = []
    for r in candidates:
        fp = r.get("filepath","").strip()
        if fp:
            filepaths.append(Path(fp))
        vid = r.get("id","").strip()
        if vid:
            ids.append(vid)

    print(f"[INFO] Found {len(candidates)} ledger row(s) to remove.")
    for fp in filepaths:
        print(f"  - audio: {fp}")
        # symlinks
        syms = find_all_symlinks_for_file(playlists_root, fp.name)
        for s in syms:
            print(f"  - symlink: {s}")

    if dry_run:
        print("[DRY-RUN] No changes written.")
        return

    # Remove files + symlinks
    for fp in filepaths:
        safe_unlink(fp)
        for s in find_all_symlinks_for_file(playlists_root, fp.name):
            safe_unlink(s)

    # Filter ledger
    keep = []
    for r in rows:
        if r in candidates:
            continue
        keep.append(r)
    write_ledger(ledger_path, keep)

    # Prune archive
    remove_archive_ids(archive_path, ids)
    print("[DONE] Track deletion complete.")

def cmd_delete_playlist(outdir: Path, playlist_title: str, dry_run: bool):
    """
    Delete an entire playlist:
      - remove all audio files referenced in ledger with matching playlist_title
      - remove symlinks folder downloads/dj_playlists/<playlist_title>
      - prune ledger rows and archive ids
    """
    ledger_path = ledger_path_for_outdir(outdir)
    archive_path = archive_path_for_outdir(outdir)
    downloads_root = outdir.parent if outdir.name == "audio" else outdir
    playlists_root = downloads_root / "dj_playlists"
    pl_dir = playlists_root / playlist_title

    rows = read_ledger(ledger_path)
    if not rows:
        print("[INFO] Ledger is empty; nothing to delete.")
        return
    targets = [r for r in rows if (r.get("playlist_title","").strip() == playlist_title)]
    if not targets:
        print(f"[WARN] No ledger rows for playlist '{playlist_title}'.")
        # still remove symlink folder if it exists
        if pl_dir.exists():
            print(f"[INFO] Removing playlist folder: {pl_dir}")
            if not dry_run:
                shutil.rmtree(pl_dir, ignore_errors=True)
        return

    ids = []
    files = []
    for r in targets:
        vid = r.get("id","").strip()
        if vid:
            ids.append(vid)
        fp = r.get("filepath","").strip()
        if fp:
            files.append(Path(fp))

    print(f"[INFO] Playlist '{playlist_title}': {len(files)} file(s), {len(ids)} id(s).")
    for f in files:
        print(f"  - audio: {f}")
    print(f"  - symlink folder: {pl_dir}")

    if dry_run:
        print("[DRY-RUN] No changes written.")
        return

    # Remove audio files
    for f in files:
        safe_unlink(f)

    # Remove playlist symlink dir
    if pl_dir.exists():
        shutil.rmtree(pl_dir, ignore_errors=True)

    # Re-write ledger without those rows
    keep = [r for r in rows if r not in targets]
    write_ledger(ledger_path, keep)

    # Prune archive
    remove_archive_ids(archive_path, ids)
    print("[DONE] Playlist deletion complete.")

def cmd_clean_orphans(outdir: Path, dry_run: bool):
    """
    Remove broken symlinks under dj_playlists and ledger rows that point to missing files.
    """
    ledger_path = ledger_path_for_outdir(outdir)
    downloads_root = outdir.parent if outdir.name == "audio" else outdir
    playlists_root = downloads_root / "dj_playlists"

    rows = read_ledger(ledger_path)
    keep = []
    removed_rows = 0
    for r in rows:
        fp = Path(r.get("filepath",""))
        if not fp.exists():
            print(f"[CLEAN] Missing file; drop ledger: {fp}")
            removed_rows += 1
        else:
            keep.append(r)

    if not dry_run and removed_rows:
        write_ledger(ledger_path, keep)

    # Clean broken symlinks
    if playlists_root.exists():
        broken = []
        for s in playlists_root.rglob("*"):
            if s.is_symlink():
                target = s.resolve(strict=False)
                # if target path doesn't exist on disk
                try:
                    if not Path(os.readlink(s)).exists() and not target.exists():
                        broken.append(s)
                except OSError:
                    broken.append(s)
        for b in broken:
            print(f"[CLEAN] Broken symlink: {b}")
            if not dry_run:
                safe_unlink(b)

    print("[DONE] Clean orphans complete.")

def main():
    ap = argparse.ArgumentParser(description="Manage downloaded audio + ledger + archive")
    ap.add_argument("--outdir", required=True, help="Audio output directory (e.g., v2_yt_download/downloads/audio)")
    ap.add_argument("--dry-run", action="store_true", help="Show what would be deleted, do not modify anything")

    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("delete-track", help="Delete a single track by title or by full file path")
    t.add_argument("title_or_path", help="Track title (exact) OR full/relative path to audio file")

    p = sub.add_parser("delete-playlist", help="Delete a whole playlist by its title")
    p.add_argument("playlist_title", help="Exact playlist title as stored in ledger")

    c = sub.add_parser("clean-orphans", help="Remove broken symlinks and ledger rows for missing files")

    args = ap.parse_args()
    outdir = Path(args.outdir)

    if args.cmd == "delete-track":
        cmd_delete_track(outdir, args.title_or_path, args.dry_run)
    elif args.cmd == "delete-playlist":
        cmd_delete_playlist(outdir, args.playlist_title, args.dry_run)
    elif args.cmd == "clean-orphans":
        cmd_clean_orphans(outdir, args.dry_run)
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
