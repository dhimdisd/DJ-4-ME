#!/usr/bin/env python3
# library_admin.py
from __future__ import annotations
import argparse, csv, os, shutil, datetime
from pathlib import Path
from typing import Dict, List, Optional

LEDGER_NAME = "_download_ledger.csv"
ARCHIVE_NAME = "_download_archive.txt"
PLAYLISTS_DIRNAME = "dj_playlists"

# ----------------------------- utils ---------------------------------
def _safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _backup_file(path: Path):
    """Create a timestamped backup of the given file in the same directory."""
    if not path.exists():
        return None
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"{path.stem}.backup.{ts}{path.suffix}")
    shutil.copy2(path, backup)
    print(f"[BACKUP] {path.name} → {backup.name}")
    return backup

def _read_ledger(ledger_path: Path) -> list[dict]:
    rows: list[dict] = []
    if not ledger_path.exists():
        return rows
    with open(ledger_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def _write_ledger(ledger_path: Path, rows: list[dict]):
    if not rows:
        # Write header only so future appends still work
        fieldnames = ["id","title","song_url","filepath","codec","downloaded_at","status","playlist_url","playlist_title","playlist_id"]
        with open(ledger_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
        return
    # union of all keys to preserve any extra fields
    fieldnames: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(ledger_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def _read_archive(archive_path: Path) -> list[str]:
    if not archive_path.exists():
        return []
    with open(archive_path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]

def _write_archive(archive_path: Path, lines: list[str]):
    _safe_mkdir(archive_path.parent)
    with open(archive_path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")

def _ensure_symlink(link_path: Path, target: Path, dry: bool = False) -> bool:
    """Create or fix a symlink so that link_path -> relative(target). Returns True if changed."""
    _safe_mkdir(link_path.parent)
    rel = os.path.relpath(target, start=link_path.parent)

    if link_path.exists() or link_path.is_symlink():
        try:
            current = os.readlink(link_path)
            if current == rel:
                return False  # correct already
        except OSError:
            pass
        if dry:
            print(f"[DRY] replace link: {link_path} -> {rel}")
            return True
        link_path.unlink(missing_ok=True)

    if dry:
        print(f"[DRY] create link: {link_path} -> {rel}")
        return True

    link_path.symlink_to(rel)
    return True

def _playlist_dir_name(title: Optional[str], pid: Optional[str], default_singles: str) -> str:
    t = (title or "").strip()
    p = (pid or "").strip()
    if t and p:
        return f"{t} - {p}"
    if t:
        return t
    return default_singles

def _ledger_paths(base_audio_dir: Path):
    downloads_dir = base_audio_dir.parent
    playlists_root = downloads_dir / PLAYLISTS_DIRNAME
    ledger_path = downloads_dir / LEDGER_NAME
    archive_path = downloads_dir / ARCHIVE_NAME
    return downloads_dir, playlists_root, ledger_path, archive_path

def _matches_title(row: dict, title: str) -> bool:
    return (row.get("title") or "").strip() == title.strip()

def _row_id(row: dict) -> str:
    return (row.get("id") or "").strip()

def _resolve_fp(fp_str: str, downloads_dir: Path) -> Optional[Path]:
    """
    Resolve a ledger filepath string to an absolute path. Handles a few cases:
      - absolute paths: return as-is
      - 'downloads/audio/...' (common in your ledger): resolve under downloads_dir/audio
      - 'audio/...' or bare filename: resolve under downloads_dir/
    """
    if not fp_str:
        return None
    p = Path(fp_str)
    if p.is_absolute():
        return p

    parts = p.parts
    # If ledger stored "downloads/..." or "<downloads_dir.name>/...", drop that prefix
    if parts and (parts[0].lower() == "downloads" or parts[0] == downloads_dir.name):
        p = Path(*parts[1:]) if len(parts) > 1 else Path("")

    # If path now starts with "audio", keep it; otherwise treat as relative to downloads_dir
    if p.parts and p.parts[0].lower() == "audio":
        resolved = downloads_dir / p
    else:
        resolved = downloads_dir / p

    return resolved.resolve()

# ----------------------------- commands ---------------------------------

def cmd_delete_track(args):
    """
    Delete a single track by exact TITLE or by file PATH.
    Updates: audio file, playlist symlinks, ledger, archive.
    """
    audio_dir = Path(args.outdir).resolve()
    downloads_dir, playlists_root, ledger_path, archive_path = _ledger_paths(audio_dir)

    rows = _read_ledger(ledger_path)
    if not rows:
        print(f"[WARN] No ledger at {ledger_path}")
    archive_lines = _read_archive(archive_path)

    to_delete_files: set[Path] = set()
    to_delete_ids: set[str] = set()

    if args.by == "title":
        title = args.target
        for r in rows:
            if _matches_title(r, title):
                fp = _resolve_fp((r.get("filepath") or "").strip(), downloads_dir)
                if fp:
                    to_delete_files.add(fp)
                rid = _row_id(r)
                if rid:
                    to_delete_ids.add(rid)
    else:  # by path
        p = _resolve_fp(args.target, downloads_dir)
        if p:
            to_delete_files.add(p)
        # find ledger rows referencing this path (normalize both sides)
        for r in rows:
            rfp = _resolve_fp((r.get("filepath") or "").strip(), downloads_dir)
            if rfp and p and rfp == p:
                rid = _row_id(r)
                if rid:
                    to_delete_ids.add(rid)

    # Remove files + symlinks
    removed_files = 0
    for f in to_delete_files:
        # remove symlinks pointing to f
        if playlists_root.exists():
            for pl in playlists_root.glob("*"):
                if pl.is_dir():
                    link = pl / f.name
                    if link.is_symlink():
                        if args.dry_run:
                            print(f"[DRY] unlink symlink: {link}")
                        else:
                            link.unlink(missing_ok=True)
        # remove audio file
        if f.exists():
            if args.dry_run:
                print(f"[DRY] delete file: {f}")
            else:
                try:
                    f.unlink()
                except Exception as e:
                    print(f"[WARN] failed to delete {f}: {e}")
                else:
                    removed_files += 1

    # Filter ledger rows
    new_rows = []
    for r in rows:
        keep = True
        if args.by == "title" and _matches_title(r, args.target):
            keep = False
        if args.by == "path":
            rfp = _resolve_fp((r.get("filepath") or "").strip(), downloads_dir)
            p   = _resolve_fp(args.target, downloads_dir)
            if rfp and p and rfp == p:
                keep = False
        if keep:
            new_rows.append(r)

    # Filter archive (entries begin with "youtube <id>" typically)
    def _archive_keep(line: str) -> bool:
        for rid in to_delete_ids:
            if rid and rid in line:
                return False
        return True

    new_archive = [ln for ln in archive_lines if _archive_keep(ln)]

    # Write back (unless dry-run)
    if args.dry_run:
        print(f"[DRY] would write ledger ({len(new_rows)} rows) and archive ({len(new_archive)} lines)")
    else:
        _backup_file(ledger_path); _backup_file(archive_path)
        _write_ledger(ledger_path, new_rows)
        _write_archive(archive_path, new_archive)

    print(f"[DONE] removed {removed_files} audio files; updated ledger & archive ({'dry-run' if args.dry_run else 'written'})")

def cmd_delete_playlist(args):
    """
    Delete an entire playlist by TITLE (the human title; folder may be 'Title - ID').
    - Removes playlist symlink folder
    - Removes audio files unless --keep-audio
    - Cleans ledger rows + archive entries for that playlist
    """
    audio_dir = Path(args.outdir).resolve()
    downloads_dir, playlists_root, ledger_path, archive_path = _ledger_paths(audio_dir)

    rows = _read_ledger(ledger_path)
    archive_lines = _read_archive(archive_path)

    title = args.title.strip()
    affected_rows = [r for r in rows if (r.get("playlist_title") or "").strip() == title]

    pid_counts: dict[str,int] = {}
    for r in affected_rows:
        pid = (r.get("playlist_id") or "").strip()
        if pid:
            pid_counts[pid] = pid_counts.get(pid, 0) + 1
    playlist_dirs = []
    if pid_counts:
        best_pid = max(pid_counts.items(), key=lambda kv: kv[1])[0]
        playlist_dirs.append(playlists_root / f"{title} - {best_pid}")
    playlist_dirs.append(playlists_root / title)  # fallback

    # Remove symlink folder(s)
    removed_links = 0
    for d in playlist_dirs:
        if d.exists() and d.is_dir():
            for item in d.iterdir():
                if item.is_symlink():
                    removed_links += 1
            if args.dry_run:
                print(f"[DRY] remove playlist folder: {d}")
            else:
                shutil.rmtree(d, ignore_errors=True)

    # Remove audio files (unless keep-audio)
    deleted_audio = 0
    if not args.keep_audio:
        for r in affected_rows:
            fp = _resolve_fp((r.get("filepath") or "").strip(), downloads_dir)
            if not fp:
                continue
            if fp.exists():
                if args.dry_run:
                    print(f"[DRY] delete audio: {fp}")
                else:
                    try:
                        fp.unlink()
                        deleted_audio += 1
                    except Exception as e:
                        print(f"[WARN] failed to delete {fp}: {e}")

    # Update ledger
    new_rows = [r for r in rows if (r.get("playlist_title") or "").strip() != title]

    # Update archive by removing IDs that belonged to this playlist
    ids_to_remove = set((r.get("id") or "").strip() for r in affected_rows if (r.get("id") or "").strip())
    new_archive = []
    for ln in archive_lines:
        if any(rid and rid in ln for rid in ids_to_remove):
            continue
        new_archive.append(ln)

    if args.dry_run:
        print(f"[DRY] would write ledger ({len(new_rows)} rows) and archive ({len(new_archive)} lines)")
    else:
        _backup_file(ledger_path); _backup_file(archive_path)
        _write_ledger(ledger_path, new_rows)
        _write_archive(archive_path, new_archive)

    print(f"[DONE] removed links: {removed_links}, removed audio: {deleted_audio}, updated ledger & archive ({'dry-run' if args.dry_run else 'written'})")

def cmd_clean_orphans(args):
    """
    - Remove broken symlinks in dj_playlists
    - Remove ledger rows whose audio filepath is missing (with proper path resolution)
    - Remove archive entries whose IDs no longer exist on disk (based on dropped rows)
    """
    audio_dir = Path(args.outdir).resolve()
    downloads_dir, playlists_root, ledger_path, archive_path = _ledger_paths(audio_dir)

    # 1) broken symlinks
    removed_links = 0
    if playlists_root.exists():
        for root in playlists_root.glob("*"):
            if not root.is_dir():
                continue
            for link in root.iterdir():
                if link.is_symlink():
                    target = link.resolve(strict=False)
                    if not target.exists():
                        if args.dry_run:
                            print(f"[DRY] remove broken link: {link}")
                        else:
                            link.unlink(missing_ok=True)
                        removed_links += 1

    # 2) ledger rows with missing files (resolve relative paths to downloads_dir)
    rows = _read_ledger(ledger_path)
    kept, dropped = [], []
    for r in rows:
        fp_str = (r.get("filepath") or "").strip()
        p = _resolve_fp(fp_str, downloads_dir) if fp_str else None
        if p and p.exists():
            kept.append(r)
        else:
            if args.verbose:
                print(f"[PRUNE] missing on disk → drop: title='{r.get('title','')}' path='{fp_str}' resolved='{p}'")
            dropped.append(r)

    if args.dry_run:
        print(f"[DRY] would write ledger ({len(kept)} rows), dropped {len(dropped)}")
    else:
        _backup_file(ledger_path)
        _write_ledger(ledger_path, kept)

    # 3) archive cleanup (remove IDs of dropped rows)
    archive_lines = _read_archive(archive_path)
    drop_ids = set(_row_id(r) for r in dropped if _row_id(r))
    new_archive = []
    for ln in archive_lines:
        if any(rid and rid in ln for rid in drop_ids):
            continue
        new_archive.append(ln)

    if args.dry_run:
        print(f"[DRY] would write archive ({len(new_archive)} lines)")
    else:
        _backup_file(archive_path)
        _write_archive(archive_path, new_archive)

    print(f"[DONE] removed broken links: {removed_links}; pruned ledger: {len(dropped)}; archive cleaned ({'dry-run' if args.dry_run else 'written'})")

def cmd_sync_playlists(args):
    """
    Rebuild/repair symlinks under downloads/dj_playlists/ based on the ledger.
    Then normalize: convert any real audio files inside playlist folders into
    symlinks pointing to downloads/audio/ when a same-named file exists there.
    """
    audio_dir = Path(args.outdir).resolve()
    downloads_dir, playlists_root, ledger_path, _archive_path = _ledger_paths(audio_dir)

    singles_name = args.singles_name
    exts = tuple(x.lower() for x in args.exts)  # e.g. (".mp3", ".wav", ".m4a", ".flac")

    rows = _read_ledger(ledger_path)
    _safe_mkdir(playlists_root)

    # ----- Pass 1: build symlinks from ledger -----
    touched, missing_files = 0, 0
    for r in rows:
        fp_str = (r.get("filepath") or "").strip()
        fpath = _resolve_fp(fp_str, downloads_dir) if fp_str else None
        if not fpath or not fpath.exists():
            missing_files += 1
            if args.verbose:
                print(f"[SKIP] missing audio (no link): title='{r.get('title','')}' path='{fp_str}' resolved='{fpath}'")
            continue

        ptitle = r.get("playlist_title")
        pid    = r.get("playlist_id")
        subdir = _playlist_dir_name(ptitle, pid, default_singles=singles_name)
        playlist_dir = playlists_root / subdir
        _safe_mkdir(playlist_dir)

        link = playlist_dir / fpath.name
        changed = _ensure_symlink(link, fpath, dry=args.dry_run)
        if changed:
            touched += 1

    print(f"[SYNC] ledger rows: {len(rows)} | links created/updated: {touched}{' (dry-run)' if args.dry_run else ''}")
    if missing_files:
        print(f"[SYNC] missing audio files (not linked): {missing_files}")

    # ----- Pass 2: normalize stray real files inside playlist folders -----
    if args.normalize:
        norm_done, kept = 0, 0
        for pl_dir in playlists_root.glob("*"):
            if not pl_dir.is_dir():
                continue
            for item in pl_dir.iterdir():
                if item.is_symlink():
                    continue
                if not item.is_file():
                    continue
                if item.suffix.lower() not in exts:
                    continue

                candidate = audio_dir / item.name
                if candidate.exists():
                    if args.verbose:
                        print(f"[NORM] replace real file with symlink: {item} -> {candidate}")
                    if args.dry_run:
                        norm_done += 1
                    else:
                        try:
                            item.unlink()
                            _ensure_symlink(item, candidate, dry=False)
                            norm_done += 1
                        except Exception as e:
                            print(f"[WARN] failed to normalize {item}: {e}")
                else:
                    kept += 1
                    if args.verbose:
                        print(f"[NORM] kept as-is (no audio match): {item}")
        print(f"[NORM] normalized files → symlinks: {norm_done}{' (dry-run)' if args.dry_run else ''} | kept: {kept}")

# ----------------------------- main ---------------------------------

def main():
    ap = argparse.ArgumentParser(description="Maintain downloads/audio, dj_playlists/, ledger and archive")
    sp = ap.add_subparsers(dest="cmd", required=True)

    # delete-track
    p = sp.add_parser("delete-track", help="Delete one track by TITLE or PATH (updates ledger + archive + symlinks)")
    p.add_argument("target", help="Exact title (when --by title) or a path (when --by path)")
    p.add_argument("--outdir", required=True, help="downloads/audio directory")
    p.add_argument("--by", choices=["title","path"], default="title")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_delete_track)

    # delete-playlist
    p = sp.add_parser("delete-playlist", help="Delete an entire playlist by its title (folder may be 'Title - ID')")
    p.add_argument("title", help="Playlist title (human name)")
    p.add_argument("--outdir", required=True, help="downloads/audio directory")
    p.add_argument("--keep-audio", action="store_true", help="Keep audio files; only remove playlist symlinks + ledger + archive")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_delete_playlist)

    # clean-orphans
    p = sp.add_parser("clean-orphans", help="Remove broken symlinks; prune ledger rows with missing files; tidy archive")
    p.add_argument("--outdir", required=True, help="downloads/audio directory")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_clean_orphans)

    # sync-playlists
    p = sp.add_parser("sync-playlists", help="Ensure playlist/singles symlinks exist based on ledger; normalize real files into symlinks")
    p.add_argument("--outdir", required=True, help="downloads/audio directory (where real audio lives)")
    p.add_argument("--singles-name", default="Singles", help="Folder name for tracks without playlist info")
    p.add_argument("--exts", nargs="+", default=[".mp3",".wav",".m4a",".flac"], help="Audio extensions to normalize")
    p.add_argument("--normalize", action="store_true", help="Convert real files in playlist folders to symlinks when possible")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_sync_playlists)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()