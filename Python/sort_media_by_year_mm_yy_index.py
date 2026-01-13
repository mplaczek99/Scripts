#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple

MEDIA_EXTS = [
    "jpg","jpeg","png","heic","tif","tiff",
    "mp4","mov","avi","mkv","m4v","3gp",
]

DATE_FORMATS = [
    "%Y:%m:%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y:%m:%d %H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S%z",
]

PHOTO_EXTS = {".jpg",".jpeg",".png",".heic",".tif",".tiff"}
VIDEO_EXTS = {".mp4",".mov",".avi",".mkv",".m4v",".3gp"}

@dataclass(frozen=True)
class Item:
    path: Path
    dt: datetime
    year: int
    month: int
    ext: str

def parse_exif_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    for fmt in DATE_FORMATS:
        try:
            # Drop tzinfo for consistent sorting/formatting
            return datetime.strptime(value, fmt).replace(tzinfo=None)
        except ValueError:
            pass
    return None

def first_dt(meta: dict, keys: List[str]) -> Optional[datetime]:
    for k in keys:
        dt = parse_exif_dt(meta.get(k))
        if dt:
            return dt
    return None

def should_skip(p: Path) -> bool:
    n = p.name
    # skip AppleDouble and common dotfiles
    return n.startswith("._") or n.startswith(".")

def run_exiftool_json(root: Path) -> List[dict]:
    cmd = ["exiftool", "-r", "-json"]
    for ext in MEDIA_EXTS:
        cmd += ["-ext", ext]

    # Ask for key date tags. "CreationDate" is the important one for your MOVs.
    cmd += [
        "-DateTimeOriginal",
        "-CreateDate",
        "-ModifyDate",
        "-CreationDate",       # often maps to Keys:CreationDate for QuickTime
        "-MediaCreateDate",
        "-TrackCreateDate",
        "-FileModifyDate",
        "-FileName",
        "-Directory",
        str(root),
    ]
    out = subprocess.check_output(cmd)
    return json.loads(out.decode("utf-8", errors="replace"))

def unique_target_path(target: Path) -> Path:
    if not target.exists():
        return target
    stem, suf, parent = target.stem, target.suffix, target.parent
    i = 1
    while True:
        cand = parent / f"{stem}_{i}{suf}"
        if not cand.exists():
            return cand
        i += 1

def choose_datetime(meta: dict, p: Path, strict_embedded: bool) -> Optional[datetime]:
    ext = p.suffix.lower()

    if ext in PHOTO_EXTS:
        # Photo priority
        dt = first_dt(meta, ["DateTimeOriginal", "CreateDate", "ModifyDate"])
        if dt:
            return dt

    if ext in VIDEO_EXTS:
        # Video priority: Keys:CreationDate first (shows as "CreationDate" in JSON),
        # then other QuickTime dates.
        dt = first_dt(meta, ["CreationDate", "MediaCreateDate", "TrackCreateDate", "CreateDate"])
        if dt:
            return dt

    if strict_embedded:
        return None

    # Fallback to file times (not recommended for your goal)
    dt = parse_exif_dt(meta.get("FileModifyDate"))
    if dt:
        return dt
    return datetime.fromtimestamp(p.stat().st_mtime)

def main():
    ap = argparse.ArgumentParser(
        description="Sort media into YYYY/ and rename to MM-YY-####.ext using embedded creation dates."
    )
    ap.add_argument("source", help="Source directory (media root)")
    ap.add_argument("--dry-run", action="store_true", help="Print actions but do not move/rename")
    ap.add_argument("--dest", default=None, help="Destination root (default: same as source)")
    ap.add_argument("--start-index", type=int, default=1, help="Starting index for each month-year (default: 1)")
    ap.add_argument("--strict-embedded", action="store_true",
                    help="Only use embedded metadata dates; skip files that lack them (recommended).")
    ap.add_argument("--unknown-folder", default="UnknownDate",
                    help="If not strict, files with no usable date go here (default: UnknownDate).")
    args = ap.parse_args()

    src_root = Path(args.source).resolve()
    dest_root = Path(args.dest).resolve() if args.dest else src_root
    if not src_root.is_dir():
        raise SystemExit(f"Source is not a directory: {src_root}")

    metas = run_exiftool_json(src_root)

    items: List[Item] = []
    unknown: List[Path] = []

    for m in metas:
        directory = m.get("Directory")
        filename = m.get("FileName")
        if not directory or not filename:
            continue
        p = Path(directory) / filename

        if should_skip(p):
            continue
        if not p.exists() or not p.is_file():
            continue

        dt = choose_datetime(m, p, strict_embedded=args.strict_embedded)
        if not dt:
            unknown.append(p)
            continue

        items.append(Item(path=p, dt=dt, year=dt.year, month=dt.month, ext=p.suffix.lower()))

    # Group by (year, month)
    groups: Dict[Tuple[int, int], List[Item]] = {}
    for it in items:
        groups.setdefault((it.year, it.month), []).append(it)

    total_actions = 0

    for (year, month), group in sorted(groups.items()):
        group_sorted = sorted(group, key=lambda x: (x.dt, str(x.path)))
        year_dir = dest_root / f"{year:04d}"
        mm = f"{month:02d}"
        yy = f"{year % 100:02d}"

        if not args.dry_run:
            year_dir.mkdir(parents=True, exist_ok=True)

        idx = args.start_index
        for it in group_sorted:
            new_name = f"{mm}-{yy}-{idx:04d}{it.ext}"
            target = unique_target_path(year_dir / new_name)

            if it.path.resolve() != target.resolve():
                print(f"{it.path}  ->  {target}")
                total_actions += 1
                if not args.dry_run:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(it.path), str(target))
            idx += 1

    # Handle unknowns if not strict
    if unknown and not args.strict_embedded:
        unk_dir = dest_root / args.unknown_folder
        if not args.dry_run:
            unk_dir.mkdir(parents=True, exist_ok=True)
        for p in sorted(unknown, key=str):
            target = unique_target_path(unk_dir / p.name)
            print(f"{p}  ->  {target}")
            total_actions += 1
            if not args.dry_run:
                shutil.move(str(p), str(target))

    if args.strict_embedded and unknown:
        print(f"\nSkipped (no embedded date found): {len(unknown)} files (use --strict-embedded off to move them).")

    print(f"\nDone. Planned actions: {total_actions}{' (dry-run)' if args.dry_run else ''}")

if __name__ == "__main__":
    main()
