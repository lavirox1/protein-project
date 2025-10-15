#!/usr/bin/env python3
import argparse, os, sys, shutil, gzip, zipfile, tarfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PDB_LIKE_EXTS = (".pdb", ".cif")

def iter_archives(root: Path):
    for p in root.rglob("*"):
        if not p.is_file(): continue
        suffs = p.suffixes
        if not suffs: continue
        s2 = "".join(suffs[-2:])
        if p.suffix in (".gz",) or s2 in (".tar.gz",) or p.suffix in (".tgz", ".tar", ".zip"):
            yield p

def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def decompress_gz(src: Path, dst_dir: Path):
    # handles *.pdb.gz / *.cif.gz and generic .gz
    base = src.name[:-3]  # strip .gz
    out = dst_dir / base
    ensure_dir(dst_dir)
    if out.exists(): return f"skip {src} (exists)"
    with gzip.open(src, "rb") as f_in, open(out, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out, length=1024*1024)
    return f"ok   {src} -> {out.name}"

def extract_tar_like(src: Path, dst_dir: Path):
    ensure_dir(dst_dir)
    mode = "r:gz" if "".join(src.suffixes[-2:]) == ".tar.gz" or src.suffix == ".tgz" else "r:"
    count = 0
    with tarfile.open(src, mode) as tf:
        for m in tf.getmembers():
            if not m.isfile(): continue
            name = Path(m.name).name
            if not name.lower().endswith(PDB_LIKE_EXTS): continue
            out = dst_dir / f"{src.stem}__{name}"
            if out.exists(): continue
            f = tf.extractfile(m)
            if f is None: continue
            with open(out, "wb") as g:
                shutil.copyfileobj(f, g, length=1024*1024)
            count += 1
    return f"ok   {src} ({count} files)"

def extract_zip(src: Path, dst_dir: Path):
    ensure_dir(dst_dir)
    count = 0
    with zipfile.ZipFile(src, "r") as zf:
        for n in zf.namelist():
            name = Path(n).name
            if not name.lower().endswith(PDB_LIKE_EXTS): continue
            out = dst_dir / f"{src.stem}__{name}"
            if out.exists(): continue
            with zf.open(n) as f_in, open(out, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out, length=1024*1024)
            count += 1
    return f"ok   {src} ({count} files)"

def worker(path: Path, outdir: Path):
    suffs = path.suffixes
    s2 = "".join(suffs[-2:]) if len(suffs) >= 2 else ""
    try:
        if s2 in (".pdb.gz", ".cif.gz"):
            return decompress_gz(path, outdir)
        if s2 == ".tar.gz" or path.suffix in (".tgz", ".tar"):
            return extract_tar_like(path, outdir)
        if path.suffix == ".zip":
            return extract_zip(path, outdir)
        # generic .gz (we'll still decompress)
        if path.suffix == ".gz":
            return decompress_gz(path, outdir)
        return f"skip {path} (unknown)"
    except Exception as e:
        return f"ERR  {path}: {e}"

def main():
    ap = argparse.ArgumentParser(description="Extract AFDB archives safely")
    ap.add_argument("--src", required=True, help="AFDB root with many archives")
    ap.add_argument("--dst", required=True, help="Output directory for extracted PDB/CIF")
    ap.add_argument("--workers", type=int, default=min(8, (os.cpu_count() or 4)))
    args = ap.parse_args()

    src = Path(args.src); dst = Path(args.dst); ensure_dir(dst)
    todo = list(iter_archives(src))
    if not todo:
        print("Nothing to extract."); return
    print(f"Found {len(todo)} archives. Extracting to: {dst}")
    ok = err = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(worker, p, dst): p for p in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            msg = fut.result()
            print(f"[{i}/{len(todo)}] {msg}")
            if msg.startswith("ERR"): err += 1
            elif msg.startswith("ok"): ok += 1
    print(f"Done. ok={ok}, err={err}, out_dir={dst}")

if __name__ == "__main__":
    main()