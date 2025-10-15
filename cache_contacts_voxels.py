#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Precompute contact maps (L×L) and voxel grids (4×32×32×32) from the Cα PDBs
we wrote in tfds_to_fasta_pdb.py. Speeds up PyTorch training.

Requires:
  pip install biopython numpy tqdm
"""

import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from Bio.PDB import PDBParser

'''def read_ca_coords(pdb_path: Path):
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("x", str(pdb_path))
    coords = []
    for model in struct:
        for chain in model:
            for res in chain:
                if "CA" in res:
                    coords.append(res["CA"].get_vector().get_array())
    return np.asarray(coords, dtype=np.float32)  # [L,3]'''

# Inside cache_contacts_voxels.py as a fallback:
def read_ca_coords_loose(pdb_path):
    coords = []
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                try:
                    x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                    coords.append((x,y,z))
                except ValueError:
                    pass
    return np.asarray(coords, dtype=np.float32)

def contact_map_from_coords(ca_xyz: np.ndarray, cutoff=8.0, max_len=512):
    L = min(len(ca_xyz), max_len)
    C = np.zeros((max_len, max_len), dtype=np.float32)
    for i in range(L):
        for j in range(L):
            dij = np.linalg.norm(ca_xyz[i] - ca_xyz[j])
            C[i, j] = 1.0 if dij < cutoff else 0.0
    return C

def voxels_from_coords(ca_xyz: np.ndarray, box=32, spacing=1.5):
    """
    4-channel grid (C,N,O,S). With CA-only we mark channel-0 (C) where CA lies.
    """
    if ca_xyz.size == 0:
        return np.zeros((4, box, box, box), dtype=np.float32)
    P = ca_xyz - ca_xyz.mean(axis=0)     # center
    half = (box // 2) * spacing
    V = np.zeros((4, box, box, box), dtype=np.float32)
    for p in P:
        if np.any(np.abs(p) > half): 
            continue
        idx = np.clip(((p + half) / spacing).astype(int), 0, box-1)
        V[0, idx[0], idx[1], idx[2]] = 1.0  # put CA into carbon channel
    return V

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb_dir", type=str, default="data/casp7_assets/pdb_ca")
    ap.add_argument("--out_dir", type=str, default="data/casp7_cache")
    ap.add_argument("--cutoff", type=float, default=8.0)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--box", type=int, default=32)
    ap.add_argument("--spacing", type=float, default=1.5)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    pdb_dir = Path(args.pdb_dir)
    out_dir = Path(args.out_dir)
    c_dir = out_dir / "contacts"; c_dir.mkdir(parents=True, exist_ok=True)
    v_dir = out_dir / "voxels";   v_dir.mkdir(parents=True, exist_ok=True)

    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    if args.limit: pdb_files = pdb_files[:args.limit]

    for pdb in tqdm(pdb_files, desc="[cache]"):
        ca = read_ca_coords_loose(pdb)
        C  = contact_map_from_coords(ca, cutoff=args.cutoff, max_len=args.max_len)
        V  = voxels_from_coords(ca, box=args.box, spacing=args.spacing)

        np.save(c_dir / f"{pdb.stem}.npy", C)
        np.save(v_dir / f"{pdb.stem}.npy", V)

    print(f"Done. Saved contacts -> {c_dir}, voxels -> {v_dir}")

if __name__ == "__main__":
    main()