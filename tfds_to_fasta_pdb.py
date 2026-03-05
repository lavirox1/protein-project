#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Downloads and converts the ProteinNet CASP7 dataset (via TensorFlow Datasets)
into FASTA files, CA-only PDBs, and a manifest CSV for training.

Output:
  <out_dir>/fasta/<id>.fasta
  <out_dir>/pdb_ca/<id>.pdb
  <out_dir>/manifest.csv

Usage: python tfds_to_fasta_pdb.py --out_dir data/casp7_assets
Requires: pip install tensorflow tensorflow-datasets numpy pandas tqdm
"""
import csv, argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import tensorflow_datasets as tfds

def write_fasta(path: Path, rec_id: str, seq: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f">{rec_id}\n")
        for i in range(0, len(seq), 80):
            f.write(seq[i:i+80] + "\n")

# Map 1-letter -> 3-letter residue names
AA3 = {
    "A":"ALA","C":"CYS","D":"ASP","E":"GLU","F":"PHE",
    "G":"GLY","H":"HIS","I":"ILE","K":"LYS","L":"LEU",
    "M":"MET","N":"ASN","P":"PRO","Q":"GLN","R":"ARG",
    "S":"SER","T":"THR","V":"VAL","W":"TRP","Y":"TYR",
    "X":"UNK"
}

def write_ca_pdb(path: Path, seq: str, ca_xyz: np.ndarray):
    """
    Strict PDB fixed-width writer for Cα-only trace.
    Columns (PDB v3.3):
    1-6  Record name  'ATOM  '
    7-11 Serial       (right)
    13-16 Atom name   (' CA ')
    17   AltLoc       ' '
    18-20 ResName     (3-letter, right)
    22   ChainID      'A'
    23-26 ResSeq      (right)
    27   iCode        ' '
    31-38 x, 39-46 y, 47-54 z (8.3f)
    55-60 Occupancy   (6.2f)
    61-66 TempFactor  (6.2f)
    77-78 Element     (right)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        serial = 1
        for i, (aa1, (x, y, z)) in enumerate(zip(seq, ca_xyz.tolist()), start=1):
            res3 = AA3.get(aa1.upper(), "UNK")
            line = (
                f"{'ATOM':<6}"           # 1-6
                f"{serial:>5}"           # 7-11
                f" "                      # 12
                f"{'CA':^4}"             # 13-16 atom name centered in 4 cols
                f" "                      # 17 altLoc
                f"{res3:>3}"             # 18-20 resName
                f" "                      # 21
                f"{'A':1}"               # 22 chainID
                f"{i:>4}"                # 23-26 resSeq
                f" "                      # 27 iCode
                f"   "                    # 28-30 (3 spaces)
                f"{x:8.3f}{y:8.3f}{z:8.3f}"  # 31-54 coords
                f"{1.00:6.2f}"            # 55-60 occupancy
                f"{50.00:6.2f}"           # 61-66 tempFactor
                f"{'':>10}"               # 67-76
                f"{'C':>2}"               # 77-78 element
                f"{'':>2}"                # 79-80
            )
            f.write(line + "\n")
            serial += 1
        f.write("END\n")

def tfds_iter(split: str, data_dir: str):
    # Return both dataset and dataset info so we can decode 'primary' -> amino letters
    return tfds.load("protein_net/casp7", split=split, data_dir=data_dir, download=True, with_info=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data/casp7_assets")
    ap.add_argument("--data_dir", type=str, default="data/tfds")   # TFDS cache
    ap.add_argument("--splits", type=str, default="train_70,validation,test")
    ap.add_argument("--limit", type=int, default=0, help="0 = all examples")
    args = ap.parse_args()

    out = Path(args.out_dir)
    fasta_dir = out / "fasta"
    pdb_dir   = out / "pdb_ca"
    out.mkdir(parents=True, exist_ok=True)

    manifest_path = out / "manifest.csv"
    with open(manifest_path, "w", newline="") as mf:
        writer = csv.writer(mf)
        writer.writerow(["id","split","fasta_path","pdb_path","length"])

        for split in args.splits.split(","):
            ds, ds_info = tfds_iter(split, args.data_dir)
            to_aa = ds_info.features['primary'].int2str  # function: int -> amino-acid letter

            count = 0
            for ex in tqdm(ds.as_numpy_iterator(), desc=f"[{split}]"):
                rec_id = ex["id"].decode() if isinstance(ex["id"], (bytes, bytearray)) else str(ex["id"])

                # 'primary' is an int array (0..19). Map back to letters.
                prim = ex["primary"]                      # numpy array of ints, shape [L]
                seq = "".join(to_aa(int(i)) for i in prim.tolist())

                tertiary = ex["tertiary"].astype(np.float32)  # shape (L,3) Cα coords

                fasta_path = fasta_dir / f"{rec_id}.fasta"
                pdb_path   = pdb_dir   / f"{rec_id}.pdb"

                write_fasta(fasta_path, rec_id, seq)
                write_ca_pdb(pdb_path, seq, tertiary)

                writer.writerow([rec_id, split, str(fasta_path), str(pdb_path), len(seq)])

                count += 1
                if args.limit and count >= args.limit:
                    break

    print(f"Done. Wrote manifest: {manifest_path}")

if __name__ == "__main__":
    main()