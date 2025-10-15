#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
End-to-end protein design scaffold in ONE file.

Flow:
1) Scan datasets (ProteinNet/AFDB/Swiss-Prot) -> manifests
2) Preprocess (optional PLM embeddings) + light augmentation
3) Train sequence models: SeqGen_CNNBiLSTM + GRU(TimeNet-like)
4) Train judges: 2D (DenseNet121, NASNet-A) on contact maps -> pLDDT(mean)
                 3D CNN on voxel grids -> pLDDT(mean)
5) Train structure predictor: Distogram DenseNet-U (distogram over bins)
6) Generate sequence with user constraints via ensemble (SeqGen+GRU)
7) Predict structure from distogram -> MDS -> smooth -> Cα-only PDB
8) Score predicted PDB with 2D/3D judges -> proxy "pLDDT" (write into B-factor)
9) Visualize (py3Dmol or PyMOL) + print validity stats

Notes:
- Replace DATA_ROOTS below with your paths.
- This is a scaffold; feel free to refactor into modules as you grow.
"""

import os, sys, json, math, time, random, subprocess, argparse, gzip
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from Bio import SeqIO
# NOTE: Removed all Bio.PDB / PDBParser usage. Everything uses "loose" readers.

# -----------------------------
# CONFIG
# -----------------------------
AMINO = "ACDEFGHIKLMNPQRSTVWY"
AA2IDX = {a:i for i,a in enumerate(AMINO)}
MAX_LEN = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# Optional protein language model (set to None to use one-hot)
PLM_NAME = "Rostlab/prot_bert"    # or None

# Data roots (EDIT THESE)
D_PROTEINNET = Path("data/casp7_assets")
D_AFDB       = Path("data/afdb_extracted")   # can also point to a folder with .pdb.gz / .cif.gz
D_SWISSPROT  = Path("data/swissprot")
D_PN_CACHE   = Path("data/casp7_cache")  # <— cached contacts/voxels from cache_contacts_voxels.py

CHECKPOINT_DIR = Path("checkpoints"); CHECKPOINT_DIR.mkdir(exist_ok=True)

# -----------------------------
# UTILS (augment/one-hot/softmaxT)
# -----------------------------
SIM_GROUPS = [set("IVLMA"), set("ST"), set("DENQ"), set("KRH"), set("FYW"), set("PG"), set("C")]

def similar_mutation(a):
    for g in SIM_GROUPS:
        if a in g: return random.choice(list(g))
    return a

def augment_sequence(seq, p_sub=0.03, p_mask=0.02, mask_tok="X"):
    s = list(seq)
    for i,ch in enumerate(s):
        r = random.random()
        if r < p_sub: s[i] = similar_mutation(ch)
        elif r < p_sub + p_mask: s[i] = mask_tok
    return "".join(s)

def one_hot(seq):
    x = torch.zeros(MAX_LEN, len(AMINO))
    for i,a in enumerate(seq[:MAX_LEN]):
        if a in AA2IDX: x[i, AA2IDX[a]] = 1.0
    return x

def softmax_T(logits, T=1.0):
    z = logits / max(T,1e-6)
    z -= z.max()
    return torch.softmax(z, dim=-1)

import torch.nn as nn

def freeze_bn(m):
    if isinstance(m, nn.BatchNorm2d) or m.__class__.__name__.startswith("BatchNorm"):
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

# -----------------------------
# GZIP-AWARE TEXT OPEN
# -----------------------------
def _open_text(path):
    s = str(path)
    if s.endswith(".gz"):
        return gzip.open(s, "rt", errors="ignore")
    return open(s, "r", errors="ignore")

# -----------------------------
# "LOOSE" PDB READERS (no Bio.PDB)
# -----------------------------
def _safe_float(s, default=None):
    try:
        return float(s)
    except Exception:
        return default

def read_ca_coords_loose(pdb_path: str) -> np.ndarray:
    """Return Nx3 CA coordinates; tolerant to spacing/format quirks; supports .gz."""
    coords = []
    with _open_text(pdb_path) as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if line[12:16].strip() != "CA":
                continue
            x = _safe_float(line[30:38]); y = _safe_float(line[38:46]); z = _safe_float(line[46:54])
            if x is None or y is None or z is None:
                parts = line.split()
                nums = [ _safe_float(tok) for tok in parts if _safe_float(tok) is not None ]
                if len(nums) >= 3:
                    x,y,z = nums[-3], nums[-2], nums[-1]
                else:
                    continue
            coords.append((x,y,z))
    if not coords:
        return np.zeros((0,3), dtype=np.float32)
    return np.asarray(coords, dtype=np.float32)

def read_atoms_loose(pdb_path: str, elements={"C","N","O","S"}) -> List[Tuple[str, np.ndarray]]:
    """Return list of (element, xyz) for selected elements; robust & .gz-aware."""
    out = []
    with _open_text(pdb_path) as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            el = line[76:78].strip().upper()
            if not el:
                name = line[12:16].strip()
                el = name[0].upper() if name else ""
            if el not in elements:
                continue
            x = _safe_float(line[30:38]); y = _safe_float(line[38:46]); z = _safe_float(line[46:54])
            if x is None or y is None or z is None:
                parts = line.split()
                nums = [ _safe_float(tok) for tok in parts if _safe_float(tok) is not None ]
                if len(nums) >= 3:
                    x,y,z = nums[-3], nums[-2], nums[-1]
                else:
                    continue
            out.append((el, np.array([x,y,z], dtype=np.float32)))
    return out

def plddt_stats_from_pdb(pdb_path: str) -> Optional[dict]:
    """Estimate pLDDT stats from B-factor of CA atoms; tolerant & .gz-aware."""
    vals = []
    with _open_text(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            b = _safe_float(line[60:66])
            if b is None:
                parts = line.strip().split()
                for tok in reversed(parts):
                    b = _safe_float(tok)
                    if b is not None:
                        break
            if b is not None:
                vals.append(b)
    if not vals:
        return None
    p = np.asarray(vals, dtype=float)
    return dict(mean=float(p.mean()),
                median=float(np.median(p)),
                pct_ge_70=float((p>=70).mean()*100.0),
                pct_lt_50=float((p<50).mean()*100.0),
                length=int(p.size))

# -----------------------------
# SCAN DATASETS
# -----------------------------
def scan_proteinnet(root: Path):
    items = []
    fasta_dir = root / "fasta"
    pdb_dir   = root / "pdb_ca"
    fasta_map = {p.stem: p for p in fasta_dir.glob("*.fasta")}
    pdb_map   = {p.stem: p for p in pdb_dir.glob("*.pdb")}
    common = sorted(set(fasta_map).intersection(pdb_map))
    for k in common:
        items.append((str(fasta_map[k].resolve()), str(pdb_map[k].resolve())))
    return items

def _possible_fasta_for(p: Path) -> Optional[str]:
    """
    Try to locate a sibling FASTA for a given structure file.
    Handles .pdb, .cif, .pdb.gz, .cif.gz by checking both suffix and stripped .gz.
    """
    # direct .fasta next to exact name minus last suffix
    cand1 = p.with_suffix(".fasta")
    if cand1.exists():
        return str(cand1)
    # if .gz, strip then try again
    if "".join(p.suffixes[-2:]).lower() in (".pdb.gz", ".cif.gz"):
        cand2 = p.with_suffix("").with_suffix(".fasta")
        if cand2.exists():
            return str(cand2)
    return None

def scan_afdb(root: Path, max_files: Optional[int] = None):
    """
    Stream AFDB files; **PDB only** (.pdb, .pdb.gz).
    """
    items = []
    taken = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        if not (name.endswith(".pdb") or name.endswith(".pdb.gz")):
            continue  # <-- ignore .cif / .cif.gz
        fasta = _possible_fasta_for(p)
        items.append((fasta, str(p)))
        taken += 1
        if max_files is not None and taken >= max_files:
            break
    return items

def scan_swissprot(root: Path):
    return [(str(f), None) for f in sorted(root.rglob("*.fasta"))]

# -----------------------------
# STRUCTURE IO (contact map / voxels) using loose readers
# -----------------------------
def contact_map_from_pdb(pdb_path, cutoff=8.0):
    ca = read_ca_coords_loose(pdb_path)
    L = int(min(len(ca), MAX_LEN))
    C = np.zeros((MAX_LEN, MAX_LEN), dtype=np.float32)
    if L == 0:
        return C
    P = ca[:L]
    dif = P[:,None,:] - P[None,:,:]
    D = np.linalg.norm(dif, axis=-1)
    C[:L,:L] = (D < float(cutoff)).astype(np.float32)
    return C

def voxels_from_pdb(pdb_path, box=32, spacing=1.5):
    atoms = read_atoms_loose(pdb_path, elements={"C","N","O","S"})
    if not atoms:
        return np.zeros((4, box, box, box), dtype=np.float32)
    coords = np.stack([p for _,p in atoms])
    center = coords.mean(axis=0)
    CH = {"C":0,"N":1,"O":2,"S":3}
    half = (box//2)*spacing
    grid = np.zeros((4, box, box, box), dtype=np.float32)
    for el,p in atoms:
        q = p - center
        if np.any(np.abs(q) > half):
            continue
        idx = np.clip(((q+half)/spacing).astype(int), 0, box-1)
        grid[CH[el], idx[0], idx[1], idx[2]] = 1.0
    return grid

def validity_percent(stats: dict) -> float:
    v = (0.6*(stats["mean"]/100.0) +
         0.3*(stats["pct_ge_70"]/100.0) +
         0.1*(1.0 - stats["pct_lt_50"]/100.0))
    return round(100*v, 2)

# -----------------------------
# CACHE HELPERS (use precomputed assets if available)
# -----------------------------
def load_cached_contact_by_stem(stem: str) -> Optional[np.ndarray]:
    if not D_PN_CACHE:
        return None
    p = D_PN_CACHE / "contacts" / f"{stem}.npy"
    if p.exists():
        return np.load(p, allow_pickle=False)
    return None

def load_cached_voxel_by_stem(stem: str) -> Optional[np.ndarray]:
    if not D_PN_CACHE:
        return None
    p = D_PN_CACHE / "voxels" / f"{stem}.npy"
    if p.exists():
        return np.load(p, allow_pickle=False)
    return None

# -----------------------------
# OPTIONAL PLM (ProtBERT/ESM)
# -----------------------------
tokenizer = None
plm_model = None
def init_plm(name: Optional[str]):
    global tokenizer, plm_model
    if name is None:
        tokenizer = None; plm_model = None; return
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(name, do_lower_case=False)
    plm_model = AutoModel.from_pretrained(name).to(DEVICE).eval()

@torch.no_grad()
def plm_embed(seq: str) -> torch.Tensor:
    if plm_model is None:
        return one_hot(seq)
    spaced = " ".join(list(seq[:MAX_LEN]))
    t = tokenizer(spaced, return_tensors="pt", add_special_tokens=True)
    t = {k:v.to(DEVICE) for k,v in t.items()}
    out = plm_model(**t).last_hidden_state[:,1:-1,:].squeeze(0)[:MAX_LEN]
    if out.size(0) < MAX_LEN:
        pad = torch.zeros(MAX_LEN-out.size(0), out.size(1), device=out.device)
        out = torch.cat([out, pad], dim=0)
    return out  # [L,D]

def infer_feat_dim_quick() -> int:
    x = plm_embed("M"*8)
    return int(x.shape[-1])

# -----------------------------
# DATASET
# -----------------------------
class SequenceDataset(Dataset):
    """Yields (X_seq [L,F], C_map [L,L], y [L])"""
    def __init__(self, manifest, training=True):
        self.manifest = manifest
        self.training = training
    def __len__(self): return len(self.manifest)
    def __getitem__(self, i):
        fasta, pdb = self.manifest[i]
        seq = str(next(SeqIO.parse(fasta, "fasta")).seq)[:MAX_LEN]
        if self.training:
            seq = augment_sequence(seq)

        # Inputs
        X = plm_embed(seq)  # [MAX_LEN, F]

        # Contact map (cached if available)
        C = torch.zeros(MAX_LEN, MAX_LEN, dtype=torch.float32)
        if pdb:
            stem = Path(pdb).stem
            C_cached = load_cached_contact_by_stem(stem)
            if C_cached is not None:
                C = torch.tensor(C_cached, dtype=torch.float32)
            else:
                C = torch.tensor(contact_map_from_pdb(pdb), dtype=torch.float32)

        # Next-token targets (fixed to MAX_LEN)
        idxs_list = [AA2IDX.get(a, 0) for a in seq[:MAX_LEN]]
        idxs = torch.tensor(idxs_list, dtype=torch.long)           # [L]
        y_roll = torch.roll(idxs, shifts=-1)                       # [L]

        y = torch.zeros(MAX_LEN, dtype=torch.long)                 # [MAX_LEN]
        L = len(idxs_list)
        if L > 0:
            y[:L] = y_roll
        return X, C, y

# -----------------------------
# MODELS: SeqGen + GRU  (UNCHANGED)
# -----------------------------
class SeqGen_CNNBiLSTM(nn.Module):
    def __init__(self, feat_dim, hid=256, out_classes=20):
        super().__init__()
        self.conv = nn.Conv1d(feat_dim, 128, 5, padding=2)
        self.bn   = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(128, hid, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hid*2, out_classes)
    def forward(self, X):  # X:[B,L,F]
        x = X.permute(0,2,1)                 # [B,F,L]
        x = F.relu(self.bn(self.conv(x)))    # [B,128,L]
        x = self.pool(x)                     # [B,128,L/2]
        x = x.permute(0,2,1)                 # [B,L/2,128]
        y,_= self.lstm(x)                    # [B,L/2,2*hid]
        return self.head(y)                  # [B,L/2,20]

class TimeNetLike_GRU(nn.Module):
    def __init__(self, feat_dim, hid=384, layers=2, out_classes=20):
        super().__init__()
        self.proj = nn.Linear(feat_dim, 256)
        self.gru  = nn.GRU(256, hid, num_layers=layers, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hid*2, out_classes)
    def forward(self, X):
        x = F.relu(self.proj(X))             # [B,L,256]
        y,_ = self.gru(x)                    # [B,L,2*hid]
        return self.head(y)                  # [B,L,20]

# -----------------------------
# TRAIN LOOPS (sequence models)  (UNCHANGED)
# -----------------------------
def train_seq_model(model, loader, val_loader=None, epochs=3, lr=1e-3, name="seq"):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = 1e9; path = CHECKPOINT_DIR/f"{name}.pt"
    for ep in range(1, epochs+1):
        model.train(); losses=[]; accs=[]
        pbar = tqdm(loader, desc=f"[{name}] ep{ep}")
        for X,C,y in pbar:
            X,y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)                # [B,L',20] or [B,L,20]
            B,Lp,Cv = logits.shape
            loss = F.cross_entropy(logits.reshape(-1,Cv), y[:, :Lp].reshape(-1))
            with torch.no_grad():
                acc = (logits.argmax(-1) == y[:, :Lp]).float().mean().item()
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item()); accs.append(acc)
            pbar.set_postfix(loss=np.mean(losses), acc=np.mean(accs))
        if val_loader:
            v = eval_seq_loss(model, val_loader)
            print(f"[{name}] val_loss={v:.4f}")
            if v < best: best=v; torch.save(model.state_dict(), path)
        else:
            torch.save(model.state_dict(), path)
    print(f"Saved best {name} -> {path}"); return path

@torch.no_grad()
def eval_seq_loss(model, loader):
    model.eval().to(DEVICE); losses=[]
    for X,C,y in loader:
        X,y = X.to(DEVICE), y.to(DEVICE)
        logits = model(X); B,Lp,Cv = logits.shape
        loss = F.cross_entropy(logits.reshape(-1,Cv), y[:, :Lp].reshape(-1))
        losses.append(loss.item())
    return float(np.mean(losses))

# -----------------------------
# 2D JUDGES (DenseNet/NASNet via timm)
# -----------------------------
import timm

def _adapt_first_conv_to_1ch(m):
    if getattr(m, "in_channels", 3) == 3:
        w = m.weight.data
        m.weight = nn.Parameter(w.mean(dim=1, keepdim=True))

class DenseNet2DScorer(nn.Module):
    def __init__(self, out_dim=1):
        super().__init__()
        self.backbone = timm.create_model("densenet121", pretrained=True, in_chans=1, num_classes=out_dim)
        try:
            _adapt_first_conv_to_1ch(self.backbone.features.conv0)
        except Exception:
            pass
    def forward(self, C):   # C:[B,1,L,L]
        return self.backbone(C)

class NASNet2DScorer(nn.Module):
    def __init__(self, out_dim=1):
        super().__init__()
        self.backbone = timm.create_model("nasnetalarge", pretrained=True, in_chans=1, num_classes=out_dim)
    def forward(self, C):
        return self.backbone(C)

# -----------------------------
# 3D JUDGE
# -----------------------------
class Volume3DScorer(nn.Module):
    def __init__(self, in_ch=4):
        super().__init__()
        self.c1 = nn.Conv3d(in_ch, 16, 3, padding=1)
        self.c2 = nn.Conv3d(16, 32, 3, padding=1)
        self.c3 = nn.Conv3d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool3d(2)
        self.gap  = nn.AdaptiveAvgPool3d(4)      # -> [B,64,4,4,4] regardless of input
        self.fc   = nn.Linear(64*4*4*4, 1)

    def forward(self, V):
        x = F.relu(self.c1(V))
        x = self.pool(F.relu(self.c2(x)))
        x = self.pool(F.relu(self.c3(x)))
        x = self.gap(x)              # [B,64,4,4,4]
        x = x.flatten(1)             # [B,4096]
        return self.fc(x)

# -----------------------------
# Distogram utils (labels + pairwise feats) with loose CA reader
# -----------------------------
def distogram_from_pdb(pdb_path, L, bins=np.linspace(2.0, 20.0, 64)):
    cas = read_ca_coords_loose(pdb_path)
    L = int(min(L, len(cas)))
    D = np.full((L, L), 25.0, dtype=np.float32)
    if L > 0:
        P = cas[:L]
        dif = P[:,None,:] - P[None,:,:]
        D = np.linalg.norm(dif, axis=-1).astype(np.float32)
    centers = 0.5*(bins[1:]+bins[:-1])  # 63 centers
    idx = np.digitize(D, bins[1:])
    idx = np.clip(idx, 0, len(centers)-1)
    return torch.tensor(idx, dtype=torch.long), torch.tensor(centers, dtype=torch.float32)

def pairwise_features(res_emb):  # [L,F] -> [L,L,4F]
    Ei = res_emb.unsqueeze(1)    # [L,1,F]
    Ej = res_emb.unsqueeze(0)    # [1,L,F]
    diff = (Ei - Ej).abs()
    prod = Ei * Ej
    pw = torch.cat([Ei.expand(-1,Ej.size(1),-1),
                    Ej.expand(Ei.size(0),-1,-1),
                    diff, prod], dim=-1)
    return pw

def ca_coords_and_len(pdb_path: str, Lseq: int):
    """
    Use the same loose CA reader everywhere; return (coords[:L], L) with L = min(Lseq, L_CA).
    """
    cas = read_ca_coords_loose(pdb_path)  # robust token/fixed-width hybrid
    L = int(min(Lseq, len(cas)))
    if L <= 0:
        return np.zeros((0,3), np.float32), 0
    return cas[:L], L

def true_distance_matrix_from_pdb(pdb_path, Lseq):
    """
    True Cα–Cα distance matrix using the loose reader; guard against absurd coordinates.
    """
    P, L = ca_coords_and_len(pdb_path, Lseq)
    if L <= 1:
        return torch.zeros(0, 0, dtype=torch.float32)

    # sanity guard: if coords are absurdly large, skip by returning NaNs
    if np.max(np.abs(P)) > 5000:  # >> realistic Angstrom ranges
        return torch.full((L, L), float("nan"), dtype=torch.float32)

    dif = P[:, None, :] - P[None, :, :]
    D = np.linalg.norm(dif, axis=-1).astype(np.float32)
    return torch.tensor(D, dtype=torch.float32)

@torch.no_grad()
def eval_distogram(model, val_manifest, Lmax=256, bin_centers=None, exclude_k=5,
                   true_max_cutoff=10000.0, verbose=True):
    """
    Returns (val_CE, val_RMSE). Masks |i-j| <= exclude_k and clamps both
    predicted and true distances to the model's range [2,20] Å for RMSE.
    """
    if not val_manifest:
        return float("nan"), float("nan")

    model.eval().to(DEVICE)
    ce_losses, rmses = [], []

    # default bin centers (63 bins from 2..20 Å)
    if bin_centers is None:
        bins = np.linspace(2.0, 20.0, 64)
        bin_centers = torch.tensor(0.5 * (bins[1:] + bins[:-1]),
                                   dtype=torch.float32, device=DEVICE)

    # counters (debug)
    n_total = 0; n_ce = 0; n_rmse_kept = 0
    n_skip_len = 0; n_skip_nonfinite = 0; n_skip_cutoff = 0

    for fasta, pdb in val_manifest:
        if fasta is None or pdb is None:
            continue
        n_total += 1

        # --- align lengths ---
        seq = str(next(SeqIO.parse(fasta, "fasta")).seq)
        L_seq = min(len(seq), MAX_LEN, Lmax)
        _, L_pdb = ca_coords_and_len(pdb, L_seq)
        L = min(L_seq, L_pdb)
        if L < 2:
            n_skip_len += 1
            continue

        # --- build input [1,3,L,L] ---
        emb = plm_embed(seq).detach().cpu()[:L]           # [L,F]
        pw  = pairwise_features(emb)                      # [L,L,4F]
        C_in = 3
        if pw.shape[-1] % C_in != 0:
            pad = torch.zeros(pw.shape[0], pw.shape[1], C_in - (pw.shape[-1] % C_in))
            pw = torch.cat([pw, pad], dim=-1)
        X  = pw.view(L, L, C_in, -1).mean(-1).permute(2,0,1).unsqueeze(0).to(DEVICE)

        # --- labels [1,L,L] ---
        Y_idx, _ = distogram_from_pdb(pdb, L)             # [L,L]
        Y = Y_idx.unsqueeze(0).to(DEVICE)                 # [1,L,L]

        # --- logits & CE ---
        logits = model(X)                                  # [1, n_bins, L, L]
        if exclude_k > 0:
            ce_map = F.cross_entropy(logits, Y, reduction="none").squeeze(0)  # [L,L]
            # build mask to exclude |i-j| <= exclude_k
            mask = torch.ones((L, L), dtype=torch.bool, device=ce_map.device)
            for d in range(-exclude_k, exclude_k+1):
                mask &= ~torch.diag(torch.ones(L - abs(d), dtype=torch.bool, device=ce_map.device), diagonal=d)
            ce_losses.append(ce_map[mask].mean().item())
        else:
            ce_losses.append(F.cross_entropy(logits, Y).item())
        n_ce += 1

        # --- RMSE in Å (clipped to model range and masked near-diagonal) ---
        Dhat = expected_dist_matrix(logits.squeeze(0), bin_centers)  # [L,L], device
        Dtrue = true_distance_matrix_from_pdb(pdb, L)                # [L,L], cpu

        if (not torch.isfinite(Dhat).all()) or (not torch.isfinite(Dtrue).all()) or Dtrue.numel() == 0:
            n_skip_nonfinite += 1
            continue

        # optional sanity cutoff on the *raw* Dtrue (detect totally broken parses)
        if true_max_cutoff is not None and float(Dtrue.max()) > float(true_max_cutoff):
            n_skip_cutoff += 1
            continue

        # clamp BOTH to the model's range (fair RMSE)
        Dhat_c  = Dhat.clamp(2.0, 20.0).cpu().float()
        Dtrue_c = Dtrue.clamp(2.0, 20.0).float()

        if exclude_k > 0:
            m = torch.ones((L, L), dtype=torch.bool)
            for d in range(-exclude_k, exclude_k+1):
                m &= ~torch.diag(torch.ones(L - abs(d), dtype=torch.bool), diagonal=d)
            diff = (Dhat_c - Dtrue_c)[m]
        else:
            diff = (Dhat_c - Dtrue_c).reshape(-1)

        rmse = torch.sqrt((diff ** 2).mean()).item()
        if not np.isfinite(rmse):
            n_skip_nonfinite += 1
            continue
        rmses.append(rmse); n_rmse_kept += 1

        # print stats for the first kept sample
        if verbose and n_rmse_kept == 1:
            from tqdm import tqdm
            try:
                tqdm.write(
                    f"bin_centers: {bin_centers.min().item():.2f}..{bin_centers.max().item():.2f} | "
                    f"Dhat: {float(Dhat.min().cpu()):.2f}..{float(Dhat.max().cpu()):.2f} | "
                    f"Dtrue: {float(Dtrue.min()):.2f}..{float(Dtrue.max()):.2f} | L={L}"
                )
            except Exception:
                print("bin_centers:", bin_centers.min().item(), bin_centers.max().item())
                print("Dhat stats:", float(Dhat.min().cpu()), float(Dhat.max().cpu()))
                print("Dtrue stats:", float(Dtrue.min()), float(Dtrue.max()))
                print("L =", L)

    if verbose:
        print(f"[eval_dist] total={n_total}  CE_samples={n_ce}  RMSE_used={n_rmse_kept}  "
              f"skip_len<{2}={n_skip_len}  skip_nonfinite={n_skip_nonfinite}  "
              f"skip_cutoff>{true_max_cutoff}={n_skip_cutoff}")

    val_ce   = float(np.mean(ce_losses)) if ce_losses else float("nan")
    val_rmse = float(np.mean(rmses))     if rmses     else float("nan")
    return val_ce, val_rmse

# -----------------------------
# Distogram DenseNet-U (DenseNet encoder + upsampling head)
# -----------------------------

class DistogramDenseNetU(nn.Module):
    def __init__(self, c_in=3, n_bins=63):
        super().__init__()
        self.backbone = timm.create_model("densenet121", pretrained=True, features_only=True, out_indices=(0,1,2,3))
        self.proj_in  = nn.Conv2d(c_in, 3, kernel_size=1)  # to 3ch if needed
        chs = self.backbone.feature_info.channels()  # e.g., [64,128,256,1024]
        self.up3 = nn.ConvTranspose2d(chs[-1], 256, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(256+chs[2], 128, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(128+chs[1], 64,  2, stride=2)
        self.up0 = nn.ConvTranspose2d(64+chs[0], 64,   2, stride=2)
        self.head = nn.Conv2d(64, n_bins, 1)

    def _resize(self, x, ref):
        # bilinear resize to ref’s HxW if needed
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, x):  # x:[B,C_in,L,L]
        B,C,H,W = x.shape
        x = self.proj_in(x)

        # 1) pad to multiples of 16 so the down/up path aligns cleanly
        H0, W0 = H, W
        pad_h = (16 - H % 16) % 16
        pad_w = (16 - W % 16) % 16
        if pad_h or pad_w:
            # F.pad order for 4D tensors is (left, right, top, bottom) on W then H
            x = F.pad(x, (0, pad_w, 0, pad_h))

        # 2) backbone features
        f0, f1, f2, f3 = self.backbone(x)  # low -> high depth

        # 3) up path with safe resizing before concat
        u3 = self.up3(f3)                  # upsample deepest
        u3 = self._resize(u3, f2)
        u2 = self.up2(torch.cat([u3, f2], dim=1))

        u2 = self._resize(u2, f1)
        u1 = self.up1(torch.cat([u2, f1], dim=1))

        u1 = self._resize(u1, f0)
        u0 = self.up0(torch.cat([u1, f0], dim=1))

        out = self.head(u0)

        # 4) crop back to original HxW (undo padding)
        if pad_h or pad_w:
            out = out[..., :H0, :W0]

        # extra guard: ensure exact (H,W)
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)

        return out  # [B,n_bins,H,W]

# -----------------------------
# Distogram training helpers
# -----------------------------

def build_dist_batch(batch_records, Lmax=256):
    Xs, Ys = [], []
    centers = None
    for fasta, pdb in batch_records:
        if fasta is None or pdb is None:
            continue

        seq = str(next(SeqIO.parse(fasta, "fasta")).seq)
        L_seq = min(len(seq), MAX_LEN, Lmax)

        # align to PDB CA count (strict)
        _, L_pdb = ca_coords_and_len(pdb, L_seq)
        L = min(L_seq, L_pdb)
        if L < 2:
            continue

        emb = plm_embed(seq).detach().cpu()[:L]       # [L,F]
        pw  = pairwise_features(emb)                  # [L,L,4F]

        C_in = 3
        if pw.shape[-1] % C_in != 0:
            pad = torch.zeros(pw.shape[0], pw.shape[1], C_in - (pw.shape[-1] % C_in))
            pw = torch.cat([pw, pad], dim=-1)
        pw = pw.view(L, L, C_in, -1).mean(-1)         # [L,L,3]
        pw = pw.permute(2,0,1)                        # [3,L,L]

        y, ctr = distogram_from_pdb(pdb, L)           # [L,L]
        Xs.append(pw); Ys.append(y); centers = ctr

    if len(Xs) == 0:
        X = torch.zeros(1,3,8,8)
        Y = torch.zeros(1,8,8, dtype=torch.long)
        centers = torch.linspace(2.0, 20.0, 63)
        return X, Y, centers

    X = torch.stack(Xs, 0)                            # [B,3,L,L] (use bs=1)
    Y = torch.stack(Ys, 0)                            # [B,L,L]
    return X, Y, centers


def train_distogram(model, train_manifest, val_manifest, Lmax=256, epochs=2, bs=1, lr=1e-4, name="dist_densenetU"):
    """
    Train with per-epoch validation:
      - Prints train CE, val CE, val RMSE(Å)
      - Saves best-by-val CE to checkpoints/{name}.pt
      NOTE: Keep bs=1 unless you pad to a common L, since sequences vary in length.
    """
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    path = CHECKPOINT_DIR / f"{name}.pt"

    # define bin centers once (consistency for eval & prediction)
    bins = np.linspace(2.0, 20.0, 64)
    bin_centers = torch.tensor(0.5 * (bins[1:] + bins[:-1]), dtype=torch.float32).to(DEVICE)

    best_val = float("inf")

    for ep in range(1, epochs+1):
        model.train()
        model.apply(freeze_bn)
        losses = []

        # iterate in bs=1 chunks to avoid variable-L stacking issues
        for i in tqdm(range(0, len(train_manifest), bs), desc=f"[{name}] ep{ep}"):
            batch = train_manifest[i:i+bs]
            X, Y, _centers_unused = build_dist_batch(batch, Lmax=Lmax)  # X:[1,3,L,L], Y:[1,L,L]
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            logits = model(X)                 # [1,n_bins,L,L]
            loss = F.cross_entropy(logits, Y)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())

        train_ce = float(np.mean(losses)) if losses else float("nan")

        # validation
        if val_manifest:
            val_ce, val_rmse = eval_distogram(model, val_manifest, Lmax=Lmax, bin_centers=bin_centers, exclude_k=5, true_max_cutoff=10000.0)
            print(f"[{name}] train CE={train_ce:.4f} | val CE={val_ce:.4f} | val RMSE={val_rmse:.3f} Å")
            # save best-by-val CE
            if val_ce < best_val:
                best_val = val_ce
                torch.save(model.state_dict(), path)
        else:
            print(f"[{name}] train CE={train_ce:.4f}")
            torch.save(model.state_dict(), path)

    print(f"Saved best {name} -> {path}")
    # return CPU tensor for downstream use
    return path, bin_centers.detach().cpu()

# -----------------------------
# Reconstruction: distogram -> coords
# -----------------------------
def expected_dist_matrix(logits, bin_centers):  # logits: [n_bins, L, L]
    probs = torch.softmax(logits, dim=0)
    Dhat = (probs * bin_centers.view(-1,1,1).to(logits.device)).sum(0)
    Dhat = 0.5*(Dhat + Dhat.t())
    Dhat.fill_diagonal_(0.0)
    return Dhat

def cmds_from_dist(D):
    D = D.detach().cpu().numpy()
    n = D.shape[0]
    J = np.eye(n) - np.ones((n,n))/n
    B = -0.5 * J.dot(D**2).dot(J)
    vals, vecs = np.linalg.eigh(B)
    idx = np.argsort(vals)[::-1][:3]
    Lm = np.diag(np.sqrt(np.maximum(vals[idx], 0)))
    X = vecs[:, idx].dot(Lm)
    return torch.tensor(X, dtype=torch.float32)

def smooth_refine(coords, iters=400, lr=5e-3, w_bond=10.0):
    # make a leaf param we can optimize
    coords = coords.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([coords], lr=lr)

    for _ in range(iters):
        # re-enable autograd even if caller is under torch.no_grad()
        with torch.enable_grad():
            diffs  = coords[1:] - coords[:-1]
            bond   = (diffs.norm(dim=1) - 3.8)**2
            smooth = ((coords[2:] - 2*coords[1:-1] + coords[:-2])**2).sum(dim=1)
            loss   = w_bond * bond.mean() + 0.1 * smooth.mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

    return coords.detach()

AA1_TO_AA3 = {
    "A":"ALA","C":"CYS","D":"ASP","E":"GLU","F":"PHE","G":"GLY","H":"HIS",
    "I":"ILE","K":"LYS","L":"LEU","M":"MET","N":"ASN","P":"PRO","Q":"GLN",
    "R":"ARG","S":"SER","T":"THR","V":"VAL","W":"TRP","Y":"TYR"
}

def write_ca_pdb(seq, ca_coords, out_path, proxy_plddt=None, chain_id="A"):
    with open(out_path, "w") as f:
        for i, (aa, xyz) in enumerate(zip(seq, ca_coords.tolist()), start=1):
            x, y, z = xyz
            resn = AA1_TO_AA3.get(aa, "UNK")
            b = 0.0 if proxy_plddt is None else float(proxy_plddt)  # pLDDT proxy in B-factor
            # PDB fixed-width fields (CA only)
            f.write(
                f"ATOM  {i:5d}  CA  {resn:>3s} {chain_id}{i:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 {b:6.2f}           C\n"
            )
        f.write("TER\nEND\n")

# -----------------------------
# Judges training (2D/3D) on AFDB labels
# -----------------------------

@torch.no_grad()
def eval_2d_scorer(model, data):
    model.eval().to(DEVICE)
    losses = []
    for pdb, y in data:
        C = torch.tensor(contact_map_from_pdb(pdb)).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
        tgt = torch.tensor([[y]], dtype=torch.float32).to(DEVICE)
        pred = torch.sigmoid(model(C))
        losses.append(F.mse_loss(pred, tgt).item())
    return float(np.mean(losses)) if losses else float("nan")

@torch.no_grad()
def eval_3d_scorer(model, data):
    model.eval().to(DEVICE)
    losses = []
    for pdb, y in data:
        V = torch.tensor(voxels_from_pdb(pdb), dtype=torch.float32).unsqueeze(0).to(DEVICE)
        tgt = torch.tensor([[y]], dtype=torch.float32).to(DEVICE)
        pred = torch.sigmoid(model(V))
        losses.append(F.mse_loss(pred, tgt).item())
    return float(np.mean(losses)) if losses else float("nan")

def ds_contact_reg(manifest):  # list[(fasta|None, pdb)]
    data=[]
    for fasta,pdb in manifest:
        if not pdb: continue
        st = plddt_stats_from_pdb(pdb)
        if not st: continue
        data.append((pdb, st["mean"]/100.0))
    return data

def ds_voxel_reg(manifest):
    data=[]
    for fasta,pdb in manifest:
        if not pdb: continue
        st = plddt_stats_from_pdb(pdb)
        if not st: continue
        data.append((pdb, st["mean"]/100.0))
    return data

def train_2d_scorer(model, data, val_data=None, epochs=2, lr=1e-4, name="2d"):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    path = CHECKPOINT_DIR/f"{name}.pt"
    best_val = float("inf")
    for ep in range(1, epochs+1):
        model.train(); losses=[]
        for pdb, y in tqdm(data, desc=f"[{name}] ep{ep}"):
            C = torch.tensor(contact_map_from_pdb(pdb)).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
            tgt = torch.tensor([[y]], dtype=torch.float32).to(DEVICE)
            pred = torch.sigmoid(model(C))
            loss = F.mse_loss(pred, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        train_mse = float(np.mean(losses)) if losses else float("nan")

        if val_data:
            val_mse = eval_2d_scorer(model, val_data)
            print(f"[{name}] train MSE={train_mse:.4f} | val MSE={val_mse:.4f}")
            if val_mse < best_val:
                best_val = val_mse
                torch.save(model.state_dict(), path)
        else:
            print(f"[{name}] train MSE={train_mse:.4f}")
            torch.save(model.state_dict(), path)

    print(f"Saved best {name} -> {path}")
    return path

def train_3d_scorer(model, data, val_data=None, epochs=2, lr=1e-4, name="3d"):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    path = CHECKPOINT_DIR/f"{name}.pt"
    best_val = float("inf")
    for ep in range(1, epochs+1):
        model.train(); losses=[]
        for pdb, y in tqdm(data, desc=f"[{name}] ep{ep}"):
            V = torch.tensor(voxels_from_pdb(pdb), dtype=torch.float32).unsqueeze(0).to(DEVICE)
            tgt = torch.tensor([[y]], dtype=torch.float32).to(DEVICE)
            pred = torch.sigmoid(model(V))
            loss = F.mse_loss(pred, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        train_mse = float(np.mean(losses)) if losses else float("nan")

        if val_data:
            val_mse = eval_3d_scorer(model, val_data)
            print(f"[{name}] train MSE={train_mse:.4f} | val MSE={val_mse:.4f}")
            if val_mse < best_val:
                best_val = val_mse
                torch.save(model.state_dict(), path)
        else:
            print(f"[{name}] train MSE={train_mse:.4f}")
            torch.save(model.state_dict(), path)

    print(f"Saved best {name} -> {path}")
    return path

# -----------------------------
# ENSEMBLE GENERATION (SeqGen + GRU)  (UNCHANGED)
# -----------------------------
@torch.no_grad()
def ensemble_next_logits(models: List[nn.Module], prefix: str):
    X = plm_embed(prefix).unsqueeze(0).to(DEVICE)  # [1,L,F]
    outs=[]
    for m in models:
        m.eval().to(DEVICE)
        out = m(X)            # [1,L',20] or [1,L,20]
        outs.append(out[:,-1,:])
    return torch.mean(torch.stack(outs, dim=0), dim=0).squeeze(0)  # [20]

def generate_sequence(models, seed, steps, T=0.9, fixed_positions=None, forbid=set(), must_include=None):
    seq = list(seed)
    for t in range(steps):
        if fixed_positions and len(seq) in fixed_positions:
            seq.append(fixed_positions[len(seq)]); continue
        logits = ensemble_next_logits(models, "".join(seq))
        mask = torch.zeros_like(logits)
        for a in forbid: mask[AA2IDX[a]] = -1e9
        probs = softmax_T(logits + mask, T=T).cpu().numpy()
        idx = np.random.choice(len(AMINO), p=probs)
        seq.append(AMINO[idx])
    s = "".join(seq)
    if must_include and must_include not in s:
        pass
    return s

# -----------------------------
# DISTOGRAM -> PDB (prediction)
# -----------------------------

@torch.no_grad()
def predict_pdb_from_sequence(seq, dist_model, bin_centers, Lmax=256, out_path=None):
    L = min(Lmax, MAX_LEN, len(seq))
    if L < 2:
        raise ValueError("Sequence too short.")

    seq = seq[:L]
    dist_model.eval().to(DEVICE)

    emb = plm_embed(seq).detach().cpu()[:L]           # [L,F]
    pw  = pairwise_features(emb)                      # [L,L,4F]
    C_in = 3
    if pw.shape[-1] % C_in != 0:
        pad = torch.zeros(pw.shape[0], pw.shape[1], C_in - (pw.shape[-1] % C_in))
        pw = torch.cat([pw, pad], dim=-1)
    pw = pw.view(L, L, C_in, -1).mean(-1).permute(2,0,1).unsqueeze(0).to(DEVICE)  # [1,3,L,L]

    logits = dist_model(pw).squeeze(0)                # [n_bins,L,L]
    Dhat   = expected_dist_matrix(logits, bin_centers)
    X0     = cmds_from_dist(Dhat)
    X      = smooth_refine(X0, iters=400, lr=5e-3)

    if out_path:
        write_ca_pdb(seq, X.cpu(), out_path)
    return X.cpu(), Dhat.cpu()

# -----------------------------
# Proxy "pLDDT" from judges & write PDB
# -----------------------------

@torch.no_grad()
def proxy_plddt_for_pred(coords, seq, den2d: nn.Module, nas2d: nn.Module, vol3d: nn.Module):
    # ensure models live on the same device as inputs
    den2d = den2d.to(DEVICE).eval()
    nas2d = nas2d.to(DEVICE).eval()
    vol3d = vol3d.to(DEVICE).eval()

    P = coords.numpy()
    L = P.shape[0]
    C = np.zeros((L, L), dtype=np.float32)  # (smaller/faster than MAX_LEN x MAX_LEN)
    for i in range(L):
        for j in range(L):
            C[i,j] = 1.0 if np.linalg.norm(P[i]-P[j]) < 8.0 else 0.0
    C_t = torch.tensor(C).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

    s2d1 = torch.sigmoid(den2d(C_t)).item()
    s2d2 = torch.sigmoid(nas2d(C_t)).item()

    box=32; spacing=1.5
    coords_centered = P - P.mean(axis=0)
    half = (box//2)*spacing
    V = np.zeros((4,box,box,box), dtype=np.float32)
    for p in coords_centered:
        if np.any(np.abs(p) > half): continue
        idx = np.clip(((p+half)/spacing).astype(int), 0, box-1)
        V[0, idx[0], idx[1], idx[2]] = 1.0
    V_t = torch.tensor(V).unsqueeze(0).to(DEVICE)

    s3d = torch.sigmoid(vol3d(V_t)).item()
    proxy = 100.0 * (0.4*s2d1 + 0.4*s2d2 + 0.2*s3d)
    return max(0.0, min(100.0, proxy))

# -----------------------------
# PyMOL render (optional) & py3Dmol view helper
# -----------------------------

def render_matplotlib_ca(pdb_path: str, out_png: str):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    P = read_ca_coords_loose(pdb_path)
    if P.size == 0:
        print("[WARN] No CA coords to render.")
        return
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(P[:,0], P[:,1], P[:,2], linewidth=1)
    ax.scatter(P[:,0], P[:,1], P[:,2], s=4)
    ax.set_axis_off()
    ax.view_init(elev=20, azim=30)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[matplotlib] Saved {out_png}")

from pathlib import Path
import subprocess, sys

def render_pymol_png(pdb_path: str, out_png: str, pymol_exe: str):
    """
    Call the real PyMOL executable and save a PNG to an absolute path.
    Examples:
      pymol_exe=r"C:\\Users\\Anvay Borade\\AppData\\Local\\Schrodinger\\PyMOL2\\PyMOL.exe"
    """
    pdb_abs = Path(pdb_path).resolve()
    out_abs = Path(out_png).resolve()
    out_abs.parent.mkdir(parents=True, exist_ok=True)

    # Use POSIX-style slashes & quotes in PML to avoid Windows spacing issues
    pdb_q = str(pdb_abs).replace("\\", "/")
    out_q = str(out_abs).replace("\\", "/")

    pml = f'''
        reinitialize
        load "{pdb_q}", prot
        hide everything
        show cartoon, prot
        set ray_opaque_background, off
        bg_color white
        ray 1200,900
        png "{out_q}", dpi=300
        quit
        '''
    script = Path("render_tmp.pml")
    script.write_text(pml)

    try:
        # run PyMOL; no reliance on python -m pymol
        subprocess.run([pymol_exe, "-cq", str(script.resolve())], check=True)
    except Exception as e:
        print("[WARN] PyMOL failed; falling back to matplotlib:", e)
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa
            import numpy as np
            P = read_ca_coords_loose(str(pdb_abs))
            if P.size == 0:
                print("[WARN] No CA coords to render.")
            else:
                fig = plt.figure(figsize=(6,5))
                ax = fig.add_subplot(111, projection='3d')
                ax.plot(P[:,0], P[:,1], P[:,2], linewidth=1)
                ax.scatter(P[:,0], P[:,1], P[:,2], s=4)
                ax.set_axis_off(); ax.view_init(elev=20, azim=30)
                fig.savefig(str(out_abs), dpi=300, bbox_inches='tight'); plt.close(fig)
                print(f"[matplotlib] Saved {out_abs}")
        finally:
            if script.exists(): script.unlink(missing_ok=True)
        return str(out_abs)

    # verify where the file actually landed
    if out_abs.exists():
        print(f"[PyMOL] Saved {out_abs}")
    else:
        print("[WARN] PyMOL returned but file not found at expected path.")
        print("  Expected:", out_abs)
        print("  CWD was :", Path.cwd())
        # small rescue: look for any .png with same stem in current dir
        candidates = list(out_abs.parent.glob(out_abs.stem + "*.png"))
        if candidates:
            print("  Found candidates:", [str(c) for c in candidates])

    if script.exists():
        script.unlink(missing_ok=True)
    return str(out_abs)

def view_pdb_py3Dmol(pdb_file, width=800, height=600):
    try:
        import py3Dmol
    except ImportError:
        print("py3Dmol not installed (pip install py3Dmol).")
        return None
    pdb_data = Path(pdb_file).read_text()
    view = py3Dmol.view(width=width, height=600)
    view.addModel(pdb_data, 'pdb')
    view.setStyle({'cartoon': {}})
    view.zoomTo()
    return view

# -----------------------------
# MAIN ORCHESTRATION (with CLI switches)
# -----------------------------
def split_manifest(man, ratio=0.9):
    random.shuffle(man); n=int(ratio*len(man))
    return man[:n], man[n:]

def main():
    parser = argparse.ArgumentParser(description="Protein end2end pipeline (modular)")
    parser.add_argument("--no-plm", action="store_true", help="Disable PLM; use one-hot")
    # which stages to run
    parser.add_argument("--train-seq", action="store_true", help="Train sequence models (CNN+BiLSTM and GRU)")
    parser.add_argument("--train-densenet2d", action="store_true", help="Train DenseNet 2D scorer")
    parser.add_argument("--train-nasnet2d", action="store_true", help="Train NASNet 2D scorer")
    parser.add_argument("--train-vol3d", action="store_true", help="Train 3D voxel scorer")
    parser.add_argument("--train-dist", action="store_true", help="Train Distogram DenseNet-U")
    parser.add_argument("--predict", action="store_true", help="Generate sequence and predict structure using trained models")
    parser.add_argument("--seq-epochs", type=int, default=3)
    parser.add_argument("--d2-epochs", type=int, default=2)
    parser.add_argument("--d3-epochs", type=int, default=2)
    parser.add_argument("--dist-epochs", type=int, default=2)
    parser.add_argument("--dist-bs", type=int, default=1)
    parser.add_argument("--seed-seq", type=str, default="M")
    parser.add_argument("--gen-len", type=int, default=200)
    parser.add_argument("--afdb-limit", type=int, default=500000, help="Max AFDB files to scan (avoids RAM blow-ups)")
    args = parser.parse_args()

    # 0) init PLM
    init_plm(None if args.no_plm else PLM_NAME)

    # 1) manifests
    man_pn = scan_proteinnet(D_PROTEINNET)
    man_af = scan_afdb(D_AFDB, max_files=args.afdb_limit)  # streaming + capped
    man_sp = scan_swissprot(D_SWISSPROT)

    if len(man_pn) == 0:
        print("No ProteinNet FASTA+PDB found under", D_PROTEINNET)

    tr_pn, va_pn = split_manifest(man_pn) if man_pn else ([],[])
    tr_af, va_af = split_manifest(man_af) if man_af else ([],[])

    feat_dim = infer_feat_dim_quick()

    # ---------------- Train sequence models (optional) ----------------
    if args.train_seq:
        if not tr_pn:
            print("[WARN] Cannot train seq models: ProteinNet manifest empty.")
        else:
            ds_tr = SequenceDataset(tr_pn, training=True)
            ds_va = SequenceDataset(va_pn, training=False)
            dl_tr = DataLoader(ds_tr, batch_size=4, shuffle=True)
            dl_va = DataLoader(ds_va, batch_size=4)
            m_seq = SeqGen_CNNBiLSTM(feat_dim)
            m_gru = TimeNetLike_GRU(feat_dim)
            ck_seq = train_seq_model(m_seq, dl_tr, dl_va, epochs=args.seq_epochs, lr=1e-3, name="seqgen_cnnlstm")
            ck_gru = train_seq_model(m_gru, dl_tr, dl_va, epochs=args.seq_epochs, lr=1e-3, name="timenet_gru")
            print("Seq models saved:", ck_seq, ck_gru)

    # ---------------- Train 2D/3D judges (optional) ----------------
    den2d_path = CHECKPOINT_DIR/"densenet2d.pt"
    nas2d_path = CHECKPOINT_DIR/"nasnet2d.pt"
    vol3d_path = CHECKPOINT_DIR/"vol3d.pt"

    if args.train_densenet2d or args.train_nasnet2d:
        if not tr_af:
            print("[WARN] AFDB set empty; 2D scorers need PDBs with B-factors.")
        else:
            # cap for speed; adjust 300 if you want more/less per epoch
            data_2d_tr = ds_contact_reg(tr_af[:min(300, len(tr_af))])
            data_2d_va = ds_contact_reg(va_af[:min(300, len(va_af))]) if va_af else []
            if args.train_densenet2d:
                den2d = DenseNet2DScorer(1)
                train_2d_scorer(den2d, data_2d_tr, val_data=data_2d_va, epochs=args.d2_epochs, name="densenet2d")
            if args.train_nasnet2d:
                nas2d = NASNet2DScorer(1)
                train_2d_scorer(nas2d, data_2d_tr, val_data=data_2d_va, epochs=args.d2_epochs, name="nasnet2d")

    if args.train_vol3d:
        if not tr_af:
            print("[WARN] AFDB set empty; 3D scorer needs PDBs.")
        else:
            data_3d_tr = ds_voxel_reg(tr_af[:min(300, len(tr_af))])
            data_3d_va = ds_voxel_reg(va_af[:min(300, len(va_af))]) if va_af else []
            vol3d = Volume3DScorer()
            train_3d_scorer(vol3d, data_3d_tr, val_data=data_3d_va, epochs=args.d3_epochs, name="vol3d")

    # ---------------- Train distogram predictor (optional) ----------------
    dist_path = CHECKPOINT_DIR/"dist_densenetU.pt"
    if args.train_dist:
        if not tr_pn:
            print("[WARN] Cannot train distogram: ProteinNet manifest empty.")
        else:
            dist_model = DistogramDenseNetU(c_in=3, n_bins=63)
            ck_dist, bin_centers = train_distogram(dist_model, tr_pn[:200], va_pn[:50], Lmax=256, epochs=args.dist_epochs, bs=args.dist_bs, name="dist_densenetU")
            print("Distogram model saved:", ck_dist)

    # ---------------- Predict/generate (optional) ----------------
    if args.predict:
        # Load seq ensemble (pretrained)
        m_seq = SeqGen_CNNBiLSTM(feat_dim)
        m_gru = TimeNetLike_GRU(feat_dim)
        seq_ck = CHECKPOINT_DIR/"seqgen_cnnlstm.pt"
        gru_ck = CHECKPOINT_DIR/"timenet_gru.pt"
        if seq_ck.exists():
            m_seq.load_state_dict(torch.load(seq_ck, map_location=DEVICE))
        else:
            print(f"[WARN] Missing {seq_ck}, generation will be poor.")
        if gru_ck.exists():
            m_gru.load_state_dict(torch.load(gru_ck, map_location=DEVICE))
        else:
            print(f"[WARN] Missing {gru_ck}, generation will be poor.")
        seq_ensemble = [m_seq, m_gru]

                # Load judges
        den2d = DenseNet2DScorer(1)
        nas2d = NASNet2DScorer(1)
        vol3d = Volume3DScorer()
        if den2d_path.exists(): den2d.load_state_dict(torch.load(den2d_path, map_location=DEVICE))
        if nas2d_path.exists(): nas2d.load_state_dict(torch.load(nas2d_path, map_location=DEVICE))
        if vol3d_path.exists(): vol3d.load_state_dict(torch.load(vol3d_path, map_location=DEVICE))
        # ---> ensure on device for inference
        den2d = den2d.to(DEVICE).eval()
        nas2d = nas2d.to(DEVICE).eval()
        vol3d = vol3d.to(DEVICE).eval()

        # Load distogram model + centers
        dist_model = DistogramDenseNetU(c_in=3, n_bins=63)
        if dist_path.exists():
            dist_model.load_state_dict(torch.load(dist_path, map_location=DEVICE))
        # ---> ensure on device
        dist_model = dist_model.to(DEVICE).eval()

        # bin centers on same device
        bin_centers = torch.tensor(
            0.5*(np.linspace(2.0,20.0,64)[1:] + np.linspace(2.0,20.0,64)[:-1]),
            dtype=torch.float32, device=DEVICE
        )

        # Example constrained generation
        user_residues = list("CCHH")
        length = int(args.gen_len); seed = args.seed_seq
        fixed = {}
        step  = max(1, length // (len(user_residues)+1))
        pos = step
        for a in user_residues:
            fixed[pos] = a; pos += step

        gen_seq = generate_sequence(seq_ensemble, seed=seed, steps=length-len(seed),
                                    T=0.9, fixed_positions=fixed, forbid=set(), must_include=None)
        print("Generated sequence (first 80):", gen_seq[:80], "...")

        coords, Dhat = predict_pdb_from_sequence(gen_seq, dist_model.to(DEVICE), bin_centers, Lmax=256, out_path="pred_ca_tmp.pdb")
        proxy = proxy_plddt_for_pred(coords, gen_seq, den2d, nas2d, vol3d)
        out_pdb = "generated_predicted.pdb"
        write_ca_pdb(gen_seq, coords, out_pdb, proxy_plddt=proxy)
        print(f"Wrote predicted PDB: {out_pdb}  (proxy pLDDT ~ {proxy:.1f})")

        st = plddt_stats_from_pdb(out_pdb)
        if st:
            print("Proxy pLDDT stats:", st)
            print("Validity% (proxy):", validity_percent(st))
        try:
            render_pymol_png(
    pdb_path="generated_predicted.pdb",
    out_png=r'C:\Users\Anvay Borade\Downloads',
    pymol_exe=r'C:\Users\Anvay Borade\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\PyMOL (Anaconda3 (64-bit))\PyMOL.lnk')

            print("Rendered PyMOL image -> generated_predicted.png")
        except Exception as e:
            print("PyMOL render skipped:", e)

if __name__ == "__main__":
    main()