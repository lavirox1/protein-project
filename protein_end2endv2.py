#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Protein Project — Simple, Single-File Pipeline (Beginner-Friendly)

What this file does (high-level):
1) Loads/embeds sequences (optionally via a PLM; one-hot fallback)
2) Uses a simple GRU to generate a new amino-acid sequence
3) Predicts a distogram (pairwise residue distances) with a DenseNet-U (2D)
4) Converts the distogram to 3D Cα coordinates (via classical MDS + a tiny smoother)
5) Reconstructs an idealized backbone (N, CA, C, O) from the Cα trace
6) Scores the structure with simple 2D/3D "judges" (optional, if checkpoints exist)
7) Writes a PDB file to disk (full backbone or CA-only)

Notes:
- We removed the CNN+BiLSTM model (only GRU remains).
- We removed PyMOL/py3Dmol rendering entirely.
- We added `--max-len`, full-backbone writer (with occupancy & B-factor), and weighted ensemble.
- Your older .pt checkpoints for GRU, 2D/3D judges, and Distogram DenseNet-U still work (we didn't change those nets).
"""

# ==============================
# Imports
# ==============================
import os, json, math, random, argparse, gzip
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    # For reading FASTA files easily; you can replace with a tiny FASTA reader if you prefer.
    from Bio import SeqIO
except ImportError:
    raise SystemExit("Please install Biopython: pip install biopython")

# Some models rely on timm backbones (DenseNet, NASNet)
try:
    import timm
except ImportError:
    raise SystemExit("Please install timm: pip install timm")

# ==============================
# Global Config / Constants
# ==============================
AMINO = "ACDEFGHIKLMNPQRSTVWY"
AA2IDX = {a: i for i, a in enumerate(AMINO)}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# These are folders you probably already have from your setup
D_PROTEINNET = Path("data/casp7_assets")
D_AFDB       = Path("data/afdb_extracted")
D_SWISSPROT  = Path("data/swissprot")
CHECKPOINT_DIR = Path("checkpoints"); CHECKPOINT_DIR.mkdir(exist_ok=True)

# PLM name (optional). If unavailable or you set --no-plm, we will use one-hot embeddings instead.
PLM_NAME = "Rostlab/prot_bert"

# Simple AA1->AA3 mapping for PDB writing
AA1_TO_AA3 = {
    "A":"ALA","C":"CYS","D":"ASP","E":"GLU","F":"PHE","G":"GLY","H":"HIS",
    "I":"ILE","K":"LYS","L":"LEU","M":"MET","N":"ASN","P":"PRO","Q":"GLN",
    "R":"ARG","S":"SER","T":"THR","V":"VAL","W":"TRP","Y":"TYR"
}

# ==============================
# Small Utility Functions
# ==============================

def set_seeds(seed: int = 42):
    """Make things reproducible-ish."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def _open_text(path: str):
    """Open normal or .gz text files."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", errors="ignore")
    return open(path, "r", errors="ignore")

def _safe_float(x, default=None):
    """Try to read a number from a string; return default on failure."""
    try:
        return float(x)
    except Exception:
        return default

# ==============================
# Loose PDB Readers (Cα + simple atoms)
# ==============================

def read_ca_coords_loose(pdb_path: str) -> np.ndarray:
    """
    Read only Cα (CA) coordinates from a PDB (or .pdb.gz).
    We don't depend on strict column widths; we try to parse robustly.
    Returns: (N,3) float32 array. Empty array if not found.
    """
    coords = []
    with _open_text(pdb_path) as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if line[12:16].strip() != "CA":
                continue
            # Try fixed columns first
            x = _safe_float(line[30:38]); y = _safe_float(line[38:46]); z = _safe_float(line[46:54])
            # Fallback to tokenized parse if needed
            if x is None or y is None or z is None:
                parts = line.split()
                nums = [_safe_float(tok) for tok in parts if _safe_float(tok) is not None]
                if len(nums) >= 3:
                    x, y, z = nums[-3], nums[-2], nums[-1]
                else:
                    continue
            coords.append((x, y, z))
    if not coords:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(coords, dtype=np.float32)

def plddt_stats_from_pdb(pdb_path: str) -> Optional[dict]:
    """
    Read B-factors of the CA atoms and compute quick stats.
    We use this as a proxy for pLDDT if your judges wrote it there.
    """
    vals = []
    with _open_text(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            b = _safe_float(line[60:66])
            if b is None:
                # Try tokenized fallback
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
    return dict(
        mean=float(p.mean()),
        median=float(np.median(p)),
        pct_ge_70=float((p >= 70).mean() * 100.0),
        pct_lt_50=float((p < 50).mean() * 100.0),
        length=int(p.size),
    )

# ==============================
# Contact Map / Voxel (simple, local uses)
# ==============================

def contact_map_from_coords(P: np.ndarray, cutoff: float = 8.0) -> np.ndarray:
    """
    Build a simple contact map (1 if distance<cutoff else 0) from a (N,3) array.
    """
    L = P.shape[0]
    C = np.zeros((L, L), dtype=np.float32)
    if L == 0:
        return C
    dif = P[:, None, :] - P[None, :, :]
    D = np.linalg.norm(dif, axis=-1)
    C[:] = (D < float(cutoff)).astype(np.float32)
    return C

def simple_voxels_from_ca(P: np.ndarray, box: int = 32, spacing: float = 1.5) -> np.ndarray:
    """
    Very simple occupancy voxel grid around the centered CA coordinates.
    Only single channel (Cα as 'C') to keep it beginner-friendly.
    """
    if P.size == 0:
        return np.zeros((1, box, box, box), dtype=np.float32)
    center = P.mean(axis=0)
    half = (box // 2) * spacing
    V = np.zeros((1, box, box, box), dtype=np.float32)
    for p in (P - center):
        if np.any(np.abs(p) > half):
            continue
        idx = np.clip(((p + half) / spacing).astype(int), 0, box - 1)
        V[0, idx[0], idx[1], idx[2]] = 1.0
    return V

# ==============================
# PLM (optional) and one-hot embeddings
# ==============================
tokenizer = None
plm_model = None

def init_plm(use_plm: bool, name: str):
    """Loads a transformer protein model if requested; else leaves globals None."""
    global tokenizer, plm_model
    if not use_plm:
        tokenizer, plm_model = None, None
        return
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(name, do_lower_case=False)
    plm_model = AutoModel.from_pretrained(name).to(DEVICE).eval()

def one_hot(seq: str, max_len: int) -> torch.Tensor:
    """Simple one-hot encoding [max_len, 20]."""
    x = torch.zeros(max_len, len(AMINO))
    for i, a in enumerate(seq[:max_len]):
        if a in AA2IDX:
            x[i, AA2IDX[a]] = 1.0
    return x

@torch.no_grad()
def embed_sequence(seq: str, max_len: int) -> torch.Tensor:
    """
    Returns [max_len, F] tensor. If PLM is loaded, use it.
    Else, return one-hot features.
    """
    if plm_model is None:
        return one_hot(seq, max_len)
    spaced = " ".join(list(seq[:max_len]))
    t = tokenizer(spaced, return_tensors="pt", add_special_tokens=True)
    t = {k: v.to(DEVICE) for k, v in t.items()}
    out = plm_model(**t).last_hidden_state[:, 1:-1, :].squeeze(0)[:max_len]
    # pad to max_len if needed
    if out.size(0) < max_len:
        pad = torch.zeros(max_len - out.size(0), out.size(1), device=out.device)
        out = torch.cat([out, pad], dim=0)
    return out

def infer_feat_dim_quick(max_len: int) -> int:
    """Small helper to figure embedding dimension once."""
    x = embed_sequence("M" * 8, max_len)
    return int(x.shape[-1])

# ==============================
# Simple Sequence Dataset (for GRU training)
# ==============================

class SequenceDataset(Dataset):
    """
    Very small dataset wrapper for (FASTA path, optional PDB path).
    Only used when training the GRU.
    """
    def __init__(self, manifest, max_len: int, training: bool = True):
        self.manifest = manifest
        self.max_len = max_len
        self.training = training

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, i):
        fasta, pdb = self.manifest[i]
        seq = str(next(SeqIO.parse(fasta, "fasta")).seq)[:self.max_len]

        # inputs = embedded sequence
        X = embed_sequence(seq, self.max_len)   # [max_len, F]

        # targets = next-token (teacher forcing)
        idxs = [AA2IDX.get(a, 0) for a in seq]
        y = torch.zeros(self.max_len, dtype=torch.long)
        if len(idxs) > 0:
            roll = idxs[1:] + [idxs[-1]]
            y[:len(roll)] = torch.tensor(roll, dtype=torch.long)
        return X, y

# ==============================
# Simple GRU Sequence Model
# ==============================

class TimeNetLike_GRU(nn.Module):
    """
    Beginner-friendly GRU:
    - Project features to a fixed size (256)
    - Bidirectional GRU layers
    - Linear head to 20 AA logits
    """
    def __init__(self, feat_dim: int, hid: int = 384, layers: int = 2, out_classes: int = 20):
        super().__init__()
        self.proj = nn.Linear(feat_dim, 256)
        self.gru  = nn.GRU(256, hid, num_layers=layers, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hid * 2, out_classes)

    def forward(self, X):  # X: [B, L, F]
        x = F.relu(self.proj(X))     # [B, L, 256]
        y, _ = self.gru(x)           # [B, L, 2*hid]
        return self.head(y)          # [B, L, 20]

def train_seq_model(model, loader, val_loader=None, epochs=3, lr=1e-3, name="timenet_gru"):
    """Simplified training loop for the GRU model."""
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = 1e9
    path = CHECKPOINT_DIR / f"{name}.pt"
    for ep in range(1, epochs + 1):
        model.train()
        losses, accs = [], []
        pbar = tqdm(loader, desc=f"[{name}] ep{ep}")
        for X, y in pbar:
            X = X.to(DEVICE).float().unsqueeze(0) if X.dim() == 2 else X.to(DEVICE).float()
            y = y.to(DEVICE)
            logits = model(X)                 # [B, L, 20]
            B, L, C = logits.shape
            loss = F.cross_entropy(logits.reshape(-1, C), y.reshape(-1))
            with torch.no_grad():
                acc = (logits.argmax(-1) == y).float().mean().item()
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item()); accs.append(acc)
            pbar.set_postfix(loss=np.mean(losses), acc=np.mean(accs))
        # save last (or best if val given)
        if val_loader:
            v = eval_seq_loss(model, val_loader)
            if v < best: best = v; torch.save(model.state_dict(), path)
        else:
            torch.save(model.state_dict(), path)
    print(f"Saved GRU -> {path}")
    return path

@torch.no_grad()
def eval_seq_loss(model, loader):
    """Validation loss helper."""
    model.eval().to(DEVICE)
    losses = []
    for X, y in loader:
        X = X.to(DEVICE).float().unsqueeze(0) if X.dim() == 2 else X.to(DEVICE).float()
        y = y.to(DEVICE)
        logits = model(X)
        B, L, C = logits.shape
        loss = F.cross_entropy(logits.reshape(-1, C), y.reshape(-1))
        losses.append(loss.item())
    return float(np.mean(losses))

# ==============================
# Distogram (U-Net-like) — same idea, beginner-wrapped
# ==============================

class DistogramDenseNetU(nn.Module):
    """
    This predicts distogram logits: [n_bins, L, L]
    We feed 3 channels of pairwise features (simple mean-reduced).
    """
    def __init__(self, c_in=3, n_bins=63):
        super().__init__()
        self.backbone = timm.create_model(
            "densenet121", pretrained=True, features_only=True, out_indices=(0,1,2,3)
        )
        self.proj_in = nn.Conv2d(c_in, 3, kernel_size=1)
        chs = self.backbone.feature_info.channels()  # e.g., [64,128,256,1024]
        self.up3 = nn.ConvTranspose2d(chs[-1], 256, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(256 + chs[2], 128, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(128 + chs[1], 64,  2, stride=2)
        self.up0 = nn.ConvTranspose2d(64 + chs[0], 64,   2, stride=2)
        self.head = nn.Conv2d(64, n_bins, 1)

    def _resize(self, x, ref):
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, x):  # x: [B, 3, L, L]
        B, C, H, W = x.shape
        x = self.proj_in(x)

        # Pad to multiple of 16 for nicer down/up alignment
        H0, W0 = H, W
        ph = (16 - H % 16) % 16
        pw = (16 - W % 16) % 16
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))

        f0, f1, f2, f3 = self.backbone(x)
        u3 = self._resize(self.up3(f3), f2)
        u2 = self._resize(self.up2(torch.cat([u3, f2], dim=1)), f1)
        u1 = self._resize(self.up1(torch.cat([u2, f1], dim=1)), f0)
        u0 = self.up0(torch.cat([u1, f0], dim=1))
        out = self.head(u0)

        # Crop back to original size
        if ph or pw:
            out = out[..., :H0, :W0]
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return out  # [B, n_bins, L, L]

def pairwise_features_simple(res_emb: torch.Tensor) -> torch.Tensor:
    """
    Build easy pairwise features from per-residue embeddings.
    We just take: [Ei, Ej, |Ei-Ej|, Ei*Ej] -> mean-reduced to 3 channels.
    """
    L, Fd = res_emb.shape
    Ei = res_emb.unsqueeze(1)          # [L,1,F]
    Ej = res_emb.unsqueeze(0)          # [1,L,F]
    diff = (Ei - Ej).abs()
    prod = Ei * Ej
    pw = torch.cat([Ei.expand(-1, L, -1),
                    Ej.expand(L, -1, -1),
                    diff, prod], dim=-1)  # [L,L,4F]
    # Reduce to 3 channels by simple chunked mean
    C_in = 3
    # pad to be divisible by 3
    pad = (C_in - (pw.shape[-1] % C_in)) % C_in
    if pad:
        pw = torch.cat([pw, torch.zeros(L, L, pad, device=pw.device)], dim=-1)
    pw = pw.view(L, L, C_in, -1).mean(-1)  # [L, L, 3]
    return pw.permute(2, 0, 1)             # [3, L, L]

def expected_dist_matrix(logits: torch.Tensor, bin_centers: torch.Tensor) -> torch.Tensor:
    """Convert logits [n_bins,L,L] to expected distances [L,L]."""
    probs = torch.softmax(logits, dim=0)
    D = (probs * bin_centers.view(-1, 1, 1).to(logits.device)).sum(0)
    D = 0.5 * (D + D.t())
    D.fill_diagonal_(0.0)
    return D

def cmds_from_dist(D: torch.Tensor) -> torch.Tensor:
    """
    Classical MDS from a distance matrix -> 3D coords (Cα).
    Returns [L, 3] tensor.
    """
    A = D.detach().cpu().numpy()
    n = A.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J.dot(A ** 2).dot(J)
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1][:3]
    Lm = np.diag(np.sqrt(np.maximum(eigvals[idx], 0)))
    X = eigvecs[:, idx].dot(Lm)
    return torch.tensor(X, dtype=torch.float32)

def smooth_refine(coords: torch.Tensor, iters: int = 200, lr: float = 5e-3) -> torch.Tensor:
    """
    Tiny smoothing/refinement on Cα coords to encourage 3.8 Å bonds and smoothness.
    """
    coords = coords.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([coords], lr=lr)
    for _ in range(iters):
        with torch.enable_grad():
            diffs = coords[1:] - coords[:-1]
            bond  = (diffs.norm(dim=1) - 3.8) ** 2
            smooth = ((coords[2:] - 2 * coords[1:-1] + coords[:-2]) ** 2).sum(dim=1) if coords.size(0) > 2 else torch.tensor(0.0, device=coords.device)
            loss = 10.0 * bond.mean() + 0.1 * (smooth.mean() if coords.size(0) > 2 else 0.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return coords.detach()

# ==============================
# Full-Backbone Reconstruction (N, CA, C, O)
# ==============================

def _safe_norm(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + eps)

def reconstruct_backbone_from_ca(ca: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Build simple idealized backbone atoms from a Cα trace using local tangents.
    This is intentionally simple (beginner-friendly), not a perfect peptide geometry.
    Returns dict with 'N','CA','C','O' arrays of shape [L,3].
    """
    L = ca.shape[0]
    if L == 0:
        z = np.zeros((0, 3), dtype=np.float32)
        return {"N": z, "CA": z, "C": z, "O": z}
    if L == 1:
        return {"N": ca.copy(), "CA": ca.copy(), "C": ca.copy(), "O": ca.copy()}

    # Ideal bond lengths (Å)
    d_CA_N = 1.458
    d_CA_C = 1.525
    d_C_O  = 1.231

    # Build rough tangents and normals
    ca_pad = np.vstack([ca[0], ca, ca[-1]])
    t = np.zeros_like(ca)
    n = np.zeros_like(ca)
    b = np.zeros_like(ca)

    for i in range(L):
        fwd = ca_pad[i + 2] - ca_pad[i + 1]
        bwd = ca_pad[i + 1] - ca_pad[i]
        tt = fwd + bwd
        t[i] = _safe_norm(tt if np.linalg.norm(tt) > 1e-6 else (fwd if i < L - 1 else bwd))

    for i in range(L):
        if 0 < i < L - 1:
            curv = ca[i + 1] - 2 * ca[i] + ca[i - 1]
        else:
            curv = (ca[min(i + 1, L - 1)] - ca[max(i - 1, 0)])
        curv = curv - np.dot(curv, t[i]) * t[i]
        # pick a fallback normal if curv≈0
        if np.linalg.norm(curv) < 1e-6:
            fallback = np.array([1.0, 0.0, 0.0]) if abs(t[i][0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            curv = np.cross(t[i], fallback)
        n[i] = _safe_norm(curv)
        b[i] = _safe_norm(np.cross(t[i], n[i]))

    # Small lifts to mimic peptide plane tilt (very rough)
    lift_N = 0.2
    lift_C = 0.2
    tilt_O = 0.3

    N = np.zeros_like(ca); CA = ca.copy(); C = np.zeros_like(ca); O = np.zeros_like(ca)
    for i in range(L):
        N[i] = CA[i] - d_CA_N * t[i] + lift_N * n[i]
        C[i] = CA[i] + d_CA_C * t[i] + lift_C * n[i]
        O[i] = C[i] - d_C_O * n[i] + tilt_O * b[i]
    return {"N": N, "CA": CA, "C": C, "O": O}

# ==============================
# Simple PDB Writers
# ==============================

def write_ca_pdb(seq: str, ca_coords: torch.Tensor, out_path: str,
                 chain_id: str = "A", occ: float = 1.00, bfactor: float = 0.0):
    """
    Write only Cα atoms to PDB. Occupancy and B-factor are written.
    """
    ca = ca_coords.detach().cpu().numpy()
    with open(out_path, "w") as f:
        atom_id = 1
        for i, (aa, xyz) in enumerate(zip(seq, ca), start=1):
            x, y, z = xyz
            resn = AA1_TO_AA3.get(aa, "UNK")
            f.write(
                f"ATOM  {atom_id:5d}  CA  {resn:>3s} {chain_id}{i:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{bfactor:6.2f}           C\n"
            )
            atom_id += 1
        f.write("TER\nEND\n")

def write_backbone_pdb(seq: str, ca_coords: torch.Tensor, out_path: str,
                       chain_id: str = "A", occ: float = 1.00,
                       bfactor: float = 0.0):
    """
    Write N, CA, C, O atoms per residue using the idealized reconstruction.
    """
    ca = ca_coords.detach().cpu().numpy()
    bb = reconstruct_backbone_from_ca(ca)
    L = min(len(seq), ca.shape[0])
    with open(out_path, "w") as f:
        atom_id = 1
        for i in range(L):
            resn = AA1_TO_AA3.get(seq[i], "UNK")
            resi = i + 1
            for atom_name in ("N", "CA", "C", "O"):
                x, y, z = bb[atom_name][i]
                elem = atom_name[0]
                f.write(
                    f"ATOM  {atom_id:5d} {atom_name:<4s}{resn:>3s} {chain_id}{resi:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{bfactor:6.2f}           {elem:>2s}\n"
                )
                atom_id += 1
        f.write("TER\nEND\n")

# ==============================
# Judges (very small)
# ==============================

def _adapt_first_conv_to_1ch(m):
    """Make the first conv 1-channel by averaging existing RGB weights if needed."""
    if getattr(m, "in_channels", 3) == 3:
        w = m.weight.data
        m.weight = nn.Parameter(w.mean(dim=1, keepdim=True))

class DenseNet2DScorer(nn.Module):
    """Takes a 1-channel contact map and predicts a scalar quality score (0..1)."""
    def __init__(self, out_dim=1):
        super().__init__()
        self.backbone = timm.create_model("densenet121", pretrained=True, in_chans=1, num_classes=out_dim)
        try:
            _adapt_first_conv_to_1ch(self.backbone.features.conv0)
        except Exception:
            pass
    def forward(self, C):  # C: [B,1,L,L]
        return self.backbone(C)

class Volume3DScorer(nn.Module):
    """Takes a small 3D voxel and predicts a scalar score (0..1)."""
    def __init__(self, in_ch=1):
        super().__init__()
        self.c1 = nn.Conv3d(in_ch, 16, 3, padding=1)
        self.c2 = nn.Conv3d(16, 32, 3, padding=1)
        self.c3 = nn.Conv3d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool3d(2)
        self.gap  = nn.AdaptiveAvgPool3d(4)
        self.fc   = nn.Linear(64 * 4 * 4 * 4, 1)
    def forward(self, V):  # V: [B,1,D,H,W]
        x = F.relu(self.c1(V))
        x = self.pool(F.relu(self.c2(x)))
        x = self.pool(F.relu(self.c3(x)))
        x = self.gap(x).flatten(1)
        return self.fc(x)

@torch.no_grad()
def proxy_plddt_for_pred(ca_coords: torch.Tensor) -> float:
    """
    Very light proxy using internal geometry only, to keep it simple:
    - Build a contact map from predicted Cα coords
    - Score with a 2D net if checkpoint exists (optional)
    - Also build a tiny voxel and score with a 3D net if checkpoint exists (optional)
    If checkpoints are missing, returns 70.0 as a neutral default.
    """
    den2d_path = CHECKPOINT_DIR / "densenet2d.pt"
    vol3d_path = CHECKPOINT_DIR / "vol3d.pt"

    P = ca_coords.detach().cpu().numpy()
    if P.shape[0] < 2:
        return 50.0

    # Start with a neutral score
    fallback = 70.0
    scores = []

    # 2D score (contact map)
    if den2d_path.exists():
        den2d = DenseNet2DScorer(1).to(DEVICE).eval()
        den2d.load_state_dict(torch.load(den2d_path, map_location=DEVICE))
        C = torch.tensor(contact_map_from_coords(P), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        s2d = torch.sigmoid(den2d(C)).item()
        scores.append(s2d)

    # 3D score (simple voxel of Cα)
    if vol3d_path.exists():
        vol3d = Volume3DScorer(1).to(DEVICE).eval()
        V = torch.tensor(simple_voxels_from_ca(P), dtype=torch.float32).unsqueeze(0).to(DEVICE)
        s3d = torch.sigmoid(vol3d(V)).item()
        scores.append(s3d)

    if not scores:
        return fallback
    # Combine into "pLDDT-like" number
    return float(100.0 * np.clip(np.mean(scores), 0.0, 1.0))

# ==============================
# Distogram helpers (bins)
# ==============================

def get_default_bin_centers(n_bins: int = 63) -> torch.Tensor:
    """Centers for 63 bins in [2.0, 20.0] Å."""
    edges = np.linspace(2.0, 20.0, n_bins + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    return torch.tensor(centers, dtype=torch.float32, device=DEVICE)

# ==============================
# Weighted Ensemble for Next Token (future-proof)
# ==============================

def parse_weights(s: Optional[str], n: int) -> List[float]:
    """Convert '0.6,0.4' -> normalized list. Fall back to equal weights."""
    if not s:
        return [1.0 / n] * n
    w = [float(x) for x in s.split(",") if x.strip()]
    if len(w) != n:
        print(f"[WARN] ens-weights had {len(w)} entries but {n} models given; using equal weights")
        return [1.0 / n] * n
    ssum = sum(w)
    if ssum <= 0:
        return [1.0 / n] * n
    return [x / ssum for x in w]

@torch.no_grad()
def ensemble_next_logits_weighted(models: List[nn.Module], weights: List[float], prefix: str, max_len: int) -> torch.Tensor:
    """
    Embed the prefix, run each model, take the last-position logits, and
    combine with the given weights. Returns [20] logits.
    """
    X = embed_sequence(prefix, max_len).unsqueeze(0).to(DEVICE).float()  # [1, L, F]
    out = None
    for m, w in zip(models, weights):
        m = m.to(DEVICE).eval()
        logits = m(X)[:, -1, :]  # [1,20]
        out = logits * w if out is None else out + logits * w
    return out.squeeze(0)  # [20]

def softmax_T(logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    """Temperature softmax."""
    z = logits / max(T, 1e-6)
    z = z - z.max()
    return torch.softmax(z, dim=-1)

def generate_sequence(models: List[nn.Module], weights: List[float],
                      seed: str, total_len: int, max_len: int,
                      T: float = 0.9) -> str:
    """
    Auto-regressive generation with a tiny ensemble (works fine with a single GRU).
    """
    seq = list(seed)
    steps = max(0, total_len - len(seq))
    for _ in range(steps):
        logits = ensemble_next_logits_weighted(models, weights, "".join(seq), max_len)
        probs = softmax_T(logits, T=T).cpu().numpy()
        idx = np.random.choice(len(AMINO), p=probs)
        seq.append(AMINO[idx])
    return "".join(seq[:total_len])

# ==============================
# Predict: seq -> distogram -> coords -> PDB
# ==============================

@torch.no_grad()
def predict_structure_and_write_pdb(seq: str, dist_model: nn.Module,
                                    bin_centers: torch.Tensor,
                                    max_len: int,
                                    out_pdb: str,
                                    full_backbone: bool,
                                    occ: float,
                                    bfactor: Optional[float]) -> Tuple[torch.Tensor, float]:
    """
    Full pipeline for a single sequence:
    1) Embed sequence
    2) Build pairwise features and run distogram model
    3) Convert to distances, then to 3D Cα coords
    4) Smooth a bit
    5) Write CA-only or full backbone PDB
    6) Return coords and proxy score used as B-factor if needed
    """
    L = min(len(seq), max_len)
    seq = seq[:L]

    # pairwise features -> [1,3,L,L]
    emb = embed_sequence(seq, L)                    # [L, F]
    pw  = pairwise_features_simple(emb.to(DEVICE))  # [3, L, L]
    X   = pw.unsqueeze(0)                           # [1, 3, L, L]

    # distogram logits -> expected distances -> coords
    dist_model = dist_model.to(DEVICE).eval()
    logits = dist_model(X).squeeze(0)               # [n_bins, L, L]
    Dhat   = expected_dist_matrix(logits, bin_centers)  # [L, L]
    X0     = cmds_from_dist(Dhat)                   # [L, 3]
    coords = smooth_refine(X0, iters=200, lr=5e-3)  # small smoothing

    # get a proxy score if B-factor not forced
    proxy = proxy_plddt_for_pred(coords) if bfactor is None else bfactor
    b_use = float(proxy)

    # write PDB
    if full_backbone:
        write_backbone_pdb(seq, coords, out_pdb, chain_id="A", occ=occ, bfactor=b_use)
    else:
        write_ca_pdb(seq, coords, out_pdb, chain_id="A", occ=occ, bfactor=b_use)

    return coords, proxy

# ==============================
# Simple dataset scanning (optional)
# ==============================

def _possible_fasta_for(p: Path) -> Optional[str]:
    """Try to find a sibling .fasta file for a given .pdb or .pdb.gz path."""
    cand1 = p.with_suffix(".fasta")
    if cand1.exists():
        return str(cand1)
    if "".join(p.suffixes[-2:]).lower() in (".pdb.gz",):
        cand2 = p.with_suffix("").with_suffix(".fasta")
        if cand2.exists():
            return str(cand2)
    return None

def scan_proteinnet(root: Path):
    """Return list of (fasta, pdb_ca) pairs if present."""
    items = []
    fasta_dir = root / "fasta"
    pdb_dir   = root / "pdb_ca"
    if fasta_dir.exists() and pdb_dir.exists():
        fasta_map = {p.stem: p for p in fasta_dir.glob("*.fasta")}
        pdb_map   = {p.stem: p for p in pdb_dir.glob("*.pdb")}
        for k in sorted(set(fasta_map).intersection(pdb_map)):
            items.append((str(fasta_map[k].resolve()), str(pdb_map[k].resolve())))
    return items

def scan_afdb(root: Path, max_files: Optional[int] = None):
    """Stream AFDB .pdb or .pdb.gz files; try to find sibling FASTAs when possible."""
    if not root.exists():
        return []
    items = []
    for i, p in enumerate(root.rglob("*")):
        if not p.is_file():
            continue
        name = p.name.lower()
        if not (name.endswith(".pdb") or name.endswith(".pdb.gz")):
            continue
        fasta = _possible_fasta_for(p)
        items.append((fasta, str(p)))
        if max_files is not None and len(items) >= max_files:
            break
    return items

# ==============================
# CLI and Main
# ==============================

def main():
    parser = argparse.ArgumentParser(description="Protein Project — Simple Pipeline")
    # Switch PLM off if you want faster/no-internet runs (one-hot only)
    parser.add_argument("--no-plm", action="store_true", help="Disable PLM; use one-hot embeddings")

    # Training toggles (GRU / Judges / Distogram)
    parser.add_argument("--train-gru", action="store_true", help="Train only the GRU sequence model")
    parser.add_argument("--train-densenet2d", action="store_true")
    parser.add_argument("--train-vol3d", action="store_true")
    parser.add_argument("--train-dist", action="store_true")

    # Prediction
    parser.add_argument("--predict", action="store_true", help="Generate sequence and predict structure")

    # Core hyperparams
    parser.add_argument("--seq-epochs", type=int, default=3)
    parser.add_argument("--d2-epochs", type=int, default=2)
    parser.add_argument("--d3-epochs", type=int, default=2)
    parser.add_argument("--dist-epochs", type=int, default=2)

    # Generation controls
    parser.add_argument("--seed-seq", type=str, default="M", help="Starting residues")
    parser.add_argument("--gen-len", type=int, default=200, help="Total length to generate")
    parser.add_argument("--ens-weights", type=str, default=None, help='Comma-separated model weights, e.g. "1.0"')

    # Length handling (NEW)
    parser.add_argument("--max-len", type=int, default=256, help="Global cap for embeddings and tensors")

    # Output PDB choices (NEW)
    parser.add_argument("--full-backbone", action="store_true", help="Write N, CA, C, O instead of CA-only")
    parser.add_argument("--occ", type=float, default=1.00, help="PDB occupancy for all atoms")
    parser.add_argument("--bfactor", type=float, default=None, help="Fixed B-factor (if omitted, we use a proxy)")

    # AFDB scan cap (to avoid huge scans)
    parser.add_argument("--afdb-limit", type=int, default=1000)

    args = parser.parse_args()

    # honor max-len and random seeds
    MAX_LEN = int(args.max_len)
    set_seeds(42)

    # init PLM or not
    init_plm(use_plm=(not args.no_plm), name=PLM_NAME)

    # manifests (optional for training)
    man_pn = scan_proteinnet(D_PROTEINNET)
    man_af = scan_afdb(D_AFDB, max_files=args.afdb_limit)

    # figure feature dimension for GRU
    feat_dim = infer_feat_dim_quick(MAX_LEN)

    # ---------------- Train GRU (optional) ----------------
    if args.train_gru:
        if not man_pn:
            print("[WARN] No ProteinNet data found; GRU training skipped.")
        else:
            ds_tr = SequenceDataset(man_pn, max_len=MAX_LEN, training=True)
            dl_tr = DataLoader(ds_tr, batch_size=1, shuffle=True)  # small and simple
            m_gru = TimeNetLike_GRU(feat_dim)
            train_seq_model(m_gru, dl_tr, val_loader=None, epochs=args.seq_epochs, lr=1e-3, name="timenet_gru")

    # ---------------- Train judges (optional; simple versions) ----------------
    # Tip: These examples require crafting small (pdb, score) datasets; for brevity we skip full loops here.
    if args.train_densenet2d:
        print("[Info] Training DenseNet2D is not fully wired to a dataset here (kept simple).")
        # You can create small (C-map -> score) pairs and call a similar training loop if needed.

    if args.train_vol3d:
        print("[Info] Training Volume3D is not fully wired to a dataset here (kept simple).")
        # Same note as above.

    # ---------------- Train distogram (optional) ----------------
    if args.train_dist:
        print("[Info] Distogram training loop omitted for brevity in this beginner file.")
        # You can reuse your earlier training code if needed; model definition is unchanged.

    # ---------------- Predict / Generate ----------------
    if args.predict:
        # 1) Load GRU (required for generation)
        gru_ck = CHECKPOINT_DIR / "timenet_gru.pt"
        m_gru = TimeNetLike_GRU(feat_dim)
        if gru_ck.exists():
            m_gru.load_state_dict(torch.load(gru_ck, map_location=DEVICE))
        else:
            print(f"[WARN] Missing {gru_ck}; generation quality will be poor if untrained.")

        # Keep an ensemble list (even if single model) + weights
        seq_models = [m_gru]
        ens_w = parse_weights(args.ens_weights, len(seq_models))

        # 2) Make a sequence
        total_len = int(args.gen_len)
        seed = args.seed_seq
        gen_seq = generate_sequence(seq_models, ens_w, seed=seed, total_len=total_len, max_len=MAX_LEN, T=0.9)
        print("Generated sequence (first 80):", gen_seq[:80], "... len=", len(gen_seq))

        # 3) Load distogram model + bin centers
        dist_path = CHECKPOINT_DIR / "dist_densenetU.pt"
        dist_model = DistogramDenseNetU(c_in=3, n_bins=63)
        if dist_path.exists():
            dist_model.load_state_dict(torch.load(dist_path, map_location=DEVICE))
        else:
            print(f"[WARN] Missing {dist_path}; structure prediction will be poor if untrained.")
        bin_centers = get_default_bin_centers(63)

        # 4) Predict structure and write PDB (no rendering)
        out_pdb = "generated_predicted.pdb"
        coords, proxy = predict_structure_and_write_pdb(
            gen_seq, dist_model, bin_centers, MAX_LEN,
            out_pdb, full_backbone=args.full_backbone, occ=args.occ, bfactor=args.bfactor
        )
        print(f"Wrote PDB -> {out_pdb} (B-factor used ~ {proxy:.1f})")

        # 5) Quick stats from written file
        st = plddt_stats_from_pdb(out_pdb)
        if st:
            print("B-factor stats on CA (proxy pLDDT style):", st)

if __name__ == "__main__":
    main()