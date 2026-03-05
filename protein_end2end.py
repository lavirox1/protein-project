#!/usr/bin/env python3
import random, argparse
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from collections import Counter

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

AMINO      = "ACDEFGHIKLMNPQRSTVWY"
AA2IDX     = {a: i for i, a in enumerate(AMINO)}
AA1_TO_AA3 = {
    "A":"ALA","C":"CYS","D":"ASP","E":"GLU","F":"PHE","G":"GLY","H":"HIS",
    "I":"ILE","K":"LYS","L":"LEU","M":"MET","N":"ASN","P":"PRO","Q":"GLN",
    "R":"ARG","S":"SER","T":"THR","V":"VAL","W":"TRP","Y":"TYR"
}
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = Path("checkpoints")
PLM_NAME       = "Rostlab/prot_bert"
MAX_LEN        = 512

tokenizer = None
plm_model = None


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_plm():
    global tokenizer, plm_model
    print("Loading ProtBERT (first run downloads ~420MB)...")
    try:
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(PLM_NAME, do_lower_case=False)
        plm_model = AutoModel.from_pretrained(PLM_NAME).to(DEVICE).eval()
        print(f"ProtBERT ready on {DEVICE}.")
    except Exception as e:
        raise SystemExit(f"Failed to load ProtBERT: {e}\nInstall with: pip install transformers")


@torch.no_grad()
def embed_sequence(seq: str, length: int) -> torch.Tensor:
    spaced = " ".join(list(seq[:length]))
    t = tokenizer(spaced, return_tensors="pt", add_special_tokens=True)
    t = {k: v.to(DEVICE) for k, v in t.items()}
    out = plm_model(**t).last_hidden_state[:, 1:-1, :].squeeze(0)[:length]
    if out.size(0) < length:
        pad = torch.zeros(length - out.size(0), out.size(1), device=out.device)
        out = torch.cat([out, pad], dim=0)
    return out  # [length, 1024]


# ── Models ────────────────────────────────────────────────────────────────────

class SeqGen_CNNBiLSTM(nn.Module):
    def __init__(self, feat_dim: int, hid: int = 256, out_classes: int = 20):
        super().__init__()
        self.conv = nn.Conv1d(feat_dim, 128, kernel_size=5, padding=2)
        self.bn   = nn.BatchNorm1d(128)
        self.lstm = nn.LSTM(128, hid, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hid * 2, out_classes)

    def forward(self, X):  # [B, L, F]
        x = F.relu(self.bn(self.conv(X.permute(0, 2, 1)))).permute(0, 2, 1)
        y, _ = self.lstm(x)
        return self.head(y)


class TimeNetLike_GRU(nn.Module):
    def __init__(self, feat_dim: int, hid: int = 384, layers: int = 2, out_classes: int = 20):
        super().__init__()
        self.proj = nn.Linear(feat_dim, 256)
        self.gru  = nn.GRU(256, hid, num_layers=layers, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hid * 2, out_classes)

    def forward(self, X):  # [B, L, F]
        x = F.relu(self.proj(X))
        y, _ = self.gru(x)
        return self.head(y)


class DistogramDenseNetU(nn.Module):
    def __init__(self, c_in=3, n_bins=63):
        super().__init__()
        self.backbone = timm.create_model(
            "densenet121", pretrained=False, features_only=True, out_indices=(0,1,2,3)
        )
        self.proj_in = nn.Conv2d(c_in, 3, kernel_size=1)
        chs = self.backbone.feature_info.channels()
        self.up3 = nn.ConvTranspose2d(chs[-1], 256, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(256 + chs[2], 128, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(128 + chs[1], 64,  2, stride=2)
        self.up0 = nn.ConvTranspose2d(64  + chs[0], 64,  2, stride=2)
        self.head = nn.Conv2d(64, n_bins, 1)

    def _resize(self, x, ref):
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, x):  # [B, 3, L, L]
        B, C, H, W = x.shape
        x = self.proj_in(x)
        ph = (16 - H % 16) % 16
        pw = (16 - W % 16) % 16
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
        f0, f1, f2, f3 = self.backbone(x)
        u3 = self._resize(self.up3(f3), f2)
        u2 = self._resize(self.up2(torch.cat([u3, f2], 1)), f1)
        u1 = self._resize(self.up1(torch.cat([u2, f1], 1)), f0)
        u0 = self.up0(torch.cat([u1, f0], 1))
        out = self.head(u0)
        if ph or pw:
            out = out[..., :H, :W]
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return out


class DenseNet2DScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("densenet121", pretrained=False, in_chans=1, num_classes=1)

    def forward(self, C):  # [B, 1, L, L]
        return self.backbone(C)


class Volume3DScorer(nn.Module):
    def __init__(self, in_ch=4):
        super().__init__()
        self.c1   = nn.Conv3d(in_ch, 16, 3, padding=1)
        self.c2   = nn.Conv3d(16, 32, 3, padding=1)
        self.c3   = nn.Conv3d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool3d(2)
        self.gap  = nn.AdaptiveAvgPool3d(4)
        self.fc   = nn.Linear(64 * 4 * 4 * 4, 1)

    def forward(self, V):  # [B, 4, D, H, W]
        x = F.relu(self.c1(V))
        x = self.pool(F.relu(self.c2(x)))
        x = self.pool(F.relu(self.c3(x)))
        return self.fc(self.gap(x).flatten(1))


# ── Checkpoint loading ────────────────────────────────────────────────────────

def load_seq_models() -> Tuple[List[nn.Module], List[float]]:
    cnn_path = CHECKPOINT_DIR / "seqgen_cnnlstm.pt"
    gru_path = CHECKPOINT_DIR / "timenet_gru.pt"
    models, weights = [], []

    for path, ModelClass, cls_name, dim_key in [
        (cnn_path, SeqGen_CNNBiLSTM, "CNN+BiLSTM", "conv.weight"),
        (gru_path, TimeNetLike_GRU,  "GRU",        "proj.weight"),
    ]:
        if not path.exists():
            print(f"  {cls_name}: not found, skipping.")
            continue
        try:
            ck       = torch.load(path, map_location=DEVICE)
            feat_dim = ck[dim_key].shape[1]
            m        = ModelClass(feat_dim).to(DEVICE).eval()
            m.load_state_dict(ck)
            models.append(m)
            weights.append(1.0)
            print(f"  {cls_name}: loaded")
        except Exception as e:
            print(f"  {cls_name}: failed — {e}")

    if not models:
        raise SystemExit("No sequence model checkpoints found in checkpoints/")

    total = sum(weights)
    return models, [w / total for w in weights]


# ── Sequence generation ────────────────────────────────────────────────────────

def _top_k_top_p_filter(probs: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    keep = torch.ones_like(sorted_probs, dtype=torch.bool)
    if top_k > 0:
        keep[top_k:] = False
    if top_p > 0.0:
        keep = keep & (torch.cumsum(sorted_probs, 0) <= top_p)
        keep[0] = True
    mask = torch.zeros_like(probs, dtype=torch.bool)
    mask[sorted_idx[keep]] = True
    filtered = torch.where(mask, probs, torch.zeros_like(probs))
    s = filtered.sum()
    return filtered / s if s > 0 else probs


def _apply_rep_penalty(logits: torch.Tensor, tokens: list, penalty: float) -> torch.Tensor:
    for idx, cnt in Counter(tokens).items():
        logits[idx] /= penalty ** cnt
    return logits


def _build_ngram_set(tokens: list, n: int) -> set:
    if n <= 0 or len(tokens) < n:
        return set()
    return {tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}


@torch.no_grad()
def generate_sequence(models, weights, seed, total_len, T, top_k, top_p,
                      rep_penalty, no_repeat_ngram) -> str:
    seq   = list(seed)
    steps = max(0, total_len - len(seq))

    def get_logits(prefix):
        X   = embed_sequence(prefix, min(len(prefix), MAX_LEN)).unsqueeze(0).float()
        out = None
        for m, w in zip(models, weights):
            logits = m(X)[:, -1, :]
            out    = logits * w if out is None else out + logits * w
        return out.squeeze(0)

    for _ in tqdm(range(steps), desc="Generating", unit="aa"):
        logits = get_logits("".join(seq))
        tokens = [AA2IDX[a] for a in seq if a in AA2IDX]
        logits = _apply_rep_penalty(logits.clone(), tokens, rep_penalty)
        probs  = torch.softmax(logits / max(T, 1e-6), dim=-1)
        probs  = _top_k_top_p_filter(probs, top_k, top_p)
        ngrams = _build_ngram_set(tokens, no_repeat_ngram)
        cand   = None

        for _ in range(20):
            idx  = int(torch.multinomial(probs, 1))
            gram = tuple(tokens[-(no_repeat_ngram-1):] + [idx]) if len(tokens) >= no_repeat_ngram - 1 else None
            if gram is None or gram not in ngrams:
                cand = idx
                break
            probs[idx] = 0.0
            s = probs.sum()
            if s <= 0:
                cand = int(logits.argmax())
                break
            probs = probs / s

        seq.append(AMINO[cand if cand is not None else int(logits.argmax())])

    return "".join(seq[:total_len])


# ── Structure prediction ───────────────────────────────────────────────────────

def pairwise_features(emb: torch.Tensor) -> torch.Tensor:
    L  = emb.shape[0]
    Ei = emb.unsqueeze(1)
    Ej = emb.unsqueeze(0)
    pw  = torch.cat([Ei.expand(-1,L,-1), Ej.expand(L,-1,-1), (Ei-Ej).abs(), Ei*Ej], dim=-1)
    pad = (3 - pw.shape[-1] % 3) % 3
    if pad:
        pw = torch.cat([pw, torch.zeros(L, L, pad, device=pw.device)], dim=-1)
    return pw.view(L, L, 3, -1).mean(-1).permute(2, 0, 1)  # [3, L, L]


def dist_matrix_from_logits(logits: torch.Tensor, bin_centers: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=0)
    D     = (probs * bin_centers.view(-1, 1, 1)).sum(0)
    D     = 0.5 * (D + D.t())
    D.fill_diagonal_(0.0)
    return D


def coords_from_dist(D: torch.Tensor) -> torch.Tensor:
    A          = D.cpu().numpy()
    n          = A.shape[0]
    J          = np.eye(n) - np.ones((n, n)) / n
    B          = -0.5 * J @ (A ** 2) @ J
    vals, vecs = np.linalg.eigh(B)
    idx        = np.argsort(vals)[::-1][:3]
    X          = vecs[:, idx] @ np.diag(np.sqrt(np.maximum(vals[idx], 0)))
    return torch.tensor(X, dtype=torch.float32)


def smooth_coords(coords: torch.Tensor, iters=100, lr=5e-3) -> torch.Tensor:
    coords = coords.clone().detach().requires_grad_(True)
    opt    = torch.optim.Adam([coords], lr=lr)
    for _ in range(iters):
        with torch.enable_grad():
            bond   = ((coords[1:] - coords[:-1]).norm(dim=1) - 3.8) ** 2
            smooth = ((coords[2:] - 2*coords[1:-1] + coords[:-2]) ** 2).sum(1) \
                     if coords.size(0) > 2 else torch.tensor(0.)
            loss   = 10.0 * bond.mean() + 0.1 * smooth.mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return coords.detach()


# ── Scoring ────────────────────────────────────────────────────────────────────

def contact_map(P: np.ndarray, cutoff=8.0) -> np.ndarray:
    D = np.linalg.norm(P[:, None] - P[None, :], axis=-1)
    return (D < cutoff).astype(np.float32)


def voxels_from_ca(P: np.ndarray, box=32, spacing=1.5) -> np.ndarray:
    V    = np.zeros((4, box, box, box), dtype=np.float32)
    half = (box // 2) * spacing
    for p in (P - P.mean(0)):
        if np.any(np.abs(p) > half):
            continue
        idx        = np.clip(((p + half) / spacing).astype(int), 0, box - 1)
        V[0, idx[0], idx[1], idx[2]] = 1.0
    return V


@torch.no_grad()
def score_structure(coords: torch.Tensor) -> float:
    P      = coords.cpu().numpy()
    L      = P.shape[0]
    scores = []

    den2d_path = CHECKPOINT_DIR / "densenet2d.pt"
    vol3d_path = CHECKPOINT_DIR / "vol3d.pt"

    if den2d_path.exists() and L >= 32:
        try:
            m = DenseNet2DScorer().to(DEVICE).eval()
            m.load_state_dict(torch.load(den2d_path, map_location=DEVICE))
            C = torch.tensor(contact_map(P)).unsqueeze(0).unsqueeze(0).to(DEVICE)
            scores.append(torch.sigmoid(m(C)).item())
        except Exception as e:
            print(f"  2D scorer skipped: {e}")

    if vol3d_path.exists():
        try:
            m = Volume3DScorer(in_ch=4).to(DEVICE).eval()
            m.load_state_dict(torch.load(vol3d_path, map_location=DEVICE))
            V = torch.tensor(voxels_from_ca(P)).unsqueeze(0).to(DEVICE)
            scores.append(torch.sigmoid(m(V)).item())
        except Exception as e:
            print(f"  3D scorer skipped: {e}")

    return float(100.0 * np.clip(np.mean(scores), 0, 1)) if scores else 70.0


# ── PDB writing ────────────────────────────────────────────────────────────────

def _safe_norm(v: np.ndarray, eps=1e-8) -> np.ndarray:
    return v / (np.linalg.norm(v) + eps)


def build_backbone(ca: np.ndarray) -> Dict[str, np.ndarray]:
    L = ca.shape[0]
    if L == 0:
        z = np.zeros((0, 3), dtype=np.float32)
        return {"N": z, "CA": z, "C": z, "O": z}

    ca_pad = np.vstack([ca[0], ca, ca[-1]])
    t = np.zeros_like(ca)
    n = np.zeros_like(ca)
    b = np.zeros_like(ca)

    for i in range(L):
        tt   = (ca_pad[i+2] - ca_pad[i+1]) + (ca_pad[i+1] - ca_pad[i])
        t[i] = _safe_norm(tt)

    for i in range(L):
        curv  = ca[i+1] - 2*ca[i] + ca[i-1] if 0 < i < L-1 else ca[min(i+1,L-1)] - ca[max(i-1,0)]
        curv -= np.dot(curv, t[i]) * t[i]
        if np.linalg.norm(curv) < 1e-6:
            fb   = np.array([1.,0.,0.]) if abs(t[i][0]) < 0.9 else np.array([0.,1.,0.])
            curv = np.cross(t[i], fb)
        n[i] = _safe_norm(curv)
        b[i] = _safe_norm(np.cross(t[i], n[i]))

    N = ca - 1.458 * t + 0.2 * n
    C = ca + 1.525 * t + 0.2 * n
    O = C  - 1.231 * n + 0.3 * b
    return {"N": N, "CA": ca.copy(), "C": C, "O": O}


def write_pdb(seq: str, coords: torch.Tensor, path: str, bfactor: float):
    ca  = coords.cpu().numpy()
    bb  = build_backbone(ca)
    L   = min(len(seq), ca.shape[0])
    with open(path, "w") as f:
        atom_id = 1
        for i in range(L):
            resn = AA1_TO_AA3.get(seq[i], "UNK")
            for atom in ("N", "CA", "C", "O"):
                x, y, z = bb[atom][i]
                f.write(
                    f"ATOM  {atom_id:5d} {atom:<4s}{resn:>3s} A{i+1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{bfactor:6.2f}           {atom[0]:>2s}\n"
                )
                atom_id += 1
        f.write("TER\nEND\n")


def pdb_stats(path: str) -> Optional[dict]:
    vals = []
    with open(path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                try:
                    vals.append(float(line[60:66]))
                except ValueError:
                    pass
    if not vals:
        return None
    p = np.array(vals)
    return {
        "length":    int(p.size),
        "mean_b":    round(float(p.mean()), 2),
        "pct_ge_70": round(float((p >= 70).mean() * 100), 1),
        "pct_lt_50": round(float((p <  50).mean() * 100), 1),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Protein generation + structure prediction")
    parser.add_argument("--predict",         action="store_true")
    parser.add_argument("--gen-len",         type=int,   default=512)
    parser.add_argument("--seed-seq",        type=str,   default="M")
    parser.add_argument("--temp",            type=float, default=0.9)
    parser.add_argument("--top-k",           type=int,   default=10)
    parser.add_argument("--top-p",           type=float, default=0.9)
    parser.add_argument("--rep-penalty",     type=float, default=1.2)
    parser.add_argument("--no-repeat-ngram", type=int,   default=3)
    parser.add_argument("--out",             type=str,   default="generated_predicted.pdb")
    args = parser.parse_args()

    set_seeds()
    print(f"Device: {DEVICE}")

    if not args.predict:
        parser.print_help()
        return

    init_plm()

    print("\nLoading sequence models...")
    seq_models, seq_weights = load_seq_models()

    total_len = min(args.gen_len, MAX_LEN)
    print(f"\nGenerating sequence (length={total_len})...")
    seq = generate_sequence(
        seq_models, seq_weights,
        seed=args.seed_seq,
        total_len=total_len,
        T=args.temp,
        top_k=args.top_k,
        top_p=args.top_p,
        rep_penalty=args.rep_penalty,
        no_repeat_ngram=args.no_repeat_ngram,
    )
    print(f"Sequence ({len(seq)} aa): {seq[:80]}{'...' if len(seq) > 80 else ''}")

    print("\nPredicting structure...")
    dist_path  = CHECKPOINT_DIR / "dist_densenetU.pt"
    dist_model = DistogramDenseNetU(c_in=3, n_bins=63).to(DEVICE).eval()
    if dist_path.exists():
        dist_model.load_state_dict(torch.load(dist_path, map_location=DEVICE))
    else:
        print("  Warning: distogram checkpoint not found, using untrained model.")

    bin_centers = torch.tensor(
        0.5 * (np.linspace(2., 20., 64)[1:] + np.linspace(2., 20., 64)[:-1]),
        dtype=torch.float32, device=DEVICE
    )

    with torch.no_grad():
        emb    = embed_sequence(seq, len(seq))
        pw     = pairwise_features(emb.to(DEVICE))
        logits = dist_model(pw.unsqueeze(0)).squeeze(0)
        D      = dist_matrix_from_logits(logits, bin_centers)
        coords = smooth_coords(coords_from_dist(D))

    print("\nScoring structure...")
    proxy = score_structure(coords)
    print(f"  Proxy pLDDT: {proxy:.1f}")

    write_pdb(seq, coords, args.out, bfactor=proxy)

    stats = pdb_stats(args.out)
    if stats:
        print(f"\nPDB written -> {args.out}")
        print(f"  Length:       {stats['length']} residues")
        print(f"  Mean B-factor:{stats['mean_b']}")
        print(f"  pLDDT >= 70:  {stats['pct_ge_70']}%")
        print(f"  pLDDT <  50:  {stats['pct_lt_50']}%")


if __name__ == "__main__":
    main()
