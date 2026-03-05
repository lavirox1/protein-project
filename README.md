# Protein Design Pipeline

Generates a novel protein sequence and predicts its 3D structure, saved as a PDB file.

## How it works

1. **Sequence generation** — ProtBERT embeds the growing sequence at each step. A CNN+BiLSTM and a GRU model (ensemble) predict the next amino acid, one at a time.
2. **Structure prediction** — The sequence is fed into a distogram model that predicts pairwise distances between residues. These distances are converted to 3D coordinates using MDS, then smoothed to enforce realistic bond lengths.
3. **Scoring** — A 2D contact map scorer and a 3D voxel scorer give a proxy confidence score (0–100), written into the PDB as the B-factor column.

## Requirements

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers timm biopython tqdm numpy
```

> First run will download ProtBERT (~420MB), cached automatically after that.

## Usage

```
python protein_end2end.py --predict
```

This generates a 512 residue protein and writes `generated_predicted.pdb`.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--gen-len` | 512 | Length of the generated sequence (max 512) |
| `--seed-seq` | M | Starting amino acid(s). M = Methionine, the biological start |
| `--temp` | 0.9 | Sampling temperature. Higher = more random |
| `--top-k` | 10 | Only sample from the top K amino acids at each step |
| `--top-p` | 0.9 | Nucleus sampling cutoff |
| `--rep-penalty` | 1.2 | Penalises repeating the same amino acids too often |
| `--out` | generated_predicted.pdb | Output PDB filename |

### Example

```
python protein_end2end.py --predict --gen-len 256 --out my_protein.pdb
```

## Output

```
Sequence (512 aa): MPPPCPSCC...
Proxy pLDDT: 84.7

PDB written -> generated_predicted.pdb
  Length:       512 residues
  Mean B-factor:84.73
  pLDDT >= 70:  100.0%
  pLDDT <  50:  0.0%
```

The PDB file can be opened in [PyMOL](https://pymol.org) or [ChimeraX](https://www.rbvi.ucsf.edu/chimerax/) to visualise the 3D structure. Residues are coloured by B-factor (proxy confidence).

## Checkpoints

| File | Description |
|------|-------------|
| `seqgen_cnnlstm.pt` | CNN+BiLSTM sequence model |
| `timenet_gru.pt` | GRU sequence model |
| `dist_densenetU.pt` | Distogram prediction model |
| `densenet2d.pt` | 2D contact map scorer |
| `vol3d.pt` | 3D voxel scorer |

## Notes

- Runs on CPU but is significantly faster with an NVIDIA GPU.
- The proxy pLDDT score is not the same as AlphaFold's pLDDT — it's an approximation from models trained on limited data.
- All models were trained with ProtBERT embeddings (dim=1024). Do not use `--no-plm` flags from older versions.
