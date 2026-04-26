# MIRAGE — MRI-Informed Representation for Alzheimer's Graph Embeddings

## Overview

MIRAGE is an end-to-end Alzheimer's disease classification pipeline built on the OASIS-1 neuroimaging dataset. The pipeline combines classical machine learning baselines, Graph Attention Networks (GAT), and a semi-supervised Variational Autoencoder (VAE) image encoder to classify subjects as healthy controls (CDR = 0) or dementia patients (CDR > 0). Starting from raw ANALYZE 7.5 MRI volumes, the pipeline extracts tissue segmentation volumes, constructs brain graphs where nodes represent CSF, Gray Matter, and White Matter compartments, and progressively enriches node features from hand-crafted scalars to learned 64-dimensional VAE embeddings. A supervised fine-tuning stage nudges the VAE latent space toward discriminative representations, and the final MIRAGE model fuses graph-level structural embeddings with normalized clinical features for binary classification.

---

## Pipeline Architecture

Each phase is implemented as a standalone script that reads from and writes to the project root. Scripts must be run in order.

1. **`01_build_master_index.py`** — Crawls disc1–disc8 to locate `_t88_masked_gfc_fseg.hdr` files, merges with the official OASIS-1 clinical Excel metadata, drops subjects missing CDR or MRI files. **Output:** `master_index.csv` (161 subjects).

2. **`02_tabular_baselines.py`** — Evaluates Logistic Regression, SVM (RBF), and Random Forest on demographic/structural features (Age, Educ, SES, eTIV, nWBV, ASF) using Stratified 5-Fold CV. Explicitly excludes MMSE to avoid data leakage. **Output:** console metrics table.

3. **`03_extract_nodes.py`** — Loads each subject's FSL FAST segmentation mask (`.hdr`/`.img`) using nibabel. Hardcodes voxel volume = 1.0 mm³ (t88 space guarantee) to bypass corrupted ANALYZE 7.5 metadata. Counts voxels per tissue label (1=CSF, 2=GM, 3=WM). **Output:** `node_features.csv`.

4. **`04_build_graphs.py`** — Merges `master_index.csv` with `node_features.csv`, normalizes all features with StandardScaler, and constructs one PyTorch Geometric `Data` object per subject: 3 nodes (CSF/GM/WM), 5 node features each, fully connected edge topology. **Output:** `oasis_graphs.pt` (161 graphs).

5. **`05_train_gnn.py`** — Trains and evaluates a 2-layer GAT (GATConv 5→64→32, global mean pool, BCEWithLogitsLoss) using Stratified 5-Fold CV with early stopping. **Output:** console metrics, `best_gnn.pth`.

6. **`06_interpret_attention.py`** — Trains the GAT on the full dataset and extracts per-edge attention weights from the first GATConv layer. Averages across 4 heads and 161 graphs to produce a ranked leaderboard of tissue connections. **Output:** console attention leaderboard.

7. **`07_generate_figures.py`** — Generates three publication-quality figures at 300 DPI: confusion matrix, ROC curve with AUC, and attention weight bar chart. **Output:** `fig1_confusion_matrix.png`, `fig2_roc_curve.png`, `fig3_attention_weights.png`.

8. **`08_ablation_study.py`** — Ablation study: strips all features except tissue volume (`graph.x = graph.x[:, 0:1]`) and re-runs 5-fold CV with `in_channels=1`. Quantifies the contribution of demographic features. **Output:** console comparison table.

9. **`09a_extract_slices.py`** *(local Mac)* — For each subject, loads the skull-stripped atlas-space MRI (`_t88_masked_gfc.img`) and FSL segmentation mask. For each tissue type, finds the center of mass, extracts axial/coronal/sagittal slices, resizes to 64×64, and stacks as a (3, 64, 64) tensor. **Output:** `slice_dataset.npz` (296 subjects × 3 tissues × 3 planes × 64×64).

10. **`09b_vae_train.ipynb`** *(Google Colab, GPU required)* — Trains a convolutional VAE (encoder: 3 conv blocks → 64-dim latent; decoder: 3 transposed conv blocks) on all 296 subjects (self-supervised, no labels). Uses β-VAE loss (β=0.5). **Output:** `best_vae.pth`, `vae_embeddings.npz`, `vae_anomaly_scores.csv`.

11. **`09c_build_graphs_v2.py`** — Replaces scalar node features with VAE embeddings (normalized to zero mean/unit variance). Adds a 5th clinical feature (Age×nWBV interaction). Saves fitted scalers for inference. **Output:** `oasis_graphs_v2.pt`, `clinical_scaler.pkl`, `embedding_scaler.pkl`.

12. **`09d_train_mirage.py`** — Trains the MIRAGE model: GATConv(64→128, 4 heads) → GATConv(128→32) → global mean pool → concat clinical(5) → Linear(37→16→1). Uses Age-CDR stratified 5-fold CV, BCEWithLogitsLoss with pos_weight=93/68, gradient clipping. **Output:** `best_mirage.pth`, `fig4_mirage_roc.png`, `fig5_mirage_confusion.png`, `fig6_mirage_vs_baseline.png`.

13. **`09e_finetune_vae.py`** — Supervised fine-tuning of the VAE encoder on 161 labeled subjects. Freezes all layers except `encoder_fc.1`, `fc_mu`, `fc_logvar`. Uses differential learning rates (encoder 1e-4, head 1e-3) and AUC-based early stopping. Rebuilds `oasis_graphs_v3.pt` with fine-tuned embeddings. **Output:** `best_finetune.pth`, `vae_embeddings_finetuned.npz`, `oasis_graphs_v3.pt`.

---

## Key Results

All models evaluated with Stratified 5-Fold Cross-Validation on 161 labeled OASIS-1 subjects (93 healthy, 68 dementia). Features used: tissue volumes (CSF, GM, WM), Age, Education, eTIV, nWBV, ASF. MMSE excluded to prevent data leakage.

| Model                | ROC-AUC        | Accuracy       | Sensitivity    | Specificity    |
|----------------------|----------------|----------------|----------------|----------------|
| Logistic Regression  | 0.7933         | —              | —              | —              |
| Original GAT         | 0.8360 ± 0.0260 | 0.7451 ± 0.0256 | 0.7385 ± 0.0917 | 0.7538 ± 0.1064 |
| MIRAGE (VAE+GAT)     | 0.8014 ± 0.0615 | 0.7394 ± 0.0564 | **0.7659 ± 0.0493** | 0.7199 ± 0.0657 |

The Original GAT achieves the highest ROC-AUC (0.836) using hand-crafted volumetric features derived directly from FSL FAST tissue segmentation. MIRAGE, which replaces these expert-engineered features with 64-dimensional VAE embeddings learned from 2.5D MRI slices, achieves a competitive 0.801 AUC while demonstrating meaningfully better sensitivity (0.766 vs 0.739) — a clinically important property for a screening tool. The modest gap between MIRAGE and the Original GAT reflects a well-established finding in medical imaging: at small dataset scales (N=161), domain-informed hand-crafted features remain highly competitive with learned representations. The Age-CDR stratified cross-validation strategy reduced fold-to-fold variance by 49% (std: 0.120 → 0.062), confirming that naive binary stratification produces unstable estimates when age and disease severity are confounded.

---

## Dataset

This project uses the **OASIS-1 Cross-Sectional MRI Dataset** (Open Access Series of Imaging Studies).

- **Total subjects:** 296 (used for VAE pretraining)
- **Labeled subjects:** 161 (with CDR scores; used for classification)
  - Healthy controls (CDR = 0.0): 93
  - Dementia (CDR > 0.0): 68 — of which CDR=0.5: 49, CDR=1.0: 18, CDR=2.0: 1
- **Age range:** 18–96 years
- **MRI type:** T1-weighted MPRAGE, registered to Talairach-Tournoux (t88) atlas space

Raw data is **not included** in this repository due to size (~36 GB across 8 discs). Download from [oasis-brains.org](https://www.oasis-brains.org/) and place the unzipped disc folders in the project root:

```
MIRAGE/
├── disc1/
│   ├── OAS1_0001_MR1/
│   │   ├── RAW/
│   │   ├── PROCESSED/MPRAGE/T88_111/   ← *_t88_masked_gfc.img (CNN input)
│   │   └── FSL_SEG/                    ← *_fseg.img (segmentation mask)
│   └── ...
├── disc2/ ... disc8/
```

The clinical metadata Excel file (`oasis_cross-sectional-*.xlsx`) is also available from oasis-brains.org.

---

## Installation

```bash
git clone https://github.com/yourusername/MIRAGE.git
cd MIRAGE
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**PyTorch Geometric extensions** (optional, for performance):
```bash
pip install pyg-lib torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.11.0+cpu.html
```

---

## Usage

Scripts must be run in order from the project root (where `disc1/`–`disc8/` are located):

```bash
python scripts/01_build_master_index.py
python scripts/02_tabular_baselines.py
python scripts/03_extract_nodes.py
python scripts/04_build_graphs.py
python scripts/05_train_gnn.py
python scripts/06_interpret_attention.py
python scripts/07_generate_figures.py
python scripts/08_ablation_study.py
python scripts/09a_extract_slices.py
# Upload slice_dataset.npz + slice_metadata.csv to Google Drive
# Run scripts/09b_vae_train.ipynb on Google Colab (GPU required)
# Download best_vae.pth + vae_embeddings.npz back to project root
python scripts/09c_build_graphs_v2.py
python scripts/09d_train_mirage.py
python scripts/09e_finetune_vae.py
python scripts/09d_train_mirage.py   # re-run on oasis_graphs_v3.pt
```

**Notes:**
- `disc1/`–`disc8/` must be present in the project root before running scripts 01–09a
- `09b_vae_train.ipynb` runs on Google Colab and requires a GPU runtime (T4 or better)
- All paths in scripts point to `/Users/yuvalshah/Desktop/ATML_PROJECT/` — update `PROJECT_ROOT` constants if running elsewhere

---

## Project Structure

```
MIRAGE/
├── scripts/          Python pipeline scripts (01–09e) and Colab notebook
├── figures/          Publication-quality output figures (300 DPI PNG)
├── outputs/          LaTeX paper sections
├── data/
│   ├── raw/          Placeholder — place disc1–disc8 here (not tracked by git)
│   └── processed/    Placeholder — generated CSVs and graph files land here
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Citation

```bibtex
@misc{mirage2026,
  title   = {MIRAGE: MRI-Informed Representation for Alzheimer's Graph Embeddings},
  author  = {Shah, Yuval},
  year    = {2026},
  note    = {OASIS-1 dataset, Graph Attention Networks, Variational Autoencoder}
}
```

---

## Acknowledgements

- **OASIS-1 Dataset:** Marcus, D.S., Wang, T.H., Parker, J., Csernansky, J.G., Morris, J.C., Buckner, R.L. (2007). *Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI data in young, middle aged, nondemented, and demented older adults.* Journal of Cognitive Neuroscience, 19(9), 1498–1507.
- Funded by grants P50 AG05681, P01 AG03991, R01 AG021910, P50 MH071616, U24 RR021382, R01 MH56584.
