#!/usr/bin/env python3
"""
MIRAGE Pipeline — Step 1: Build Enhanced Graph Dataset
=======================================================
Replaces the original 5-dim scalar node features with 64-dim VAE embeddings,
and attaches a normalized 4-dim clinical feature vector as a graph-level
attribute. Produces oasis_graphs_v2.pt ready for MIRAGE training.

Node features:  graph.x        shape (3, 64)  — VAE embedding per tissue
Clinical feats: graph.clinical shape (4,)     — [Age, Educ, eTIV, nWBV]
Label:          graph.y        shape (1,)     — binary AD_label
Subject ID:     graph.subject_id              — string, for traceability

Phase 9c: Graph v2 Construction
Author: Medical Data Engineering Team
Date: 2026-04-19
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT      = Path('/Users/yuvalshah/Desktop/ATML_PROJECT')
VAE_EMBEDDINGS    = PROJECT_ROOT / 'vae_embeddings.npz'
MASTER_INDEX      = PROJECT_ROOT / 'master_index.csv'
ORIGINAL_GRAPHS   = PROJECT_ROOT / 'oasis_graphs.pt'
OUTPUT_GRAPHS     = PROJECT_ROOT / 'oasis_graphs_v2.pt'
OUTPUT_SCALER     = PROJECT_ROOT / 'clinical_scaler.pkl'
OUTPUT_EMB_SCALER = PROJECT_ROOT / 'embedding_scaler.pkl'

CLINICAL_COLS     = ['Age', 'Educ', 'eTIV', 'nWBV']
CLINICAL_COLS_ALL = ['Age', 'Educ', 'eTIV', 'nWBV', 'Age_nWBV']  # 5 features incl. interaction

torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("=" * 60)
    logger.info("MIRAGE: BUILD GRAPH DATASET V2")
    logger.info("=" * 60)

    # ── Step 1: Load all inputs ──────────────────────────────────────────────
    logger.info("Step 1 — Loading inputs...")

    try:
        emb_npz = np.load(str(VAE_EMBEDDINGS), allow_pickle=True)
        emb_keys = list(emb_npz.keys())
        logger.info(f"  VAE embeddings: {len(emb_keys)} subjects, "
                    f"shape per subject: {emb_npz[emb_keys[0]].shape}")
    except Exception as e:
        logger.error(f"Failed to load vae_embeddings.npz: {e}")
        raise

    try:
        df = pd.read_csv(str(MASTER_INDEX))
        logger.info(f"  master_index.csv: {len(df)} subjects, "
                    f"columns: {list(df.columns)}")
    except Exception as e:
        logger.error(f"Failed to load master_index.csv: {e}")
        raise

    try:
        orig_graphs = torch.load(str(ORIGINAL_GRAPHS), weights_only=False)
        logger.info(f"  oasis_graphs.pt: {len(orig_graphs)} graphs, "
                    f"original x shape: {orig_graphs[0].x.shape}")
    except Exception as e:
        logger.error(f"Failed to load oasis_graphs.pt: {e}")
        raise

    # Compute binary AD_label
    df['AD_label'] = (df['CDR'] > 0.0).astype(int)
    logger.info(f"  AD_label: {(df['AD_label']==0).sum()} healthy, "
                f"{(df['AD_label']==1).sum()} dementia")

    # ── Step 2: Build and normalize clinical feature vectors ─────────────────
    logger.info("\nStep 2 — Building clinical feature vectors...")

    # Check for missing values
    missing = df[CLINICAL_COLS].isnull().sum()
    if missing.sum() > 0:
        logger.warning(f"  Missing values detected: {missing[missing>0].to_dict()}")
        logger.warning("  Filling with column medians before scaling")
        df[CLINICAL_COLS] = df[CLINICAL_COLS].fillna(df[CLINICAL_COLS].median())

    # Add Age×nWBV interaction feature BEFORE scaling
    df['Age_nWBV'] = df['Age'] * df['nWBV']
    logger.info(f"  Age×nWBV interaction feature added "
                f"(range: [{df['Age_nWBV'].min():.1f}, {df['Age_nWBV'].max():.1f}])")

    scaler = StandardScaler()
    clinical_scaled = scaler.fit_transform(df[CLINICAL_COLS_ALL].values)

    # Save fitted scaler for inference
    joblib.dump(scaler, str(OUTPUT_SCALER))
    logger.info(f"  Clinical scaler saved: {OUTPUT_SCALER}")
    logger.info(f"  Clinical features (5): {CLINICAL_COLS_ALL}")
    logger.info(f"  Scaled range: [{clinical_scaled.min():.3f}, "
                f"{clinical_scaled.max():.3f}]")

    # Build lookup: subject_id -> (clinical_vector, AD_label)
    clinical_lookup = {}
    for i, row in df.iterrows():
        clinical_lookup[row['ID']] = {
            'clinical': clinical_scaled[i].astype(np.float32),
            'ad_label': int(row['AD_label'])
        }

    # ── Step 2b: Normalize VAE embeddings across all subjects × tissues ────────
    logger.info("\nStep 2b — Normalizing VAE embeddings...")

    # subject_ids_ordered is defined here for the normalization step
    subject_ids_ordered = df['ID'].tolist()

    # Collect all 161 × 3 = 483 embedding vectors into (483, 64)
    all_emb_list = []
    for sid in subject_ids_ordered:
        emb = emb_npz[sid].astype(np.float32)   # (3, 64)
        all_emb_list.append(emb)                 # 3 rows each
    all_emb_matrix = np.vstack(all_emb_list)     # (483, 64)

    # Print before stats
    logger.info(f"  Before normalization — mean: {all_emb_matrix.mean():.4f}, "
                f"std: {all_emb_matrix.std():.4f}")
    logger.info(f"  Before normalization — min:  {all_emb_matrix.min():.4f}, "
                f"max: {all_emb_matrix.max():.4f}")

    # Fit StandardScaler on the full (483, 64) matrix
    emb_scaler = StandardScaler()
    all_emb_norm = emb_scaler.fit_transform(all_emb_matrix)  # (483, 64)

    # Print after stats — should be ~0.0 mean, ~1.0 std
    logger.info(f"  After normalization  — mean: {all_emb_norm.mean():.4f}, "
                f"std: {all_emb_norm.std():.4f}")
    logger.info(f"  After normalization  — min:  {all_emb_norm.min():.4f}, "
                f"max: {all_emb_norm.max():.4f}")

    # Save fitted embedding scaler
    joblib.dump(emb_scaler, str(OUTPUT_EMB_SCALER))
    logger.info(f"  Embedding scaler saved: {OUTPUT_EMB_SCALER}")

    # Reshape back to per-subject (3, 64) lookup
    emb_normalized = {}
    for i, sid in enumerate(subject_ids_ordered):
        emb_normalized[sid] = all_emb_norm[i*3 : i*3+3].astype(np.float32)

    # ── Step 3: Build new graph objects ──────────────────────────────────────
    logger.info("\nStep 3 — Building new graph objects...")

    # subject_ids_ordered already defined in Step 2b above

    # Verify all subject IDs have embeddings
    missing_emb = [sid for sid in subject_ids_ordered if sid not in emb_npz]
    if missing_emb:
        logger.error(f"Missing embeddings for {len(missing_emb)} subjects: "
                     f"{missing_emb[:5]}")
        raise ValueError("Embedding coverage incomplete")

    new_graphs = []
    shape_errors = []

    for idx, (orig_graph, subject_id) in enumerate(
            zip(orig_graphs, subject_ids_ordered)):
        try:
            # VAE embedding: (3, 64) — normalized
            embedding = emb_normalized[subject_id]
            assert embedding.shape == (3, 64), \
                f"Expected (3,64), got {embedding.shape}"

            # Clinical vector: (5,)
            info = clinical_lookup[subject_id]
            clinical_vec = info['clinical']
            assert clinical_vec.shape == (5,), \
                f"Expected (5,), got {clinical_vec.shape}"

            # Build new Data object
            new_graph = Data(
                x           = torch.tensor(embedding,    dtype=torch.float32),
                edge_index  = orig_graph.edge_index.clone(),
                y           = torch.tensor([info['ad_label']], dtype=torch.long),
                clinical    = torch.tensor(clinical_vec, dtype=torch.float32),
                subject_id  = subject_id
            )

            new_graphs.append(new_graph)

        except Exception as e:
            logger.warning(f"  Failed for {subject_id}: {e}")
            shape_errors.append(subject_id)
            continue

    logger.info(f"  Built {len(new_graphs)} graphs successfully")
    if shape_errors:
        logger.warning(f"  Failed: {shape_errors}")

    # ── Step 4: Validate and save ─────────────────────────────────────────────
    logger.info("\nStep 4 — Validation...")

    print()
    print("First 3 graphs:")
    print("-" * 55)
    for g in new_graphs[:3]:
        print(f"  subject_id : {g.subject_id}")
        print(f"  x shape    : {g.x.shape}")
        x_np = g.x.numpy()
        print(f"  x min/max/mean: {x_np.min():.4f} / {x_np.max():.4f} / {x_np.mean():.4f}")
        print(f"  clinical   : {g.clinical.shape}  values: "
              f"{g.clinical.numpy().round(3)}")
        print(f"  y          : {g.y.item()}")
        print()

    # Full validation
    x_ok       = all(g.x.shape       == torch.Size([3, 64]) for g in new_graphs)
    clin_ok    = all(g.clinical.shape == torch.Size([5])     for g in new_graphs)
    edge_ok    = all(g.edge_index.shape == torch.Size([2, 6]) for g in new_graphs)
    label_vals = set(g.y.item() for g in new_graphs)

    print(f"Validation checks:")
    print(f"  All x shapes (3,64):       {'✓' if x_ok    else '✗'}")
    print(f"  All clinical shapes (5,):  {'✓' if clin_ok else '✗'}")
    print(f"  All edge_index (2,6):      {'✓' if edge_ok else '✗'}")
    print(f"  Label values present:      {sorted(label_vals)}")
    print(f"  AD_label=0 (Healthy):      "
          f"{sum(g.y.item()==0 for g in new_graphs)}")
    print(f"  AD_label=1 (Dementia):     "
          f"{sum(g.y.item()==1 for g in new_graphs)}")

    if not (x_ok and clin_ok and edge_ok):
        raise RuntimeError("Validation failed — check shape errors above")

    # Save
    torch.save(new_graphs, str(OUTPUT_GRAPHS))
    size_kb = OUTPUT_GRAPHS.stat().st_size / 1024
    print()
    print(f"Saved: {OUTPUT_GRAPHS}")
    print(f"File size: {size_kb:.1f} KB")
    print()
    logger.info("✓ oasis_graphs_v2.pt built successfully")


if __name__ == '__main__':
    main()
