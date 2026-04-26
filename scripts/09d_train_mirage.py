#!/usr/bin/env python3
"""
MIRAGE Pipeline — Step 2: Train & Evaluate MIRAGE GAT
======================================================
Trains the MIRAGE model (VAE-enhanced GAT + clinical stream) using
Stratified 5-Fold CV and compares against published baselines.

Architecture:
  Graph stream:   GATConv(64→128) → GATConv(128→32) → global_mean_pool
  Clinical stream: 4-dim vector concatenated after pooling
  Classifier:     Linear(36→16) → ELU → Dropout → Linear(16→1)

Outputs:
  best_mirage.pth          — best fold model weights
  fig4_mirage_roc.png      — ROC curve
  fig5_mirage_confusion.png — confusion matrix
  fig6_mirage_vs_baseline.png — grouped bar chart vs baselines

Phase 9d: MIRAGE Training & Evaluation
Author: Medical Data Engineering Team
Date: 2026-04-19
"""

import logging
import warnings
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, recall_score,
    confusion_matrix, roc_curve, auc
)

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT   = Path('/Users/yuvalshah/Desktop/ATML_PROJECT')
GRAPHS_V2      = PROJECT_ROOT / 'oasis_graphs_v3.pt'
MASTER_INDEX   = PROJECT_ROOT / 'master_index.csv'
BEST_MODEL_OUT = PROJECT_ROOT / 'best_mirage.pth'

SKIP_TRAINING  = False   # Set False to run full training

RANDOM_STATE   = 42
N_SPLITS       = 5
BATCH_SIZE     = 16
MAX_EPOCHS     = 200
LR             = 0.001
WEIGHT_DECAY   = 1e-4
PATIENCE_ES    = 30    # early stopping
PATIENCE_LR    = 15    # scheduler
POS_WEIGHT     = 93 / 68   # ~1.368 — healthy / dementia ratio

# Baseline results for comparison table
BASELINES = {
    'Logistic Regression': {
        'roc_auc': (0.7933, None),
        'accuracy': (None, None),
        'sensitivity': (None, None),
        'specificity': (None, None),
    },
    'Original GAT': {
        'roc_auc':     (0.8360, 0.0260),
        'accuracy':    (0.7451, 0.0256),
        'sensitivity': (0.7385, 0.0917),
        'specificity': (0.7538, 0.1064),
    },
}

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

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
# MODEL ARCHITECTURE
# ============================================================================

class MIRAGEModel(nn.Module):
    """
    MIRAGE: Multi-modal Integrated Representation for Alzheimer's Graph Encoding

    Graph stream:
      GATConv(64, 32, heads=4, concat=True)  → (128,) per node
      ELU → Dropout(0.3)
      GATConv(128, 32, heads=1, concat=False) → (32,) per node
      global_mean_pool                         → (32,) per graph

    Clinical stream:
      4-dim normalized vector bypasses GAT entirely

    Fusion + Classifier:
      concat([graph_emb(32), clinical(4)]) → (36,)
      Linear(36, 16) → ELU → Dropout(0.2) → Linear(16, 1)
    """

    def __init__(self, node_feat_dim: int = 64, clinical_dim: int = 5,
                 hidden: int = 32, heads: int = 4, dropout: float = 0.3):
        super().__init__()

        # Graph stream
        self.conv1 = GATConv(
            in_channels=node_feat_dim,
            out_channels=hidden,
            heads=heads,
            concat=True,
            dropout=dropout
        )
        self.conv2 = GATConv(
            in_channels=hidden * heads,   # 128
            out_channels=hidden,          # 32
            heads=1,
            concat=False,
            dropout=dropout
        )

        # Classifier (graph_emb=32 + clinical=5 = 37)
        fused_dim = hidden + clinical_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 16),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )

        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        clinical = data.clinical   # (batch_size, 4)

        # Graph stream
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)

        # Pool to graph level
        x = global_mean_pool(x, batch)   # (batch_size, 32)

        # Fuse with clinical features
        # PyG batches graph.clinical as (batch_size * 5,) — reshape to (batch_size, 5)
        clinical = clinical.view(-1, 5)
        fused = torch.cat([x, clinical], dim=1)   # (batch_size, 37)

        return self.classifier(fused)             # (batch_size, 1)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_graphs(path: Path) -> List[Data]:
    logger.info(f"Loading {path}...")
    graphs = torch.load(str(path), weights_only=False)
    logger.info(f"  Loaded {len(graphs)} graphs")
    logger.info(f"  x shape: {graphs[0].x.shape}")
    logger.info(f"  clinical shape: {graphs[0].clinical.shape}")
    logger.info(f"  y: {graphs[0].y}")
    return graphs


def extract_labels(graphs: List[Data]) -> np.ndarray:
    return np.array([g.y.item() for g in graphs])


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out  = model(data).view(-1)
        loss = criterion(out, data.y.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for data in loader:
            data  = data.to(device)
            out   = model(data).view(-1)
            probs = torch.sigmoid(out)
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(data.y.cpu().numpy().tolist())
    return np.array(all_probs), np.array(all_labels)


def calc_metrics(y_true, y_prob) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'accuracy':    accuracy_score(y_true, y_pred),
        'roc_auc':     roc_auc_score(y_true, y_prob),
        'sensitivity': recall_score(y_true, y_pred),
        'specificity': tn / (tn + fp),
    }


# ============================================================================
# CROSS-VALIDATION
# ============================================================================

def run_cross_validation(graphs, labels, device, subj_lookup=None):
    logger.info("=" * 60)
    logger.info("STRATIFIED 5-FOLD CROSS-VALIDATION")
    logger.info("=" * 60)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                          random_state=RANDOM_STATE)

    fold_results   = []
    all_y_true     = []
    all_y_prob     = []
    best_auc       = -1.0
    best_weights   = None

    pos_weight_tensor = torch.tensor([POS_WEIGHT], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # Build combined CDR-tier × Age-tier stratification labels
    def get_strat_label(sid):
        if subj_lookup is None:
            return 0
        info = subj_lookup.get(sid, {})
        cdr  = info.get('CDR', 0.0)
        age  = info.get('Age', 75.0)
        cdr_tier = 0 if cdr == 0.0 else (1 if cdr == 0.5 else 2)
        age_tier = 0 if age < 70 else (1 if age <= 80 else 2)
        return cdr_tier * 3 + age_tier   # 9 possible bins

    strat_labels = np.array([get_strat_label(g.subject_id) for g in graphs])
    logger.info(f"Stratification bins used: {sorted(set(strat_labels.tolist()))}")

    for fold, (train_idx, test_idx) in enumerate(
            skf.split(np.zeros(len(strat_labels)), strat_labels), 1):

        logger.info(f"\n{'='*50}")
        logger.info(f"FOLD {fold}/{N_SPLITS}  "
                    f"(train={len(train_idx)}, test={len(test_idx)})")
        logger.info(f"{'='*50}")

        train_graphs = [graphs[i] for i in train_idx]
        test_graphs  = [graphs[i] for i in test_idx]

        # ── FOLD DIAGNOSTICS ─────────────────────────────────────────────────
        if subj_lookup is not None:
            from collections import Counter

            def fold_stats(graph_list, split_name):
                cdr_counter  = Counter()
                label_counter = Counter()
                ages, nwbvs_h, nwbvs_d = [], [], []
                rows = []
                for g in graph_list:
                    sid  = g.subject_id
                    info = subj_lookup.get(sid, {})
                    cdr  = info.get('CDR', float('nan'))
                    lbl  = g.y.item()
                    age  = info.get('Age', float('nan'))
                    nwbv = info.get('nWBV', float('nan'))
                    cdr_counter[cdr]   += 1
                    label_counter[lbl] += 1
                    ages.append(age)
                    if lbl == 0:
                        nwbvs_h.append(nwbv)
                    else:
                        nwbvs_d.append(nwbv)
                    rows.append((sid, cdr, lbl, age, nwbv))
                return cdr_counter, label_counter, ages, nwbvs_h, nwbvs_d, rows

            # ── TEST SET ──────────────────────────────────────────────────────
            t_cdr, t_lbl, t_ages, t_nwbv_h, t_nwbv_d, t_rows = \
                fold_stats(test_graphs, 'TEST')

            print(f"\n{'─'*65}")
            print(f"  FOLD {fold} — TEST SET ({len(test_graphs)} subjects)")
            print(f"{'─'*65}")
            print(f"  {'Subject ID':<18} {'CDR':>5} {'AD':>4} {'Age':>5} {'nWBV':>7}")
            print(f"  {'─'*50}")
            for sid, cdr, lbl, age, nwbv in sorted(t_rows, key=lambda r: r[1]):
                print(f"  {sid:<18} {cdr:>5.1f} {lbl:>4}  {age:>5.1f}  {nwbv:>7.3f}")

            print(f"\n  CDR breakdown (test):")
            for cdr_val in sorted(t_cdr.keys()):
                print(f"    CDR={cdr_val}: {t_cdr[cdr_val]}")
            print(f"  Healthy (AD=0): {t_lbl[0]}   Dementia (AD=1): {t_lbl[1]}")

            # ── TRAIN SET ─────────────────────────────────────────────────────
            r_cdr, r_lbl, r_ages, r_nwbv_h, r_nwbv_d, _ = \
                fold_stats(train_graphs, 'TRAIN')

            print(f"\n  FOLD {fold} — TRAIN SET ({len(train_graphs)} subjects)")
            print(f"  CDR breakdown (train):")
            for cdr_val in sorted(r_cdr.keys()):
                print(f"    CDR={cdr_val}: {r_cdr[cdr_val]}")
            print(f"  Healthy (AD=0): {r_lbl[0]}   Dementia (AD=1): {r_lbl[1]}")

            import numpy as _np
            print(f"\n  Mean Age   — Healthy: {_np.nanmean([subj_lookup[g.subject_id]['Age'] for g in train_graphs if g.y.item()==0]):.1f}"
                  f"   Dementia: {_np.nanmean([subj_lookup[g.subject_id]['Age'] for g in train_graphs if g.y.item()==1]):.1f}")
            print(f"  Mean nWBV  — Healthy: {_np.nanmean(r_nwbv_h):.3f}"
                  f"   Dementia: {_np.nanmean(r_nwbv_d):.3f}")
            print(f"{'─'*65}\n")

        # ── SKIP TRAINING FLAG ────────────────────────────────────────────────
        if SKIP_TRAINING:
            continue

        train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE,
                                  shuffle=True)
        test_loader  = DataLoader(test_graphs,  batch_size=BATCH_SIZE,
                                  shuffle=False)

        # Fresh model per fold
        model = MIRAGEModel().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                     weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=PATIENCE_LR, factor=0.5
        )

        best_val   = float('inf')
        patience_c = 0
        best_fold_weights = None

        for epoch in range(1, MAX_EPOCHS + 1):
            train_loss = train_epoch(model, train_loader, optimizer,
                                     criterion, device)

            # Validation loss
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data in test_loader:
                    data = data.to(device)
                    out  = model(data).view(-1)
                    val_loss += criterion(out, data.y.float()).item() \
                                * data.num_graphs
            val_loss /= len(test_loader.dataset)

            scheduler.step(val_loss)

            if epoch % 40 == 0:
                lr_now = optimizer.param_groups[0]['lr']
                logger.info(f"  Epoch {epoch:>3}/{MAX_EPOCHS}  "
                            f"train={train_loss:.4f}  "
                            f"val={val_loss:.4f}  lr={lr_now:.2e}")

            if val_loss < best_val:
                best_val = val_loss
                patience_c = 0
                best_fold_weights = {k: v.clone()
                                     for k, v in model.state_dict().items()}
            else:
                patience_c += 1

            if patience_c >= PATIENCE_ES:
                logger.info(f"  Early stopping at epoch {epoch}")
                break

        # Evaluate with best weights
        model.load_state_dict(best_fold_weights)
        y_prob, y_true = evaluate(model, test_loader, device)
        metrics = calc_metrics(y_true, y_prob)

        logger.info(f"\nFold {fold} results:")
        for k, v in metrics.items():
            logger.info(f"  {k:12s}: {v:.4f}")

        fold_results.append(metrics)
        all_y_true.extend(y_true.tolist())
        all_y_prob.extend(y_prob.tolist())

        # Track best fold model
        if metrics['roc_auc'] > best_auc:
            best_auc     = metrics['roc_auc']
            best_weights = best_fold_weights

    # Save best fold model (only if training was not skipped)
    if not SKIP_TRAINING and best_weights is not None:
        torch.save(best_weights, str(BEST_MODEL_OUT))
        logger.info(f"\nBest fold model saved: {BEST_MODEL_OUT} "
                    f"(AUC={best_auc:.4f})")

    if SKIP_TRAINING:
        logger.info("\nSKIP_TRAINING=True — diagnostics complete, no training run.")
        return [], np.array([]), np.array([])

    return fold_results, np.array(all_y_true), np.array(all_y_prob)


# ============================================================================
# FIGURES
# ============================================================================

def fig_roc(y_true, y_prob, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc     = auc(fpr, tpr)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr, tpr, color='darkred', lw=2.5,
            label=f'MIRAGE (AUC = {roc_auc:.4f})')
    ax.plot(fpr, tpr, color='darkred', lw=2.5)
    # Original GAT reference
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--',
            label='Random Classifier (AUC = 0.5000)')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate',  fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve — MIRAGE Model\n5-Fold Cross-Validation',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")
    return roc_auc


def fig_confusion(y_true, y_prob, save_path):
    y_pred = (y_prob >= 0.5).astype(int)
    cm     = confusion_matrix(y_true, y_pred)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=['Healthy Control', 'Dementia'],
                yticklabels=['Healthy Control', 'Dementia'],
                cbar_kws={'label': 'Count'},
                square=True, linewidths=1, linecolor='gray', ax=ax)
    ax.set_title('Confusion Matrix — MIRAGE Model\n'
                 '5-Fold Cross-Validation (Combined)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('True Label',      fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


def fig_comparison(mirage_summary, save_path):
    metrics     = ['roc_auc', 'accuracy', 'sensitivity', 'specificity']
    metric_lbls = ['ROC-AUC', 'Accuracy', 'Sensitivity', 'Specificity']
    models      = ['Logistic\nRegression', 'Original\nGAT', 'MIRAGE\n(VAE+GAT)']

    # Build value matrix (3 models × 4 metrics)
    values = np.zeros((3, 4))
    errors = np.zeros((3, 4))

    # Logistic Regression
    values[0] = [0.7933, 0.0, 0.0, 0.0]   # only AUC known
    errors[0] = [0.0,    0.0, 0.0, 0.0]

    # Original GAT
    orig = BASELINES['Original GAT']
    for j, m in enumerate(metrics):
        values[1, j] = orig[m][0] if orig[m][0] else 0.0
        errors[1, j] = orig[m][1] if orig[m][1] else 0.0

    # MIRAGE
    for j, m in enumerate(metrics):
        values[2, j] = mirage_summary[m]['mean']
        errors[2, j] = mirage_summary[m]['std']

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(13, 7))
    x       = np.arange(len(metrics))
    width   = 0.22
    colors  = ['#4682B4', '#2E8B57', '#8B0000']

    for i, (model, color) in enumerate(zip(models, colors)):
        offset = (i - 1) * width
        bars   = ax.bar(x + offset, values[i], width,
                        yerr=errors[i], capsize=4,
                        label=model, color=color, alpha=0.85,
                        error_kw={'elinewidth': 1.5})
        # Value labels on bars
        for bar, val in zip(bars, values[i]):
            if val > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.012,
                        f'{val:.3f}', ha='center', va='bottom',
                        fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(metric_lbls, fontsize=12)
    ax.set_ylim([0.5, 1.05])
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('MIRAGE vs Baseline Models — All Metrics\n'
                 '5-Fold Stratified Cross-Validation',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


# ============================================================================
# RESULTS DISPLAY
# ============================================================================

def print_comparison_table(mirage_summary):
    def fmt(mean, std):
        if mean is None:
            return '—'
        if std is None:
            return f'{mean:.4f}'
        return f'{mean:.4f}±{std:.4f}'

    print()
    print("=" * 82)
    print("  MIRAGE RESULTS vs BASELINE")
    print("=" * 82)
    print(f"  {'Model':<24} {'ROC-AUC':<16} {'Accuracy':<16} "
          f"{'Sensitivity':<16} {'Specificity':<16}")
    print("-" * 82)

    # Logistic Regression
    lr = BASELINES['Logistic Regression']
    print(f"  {'Logistic Regression':<24} "
          f"{fmt(lr['roc_auc'][0], None):<16} "
          f"{fmt(None, None):<16} "
          f"{fmt(None, None):<16} "
          f"{fmt(None, None):<16}")

    # Original GAT
    og = BASELINES['Original GAT']
    print(f"  {'Original GAT':<24} "
          f"{fmt(og['roc_auc'][0],     og['roc_auc'][1]):<16} "
          f"{fmt(og['accuracy'][0],    og['accuracy'][1]):<16} "
          f"{fmt(og['sensitivity'][0], og['sensitivity'][1]):<16} "
          f"{fmt(og['specificity'][0], og['specificity'][1]):<16}")

    # MIRAGE
    m = mirage_summary
    print(f"  {'MIRAGE (VAE+GAT)':<24} "
          f"{fmt(m['roc_auc']['mean'],     m['roc_auc']['std']):<16} "
          f"{fmt(m['accuracy']['mean'],    m['accuracy']['std']):<16} "
          f"{fmt(m['sensitivity']['mean'], m['sensitivity']['std']):<16} "
          f"{fmt(m['specificity']['mean'], m['specificity']['std']):<16}")

    print("=" * 82)

    # Delta vs Original GAT
    delta     = m['roc_auc']['mean'] - og['roc_auc'][0]
    delta_pct = delta / og['roc_auc'][0] * 100
    sign      = '+' if delta >= 0 else ''
    print(f"\n  Delta vs Original GAT:  "
          f"ROC-AUC {sign}{delta:.4f} ({sign}{delta_pct:.2f}%)")
    print("=" * 82)
    print()


# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("=" * 60)
    logger.info("MIRAGE: TRAINING & EVALUATION")
    logger.info("=" * 60)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Device: MPS (Apple Metal GPU)")
    else:
        device = torch.device('cpu')
        logger.info("Device: CPU")

    try:
        # Load data
        graphs = load_graphs(GRAPHS_V2)
        labels = extract_labels(graphs)
        logger.info(f"Labels: {np.bincount(labels)}")

        # Load master_index for CDR/clinical lookup during diagnostics
        import pandas as pd
        master_df = pd.read_csv(str(MASTER_INDEX))
        master_df['AD_label'] = (master_df['CDR'] > 0.0).astype(int)
        # Build lookup: subject_id -> row dict
        subj_lookup = master_df.set_index('ID').to_dict('index')

        # Verify clinical attribute exists
        assert hasattr(graphs[0], 'clinical'), \
            "graph.clinical missing — run 09c_build_graphs_v2.py first"

        # Cross-validation
        fold_results, all_y_true, all_y_prob = run_cross_validation(
            graphs, labels, device, subj_lookup
        )

        # Aggregate metrics (only if training was run)
        if SKIP_TRAINING or len(fold_results) == 0:
            logger.info("SKIP_TRAINING=True — skipping aggregation and figures.")
            return

        mirage_summary = {}
        for metric in ['accuracy', 'roc_auc', 'sensitivity', 'specificity']:
            vals = [f[metric] for f in fold_results]
            mirage_summary[metric] = {
                'mean': float(np.mean(vals)),
                'std':  float(np.std(vals)),
                'vals': vals
            }

        # Print comparison table
        print_comparison_table(mirage_summary)

        # Generate figures
        logger.info("Generating figures...")
        roc_auc_combined = fig_roc(
            all_y_true, all_y_prob,
            PROJECT_ROOT / 'fig4_mirage_roc.png'
        )
        fig_confusion(
            all_y_true, all_y_prob,
            PROJECT_ROOT / 'fig5_mirage_confusion.png'
        )
        fig_comparison(
            mirage_summary,
            PROJECT_ROOT / 'fig6_mirage_vs_baseline.png'
        )

        logger.info(f"\n✓ MIRAGE training complete!")
        logger.info(f"  Combined ROC-AUC (all folds): {roc_auc_combined:.4f}")
        logger.info(f"  Mean ROC-AUC (5-fold):        "
                    f"{mirage_summary['roc_auc']['mean']:.4f} ± "
                    f"{mirage_summary['roc_auc']['std']:.4f}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
