#!/usr/bin/env python3
"""
MIRAGE Pipeline — Step 3: Supervised VAE Fine-Tuning
=====================================================
Fine-tunes the VAE encoder on 161 labeled subjects to make the latent
space more discriminative for AD classification, while preserving the
low-level visual features learned during unsupervised pretraining.

Phase 9e: Supervised Fine-Tuning
"""

import logging
import warnings
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT       = Path('/Users/yuvalshah/Desktop/ATML_PROJECT')
VAE_WEIGHTS        = PROJECT_ROOT / 'best_vae.pth'
SLICE_NPZ          = PROJECT_ROOT / 'slice_dataset.npz'
SLICE_META         = PROJECT_ROOT / 'slice_metadata.csv'
MASTER_INDEX       = PROJECT_ROOT / 'master_index.csv'
CLINICAL_SCALER    = PROJECT_ROOT / 'clinical_scaler.pkl'
EMB_SCALER         = PROJECT_ROOT / 'embedding_scaler.pkl'
ORIGINAL_GRAPHS    = PROJECT_ROOT / 'oasis_graphs.pt'

OUTPUT_WEIGHTS     = PROJECT_ROOT / 'best_finetune.pth'
OUTPUT_EMBEDDINGS  = PROJECT_ROOT / 'vae_embeddings_finetuned.npz'
OUTPUT_EMB_SCALER_FT = PROJECT_ROOT / 'embedding_scaler_finetuned.pkl'
OUTPUT_GRAPHS_V3   = PROJECT_ROOT / 'oasis_graphs_v3.pt'
OUTPUT_LOSS_CURVE  = PROJECT_ROOT / 'finetune_loss_curve.png'

CLINICAL_COLS      = ['Age', 'Educ', 'eTIV', 'nWBV']
LATENT_DIM         = 64
CLINICAL_DIM       = 5   # Age, Educ, eTIV, nWBV, Age×nWBV
BATCH_SIZE         = 16
MAX_EPOCHS         = 100
PATIENCE_ES        = 10
PATIENCE_LR        = 5
POS_WEIGHT         = 93 / 68
RANDOM_STATE       = 42

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
logging.getLogger('nibabel').setLevel(logging.ERROR)


# ============================================================================
# STEP 1 — BrainVAE Architecture (identical to 09b_vae_train.ipynb)
# ============================================================================

class BrainVAE(nn.Module):
    """Exact replica of the architecture used in 09b_vae_train.ipynb."""

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
        )
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.LeakyReLU(0.2),
        )
        self.fc_mu     = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128 * 8 * 8),
            nn.LeakyReLU(0.2),
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns mu only — used for embedding extraction."""
        h = self.encoder_fc(self.encoder_conv(x))
        return self.fc_mu(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def decode(self, z):
        return self.decoder_conv(self.decoder_fc(z).view(-1, 128, 8, 8))

    def forward(self, x):
        h      = self.encoder_fc(self.encoder_conv(x))
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar


# ============================================================================
# STEP 3 — Classification Head and FinetuneModel
# ============================================================================

class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int = LATENT_DIM + CLINICAL_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


class FinetuneModel(nn.Module):
    """
    Combines frozen VAE encoder with a trainable classification head.
    Input:  slices (3, 3, 64, 64) + clinical (5,)
    Output: logit (1,)
    """

    def __init__(self, vae: BrainVAE, emb_scaler, device):
        super().__init__()
        self.vae        = vae
        self.head       = ClassificationHead(LATENT_DIM + CLINICAL_DIM)
        self.emb_scaler = emb_scaler
        self.device     = device

    def forward(self, slices: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
        # slices: (B, 3, 3, 64, 64)  B=batch, 3 tissues, 3 planes, 64x64
        B = slices.shape[0]
        tissue_embeddings = []
        for t in range(3):
            tissue_input = slices[:, t, :, :, :]   # (B, 3, 64, 64)
            mu = self.vae.encode(tissue_input)       # (B, 64)
            tissue_embeddings.append(mu)
        # Mean pool across 3 tissues → (B, 64)
        pooled = torch.stack(tissue_embeddings, dim=1).mean(dim=1)

        # Apply embedding scaler (numpy → back to tensor)
        pooled_np   = pooled.detach().cpu().numpy()
        pooled_norm = self.emb_scaler.transform(pooled_np)
        pooled_t    = torch.tensor(pooled_norm, dtype=torch.float32).to(self.device)

        # Concatenate clinical → (B, 69)
        fused = torch.cat([pooled_t, clinical], dim=1)
        return self.head(fused)


# ============================================================================
# STEP 4 — Dataset
# ============================================================================

class LabeledBrainDataset(Dataset):
    def __init__(self, subject_ids: List[str], npz, clinical_array: np.ndarray,
                 labels: np.ndarray):
        self.subject_ids   = subject_ids
        self.npz           = npz
        self.clinical      = clinical_array.astype(np.float32)
        self.labels        = labels.astype(np.float32)

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        sid      = self.subject_ids[idx]
        slices   = torch.tensor(self.npz[sid].astype(np.float32))  # (3,3,64,64)
        clinical = torch.tensor(self.clinical[idx])                 # (4,)
        label    = torch.tensor(self.labels[idx])                   # scalar
        return slices, clinical, label, sid


# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("=" * 60)
    logger.info("STEP 9e: SUPERVISED VAE FINE-TUNING")
    logger.info("=" * 60)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Device: MPS (Apple Metal GPU)")
    else:
        device = torch.device('cpu')
        logger.info("Device: CPU")

    try:
        # ── STEP 1: Load BrainVAE and weights ───────────────────────────────
        logger.info("\n" + "─" * 50)
        logger.info("STEP 1 — Loading BrainVAE weights")
        logger.info("─" * 50)

        vae = BrainVAE(latent_dim=LATENT_DIM)
        state = torch.load(str(VAE_WEIGHTS), map_location='cpu', weights_only=False)
        vae.load_state_dict(state)
        logger.info(f"  Loaded weights from {VAE_WEIGHTS}")

        # ── STEP 2: Freeze all, then unfreeze encoder_fc.1, fc_mu, fc_logvar ─
        logger.info("\n" + "─" * 50)
        logger.info("STEP 2 — Selective freezing")
        logger.info("─" * 50)

        # Freeze everything
        for param in vae.parameters():
            param.requires_grad = False

        # Unfreeze: encoder_fc[1] (the Linear layer, index 1 in Sequential),
        # fc_mu, fc_logvar
        for param in vae.encoder_fc[1].parameters():
            param.requires_grad = True
        for param in vae.fc_mu.parameters():
            param.requires_grad = True
        for param in vae.fc_logvar.parameters():
            param.requires_grad = True

        # Parameter summary
        print("\nParameter freeze status:")
        print(f"  {'Layer':<35} {'Trainable':>10}  {'Params':>10}")
        print("  " + "-" * 60)
        total_trainable = 0
        total_frozen    = 0
        for name, param in vae.named_parameters():
            n = param.numel()
            status = "TRAINABLE" if param.requires_grad else "frozen"
            if param.requires_grad:
                total_trainable += n
            else:
                total_frozen += n
            print(f"  {name:<35} {status:>10}  {n:>10,}")
        print(f"\n  Total trainable (encoder): {total_trainable:,}")
        print(f"  Total frozen:              {total_frozen:,}")

        vae = vae.to(device)

        # ── Load scalers ─────────────────────────────────────────────────────
        clin_scaler = joblib.load(str(CLINICAL_SCALER))
        emb_scaler  = joblib.load(str(EMB_SCALER))
        logger.info(f"\n  Clinical scaler loaded ({clin_scaler.n_features_in_} features)")
        logger.info(f"  Embedding scaler loaded ({emb_scaler.n_features_in_} features)")

        # ── STEP 4: Data preparation ─────────────────────────────────────────
        logger.info("\n" + "─" * 50)
        logger.info("STEP 4 — Data preparation")
        logger.info("─" * 50)

        npz      = np.load(str(SLICE_NPZ), allow_pickle=True)
        meta_df  = pd.read_csv(str(SLICE_META))
        master   = pd.read_csv(str(MASTER_INDEX))

        # Use only labeled subjects
        labeled  = meta_df.dropna(subset=['CDR']).copy()
        labeled['AD_label'] = (labeled['CDR'] > 0.0).astype(int)
        logger.info(f"  Labeled subjects: {len(labeled)}")

        # Merge clinical features
        merged = labeled.merge(
            master[['ID'] + CLINICAL_COLS],
            left_on='subject_id', right_on='ID', how='inner'
        )
        logger.info(f"  After merge with clinical: {len(merged)}")

        subject_ids = merged['subject_id'].tolist()
        labels_arr  = merged['AD_label'].values
        # Fill missing, add Age×nWBV interaction BEFORE scaling
        merged_clin = merged[CLINICAL_COLS].fillna(merged[CLINICAL_COLS].median()).copy()
        merged_clin['Age_nWBV'] = merged_clin['Age'] * merged_clin['nWBV']
        clinical_raw = merged_clin[CLINICAL_COLS + ['Age_nWBV']].values
        clinical_scaled = clin_scaler.transform(clinical_raw).astype(np.float32)

        # Stratified 80/20 split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2,
                                     random_state=RANDOM_STATE)
        train_idx, val_idx = next(sss.split(subject_ids, labels_arr))

        train_ids  = [subject_ids[i] for i in train_idx]
        val_ids    = [subject_ids[i] for i in val_idx]
        train_clin = clinical_scaled[train_idx]
        val_clin   = clinical_scaled[val_idx]
        train_lbl  = labels_arr[train_idx]
        val_lbl    = labels_arr[val_idx]

        train_ds = LabeledBrainDataset(train_ids, npz, train_clin, train_lbl)
        val_ds   = LabeledBrainDataset(val_ids,   npz, val_clin,   val_lbl)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                  shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=0)

        print(f"\n  Train: {len(train_ds)} subjects  "
              f"(healthy={sum(train_lbl==0)}, dementia={sum(train_lbl==1)})")
        print(f"  Val:   {len(val_ds)} subjects  "
              f"(healthy={sum(val_lbl==0)}, dementia={sum(val_lbl==1)})")

        # ── STEP 3 + 5: Build FinetuneModel and optimizer ────────────────────
        logger.info("\n" + "─" * 50)
        logger.info("STEP 3+5 — FinetuneModel and optimizer")
        logger.info("─" * 50)

        model = FinetuneModel(vae, emb_scaler, device).to(device)

        # Differential learning rates
        encoder_params = list(vae.encoder_fc[1].parameters()) + \
                         list(vae.fc_mu.parameters()) + \
                         list(vae.fc_logvar.parameters())
        head_params    = list(model.head.parameters())

        optimizer = torch.optim.Adam([
            {'params': encoder_params, 'lr': 1e-4},
            {'params': head_params,    'lr': 1e-3},
        ], weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=PATIENCE_LR, factor=0.5
        )

        pos_w     = torch.tensor([POS_WEIGHT], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        total_head_params = sum(p.numel() for p in head_params)
        total_enc_params  = sum(p.numel() for p in encoder_params)
        print(f"\n  Encoder trainable params: {total_enc_params:,}  lr=1e-4")
        print(f"  Head trainable params:    {total_head_params:,}  lr=1e-3")

        # ── STEP 6: Training loop ─────────────────────────────────────────────
        logger.info("\n" + "─" * 50)
        logger.info("STEP 6 — Training")
        logger.info("─" * 50)

        train_losses, val_losses, val_aucs = [], [], []
        best_val_auc   = 0.0
        patience_count = 0
        best_epoch     = 0

        print(f"\n{'Epoch':>6} | {'Train Loss':>11} | {'Val Loss':>10} | "
              f"{'Val AUC':>9} | {'LR enc':>10} | {'LR head':>10}")
        print("-" * 72)

        for epoch in range(1, MAX_EPOCHS + 1):
            # Train
            model.train()
            epoch_train = 0.0
            for slices, clinical, labels_b, _ in train_loader:
                slices   = slices.to(device)
                clinical = clinical.to(device)
                labels_b = labels_b.to(device)

                optimizer.zero_grad()
                logits = model(slices, clinical).view(-1)
                loss   = criterion(logits, labels_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    encoder_params + head_params, max_norm=1.0
                )
                optimizer.step()
                epoch_train += loss.item() * slices.size(0)
            epoch_train /= len(train_loader.dataset)

            # Validate
            model.eval()
            epoch_val  = 0.0
            all_probs  = []
            all_labels = []
            with torch.no_grad():
                for slices, clinical, labels_b, _ in val_loader:
                    slices   = slices.to(device)
                    clinical = clinical.to(device)
                    labels_b = labels_b.to(device)
                    logits   = model(slices, clinical).view(-1)
                    loss     = criterion(logits, labels_b)
                    epoch_val += loss.item() * slices.size(0)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    all_probs.extend(probs.tolist())
                    all_labels.extend(labels_b.cpu().numpy().tolist())
            epoch_val /= len(val_loader.dataset)

            try:
                val_auc = roc_auc_score(all_labels, all_probs)
            except Exception:
                val_auc = 0.5

            scheduler.step(val_auc)
            lr_enc  = optimizer.param_groups[0]['lr']
            lr_head = optimizer.param_groups[1]['lr']

            train_losses.append(epoch_train)
            val_losses.append(epoch_val)
            val_aucs.append(val_auc)

            print(f"{epoch:>6} | {epoch_train:>11.5f} | {epoch_val:>10.5f} | "
                  f"{val_auc:>9.4f} | {lr_enc:>10.2e} | {lr_head:>10.2e}")

            if val_auc > best_val_auc:
                best_val_auc  = val_auc
                best_epoch    = epoch
                patience_count = 0
                torch.save({
                    'vae_state':  vae.state_dict(),
                    'head_state': model.head.state_dict(),
                    'epoch':      epoch,
                    'val_loss':   epoch_val,
                    'val_auc':    val_auc,
                }, str(OUTPUT_WEIGHTS))
            else:
                patience_count += 1

            if patience_count >= PATIENCE_ES:
                logger.info(f"\nEarly stopping at epoch {epoch}. "
                            f"Best epoch: {best_epoch} (val_auc={best_val_auc:.4f})")
                break

        # Loss + AUC curve
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(train_losses, label='Train Loss', lw=2)
        ax1.plot(val_losses,   label='Val Loss',   lw=2)
        ax1.axvline(best_epoch - 1, color='red', linestyle='--',
                    label=f'Best epoch ({best_epoch})')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
        ax1.set_title('Fine-Tuning Loss Curve', fontweight='bold')
        ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2.plot(val_aucs, color='darkgreen', lw=2, label='Val ROC-AUC')
        ax2.axvline(best_epoch - 1, color='red', linestyle='--',
                    label=f'Best epoch ({best_epoch})')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('ROC-AUC')
        ax2.set_title('Validation AUC During Fine-Tuning', fontweight='bold')
        ax2.legend(); ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(str(OUTPUT_LOSS_CURVE), dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Loss curve saved: {OUTPUT_LOSS_CURVE}")

        # ── STEP 7: Extract fine-tuned embeddings ────────────────────────────
        logger.info("\n" + "─" * 50)
        logger.info("STEP 7 — Extracting fine-tuned embeddings")
        logger.info("─" * 50)

        # Load best weights
        ckpt = torch.load(str(OUTPUT_WEIGHTS), map_location='cpu',
                          weights_only=False)
        vae.load_state_dict(ckpt['vae_state'])
        model.head.load_state_dict(ckpt['head_state'])
        vae.eval(); model.eval()
        logger.info(f"  Best checkpoint loaded (epoch {ckpt['epoch']}, "
                    f"val_auc={ckpt['val_auc']:.4f})")

        # Extract embeddings for all 161 labeled subjects (raw, unscaled)
        all_embeddings_raw = {}
        with torch.no_grad():
            for sid in subject_ids:
                arr = npz[sid].astype(np.float32)   # (3, 3, 64, 64)
                tissue_mus = []
                for t in range(3):
                    x_t = torch.tensor(arr[t]).unsqueeze(0).to(device)  # (1,3,64,64)
                    mu  = vae.encode(x_t).squeeze(0).cpu().numpy()       # (64,)
                    tissue_mus.append(mu)
                all_embeddings_raw[sid] = np.stack(tissue_mus, axis=0)  # (3, 64)

        # Print raw stats before normalization
        raw_vals = np.concatenate([v.flatten() for v in all_embeddings_raw.values()])
        logger.info(f"  Raw embedding stats — mean: {raw_vals.mean():.4f}, "
                    f"std: {raw_vals.std():.4f}, "
                    f"min: {raw_vals.min():.4f}, max: {raw_vals.max():.4f}")

        # Fit NEW StandardScaler on fine-tuned embeddings matrix (483, 64)
        all_emb_matrix = np.vstack(
            [all_embeddings_raw[sid] for sid in subject_ids]
        )  # (483, 64)
        ft_emb_scaler = StandardScaler()
        ft_emb_scaler.fit(all_emb_matrix)

        # Save the new scaler
        joblib.dump(ft_emb_scaler, str(OUTPUT_EMB_SCALER_FT))
        logger.info(f"  Fine-tuned embedding scaler saved: {OUTPUT_EMB_SCALER_FT}")

        # Transform all embeddings with the new scaler
        all_embeddings = {}
        for sid in subject_ids:
            raw = all_embeddings_raw[sid]                              # (3, 64)
            all_embeddings[sid] = ft_emb_scaler.transform(raw).astype(np.float32)

        # Save
        np.savez_compressed(str(OUTPUT_EMBEDDINGS), **all_embeddings)
        emb_size_kb = OUTPUT_EMBEDDINGS.stat().st_size / 1024
        logger.info(f"  Saved {len(all_embeddings)} embeddings → "
                    f"{OUTPUT_EMBEDDINGS} ({emb_size_kb:.1f} KB)")

        # Embedding statistics after new normalization
        all_vals = np.concatenate([v.flatten() for v in all_embeddings.values()])
        print(f"\n  Embedding statistics (new fine-tuned scaler):")
        print(f"    Mean: {all_vals.mean():.4f}  (target ~0.0)")
        print(f"    Std:  {all_vals.std():.4f}   (target ~1.0)")
        print(f"    Min:  {all_vals.min():.4f}")
        print(f"    Max:  {all_vals.max():.4f}")

        # Final val AUC report
        print(f"\n  Fine-tune classifier val ROC-AUC: {ckpt['val_auc']:.4f}")
        print(f"  Best epoch: {ckpt['epoch']}")

        # ── STEP 8: Rebuild graphs v3 ─────────────────────────────────────────
        logger.info("\n" + "─" * 50)
        logger.info("STEP 8 — Building oasis_graphs_v3.pt")
        logger.info("─" * 50)

        from torch_geometric.data import Data as PyGData

        master_df = pd.read_csv(str(MASTER_INDEX))
        master_df['AD_label'] = (master_df['CDR'] > 0.0).astype(int)

        # Clinical features — add Age×nWBV interaction before scaling
        master_df[CLINICAL_COLS] = master_df[CLINICAL_COLS].fillna(
            master_df[CLINICAL_COLS].median()
        )
        master_df['Age_nWBV'] = master_df['Age'] * master_df['nWBV']
        clin_scaled_v3 = clin_scaler.transform(
            master_df[CLINICAL_COLS + ['Age_nWBV']].values
        ).astype(np.float32)

        orig_graphs = torch.load(str(ORIGINAL_GRAPHS), weights_only=False)
        subject_ids_ordered = master_df['ID'].tolist()

        new_graphs_v3 = []
        for idx, (orig_g, sid) in enumerate(zip(orig_graphs, subject_ids_ordered)):
            emb = all_embeddings[sid]   # (3, 64) normalized with ft_emb_scaler
            g = PyGData(
                x          = torch.tensor(emb,              dtype=torch.float32),
                edge_index = orig_g.edge_index.clone(),
                y          = torch.tensor([master_df.iloc[idx]['AD_label']],
                                          dtype=torch.long),
                clinical   = torch.tensor(clin_scaled_v3[idx], dtype=torch.float32),
                subject_id = sid,
            )
            new_graphs_v3.append(g)

        torch.save(new_graphs_v3, str(OUTPUT_GRAPHS_V3))
        size_kb = OUTPUT_GRAPHS_V3.stat().st_size / 1024

        print(f"\n  oasis_graphs_v3.pt: {len(new_graphs_v3)} graphs, "
              f"{size_kb:.1f} KB")
        print(f"  First graph x shape:       {new_graphs_v3[0].x.shape}")
        print(f"  First graph clinical shape: {new_graphs_v3[0].clinical.shape}")
        print(f"  AD_label=0: {sum(g.y.item()==0 for g in new_graphs_v3)}")
        print(f"  AD_label=1: {sum(g.y.item()==1 for g in new_graphs_v3)}")

        logger.info("\n✓ 09e_finetune_vae.py complete!")
        logger.info(f"  Outputs:")
        logger.info(f"    {OUTPUT_WEIGHTS}")
        logger.info(f"    {OUTPUT_EMBEDDINGS}")
        logger.info(f"    {OUTPUT_EMB_SCALER_FT}")
        logger.info(f"    {OUTPUT_GRAPHS_V3}")
        logger.info(f"    {OUTPUT_LOSS_CURVE}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
