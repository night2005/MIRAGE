#!/usr/bin/env python3
"""
OASIS-1 2.5D Slice Extraction
==============================
Extracts tissue-masked 2.5D slices (axial/coronal/sagittal) from all 296
subjects and saves them as a compressed numpy archive for Colab CNN training.

For each subject × tissue type (CSF, GM, WM):
  1. Mask the normalized MRI with the tissue segmentation
  2. Find the tissue center of mass
  3. Extract axial, coronal, sagittal slices at that center
  4. Resize each slice to 64×64
  5. Stack as (3, 64, 64) — 3 channels = 3 anatomical planes

Output shape per subject: (3, 3, 64, 64)  →  tissue × plane × H × W

Phase 9a: 2.5D Slice Extraction (local Mac preprocessing)
Author: Medical Data Engineering Team
Date: 2026-04-19
"""

import logging
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import center_of_mass
from skimage.transform import resize

# Suppress all warnings including nibabel's "very large origin values" spam
warnings.filterwarnings('ignore')

# Silence nibabel's internal logger specifically — it emits an INFO-level
# "very large origin values" message for every ANALYZE 7.5 file loaded,
# which floods the terminal with hundreds of duplicate lines.
logging.getLogger('nibabel').setLevel(logging.ERROR)
logging.getLogger('nibabel.analyze').setLevel(logging.ERROR)
logging.getLogger('nibabel.spm2analyze').setLevel(logging.ERROR)

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT  = Path('/Users/yuvalshah/Desktop/ATML_PROJECT')
DATA_ROOT     = PROJECT_ROOT
DISCS         = [f'disc{i}' for i in range(1, 9)]
MASTER_INDEX  = PROJECT_ROOT / 'master_index.csv'
OUTPUT_NPZ    = PROJECT_ROOT / 'slice_dataset.npz'
OUTPUT_META   = PROJECT_ROOT / 'slice_metadata.csv'

SLICE_SIZE    = 64          # resize all slices to 64×64
TISSUE_LABELS = {           # FSL FAST segmentation labels
    'CSF': 1,
    'GM':  2,
    'WM':  3,
}

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Re-silence nibabel after basicConfig (basicConfig can re-enable root handlers)
logging.getLogger('nibabel').setLevel(logging.ERROR)
logging.getLogger('nibabel.analyze').setLevel(logging.ERROR)
logging.getLogger('nibabel.spm2analyze').setLevel(logging.ERROR)


# ============================================================================
# PATH RESOLVER
# ============================================================================

def find_subject_files(subject_dir: Path):
    """
    Resolve t88_masked_gfc.img and _fseg.img paths for a subject.
    Uses glob to handle _n3_ vs _n4_ naming variation.

    Returns:
        (t88_path, fseg_path) as Path objects, or (None, None) if missing.
    """
    t88_matches = glob.glob(
        str(subject_dir / 'PROCESSED' / 'MPRAGE' / 'T88_111' / '*_t88_masked_gfc.img')
    )
    fseg_matches = glob.glob(
        str(subject_dir / 'FSL_SEG' / '*_fseg.img')
    )

    t88  = Path(t88_matches[0])  if t88_matches  else None
    fseg = Path(fseg_matches[0]) if fseg_matches else None
    return t88, fseg


# ============================================================================
# NORMALIZATION
# ============================================================================

def normalize_volume(vol: np.ndarray) -> np.ndarray:
    """
    Per-subject z-score normalization using only non-zero (brain) voxels.
    Zero voxels (background/skull-strip) remain zero after normalization.
    """
    brain_mask = vol > 0
    if brain_mask.sum() == 0:
        return vol.astype(np.float32)

    mean = vol[brain_mask].mean()
    std  = vol[brain_mask].std()

    if std < 1e-6:          # degenerate volume guard
        return vol.astype(np.float32)

    normalized = np.zeros_like(vol, dtype=np.float32)
    normalized[brain_mask] = (vol[brain_mask] - mean) / std
    return normalized


# ============================================================================
# SLICE EXTRACTION
# ============================================================================

def extract_tissue_slices(
    mri_norm: np.ndarray,
    fseg: np.ndarray,
    tissue_label: int
) -> np.ndarray:
    """
    Extract a (3, 64, 64) 2.5D representation for one tissue type.

    Steps:
      1. Build binary tissue mask from fseg label
      2. Apply mask to normalized MRI
      3. Find center of mass of the tissue mask
      4. Extract axial / coronal / sagittal slices at CoM
      5. Resize each to 64×64
      6. Stack → (3, 64, 64)

    Args:
        mri_norm:     Normalized MRI volume, shape (X, Y, Z), float32
        fseg:         Segmentation mask, shape (X, Y, Z), int
        tissue_label: Integer label (1=CSF, 2=GM, 3=WM)

    Returns:
        np.ndarray of shape (3, 64, 64), float32
    """
    X, Y, Z = mri_norm.shape

    # Step 1 — binary tissue mask
    tissue_mask = (fseg == tissue_label).astype(np.float32)

    # Step 2 — masked MRI (zero outside tissue)
    masked_vol = mri_norm * tissue_mask

    # Step 3 — center of mass (falls back to volume center if tissue absent)
    if tissue_mask.sum() > 0:
        cx, cy, cz = center_of_mass(tissue_mask)
        cx = int(np.clip(round(cx), 0, X - 1))
        cy = int(np.clip(round(cy), 0, Y - 1))
        cz = int(np.clip(round(cz), 0, Z - 1))
    else:
        cx, cy, cz = X // 2, Y // 2, Z // 2

    # Step 4 — extract three orthogonal slices at CoM
    axial    = masked_vol[cx, :, :]        # shape (Y, Z)
    coronal  = masked_vol[:, cy, :]        # shape (X, Z)
    sagittal = masked_vol[:, :, cz]        # shape (X, Y)

    # Step 5 — resize each to 64×64
    def to_64(arr):
        resized = resize(
            arr,
            (SLICE_SIZE, SLICE_SIZE),
            order=1,                    # bilinear
            mode='constant',
            cval=0.0,
            anti_aliasing=True,
            preserve_range=True
        )
        return resized.astype(np.float32)

    axial_64    = to_64(axial)
    coronal_64  = to_64(coronal)
    sagittal_64 = to_64(sagittal)

    # Step 6 — stack → (3, 64, 64)
    return np.stack([axial_64, coronal_64, sagittal_64], axis=0)


# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("=" * 60)
    logger.info("OASIS-1 2.5D SLICE EXTRACTION")
    logger.info("=" * 60)
    logger.info(f"Data root : {DATA_ROOT}")
    logger.info(f"Output NPZ: {OUTPUT_NPZ}")
    logger.info(f"Output CSV: {OUTPUT_META}")
    logger.info(f"Slice size: {SLICE_SIZE}×{SLICE_SIZE}")
    logger.info("=" * 60)

    # ── Load master_index for CDR/AD_label lookup ────────────────────────────
    master_df = pd.read_csv(MASTER_INDEX)
    master_df['AD_label'] = (master_df['CDR'] > 0.0).astype(int)
    label_lookup = master_df.set_index('ID')[['CDR', 'AD_label']].to_dict('index')
    logger.info(f"Loaded master_index: {len(master_df)} labeled subjects")

    # ── Collect all subject directories ─────────────────────────────────────
    subject_dirs = []
    for disc in DISCS:
        disc_path = DATA_ROOT / disc
        if not disc_path.exists():
            logger.warning(f"Disc not found: {disc_path}")
            continue
        dirs = sorted([
            d for d in disc_path.iterdir()
            if d.is_dir() and d.name.startswith('OAS1_')
        ])
        subject_dirs.extend(dirs)

    logger.info(f"Found {len(subject_dirs)} subject folders")
    logger.info("Starting extraction...\n")

    # ── Processing loop ──────────────────────────────────────────────────────
    slice_data   = {}   # subject_id -> np.ndarray (3, 3, 64, 64)
    meta_records = []
    failed       = []

    for idx, subject_dir in enumerate(subject_dirs, 1):
        subject_id = subject_dir.name
        disc_num   = int(subject_dir.parts[-2].replace('disc', ''))

        try:
            # Resolve file paths
            t88_path, fseg_path = find_subject_files(subject_dir)

            if t88_path is None:
                raise FileNotFoundError(f"t88_masked_gfc.img not found")
            if fseg_path is None:
                raise FileNotFoundError(f"_fseg.img not found")

            # Load MRI volume
            mri_img  = nib.load(str(t88_path))
            mri_vol  = mri_img.get_fdata().squeeze().astype(np.float32)

            # Load segmentation mask
            fseg_img = nib.load(str(fseg_path))
            fseg_vol = np.round(fseg_img.get_fdata()).squeeze().astype(np.int8)

            # Normalize MRI per-subject
            mri_norm = normalize_volume(mri_vol)

            # Extract slices for each tissue type → (3, 3, 64, 64)
            tissue_slices = []
            for tissue_name, tissue_label in TISSUE_LABELS.items():
                slices = extract_tissue_slices(mri_norm, fseg_vol, tissue_label)
                tissue_slices.append(slices)   # each (3, 64, 64)

            subject_array = np.stack(tissue_slices, axis=0)  # (3, 3, 64, 64)
            slice_data[subject_id] = subject_array

            # Metadata record
            cdr_info = label_lookup.get(subject_id, {})
            meta_records.append({
                'subject_id': subject_id,
                'disc':       disc_num,
                'CDR':        cdr_info.get('CDR',      float('nan')),
                'AD_label':   cdr_info.get('AD_label', float('nan')),
            })

        except Exception as e:
            logger.warning(f"FAILED [{subject_id}]: {e}")
            failed.append({'subject_id': subject_id, 'error': str(e)})
            continue

        # Progress
        if idx % 50 == 0:
            logger.info(f"  Progress: {idx}/{len(subject_dirs)} subjects processed...")

    logger.info(f"\nExtraction loop complete.")
    logger.info(f"  Successful : {len(slice_data)}")
    logger.info(f"  Failed     : {len(failed)}")

    # ── Save NPZ ─────────────────────────────────────────────────────────────
    logger.info(f"\nSaving compressed NPZ to {OUTPUT_NPZ} ...")
    np.savez_compressed(str(OUTPUT_NPZ), **slice_data)

    # ── Save metadata CSV ─────────────────────────────────────────────────────
    meta_df = pd.DataFrame(meta_records)
    meta_df.to_csv(OUTPUT_META, index=False)
    logger.info(f"Metadata CSV saved: {OUTPUT_META}")

    # ── Validation report ─────────────────────────────────────────────────────
    npz_size_mb = OUTPUT_NPZ.stat().st_size / 1e6

    # Compute global stats across all arrays
    all_vals = np.concatenate([v.flatten() for v in slice_data.values()])
    nonzero_vals = all_vals[all_vals != 0]

    print()
    print("=" * 60)
    print("EXTRACTION COMPLETE — VALIDATION REPORT")
    print("=" * 60)
    print(f"Total subjects processed : {len(slice_data)} / {len(subject_dirs)}")
    print(f"Failed subjects          : {len(failed)}")
    if failed:
        for f in failed:
            print(f"  ✗ {f['subject_id']}: {f['error']}")
    print()
    print(f"Output array shape per subject : (3, 3, 64, 64)")
    print(f"  Axis 0 — tissue  : CSF | GM | WM")
    print(f"  Axis 1 — plane   : Axial | Coronal | Sagittal")
    print(f"  Axis 2-3 — image : 64 × 64 pixels")
    print()
    print(f"slice_dataset.npz size   : {npz_size_mb:.1f} MB")
    print()
    print(f"Global intensity stats (non-zero voxels):")
    print(f"  Min  : {nonzero_vals.min():.4f}")
    print(f"  Max  : {nonzero_vals.max():.4f}")
    print(f"  Mean : {nonzero_vals.mean():.4f}")
    print(f"  Std  : {nonzero_vals.std():.4f}")
    print()
    print(f"Metadata CSV             : {OUTPUT_META}")
    print(f"  Labeled subjects (CDR known) : {meta_df['CDR'].notna().sum()}")
    print(f"  Unlabeled subjects           : {meta_df['CDR'].isna().sum()}")
    print(f"  AD_label=0 (Healthy)         : {(meta_df['AD_label']==0).sum()}")
    print(f"  AD_label=1 (Dementia)        : {(meta_df['AD_label']==1).sum()}")
    print()
    print(f"✓ Ready to upload slice_dataset.npz + slice_metadata.csv to Google Drive")
    print("=" * 60)


if __name__ == '__main__':
    main()
