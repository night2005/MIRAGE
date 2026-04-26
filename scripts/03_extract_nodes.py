#!/usr/bin/env python3
"""
OASIS-1 Node Feature Extraction
================================
Production-grade script to extract volumetric features from 3D MRI segmentation masks.
Processes FSL_SEG tissue segmentation files and generates node features for GNN processing.

Phase 3: Node Feature Extraction
Author: Medical Data Engineering Team
Date: 2026-04-11
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import warnings

import pandas as pd
import numpy as np
import nibabel as nib

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

MASTER_INDEX_PATH = Path('master_index.csv')
OUTPUT_CSV_PATH = Path('node_features.csv')


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    """Configure professional logging for the pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# ============================================================================
# CORE EXTRACTION FUNCTIONS
# ============================================================================

def extract_region_volumes(hdr_path: str) -> Optional[Dict[str, float]]:
    """
    Extract volumetric features from a 3D segmentation mask.
    
    The segmentation mask contains integer labels representing different tissue types:
    - 0: Background (air) - ignored
    - 1: CSF (Cerebrospinal Fluid)
    - 2: Gray Matter
    - 3: White Matter
    
    For each tissue type, calculates total volume in mm³.
    
    NOTE: OASIS-1 t88-registered images have corrupted voxel zoom metadata.
    All images are in 1x1x1 mm voxel space, so volume = voxel count.
    
    Args:
        hdr_path: Path to the .hdr file (ANALYZE format)
        
    Returns:
        Dictionary mapping tissue type to volume_mm3, or None if loading fails
    """
    # Tissue label mapping for FSL FAST segmentation
    TISSUE_LABELS = {
        1: 'CSF_volume_mm3',
        2: 'GrayMatter_volume_mm3',
        3: 'WhiteMatter_volume_mm3'
    }
    
    try:
        # Load the NIfTI/ANALYZE image
        img = nib.load(hdr_path)
        
        # Get the 3D data array and cast to integers
        # Round first to handle floating-point values, then cast to int
        data = np.round(img.get_fdata()).astype(int)
        
        # Safely squeeze to remove singleton dimensions
        data = np.squeeze(data)
        
        # Hardcoded voxel volume for t88 space (1x1x1 mm)
        voxel_volume_mm3 = 1.0
        
        logger.debug(f"Image shape: {data.shape}, Voxel volume: {voxel_volume_mm3} mm³")
        
        # Get unique labels (tissue classes)
        unique_labels = np.unique(data)
        
        logger.debug(f"Unique labels found: {unique_labels}")
        
        # Calculate volume for each tissue type
        tissue_volumes = {}
        
        for label in unique_labels:
            if label == 0:
                # Skip background
                continue
            
            # Count voxels with this label
            voxel_count = np.sum(data == label)
            
            # Calculate total volume in mm³ (equals voxel count for 1x1x1 mm voxels)
            volume_mm3 = voxel_count * voxel_volume_mm3
            
            # Map to tissue name if known, otherwise use generic name
            if label in TISSUE_LABELS:
                tissue_name = TISSUE_LABELS[label]
            else:
                tissue_name = f'region_{int(label)}_volume_mm3'
                logger.warning(f"Unknown tissue label {label} found in {hdr_path}")
            
            tissue_volumes[tissue_name] = volume_mm3
        
        logger.debug(f"Extracted {len(tissue_volumes)} tissue volumes")
        
        return tissue_volumes
        
    except FileNotFoundError:
        logger.warning(f"File not found: {hdr_path}")
        return None
    except Exception as e:
        logger.error(f"Error processing {hdr_path}: {e}")
        return None


# ============================================================================
# DATA PROCESSING
# ============================================================================

def load_master_index(csv_path: Path) -> pd.DataFrame:
    """
    Load the master index CSV.
    
    Args:
        csv_path: Path to master_index.csv
        
    Returns:
        DataFrame with subject metadata and file paths
    """
    logger.info(f"Loading master index from: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} subjects")
        
        # Verify required columns exist
        required_cols = ['ID', 'CDR', 'hdr_path']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to load master index: {e}")
        raise


def process_all_subjects(master_df: pd.DataFrame) -> List[Dict]:
    """
    Process all subjects and extract volumetric features.
    
    Args:
        master_df: DataFrame with subject metadata
        
    Returns:
        List of dictionaries containing subject ID, label, and regional volumes
    """
    logger.info("=" * 60)
    logger.info("EXTRACTING VOLUMETRIC FEATURES")
    logger.info("=" * 60)
    
    feature_records = []
    successful = 0
    failed = 0
    
    for idx, row in master_df.iterrows():
        subject_id = row['ID']
        cdr = row['CDR']
        hdr_path = row['hdr_path']
        
        # Create binary AD label (0 = healthy, 1 = dementia)
        ad_label = 1 if cdr > 0.0 else 0
        
        logger.info(f"Processing [{idx+1}/{len(master_df)}]: {subject_id}")
        
        # Extract regional volumes
        region_volumes = extract_region_volumes(hdr_path)
        
        if region_volumes is None:
            logger.warning(f"  Failed to extract features for {subject_id}")
            failed += 1
            continue
        
        # Create feature record
        record = {
            'subject_id': subject_id,
            'AD_label': ad_label,
            'CDR': cdr
        }
        
        # Add all regional volumes
        record.update(region_volumes)
        
        feature_records.append(record)
        successful += 1
        
        logger.info(f"  Extracted {len(region_volumes)} regional volumes")
        
        # Progress update every 25 subjects
        if (idx + 1) % 25 == 0:
            logger.info(f"Progress: {idx+1}/{len(master_df)} subjects processed")
    
    logger.info("=" * 60)
    logger.info("EXTRACTION COMPLETE")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info("=" * 60)
    
    return feature_records


def create_feature_dataframe(feature_records: List[Dict]) -> pd.DataFrame:
    """
    Convert feature records to DataFrame and handle missing values.
    
    Args:
        feature_records: List of dictionaries with extracted features
        
    Returns:
        Clean DataFrame with all features
    """
    logger.info("Creating feature DataFrame...")
    
    # Convert to DataFrame
    df = pd.DataFrame(feature_records)
    
    # Fill NaN values with 0 (missing tissue classes)
    # This can happen if a subject is missing a specific segmentation label
    df = df.fillna(0)
    
    logger.info(f"Feature DataFrame shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Display summary statistics
    logger.info("\nFeature summary:")
    logger.info(f"  Total subjects: {len(df)}")
    logger.info(f"  Healthy controls (AD_label=0): {(df['AD_label']==0).sum()}")
    logger.info(f"  Dementia cases (AD_label=1): {(df['AD_label']==1).sum()}")
    
    # Count regional volume features
    volume_cols = [col for col in df.columns if 'volume_mm3' in col]
    logger.info(f"  Regional volume features: {len(volume_cols)}")
    
    return df


def save_features(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save feature DataFrame to CSV.
    
    Args:
        df: Feature DataFrame
        output_path: Path to save CSV
    """
    logger.info(f"Saving features to: {output_path.resolve()}")
    
    try:
        df.to_csv(output_path, index=False)
        
        if output_path.exists():
            file_size = output_path.stat().st_size / 1024  # KB
            logger.info(f"File saved successfully ({file_size:.2f} KB)")
        else:
            logger.error("Failed to create output CSV file")
            
    except Exception as e:
        logger.error(f"Error saving CSV: {e}")
        raise


def print_feature_summary(df: pd.DataFrame) -> None:
    """
    Print comprehensive summary of extracted features.
    
    Args:
        df: Feature DataFrame
    """
    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Total subjects: {len(df)}")
    print(f"Total features: {df.shape[1]}")
    print("=" * 60)
    
    print("\nCLASS DISTRIBUTION:")
    print(f"  Healthy controls (AD_label=0): {(df['AD_label']==0).sum()}")
    print(f"  Dementia cases (AD_label=1): {(df['AD_label']==1).sum()}")
    
    print("\nCDR DISTRIBUTION:")
    print(df['CDR'].value_counts().sort_index())
    
    # Analyze tissue volumes
    volume_cols = [col for col in df.columns if 'volume_mm3' in col]
    
    print(f"\nTISSUE VOLUME FEATURES: {len(volume_cols)}")
    if volume_cols:
        print("\nTissue volumes (mean ± std):")
        for col in sorted(volume_cols):
            mean_vol = df[col].mean()
            std_vol = df[col].std()
            min_vol = df[col].min()
            max_vol = df[col].max()
            tissue_name = col.replace('_volume_mm3', '').replace('_', ' ')
            print(f"  {tissue_name}:")
            print(f"    Mean: {mean_vol:.2f} mm³")
            print(f"    Std:  {std_vol:.2f} mm³")
            print(f"    Range: [{min_vol:.2f}, {max_vol:.2f}] mm³")
    
    print("\n" + "=" * 60)
    print("SAMPLE DATA (first 5 rows):")
    print("=" * 60)
    
    # Display key columns
    display_cols = ['subject_id', 'AD_label', 'CDR'] + volume_cols
    available_cols = [col for col in display_cols if col in df.columns]
    print(df[available_cols].head().to_string())
    
    print("\n" + "=" * 60)
    print("DATAFRAME INFO:")
    print("=" * 60)
    df.info()
    
    print("\n" + "=" * 60)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function for node feature extraction.
    
    Pipeline:
    1. Load master index
    2. Process all subjects and extract volumetric features
    3. Create feature DataFrame
    4. Save to CSV
    5. Display summary
    """
    logger.info("=" * 60)
    logger.info("OASIS-1 NODE FEATURE EXTRACTION")
    logger.info("=" * 60)
    logger.info(f"Master index: {MASTER_INDEX_PATH.resolve()}")
    logger.info(f"Output CSV: {OUTPUT_CSV_PATH.resolve()}")
    logger.info("=" * 60)
    
    try:
        # Step 1: Load master index
        master_df = load_master_index(MASTER_INDEX_PATH)
        
        # Step 2: Process all subjects
        feature_records = process_all_subjects(master_df)
        
        if len(feature_records) == 0:
            logger.error("No features extracted! Check file paths and data integrity.")
            return
        
        # Step 3: Create feature DataFrame
        feature_df = create_feature_dataframe(feature_records)
        
        # Step 4: Save to CSV
        save_features(feature_df, OUTPUT_CSV_PATH)
        
        # Step 5: Display summary
        print_feature_summary(feature_df)
        
        logger.info("\n✓ Node feature extraction complete!")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
