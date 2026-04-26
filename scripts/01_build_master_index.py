#!/usr/bin/env python3
"""
OASIS-1 Master Index Builder
=============================
Production-grade script to crawl OASIS-1 disc data, merge with clinical metadata,
and generate a master index CSV for Alzheimer's classification pipeline.

Author: Medical Data Engineering Team
Date: 2026-04-11
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_ROOT = Path('.')  # Parent folder containing disc1, disc2, ..., disc8
METADATA_EXCEL_PATH = Path('/Users/yuvalshah/Desktop/ATML_PROJECT/oasis_cross-sectional-5708aa0a98d82080.xlsx')
OUTPUT_CSV_PATH = Path('master_index.csv')


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
# CORE FUNCTIONS
# ============================================================================

def load_metadata(excel_path: Path) -> pd.DataFrame:
    """
    Load clinical metadata from the official OASIS-1 Excel file.
    
    Args:
        excel_path: Path to the Excel metadata file
        
    Returns:
        DataFrame containing clinical metadata
        
    Raises:
        FileNotFoundError: If Excel file doesn't exist
        Exception: For other loading errors
    """
    logger.info(f"Loading metadata from: {excel_path}")
    
    try:
        if not excel_path.exists():
            raise FileNotFoundError(f"Metadata Excel file not found: {excel_path}")
        
        metadata_df = pd.read_excel(excel_path, engine='openpyxl')
        logger.info(f"Successfully loaded {len(metadata_df)} subjects from metadata")
        logger.info(f"Columns: {list(metadata_df.columns)}")
        
        return metadata_df
        
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        raise


def find_mri_paths(data_root: Path) -> List[Dict[str, str]]:
    """
    Crawl disc directories to find preprocessed MRI files.
    
    Searches for files ending with '_t88_masked_gfc_fseg.hdr' in the FSL_SEG
    folder of each subject directory across all 8 discs.
    
    Args:
        data_root: Root directory containing disc1, disc2, ..., disc8
        
    Returns:
        List of dictionaries with keys: 'ID', 'hdr_path', 'img_path', 'disc'
    """
    logger.info(f"Starting MRI file crawl from: {data_root.resolve()}")
    
    mri_records = []
    total_folders = 0
    found_mri = 0
    
    # Iterate through disc1 to disc8
    for disc_num in range(1, 9):
        disc_dir = data_root / f'disc{disc_num}'
        
        if not disc_dir.exists():
            logger.warning(f"Disc directory not found: {disc_dir}")
            continue
        
        logger.info(f"Scanning {disc_dir}...")
        
        # Find all subject folders (starting with OAS1_)
        try:
            subject_dirs = sorted([
                d for d in disc_dir.iterdir() 
                if d.is_dir() and d.name.startswith('OAS1_')
            ])
        except Exception as e:
            logger.error(f"Error reading disc directory {disc_dir}: {e}")
            continue
        
        for subject_dir in subject_dirs:
            total_folders += 1
            subject_id = subject_dir.name
            
            try:
                # Look in FSL_SEG folder for the t88_masked_gfc_fseg files
                fsl_seg_dir = subject_dir / 'FSL_SEG'
                
                if not fsl_seg_dir.exists():
                    continue
                
                # Find files ending with _t88_masked_gfc_fseg.hdr and .img
                hdr_files = list(fsl_seg_dir.glob('*_t88_masked_gfc_fseg.hdr'))
                img_files = list(fsl_seg_dir.glob('*_t88_masked_gfc_fseg.img'))
                
                if len(hdr_files) > 0 and len(img_files) > 0:
                    hdr_path = hdr_files[0]
                    img_path = img_files[0]
                    
                    # Verify both files exist
                    if hdr_path.exists() and img_path.exists():
                        mri_records.append({
                            'ID': subject_id,
                            'hdr_path': str(hdr_path.resolve()),
                            'img_path': str(img_path.resolve()),
                            'disc': disc_num
                        })
                        found_mri += 1
                        
                        if found_mri % 50 == 0:
                            logger.info(f"Found {found_mri} MRI files...")
                            
            except Exception as e:
                logger.warning(f"Error processing {subject_id}: {e}")
                continue
    
    logger.info("=" * 60)
    logger.info("MRI CRAWL COMPLETE")
    logger.info(f"Total subject folders scanned: {total_folders}")
    logger.info(f"MRI files found: {found_mri}")
    logger.info("=" * 60)
    
    return mri_records


def merge_and_clean_data(
    metadata_df: pd.DataFrame,
    mri_records: List[Dict[str, str]]
) -> pd.DataFrame:
    """
    Merge metadata with MRI file paths and clean the dataset.
    
    Performs inner join to keep only subjects with both metadata and MRI files,
    then drops rows with missing CDR values.
    
    Args:
        metadata_df: DataFrame with clinical metadata
        mri_records: List of dictionaries with MRI file paths
        
    Returns:
        Clean DataFrame ready for analysis
    """
    logger.info("Merging metadata with MRI file paths...")
    
    # Convert MRI records to DataFrame
    mri_df = pd.DataFrame(mri_records)
    logger.info(f"MRI DataFrame created with {len(mri_df)} records")
    
    # Merge on subject ID (inner join to keep only subjects with both)
    logger.info(f"Metadata subjects before merge: {len(metadata_df)}")
    logger.info(f"MRI files found: {len(mri_df)}")
    
    merged_df = metadata_df.merge(mri_df, on='ID', how='inner')
    logger.info(f"After inner merge: {len(merged_df)} subjects")
    
    # Drop rows with missing CDR values
    initial_count = len(merged_df)
    merged_df = merged_df.dropna(subset=['CDR'])
    dropped_cdr = initial_count - len(merged_df)
    
    if dropped_cdr > 0:
        logger.warning(f"Dropped {dropped_cdr} subjects with missing CDR values")
    else:
        logger.info("No subjects with missing CDR values")
    
    # Verify no missing MRI paths
    missing_hdr = merged_df['hdr_path'].isna().sum()
    missing_img = merged_df['img_path'].isna().sum()
    
    if missing_hdr > 0 or missing_img > 0:
        logger.warning(f"Found {missing_hdr} missing HDR and {missing_img} missing IMG paths")
        merged_df = merged_df.dropna(subset=['hdr_path', 'img_path'])
        logger.info(f"After dropping missing MRI paths: {len(merged_df)} subjects")
    
    return merged_df


def save_master_index(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save the master index DataFrame to CSV.
    
    Args:
        df: Clean DataFrame to save
        output_path: Path where CSV should be saved
    """
    logger.info(f"Saving master index to: {output_path.resolve()}")
    
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


def print_summary_statistics(df: pd.DataFrame, metadata_count: int, mri_count: int) -> None:
    """
    Print comprehensive summary statistics about the final dataset.
    
    Args:
        df: Final clean DataFrame
        metadata_count: Original number of subjects in metadata
        mri_count: Number of MRI files found
    """
    print("\n" + "=" * 60)
    print("FINAL DATASET SUMMARY")
    print("=" * 60)
    print(f"Total subjects in metadata Excel: {metadata_count}")
    print(f"Subjects with MRI files found on discs: {mri_count}")
    print(f"Final valid subjects (with MRI + CDR): {len(df)}")
    print(f"Subjects excluded (no MRI): {metadata_count - mri_count}")
    print(f"Subjects excluded (missing CDR): {mri_count - len(df)}")
    print("=" * 60)
    
    print("\nCDR CLASS DISTRIBUTION:")
    print(df['CDR'].value_counts().sort_index())
    
    print("\nCDR PERCENTAGES:")
    cdr_pct = df['CDR'].value_counts(normalize=True).sort_index() * 100
    for cdr_val, pct in cdr_pct.items():
        print(f"  CDR {cdr_val}: {pct:.2f}%")
    
    print("\n" + "=" * 60)
    print("SAMPLE DATA (first 5 rows):")
    print("=" * 60)
    
    # Display key columns
    key_cols = ['ID', 'M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'CDR', 
                'eTIV', 'nWBV', 'ASF', 'hdr_path', 'disc']
    available_cols = [col for col in key_cols if col in df.columns]
    print(df[available_cols].head().to_string())
    
    print("\n" + "=" * 60)
    print("DATAFRAME INFO:")
    print("=" * 60)
    df.info()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function for the OASIS-1 master index builder.
    
    Orchestrates the entire pipeline:
    1. Load metadata from Excel
    2. Crawl discs for MRI files
    3. Merge and clean data
    4. Save master index
    5. Display summary statistics
    """
    logger.info("=" * 60)
    logger.info("OASIS-1 MASTER INDEX BUILDER")
    logger.info("=" * 60)
    logger.info(f"Data root: {DATA_ROOT.resolve()}")
    logger.info(f"Metadata Excel: {METADATA_EXCEL_PATH}")
    logger.info(f"Output CSV: {OUTPUT_CSV_PATH.resolve()}")
    logger.info("=" * 60)
    
    try:
        # Step 1: Load metadata
        metadata_df = load_metadata(METADATA_EXCEL_PATH)
        metadata_count = len(metadata_df)
        
        # Step 2: Find MRI files
        mri_records = find_mri_paths(DATA_ROOT)
        mri_count = len(mri_records)
        
        if mri_count == 0:
            logger.error("No MRI files found! Check DATA_ROOT path and disc structure.")
            return
        
        # Step 3: Merge and clean
        final_df = merge_and_clean_data(metadata_df, mri_records)
        
        if len(final_df) == 0:
            logger.error("No valid subjects after merge and cleaning!")
            return
        
        # Step 4: Save master index
        save_master_index(final_df, OUTPUT_CSV_PATH)
        
        # Step 5: Display summary
        print_summary_statistics(final_df, metadata_count, mri_count)
        
        logger.info("\n✓ Master index build complete!")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
