#!/usr/bin/env python3
"""
OASIS-1 Graph Construction for PyTorch Geometric
=================================================
Production-grade script to fuse tabular metadata and volumetric features
into PyTorch Geometric Data objects for Graph Neural Network processing.

Phase 4: Graph Construction
Author: PyTorch Geometric Data Engineering Team
Date: 2026-04-11
"""

import logging
from pathlib import Path
from typing import List, Tuple
import warnings

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

MASTER_INDEX_PATH = Path('master_index.csv')
NODE_FEATURES_PATH = Path('node_features.csv')
OUTPUT_GRAPH_PATH = Path('oasis_graphs.pt')

# Feature columns for normalization
VOLUME_FEATURES = ['CSF_volume_mm3', 'GrayMatter_volume_mm3', 'WhiteMatter_volume_mm3']
TABULAR_FEATURES = ['Age', 'Educ', 'eTIV', 'nWBV']


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
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_and_merge_data(
    master_path: Path,
    node_features_path: Path
) -> pd.DataFrame:
    """
    Load and merge master index with node features.
    
    Args:
        master_path: Path to master_index.csv
        node_features_path: Path to node_features.csv
        
    Returns:
        Merged DataFrame with all features
    """
    logger.info("Loading data files...")
    
    try:
        # Load master index
        master_df = pd.read_csv(master_path)
        logger.info(f"Loaded master index: {len(master_df)} subjects")
        
        # Load node features
        node_df = pd.read_csv(node_features_path)
        logger.info(f"Loaded node features: {len(node_df)} subjects")
        
        # Merge on subject ID
        # master_index uses 'ID', node_features uses 'subject_id'
        merged_df = master_df.merge(
            node_df,
            left_on='ID',
            right_on='subject_id',
            how='inner'
        )
        
        logger.info(f"Merged dataset: {len(merged_df)} subjects")
        
        # Verify required columns exist
        required_cols = VOLUME_FEATURES + TABULAR_FEATURES + ['AD_label']
        missing_cols = [col for col in required_cols if col not in merged_df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        logger.info(f"All required columns present: {required_cols}")
        
        return merged_df
        
    except Exception as e:
        logger.error(f"Failed to load and merge data: {e}")
        raise


def normalize_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler, StandardScaler]:
    """
    Normalize volume and tabular features using StandardScaler.
    
    Args:
        df: DataFrame with raw features
        
    Returns:
        Tuple of (normalized_df, volume_scaler, tabular_scaler)
    """
    logger.info("Normalizing features...")
    
    # Create a copy to avoid modifying original
    df_normalized = df.copy()
    
    # Normalize volume features
    volume_scaler = StandardScaler()
    df_normalized[VOLUME_FEATURES] = volume_scaler.fit_transform(df[VOLUME_FEATURES])
    
    logger.info("Volume features normalized:")
    for feat in VOLUME_FEATURES:
        mean_val = df_normalized[feat].mean()
        std_val = df_normalized[feat].std()
        logger.info(f"  {feat}: mean={mean_val:.4f}, std={std_val:.4f}")
    
    # Normalize tabular features
    tabular_scaler = StandardScaler()
    df_normalized[TABULAR_FEATURES] = tabular_scaler.fit_transform(df[TABULAR_FEATURES])
    
    logger.info("Tabular features normalized:")
    for feat in TABULAR_FEATURES:
        mean_val = df_normalized[feat].mean()
        std_val = df_normalized[feat].std()
        logger.info(f"  {feat}: mean={mean_val:.4f}, std={std_val:.4f}")
    
    return df_normalized, volume_scaler, tabular_scaler


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def create_edge_index() -> torch.Tensor:
    """
    Create edge index for a fully connected 3-node graph (no self-loops).
    
    Edges:
    - Node 0 (CSF) <-> Node 1 (Gray Matter)
    - Node 1 (Gray Matter) <-> Node 2 (White Matter)
    - Node 0 (CSF) <-> Node 2 (White Matter)
    
    Returns:
        Edge index tensor of shape [2, 6]
    """
    # Fully connected graph with 3 nodes (bidirectional, no self-loops)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 0, 2],  # Source nodes
        [1, 0, 2, 1, 2, 0]   # Target nodes
    ], dtype=torch.long)
    
    return edge_index


def create_graph_for_subject(row: pd.Series, edge_index: torch.Tensor) -> Data:
    """
    Create a PyTorch Geometric Data object for a single subject.
    
    Graph structure:
    - 3 nodes representing CSF, Gray Matter, and White Matter
    - Each node has features: [tissue_volume, Age, Educ, eTIV, nWBV]
    - Fully connected graph (6 bidirectional edges)
    - Binary label for AD classification
    
    Args:
        row: DataFrame row with normalized features
        edge_index: Pre-computed edge index tensor
        
    Returns:
        PyTorch Geometric Data object
    """
    # Extract normalized features
    csf_vol = row['CSF_volume_mm3']
    gm_vol = row['GrayMatter_volume_mm3']
    wm_vol = row['WhiteMatter_volume_mm3']
    
    age = row['Age']
    educ = row['Educ']
    etiv = row['eTIV']
    nwbv = row['nWBV']
    
    # Construct node feature matrix [3 nodes, 5 features each]
    # Node 0: CSF
    # Node 1: Gray Matter
    # Node 2: White Matter
    x = torch.tensor([
        [csf_vol, age, educ, etiv, nwbv],  # Node 0: CSF
        [gm_vol, age, educ, etiv, nwbv],   # Node 1: Gray Matter
        [wm_vol, age, educ, etiv, nwbv]    # Node 2: White Matter
    ], dtype=torch.float)
    
    # Extract label
    y = torch.tensor([row['AD_label']], dtype=torch.long)
    
    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, y=y)
    
    return data


def build_all_graphs(df: pd.DataFrame) -> List[Data]:
    """
    Build PyTorch Geometric graphs for all subjects.
    
    Args:
        df: DataFrame with normalized features
        
    Returns:
        List of PyTorch Geometric Data objects
    """
    logger.info("=" * 60)
    logger.info("BUILDING GRAPHS")
    logger.info("=" * 60)
    
    # Create edge index once (same for all graphs)
    edge_index = create_edge_index()
    logger.info(f"Edge index shape: {edge_index.shape}")
    logger.info(f"Edge index:\n{edge_index}")
    
    graph_list = []
    
    for idx, row in df.iterrows():
        try:
            graph = create_graph_for_subject(row, edge_index)
            graph_list.append(graph)
            
            if (idx + 1) % 50 == 0:
                logger.info(f"Created {idx + 1}/{len(df)} graphs...")
                
        except Exception as e:
            logger.error(f"Failed to create graph for subject {row.get('ID', 'unknown')}: {e}")
            continue
    
    logger.info("=" * 60)
    logger.info(f"GRAPH CONSTRUCTION COMPLETE: {len(graph_list)} graphs created")
    logger.info("=" * 60)
    
    return graph_list


# ============================================================================
# SAVING & VALIDATION
# ============================================================================

def save_graphs(graph_list: List[Data], output_path: Path) -> None:
    """
    Save graph list to disk using torch.save.
    
    Args:
        graph_list: List of PyTorch Geometric Data objects
        output_path: Path to save the graphs
    """
    logger.info(f"Saving graphs to: {output_path.resolve()}")
    
    try:
        torch.save(graph_list, output_path)
        
        if output_path.exists():
            file_size = output_path.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"Graphs saved successfully ({file_size:.2f} MB)")
        else:
            logger.error("Failed to save graphs")
            
    except Exception as e:
        logger.error(f"Error saving graphs: {e}")
        raise


def print_graph_summary(graph_list: List[Data], df: pd.DataFrame) -> None:
    """
    Print comprehensive summary of the graph dataset.
    
    Args:
        graph_list: List of PyTorch Geometric Data objects
        df: Original DataFrame for statistics
    """
    print("\n" + "=" * 60)
    print("GRAPH DATASET SUMMARY")
    print("=" * 60)
    print(f"Total graphs: {len(graph_list)}")
    print("=" * 60)
    
    # Class distribution
    labels = [graph.y.item() for graph in graph_list]
    healthy_count = sum(1 for label in labels if label == 0)
    dementia_count = sum(1 for label in labels if label == 1)
    
    print("\nCLASS DISTRIBUTION:")
    print(f"  Healthy controls (y=0): {healthy_count}")
    print(f"  Dementia cases (y=1): {dementia_count}")
    print(f"  Class balance: {healthy_count/len(graph_list)*100:.1f}% / {dementia_count/len(graph_list)*100:.1f}%")
    
    # Graph structure
    print("\nGRAPH STRUCTURE:")
    print(f"  Nodes per graph: {graph_list[0].num_nodes}")
    print(f"  Edges per graph: {graph_list[0].num_edges}")
    print(f"  Node features: {graph_list[0].num_node_features}")
    print(f"  Undirected: {graph_list[0].is_undirected()}")
    
    # Feature details
    print("\nNODE FEATURE COMPOSITION:")
    print("  Each node has 5 features:")
    print("    [0] Tissue volume (normalized)")
    print("    [1] Age (normalized)")
    print("    [2] Education years (normalized)")
    print("    [3] eTIV - Estimated Total Intracranial Volume (normalized)")
    print("    [4] nWBV - Normalized Whole Brain Volume (normalized)")
    
    print("\nNODE TYPES:")
    print("  Node 0: CSF (Cerebrospinal Fluid)")
    print("  Node 1: Gray Matter")
    print("  Node 2: White Matter")
    
    # First graph details
    print("\n" + "=" * 60)
    print("FIRST GRAPH DETAILS:")
    print("=" * 60)
    first_graph = graph_list[0]
    print(f"\nGraph object: {first_graph}")
    print(f"\nNode features (x):\n{first_graph.x}")
    print(f"\nEdge index:\n{first_graph.edge_index}")
    print(f"\nLabel (y): {first_graph.y.item()}")
    
    # Statistics
    print("\n" + "=" * 60)
    print("FEATURE STATISTICS (after normalization):")
    print("=" * 60)
    
    # Collect all node features
    all_features = torch.cat([graph.x for graph in graph_list], dim=0)
    feature_names = ['Tissue Volume', 'Age', 'Education', 'eTIV', 'nWBV']
    
    print("\nNode feature statistics:")
    for i, name in enumerate(feature_names):
        mean = all_features[:, i].mean().item()
        std = all_features[:, i].std().item()
        min_val = all_features[:, i].min().item()
        max_val = all_features[:, i].max().item()
        print(f"  {name}:")
        print(f"    Mean: {mean:.4f}, Std: {std:.4f}")
        print(f"    Range: [{min_val:.4f}, {max_val:.4f}]")
    
    print("\n" + "=" * 60)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function for graph construction.
    
    Pipeline:
    1. Load and merge data
    2. Normalize features
    3. Build graphs for all subjects
    4. Save graphs
    5. Display summary
    """
    logger.info("=" * 60)
    logger.info("OASIS-1 GRAPH CONSTRUCTION")
    logger.info("=" * 60)
    logger.info(f"Master index: {MASTER_INDEX_PATH.resolve()}")
    logger.info(f"Node features: {NODE_FEATURES_PATH.resolve()}")
    logger.info(f"Output graphs: {OUTPUT_GRAPH_PATH.resolve()}")
    logger.info("=" * 60)
    
    try:
        # Step 1: Load and merge data
        merged_df = load_and_merge_data(MASTER_INDEX_PATH, NODE_FEATURES_PATH)
        
        # Step 2: Normalize features
        normalized_df, volume_scaler, tabular_scaler = normalize_features(merged_df)
        
        # Step 3: Build graphs
        graph_list = build_all_graphs(normalized_df)
        
        if len(graph_list) == 0:
            logger.error("No graphs created! Check data integrity.")
            return
        
        # Step 4: Save graphs
        save_graphs(graph_list, OUTPUT_GRAPH_PATH)
        
        # Step 5: Display summary
        print_graph_summary(graph_list, normalized_df)
        
        logger.info("\n✓ Graph construction complete!")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
