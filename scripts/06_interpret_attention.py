#!/usr/bin/env python3
"""
OASIS-1 GAT Attention Interpretability
=======================================
Production-grade script to train a GAT model on the full dataset and extract
attention weights to understand which tissue connections drive AD predictions.

Phase 6: Model Interpretability
Author: PyTorch Geometric Deep Learning Team
Date: 2026-04-11
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

GRAPH_DATA_PATH = Path('oasis_graphs.pt')
RANDOM_STATE = 42
BATCH_SIZE = 16
EPOCHS = 80
LEARNING_RATE = 0.005
WEIGHT_DECAY = 1e-4

# Node mapping
NODE_NAMES = {
    0: 'CSF',
    1: 'Gray Matter',
    2: 'White Matter'
}

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


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
# INTERPRETABLE MODEL ARCHITECTURE
# ============================================================================

class InterpretableGAT(nn.Module):
    """
    Interpretable Graph Attention Network for Alzheimer's Disease classification.
    
    This model stores attention weights from the first GAT layer for interpretability.
    
    Architecture:
    - GAT Layer 1: 5 -> 16 features (4 attention heads) - WITH attention weights
    - ELU activation
    - GAT Layer 2: 64 -> 16 features (1 attention head)
    - Global mean pooling
    - Linear classifier: 16 -> 1 (binary classification)
    """
    
    def __init__(self, in_channels: int = 5, hidden_channels: int = 16, 
                 num_heads: int = 4, dropout: float = 0.3):
        """
        Initialize the interpretable GAT model.
        
        Args:
            in_channels: Number of input node features
            hidden_channels: Number of hidden features
            num_heads: Number of attention heads in first layer
            dropout: Dropout probability
        """
        super(InterpretableGAT, self).__init__()
        
        # First GAT layer with multiple attention heads
        self.conv1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=num_heads,
            dropout=dropout
        )
        
        # Second GAT layer (single head, no concatenation)
        self.conv2 = GATConv(
            in_channels=hidden_channels * num_heads,
            out_channels=hidden_channels,
            heads=1,
            concat=False,
            dropout=dropout
        )
        
        # Final classifier
        self.classifier = nn.Linear(hidden_channels, 1)
        
        self.dropout = dropout
        
        # Storage for attention weights (for interpretability)
        self.last_edge_index = None
        self.last_alpha = None
        
    def forward(self, data):
        """
        Forward pass through the network with attention weight extraction.
        
        Args:
            data: PyTorch Geometric Data object with x, edge_index, batch
            
        Returns:
            Logits for binary classification
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # First GAT layer + activation (RETURN ATTENTION WEIGHTS)
        x, (edge_index_out, alpha) = self.conv1(x, edge_index, return_attention_weights=True)
        
        # Store attention weights for interpretability
        self.last_edge_index = edge_index_out
        self.last_alpha = alpha
        
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second GAT layer + activation
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        
        # Global pooling (aggregate node features to graph-level)
        x = global_mean_pool(x, batch)
        
        # Classifier
        x = self.classifier(x)
        
        return x


# ============================================================================
# DATA LOADING
# ============================================================================

def load_graph_dataset(data_path: Path) -> List[Data]:
    """
    Load the graph dataset from disk.
    
    Args:
        data_path: Path to the saved graph dataset
        
    Returns:
        List of PyTorch Geometric Data objects
    """
    logger.info(f"Loading graph dataset from: {data_path}")
    
    try:
        # Load with weights_only=False for PyTorch 2.6+
        graphs = torch.load(data_path, weights_only=False)
        logger.info(f"Loaded {len(graphs)} graphs")
        
        # Class distribution
        labels = [graph.y.item() for graph in graphs]
        healthy = sum(1 for l in labels if l == 0)
        dementia = sum(1 for l in labels if l == 1)
        logger.info(f"Class distribution: Healthy={healthy}, Dementia={dementia}")
        
        return graphs
        
    except Exception as e:
        logger.error(f"Failed to load graph dataset: {e}")
        raise


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                criterion: nn.Module, device: torch.device) -> float:
    """
    Train the model for one epoch.
    
    Args:
        model: The GAT model
        loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        out = model(data)
        
        # Ensure output is 1D for BCEWithLogitsLoss
        out = out.view(-1)
        
        # Calculate loss
        loss = criterion(out, data.y.float())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
    
    return total_loss / len(loader.dataset)


def train_full_model(graphs: List[Data], device: torch.device) -> InterpretableGAT:
    """
    Train the interpretable GAT model on the full dataset.
    
    Args:
        graphs: List of all graph Data objects
        device: Device to train on
        
    Returns:
        Trained model
    """
    logger.info("=" * 60)
    logger.info("TRAINING FINAL MODEL ON FULL DATASET")
    logger.info("=" * 60)
    logger.info(f"Total samples: {len(graphs)}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Epochs: {EPOCHS}")
    logger.info(f"Learning rate: {LEARNING_RATE}")
    logger.info(f"Weight decay: {WEIGHT_DECAY}")
    logger.info("=" * 60)
    
    # Create data loader
    loader = DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize model
    model = InterpretableGAT(
        in_channels=5,
        hidden_channels=16,
        num_heads=4,
        dropout=0.3
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Initialize optimizer and loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, loader, optimizer, criterion, device)
        
        if epoch % 20 == 0:
            logger.info(f"Epoch {epoch}/{EPOCHS}, Loss: {train_loss:.4f}")
    
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    
    return model


# ============================================================================
# ATTENTION EXTRACTION
# ============================================================================

def extract_attention_weights(
    model: InterpretableGAT,
    graphs: List[Data],
    device: torch.device
) -> Dict[str, List[float]]:
    """
    Extract attention weights from the trained model for all graphs.
    
    Args:
        model: Trained interpretable GAT model
        graphs: List of graph Data objects
        device: Device to run inference on
        
    Returns:
        Dictionary mapping edge names to lists of attention weights
    """
    logger.info("\n" + "=" * 60)
    logger.info("EXTRACTING ATTENTION WEIGHTS")
    logger.info("=" * 60)
    
    model.eval()
    
    # Dictionary to accumulate attention weights for each edge type
    edge_attention = defaultdict(list)
    
    with torch.no_grad():
        for idx, graph in enumerate(graphs):
            # Move graph to device
            graph = graph.to(device)
            
            # Forward pass (stores attention weights in model)
            _ = model(graph)
            
            # Extract stored attention weights
            edge_index = model.last_edge_index
            alpha = model.last_alpha
            
            if edge_index is None or alpha is None:
                logger.warning(f"No attention weights for graph {idx}")
                continue
            
            # Average across attention heads (dim=1)
            # alpha shape: [num_edges, num_heads] -> [num_edges]
            alpha_mean = alpha.mean(dim=1)
            
            # Process each edge
            for i in range(edge_index.size(1)):
                src_node = edge_index[0, i].item()
                dst_node = edge_index[1, i].item()
                attention_weight = alpha_mean[i].item()
                
                # Map node indices to tissue names
                src_name = NODE_NAMES[src_node]
                dst_name = NODE_NAMES[dst_node]
                
                # Create edge name
                edge_name = f"{src_name} → {dst_name}"
                
                # Store attention weight
                edge_attention[edge_name].append(attention_weight)
            
            if (idx + 1) % 50 == 0:
                logger.info(f"Processed {idx + 1}/{len(graphs)} graphs...")
    
    logger.info(f"Extracted attention weights from {len(graphs)} graphs")
    logger.info("=" * 60)
    
    return edge_attention


def calculate_average_attention(edge_attention: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate average attention statistics for each edge type.
    
    Args:
        edge_attention: Dictionary mapping edge names to lists of weights
        
    Returns:
        Dictionary with mean, std, min, max for each edge
    """
    attention_stats = {}
    
    for edge_name, weights in edge_attention.items():
        weights_array = np.array(weights)
        attention_stats[edge_name] = {
            'mean': np.mean(weights_array),
            'std': np.std(weights_array),
            'min': np.min(weights_array),
            'max': np.max(weights_array),
            'count': len(weights_array)
        }
    
    return attention_stats


# ============================================================================
# RESULTS DISPLAY
# ============================================================================

def print_attention_leaderboard(attention_stats: Dict[str, Dict[str, float]]) -> None:
    """
    Print a beautifully formatted leaderboard of attention weights.
    
    Args:
        attention_stats: Dictionary with attention statistics for each edge
    """
    print("\n" + "=" * 80)
    print("ATTENTION WEIGHT LEADERBOARD")
    print("=" * 80)
    print("Which tissue connections does the GAT model pay the most attention to?")
    print("=" * 80)
    
    # Sort edges by mean attention weight (descending)
    sorted_edges = sorted(
        attention_stats.items(),
        key=lambda x: x[1]['mean'],
        reverse=True
    )
    
    print("\nRANKED BY AVERAGE ATTENTION WEIGHT:")
    print("-" * 80)
    print(f"{'Rank':<6} {'Connection':<30} {'Mean':<12} {'Std':<12} {'Range':<20}")
    print("-" * 80)
    
    for rank, (edge_name, stats) in enumerate(sorted_edges, 1):
        mean_val = stats['mean']
        std_val = stats['std']
        min_val = stats['min']
        max_val = stats['max']
        range_str = f"[{min_val:.4f}, {max_val:.4f}]"
        
        print(f"{rank:<6} {edge_name:<30} {mean_val:<12.4f} {std_val:<12.4f} {range_str:<20}")
    
    print("-" * 80)
    
    # Identify most important connections
    print("\nKEY INSIGHTS:")
    print("-" * 80)
    
    top_edge = sorted_edges[0]
    print(f"🏆 MOST IMPORTANT CONNECTION: {top_edge[0]}")
    print(f"   Average attention: {top_edge[1]['mean']:.4f}")
    print(f"   This connection receives {top_edge[1]['mean']/sorted_edges[-1][1]['mean']:.2f}x more attention than the least important")
    
    print(f"\n📊 ATTENTION DISTRIBUTION:")
    total_attention = sum(stats['mean'] for _, stats in sorted_edges)
    for edge_name, stats in sorted_edges:
        percentage = (stats['mean'] / total_attention) * 100
        print(f"   {edge_name:<30} {percentage:>6.2f}%")
    
    print("\n" + "=" * 80)
    print("BIOLOGICAL INTERPRETATION:")
    print("-" * 80)
    print("Higher attention weights indicate that the model considers these tissue")
    print("connections more important for distinguishing between healthy controls")
    print("and Alzheimer's disease patients.")
    print()
    print("The attention mechanism learns to focus on the most discriminative")
    print("relationships between brain tissue types (CSF, Gray Matter, White Matter).")
    print("=" * 80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function for attention interpretability analysis.
    
    Pipeline:
    1. Load graph dataset
    2. Train final model on full dataset
    3. Extract attention weights from all graphs
    4. Calculate average attention statistics
    5. Display attention leaderboard
    """
    logger.info("=" * 60)
    logger.info("OASIS-1 GAT ATTENTION INTERPRETABILITY")
    logger.info("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        # Step 1: Load graph dataset
        graphs = load_graph_dataset(GRAPH_DATA_PATH)
        
        # Step 2: Train final model on full dataset
        model = train_full_model(graphs, device)
        
        # Step 3: Extract attention weights
        edge_attention = extract_attention_weights(model, graphs, device)
        
        # Step 4: Calculate statistics
        attention_stats = calculate_average_attention(edge_attention)
        
        # Step 5: Display leaderboard
        print_attention_leaderboard(attention_stats)
        
        logger.info("\n✓ Attention interpretability analysis complete!")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
