#!/usr/bin/env python3
"""
OASIS-1 Graph Attention Network Training
=========================================
Production-grade script to train and evaluate a GAT model for Alzheimer's
classification using PyTorch Geometric.

Phase 5: GNN Training & Evaluation
Author: PyTorch Geometric Deep Learning Team
Date: 2026-04-11
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, recall_score,
    confusion_matrix
)

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

GRAPH_DATA_PATH = Path('oasis_graphs.pt')
RANDOM_STATE = 42
N_SPLITS = 5
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.005
WEIGHT_DECAY = 1e-4

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
# MODEL ARCHITECTURE
# ============================================================================

class ADGraphModel(nn.Module):
    """
    Graph Attention Network for Alzheimer's Disease classification.
    
    Architecture:
    - GAT Layer 1: 5 -> 16 features (4 attention heads)
    - ELU activation
    - GAT Layer 2: 64 -> 16 features (1 attention head)
    - Global mean pooling
    - Linear classifier: 16 -> 1 (binary classification)
    """
    
    def __init__(self, in_channels: int = 5, hidden_channels: int = 16, 
                 num_heads: int = 4, dropout: float = 0.3):
        """
        Initialize the GAT model.
        
        Args:
            in_channels: Number of input node features
            hidden_channels: Number of hidden features
            num_heads: Number of attention heads in first layer
            dropout: Dropout probability
        """
        super(ADGraphModel, self).__init__()
        
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
        
    def forward(self, data):
        """
        Forward pass through the network.
        
        Args:
            data: PyTorch Geometric Data object with x, edge_index, batch
            
        Returns:
            Logits for binary classification
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # First GAT layer + activation
        x = self.conv1(x, edge_index)
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
        
        # Verify structure
        first_graph = graphs[0]
        logger.info(f"Graph structure: {first_graph}")
        logger.info(f"  Nodes: {first_graph.num_nodes}")
        logger.info(f"  Edges: {first_graph.num_edges}")
        logger.info(f"  Features: {first_graph.num_node_features}")
        
        return graphs
        
    except Exception as e:
        logger.error(f"Failed to load graph dataset: {e}")
        raise


def extract_labels(graphs: List[Data]) -> np.ndarray:
    """
    Extract labels from graph dataset for stratification.
    
    Args:
        graphs: List of PyTorch Geometric Data objects
        
    Returns:
        NumPy array of labels
    """
    labels = np.array([graph.y.item() for graph in graphs])
    logger.info(f"Label distribution: {np.bincount(labels)}")
    return labels


# ============================================================================
# TRAINING & EVALUATION
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


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the model and return predictions and true labels.
    
    Args:
        model: The GAT model
        loader: Test data loader
        device: Device to evaluate on
        
    Returns:
        Tuple of (predictions, true_labels)
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            
            # Forward pass
            out = model(data)
            
            # Ensure output is 1D
            out = out.view(-1)
            
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(out)
            
            # Convert to lists
            all_preds.extend(probs.cpu().numpy().tolist())
            all_labels.extend(data.y.cpu().numpy().tolist())
    
    return np.array(all_preds), np.array(all_labels)


def calculate_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred_probs: Predicted probabilities
        
    Returns:
        Dictionary with accuracy, ROC-AUC, sensitivity, specificity
    """
    # Convert probabilities to binary predictions (threshold = 0.5)
    y_pred = (y_pred_probs >= 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_probs)
    sensitivity = recall_score(y_true, y_pred)
    
    # Calculate specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    
    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'sensitivity': sensitivity,
        'specificity': specificity
    }


# ============================================================================
# CROSS-VALIDATION
# ============================================================================

def train_with_cross_validation(
    graphs: List[Data],
    labels: np.ndarray,
    device: torch.device
) -> List[Dict[str, float]]:
    """
    Train and evaluate the model using stratified k-fold cross-validation.
    
    Args:
        graphs: List of graph Data objects
        labels: Array of labels for stratification
        device: Device to train on
        
    Returns:
        List of metric dictionaries for each fold
    """
    logger.info("=" * 60)
    logger.info("STARTING CROSS-VALIDATION")
    logger.info("=" * 60)
    logger.info(f"Number of folds: {N_SPLITS}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Epochs per fold: {EPOCHS}")
    logger.info(f"Learning rate: {LEARNING_RATE}")
    logger.info(f"Weight decay: {WEIGHT_DECAY}")
    logger.info("=" * 60)
    
    # Initialize stratified k-fold
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"FOLD {fold}/{N_SPLITS}")
        logger.info(f"{'='*60}")
        
        # Split data
        train_graphs = [graphs[i] for i in train_idx]
        test_graphs = [graphs[i] for i in test_idx]
        
        logger.info(f"Training samples: {len(train_graphs)}")
        logger.info(f"Test samples: {len(test_graphs)}")
        
        # Create data loaders
        train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)
        
        # Initialize model (fresh for each fold)
        model = ADGraphModel(
            in_channels=5,
            hidden_channels=16,
            num_heads=4,
            dropout=0.3
        ).to(device)
        
        # Initialize optimizer and loss
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        criterion = nn.BCEWithLogitsLoss()
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        patience = 15
        
        for epoch in range(1, EPOCHS + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            
            # Early stopping check
            if train_loss < best_loss:
                best_loss = train_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 20 == 0:
                logger.info(f"  Epoch {epoch}/{EPOCHS}, Loss: {train_loss:.4f}")
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"  Early stopping at epoch {epoch}")
                break
        
        # Evaluate on test set
        y_pred_probs, y_true = evaluate(model, test_loader, device)
        metrics = calculate_metrics(y_true, y_pred_probs)
        
        logger.info(f"\nFold {fold} Results:")
        logger.info(f"  Accuracy:    {metrics['accuracy']:.4f}")
        logger.info(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")
        logger.info(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        logger.info(f"  Specificity: {metrics['specificity']:.4f}")
        
        fold_results.append(metrics)
    
    return fold_results


# ============================================================================
# RESULTS DISPLAY
# ============================================================================

def print_final_results(fold_results: List[Dict[str, float]]) -> None:
    """
    Print comprehensive final results across all folds.
    
    Args:
        fold_results: List of metric dictionaries from each fold
    """
    print("\n" + "=" * 80)
    print("GRAPH ATTENTION NETWORK - FINAL RESULTS")
    print("=" * 80)
    print(f"Cross-validation: {N_SPLITS}-fold stratified")
    print(f"Model: GAT (2 layers, 4 attention heads)")
    print(f"Training: {EPOCHS} epochs max, early stopping enabled")
    print("=" * 80)
    
    # Calculate mean and std for each metric
    metrics_summary = {}
    for metric in ['accuracy', 'roc_auc', 'sensitivity', 'specificity']:
        values = [fold[metric] for fold in fold_results]
        metrics_summary[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values
        }
    
    print("\nPER-FOLD RESULTS:")
    print("-" * 80)
    for i, fold_result in enumerate(fold_results, 1):
        print(f"\nFold {i}:")
        print(f"  Accuracy:    {fold_result['accuracy']:.4f}")
        print(f"  ROC-AUC:     {fold_result['roc_auc']:.4f}")
        print(f"  Sensitivity: {fold_result['sensitivity']:.4f}")
        print(f"  Specificity: {fold_result['specificity']:.4f}")
    
    print("\n" + "-" * 80)
    print("AVERAGE RESULTS (Mean ± Std):")
    print("-" * 80)
    print(f"Accuracy:    {metrics_summary['accuracy']['mean']:.4f} ± {metrics_summary['accuracy']['std']:.4f}")
    print(f"ROC-AUC:     {metrics_summary['roc_auc']['mean']:.4f} ± {metrics_summary['roc_auc']['std']:.4f}")
    print(f"Sensitivity: {metrics_summary['sensitivity']['mean']:.4f} ± {metrics_summary['sensitivity']['std']:.4f}")
    print(f"Specificity: {metrics_summary['specificity']['mean']:.4f} ± {metrics_summary['specificity']['std']:.4f}")
    
    print("\n" + "=" * 80)
    print("COMPARISON WITH BASELINE MODELS")
    print("=" * 80)
    print("\nModel: Graph Attention Network")
    print(f"  Accuracy:    {metrics_summary['accuracy']['mean']:.4f} ± {metrics_summary['accuracy']['std']:.4f}")
    print(f"  ROC-AUC:     {metrics_summary['roc_auc']['mean']:.4f} ± {metrics_summary['roc_auc']['std']:.4f}")
    print(f"  Sensitivity: {metrics_summary['sensitivity']['mean']:.4f} ± {metrics_summary['sensitivity']['std']:.4f}")
    print(f"  Specificity: {metrics_summary['specificity']['mean']:.4f} ± {metrics_summary['specificity']['std']:.4f}")
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS:")
    print("-" * 80)
    print(f"• Best metric: {max(metrics_summary.items(), key=lambda x: x[1]['mean'])[0].upper()}")
    print(f"  Value: {max(metrics_summary.values(), key=lambda x: x['mean'])['mean']:.4f}")
    print(f"• Most stable metric: {min(metrics_summary.items(), key=lambda x: x[1]['std'])[0].upper()}")
    print(f"  Std: {min(metrics_summary.values(), key=lambda x: x['std'])['std']:.4f}")
    print("=" * 80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function for GNN training and evaluation.
    
    Pipeline:
    1. Load graph dataset
    2. Extract labels for stratification
    3. Train with cross-validation
    4. Display final results
    """
    logger.info("=" * 60)
    logger.info("OASIS-1 GRAPH ATTENTION NETWORK TRAINING")
    logger.info("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        # Step 1: Load graph dataset
        graphs = load_graph_dataset(GRAPH_DATA_PATH)
        
        # Step 2: Extract labels
        labels = extract_labels(graphs)
        
        # Step 3: Train with cross-validation
        fold_results = train_with_cross_validation(graphs, labels, device)
        
        # Step 4: Display final results
        print_final_results(fold_results)
        
        logger.info("\n✓ GNN training and evaluation complete!")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
