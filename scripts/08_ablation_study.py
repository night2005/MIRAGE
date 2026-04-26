#!/usr/bin/env python3
"""
OASIS-1 Ablation Study: Topology & Volume Only
===============================================
Production-grade script to evaluate GAT performance when using ONLY volumetric
features and graph topology, removing all tabular demographic features.

Phase 8: Ablation Study
Author: PyTorch ML Research Team
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
# MODEL ARCHITECTURE (ABLATED)
# ============================================================================

class ADGraphModelAblated(nn.Module):
    """
    Ablated Graph Attention Network for Alzheimer's Disease classification.
    
    ABLATION: Uses only 1 input feature (tissue volume) instead of 5.
    Removes all demographic features (Age, Educ, eTIV, nWBV).
    
    Architecture:
    - GAT Layer 1: 1 -> 16 features (4 attention heads)
    - ELU activation
    - GAT Layer 2: 64 -> 16 features (1 attention head)
    - Global mean pooling
    - Linear classifier: 16 -> 1 (binary classification)
    """
    
    def __init__(self, in_channels: int = 1, hidden_channels: int = 16, 
                 num_heads: int = 4, dropout: float = 0.3):
        """
        Initialize the ablated GAT model.
        
        Args:
            in_channels: Number of input node features (1 for ablation)
            hidden_channels: Number of hidden features
            num_heads: Number of attention heads in first layer
            dropout: Dropout probability
        """
        super(ADGraphModelAblated, self).__init__()
        
        # First GAT layer with multiple attention heads
        # CRITICAL: in_channels=1 for ablation study
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
# DATA LOADING & ABLATION
# ============================================================================

def load_and_ablate_graphs(data_path: Path) -> List[Data]:
    """
    Load graph dataset and perform feature ablation.
    
    ABLATION: Keep only the first feature (tissue volume), remove all others.
    Original features: [Volume, Age, Educ, eTIV, nWBV]
    Ablated features: [Volume]
    
    Args:
        data_path: Path to the saved graph dataset
        
    Returns:
        List of ablated PyTorch Geometric Data objects
    """
    logger.info(f"Loading graph dataset from: {data_path}")
    
    try:
        # Load with weights_only=False for PyTorch 2.6+
        graphs = torch.load(data_path, weights_only=False)
        logger.info(f"Loaded {len(graphs)} graphs")
        
        # Verify original structure
        first_graph = graphs[0]
        logger.info(f"Original graph structure: {first_graph}")
        logger.info(f"  Original num_node_features: {first_graph.num_node_features}")
        
        # Perform ablation: keep only first feature (Volume)
        logger.info("\n" + "=" * 60)
        logger.info("PERFORMING FEATURE ABLATION")
        logger.info("=" * 60)
        logger.info("Removing features: Age, Educ, eTIV, nWBV")
        logger.info("Keeping only: Tissue Volume (feature index 0)")
        logger.info("=" * 60)
        
        ablated_graphs = []
        for graph in graphs:
            # Create a copy to avoid modifying original
            ablated_graph = Data(
                x=graph.x[:, 0:1],  # Keep only first column (Volume)
                edge_index=graph.edge_index,
                y=graph.y
            )
            ablated_graphs.append(ablated_graph)
        
        # Verify ablation
        first_ablated = ablated_graphs[0]
        logger.info(f"\nAblated graph structure: {first_ablated}")
        logger.info(f"  Ablated num_node_features: {first_ablated.num_node_features}")
        
        if first_ablated.num_node_features != 1:
            raise ValueError(f"Ablation failed! Expected 1 feature, got {first_ablated.num_node_features}")
        
        logger.info("✓ Ablation successful: num_node_features = 1")
        logger.info("=" * 60)
        
        return ablated_graphs
        
    except Exception as e:
        logger.error(f"Failed to load and ablate graph dataset: {e}")
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
    Train and evaluate the ablated model using stratified k-fold cross-validation.
    
    Args:
        graphs: List of ablated graph Data objects
        labels: Array of labels for stratification
        device: Device to train on
        
    Returns:
        List of metric dictionaries for each fold
    """
    logger.info("\n" + "=" * 60)
    logger.info("STARTING CROSS-VALIDATION (ABLATED MODEL)")
    logger.info("=" * 60)
    logger.info(f"Number of folds: {N_SPLITS}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Epochs per fold: {EPOCHS}")
    logger.info(f"Learning rate: {LEARNING_RATE}")
    logger.info(f"Weight decay: {WEIGHT_DECAY}")
    logger.info(f"Input features: 1 (Volume only)")
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
        # CRITICAL: in_channels=1 for ablated model
        model = ADGraphModelAblated(
            in_channels=1,
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

def print_ablation_results(fold_results: List[Dict[str, float]]) -> None:
    """
    Print comprehensive ablation study results.
    
    Args:
        fold_results: List of metric dictionaries from each fold
    """
    print("\n" + "=" * 80)
    print("ABLATION STUDY: TOPOLOGY & VOLUME ONLY")
    print("=" * 80)
    print("Features Used: Tissue Volume ONLY (1 feature)")
    print("Features Removed: Age, Education, eTIV, nWBV (4 features)")
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
    print("ABLATED MODEL - AVERAGE RESULTS (Mean ± Std):")
    print("-" * 80)
    print(f"Accuracy:    {metrics_summary['accuracy']['mean']:.4f} ± {metrics_summary['accuracy']['std']:.4f}")
    print(f"ROC-AUC:     {metrics_summary['roc_auc']['mean']:.4f} ± {metrics_summary['roc_auc']['std']:.4f}")
    print(f"Sensitivity: {metrics_summary['sensitivity']['mean']:.4f} ± {metrics_summary['sensitivity']['std']:.4f}")
    print(f"Specificity: {metrics_summary['specificity']['mean']:.4f} ± {metrics_summary['specificity']['std']:.4f}")
    
    print("\n" + "=" * 80)
    print("COMPARISON WITH FULL MODEL")
    print("=" * 80)
    print("\nFull Model (5 features: Volume + Age + Educ + eTIV + nWBV):")
    print("  Accuracy:    0.7451 ± 0.0256")
    print("  ROC-AUC:     0.8360 ± 0.0260")
    print("  Sensitivity: 0.7385 ± 0.0917")
    print("  Specificity: 0.7538 ± 0.1064")
    
    print("\nAblated Model (1 feature: Volume only):")
    print(f"  Accuracy:    {metrics_summary['accuracy']['mean']:.4f} ± {metrics_summary['accuracy']['std']:.4f}")
    print(f"  ROC-AUC:     {metrics_summary['roc_auc']['mean']:.4f} ± {metrics_summary['roc_auc']['std']:.4f}")
    print(f"  Sensitivity: {metrics_summary['sensitivity']['mean']:.4f} ± {metrics_summary['sensitivity']['std']:.4f}")
    print(f"  Specificity: {metrics_summary['specificity']['mean']:.4f} ± {metrics_summary['specificity']['std']:.4f}")
    
    print("\nPerformance Change:")
    print(f"  Accuracy:    {(metrics_summary['accuracy']['mean'] - 0.7451):.4f} ({(metrics_summary['accuracy']['mean'] - 0.7451)/0.7451*100:+.2f}%)")
    print(f"  ROC-AUC:     {(metrics_summary['roc_auc']['mean'] - 0.8360):.4f} ({(metrics_summary['roc_auc']['mean'] - 0.8360)/0.8360*100:+.2f}%)")
    print(f"  Sensitivity: {(metrics_summary['sensitivity']['mean'] - 0.7385):.4f} ({(metrics_summary['sensitivity']['mean'] - 0.7385)/0.7385*100:+.2f}%)")
    print(f"  Specificity: {(metrics_summary['specificity']['mean'] - 0.7538):.4f} ({(metrics_summary['specificity']['mean'] - 0.7538)/0.7538*100:+.2f}%)")
    
    print("\n" + "=" * 80)
    print("KEY FINDINGS:")
    print("-" * 80)
    
    roc_change = (metrics_summary['roc_auc']['mean'] - 0.8360) / 0.8360 * 100
    if roc_change < -5:
        print("• Removing demographic features SIGNIFICANTLY DEGRADES performance")
        print("• Tabular features (Age, Education, eTIV, nWBV) are CRITICAL for classification")
    elif roc_change < 0:
        print("• Removing demographic features slightly reduces performance")
        print("• Tabular features provide valuable complementary information")
    else:
        print("• Model maintains performance with volume and topology alone")
        print("• Graph structure captures sufficient discriminative information")
    
    print(f"• ROC-AUC change: {roc_change:+.2f}%")
    print("• This ablation study demonstrates the contribution of demographic features")
    print("=" * 80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function for ablation study.
    
    Pipeline:
    1. Load and ablate graph dataset (keep only volume feature)
    2. Extract labels for stratification
    3. Train with cross-validation
    4. Display ablation results with comparison
    """
    logger.info("=" * 60)
    logger.info("OASIS-1 ABLATION STUDY: TOPOLOGY & VOLUME ONLY")
    logger.info("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        # Step 1: Load and ablate graph dataset
        graphs = load_and_ablate_graphs(GRAPH_DATA_PATH)
        
        # Step 2: Extract labels
        labels = extract_labels(graphs)
        
        # Step 3: Train with cross-validation
        fold_results = train_with_cross_validation(graphs, labels, device)
        
        # Step 4: Display ablation results
        print_ablation_results(fold_results)
        
        logger.info("\n✓ Ablation study complete!")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
