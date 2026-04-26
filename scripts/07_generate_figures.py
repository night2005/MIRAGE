#!/usr/bin/env python3
"""
OASIS-1 Publication-Quality Figure Generation
==============================================
Production-grade script to generate high-DPI figures for research paper:
- Figure 1: Confusion Matrix
- Figure 2: ROC Curve with AUC
- Figure 3: Attention Weight Bar Chart

Phase 7: Visualization & Publication
Author: Data Visualization & PyTorch Engineering Team
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc
)

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

GRAPH_DATA_PATH = Path('oasis_graphs.pt')
RANDOM_STATE = 42
N_SPLITS = 5
BATCH_SIZE = 16
EPOCHS_CV = 100
EPOCHS_FULL = 80
LEARNING_RATE = 0.005
WEIGHT_DECAY = 1e-4
DPI = 300

# Node mapping
NODE_NAMES = {
    0: 'CSF',
    1: 'Gray Matter',
    2: 'White Matter'
}

# Set random seeds
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')


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
# MODEL ARCHITECTURES
# ============================================================================

class ADGraphModel(nn.Module):
    """Standard GAT model for cross-validation."""
    
    def __init__(self, in_channels: int = 5, hidden_channels: int = 16, 
                 num_heads: int = 4, dropout: float = 0.3):
        super(ADGraphModel, self).__init__()
        
        self.conv1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=num_heads,
            dropout=dropout
        )
        
        self.conv2 = GATConv(
            in_channels=hidden_channels * num_heads,
            out_channels=hidden_channels,
            heads=1,
            concat=False,
            dropout=dropout
        )
        
        self.classifier = nn.Linear(hidden_channels, 1)
        self.dropout = dropout
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        
        return x


class InterpretableGAT(nn.Module):
    """Interpretable GAT model with attention weight extraction."""
    
    def __init__(self, in_channels: int = 5, hidden_channels: int = 16, 
                 num_heads: int = 4, dropout: float = 0.3):
        super(InterpretableGAT, self).__init__()
        
        self.conv1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=num_heads,
            dropout=dropout
        )
        
        self.conv2 = GATConv(
            in_channels=hidden_channels * num_heads,
            out_channels=hidden_channels,
            heads=1,
            concat=False,
            dropout=dropout
        )
        
        self.classifier = nn.Linear(hidden_channels, 1)
        self.dropout = dropout
        
        self.last_edge_index = None
        self.last_alpha = None
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x, (edge_index_out, alpha) = self.conv1(x, edge_index, return_attention_weights=True)
        self.last_edge_index = edge_index_out
        self.last_alpha = alpha
        
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        
        return x


# ============================================================================
# DATA LOADING
# ============================================================================

def load_graph_dataset(data_path: Path) -> List[Data]:
    """Load the graph dataset from disk."""
    logger.info(f"Loading graph dataset from: {data_path}")
    graphs = torch.load(data_path, weights_only=False)
    logger.info(f"Loaded {len(graphs)} graphs")
    return graphs


def extract_labels(graphs: List[Data]) -> np.ndarray:
    """Extract labels from graph dataset."""
    return np.array([graph.y.item() for graph in graphs])


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                criterion: nn.Module, device: torch.device) -> float:
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        out = model(data)
        out = out.view(-1)
        loss = criterion(out, data.y.float())
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
    
    return total_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate the model and return predictions and true labels."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            out = out.view(-1)
            probs = torch.sigmoid(out)
            
            all_preds.extend(probs.cpu().numpy().tolist())
            all_labels.extend(data.y.cpu().numpy().tolist())
    
    return np.array(all_preds), np.array(all_labels)


def cross_validation_predictions(graphs: List[Data], labels: np.ndarray, 
                                  device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform cross-validation and collect all predictions.
    
    Returns:
        Tuple of (y_true, y_prob, y_pred)
    """
    logger.info("=" * 60)
    logger.info("RUNNING CROSS-VALIDATION FOR FIGURES 1 & 2")
    logger.info("=" * 60)
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    all_y_true = []
    all_y_prob = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), 1):
        logger.info(f"Processing Fold {fold}/{N_SPLITS}...")
        
        train_graphs = [graphs[i] for i in train_idx]
        test_graphs = [graphs[i] for i in test_idx]
        
        train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)
        
        model = ADGraphModel().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = nn.BCEWithLogitsLoss()
        
        # Train with early stopping
        best_loss = float('inf')
        patience_counter = 0
        patience = 15
        
        for epoch in range(1, EPOCHS_CV + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            
            if train_loss < best_loss:
                best_loss = train_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
        
        # Evaluate
        y_prob, y_true = evaluate(model, test_loader, device)
        all_y_true.extend(y_true)
        all_y_prob.extend(y_prob)
    
    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)
    all_y_pred = (all_y_prob >= 0.5).astype(int)
    
    logger.info(f"Collected {len(all_y_true)} predictions across all folds")
    logger.info("=" * 60)
    
    return all_y_true, all_y_prob, all_y_pred


# ============================================================================
# ATTENTION EXTRACTION
# ============================================================================

def train_full_model(graphs: List[Data], device: torch.device) -> InterpretableGAT:
    """Train interpretable GAT on full dataset."""
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING FULL MODEL FOR FIGURE 3")
    logger.info("=" * 60)
    
    loader = DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=True)
    model = InterpretableGAT().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(1, EPOCHS_FULL + 1):
        train_loss = train_epoch(model, loader, optimizer, criterion, device)
        if epoch % 20 == 0:
            logger.info(f"Epoch {epoch}/{EPOCHS_FULL}, Loss: {train_loss:.4f}")
    
    logger.info("Training complete")
    logger.info("=" * 60)
    
    return model


def extract_attention_weights(model: InterpretableGAT, graphs: List[Data], 
                               device: torch.device) -> Dict[str, float]:
    """Extract and average attention weights across all graphs."""
    logger.info("Extracting attention weights...")
    
    model.eval()
    edge_attention = defaultdict(list)
    
    with torch.no_grad():
        for graph in graphs:
            graph = graph.to(device)
            _ = model(graph)
            
            edge_index = model.last_edge_index
            alpha = model.last_alpha
            
            if edge_index is None or alpha is None:
                continue
            
            alpha_mean = alpha.mean(dim=1)
            
            for i in range(edge_index.size(1)):
                src_node = edge_index[0, i].item()
                dst_node = edge_index[1, i].item()
                attention_weight = alpha_mean[i].item()
                
                src_name = NODE_NAMES[src_node]
                dst_name = NODE_NAMES[dst_node]
                edge_name = f"{src_name} → {dst_name}"
                
                edge_attention[edge_name].append(attention_weight)
    
    # Calculate mean attention for each edge
    mean_attention = {edge: np.mean(weights) for edge, weights in edge_attention.items()}
    
    logger.info(f"Extracted attention for {len(mean_attention)} edge types")
    
    return mean_attention


# ============================================================================
# FIGURE GENERATION
# ============================================================================

def generate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Generate and save confusion matrix (Figure 1)."""
    logger.info("\nGenerating Figure 1: Confusion Matrix...")
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Healthy Control', 'Dementia'],
        yticklabels=['Healthy Control', 'Dementia'],
        cbar_kws={'label': 'Count'},
        square=True,
        linewidths=1,
        linecolor='gray'
    )
    
    plt.title('Confusion Matrix - GAT Model Performance\n5-Fold Cross-Validation', 
              fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('fig1_confusion_matrix.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    
    logger.info("✓ Saved: fig1_confusion_matrix.png")


def generate_roc_curve(y_true: np.ndarray, y_prob: np.ndarray) -> None:
    """Generate and save ROC curve (Figure 2)."""
    logger.info("Generating Figure 2: ROC Curve...")
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Create figure
    plt.figure(figsize=(8, 7))
    
    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkblue', lw=2.5, 
             label=f'GAT Model (AUC = {roc_auc:.4f})')
    
    # Plot diagonal baseline
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', 
             label='Random Classifier (AUC = 0.5000)')
    
    # Formatting
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curve - Alzheimer\'s Disease Classification\n5-Fold Cross-Validation', 
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='lower right', fontsize=11, frameon=True, shadow=True)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fig2_roc_curve.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Saved: fig2_roc_curve.png (AUC = {roc_auc:.4f})")


def generate_attention_bar_chart(mean_attention: Dict[str, float]) -> None:
    """Generate and save attention weight bar chart (Figure 3)."""
    logger.info("Generating Figure 3: Attention Weight Bar Chart...")
    
    # Sort by attention weight
    sorted_edges = sorted(mean_attention.items(), key=lambda x: x[1], reverse=True)
    edge_names = [edge for edge, _ in sorted_edges]
    attention_values = [attn for _, attn in sorted_edges]
    
    # Create colors (top 3 in dark red, rest in steel blue)
    colors = ['#8B0000' if i < 3 else '#4682B4' for i in range(len(edge_names))]
    
    # Create figure
    plt.figure(figsize=(10, 7))
    
    # Create horizontal bar chart
    bars = plt.barh(edge_names, attention_values, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, attention_values)):
        plt.text(value + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{value:.4f}', 
                va='center', fontsize=10, fontweight='bold')
    
    # Formatting
    plt.xlabel('Mean Attention Weight', fontsize=12, fontweight='bold')
    plt.ylabel('Tissue Connection', fontsize=12, fontweight='bold')
    plt.title('GAT Attention Weights by Tissue Connection\nTrained on Full Dataset (N=161)', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#8B0000', edgecolor='black', label='Top 3 Connections'),
        Patch(facecolor='#4682B4', edgecolor='black', label='Other Connections')
    ]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=10, frameon=True, shadow=True)
    
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig3_attention_weights.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    
    logger.info("✓ Saved: fig3_attention_weights.png")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function for figure generation.
    
    Pipeline:
    1. Load graph dataset
    2. Run cross-validation and collect predictions
    3. Generate confusion matrix (Figure 1)
    4. Generate ROC curve (Figure 2)
    5. Train full model and extract attention weights
    6. Generate attention bar chart (Figure 3)
    """
    logger.info("=" * 60)
    logger.info("PUBLICATION-QUALITY FIGURE GENERATION")
    logger.info("=" * 60)
    logger.info(f"Output DPI: {DPI}")
    logger.info("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        # Step 1: Load data
        graphs = load_graph_dataset(GRAPH_DATA_PATH)
        labels = extract_labels(graphs)
        
        # Step 2: Cross-validation for Figures 1 & 2
        y_true, y_prob, y_pred = cross_validation_predictions(graphs, labels, device)
        
        # Step 3: Generate Figure 1 - Confusion Matrix
        generate_confusion_matrix(y_true, y_pred)
        
        # Step 4: Generate Figure 2 - ROC Curve
        generate_roc_curve(y_true, y_prob)
        
        # Step 5: Train full model for Figure 3
        model = train_full_model(graphs, device)
        
        # Step 6: Extract attention weights
        mean_attention = extract_attention_weights(model, graphs, device)
        
        # Step 7: Generate Figure 3 - Attention Bar Chart
        generate_attention_bar_chart(mean_attention)
        
        # Summary
        print("\n" + "=" * 60)
        print("FIGURE GENERATION COMPLETE")
        print("=" * 60)
        print("Generated files:")
        print("  ✓ fig1_confusion_matrix.png")
        print("  ✓ fig2_roc_curve.png")
        print("  ✓ fig3_attention_weights.png")
        print()
        print(f"All figures saved at {DPI} DPI for publication quality.")
        print("=" * 60)
        
        logger.info("\n✓ All figures generated successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
