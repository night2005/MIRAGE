#!/usr/bin/env python3
"""
OASIS-1 Tabular Baseline Classifiers
=====================================
Production-grade script to evaluate baseline ML models for Alzheimer's classification
using demographic and structural brain features (excluding cognitive tests).

Phase 2: Tabular Baselines
Author: Medical Data Science Team
Date: 2026-04-11
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, roc_auc_score, recall_score, 
    confusion_matrix, make_scorer
)

warnings.filterwarnings('ignore', category=FutureWarning)


# ============================================================================
# CONFIGURATION
# ============================================================================

MASTER_INDEX_PATH = Path('master_index.csv')
RANDOM_STATE = 42
N_SPLITS = 5

# Feature columns (excluding MMSE cognitive test)
FEATURE_COLS = ['Age', 'Educ', 'SES', 'eTIV', 'nWBV', 'ASF']


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
# DATA LOADING & PREPARATION
# ============================================================================

def load_and_prepare_data(csv_path: Path) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load master index and prepare features and target for classification.
    
    Creates binary target: AD_label (0 if CDR == 0.0, else 1)
    Uses only demographic and structural features, excluding cognitive tests.
    
    Args:
        csv_path: Path to master_index.csv
        
    Returns:
        Tuple of (full_df, X, y) where X is feature matrix and y is binary target
    """
    logger.info(f"Loading data from: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} subjects")
        
        # Create binary target: 0 = healthy control (CDR=0), 1 = dementia (CDR>0)
        df['AD_label'] = (df['CDR'] > 0.0).astype(int)
        
        logger.info(f"Class distribution:")
        logger.info(f"  Healthy controls (CDR=0): {(df['AD_label']==0).sum()}")
        logger.info(f"  Dementia (CDR>0): {(df['AD_label']==1).sum()}")
        
        # Extract features and target
        X = df[FEATURE_COLS].values
        y = df['AD_label'].values
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Features used: {FEATURE_COLS}")
        
        # Check for missing values
        missing_counts = df[FEATURE_COLS].isnull().sum()
        if missing_counts.sum() > 0:
            logger.info("Missing values detected:")
            for col, count in missing_counts[missing_counts > 0].items():
                logger.info(f"  {col}: {count} missing")
        
        return df, X, y
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

def create_pipeline(classifier) -> Pipeline:
    """
    Create sklearn pipeline with imputation, scaling, and classification.
    
    Pipeline steps:
    1. SimpleImputer (median strategy) - handles missing SES values
    2. StandardScaler - normalizes features with different scales
    3. Classifier - the ML model
    
    Args:
        classifier: Sklearn classifier instance
        
    Returns:
        Complete sklearn Pipeline
    """
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', classifier)
    ])
    return pipeline


def get_models() -> Dict[str, Pipeline]:
    """
    Define all baseline models to evaluate.
    
    Returns:
        Dictionary mapping model names to pipeline instances
    """
    models = {
        'Logistic Regression': create_pipeline(
            LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
        ),
        'SVM (RBF)': create_pipeline(
            SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)
        ),
        'Random Forest': create_pipeline(
            RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        )
    }
    return models


# ============================================================================
# CUSTOM METRICS
# ============================================================================

def specificity_score(y_true, y_pred):
    """
    Calculate specificity (True Negative Rate).
    
    Specificity = TN / (TN + FP)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Specificity score
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_model(
    model: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold
) -> Dict[str, float]:
    """
    Evaluate a model using stratified k-fold cross-validation.
    
    Computes multiple metrics across all folds:
    - Accuracy
    - ROC-AUC
    - Sensitivity (Recall)
    - Specificity
    
    Args:
        model: Sklearn pipeline to evaluate
        X: Feature matrix
        y: Target labels
        cv: Cross-validation splitter
        
    Returns:
        Dictionary with mean and std for each metric
    """
    # Define scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'roc_auc': 'roc_auc',
        'sensitivity': 'recall',  # Recall is sensitivity (TPR)
        'specificity': make_scorer(specificity_score)
    }
    
    # Perform cross-validation
    cv_results = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1
    )
    
    # Calculate mean and std for each metric
    results = {}
    for metric in ['accuracy', 'roc_auc', 'sensitivity', 'specificity']:
        scores = cv_results[f'test_{metric}']
        results[f'{metric}_mean'] = np.mean(scores)
        results[f'{metric}_std'] = np.std(scores)
    
    return results


def evaluate_all_models(
    models: Dict[str, Pipeline],
    X: np.ndarray,
    y: np.ndarray
) -> pd.DataFrame:
    """
    Evaluate all models and compile results into a DataFrame.
    
    Args:
        models: Dictionary of model name to pipeline
        X: Feature matrix
        y: Target labels
        
    Returns:
        DataFrame with evaluation results for all models
    """
    logger.info("=" * 60)
    logger.info("EVALUATING BASELINE MODELS")
    logger.info("=" * 60)
    logger.info(f"Cross-validation: {N_SPLITS}-fold stratified")
    logger.info(f"Random state: {RANDOM_STATE}")
    logger.info("=" * 60)
    
    # Setup stratified k-fold
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    results_list = []
    
    for model_name, model in models.items():
        logger.info(f"\nEvaluating: {model_name}")
        
        try:
            results = evaluate_model(model, X, y, cv)
            results['model'] = model_name
            results_list.append(results)
            
            logger.info(f"  Accuracy: {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")
            logger.info(f"  ROC-AUC:  {results['roc_auc_mean']:.4f} ± {results['roc_auc_std']:.4f}")
            logger.info(f"  Sensitivity: {results['sensitivity_mean']:.4f} ± {results['sensitivity_std']:.4f}")
            logger.info(f"  Specificity: {results['specificity_mean']:.4f} ± {results['specificity_std']:.4f}")
            
        except Exception as e:
            logger.error(f"  Failed to evaluate {model_name}: {e}")
            continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results_list)
    
    # Reorder columns for better readability
    col_order = [
        'model',
        'accuracy_mean', 'accuracy_std',
        'roc_auc_mean', 'roc_auc_std',
        'sensitivity_mean', 'sensitivity_std',
        'specificity_mean', 'specificity_std'
    ]
    results_df = results_df[col_order]
    
    return results_df


# ============================================================================
# RESULTS DISPLAY
# ============================================================================

def print_results_table(results_df: pd.DataFrame) -> None:
    """
    Print formatted results table to console.
    
    Args:
        results_df: DataFrame with evaluation results
    """
    print("\n" + "=" * 80)
    print("BASELINE MODEL COMPARISON")
    print("=" * 80)
    print(f"Dataset: {len(results_df)} models evaluated")
    print(f"Cross-validation: {N_SPLITS}-fold stratified")
    print(f"Features: {', '.join(FEATURE_COLS)}")
    print("=" * 80)
    
    print("\nDETAILED RESULTS:")
    print("-" * 80)
    
    for idx, row in results_df.iterrows():
        print(f"\n{row['model']}")
        print(f"  Accuracy:    {row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f}")
        print(f"  ROC-AUC:     {row['roc_auc_mean']:.4f} ± {row['roc_auc_std']:.4f}")
        print(f"  Sensitivity: {row['sensitivity_mean']:.4f} ± {row['sensitivity_std']:.4f}")
        print(f"  Specificity: {row['specificity_mean']:.4f} ± {row['specificity_std']:.4f}")
    
    print("\n" + "-" * 80)
    print("SUMMARY TABLE (Mean ± Std):")
    print("-" * 80)
    
    # Create formatted summary table
    summary_data = []
    for idx, row in results_df.iterrows():
        summary_data.append({
            'Model': row['model'],
            'Accuracy': f"{row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f}",
            'ROC-AUC': f"{row['roc_auc_mean']:.4f} ± {row['roc_auc_std']:.4f}",
            'Sensitivity': f"{row['sensitivity_mean']:.4f} ± {row['sensitivity_std']:.4f}",
            'Specificity': f"{row['specificity_mean']:.4f} ± {row['specificity_std']:.4f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    print("\n" + "=" * 80)
    
    # Identify best model for each metric
    print("\nBEST MODELS BY METRIC:")
    print("-" * 80)
    
    best_acc_idx = results_df['accuracy_mean'].idxmax()
    best_auc_idx = results_df['roc_auc_mean'].idxmax()
    best_sens_idx = results_df['sensitivity_mean'].idxmax()
    best_spec_idx = results_df['specificity_mean'].idxmax()
    
    print(f"Best Accuracy:    {results_df.loc[best_acc_idx, 'model']} "
          f"({results_df.loc[best_acc_idx, 'accuracy_mean']:.4f})")
    print(f"Best ROC-AUC:     {results_df.loc[best_auc_idx, 'model']} "
          f"({results_df.loc[best_auc_idx, 'roc_auc_mean']:.4f})")
    print(f"Best Sensitivity: {results_df.loc[best_sens_idx, 'model']} "
          f"({results_df.loc[best_sens_idx, 'sensitivity_mean']:.4f})")
    print(f"Best Specificity: {results_df.loc[best_spec_idx, 'model']} "
          f"({results_df.loc[best_spec_idx, 'specificity_mean']:.4f})")
    
    print("=" * 80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function for tabular baseline evaluation.
    
    Pipeline:
    1. Load and prepare data
    2. Define models
    3. Evaluate all models with stratified k-fold CV
    4. Display results
    """
    logger.info("=" * 60)
    logger.info("OASIS-1 TABULAR BASELINE CLASSIFIERS")
    logger.info("=" * 60)
    
    try:
        # Step 1: Load and prepare data
        df, X, y = load_and_prepare_data(MASTER_INDEX_PATH)
        
        # Step 2: Define models
        models = get_models()
        logger.info(f"\nModels to evaluate: {list(models.keys())}")
        
        # Step 3: Evaluate all models
        results_df = evaluate_all_models(models, X, y)
        
        # Step 4: Display results
        print_results_table(results_df)
        
        logger.info("\n✓ Baseline evaluation complete!")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
