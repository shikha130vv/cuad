#!/usr/bin/env python3
"""
CUAD Bedrock Labeler Evaluation Script

This script evaluates the performance of the bedrock labeler by comparing its predictions
against ground truth labels from the CUAD dataset.

It calculates:
1. Overall metrics: Accuracy, Precision (macro/micro), Recall (macro/micro), F1 Score (macro/micro)
2. Per-category metrics: Accuracy, Precision, Recall, F1 Score for each of the 41 CUAD categories

Usage:
    python evaluate_labeler.py --predictions <predictions.csv> --ground-truth <ground_truth.csv>
    python evaluate_labeler.py --predictions <predictions.csv> --ground-truth <ground_truth.csv> --output results.json
    python evaluate_labeler.py --predictions <predictions.csv> --ground-truth <ground_truth.csv> --output-format csv

Author: CUAD Evaluation Team
Date: 2025-11-17
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# The 41 CUAD categories (standard order)
CUAD_CATEGORIES = [
    "Document Name",
    "Parties",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
    "Renewal Term",
    "Notice Period To Terminate Renewal",
    "Governing Law",
    "Most Favored Nation",
    "Non-Compete",
    "Exclusivity",
    "No-Solicit Of Customers",
    "No-Solicit Of Employees",
    "Non-Disparagement",
    "Termination For Convenience",
    "Rofr/Rofo/Rofn",
    "Change Of Control",
    "Anti-Assignment",
    "Revenue/Profit Sharing",
    "Price Restrictions",
    "Minimum Commitment",
    "Volume Restriction",
    "Ip Ownership Assignment",
    "Joint Ip Ownership",
    "License Grant",
    "Non-Transferable License",
    "Affiliate License-Licensor",
    "Affiliate License-Licensee",
    "Unlimited/All-You-Can-Eat-License",
    "Irrevocable Or Perpetual License",
    "Source Code Escrow",
    "Post-Termination Services",
    "Audit Rights",
    "Uncapped Liability",
    "Cap On Liability",
    "Liquidated Damages",
    "Warranty Duration",
    "Insurance",
    "Covenant Not To Sue",
    "Third Party Beneficiary",
    "Termination For Cause"
]


def normalize_category_name(category: str) -> str:
    """
    Normalize category name to match CSV column format.
    
    Args:
        category: Original category name
        
    Returns:
        Normalized category name (lowercase, underscores, no special chars)
    """
    return category.lower().replace(" ", "_").replace("/", "_").replace("-", "_")


def load_csv_data(csv_path: str) -> Tuple[List[str], Dict[str, Dict[str, int]]]:
    """
    Load predictions or ground truth from CSV file.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        Tuple of (contract_ids, labels_dict) where labels_dict maps
        contract_id -> {category: label}
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    contract_ids = []
    labels_dict = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            contract_id = row.get('contract_id')
            if not contract_id:
                continue
            
            contract_ids.append(contract_id)
            labels_dict[contract_id] = {}
            
            # Extract label for each category
            for category in CUAD_CATEGORIES:
                normalized_cat = normalize_category_name(category)
                label_col = f'{normalized_cat}_label'
                
                # Get label value, default to 0 if missing
                label_value = row.get(label_col, '0')
                
                # Convert to int, handle various formats
                try:
                    label = int(float(label_value)) if label_value else 0
                except (ValueError, TypeError):
                    label = 0
                
                labels_dict[contract_id][category] = label
    
    return contract_ids, labels_dict


def align_datasets(
    pred_ids: List[str],
    pred_labels: Dict[str, Dict[str, int]],
    gt_ids: List[str],
    gt_labels: Dict[str, Dict[str, int]]
) -> Tuple[List[str], Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]:
    """
    Align prediction and ground truth datasets by contract IDs.
    
    Args:
        pred_ids: List of prediction contract IDs
        pred_labels: Prediction labels dictionary
        gt_ids: List of ground truth contract IDs
        gt_labels: Ground truth labels dictionary
        
    Returns:
        Tuple of (common_ids, aligned_pred_labels, aligned_gt_labels)
    """
    # Find common contract IDs
    pred_id_set = set(pred_ids)
    gt_id_set = set(gt_ids)
    common_ids = sorted(pred_id_set & gt_id_set)
    
    if not common_ids:
        raise ValueError(
            "No common contract IDs found between predictions and ground truth!\n"
            f"Prediction IDs sample: {pred_ids[:5]}\n"
            f"Ground truth IDs sample: {gt_ids[:5]}"
        )
    
    # Create aligned dictionaries
    aligned_pred = {cid: pred_labels[cid] for cid in common_ids}
    aligned_gt = {cid: gt_labels[cid] for cid in common_ids}
    
    # Report alignment statistics
    print(f"\n{'='*60}")
    print(f"Dataset Alignment:")
    print(f"  Prediction contracts: {len(pred_ids)}")
    print(f"  Ground truth contracts: {len(gt_ids)}")
    print(f"  Common contracts: {len(common_ids)}")
    print(f"  Prediction-only contracts: {len(pred_id_set - gt_id_set)}")
    print(f"  Ground truth-only contracts: {len(gt_id_set - pred_id_set)}")
    print(f"{'='*60}\n")
    
    return common_ids, aligned_pred, aligned_gt


def calculate_overall_metrics(
    pred_labels: Dict[str, Dict[str, int]],
    gt_labels: Dict[str, Dict[str, int]],
    contract_ids: List[str]
) -> Dict[str, float]:
    """
    Calculate overall metrics across all categories and contracts.
    
    Args:
        pred_labels: Prediction labels dictionary
        gt_labels: Ground truth labels dictionary
        contract_ids: List of contract IDs to evaluate
        
    Returns:
        Dictionary of overall metrics
    """
    # Flatten all predictions and ground truth into single arrays
    y_pred = []
    y_true = []
    
    for contract_id in contract_ids:
        for category in CUAD_CATEGORIES:
            y_pred.append(pred_labels[contract_id].get(category, 0))
            y_true.append(gt_labels[contract_id].get(category, 0))
    
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'total_samples': len(y_true),
        'total_positive_predictions': int(np.sum(y_pred)),
        'total_positive_ground_truth': int(np.sum(y_true))
    }
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_positives'] = int(tp)
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    
    return metrics


def calculate_category_metrics(
    pred_labels: Dict[str, Dict[str, int]],
    gt_labels: Dict[str, Dict[str, int]],
    contract_ids: List[str],
    category: str
) -> Dict[str, Any]:
    """
    Calculate metrics for a specific category.
    
    Args:
        pred_labels: Prediction labels dictionary
        gt_labels: Ground truth labels dictionary
        contract_ids: List of contract IDs to evaluate
        category: Category name
        
    Returns:
        Dictionary of category-specific metrics
    """
    # Extract predictions and ground truth for this category
    y_pred = []
    y_true = []
    
    for contract_id in contract_ids:
        y_pred.append(pred_labels[contract_id].get(category, 0))
        y_true.append(gt_labels[contract_id].get(category, 0))
    
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    # Calculate metrics
    metrics = {
        'category': category,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'support': int(np.sum(y_true)),  # Number of positive samples in ground truth
        'total_samples': len(y_true),
        'positive_predictions': int(np.sum(y_pred)),
        'positive_ground_truth': int(np.sum(y_true))
    }
    
    # Calculate confusion matrix
    if len(np.unique(y_true)) > 1:  # Only if we have both classes
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
    else:
        # Handle edge case where all samples are one class
        if np.all(y_true == 0):
            metrics['true_negatives'] = int(np.sum((y_pred == 0) & (y_true == 0)))
            metrics['false_positives'] = int(np.sum((y_pred == 1) & (y_true == 0)))
            metrics['true_positives'] = 0
            metrics['false_negatives'] = 0
        else:
            metrics['true_positives'] = int(np.sum((y_pred == 1) & (y_true == 1)))
            metrics['false_negatives'] = int(np.sum((y_pred == 0) & (y_true == 1)))
            metrics['true_negatives'] = 0
            metrics['false_positives'] = 0
    
    return metrics


def evaluate_all_categories(
    pred_labels: Dict[str, Dict[str, int]],
    gt_labels: Dict[str, Dict[str, int]],
    contract_ids: List[str]
) -> List[Dict[str, Any]]:
    """
    Calculate metrics for all categories.
    
    Args:
        pred_labels: Prediction labels dictionary
        gt_labels: Ground truth labels dictionary
        contract_ids: List of contract IDs to evaluate
        
    Returns:
        List of dictionaries containing metrics for each category
    """
    category_metrics = []
    
    for category in CUAD_CATEGORIES:
        metrics = calculate_category_metrics(pred_labels, gt_labels, contract_ids, category)
        category_metrics.append(metrics)
    
    return category_metrics


def print_results(
    overall_metrics: Dict[str, float],
    category_metrics: List[Dict[str, Any]]
):
    """
    Print evaluation results to console in a readable format.
    
    Args:
        overall_metrics: Dictionary of overall metrics
        category_metrics: List of category-specific metrics
    """
    print("\n" + "="*80)
    print("CUAD BEDROCK LABELER EVALUATION RESULTS")
    print("="*80)
    
    # Overall metrics
    print("\n" + "-"*80)
    print("OVERALL METRICS")
    print("-"*80)
    print(f"Total Samples: {overall_metrics['total_samples']:,}")
    print(f"Positive Predictions: {overall_metrics['total_positive_predictions']:,}")
    print(f"Positive Ground Truth: {overall_metrics['total_positive_ground_truth']:,}")
    print()
    print(f"Accuracy:           {overall_metrics['accuracy']:.4f}")
    print()
    print(f"Precision (Macro):  {overall_metrics['precision_macro']:.4f}")
    print(f"Precision (Micro):  {overall_metrics['precision_micro']:.4f}")
    print(f"Precision (Weighted): {overall_metrics['precision_weighted']:.4f}")
    print()
    print(f"Recall (Macro):     {overall_metrics['recall_macro']:.4f}")
    print(f"Recall (Micro):     {overall_metrics['recall_micro']:.4f}")
    print(f"Recall (Weighted):  {overall_metrics['recall_weighted']:.4f}")
    print()
    print(f"F1 Score (Macro):   {overall_metrics['f1_macro']:.4f}")
    print(f"F1 Score (Micro):   {overall_metrics['f1_micro']:.4f}")
    print(f"F1 Score (Weighted): {overall_metrics['f1_weighted']:.4f}")
    print()
    print("Confusion Matrix:")
    print(f"  True Positives:  {overall_metrics['true_positives']:,}")
    print(f"  True Negatives:  {overall_metrics['true_negatives']:,}")
    print(f"  False Positives: {overall_metrics['false_positives']:,}")
    print(f"  False Negatives: {overall_metrics['false_negatives']:,}")
    
    # Category metrics
    print("\n" + "-"*80)
    print("PER-CATEGORY METRICS")
    print("-"*80)
    print(f"{'Category':<45} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Supp':>6}")
    print("-"*80)
    
    for metrics in category_metrics:
        print(
            f"{metrics['category']:<45} "
            f"{metrics['accuracy']:>6.3f} "
            f"{metrics['precision']:>6.3f} "
            f"{metrics['recall']:>6.3f} "
            f"{metrics['f1_score']:>6.3f} "
            f"{metrics['support']:>6}"
        )
    
    # Summary statistics
    print("-"*80)
    accuracies = [m['accuracy'] for m in category_metrics]
    precisions = [m['precision'] for m in category_metrics]
    recalls = [m['recall'] for m in category_metrics]
    f1_scores = [m['f1_score'] for m in category_metrics]
    
    print(f"{'Average':<45} "
          f"{np.mean(accuracies):>6.3f} "
          f"{np.mean(precisions):>6.3f} "
          f"{np.mean(recalls):>6.3f} "
          f"{np.mean(f1_scores):>6.3f}")
    print(f"{'Std Dev':<45} "
          f"{np.std(accuracies):>6.3f} "
          f"{np.std(precisions):>6.3f} "
          f"{np.std(recalls):>6.3f} "
          f"{np.std(f1_scores):>6.3f}")
    
    print("="*80 + "\n")


def save_results_json(
    overall_metrics: Dict[str, float],
    category_metrics: List[Dict[str, Any]],
    output_path: str
):
    """
    Save evaluation results to JSON file.
    
    Args:
        overall_metrics: Dictionary of overall metrics
        category_metrics: List of category-specific metrics
        output_path: Path to save JSON file
    """
    results = {
        'overall_metrics': overall_metrics,
        'category_metrics': category_metrics,
        'summary': {
            'num_categories': len(category_metrics),
            'avg_accuracy': float(np.mean([m['accuracy'] for m in category_metrics])),
            'avg_precision': float(np.mean([m['precision'] for m in category_metrics])),
            'avg_recall': float(np.mean([m['recall'] for m in category_metrics])),
            'avg_f1_score': float(np.mean([m['f1_score'] for m in category_metrics]))
        }
    }
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to JSON: {output_path}")


def save_results_csv(
    overall_metrics: Dict[str, float],
    category_metrics: List[Dict[str, Any]],
    output_path: str
):
    """
    Save evaluation results to CSV file.
    
    Args:
        overall_metrics: Dictionary of overall metrics
        category_metrics: List of category-specific metrics
        output_path: Path to save CSV file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save category metrics
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'category', 'accuracy', 'precision', 'recall', 'f1_score',
            'support', 'total_samples', 'positive_predictions', 'positive_ground_truth',
            'true_positives', 'true_negatives', 'false_positives', 'false_negatives'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(category_metrics)
    
    print(f"✓ Category metrics saved to CSV: {output_path}")
    
    # Save overall metrics to a separate file
    overall_path = output_file.parent / f"{output_file.stem}_overall{output_file.suffix}"
    with open(overall_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        for key, value in overall_metrics.items():
            writer.writerow([key, value])
    
    print(f"✓ Overall metrics saved to CSV: {overall_path}")


def main():
    """Evaluate CUAD Bedrock Labeler performance.
    
    All paths are hardcoded - no arguments needed.
    Simply run: python evaluate_labeler.py
    """
    
    # Hardcoded paths based on directory structure
    predictions_path = "./cuad_processing/output/labeled_contracts.csv"
    ground_truth_path = "./cuad_processing/output/cuad.csv"
    output_path = "./cuad_processing/output/evaluation_results.csv"
    output_format = "csv"  # json or csv
    output_dir = "./cuad_processing/output"
    quiet = False
    
    try:
        # Load data
        print("Loading predictions...")
        pred_ids, pred_labels = load_csv_data(predictions_path)
        print(f"  Loaded {len(pred_ids)} prediction contracts")
        
        print("Loading ground truth...")
        gt_ids, gt_labels = load_csv_data(ground_truth_path)
        print(f"  Loaded {len(gt_ids)} ground truth contracts")
        
        # Align datasets
        common_ids, aligned_pred, aligned_gt = align_datasets(
            pred_ids, pred_labels, gt_ids, gt_labels
        )
        
        # Calculate metrics
        print("Calculating overall metrics...")
        overall_metrics = calculate_overall_metrics(aligned_pred, aligned_gt, common_ids)
        
        print("Calculating per-category metrics...")
        category_metrics = evaluate_all_categories(aligned_pred, aligned_gt, common_ids)
        
        # Print results to console
        if not quiet:
            print_results(overall_metrics, category_metrics)
        
        # Save results to file
        if output_path:
            output_path = output_path
            # Determine format from extension if not specified
            if output_path.endswith('.csv'):
                save_results_csv(overall_metrics, category_metrics, output_path)
            elif output_path.endswith('.json'):
                save_results_json(overall_metrics, category_metrics, output_path)
            else:
                # Use specified format
                if output_path_format == 'csv':
                    output_path = str(Path(output_path).with_suffix('.csv'))
                    save_results_csv(overall_metrics, category_metrics, output_path)
                else:
                    output_path = str(Path(output_path).with_suffix('.json'))
                    save_results_json(overall_metrics, category_metrics, output_path)
        else:
            # Default output path
            output_dir = Path(output_path_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if output_path_format == 'csv':
                output_path = output_dir / 'evaluation_results.csv'
                save_results_csv(overall_metrics, category_metrics, str(output_path))
            else:
                output_path = output_dir / 'evaluation_results.json'
                save_results_json(overall_metrics, category_metrics, str(output_path))
        
        print("\n✓ Evaluation completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n✗ Error during evaluation: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
