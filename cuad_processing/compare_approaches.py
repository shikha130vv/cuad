#!/usr/bin/env python3
"""
CUAD Comparative Evaluation Script

This script automatically compares the performance of different approaches (Standard vs LangGraph)
for CUAD contract labeling by running evaluations on both and generating comparative analysis.

Features:
- Auto-detects prediction files from both approaches
- Minimal command-line arguments required
- Generates side-by-side comparison tables
- Shows which approach performs better for each metric
- Outputs results in console, JSON, and CSV formats

Usage:
    # Basic usage (auto-detect all files)
    python compare_approaches.py
    
    # Specify custom files
    python compare_approaches.py --ground-truth output/cuad.csv --standard-predictions output/labeled_contracts.csv
    
    # Custom output location
    python compare_approaches.py --output comparison_results.json

Author: CUAD Evaluation Team
Date: 2025-11-17
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import glob

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
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
    """Normalize category name to match CSV column format."""
    return category.lower().replace(" ", "_").replace("/", "_").replace("-", "_")


def auto_detect_file(pattern: str, description: str) -> Optional[str]:
    """
    Auto-detect a file based on a glob pattern.
    
    Args:
        pattern: Glob pattern to search for
        description: Description of the file being searched
        
    Returns:
        Path to the detected file or None if not found
    """
    matches = glob.glob(pattern)
    if matches:
        # Return the most recently modified file
        latest_file = max(matches, key=lambda x: Path(x).stat().st_mtime)
        print(f"  ✓ Auto-detected {description}: {latest_file}")
        return latest_file
    return None


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
    """Align prediction and ground truth datasets by contract IDs."""
    pred_id_set = set(pred_ids)
    gt_id_set = set(gt_ids)
    common_ids = sorted(pred_id_set & gt_id_set)
    
    if not common_ids:
        raise ValueError(
            "No common contract IDs found between predictions and ground truth!"
        )
    
    aligned_pred = {cid: pred_labels[cid] for cid in common_ids}
    aligned_gt = {cid: gt_labels[cid] for cid in common_ids}
    
    return common_ids, aligned_pred, aligned_gt


def calculate_overall_metrics(
    pred_labels: Dict[str, Dict[str, int]],
    gt_labels: Dict[str, Dict[str, int]],
    contract_ids: List[str]
) -> Dict[str, float]:
    """Calculate overall metrics across all categories and contracts."""
    y_pred = []
    y_true = []
    
    for contract_id in contract_ids:
        for category in CUAD_CATEGORIES:
            y_pred.append(pred_labels[contract_id].get(category, 0))
            y_true.append(gt_labels[contract_id].get(category, 0))
    
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'total_samples': len(y_true)
    }
    
    return metrics


def calculate_category_metrics(
    pred_labels: Dict[str, Dict[str, int]],
    gt_labels: Dict[str, Dict[str, int]],
    contract_ids: List[str],
    category: str
) -> Dict[str, Any]:
    """Calculate metrics for a specific category."""
    y_pred = []
    y_true = []
    
    for contract_id in contract_ids:
        y_pred.append(pred_labels[contract_id].get(category, 0))
        y_true.append(gt_labels[contract_id].get(category, 0))
    
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    metrics = {
        'category': category,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'support': int(np.sum(y_true))
    }
    
    return metrics


def evaluate_approach(
    pred_path: str,
    gt_path: str,
    approach_name: str
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    Evaluate a single approach.
    
    Args:
        pred_path: Path to predictions CSV
        gt_path: Path to ground truth CSV
        approach_name: Name of the approach (for logging)
        
    Returns:
        Tuple of (overall_metrics, category_metrics)
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {approach_name} Approach")
    print(f"{'='*60}")
    
    # Load data
    print(f"  Loading predictions from: {pred_path}")
    pred_ids, pred_labels = load_csv_data(pred_path)
    print(f"    ✓ Loaded {len(pred_ids)} contracts")
    
    print(f"  Loading ground truth from: {gt_path}")
    gt_ids, gt_labels = load_csv_data(gt_path)
    print(f"    ✓ Loaded {len(gt_ids)} contracts")
    
    # Align datasets
    common_ids, aligned_pred, aligned_gt = align_datasets(
        pred_ids, pred_labels, gt_ids, gt_labels
    )
    print(f"    ✓ Aligned {len(common_ids)} common contracts")
    
    # Calculate metrics
    print("  Calculating metrics...")
    overall_metrics = calculate_overall_metrics(aligned_pred, aligned_gt, common_ids)
    
    category_metrics = []
    for category in CUAD_CATEGORIES:
        metrics = calculate_category_metrics(aligned_pred, aligned_gt, common_ids, category)
        category_metrics.append(metrics)
    
    print(f"    ✓ Evaluation complete")
    
    return overall_metrics, category_metrics


def compare_metrics(
    standard_overall: Dict[str, float],
    standard_category: List[Dict[str, Any]],
    langgraph_overall: Dict[str, float],
    langgraph_category: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compare metrics between two approaches.
    
    Returns:
        Dictionary containing comparison results
    """
    comparison = {
        'overall_comparison': {},
        'category_comparison': [],
        'summary': {}
    }
    
    # Compare overall metrics
    key_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    
    for metric in key_metrics:
        standard_val = standard_overall[metric]
        langgraph_val = langgraph_overall[metric]
        diff = langgraph_val - standard_val
        pct_change = (diff / standard_val * 100) if standard_val > 0 else 0
        
        comparison['overall_comparison'][metric] = {
            'standard': standard_val,
            'langgraph': langgraph_val,
            'difference': diff,
            'percent_change': pct_change,
            'winner': 'LangGraph' if langgraph_val > standard_val else 'Standard' if standard_val > langgraph_val else 'Tie'
        }
    
    # Compare category metrics
    standard_wins = 0
    langgraph_wins = 0
    ties = 0
    
    for std_cat, lg_cat in zip(standard_category, langgraph_category):
        category = std_cat['category']
        
        # Compare F1 scores (primary metric)
        std_f1 = std_cat['f1_score']
        lg_f1 = lg_cat['f1_score']
        
        if lg_f1 > std_f1:
            winner = 'LangGraph'
            langgraph_wins += 1
        elif std_f1 > lg_f1:
            winner = 'Standard'
            standard_wins += 1
        else:
            winner = 'Tie'
            ties += 1
        
        diff = lg_f1 - std_f1
        pct_change = (diff / std_f1 * 100) if std_f1 > 0 else 0
        
        comparison['category_comparison'].append({
            'category': category,
            'standard_accuracy': std_cat['accuracy'],
            'langgraph_accuracy': lg_cat['accuracy'],
            'standard_precision': std_cat['precision'],
            'langgraph_precision': lg_cat['precision'],
            'standard_recall': std_cat['recall'],
            'langgraph_recall': lg_cat['recall'],
            'standard_f1': std_f1,
            'langgraph_f1': lg_f1,
            'f1_difference': diff,
            'f1_percent_change': pct_change,
            'winner': winner,
            'support': std_cat['support']
        })
    
    # Summary statistics
    comparison['summary'] = {
        'total_categories': len(CUAD_CATEGORIES),
        'standard_wins': standard_wins,
        'langgraph_wins': langgraph_wins,
        'ties': ties,
        'standard_win_rate': standard_wins / len(CUAD_CATEGORIES) * 100,
        'langgraph_win_rate': langgraph_wins / len(CUAD_CATEGORIES) * 100
    }
    
    return comparison


def print_comparison_results(comparison: Dict[str, Any]):
    """Print comparison results in a formatted table."""
    print("\n" + "="*100)
    print("COMPARATIVE EVALUATION RESULTS: STANDARD vs LANGGRAPH")
    print("="*100)
    
    # Overall metrics comparison
    print("\n" + "-"*100)
    print("OVERALL METRICS COMPARISON")
    print("-"*100)
    print(f"{'Metric':<20} {'Standard':>12} {'LangGraph':>12} {'Difference':>12} {'% Change':>12} {'Winner':>12}")
    print("-"*100)
    
    for metric, values in comparison['overall_comparison'].items():
        metric_name = metric.replace('_', ' ').title()
        print(
            f"{metric_name:<20} "
            f"{values['standard']:>12.4f} "
            f"{values['langgraph']:>12.4f} "
            f"{values['difference']:>+12.4f} "
            f"{values['percent_change']:>+11.2f}% "
            f"{values['winner']:>12}"
        )
    
    # Category-wise comparison
    print("\n" + "-"*100)
    print("CATEGORY-WISE F1 SCORE COMPARISON")
    print("-"*100)
    print(f"{'Category':<45} {'Standard':>10} {'LangGraph':>10} {'Diff':>10} {'% Chg':>10} {'Winner':>12}")
    print("-"*100)
    
    for cat in comparison['category_comparison']:
        print(
            f"{cat['category']:<45} "
            f"{cat['standard_f1']:>10.4f} "
            f"{cat['langgraph_f1']:>10.4f} "
            f"{cat['f1_difference']:>+10.4f} "
            f"{cat['f1_percent_change']:>+9.2f}% "
            f"{cat['winner']:>12}"
        )
    
    # Summary
    print("-"*100)
    summary = comparison['summary']
    print(f"\nSUMMARY:")
    print(f"  Total Categories: {summary['total_categories']}")
    print(f"  Standard Wins: {summary['standard_wins']} ({summary['standard_win_rate']:.1f}%)")
    print(f"  LangGraph Wins: {summary['langgraph_wins']} ({summary['langgraph_win_rate']:.1f}%)")
    print(f"  Ties: {summary['ties']}")
    
    # Determine overall winner
    if summary['langgraph_wins'] > summary['standard_wins']:
        overall_winner = "LangGraph"
    elif summary['standard_wins'] > summary['langgraph_wins']:
        overall_winner = "Standard"
    else:
        overall_winner = "Tie"
    
    print(f"\n  🏆 Overall Winner: {overall_winner}")
    print("="*100 + "\n")


def save_comparison_json(
    comparison: Dict[str, Any],
    standard_overall: Dict[str, float],
    standard_category: List[Dict[str, Any]],
    langgraph_overall: Dict[str, float],
    langgraph_category: List[Dict[str, Any]],
    output_path: str
):
    """Save comparison results to JSON file."""
    results = {
        'comparison': comparison,
        'standard_approach': {
            'overall_metrics': standard_overall,
            'category_metrics': standard_category
        },
        'langgraph_approach': {
            'overall_metrics': langgraph_overall,
            'category_metrics': langgraph_category
        }
    }
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Comparison results saved to JSON: {output_path}")


def save_comparison_csv(comparison: Dict[str, Any], output_path: str):
    """Save comparison results to CSV file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save category comparison
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'category', 'standard_accuracy', 'langgraph_accuracy',
            'standard_precision', 'langgraph_precision',
            'standard_recall', 'langgraph_recall',
            'standard_f1', 'langgraph_f1',
            'f1_difference', 'f1_percent_change', 'winner', 'support'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison['category_comparison'])
    
    print(f"✓ Category comparison saved to CSV: {output_path}")
    
    # Save overall comparison
    overall_path = output_file.parent / f"{output_file.stem}_overall{output_file.suffix}"
    with open(overall_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Standard', 'LangGraph', 'Difference', 'Percent_Change', 'Winner'])
        for metric, values in comparison['overall_comparison'].items():
            writer.writerow([
                metric,
                values['standard'],
                values['langgraph'],
                values['difference'],
                values['percent_change'],
                values['winner']
            ])
    
    print(f"✓ Overall comparison saved to CSV: {overall_path}")
    
    # Save summary
    summary_path = output_file.parent / f"{output_file.stem}_summary{output_file.suffix}"
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        for key, value in comparison['summary'].items():
            writer.writerow([key, value])
    
    print(f"✓ Summary saved to CSV: {summary_path}")


def main():
    """Compare Standard and LangGraph approaches for CUAD contract labeling.
    
    All paths are hardcoded - no arguments needed.
    Simply run: python compare_approaches.py
    """
    
    # Hardcoded paths based on directory structure
    ground_truth_path = "cuad_processing/output/cuad.csv"
    standard_predictions_path = "cuad_processing/output/labeled_contracts.csv"
    langgraph_predictions_path = "cuad_processing/output/labeled_contracts_langgraph.csv"
    output_path = "cuad_processing/output/comparison_results.json"
    output_format = "both"  # json, csv, or both
    
    try:
        print("\n" + "="*100)
        print("CUAD COMPARATIVE EVALUATION: AUTO-DETECTION MODE")
        print("="*100)
        
        # Auto-detect files if not provided
        print("\nAuto-detecting files...")
        
        # Ground truth
        if ground_truth_path is None:
            gt_path = auto_detect_file('output/cuad.csv', 'ground truth')
            if gt_path is None:
                gt_path = auto_detect_file('cuad_processing/output/cuad.csv', 'ground truth')
            if gt_path is None:
                raise FileNotFoundError(
                    "Could not auto-detect ground truth file. "
                    "Please specify with --ground-truth"
                )
        else:
            gt_path = ground_truth_path
            print(f"  ✓ Using specified ground truth: {gt_path}")
        
        # Standard predictions
        if standard_predictions_path is None:
            std_path = auto_detect_file('output/labeled_contracts.csv', 'standard predictions')
            if std_path is None:
                std_path = auto_detect_file('cuad_processing/output/labeled_contracts.csv', 'standard predictions')
            if std_path is None:
                raise FileNotFoundError(
                    "Could not auto-detect standard predictions file. "
                    "Please specify with --standard-predictions"
                )
        else:
            std_path = standard_predictions_path
            print(f"  ✓ Using specified standard predictions: {std_path}")
        
        # LangGraph predictions
        if langgraph_predictions_path is None:
            lg_path = auto_detect_file('output/labeled_contracts_langgraph.csv', 'LangGraph predictions')
            if lg_path is None:
                lg_path = auto_detect_file('cuad_processing/output/labeled_contracts_langgraph.csv', 'LangGraph predictions')
            if lg_path is None:
                raise FileNotFoundError(
                    "Could not auto-detect LangGraph predictions file. "
                    "Please specify with --langgraph-predictions"
                )
        else:
            lg_path = langgraph_predictions_path
            print(f"  ✓ Using specified LangGraph predictions: {lg_path}")
        
        # Evaluate both approaches
        standard_overall, standard_category = evaluate_approach(
            std_path, gt_path, "Standard"
        )
        
        langgraph_overall, langgraph_category = evaluate_approach(
            lg_path, gt_path, "LangGraph"
        )
        
        # Compare results
        print("\n" + "="*60)
        print("Generating Comparison Analysis")
        print("="*60)
        comparison = compare_metrics(
            standard_overall, standard_category,
            langgraph_overall, langgraph_category
        )
        
        # Print results
        print_comparison_results(comparison)
        
        # Save results
        output_path = output_path
        
        if output_path_format in ['json', 'both']:
            json_path = str(Path(output_path).with_suffix('.json'))
            save_comparison_json(
                comparison,
                standard_overall, standard_category,
                langgraph_overall, langgraph_category,
                json_path
            )
        
        if output_path_format in ['csv', 'both']:
            csv_path = str(Path(output_path).with_suffix('.csv'))
            save_comparison_csv(comparison, csv_path)
        
        print("\n✓ Comparative evaluation completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n✗ Error during comparative evaluation: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
