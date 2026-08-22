#!/usr/bin/env python3
"""
General evaluation script for recommendation models on any dataset
Based on the evaluation logic from get_metrics.py and evaluate.py
"""

import json
import pandas as pd
import numpy as np
import argparse
import os
import glob
from operator import mul
from datetime import datetime

class RecommendationEvaluator:
    def __init__(self, dataset_name, model_name, base_path=None):
        self.dataset_name = dataset_name
        self.model_name = model_name
            
        # Construct paths
        self.dataset_path = f"../datasets/{dataset_name}/future.json"
        self.predictions_path = f"../predictions/{dataset_name}/{model_name}/keyset0.json"
    
        
    
    def load_data(self):
        """Load predictions and ground truth data"""
        print(f"{datetime.now()}: Loading predictions from {self.predictions_path}")
        with open(self.predictions_path, 'r') as f:
            self.predictions = json.load(f)
        
        print(f"{datetime.now()}: Loading ground truth from {self.dataset_path}")
        with open(self.dataset_path, 'r') as f:
            self.ground_truth = json.load(f)
            
    
    def match_k(self, purchases, predictions, k=5):
        """Check which predictions match the ground truth purchases"""
        count_k = [0] * min(k, len(predictions))
        for i in range(len(count_k)):
            if predictions[i] in purchases:
                count_k[i] = 1
        return count_k
    
    def calculate_precision_at_k(self, values):
        """Calculate precision@k (k == len(values))"""
        if not values:
            return 0.0
        return sum(values) / len(values)
    
    def calculate_ndcg(self, values, total_purchases):
        """Calculate normalized DCG"""
        n = min(len(values), total_purchases)
        dcg = sum([values[i] / np.log(i + 2) for i in range(n)])
        idcg = sum([1 / np.log(i + 2) for i in range(n)])
        if idcg == 0:
            return 0
        return dcg / idcg
    

    
    def evaluate(self, ks=(5, 10)):
        """Main evaluation function for specified k values (defaults to 5 and 10)"""
        print(f"{datetime.now()}: Starting evaluation for {self.dataset_name}/{self.model_name}")

        # Ensure ks is an iterable of ints
        ks = tuple(sorted(set(int(k) for k in ks)))

        eval_rows = []
        skipped_users = 0

        for i, (user_id, user_predictions) in enumerate(self.predictions.items()):

            if user_id not in self.ground_truth:
                skipped_users += 1
                print(f"Warning: User {user_id} not found in ground truth")
                continue

            ground_truth_purchases = self.ground_truth[user_id][1]
            total_purchases = len(ground_truth_purchases)

            row = {
                'user': user_id,
                'total purchases': total_purchases,
            }

            for k in ks:
                # Build matches at k
                matches_k = self.match_k(ground_truth_purchases, user_predictions, k=k)

                # Weights for WHR@k
                weights_k = [1 - (s - 1) / k for s in range(1, k + 1)]

                # HR@k
                denom_hr = min(total_purchases, len(matches_k)) if len(matches_k) > 0 else 1
                row[f'HR@{k}'] = sum(matches_k) / denom_hr

                # WHR@k (follow original approach: weight up to purchases count)
                # Note: map stops at shortest length, mirroring original semantics
                weight_cut = weights_k[:total_purchases]
                denom_whr = sum(weight_cut) if weight_cut else 1
                row[f'WHR@{k}'] = sum(map(mul, matches_k, weight_cut)) / denom_whr

                # nDCG@k
                row[f'nDCG@{k}'] = self.calculate_ndcg(matches_k, total_purchases)

                # recall@k
                row[f'recall@{k}'] = (sum(matches_k) / total_purchases) if total_purchases > 0 else 0.0

                # precision@k (exactly at k)
                row[f'precision@{k}'] = self.calculate_precision_at_k(matches_k)

            eval_rows.append(row)

        eval_df = pd.DataFrame(eval_rows)

        # Average metrics across users (only the new metrics)
        avg_metrics = {}
        metric_cols = [
            *(f'HR@{k}' for k in ks),
            *(f'WHR@{k}' for k in ks),
            *(f'nDCG@{k}' for k in ks),
            *(f'recall@{k}' for k in ks),
            *(f'precision@{k}' for k in ks),
        ]
        for col in metric_cols:
            if col in eval_df.columns:
                avg_metrics[col] = eval_df[col].mean()

        print(f"{datetime.now()}: Evaluation completed successfully for {len(eval_df)} users!")

        return eval_df, avg_metrics
    
    def print_results(self, avg_metrics, total_users):
        """Print evaluation results (only HR@k, WHR@k, nDCG@k, recall@k, precision@k for k=5,10)"""
        print("\n" + "="*70)
        print(f"{self.dataset_name.upper()} {self.model_name.upper()} EVALUATION RESULTS")
        print("="*70)
        print(f"Dataset: {self.dataset_name}")
        print(f"Model: {self.model_name}")
        print(f"Total users evaluated: {total_users}")
        print("\nAverage Metrics:")
        for metric in [
            'HR@5', 'WHR@5', 'nDCG@5', 'recall@5', 'precision@5',
            'HR@10', 'WHR@10', 'nDCG@10', 'recall@10', 'precision@10',
        ]:
            value = avg_metrics.get(metric, None)
            if value is not None:
                print(f"{metric:15}: {value:.4f}")
            else:
                print(f"{metric:15}: N/A")
    


def find_keyset_files(dataset_name, model_name):
    """Find all keyset files in the predictions directory"""
    predictions_dir = f"../predictions/{dataset_name}/{model_name}/"
    if not os.path.exists(predictions_dir):
        return []
    
    keyset_files = glob.glob(f"{predictions_dir}/keyset*.json")
    keyset_files.sort()  # Sort for consistent ordering
    return keyset_files

def calculate_statistics(metrics_list):
    """Calculate mean and standard error for a list of metric dictionaries"""
     
    # Get all metric keys
    all_metrics = set()
    for metrics in metrics_list:
        all_metrics.update(metrics.keys())
    
    statistics = {}
    for metric in all_metrics:
        values = [metrics[metric] for metrics in metrics_list]
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1) if len(values) > 1 else 0
            std_error = std_val / np.sqrt(len(values))
            statistics[metric] = {
                'mean': mean_val,
                'std_error': std_error,
                'count': len(values)
            }
    
    return statistics

def print_keyset_statistics(statistics, total_keysets):
    """Print statistics across all keysets"""
    print("\n" + "="*70)
    print("KEYSET STATISTICS ACROSS ALL FOLDS")
    print("="*70)
    print(f"Total keysets evaluated: {total_keysets}")
    print("\nAverage Metrics (Mean ± Standard Error):")
    for metric, stats in statistics.items():
        print(f"{metric:15}: {stats['mean']:.4f} ± {stats['std_error']:.4f} (n={stats['count']})")

def save_detailed_results(eval_df, dataset_name, model_name, keyset_index):
    """Save detailed results for a specific keyset"""
    results_dir = f"../results/{dataset_name}"
    os.makedirs(results_dir, exist_ok=True)
    
    detailed_file = f"{results_dir}/{model_name}_keyset{keyset_index}_detailed_results.csv"
    eval_df.to_csv(detailed_file, index=False)
    print(f"Detailed results saved to: {detailed_file}")

def save_summary_results(statistics, dataset_name, model_name, keyset_index=None):
    """Save summary results with standard errors"""
    results_dir = f"../results/{dataset_name}"
    os.makedirs(results_dir, exist_ok=True)
    
    if keyset_index is not None:
        # Single keyset summary
        summary_file = f"{results_dir}/{model_name}_keyset{keyset_index}_summary_metrics.csv"
        summary_data = []
        for metric, value in statistics.items():
            summary_data.append({
                'metric': metric,
                'value': value,
                'std_error': None,
                'count': 1
            })
    else:
        # Multiple keysets summary with standard errors
        summary_file = f"{results_dir}/{model_name}_summary_metrics.csv"
        summary_data = []
        for metric, stats in statistics.items():
            summary_data.append({
                'metric': metric,
                'value': stats['mean'],
                'std_error': stats['std_error'],
                'count': stats['count']
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary results saved to: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate recommendation models on any dataset')
    parser.add_argument('dataset_name', help='Name of the dataset (e.g., dunnhumby, instacart)')
    parser.add_argument('model_name', help='Name of the model (e.g., tifuknn, gru4rec)')
    parser.add_argument('--keyset', type=int, help='Specific keyset index to evaluate (default: evaluate all)')

    
    args = parser.parse_args()
    
    if args.keyset is not None:
        # Evaluate specific keyset
        evaluator = RecommendationEvaluator(args.dataset_name, args.model_name)
        evaluator.predictions_path = f"../predictions/{args.dataset_name}/{args.model_name}/keyset{args.keyset}.json"
        
        evaluator.load_data()
        eval_df, avg_metrics = evaluator.evaluate()
        evaluator.print_results(avg_metrics, len(eval_df))
        
        # Save results for single keyset
        save_detailed_results(eval_df, args.dataset_name, args.model_name, args.keyset)
        save_summary_results(avg_metrics, args.dataset_name, args.model_name, args.keyset)
        
    else:
        # Evaluate all keysets
        keyset_files = find_keyset_files(args.dataset_name, args.model_name)
        
       
        print(f"Found {len(keyset_files)} keyset files:")
        
        all_metrics = []
        
        for keyset_file in keyset_files:
            # Extract keyset index from filename
            filename = os.path.basename(keyset_file)
            keyset_index = filename.replace('keyset', '').replace('.json', '')
            
            print(f"\n{datetime.now()}: Evaluating {filename}...")
            
            # Create evaluator and update path
            evaluator = RecommendationEvaluator(args.dataset_name, args.model_name)
            evaluator.predictions_path = keyset_file
            
            evaluator.load_data()
            eval_df, avg_metrics = evaluator.evaluate()
            all_metrics.append(avg_metrics)
            print(f"Completed evaluation for keyset {keyset_index}")
            
            # Save detailed results for this keyset
            save_detailed_results(eval_df, args.dataset_name, args.model_name, keyset_index)
                
        
        # Calculate statistics across all keysets
        statistics = calculate_statistics(all_metrics)
        print_keyset_statistics(statistics, len(all_metrics))
        
        # Save summary results with standard errors
        save_summary_results(statistics, args.dataset_name, args.model_name)
 
    
    print(f"\n{datetime.now()}: Evaluation completed successfully!")
        
    
if __name__ == "__main__":
    main()
