#!/usr/bin/env python3
"""
Hyperparameter tuning for TIFUKNN recommendation model
"""

import numpy as np
import json
import pandas as pd
import argparse
import os
import sys
import random
from itertools import product
from datetime import datetime
from tqdm import tqdm

# Import the TIFUKNN functions
from tifuknn import (
    temporal_decay_sum_history, 
    KNN, 
    merge_history, 
    evaluate,
    vec2label_list
)

# Import the existing evaluator
import sys
sys.path.append('../../evaluations/')
from evaluate_recommendation_model import RecommendationEvaluator


class TIFUKNNHyperparameterTuner:
    def __init__(self, dataset_name, base_path=None):
        self.dataset_name = dataset_name
        self.base_path = base_path or ""
        
        # Load dataset files
        self.history_file = f"../../datasets/{dataset_name}/history.json"
        self.future_file = f"../../datasets/{dataset_name}/future.json"
        self.keyset_file = f"../../datasets/{dataset_name}/keyset_0.json"
        
        # Load data
        self.load_data()
        
        # Initialize parameter ranges
        self.param_ranges = self._get_parameter_ranges()
        
    def load_data(self):
        """Load dataset files"""
        print(f"{datetime.now()}: Loading dataset files for {self.dataset_name}")
        
        with open(self.history_file, 'r') as f:
            self.data_history = json.load(f)
        with open(self.keyset_file, 'r') as f:
            self.keyset = json.load(f)
        with open(self.future_file, 'r') as f:
            self.ground_truth = json.load(f)
            
        # Extract dataset information
        self.input_size = self.keyset['item_num']
        self.keyset_train = self.keyset['train']
        self.keyset_val = self.keyset['val']
        self.keyset_test = self.keyset['test']
        
        print(f"{datetime.now()}: Dataset loaded successfully")
        print(f"  - Item count: {self.input_size}")
        print(f"  - Train users: {len(self.keyset_train)}")
        print(f"  - Validation users: {len(self.keyset_val)}")
        print(f"  - Test users: {len(self.keyset_test)}")
        
        # Initialize evaluator for validation set with correct paths
        self.evaluator = RecommendationEvaluator(self.dataset_name, "tifuknn", split='val')
        # Update paths to be relative to tifuknn directory
        self.evaluator.predictions_path = f"../../predictions/{self.dataset_name}/tifuknn/keyset0.json"
        self.evaluator.dataset_path = f"../../datasets/{self.dataset_name}/future.json"
        self.evaluator.keyset_path = f"../../datasets/{self.dataset_name}/keyset_0.json"

        
    def _get_parameter_ranges(self):
        """Define hyperparameter search space"""
        num_users = len(self.keyset_train)
        
        # num_nearest_neighbors: multiples of 20 or 200 between len(users)/10 and len(users)/100
        min_nn = max(10, num_users // 100)
        max_nn = num_users // 10
        nn_values = []
        for i in range(min_nn, max_nn + 1):
            if i <= 1600 and i % 200 == 0:
                nn_values.append(i)
            elif i <= 100 and i % 20 == 0:
                nn_values.append(i)
        if not nn_values:  # fallback if no valid values
            nn_values = [10, 50, 100, min(num_users, 500)]
        
        return {
            'num_nearest_neighbors': nn_values,
            'within_decay_rate': [0.5, 0.6, 0.7, 0.8, 0.9],
            'group_decay_rate': [0.5, 0.6, 0.7, 0.8, 0.9],
            'alpha': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            'num_groups': list(range(2, 11))  # 2 to 10
        }
        
    def evaluate_predictions(self, predictions):
        """Evaluate predictions using the existing RecommendationEvaluator"""
        # Save predictions temporarily
        temp_pred_path = f"../../predictions/{self.dataset_name}/tifuknn/keyset0.json"
        os.makedirs(os.path.dirname(temp_pred_path), exist_ok=True)
        
        with open(temp_pred_path, 'w') as f:
            json.dump(predictions, f)
        
        self.evaluator.load_data()
        eval_df, avg_metrics = self.evaluator.evaluate()
        return avg_metrics
    
    def run_single_experiment(self, params):
        """Run a single experiment with given parameters"""
        # Generate predictions for validation set
        predictions = evaluate(
            self.data_history, 
            self.keyset_train, 
            self.keyset_val, 
            self.input_size,
            params['num_groups'], 
            params['within_decay_rate'], 
            params['group_decay_rate'],
            params['num_nearest_neighbors'], 
            params['alpha']
        )
        
        # Evaluate predictions (k ignored downstream)
        metrics = self.evaluate_predictions(predictions)
        
        return metrics
    
    def grid_search(self, metric='nDCG@5', max_combinations=None, seed=42):
        """Perform grid search over hyperparameter space"""
        print(f"{datetime.now()}: Starting grid search for {self.dataset_name}")
        print(f"Target metric: {metric}")
        print(f"Random seed: {seed}")
        print(f"Parameter ranges: {self.param_ranges}")
        
        # Set random seeds for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        
        # Generate all parameter combinations
        param_names = list(self.param_ranges.keys())
        param_values = list(self.param_ranges.values())
        all_combinations = list(product(*param_values))
        
        if max_combinations and len(all_combinations) > max_combinations:
            print(f"Limiting search to {max_combinations} random combinations from {len(all_combinations)} total")
            selected_indices = np.random.choice(len(all_combinations), max_combinations, replace=False)
            all_combinations = [all_combinations[i] for i in selected_indices]
        
        print(f"Total combinations to test: {len(all_combinations)}")
        
        results = []
        best_score = -1
        best_params = None
        
        # Run experiments
        for i, combination in enumerate(tqdm(all_combinations, desc="Testing combinations")):
            params = dict(zip(param_names, combination))
            
            # Run experiment
            metrics = self.run_single_experiment(params)
            
            score = metrics[metric]
            results.append({
                'combination_id': i,
                'parameters': params.copy(),
                'metrics': metrics.copy(),
                'score': score
            })
            
            if score > best_score:
                best_score = score
                best_params = params.copy()

            print(f"{datetime.now()}: parameters: {params} = score: {score}")
                    
        
        # Sort results by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results, best_params, best_score
    
    def print_results(self, results, best_params, best_score, metric='nDCG@5', top_n=3):
        """Print hyperparameter tuning results"""
        print("\n" + "="*80)
        print(f"TIFUKNN HYPERPARAMETER TUNING RESULTS - {self.dataset_name.upper()}")
        print("="*80)
        print(f"Best {metric}: {best_score:.4f}")
        print(f"Best parameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        
        print(f"\nTop {top_n} results:")
        print("-" * 80)
        for i, result in enumerate(results[:top_n]):
            print(f"Rank {i+1}: {metric} = {result['score']:.4f}")
            for param, value in result['parameters'].items():
                print(f"  {param}: {value}")
            print()
    
    def save_results(self, results, best_params, best_score, metric='nDCG@5', seed=42):
        """Save results to JSON file"""
        results_data = {
            'dataset': self.dataset_name,
            'best_metric': metric,
            'best_score': best_score,
            'best_parameters': best_params,
            'all_results': results,
            'random_seed': seed,
            'timestamp': datetime.now().isoformat()
        }
        
        os.makedirs(f"../../hyperparameter_results/{self.dataset_name}", exist_ok=True)
        results_file = f"../../hyperparameter_results/{self.dataset_name}/tifuknn.json"
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"{datetime.now()}: Results saved to {results_file}")
    
    def save_best_params_to_config(self, best_params, best_score, metric='nDCG@5'):
        """Save best parameters to a global config JSON file"""
        config_file = "best_params_config.json"
        
        # Load existing config or create new one
        config = {}
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
        
        # Add or update the dataset entry
        config[self.dataset_name] = {
            'best_parameters': best_params,
            'best_score': best_score,
            'metric': metric,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save updated config
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"{datetime.now()}: Best parameters saved to config: {config_file}")
    
        
  
    
    def tune(self, metric='nDCG@5', max_combinations=None, top_n=3, seed=42):
        """Main tuning function"""
        print(f"{datetime.now()}: Starting TIFUKNN hyperparameter tuning")
        
        # Run grid search
        results, best_params, best_score = self.grid_search(metric, max_combinations, seed)
        
        # Print results
        self.print_results(results, best_params, best_score, metric, top_n)
        
        # Save results
        self.save_results(results, best_params, best_score, metric, seed)
        
        # Save best parameters to global config
        self.save_best_params_to_config(best_params, best_score, metric)
        
        return results, best_params, best_score

def main():
    parser = argparse.ArgumentParser(description='TIFUKNN Hyperparameter Tuning')
    parser.add_argument('dataset_name', help='Name of the dataset (e.g., dunnhumby, instacart)')
    parser.add_argument('--metric', '-m', default='nDCG@5', 
                       choices=['HR@5', 'WHR@5', 'nDCG@5', 'recall@5'],
                       help='Metric to optimize (default: nDCG@5)')
    parser.add_argument('--max_combinations', '-max', type=int, default=30,
                       help='Maximum number of parameter combinations to test (default: 30)')
    parser.add_argument('--top_n', '-n', type=int, default=3,
                       help='Number of top results to display (default: 10)')
    parser.add_argument('--seed', '-s', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Initialize tuner
    tuner = TIFUKNNHyperparameterTuner(args.dataset_name)
    
    # Run tuning
    results, best_params, best_score = tuner.tune(
        metric=args.metric,
        max_combinations=args.max_combinations,
        top_n=args.top_n,
        seed=args.seed
    )
    
    print(f"\n{datetime.now()}: Hyperparameter tuning completed successfully!")

if __name__ == "__main__":
    main()
