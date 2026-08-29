#!/usr/bin/env python3
"""
Hyperparameter tuning for RACF recommendation model
"""

import argparse
import json
import os
from datetime import datetime
from itertools import product
import random

import numpy as np
import pandas as pd
from scipy import sparse

# Import RACF building blocks to avoid duplication
from racf import (
    data_parser,
    build_user_item_matrix,
    compute_rc_matrix,
    compute_user_similarity,
    topk_from_rec_matrix,
)

# Import the existing evaluator
import sys
sys.path.append('../../evaluations/')
from evaluate_recommendation_model import RecommendationEvaluator


class RACFHyperparameterTuner:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

        # Data paths
        self.history_file = f"../../datasets/{dataset_name}/history.json"
        self.future_file = f"../../datasets/{dataset_name}/future.json"
        self.keyset_file = f"../../datasets/{dataset_name}/keyset_0.json"

        self.load_data()
        self.param_ranges = self._get_parameter_ranges()

    def load_data(self):
        print(f"{datetime.now()}: Loading dataset files for {self.dataset_name}")
        with open(self.history_file, 'r') as f:
            self.history = json.load(f)
        with open(self.future_file, 'r') as f:
            self.ground_truth = json.load(f)
        with open(self.keyset_file, 'r') as f:
            self.keyset = json.load(f)

        # Build training dataframe once (reuse racf.data_parser)
        self.df_train = data_parser(self.dataset_name)

        self.user_count = int(self.df_train.UID.max()) + 1 if not self.df_train.empty else 0
        self.item_count = int(self.df_train.PID.max()) + 1 if not self.df_train.empty else 0

        # Useful keysets
        self.keyset_train = self.keyset['train']
        self.keyset_val = self.keyset['val']

        # Evaluator setup
        self.evaluator = RecommendationEvaluator(self.dataset_name, "racf", split='val')
        self.evaluator.predictions_path = f"../../predictions/{self.dataset_name}/racf/keyset0.json"
        self.evaluator.dataset_path = f"../../datasets/{self.dataset_name}/future.json"
        self.evaluator.keyset_path = f"../../datasets/{self.dataset_name}/keyset_0.json"

        print(f"{datetime.now()}: Dataset loaded successfully")
        print(f"  - Users: {self.user_count}")
        print(f"  - Items: {self.item_count}")
        print(f"  - Train users: {len(self.keyset_train)}")
        print(f"  - Validation users: {len(self.keyset_val)}")

    def _get_parameter_ranges(self):


        return {
            'recency': [1, 5, 10, 20, 25, 30, 40, 50],
            'asymmetry': [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1],
            'locality': [1, 5, 10, 25, 50, 100, 200, 500]
        }

    def evaluate_predictions(self, predictions):
        # Save predictions temporarily for evaluator
        temp_pred_path = f"../../predictions/{self.dataset_name}/racf/keyset0.json"
        os.makedirs(os.path.dirname(temp_pred_path), exist_ok=True)
        with open(temp_pred_path, 'w') as f:
            json.dump(predictions, f)

        self.evaluator.load_data()
        eval_df, avg_metrics = self.evaluator.evaluate()
        return eval_df, avg_metrics

    def run_single_experiment(self, params, topk=10):
        # Build matrices per params
        item_user_matrix = build_user_item_matrix(self.df_train)
        usersim = compute_user_similarity(item_user_matrix, alpha=params['asymmetry'])
        rc_matrix = compute_rc_matrix(self.df_train, self.user_count, self.item_count, recency=params['recency'])

        user_recommendations = usersim.power(params['locality']).dot(sparse.csr_matrix(rc_matrix))

        # Build predictions for validation users only
        predictions = {}
        for uid in self.keyset_val:
            predictions[uid] = topk_from_rec_matrix(user_recommendations, uid, topk)

        eval_df, metrics = self.evaluate_predictions(predictions)
        return eval_df, metrics

    def grid_search(self, metric='nDCG@5', max_combinations=None, seed=42):
        print(f"{datetime.now()}: Starting grid search for {self.dataset_name} (RACF)")
        print(f"Target metric: {metric}")
        print(f"Random seed: {seed}")
        print(f"Parameter ranges: {self.param_ranges}")

        np.random.seed(seed)
        random.seed(seed)

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

        for i, combination in enumerate(all_combinations):
            params = dict(zip(param_names, combination))
            _, metrics = self.run_single_experiment(params)
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

        results.sort(key=lambda x: x['score'], reverse=True)
        return results, best_params, best_score

    def print_results(self, results, best_params, best_score, metric='nDCG@5', top_n=3):
        print("\n" + "="*80)
        print(f"RACF HYPERPARAMETER TUNING RESULTS - {self.dataset_name.upper()}")
        print("="*80)
        print(f"Best {metric}: {best_score:.4f}")
        print("Best parameters:")
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
        results_file = f"../../hyperparameter_results/{self.dataset_name}/racf.json"
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"{datetime.now()}: Results saved to {results_file}")

    def save_best_params_to_config(self, best_params, best_score, metric='nDCG@5'):
        config_file = "best_params_config.json"
        config = {}
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
        config[self.dataset_name] = {
            'best_parameters': best_params,
            'best_score': best_score,
            'metric': metric,
            'timestamp': datetime.now().isoformat()
        }
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"{datetime.now()}: Best parameters saved to config: {config_file}")

    def tune(self, metric='nDCG@5', max_combinations=None, top_n=3, seed=42):
        print(f"{datetime.now()}: Starting RACF hyperparameter tuning")
        results, best_params, best_score = self.grid_search(metric, max_combinations, seed)
        self.print_results(results, best_params, best_score, metric, top_n)
        self.save_results(results, best_params, best_score, metric, seed)
        self.save_best_params_to_config(best_params, best_score, metric)
        return results, best_params, best_score


def main():
    parser = argparse.ArgumentParser(description='RACF Hyperparameter Tuning')
    parser.add_argument('dataset_name', help='Name of the dataset (e.g., dunnhumby, instacart)')
    parser.add_argument('--metric', '-m', default='nDCG@5',
                        choices=['HR@5', 'WHR@5', 'nDCG@5', 'recall@5'],
                        help='Metric to optimize (default: nDCG@5)')
    parser.add_argument('--max_combinations', '-max', type=int, default=20,
                        help='Maximum number of parameter combinations to test (default: 20)')
    parser.add_argument('--top_n', '-n', type=int, default=3,
                        help='Number of top results to display (default: 3)')
    parser.add_argument('--seed', '-s', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    tuner = RACFHyperparameterTuner(args.dataset_name)
    results, best_params, best_score = tuner.tune(
        metric=args.metric,
        max_combinations=args.max_combinations,
        top_n=args.top_n,
        seed=args.seed
    )
    print(f"\n{datetime.now()}: Hyperparameter tuning completed successfully!")


if __name__ == "__main__":
    main()


