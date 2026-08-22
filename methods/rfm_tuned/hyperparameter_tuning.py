#!/usr/bin/env python3
"""
Hyperparameter tuning for RFM recommendation model
"""

import argparse
import json
import os
from datetime import datetime
import random

import numpy as np

# Import RFM building blocks to avoid duplication
from rfm_new import (
    load_data,
    compute_rfm_scores,
    compute_global_item_scores,
)

# Import the existing evaluator
import sys
sys.path.append('../../evaluations/')
from evaluate_recommendation_model import RecommendationEvaluator


class RFMHyperparameterTuner:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

        # Data paths
        self.history_file = f"../../datasets/{dataset_name}/history.json"
        self.prices_file = f"../../datasets/{dataset_name}/prices.json"
        self.dates_file = f"../../datasets/{dataset_name}/dates.json"
        self.future_file = f"../../datasets/{dataset_name}/future.json"
        self.keyset_file = f"../../datasets/{dataset_name}/keyset_0.json"

        self.load_data()
        self.param_combinations = self._generate_parameter_combinations()

    def load_data(self):
        print(f"{datetime.now()}: Loading dataset files for {self.dataset_name}")
        
        # Load history, prices, and dates
        self.data_history, self.data_prices, self.data_dates = load_data(
            self.history_file, self.prices_file, self.dates_file
        )
        
        with open(self.future_file, 'r') as f:
            self.ground_truth = json.load(f)
        with open(self.keyset_file, 'r') as f:
            self.keyset = json.load(f)

        # Useful keysets
        self.keyset_train = self.keyset['train']
        self.keyset_val = self.keyset['val']

        # Compute global item scores once (for tie-breaking)
        self.global_scores = compute_global_item_scores(self.data_history)

        # Evaluator setup
        self.evaluator = RecommendationEvaluator(self.dataset_name, "rfm")
        self.evaluator.predictions_path = f"../../predictions/{self.dataset_name}/rfm/keyset0.json"
        self.evaluator.dataset_path = f"../../datasets/{self.dataset_name}/future.json"

        print(f"{datetime.now()}: Dataset loaded successfully")
        print(f"  - Total users: {len(self.data_history)}")
        print(f"  - Train users: {len(self.keyset_train)}")
        print(f"  - Validation users: {len(self.keyset_val)}")

    def _generate_parameter_combinations(self):
        """
        Generate parameter combinations for alpha, beta, gamma that sum to 1.
        Uses a grid approach with step size 0.1.
        """
        step = 0.1
        values = [round(i * step, 1) for i in range(11)]  # [0.0, 0.1, ..., 1.0]
        
        combinations = []
        for alpha in values:
            for beta in values:
                gamma = round(1.0 - alpha - beta, 1)
                # Check if gamma is valid (non-negative and in our value range)
                if gamma >= 0 and gamma <= 1.0 and abs(alpha + beta + gamma - 1.0) < 0.01:
                    combinations.append({
                        'alpha': alpha,
                        'beta': beta,
                        'gamma': gamma
                    })
        
        print(f"Generated {len(combinations)} valid parameter combinations")
        return combinations

    def evaluate_predictions(self, predictions):
        """Evaluate predictions using the standard evaluator."""
        # Save predictions temporarily for evaluator
        temp_pred_path = f"../../predictions/{self.dataset_name}/rfm/keyset0.json"
        os.makedirs(os.path.dirname(temp_pred_path), exist_ok=True)
        with open(temp_pred_path, 'w') as f:
            json.dump(predictions, f)

        self.evaluator.load_data()
        eval_df, avg_metrics = self.evaluator.evaluate()
        return eval_df, avg_metrics

    def compute_rfm_scores_with_params(self, user_id, baskets, prices, dates, alpha, beta, gamma):
        """
        Modified version of compute_rfm_scores that accepts custom alpha, beta, gamma.
        """
        from collections import defaultdict
        
        # Ignore padding baskets [-1] at start and end
        core_baskets = baskets[1:-1] if len(baskets) > 2 else []
        core_prices = prices[1:-1] if len(prices) > 2 else []
        core_dates = dates[1:-1] if len(dates) > 2 else []
        
        if not core_baskets:
            return {}
        
        # Track metrics per item
        item_recency = {}
        item_frequency = defaultdict(int)
        item_revenue = defaultdict(float)
        
        for basket_idx, (basket, price_list, date_list) in enumerate(zip(core_baskets, core_prices, core_dates)):
            for item, price, date in zip(basket, price_list, date_list):
                if item == -1:
                    continue
                if item not in item_recency or date > item_recency[item]:
                    item_recency[item] = date
                item_frequency[item] += 1
                item_revenue[item] += price
        
        if not item_recency:
            return {}
        
        # Normalize metrics within user
        def min_max_normalize(values_dict):
            if not values_dict:
                return {}
            min_val = min(values_dict.values())
            max_val = max(values_dict.values())
            if min_val == max_val:
                return {k: 1.0 for k in values_dict.keys()}
            return {k: (v - min_val) / (max_val - min_val) for k, v in values_dict.items()}
        
        # Get total revenue per user for revenue share calculation
        total_revenue = sum(item_revenue.values())
        item_revenue_share = {item: revenue / total_revenue if total_revenue > 0 else 0.0 
                              for item, revenue in item_revenue.items()}
        
        # Normalize each metric
        normalized_recency = min_max_normalize(item_recency)
        normalized_frequency = min_max_normalize(dict(item_frequency))
        normalized_revenue = min_max_normalize(item_revenue_share)
        
        # Compute RFM score with custom weights
        rfm_scores = {}
        for item in item_recency.keys():
            rfm_scores[item] = (alpha * normalized_recency[item] + 
                               beta * normalized_frequency[item] + 
                               gamma * normalized_revenue[item])
        
        return rfm_scores

    def run_single_experiment(self, params, topk=10):
        """Run a single experiment with given parameters."""
        alpha = params['alpha']
        beta = params['beta']
        gamma = params['gamma']
        
        # Generate predictions for validation users
        predictions = {}
        for user_id in self.keyset_val:
            baskets = self.data_history.get(user_id, [])
            prices = self.data_prices.get(user_id, [])
            dates = self.data_dates.get(user_id, [])
            
            # Compute RFM scores with custom parameters
            rfm_scores = self.compute_rfm_scores_with_params(
                user_id, baskets, prices, dates, alpha, beta, gamma
            )
            
            if not rfm_scores:
                predictions[user_id] = []
                continue
            
            # Sort by RFM score (desc), then global popularity (desc), then item_id (asc)
            ranked_items = sorted(
                rfm_scores.items(),
                key=lambda x: (-x[1], -self.global_scores.get(x[0], 0), x[0])
            )
            
            predictions[user_id] = [item for item, _ in ranked_items[:topk]]
        
        eval_df, metrics = self.evaluate_predictions(predictions)
        return eval_df, metrics

    def grid_search(self, metric='nDCG@5', max_combinations=None, seed=42):
        """Perform grid search over parameter combinations."""
        print(f"{datetime.now()}: Starting grid search for {self.dataset_name} (RFM)")
        print(f"Target metric: {metric}")
        print(f"Random seed: {seed}")
        print(f"Total parameter combinations: {len(self.param_combinations)}")

        np.random.seed(seed)
        random.seed(seed)

        combinations_to_test = self.param_combinations
        
        if max_combinations and len(combinations_to_test) > max_combinations:
            print(f"Limiting search to {max_combinations} random combinations from {len(combinations_to_test)} total")
            selected_indices = np.random.choice(len(combinations_to_test), max_combinations, replace=False)
            combinations_to_test = [combinations_to_test[i] for i in selected_indices]

        print(f"Total combinations to test: {len(combinations_to_test)}")

        results = []
        best_score = -1
        best_params = None

        for i, params in enumerate(combinations_to_test):
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

            print(f"{datetime.now()}: [{i+1}/{len(combinations_to_test)}] alpha={params['alpha']:.1f}, beta={params['beta']:.1f}, gamma={params['gamma']:.1f} => {metric}={score:.4f}")

        results.sort(key=lambda x: x['score'], reverse=True)
        return results, best_params, best_score

    def print_results(self, results, best_params, best_score, metric='nDCG@5', top_n=3):
        """Print the tuning results."""
        print("\n" + "="*80)
        print(f"RFM HYPERPARAMETER TUNING RESULTS - {self.dataset_name.upper()}")
        print("="*80)
        print(f"Best {metric}: {best_score:.4f}")
        print("Best parameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print(f"  Sum: {sum(best_params.values()):.4f}")
        print(f"\nTop {top_n} results:")
        print("-" * 80)
        for i, result in enumerate(results[:top_n]):
            print(f"Rank {i+1}: {metric} = {result['score']:.4f}")
            for param, value in result['parameters'].items():
                print(f"  {param}: {value}")
            print()

    def save_results(self, results, best_params, best_score, metric='nDCG@5', seed=42):
        """Save tuning results to file."""
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
        results_file = f"../../hyperparameter_results/{self.dataset_name}/rfm.json"
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"{datetime.now()}: Results saved to {results_file}")

    def save_best_params_to_config(self, best_params, best_score, metric='nDCG@5'):
        """Save best parameters to a config file."""
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
        """Run the complete hyperparameter tuning process."""
        print(f"{datetime.now()}: Starting RFM hyperparameter tuning")
        results, best_params, best_score = self.grid_search(metric, max_combinations, seed)
        self.print_results(results, best_params, best_score, metric, top_n)
        self.save_results(results, best_params, best_score, metric, seed)
        self.save_best_params_to_config(best_params, best_score, metric)
        return results, best_params, best_score


def main():
    parser = argparse.ArgumentParser(description='RFM Hyperparameter Tuning')
    parser.add_argument('dataset_name', help='Name of the dataset (e.g., dunnhumby, instacart)')
    parser.add_argument('--metric', '-m', default='nDCG@5',
                        choices=['HR@5', 'WHR@5', 'nDCG@5', 'recall@5'],
                        help='Metric to optimize (default: nDCG@5)')
    parser.add_argument('--max_combinations', '-max', type=int, default=None,
                        help='Maximum number of parameter combinations to test (default: None, tests all)')
    parser.add_argument('--top_n', '-n', type=int, default=5,
                        help='Number of top results to display (default: 5)')
    parser.add_argument('--seed', '-s', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    tuner = RFMHyperparameterTuner(args.dataset_name)
    results, best_params, best_score = tuner.tune(
        metric=args.metric,
        max_combinations=args.max_combinations,
        top_n=args.top_n,
        seed=args.seed
    )
    print(f"\n{datetime.now()}: Hyperparameter tuning completed successfully!")


if __name__ == "__main__":
    main()

