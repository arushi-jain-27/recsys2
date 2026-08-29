#!/usr/bin/env python3
"""
Hyperparameter tuning for triple2vec recommendation model
Mirrors the structure and workflow of the Sets2Sets tuner.
"""

import os
import sys
import json
import random
from tqdm import tqdm
from itertools import product
from datetime import datetime

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Import triple2vec train/eval entrypoints defined in main.py (same directory)
from main import run_embedding, run_recommendation

# Import the existing evaluator used across methods
sys.path.append(os.path.join(os.path.dirname(__file__), '../../evaluations/'))
from evaluate_recommendation_model import RecommendationEvaluator


class Triple2VecHyperparameterTuner:
    def __init__(self, dataset_name, fold_id=0):
        self.dataset_name = dataset_name
        self.fold_id = fold_id

        # Dataset files (for sanity checks and potential aux usage)
        self.history_file = f"../../datasets/{dataset_name}/history.json"
        self.future_file = f"../../datasets/{dataset_name}/future.json"
        self.keyset_file = f"../../datasets/{dataset_name}/keyset_{fold_id}.json"

        self._load_data()
        self.param_ranges = self._get_parameter_ranges()

        # Evaluator pointing to triple2vec prediction outputs
        self.evaluator = RecommendationEvaluator(self.dataset_name, "triple2vec", split='val')
        self.evaluator.predictions_path = (
            f"../../predictions/{self.dataset_name}/triple2vec/keyset{self.fold_id}.json"
        )
        self.evaluator.dataset_path = f"../../datasets/{self.dataset_name}/future.json"
        self.evaluator.keyset_path = f"../../datasets/{self.dataset_name}/keyset_{self.fold_id}.json"

    def _load_data(self):
        """Lightweight dataset load to extract key meta info and validate paths."""
        print(f"{datetime.now()}: Loading dataset files for {self.dataset_name}")

        with open(self.history_file, 'r') as f:
            self.data_history = json.load(f)
        with open(self.keyset_file, 'r') as f:
            self.keyset = json.load(f)
        with open(self.future_file, 'r') as f:
            self.ground_truth = json.load(f)

        self.input_size = self.keyset['item_num']
        self.keyset_train = self.keyset['train']
        self.keyset_val = self.keyset['val']
        self.keyset_test = self.keyset['test']

        print(f"{datetime.now()}: Dataset loaded successfully")
        print(f"  - Item count: {self.input_size}")
        print(f"  - Train users: {len(self.keyset_train)}")
        print(f"  - Validation users: {len(self.keyset_val)}")
        print(f"  - Test users: {len(self.keyset_test)}")

    def _get_parameter_ranges(self):
        """Define hyperparameter search space for triple2vec."""
        return {
            'dim': [64, 128, 256],
            'lr': [0.0001, 0.0005, 0.001, 0.005, 0.01],
            'batch_size': [1024, 2048, 4096],
            'n_neg': [5, 10, 25, 50],
            'optimizer': ['adam'],
            'max_epoch': [200],
            'n_sample': [1000000, 5000000, 10000000],
            'n_sample_per_epoch': [1000000],
            'patience': [5],
            'l0': [-1, 0.3, 0.5, 0.7, 1.0],
            'ensemble': [False, True],
            'top_k': [10],  # evaluation/export cutoff
        }

    def _evaluate_current_predictions(self):
        """Evaluate the validation-only predictions written by triple2vec for this keyset."""
        self.evaluator.load_data()
        eval_df, avg_metrics = self.evaluator.evaluate()
        return eval_df, avg_metrics

    def run_single_experiment(self, params):
        """Train triple2vec with given params, export predictions, and evaluate validation metrics."""
        # Device/seed control

        # Train embeddings
        run_embedding(
            DATA_NAME=self.dataset_name,
            dim=params['dim'],
            lr=params['lr'],
            batch_size=params['batch_size'],
            n_neg=params['n_neg'],
            fold_id=self.fold_id,
            optimizer=params['optimizer'],
            max_epoch=params['max_epoch'],
            n_sample=params['n_sample'],
            n_sample_per_epoch=params['n_sample_per_epoch'],
            patience=params['patience'],
        )

        # Generate predictions (overwrites the same keyset file for sequential runs)
        l0 = params['l0']
        if isinstance(l0, (int, float)) and l0 < 0:
            l0 = None
        run_recommendation(
            data_name=self.dataset_name,
            dim=params['dim'],
            lr=params['lr'],
            batch_size=params['batch_size'],
            n_neg=params['n_neg'],
            l0=l0,
            fold_id=self.fold_id,
            top_k=params['top_k'],
            ensemble=params['ensemble'],
            export_flags=('validation',),
            run_ranking_eval=False,
        )

        # Evaluate predictions
        _, avg_metrics = self._evaluate_current_predictions()
        return avg_metrics

    def grid_search(self, metric='nDCG@5', max_combinations=None, seed=42):
        """Perform grid search over the hyperparameter space."""
        print(f"{datetime.now()}: Starting grid search for {self.dataset_name} (triple2vec)")
        print(f"Target metric: {metric}")
        print(f"Random seed: {seed}")
        print(f"Parameter ranges: {self.param_ranges}")

        # Reproducibility
        np.random.seed(seed)
        random.seed(seed)
        tf.set_random_seed(seed)

        # Build parameter combinations
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

        for i, combo in tqdm(enumerate(all_combinations), total=len(all_combinations), desc="Testing combinations"):
            params = dict(zip(param_names, combo))
            try:
                metrics = self.run_single_experiment(params)
                score = metrics.get(metric, float('-inf'))
                results.append({
                    'combination_id': i,
                    'parameters': params.copy(),
                    'metrics': metrics.copy(),
                    'score': score,
                })
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                print(f"{datetime.now()}: parameters: {params} = score: {score}")
            except Exception as e:
                print(f"{datetime.now()}: Error with parameters {params}: {str(e)}")
                continue

        # Sort by score desc
        results.sort(key=lambda x: x['score'], reverse=True)
        return results, best_params, best_score

    def print_results(self, results, best_params, best_score, metric='nDCG@5', top_n=3):
        print("\n" + "="*80)
        print(f"TRIPLE2VEC HYPERPARAMETER TUNING RESULTS - {self.dataset_name.upper()}")
        print("="*80)
        print(f"Best {metric}: {best_score:.4f}")
        print("Best parameters:")
        for p, v in best_params.items():
            print(f"  {p}: {v}")
        print(f"\nTop {top_n} results:")
        print("-" * 80)
        for i, r in enumerate(results[:top_n]):
            print(f"Rank {i+1}: {metric} = {r['score']:.4f}")
            for p, v in r['parameters'].items():
                print(f"  {p}: {v}")
            print()

    def save_results(self, results, best_params, best_score, metric='nDCG@5', seed=42):
        payload = {
            'dataset': self.dataset_name,
            'best_metric': metric,
            'best_score': best_score,
            'best_parameters': best_params,
            'all_results': results,
            'random_seed': seed,
            'timestamp': datetime.now().isoformat(),
        }
        os.makedirs(f"../../hyperparameter_results/{self.dataset_name}", exist_ok=True)
        out_file = f"../../hyperparameter_results/{self.dataset_name}/triple2vec.json"
        with open(out_file, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"{datetime.now()}: Results saved to {out_file}")

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
            'timestamp': datetime.now().isoformat(),
        }
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"{datetime.now()}: Best parameters saved to config: {config_file}")

    def tune(self, metric='nDCG@5', max_combinations=None, top_n=3, seed=42):
        print(f"{datetime.now()}: Starting triple2vec hyperparameter tuning")
        results, best_params, best_score = self.grid_search(metric, max_combinations, seed)
        self.print_results(results, best_params, best_score, metric, top_n)
        self.save_results(results, best_params, best_score, metric, seed)
        self.save_best_params_to_config(best_params, best_score, metric)
        return results, best_params, best_score


def main():
    import argparse
    parser = argparse.ArgumentParser(description='triple2vec Hyperparameter Tuning')
    parser.add_argument('dataset_name', help='Name of the dataset (e.g., dunnhumby, instacart)')
    parser.add_argument('--fold_id', type=int, default=0, help='Keyset/fold index (default: 0)')
    parser.add_argument('--metric', '-m', default='nDCG@5',
                        choices=['HR@5', 'WHR@5', 'nDCG@5', 'recall@5', 'precision@5',
                                 'HR@10', 'WHR@10', 'nDCG@10', 'recall@10', 'precision@10'],
                        help='Metric to optimize (default: nDCG@5)')
    parser.add_argument('--max_combinations', '-max', type=int, default=20,
                        help='Maximum number of parameter combinations to test (default: 10)')
    parser.add_argument('--top_n', '-n', type=int, default=3,
                        help='Number of top results to display (default: 3)')
    parser.add_argument('--seed', '-s', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    tuner = Triple2VecHyperparameterTuner(args.dataset_name, fold_id=args.fold_id)
    results, best_params, best_score = tuner.tune(
        metric=args.metric,
        max_combinations=args.max_combinations,
        top_n=args.top_n,
        seed=args.seed,
    )
    print(f"\n{datetime.now()}: Hyperparameter tuning completed successfully!")


if __name__ == "__main__":
    main()


