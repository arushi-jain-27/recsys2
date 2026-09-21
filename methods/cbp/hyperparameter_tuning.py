#!/usr/bin/env python3
"""
Hyperparameter tuning for CBP recommendation model
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from itertools import product

import numpy as np
import torch
from tqdm import tqdm

from cbp import load_data, predict, train_model

sys.path.append("../../evaluations/")
from evaluate_recommendation_model import RecommendationEvaluator


class CBPHyperparameterTuner:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.load_data()
        self.param_ranges = {
            "dim": [16, 32, 64],
            "lr": [0.0001, 0.001, 0.01],
            "l2": [1e-5, 1e-4, 0.001, 0.01],
            "alpha": [0.1, 0.2, 0.3, 0.9],
            "beta": [0.2, 0.4, 0.6, 0.8],
            "decay": [0.4, 0.6, 0.8],
            "batch_size": [50, 100, 200],
            "n_epoch": [20],
        }

    def load_data(self):
        print(f"{datetime.now()}: Loading dataset files for {self.dataset_name}")
        self.data_history, self.ground_truth, self.keyset = load_data(self.dataset_name, 0)
        self.keyset_train = self.keyset["train"]
        self.keyset_val = self.keyset["val"]
        self.keyset_test = self.keyset["test"]

        print(f"{datetime.now()}: Dataset loaded successfully")
        print(f"  - Item count: {self.keyset['item_num']}")
        print(f"  - Train users: {len(self.keyset_train)}")
        print(f"  - Validation users: {len(self.keyset_val)}")
        print(f"  - Test users: {len(self.keyset_test)}")

        self.evaluator = RecommendationEvaluator(self.dataset_name, "cbp", split="val")
        self.evaluator.predictions_path = f"../../predictions/{self.dataset_name}/cbp/keyset0.json"
        self.evaluator.dataset_path = f"../../datasets/{self.dataset_name}/future.json"
        self.evaluator.keyset_path = f"../../datasets/{self.dataset_name}/keyset_0.json"

    def evaluate_predictions(self, predictions):
        temp_pred_path = f"../../predictions/{self.dataset_name}/cbp/keyset0.json"
        os.makedirs(os.path.dirname(temp_pred_path), exist_ok=True)
        with open(temp_pred_path, "w") as f:
            json.dump(predictions, f)
        self.evaluator.load_data()
        eval_df, avg_metrics = self.evaluator.evaluate()
        return avg_metrics

    def run_single_experiment(self, params):
        model, itemPOP = train_model(
            self.data_history, self.ground_truth, self.keyset,
            params["dim"], params["lr"], params["l2"],
            params["alpha"], params["beta"], params["decay"],
            params["batch_size"], params["n_epoch"],
        )
        predictions = predict(model, self.data_history, self.keyset_val, itemPOP,
                              batch_size=params["batch_size"])
        metrics = self.evaluate_predictions(predictions)
        del model, itemPOP, predictions
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return metrics

    def grid_search(self, metric="nDCG@5", max_combinations=None, seed=42):
        print(f"{datetime.now()}: Starting grid search for {self.dataset_name}")
        print(f"Target metric: {metric}")
        print(f"Random seed: {seed}")
        print(f"Parameter ranges: {self.param_ranges}")

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        param_names = list(self.param_ranges.keys())
        all_combinations = list(product(*self.param_ranges.values()))
        if max_combinations and len(all_combinations) > max_combinations:
            print(f"Limiting search to {max_combinations} random combinations from {len(all_combinations)} total")
            selected = np.random.choice(len(all_combinations), max_combinations, replace=False)
            all_combinations = [all_combinations[i] for i in selected]
        print(f"Total combinations to test: {len(all_combinations)}")

        results = []
        best_score = -1
        best_params = None
        for i, combination in enumerate(tqdm(all_combinations, desc="Testing combinations")):
            params = dict(zip(param_names, combination))
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            metrics = self.run_single_experiment(params)
            score = metrics[metric]
            results.append({
                "combination_id": i,
                "parameters": params.copy(),
                "metrics": metrics.copy(),
                "score": score,
            })
            if score > best_score:
                best_score = score
                best_params = params.copy()
            print(f"{datetime.now()}: parameters: {params} = score: {score}")

        results.sort(key=lambda x: x["score"], reverse=True)
        return results, best_params, best_score

    def print_results(self, results, best_params, best_score, metric="nDCG@5", top_n=3):
        print("\n" + "=" * 80)
        print(f"CBP HYPERPARAMETER TUNING RESULTS - {self.dataset_name.upper()}")
        print("=" * 80)
        print(f"Best {metric}: {best_score:.4f}")
        print("Best parameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print(f"\nTop {top_n} results:")
        print("-" * 80)
        for i, result in enumerate(results[:top_n]):
            print(f"Rank {i+1}: {metric} = {result['score']:.4f}")
            for param, value in result["parameters"].items():
                print(f"  {param}: {value}")
            print()

    def save_results(self, results, best_params, best_score, metric="nDCG@5", seed=42):
        results_data = {
            "dataset": self.dataset_name,
            "best_metric": metric,
            "best_score": best_score,
            "best_parameters": best_params,
            "all_results": results,
            "random_seed": seed,
            "timestamp": datetime.now().isoformat(),
        }
        os.makedirs(f"../../hyperparameter_results/{self.dataset_name}", exist_ok=True)
        results_file = f"../../hyperparameter_results/{self.dataset_name}/cbp.json"
        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)
        print(f"{datetime.now()}: Results saved to {results_file}")

    def save_best_params_to_config(self, best_params, best_score, metric="nDCG@5"):
        config_file = "best_params_config.json"
        config = {}
        if os.path.exists(config_file):
            with open(config_file) as f:
                config = json.load(f)
        config[self.dataset_name] = {
            "best_parameters": best_params,
            "best_score": best_score,
            "metric": metric,
            "timestamp": datetime.now().isoformat(),
        }
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        print(f"{datetime.now()}: Best parameters saved to config: {config_file}")

    def tune(self, metric="nDCG@5", max_combinations=None, top_n=3, seed=42):
        print(f"{datetime.now()}: Starting CBP hyperparameter tuning")
        results, best_params, best_score = self.grid_search(metric, max_combinations, seed)
        self.print_results(results, best_params, best_score, metric, top_n)
        self.save_results(results, best_params, best_score, metric, seed)
        self.save_best_params_to_config(best_params, best_score, metric)
        return results, best_params, best_score


def main():
    parser = argparse.ArgumentParser(description="CBP Hyperparameter Tuning")
    parser.add_argument("dataset_name")
    parser.add_argument("--metric", "-m", default="nDCG@5",
                        choices=["HR@5", "WHR@5", "nDCG@5", "recall@5"])
    parser.add_argument("--max_combinations", "-max", type=int, default=20)
    parser.add_argument("--top_n", "-n", type=int, default=3)
    parser.add_argument("--seed", "-s", type=int, default=42)
    args = parser.parse_args()

    tuner = CBPHyperparameterTuner(args.dataset_name)
    tuner.tune(metric=args.metric, max_combinations=args.max_combinations,
               top_n=args.top_n, seed=args.seed)
    print(f"\n{datetime.now()}: Hyperparameter tuning completed successfully!")


if __name__ == "__main__":
    main()
