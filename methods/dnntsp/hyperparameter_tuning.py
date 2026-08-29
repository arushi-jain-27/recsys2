#!/usr/bin/env python3
"""
Hyperparameter tuning for DNNTSP recommendation model
"""

import numpy as np
import json
import pandas as pd
import argparse
import os
import sys
import random
import torch
import torch.nn as nn
from itertools import product
from datetime import datetime
from tqdm import tqdm

# Import the DNNTSP functions
from dnntsp import *

# Import the existing evaluator
sys.path.append(os.path.join(os.path.dirname(__file__), '../../evaluations/'))
from evaluate_recommendation_model import RecommendationEvaluator


class DNNTSPHyperparameterTuner:
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
        self.items_total = self.keyset['item_num']
        self.keyset_train = self.keyset['train']
        self.keyset_val = self.keyset['val']
        self.keyset_test = self.keyset['test']
        
        print(f"{datetime.now()}: Dataset loaded successfully")
        print(f"  - Item count: {self.items_total}")
        print(f"  - Train users: {len(self.keyset_train)}")
        print(f"  - Validation users: {len(self.keyset_val)}")
        print(f"  - Test users: {len(self.keyset_test)}")
        
        # Initialize evaluator for validation set with correct paths
        self.evaluator = RecommendationEvaluator(self.dataset_name, "dnntsp", split='val')
        # Update paths to be relative to dnntsp directory
        self.evaluator.predictions_path = f"../../predictions/{self.dataset_name}/dnntsp/keyset0.json"
        self.evaluator.dataset_path = f"../../datasets/{self.dataset_name}/future.json"
        self.evaluator.keyset_path = f"../../datasets/{self.dataset_name}/keyset_0.json"

        
    def _get_parameter_ranges(self):
        """Define hyperparameter search space"""
        return {
            'item_embed_dim': [16, 32, 64, 128],
            'batch_size': [32, 64, 128],
            'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01],
            'epochs': [3],  # Use early stopping
            'loss_function': ['multi_label_soft_loss', 'weight_mse_loss', 'mse_loss', 'bpr_loss'],
            'optim': ['Adam'],
            'weight_decay': [0, 1e-5, 1e-4]
        }
        
    def evaluate_predictions(self, predictions):
        """Evaluate predictions using the existing RecommendationEvaluator"""
        # Save predictions temporarily
        temp_pred_path = f"../../predictions/{self.dataset_name}/dnntsp/keyset0.json"
        os.makedirs(os.path.dirname(temp_pred_path), exist_ok=True)
        
        with open(temp_pred_path, 'w') as f:
            json.dump(predictions, f)
        
        self.evaluator.load_data()
        eval_df, avg_metrics = self.evaluator.evaluate()
        return avg_metrics
    
    def run_single_experiment(self, params):
        """Run a single experiment with given parameters"""
        # Set device
        use_cuda = torch.cuda.is_available()
        
        # Create model
        model = create_model(self.items_total, params['item_embed_dim'])
        
        if use_cuda:
            model = model.cuda()
        
        # Create data loaders
        train_data_loader = get_data_loader(
            history_path=self.history_file,
            future_path=self.future_file,
            keyset_path=self.keyset_file,
            data_type='train',
            batch_size=params['batch_size'],
            item_embedding_matrix=model.item_embedding
        )
        
        val_data_loader = get_data_loader(
            history_path=self.history_file,
            future_path=self.future_file,
            keyset_path=self.keyset_file,
            data_type='val',
            batch_size=params['batch_size'],
            item_embedding_matrix=model.item_embedding
        )
        
        # Create loss function
        loss_func = create_loss(
            loss_type=params['loss_function'],
            history_path=self.history_file,
            keyset_path=self.keyset_file,
            items_total=self.items_total
        )
        
        # Create optimizer
        if params['optim'] == "Adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=params['learning_rate'],
                weight_decay=params['weight_decay']
            )
        elif params['optim'] == "SGD":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=params['learning_rate'],
                momentum=0.9,
                weight_decay=params['weight_decay']
            )
        else:
            raise NotImplementedError(f"Optimizer {params['optim']} not implemented")
        
        # Model folder
        model_folder = f"../../models/dnntsp/{self.dataset_name}/keyset_0"
        os.makedirs(model_folder, exist_ok=True)
        
        # Delete old checkpoints to avoid loading wrong model dimensions
        checkpoint_path = f"{model_folder}/dnntsp_best.pth"
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print(f"Deleted old checkpoint: {checkpoint_path}")
        
        # Train the model
        actual_epochs = train_model(
            model=model,
            future_path=self.future_file,
            train_data_loader=train_data_loader,
            val_data_loader=val_data_loader,
            val_key_set=self.keyset_val,
            loss_func=loss_func,
            epochs=params['epochs'],
            optimizer=optimizer,
            model_folder=model_folder,
            top_k=10,
            should_save_model=True
        )
        
        # Update params with actual epochs completed
        params['actual_epochs'] = actual_epochs
        
        # Load the best model if checkpoint exists, otherwise use model in memory
        checkpoint_path = f"{model_folder}/dnntsp_best.pth"
        if os.path.exists(checkpoint_path):
            model = load_model(model, f"{self.dataset_name}/keyset_0")
        else:
            print(f"No checkpoint found at {checkpoint_path}, using model in memory")
        
        # Generate predictions for validation set
        predictions = evaluate(
            model, 
            val_data_loader, 
            self.keyset_val, 
            top_k=10
        )
        
        # Evaluate predictions
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
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
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
            
            try:
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
                
            except Exception as e:
                print(f"{datetime.now()}: Error with parameters {params}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
                    
        
        # Sort results by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results, best_params, best_score
    
    def print_results(self, results, best_params, best_score, metric='nDCG@5', top_n=3):
        """Print hyperparameter tuning results"""
        print("\n" + "="*80)
        print(f"DNNTSP HYPERPARAMETER TUNING RESULTS - {self.dataset_name.upper()}")
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
        results_file = f"../../hyperparameter_results/{self.dataset_name}/dnntsp.json"
        
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
        print(f"{datetime.now()}: Starting DNNTSP hyperparameter tuning")
        
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
    parser = argparse.ArgumentParser(description='DNNTSP Hyperparameter Tuning')
    parser.add_argument('dataset_name', help='Name of the dataset (e.g., dunnhumby, instacart)')
    parser.add_argument('--metric', '-m', default='nDCG@5', 
                       choices=['HR@5', 'WHR@5', 'nDCG@5', 'recall@5'],
                       help='Metric to optimize (default: nDCG@5)')
    parser.add_argument('--max_combinations', '-max', type=int, default=10,
                       help='Maximum number of parameter combinations to test (default: 10)')
    parser.add_argument('--top_n', '-n', type=int, default=3,
                       help='Number of top results to display (default: 3)')
    parser.add_argument('--seed', '-s', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Initialize tuner
    tuner = DNNTSPHyperparameterTuner(args.dataset_name)
    
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

