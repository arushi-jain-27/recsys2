#!/usr/bin/env python3
"""
Hyperparameter tuning for Sets2Sets recommendation model
"""

import numpy as np
import json
import pandas as pd
import argparse
import os
import sys
import random
import gc
import time
import torch
import torch.nn as nn
from itertools import product
from datetime import datetime
from tqdm import tqdm

# Import the Sets2Sets functions
from sets2sets import *

# Import the existing evaluator
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../evaluations/'))
from evaluate_recommendation_model import RecommendationEvaluator


class Sets2SetsHyperparameterTuner:
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


        # Calculate weights for loss function
        self.weights = np.zeros(self.input_size)
        # not needed since we restrict to 50 baskets
        # for key, value in truncated_data_history.items():
        #     if len(value[1:-1]) > params['max_length']:
        #         truncated_data_history[key] = [-1] + value[-params['max_length']-1:-1] + [-1]
        codes_freq = get_codes_frequency_no_vector(self.data_history, self.input_size, self.ground_truth.keys())
        max_freq = max(codes_freq) if max(codes_freq) > 0 else 1
        for idx in range(len(codes_freq)):
            if codes_freq[idx] > 0:
                self.weights[idx] = max_freq / codes_freq[idx]
            else:
                self.weights[idx] = 0
        
        # Initialize evaluator for validation set with correct paths
        self.evaluator = RecommendationEvaluator(self.dataset_name, "sets2sets", split='val')
        # Update paths to be relative to sets2sets directory
        self.evaluator.predictions_path = f"../../predictions/{self.dataset_name}/sets2sets/keyset0.json"
        self.evaluator.dataset_path = f"../../datasets/{self.dataset_name}/future.json"
        self.evaluator.keyset_path = f"../../datasets/{self.dataset_name}/keyset_0.json"

        
    def _get_parameter_ranges(self):
        """Define hyperparameter search space"""
        return {
            'hidden_size': [16, 32, 64],
            'num_layers': [1, 2, 3],
            'learning_rate': [0.0001,0.0005, 0.001, 0.005, 0.01],
            'num_iter': [3], # implement early stopping based on validation loss
            'labmda': [5, 10, 20],
            'max_length': [50, 80, 100],
            'dropout_p': [0, 0.1, 0.2],
            'use_embedding': [1],  # Fixed value
            'use_linear_reduction': [0]  # Fixed value
        }
        
    def evaluate_predictions(self, predictions):
        """Evaluate predictions using the existing RecommendationEvaluator"""
        # Save predictions temporarily
        temp_pred_path = f"../../predictions/{self.dataset_name}/sets2sets/keyset0.json"
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
        
        # Create models with current parameters
        encoder = EncoderRNN_new(
            self.input_size, 
            params['hidden_size'], 
            params['num_layers'], 
            use_embedding=params['use_embedding'], 
            use_linear_reduction=params['use_linear_reduction'], 
            use_cuda=use_cuda
        )
        decoder = AttnDecoderRNN_new(
            params['hidden_size'], 
            self.input_size, 
            params['num_layers'], 
            dropout_p=params['dropout_p'],
            max_length=params['max_length'],
            use_embedding=params['use_embedding'], 
            use_linear_reduction=params['use_linear_reduction'], 
            use_cuda=use_cuda
        )
        
        if use_cuda:
            encoder = encoder.cuda()
            decoder = decoder.cuda()
        

        
        # Train the model
        model_name = f"{self.dataset_name}/keyset_0"
        actual_epochs = train_model(
            self.data_history, 
            self.ground_truth, 
            self.input_size, 
            encoder, 
            decoder, 
            model_name, 
            self.keyset_train, 
            self.keyset_val, 
            self.weights, 
            1,  # next_k_step
            params['num_iter'], 
            10,  # top_k
            params['learning_rate'],
            params['max_length'],
            should_save_model=True,
            labmda=params['labmda']
        )
        
        # Update params with actual epochs completed
        params['actual_epochs'] = actual_epochs

        # load the best model
        encoder, decoder = self.load_model(encoder, decoder, model_name)
        
        # Generate predictions for validation set
        predictions = evaluate(
            self.data_history, 
            self.ground_truth, 
            encoder, 
            decoder, 
            self.input_size, 
            self.keyset_val, 
            1,  # next_k_step
            10,  # activate_codes_num
            params['max_length']
        )
        
        # Evaluate predictions
        metrics = self.evaluate_predictions(predictions)
        
        # Aggressive memory cleanup after each experiment
        del encoder, decoder, predictions
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force Python garbage collection
        gc.collect()
        
        return metrics
    
    def load_model(self, encoder, decoder, model_name, epoch=None, device=None):
        """Load model with consistent naming"""
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        directory = f'../../models/sets2sets/{model_name}/'
        
        if epoch is not None:
            encoder_path = f'{directory}encoder_epoch_{epoch}.pth'
            decoder_path = f'{directory}decoder_epoch_{epoch}.pth'
        else:
            encoder_path = f'{directory}encoder_best.pth'
            decoder_path = f'{directory}decoder_best.pth'
        
        # Load the model state dictionaries
        encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        decoder.load_state_dict(torch.load(decoder_path, map_location=device))
        
        return encoder, decoder
    
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
                
                # Additional cleanup between combinations
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Small delay to allow OS to release memory
                time.sleep(0.5)
                
            except Exception as e:
                print(f"{datetime.now()}: Error with parameters {params}: {str(e)}")
                # Cleanup even on error
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
                    
        
        # Sort results by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results, best_params, best_score
    
    def print_results(self, results, best_params, best_score, metric='nDCG@5', top_n=3):
        """Print hyperparameter tuning results"""
        print("\n" + "="*80)
        print(f"SETS2SETS HYPERPARAMETER TUNING RESULTS - {self.dataset_name.upper()}")
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
        results_file = f"../../hyperparameter_results/{self.dataset_name}/sets2sets.json"
        
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
        print(f"{datetime.now()}: Starting Sets2Sets hyperparameter tuning")
        
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
    parser = argparse.ArgumentParser(description='Sets2Sets Hyperparameter Tuning')
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
    
    # Initialize tuner
    tuner = Sets2SetsHyperparameterTuner(args.dataset_name)
    
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
