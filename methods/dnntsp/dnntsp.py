from datetime import datetime
import json
import argparse
import os
import shutil
import sys
import warnings
import random

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np

# Add the dnntsp directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../evaluations/'))

from utils.data_container import get_data_loader
from utils.loss import BPRLoss, WeightMSELoss
from utils.util import convert_to_gpu, convert_all_data_to_gpu

from model.temporal_set_prediction import temporal_set_prediction
from evaluate_recommendation_model import RecommendationEvaluator


def create_model(items_total, item_embedding_dim):
    """Create the DNNTSP model"""
    model = temporal_set_prediction(items_total=items_total,
                                    item_embedding_dim=item_embedding_dim)
    return model


def get_class_weights_from_history(history_path, keyset_path, items_total):
    """
    Calculate class weights from training users' history data
    Similar to get_class_weights but uses history.json and filters by training keyset
    """
    with open(history_path, 'r') as f:
        history_data = json.load(f)
    with open(keyset_path, 'r') as f:
        keyset = json.load(f)
    
    train_user_ids = keyset.get('train', [])
    
    item_frequency = torch.ones(items_total)
    num_baskets = 0
    
    # Only use training users' history baskets
    for user_id in train_user_ids:
        user_baskets = history_data[user_id][1:-1]  # Skip [-1] markers
        for basket in user_baskets:
            num_baskets += 1
            for item in basket:
                item_frequency[item] += 1
    
    item_frequency /= num_baskets
    max_item_frequency = torch.ones(items_total) * torch.max(item_frequency)
    weights = max_item_frequency / item_frequency
    weights = weights / torch.max(weights)
    
    return weights


def create_loss(loss_type, history_path=None, keyset_path=None, items_total=None):
    """Create loss function"""
    if loss_type == 'bpr_loss':
        loss_func = BPRLoss()
    elif loss_type == 'mse_loss':
        loss_func = WeightMSELoss()
    elif loss_type == 'weight_mse_loss':
        if history_path is None or keyset_path is None or items_total is None:
            raise ValueError("history_path, keyset_path, and items_total required for weight_mse_loss")
        weights = get_class_weights_from_history(history_path, keyset_path, items_total)
        loss_func = WeightMSELoss(weights=weights)
    elif loss_type == "multi_label_soft_loss":
        loss_func = nn.MultiLabelSoftMarginLoss(reduction="mean")
    else:
        raise ValueError("Unknown loss function.")
    return loss_func


def train_model(model: nn.Module,
                future_path: str,
                train_data_loader: DataLoader,
                val_data_loader: DataLoader,
                val_key_set: list,
                loss_func,
                epochs,
                optimizer,
                model_folder,
                top_k=10,
                should_save_model=True):
    """
    Train the DNNTSP model with early stopping and learning rate scheduling
    """
    warnings.filterwarnings('ignore')

    print(model)
    print(optimizer)

    model = convert_to_gpu(model)
    loss_func = convert_to_gpu(loss_func)

    start_time = datetime.now()
    
    best_ndcg = 0.0
    patience = 5  # Early stopping patience
    patience_counter = 0
    best_epoch = 0
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Load future data for evaluation
    with open(future_path, 'r') as f:
        data_future = json.load(f)

    # Train n_iters epochs with early stopping
    for epoch in range(epochs):
        model.train()
        print_loss_total = 0.0
        
        # Train on training data
        for step, (g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency) in enumerate(train_data_loader):
            g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency = \
                convert_all_data_to_gpu(g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency)

            # Forward pass
            output = model(g, nodes_feature, edges_weight, lengths, nodes, users_frequency)
            loss = loss_func(output, truth_data.float())
            print_loss_total += loss.cpu().data.numpy()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        
            optimizer.step()

        # Print loss
        num_batches = max(1, len(train_data_loader))
        print_loss_avg = print_loss_total / num_batches
        print(f'{datetime.now()}: Epoch {epoch} Train Loss: {print_loss_avg:.4f}')
        sys.stdout.flush()

        # Evaluate on validation set for early stopping (similar to sets2sets)
        val_predictions = evaluate(model, val_data_loader, val_key_set, top_k)
        
        # Calculate validation nDCG@5 using RecommendationEvaluator (similar to sets2sets)
        val_ndcg = 0.0
        if val_predictions:
            # Create temporary evaluator
            temp_evaluator = RecommendationEvaluator("temp", "temp", split='val')
            temp_evaluator.predictions = val_predictions
            temp_evaluator.ground_truth = {k: data_future[k] for k in val_key_set if k in data_future}
            
            # Calculate nDCG@5
            eval_df, avg_metrics = temp_evaluator.evaluate()
            val_ndcg = avg_metrics.get('nDCG@5', 0.0)
        
        print(f'{datetime.now()}: Epoch {epoch} Validation nDCG@5: {val_ndcg:.4f}')
        
        # Learning rate scheduling
        scheduler.step(val_ndcg)
        
        # Early stopping logic (similar to sets2sets)
        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            patience_counter = 0
            if should_save_model:
                save_model(model, f"{model_folder}/dnntsp_best.pth")
                print('New best model saved.')
            best_epoch = epoch
        else:
            patience_counter += 1
            print(f'No improvement for {patience_counter} epochs')
        
        if patience_counter >= patience:
            print(f'{datetime.now()}: Early stopping at epoch {epoch} (patience={patience})')
            break
     
        
        print(f'{datetime.now()}: Finish epoch: {epoch}')

    end_time = datetime.now()
    print("cost %d seconds" % (end_time - start_time).seconds)
    
    return best_epoch + 1  # Return actual best epoch


def evaluate(model, test_data_loader, user_keys, top_k=10):
    """
    Generate predictions for test users and return as dictionary
    """
    predictions = {}
    model.eval()
    user_idx = 0
    
    with torch.no_grad():
        for step, (g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency) in enumerate(test_data_loader):
            g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency = \
                convert_all_data_to_gpu(g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency)

            # Get predictions
            output = model(g, nodes_feature, edges_weight, lengths, nodes, users_frequency)
            
            # Get top-k items for each user in the batch
            batch_size_current = output.shape[0]
            for i in range(batch_size_current):
                user_id = user_keys[user_idx]
                # Get top-k predicted items
                _, top_indices = output[i].topk(k=top_k)
                predicted_items = top_indices.cpu().numpy().tolist()
                predictions[user_id] = predicted_items
                user_idx += 1
    
    return predictions


def load_model(model, model_name, device='cpu'):
    """
    Load model with consistent naming (similar to sets2sets)
    """
    directory = f'../../models/dnntsp/{model_name}/'
    model_path = f'{directory}dnntsp_best.pth'
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model file not found at {model_path}")
    
    return model


def save_model(model, model_path):
    """
    Save model with consistent naming (similar to sets2sets)
    """
    directory = os.path.dirname(model_path)
    os.makedirs(directory, exist_ok=True)
    print(f"Saving model to {model_path}")
    torch.save(model.state_dict(), model_path)


def load_data(dataset_name, keyset_index=0):
    """
    Load dataset files (wrapper around get_data_loader)
    Returns paths needed for data loading
    """
    history_file = f"../../datasets/{dataset_name}/history.json"
    future_file = f"../../datasets/{dataset_name}/future.json"
    keyset_file = f"../../datasets/{dataset_name}/keyset_{keyset_index}.json"
 
    # Load keyset to get item_num
    with open(keyset_file, 'r') as f:
        keyset = json.load(f)
    
    return history_file, future_file, keyset_file, keyset


def save_predictions(predictions, dataset_name, keyset_index):
    """
    Save predictions in consistent format (similar to sets2sets)
    """
    pred_path = f"../../predictions/{dataset_name}/dnntsp/keyset{keyset_index}.json"
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    with open(pred_path, 'w') as f:
        json.dump(predictions, f)
    print(f"Predictions saved to {pred_path}")


def load_default_parameters(dataset_name):
    """
    Load default parameters from config (if available)
    """
    config_file = "best_params_config.json"
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config[dataset_name]['best_parameters']



def main():
    """Main function with argument parsing (similar to sets2sets)"""
    
    dataset_name = sys.argv[1]
    default_params = load_default_parameters(dataset_name)
    
    # Now create the full argument parser with defaults from config
    parser = argparse.ArgumentParser(description='DNNTSP Recommendation Model')
    parser.add_argument('dataset_name', help='Name of the dataset')
    parser.add_argument('keyset_index', type=int, default=0, help='Keyset index')
    parser.add_argument('--mode', choices=['train', 'test'], default='train',
                       help='Mode: train or test')
    parser.add_argument('--item_embed_dim', type=int, default=default_params.get('item_embed_dim', 32), help='Item embedding dimension')
    parser.add_argument('--batch_size', type=int, default=default_params.get('batch_size', 64), help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=default_params.get('learning_rate', 0.001), help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--loss_function', type=str, default=default_params.get('loss_function', 'multi_label_soft_loss'),
                       choices=['bpr_loss', 'mse_loss', 'weight_mse_loss', 'multi_label_soft_loss'],
                       help='Loss function type')
    parser.add_argument('--optim', type=str, default=default_params.get('optim', 'Adam'), choices=['Adam', 'SGD'],
                       help='Optimizer')
    parser.add_argument('--weight_decay', type=float, default=default_params.get('weight_decay', 0), help='Weight decay')
    parser.add_argument('--top_k', type=int, default=10, help='Top-k for evaluation')
    
    args = parser.parse_args()

    # Print parameter information
    print(f"Dataset: {args.dataset_name}")
    print(f"Keyset Index: {args.keyset_index}")
    print(f"Mode: {args.mode}")
    print("Hyperparameters:")
    print(f"  item_embed_dim: {args.item_embed_dim}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  learning_rate: {args.learning_rate}")
    print(f"  epochs: {args.epochs}")
    print(f"  loss_function: {args.loss_function}")
    print(f"  optim: {args.optim}")
    print(f"  weight_decay: {args.weight_decay}")
    print(f"  top_k: {args.top_k}")

    # Set random seeds
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Load data
    history_path, future_path, keyset_path, keyset = load_data(args.dataset_name, args.keyset_index)
    
    items_total = keyset.get('item_num', None)
    training_key_set = keyset.get('train', [])
    val_key_set = keyset.get('val', [])
    test_key_set = keyset.get('test', [])
    print(items_total, len(training_key_set), len(val_key_set), len(test_key_set))
    
    # Create model
    use_cuda = torch.cuda.is_available()
    model = create_model(items_total, args.item_embed_dim)
    
    if use_cuda:
        model = model.cuda()
    
    # Create data loaders
    train_data_loader = get_data_loader(
        history_path=history_path,
        future_path=future_path,
        keyset_path=keyset_path,
        data_type='train',
        batch_size=args.batch_size,
        item_embedding_matrix=model.item_embedding
    )

    val_data_loader = get_data_loader(
        history_path=history_path,
        future_path=future_path,
        keyset_path=keyset_path,
        data_type='val',
        batch_size=args.batch_size,
        item_embedding_matrix=model.item_embedding
    )
    
    test_data_loader = get_data_loader(
        history_path=history_path,
        future_path=future_path,
        keyset_path=keyset_path,
        data_type='test',
        batch_size=args.batch_size,
        item_embedding_matrix=model.item_embedding
    )
    
    # Create loss function
    # For weight_mse_loss, use training users' history data (not future.json)
    loss_func = create_loss(
        loss_type=args.loss_function,
        history_path=history_path,
        keyset_path=keyset_path,
        items_total=items_total
    )
    
    # Model folder
    model_folder = f"../../models/dnntsp/{args.dataset_name}/keyset_{args.keyset_index}"
    
    if args.mode == 'train':
        os.makedirs(model_folder, exist_ok=True)
        
        # Create optimizer
        if args.optim == "Adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay
            )
        elif args.optim == "SGD":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=args.learning_rate,
                momentum=0.9
            )
        else:
            raise NotImplementedError(f"Optimizer {args.optim} not implemented")
        
        # Train model
        actual_epochs = train_model(
            model=model,
            future_path=future_path,
            train_data_loader=train_data_loader,
            val_data_loader=val_data_loader,
            val_key_set=val_key_set,
            loss_func=loss_func,
            epochs=args.epochs,
            optimizer=optimizer,
            model_folder=model_folder,
            top_k=args.top_k,
            should_save_model=True
        )
        print(f"Training completed after {actual_epochs} epochs")
    else:
        # Test mode - load best model
        model = load_model(model, f"{args.dataset_name}/keyset_{args.keyset_index}")
    
    # Generate predictions for test set 
    predicted_test = evaluate(
        model, test_data_loader, test_key_set, args.top_k
    )
    
    # Generate predictions for validation set
    predicted_val = evaluate(
        model, val_data_loader, val_key_set, args.top_k
    )
    
    # Combine validation and test predictions
    pred_dict = dict()
    pred_dict.update(predicted_val)
    pred_dict.update(predicted_test)
    
    # Save predictions
    save_predictions(pred_dict, args.dataset_name, args.keyset_index)
    print(f"Predictions saved for {len(pred_dict)} users")


if __name__ == '__main__':
    main()
