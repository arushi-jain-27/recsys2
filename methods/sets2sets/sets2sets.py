import numpy as np
import random
import sys
import os
import json
import argparse
from datetime import datetime
from tqdm import tqdm
import time
import math

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), '../../evaluations/'))
from evaluate_recommendation_model import RecommendationEvaluator

topk = 10

class EncoderRNN_new(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, use_embedding=1, use_linear_reduction=0, use_cuda=False):
        super(EncoderRNN_new, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_embedding = use_embedding
        self.use_linear_reduction = use_linear_reduction
        self.use_cuda = use_cuda
        
        # Only create layers that are actually used based on configuration
        if use_linear_reduction:
            self.reduction = nn.Linear(input_size, hidden_size)
        if use_embedding:
            self.embedding = nn.Embedding(input_size, hidden_size)
        
        if use_embedding or use_linear_reduction:
            self.gru = nn.GRU(hidden_size, hidden_size, num_layers)
        else:
            self.gru = nn.GRU(input_size, hidden_size, num_layers)

    def forward(self, input, hidden, use_average_embedding=1):
        if self.use_embedding:
            list = Variable(torch.LongTensor(input).view(-1, 1))
            if self.use_cuda:
                list = list.cuda()
            average_embedding = Variable(torch.zeros(self.hidden_size)).view(1, 1, -1)
            if self.use_cuda:
                average_embedding = average_embedding.cuda()

            for ele in list:
                embedded = self.embedding(ele).view(1, 1, -1)
                tmp = average_embedding.clone()
                average_embedding = tmp + embedded

            if use_average_embedding:
                tmp = [1] * self.hidden_size
                length = Variable(torch.FloatTensor(tmp))
                if self.use_cuda:
                    length = length.cuda()
                real_ave = average_embedding.view(-1) / length
                average_embedding = real_ave.view(1, 1, -1)

            embedding = average_embedding
        else:
            tensorized_input = torch.FloatTensor(input).clone()
            inputs = Variable(torch.unsqueeze(tensorized_input, 0).view(1, -1))
            if self.use_cuda:
                inputs = inputs.cuda()
            if self.use_linear_reduction == 1:
                reduced_input = self.reduction(inputs)
            else:
                reduced_input = inputs

            embedding = torch.unsqueeze(reduced_input, 0)

        output, hidden = self.gru(embedding, hidden)
        return output, hidden

    def initHidden(self):
        num_layers = self.gru.num_layers
        result = Variable(torch.zeros(num_layers, 1, self.hidden_size))
        if self.use_cuda:
            return result.cuda()
        else:
            return result


class AttnDecoderRNN_new(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, dropout_p=0.2, max_length=100, 
                 use_embedding=1, use_linear_reduction=0, use_cuda=False):
        super(AttnDecoderRNN_new, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.use_embedding = use_embedding
        self.use_linear_reduction = use_linear_reduction
        self.use_cuda = use_cuda

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        if use_embedding or use_linear_reduction:
            self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
            self.attn1 = nn.Linear(self.hidden_size + output_size, self.hidden_size)
        else:
            self.attn = nn.Linear(self.hidden_size + self.output_size, self.output_size)

        if use_embedding or use_linear_reduction:
            self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        else:
            self.attn_combine = nn.Linear(self.hidden_size + self.output_size, self.hidden_size)
        
        # Replace Linear(output_size, output_size) with element-wise parameter
        # This reduces parameters from O(n^2) to O(n): 14.7B → 121K params for taobao!
        self.history_weight = nn.Parameter(torch.zeros(self.output_size))
        
        self.dropout = nn.Dropout(self.dropout_p)
        # Only create reduction layer when needed (use_linear_reduction and not use_embedding)
        if use_linear_reduction and not use_embedding:
            self.reduction = nn.Linear(self.output_size, self.hidden_size)
        if use_embedding or use_linear_reduction:
            self.gru = nn.GRU(hidden_size, hidden_size, num_layers)
        else:
            self.gru = nn.GRU(hidden_size, hidden_size, num_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs, history_record, last_hidden, 
                use_average_embedding=1):
        if self.use_embedding:
            list = Variable(torch.LongTensor(input).view(-1, 1))
            if self.use_cuda:
                list = list.cuda()
            average_embedding = Variable(torch.zeros(self.hidden_size)).view(1, 1, -1)
            if self.use_cuda:
                average_embedding = average_embedding.cuda()

            for ele in list:
                embedded = self.embedding(ele).view(1, 1, -1)
                tmp = average_embedding.clone()
                average_embedding = tmp + embedded

            if use_average_embedding:
                tmp = [1] * self.hidden_size
                length = Variable(torch.FloatTensor(tmp))
                if self.use_cuda:
                    length = length.cuda()
                real_ave = average_embedding.view(-1) / length
                average_embedding = real_ave.view(1, 1, -1)

            embedding = average_embedding
        else:
            tensorized_input = torch.FloatTensor(input).clone()
            inputs = Variable(torch.unsqueeze(tensorized_input, 0).view(1, -1))
            if self.use_cuda:
                inputs = inputs.cuda()
            if self.use_linear_reduction == 1:
                reduced_input = self.reduction(inputs)
            else:
                reduced_input = inputs

            embedding = torch.unsqueeze(reduced_input, 0)

        droped_ave_embedded = self.dropout(embedding)

        history_context = Variable(torch.FloatTensor(history_record).view(1, -1))
        if self.use_cuda:
            history_context = history_context.cuda()

        attn_weights = F.softmax(
            self.attn(torch.cat((droped_ave_embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        element_attn_weights = F.softmax(
            self.attn1(torch.cat((history_context, hidden[0]), 1)), dim=1)

        output = torch.cat((droped_ave_embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        linear_output = self.out(output[0])

        # Element-wise transformation: (1, output_size) * (output_size,) = (1, output_size)
        value = torch.sigmoid(history_context * self.history_weight.unsqueeze(0)).unsqueeze(0)

        one_vec = Variable(torch.ones(self.output_size).view(1, -1))
        if self.use_cuda:
            one_vec = one_vec.cuda()

        res = history_context.clone()
        res[history_context != 0] = 1
        output = F.softmax(linear_output * (one_vec - res * value[0]) + history_context * value[0], dim=1)

        return output.view(1, -1), hidden, attn_weights

    def initHidden(self):
        num_layers = self.gru.num_layers
        result = Variable(torch.zeros(num_layers, 1, self.hidden_size))
        if self.use_cuda:
            return result.cuda()
        else:
            return result


class custom_MultiLabelLoss_torch(nn.modules.loss._Loss):
    def __init__(self, labmda, use_cuda):
        super(custom_MultiLabelLoss_torch, self).__init__()
        self.labmda = labmda
        self.use_cuda = use_cuda

    def forward(self, pred, target, weights):
        #balance the mseloss, incase that some items occurs frequently in the training dataset
        mseloss = torch.sum(weights * torch.pow((pred - target), 2))
        pred = pred.data
        target = target.data

        ones_idx_set = (target == 1).nonzero()
        zeros_idx_set = (target == 0).nonzero()

        ones_set = torch.index_select(pred, 1, ones_idx_set[:, 1])
        zeros_set = torch.index_select(pred, 1, zeros_idx_set[:, 1])

        repeat_ones = ones_set.repeat(1, zeros_set.shape[1])
        repeat_zeros_set = torch.transpose(zeros_set.repeat(ones_set.shape[1], 1), 0, 1).clone()
        repeat_zeros = repeat_zeros_set.reshape(1, -1)
        difference_val = -(repeat_ones - repeat_zeros)
        exp_val = torch.exp(difference_val)
        exp_loss = torch.sum(exp_val)
        normalized_loss = exp_loss / (zeros_set.shape[1] * ones_set.shape[1])
        set_loss = Variable(torch.FloatTensor([self.labmda * normalized_loss]), requires_grad=True)
        if self.use_cuda:
            set_loss = set_loss.cuda()
        loss = mseloss + set_loss

        return loss


def get_codes_frequency_no_vector(history_data, num_dim, key_set):
    """Calculate item frequencies for weighting"""
    result_vector = np.zeros(num_dim)
    for pid in key_set:
        for idx in history_data[pid]:
            if idx == [-1]:
                continue
            result_vector[idx] += 1
    return result_vector


def train_single_step(input_variable, target_variable, encoder, decoder, codes_inverse_freq, 
                     encoder_optimizer, decoder_optimizer, criterion, output_size, max_length):
    """Train a single step of the model"""
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = len(input_variable)
    target_length = len(target_variable)

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    if encoder.use_cuda:
        encoder_outputs = encoder_outputs.cuda()

    history_record = np.zeros(output_size)
    for ei in range(input_length - 1):
        if ei == 0: #because first basket in input variable is [-1]
            continue
        for ele in input_variable[ei]:
            history_record[ele] += 1.0 / (input_length - 2)

    for ei in range(input_length - 1):
        if ei == 0:
            continue
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei - 1] = encoder_output[0][0]

    last_input = input_variable[input_length - 2]
    decoder_hidden = encoder_hidden
    last_hidden = encoder_hidden
    decoder_input = last_input

    decoder_output, decoder_hidden, decoder_attention = decoder(
        decoder_input, decoder_hidden, encoder_outputs, history_record, last_hidden)

    #create target tensor.
    vectorized_target = np.zeros(output_size)
    for idx in target_variable[1]:
        vectorized_target[idx] = 1
    target = Variable(torch.FloatTensor(vectorized_target).reshape(1, -1))

    if encoder.use_cuda:
        target = target.cuda()
    weights = Variable(torch.FloatTensor(codes_inverse_freq).reshape(1, -1))
    if encoder.use_cuda:
        weights = weights.cuda()

    loss = criterion(decoder_output, target, weights)
    loss.backward()

    # Gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()




def train_model(data_history, data_future, output_size, encoder, decoder, model_name, 
                training_key_set, val_keyset, codes_inverse_freq, next_k_step, n_iters, top_k, learning_rate, max_length, should_save_model, labmda):
    """Train the Sets2Sets model"""
    start = time.time()
    print_loss_total = 0

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-11,
                                         weight_decay=1e-5)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-11,
                                         weight_decay=1e-5)

    total_iter = 0
    criterion = custom_MultiLabelLoss_torch(use_cuda=encoder.use_cuda, labmda=labmda)
    best_ndcg = 0.0
    patience = 5  # Early stopping patience
    patience_counter = 0
    best_epoch = 0
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        encoder_optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # train n_iters epoch with early stopping
    for j in range(n_iters):
        # get a suffle list
        key_idx = np.random.permutation(len(training_key_set))
        training_keys = []
        for idx in key_idx:
            training_keys.append(training_key_set[idx])

        # for iter in tqdm(range(0, len(training_key_set))):
        for iter in range(0, len(training_key_set)):

            # get training data and label.
            input_variable = data_history[training_keys[iter]]
            target_variable = data_future[training_keys[iter]]

            loss = train_single_step(input_variable, target_variable, encoder,
                         decoder, codes_inverse_freq, encoder_optimizer, decoder_optimizer, 
                         criterion, output_size, max_length)

            print_loss_total += loss
            total_iter += 1

        # print loss and save model
        print_loss_avg = print_loss_total / len(training_key_set)
        print_loss_total = 0
        print(f'{datetime.now()}: Epoch {j} Train Loss: {print_loss_avg}')
        sys.stdout.flush()

        # Evaluate on validation set for early stopping
        if val_keyset and len(val_keyset) > 0:
            val_predictions = evaluate(data_history, data_future, encoder, decoder, 
                                     output_size, val_keyset, next_k_step, top_k, max_length)
            
            # Calculate validation nDCG@5 using RecommendationEvaluator
            val_ndcg = 0.0
            if val_predictions:
                
                # Create temporary evaluator
                temp_evaluator = RecommendationEvaluator("temp", "temp")
                temp_evaluator.predictions = val_predictions
                temp_evaluator.ground_truth = {k: data_future[k] for k in val_keyset if k in data_future}
                
                # Calculate nDCG@5
                eval_df, avg_metrics = temp_evaluator.evaluate()
                val_ndcg = avg_metrics.get('nDCG@5', 0.0)
            
            print(f'{datetime.now()}: Epoch {j} Validation nDCG@5: {val_ndcg:.4f}')
            
            # Learning rate scheduling
            scheduler.step(val_ndcg)
            
            # Early stopping logic
            if val_ndcg > best_ndcg:
                best_ndcg = val_ndcg
                patience_counter = 0
                if should_save_model:
                    save_model(encoder, decoder, model_name)
                    print('New best model saved.')
                best_epoch = j
            else:
                patience_counter += 1
                print(f'No improvement for {patience_counter} epochs')
                
            if patience_counter >= patience:
                print(f'{datetime.now()}: Early stopping at epoch {j} (patience={patience})')
                break
        else:
            # If no validation set, just save the model
            if should_save_model:
                save_model(encoder, decoder, model_name)
                print(f'{datetime.now()}: Model is saved.')
        
        print(f'{datetime.now()}: Finish epoch: ' + str(j))
        
    
    return best_epoch + 1  # Return actual best epoch




def decoding_next_k_step(encoder, decoder, input_variable, target_variable, output_size, k, activate_codes_num, max_length):
    """Decode next k steps for evaluation"""
    encoder_hidden = encoder.initHidden()

    input_length = len(input_variable)
    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    if encoder.use_cuda:
        encoder_outputs = encoder_outputs.cuda()

    # history frequency information
    history_record = np.zeros(output_size)
    count = 0.0
    for ei in range(input_length - 1):
        if ei == 0:
            continue
        for ele in input_variable[ei]:
            history_record[ele] += 1
        count += 1.0
    history_record = history_record / count

    # basket item iterator
    for ei in range(input_length - 1):
        if ei == 0:
            continue
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei - 1] = encoder_output[0][0]

    decoder_input = input_variable[input_length - 2]
    decoder_hidden = encoder_hidden
    last_hidden = decoder_hidden
    topk = 400
    decoded_vectors = []
    prob_vectors = []
    
    # k is the number of steps need to predicted, for next basket is 1
    for di in range(k):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs, history_record, last_hidden)
        # topv is top values, topi is top indicies.
        topv, topi = decoder_output.data.topk(topk)

        # construct target vector
        vectorized_target = np.zeros(output_size)
        for idx in target_variable[di + 1]:# iter the target basket.
            vectorized_target[idx] = 1

        count = 0
        if activate_codes_num > 0:
            pick_num = activate_codes_num
        else:
            pick_num = np.sum(vectorized_target)

        tmp = []
        for ele in range(len(topi[0])):
            if count >= pick_num:
                break
            tmp.append(topi[0][ele].item())  
            count += 1
        decoded_vectors.append(tmp)
        decoder_input = tmp

        tmp = []
        for i in range(topk):
            tmp.append(topi[0][i].item()) 
        prob_vectors.append(tmp)

    return decoded_vectors, prob_vectors


def evaluate(history_data, future_data, encoder, decoder, output_size, test_key_set, 
                   next_k_step, activate_codes_num, max_length):
    """Generate predictions for test users and return as dictionary"""
    predictions = {}
    
    for iter in range(len(test_key_set)):
        user_id = test_key_set[iter]
        input_variable = history_data[user_id]
        target_variable = future_data[user_id]

        if len(target_variable) < 2 + next_k_step:
            # If user doesn't have enough future data, return empty list
            predictions[user_id] = []
            continue
            
        # Generate predictions for next k steps
        output_vectors, prob_vectors = decoding_next_k_step(encoder, decoder, input_variable, target_variable,
                                                            output_size, next_k_step, activate_codes_num, max_length)

        # For the first predicted basket (next basket), get top-k items
        if len(output_vectors) > 0:
            # Get the top-k predicted items from the first predicted basket
            # Use the activate_codes_num as the top-k limit
            top_k_limit = activate_codes_num if activate_codes_num > 0 else 10
            predicted_items = output_vectors[0][:top_k_limit] if len(output_vectors[0]) >= top_k_limit else output_vectors[0]
            predictions[user_id] = predicted_items
        else:
            predictions[user_id] = []

    return predictions




def load_model(encoder, decoder, model_name, epoch=None, device='cpu'):
    """Load model with consistent naming"""
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
    print(f"Loaded model from {directory}")
    
    return encoder, decoder


def save_model(encoder, decoder, model_name, epoch=None):
    """Save model with consistent naming"""
    directory = f'../../models/sets2sets/{model_name}/'
    os.makedirs(directory, exist_ok=True)
    
    if epoch is not None:
        encoder_path = f'{directory}encoder_epoch_{epoch}.pth'
        decoder_path = f'{directory}decoder_epoch_{epoch}.pth'
    else:
        encoder_path = f'{directory}encoder_best.pth'
        decoder_path = f'{directory}decoder_best.pth'
    
    torch.save(encoder.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)


def load_data(dataset_name, keyset_index=0, max_length=100):
    """Load dataset files with consistent path structure"""
    history_file = f"../../datasets/{dataset_name}/history.json"
    future_file = f"../../datasets/{dataset_name}/future.json"
    keyset_file = f"../../datasets/{dataset_name}/keyset_{keyset_index}.json"
    
    with open(history_file, 'r') as f:
        data_history = json.load(f)
    with open(future_file, 'r') as f:
        data_future = json.load(f)
    with open(keyset_file, 'r') as f:
        keyset = json.load(f)

    # truncate the history data to max_length baskets
    for key, value in data_history.items():
        if len(value[1:-1]) > max_length:
            data_history[key] = [-1] + value[-max_length-1:-1] + [-1]
    
    return data_history, data_future, keyset


def save_predictions(predictions, dataset_name, keyset_index):
    """Save predictions in consistent format"""
    pred_path = f"../../predictions/{dataset_name}/sets2sets/keyset{keyset_index}.json"
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    with open(pred_path, 'w') as f:
        json.dump(predictions, f)


def load_default_parameters(dataset_name):
    config_file = "best_params_config.json"
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    if dataset_name not in config:
        return {}
    return config[dataset_name]['best_parameters']


def main():
    """Main function with argument parsing"""

    dataset_name = sys.argv[1]
    default_params = load_default_parameters(dataset_name)
    
    # Now create the full argument parser with defaults from config
    parser = argparse.ArgumentParser(description='Sets2Sets Recommendation Model')
    parser.add_argument('dataset_name', help='Name of the dataset')
    parser.add_argument('keyset_index', type=int, default=0, help='Keyset index')
    parser.add_argument('--mode', choices=['train', 'test'], default='train',
                       help='Mode: train or test')
    parser.add_argument('--hidden_size', type=int, default=default_params.get('hidden_size', 16), help='Hidden size')
    parser.add_argument('--num_layers', type=int, default=default_params.get('num_layers', 2), help='Number of layers')
    parser.add_argument('--learning_rate', type=float, default=default_params.get('learning_rate', 0.0005), help='Learning rate')
    parser.add_argument('--num_iter', type=int, default=20, help='Number of iterations')
    parser.add_argument('--labmda', type=float, default=default_params.get('labmda', 5), help='Lambda parameter')
    parser.add_argument('--max_length', type=int, default=default_params.get('max_length', 50), help='Max length')
    parser.add_argument('--dropout_p', type=float, default=default_params.get('dropout_p', 0), help='Dropout probability')
    
    args = parser.parse_args()

    # Print parameter information
    print(f"Dataset: {args.dataset_name}")
    print(f"Keyset Index: {args.keyset_index}")
    print(f"Mode: {args.mode}")
    print("Hyperparameters:")
    print(f"  hidden_size: {args.hidden_size}")
    print(f"  num_layers: {args.num_layers}")
    print(f"  learning_rate: {args.learning_rate}")
    print(f"  num_iter: {args.num_iter}")
    print(f"  labmda: {args.labmda}")
    print(f"  max_length: {args.max_length}")
    print(f"  dropout_p: {args.dropout_p}")

    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Load data
    data_history, data_future, keyset = load_data(args.dataset_name, args.keyset_index, args.max_length)
    
    input_size = keyset['item_num']
    training_key_set = keyset['train']
    val_key_set = keyset['val']
    test_key_set = keyset['test']
    
    # Calculate weights
    weights = np.zeros(input_size)
    codes_freq = get_codes_frequency_no_vector(data_history, input_size, data_future.keys())
    max_freq = max(codes_freq)
    for idx in range(len(codes_freq)):
        if codes_freq[idx] > 0:
            weights[idx] = max_freq / codes_freq[idx]
        else:
            weights[idx] = 0
    
    # Create models
    use_cuda = torch.cuda.is_available()
    encoder = EncoderRNN_new(input_size, args.hidden_size, args.num_layers, 
                            use_embedding=1, use_linear_reduction=0, use_cuda=use_cuda)
    decoder = AttnDecoderRNN_new(args.hidden_size, input_size, args.num_layers, args.dropout_p, args.max_length,
                                use_embedding=1, use_linear_reduction=0, use_cuda=use_cuda)
    
    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    
    if args.mode == 'train':
        actual_epochs = train_model(data_history, data_future, input_size, encoder, decoder, 
                   f"{args.dataset_name}/keyset_{args.keyset_index}", training_key_set, val_key_set, 
                   weights, 1, args.num_iter, topk, args.learning_rate, args.max_length, should_save_model=True, labmda=args.labmda)
        print(f"Training completed after {actual_epochs} epochs")
    else:
        # Test mode - load best model and generate predictions
        encoder, decoder = load_model(encoder, decoder, f"{args.dataset_name}/keyset_{args.keyset_index}")
        
    # Generate predictions for test set
    predicted_test = evaluate(data_history, data_future, encoder, decoder, 
                                    input_size, test_key_set, 1, topk, args.max_length)
    
    # Generate predictions for validation set
    predicted_val = evaluate(data_history, data_future, encoder, decoder, 
                                    input_size, val_key_set, 1, topk, args.max_length)
    
    # Combine validation and test predictions
    pred_dict = dict()
    pred_dict.update(predicted_val)
    pred_dict.update(predicted_test)
    
    # Save predictions
    save_predictions(pred_dict, args.dataset_name, args.keyset_index)
    print(f"Predictions saved for {len(pred_dict)} users")


if __name__ == '__main__':
    main()
