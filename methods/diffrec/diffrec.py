import argparse
import copy
import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import scipy.sparse as sp
import torch
import torch.optim as optim

from dnn import DNN
from gaussian_diffusion import GaussianDiffusion, ModelMeanType

sys.path.append("../../evaluations/")
from evaluate_recommendation_model import RecommendationEvaluator


# Plain SIGIR'23 DiffRec on history interaction vectors. Original inference
# masks seen items; NBR ranking does not, so previously purchased items stay eligible.


topk = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def user_baskets(seq):
    return [b for b in seq if b != [-1]]


def load_data(dataset_name, keyset_index=0):
    with open(f"../../datasets/{dataset_name}/history.json") as f:
        data_history = json.load(f)
    with open(f"../../datasets/{dataset_name}/future.json") as f:
        data_future = json.load(f)
    with open(f"../../datasets/{dataset_name}/keyset_{keyset_index}.json") as f:
        keyset = json.load(f)
    return data_history, data_future, keyset


def load_default_parameters(dataset_name):
    with open("best_params_config.json") as f:
        config = json.load(f)
    if dataset_name not in config:
        return {}
    return config[dataset_name]["best_parameters"]


def save_predictions(predictions, dataset_name, keyset_index):
    pred_path = f"../../predictions/{dataset_name}/diffrec/keyset{keyset_index}.json"
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    with open(pred_path, "w") as f:
        json.dump(predictions, f)


def build_matrix(data_history, n_item):
    n_user = max(int(u) for u in data_history) + 1
    rows, cols, vals = [], [], []
    for uid, seq in data_history.items():
        u = int(uid)
        for basket in user_baskets(seq):
            for item in basket:
                rows.append(u)
                cols.append(item)
                vals.append(1.0)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n_user, n_item), dtype=np.float32)


def batch_dense(matrix, uids):
    return torch.from_numpy(matrix[uids].toarray()).to(device)


def predict(model, diffusion, matrix, user_ids, sampling_steps, top_k=topk, batch_size=400):
    model.eval()
    predictions = {}
    uids = [int(u) for u in user_ids]
    with torch.no_grad():
        for start in range(0, len(uids), batch_size):
            batch_uids = uids[start:start + batch_size]
            x = batch_dense(matrix, batch_uids)
            scores = diffusion.p_sample(model, x, sampling_steps, sampling_noise=False)
            top = torch.topk(scores, top_k, dim=-1).indices.cpu().tolist()
            for uid, items in zip(batch_uids, top):
                predictions[str(uid)] = items
    return predictions


def train_model(data_history, data_future, keyset, hidden, lr, weight_decay, batch_size,
                steps, noise_scale, noise_min, noise_max, n_epoch,
                emb_size=10, sampling_steps=0, reweight=True, patience=5, top_k=topk):
    n_item = keyset["item_num"]
    matrix = build_matrix(data_history, n_item)
    train_uids = [int(u) for u in data_history]
    print(f"{datetime.now()}: users={len(train_uids)} items={n_item} device={device}")

    diffusion = GaussianDiffusion(
        ModelMeanType.START_X, "linear-var", noise_scale, noise_min, noise_max, steps, device,
    ).to(device)
    out_dims = [hidden, n_item]
    in_dims = [n_item, hidden]
    model = DNN(in_dims, out_dims, emb_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_ndcg = -1
    best_state = None
    patience_counter = 0

    for epoch in range(n_epoch):
        model.train()
        random.shuffle(train_uids)
        epoch_loss = 0.0
        n_batch = 0
        for start in range(0, len(train_uids), batch_size):
            batch_uids = train_uids[start:start + batch_size]
            x = batch_dense(matrix, batch_uids)
            optimizer.zero_grad()
            loss = diffusion.training_losses(model, x, reweight)["loss"].mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batch += 1

        val_pred = predict(
            model, diffusion, matrix, keyset["val"], sampling_steps, top_k, batch_size,
        )
        evaluator = RecommendationEvaluator("temp", "temp", split="val")
        evaluator.predictions = val_pred
        evaluator.ground_truth = {uid: data_future[uid] for uid in keyset["val"]}
        _, avg_metrics = evaluator.evaluate()
        val_ndcg = avg_metrics["nDCG@5"]
        print(f"{datetime.now()}: Epoch {epoch} loss={epoch_loss / n_batch:.4f} Validation nDCG@5: {val_ndcg:.4f}")
        sys.stdout.flush()

        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")
            if patience_counter >= patience:
                print(f"{datetime.now()}: Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model, diffusion, matrix


def main():
    dataset_name = sys.argv[1]
    default_params = load_default_parameters(dataset_name)

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name")
    parser.add_argument("keyset_index", type=int, nargs="?", default=0)
    parser.add_argument("--hidden", type=int, default=default_params.get("hidden", 1000))
    parser.add_argument("--lr", type=float, default=default_params.get("lr", 5e-5))
    parser.add_argument("--weight_decay", type=float, default=default_params.get("weight_decay", 0.0))
    parser.add_argument("--batch_size", type=int, default=default_params.get("batch_size", 400))
    parser.add_argument("--steps", type=int, default=default_params.get("steps", 5))
    parser.add_argument("--noise_scale", type=float, default=default_params.get("noise_scale", 0.0001))
    parser.add_argument("--noise_min", type=float, default=default_params.get("noise_min", 0.0005))
    parser.add_argument("--noise_max", type=float, default=default_params.get("noise_max", 0.005))
    parser.add_argument("--n_epoch", type=int, default=default_params.get("n_epoch", 20))
    parser.add_argument("--emb_size", type=int, default=default_params.get("emb_size", 10))
    parser.add_argument("--sampling_steps", type=int, default=default_params.get("sampling_steps", 0))
    args = parser.parse_args()

    print(f"Dataset: {args.dataset_name}")
    print(f"Keyset Index: {args.keyset_index}")
    print("Hyperparameters:")
    for k in ["hidden", "lr", "weight_decay", "batch_size", "steps", "noise_scale",
              "noise_min", "noise_max", "n_epoch"]:
        print(f"  {k}: {getattr(args, k)}")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    data_history, data_future, keyset = load_data(args.dataset_name, args.keyset_index)
    model, diffusion, matrix = train_model(
        data_history, data_future, keyset,
        args.hidden, args.lr, args.weight_decay, args.batch_size,
        args.steps, args.noise_scale, args.noise_min, args.noise_max,
        args.n_epoch,
        args.emb_size, args.sampling_steps,
    )
    pred_dict = {}
    pred_dict.update(predict(model, diffusion, matrix, keyset["val"], args.sampling_steps,
                             batch_size=args.batch_size))
    pred_dict.update(predict(model, diffusion, matrix, keyset["test"], args.sampling_steps,
                             batch_size=args.batch_size))
    save_predictions(pred_dict, args.dataset_name, args.keyset_index)
    print(f"Predictions saved for {len(pred_dict)} users")


if __name__ == "__main__":
    main()
