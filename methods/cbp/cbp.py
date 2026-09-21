import argparse
import copy
import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("../../evaluations/")
from evaluate_recommendation_model import RecommendationEvaluator


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
    pred_path = f"../../predictions/{dataset_name}/cbp/keyset{keyset_index}.json"
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    with open(pred_path, "w") as f:
        json.dump(predictions, f)


def basket_counts(baskets, n_item):
    counts = np.zeros(n_item, dtype=np.float32)
    for basket in baskets:
        for item in basket:
            counts[item] += 1
    return counts


def l1_normalize(counts):
    total = counts.sum()
    if total > 0:
        return counts / total
    return counts


def pad_seqs(seqs, pad_idx, decay):
    max_t = max(len(s) for s in seqs)
    max_b = max(len(b) for s in seqs for b in s)
    padded = np.full((len(seqs), max_t, max_b), pad_idx, dtype=np.int64)
    decays = np.zeros((len(seqs), max_t, 1), dtype=np.float32)
    for i, seq in enumerate(seqs):
        d = len(seq) - 1
        for t, basket in enumerate(seq):
            padded[i, t, :len(basket)] = basket
            decays[i, t, 0] = decay ** d
            d -= 1
    return padded, decays


class CBP(nn.Module):
    def __init__(self, n_item, dim, alpha, beta, decay):
        super().__init__()
        self.n_item = n_item
        self.alpha = alpha
        self.beta = beta
        self.decay = decay
        self.pad_idx = n_item
        self.itemEmb = nn.Embedding(n_item + 1, dim, padding_idx=n_item)
        self.out = nn.Linear(dim, n_item)
        self.his_embds = nn.Linear(n_item, dim)

    def forward(self, seq, decay, uhis, itemPOP):
        embs = self.itemEmb(seq)
        embs2d = torch.tanh((decay * embs.sum(2)).sum(1))
        scores_trans = F.softmax(self.out(embs2d), dim=-1)
        itemPOP_his = self.his_embds(itemPOP.repeat(seq.size(0), 1))
        Iembs = torch.tanh(self.itemEmb(seq).sum(2).sum(1))
        itemBias = F.softmax(self.out(Iembs * itemPOP_his), dim=-1)
        return F.softmax(scores_trans - self.alpha * itemBias + self.beta * uhis, dim=-1)


def build_features(data_history, n_item):
    itemPOP = np.zeros(n_item, dtype=np.float32)
    hist = {}
    train_uids = []
    for uid, seq in data_history.items():
        baskets = user_baskets(seq)
        hist[uid] = baskets
        itemPOP += basket_counts(baskets, n_item)
        if len(baskets) >= 2:
            train_uids.append(uid)
    return hist, train_uids, itemPOP


def make_batch(uids, hist, n_item, pad_idx, decay, predict=False):
    seqs, uhis, targets = [], [], []
    for uid in uids:
        baskets = hist[uid]
        inp = baskets if predict else baskets[:-1]
        seqs.append(inp)
        uhis.append(l1_normalize(basket_counts(inp, n_item)))
        if not predict:
            t = np.zeros(n_item, dtype=np.float32)
            for item in baskets[-1]:
                t[item] = 1
            targets.append(t)
    padded, decays = pad_seqs(seqs, pad_idx, decay)
    out = {
        "seq": torch.from_numpy(padded).to(device),
        "decay": torch.from_numpy(decays).to(device),
        "uhis": torch.from_numpy(np.stack(uhis)).to(device),
    }
    if not predict:
        out["target"] = torch.from_numpy(np.stack(targets)).to(device)
    return out


def predict(model, data_history, user_ids, itemPOP, top_k=topk, batch_size=100):
    model.eval()
    hist = {uid: user_baskets(data_history[uid]) for uid in user_ids}
    predictions = {}
    with torch.no_grad():
        for start in range(0, len(user_ids), batch_size):
            batch_uids = user_ids[start:start + batch_size]
            batch = make_batch(batch_uids, hist, model.n_item, model.pad_idx, model.decay, predict=True)
            scores = model(batch["seq"], batch["decay"], batch["uhis"], itemPOP)
            top = torch.topk(scores, top_k, dim=-1).indices.cpu().tolist()
            for uid, items in zip(batch_uids, top):
                predictions[uid] = items
    return predictions


def train_model(data_history, data_future, keyset, dim, lr, l2, alpha, beta, decay,
                batch_size, n_epoch, patience=5, top_k=topk):
    n_item = keyset["item_num"]
    hist, train_uids, itemPOP_np = build_features(data_history, n_item)
    itemPOP = torch.from_numpy(itemPOP_np).to(device)
    print(f"{datetime.now()}: Training users={len(train_uids)} items={n_item} device={device}")

    model = CBP(n_item, dim, alpha, beta, decay).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
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
            batch = make_batch(batch_uids, hist, n_item, model.pad_idx, decay)
            scores = model(batch["seq"], batch["decay"], batch["uhis"], itemPOP)
            target = batch["target"]
            loss = -(torch.log(scores) * target + torch.log(1 - scores) * (1 - target)).sum(-1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batch += 1

        val_pred = predict(model, data_history, keyset["val"], itemPOP, top_k, batch_size)
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
    return model, itemPOP


def main():
    dataset_name = sys.argv[1]
    default_params = load_default_parameters(dataset_name)

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name")
    parser.add_argument("keyset_index", type=int, nargs="?", default=0)
    parser.add_argument("--dim", type=int, default=default_params.get("dim", 32))
    parser.add_argument("--lr", type=float, default=default_params.get("lr", 0.01))
    parser.add_argument("--l2", type=float, default=default_params.get("l2", 1e-5))
    parser.add_argument("--alpha", type=float, default=default_params.get("alpha", 0.1))
    parser.add_argument("--beta", type=float, default=default_params.get("beta", 0.2))
    parser.add_argument("--decay", type=float, default=default_params.get("decay", 0.6))
    parser.add_argument("--batch_size", type=int, default=default_params.get("batch_size", 100))
    parser.add_argument("--n_epoch", type=int, default=default_params.get("n_epoch", 20))
    args = parser.parse_args()

    print(f"Dataset: {args.dataset_name}")
    print(f"Keyset Index: {args.keyset_index}")
    print("Hyperparameters:")
    for k in ["dim", "lr", "l2", "alpha", "beta", "decay", "batch_size", "n_epoch"]:
        print(f"  {k}: {getattr(args, k)}")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    data_history, data_future, keyset = load_data(args.dataset_name, args.keyset_index)
    model, itemPOP = train_model(
        data_history, data_future, keyset,
        args.dim, args.lr, args.l2, args.alpha, args.beta, args.decay,
        args.batch_size, args.n_epoch,
    )
    pred_dict = {}
    pred_dict.update(predict(model, data_history, keyset["val"], itemPOP, batch_size=args.batch_size))
    pred_dict.update(predict(model, data_history, keyset["test"], itemPOP, batch_size=args.batch_size))
    save_predictions(pred_dict, args.dataset_name, args.keyset_index)
    print(f"Predictions saved for {len(pred_dict)} users")


if __name__ == "__main__":
    main()
