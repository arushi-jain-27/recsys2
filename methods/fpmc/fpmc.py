import argparse
import json
import os
import random
import sys
from datetime import datetime

import numpy as np
from numba import njit

sys.path.append("../../evaluations/")
from evaluate_recommendation_model import RecommendationEvaluator


topk = 10


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
    pred_path = f"../../predictions/{dataset_name}/fpmc/keyset{keyset_index}.json"
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    with open(pred_path, "w") as f:
        json.dump(predictions, f)


def build_train_instances(data_history):
    u_list, i_list, baskets = [], [], []
    for user, seq in data_history.items():
        hist = user_baskets(seq)
        uid = int(user)
        if len(hist) >= 2:
            for t in range(len(hist) - 1):
                for item in hist[t + 1]:
                    u_list.append(uid)
                    i_list.append(item)
                    baskets.append(hist[t])
        else:
            for item in hist[0]:
                u_list.append(uid)
                i_list.append(item)
                baskets.append(hist[0])

    lens = np.array([len(b) for b in baskets], dtype=np.int64)
    offsets = np.zeros(len(baskets), dtype=np.int64)
    offsets[1:] = np.cumsum(lens[:-1])
    concat = np.empty(int(lens.sum()), dtype=np.int64)
    for idx, basket in enumerate(baskets):
        concat[offsets[idx]:offsets[idx] + lens[idx]] = basket
    return np.array(u_list, dtype=np.int64), np.array(i_list, dtype=np.int64), concat, offsets, lens


@njit
def _sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    ex = np.exp(x)
    return ex / (1.0 + ex)


@njit
def _compute_x(u, i, b_concat, start, length, VUI, VIU, VLI, VIL):
    acc = 0.0
    for t in range(length):
        acc += np.dot(VIL[i], VLI[b_concat[start + t]])
    return np.dot(VUI[u], VIU[i]) + acc / length


@njit
def _learn_epoch(u_list, i_list, b_concat, offsets, lens, order, n_neg, n_item,
                 VUI, VIU, VLI, VIL, learn_rate, regular):
    n_factor = VUI.shape[1]
    for step in range(order.shape[0]):
        d_idx = order[step]
        u = u_list[d_idx]
        i = i_list[d_idx]
        start = offsets[d_idx]
        length = lens[d_idx]
        z1 = _compute_x(u, i, b_concat, start, length, VUI, VIU, VLI, VIL)
        for _ in range(n_neg):
            j = np.random.randint(0, n_item)
            while j == i:
                j = np.random.randint(0, n_item)
            z2 = _compute_x(u, j, b_concat, start, length, VUI, VIU, VLI, VIL)
            delta = 1.0 - _sigmoid(z1 - z2)

            vui_u = VUI[u].copy()
            VUI[u] += learn_rate * (delta * (VIU[i] - VIU[j]) - regular * vui_u)
            VIU[i] += learn_rate * (delta * vui_u - regular * VIU[i])
            VIU[j] += learn_rate * (-delta * vui_u - regular * VIU[j])

            eta = np.zeros(n_factor)
            for t in range(length):
                eta += VLI[b_concat[start + t]]
            eta /= length
            vil_i = VIL[i].copy()
            vil_j = VIL[j].copy()
            VIL[i] += learn_rate * (delta * eta - regular * vil_i)
            VIL[j] += learn_rate * (-delta * eta - regular * vil_j)
            inv_len = 1.0 / length
            for t in range(length):
                l = b_concat[start + t]
                VLI[l] += learn_rate * ((delta * (vil_i - vil_j) * inv_len) - regular * VLI[l])
    return VUI, VIU, VLI, VIL


class FPMC:
    def __init__(self, n_user, n_item, n_factor, learn_rate, regular):
        self.n_item = n_item
        self.learn_rate = learn_rate
        self.regular = regular
        std = 0.01
        self.VUI = np.random.normal(0, std, size=(n_user, n_factor))
        self.VIU = np.random.normal(0, std, size=(n_item, n_factor))
        self.VIL = np.random.normal(0, std, size=(n_item, n_factor))
        self.VLI = np.random.normal(0, std, size=(n_item, n_factor))

    def compute_scores(self, u, last_basket):
        scores = self.VIU.dot(self.VUI[u])
        scores += self.VIL.dot(self.VLI[np.array(last_basket)].mean(axis=0))
        return scores

    def learn_epoch(self, u_list, i_list, b_concat, offsets, lens, n_neg):
        order = np.random.permutation(len(u_list))
        self.VUI, self.VIU, self.VLI, self.VIL = _learn_epoch(
            u_list, i_list, b_concat, offsets, lens, order, n_neg, self.n_item,
            self.VUI, self.VIU, self.VLI, self.VIL, self.learn_rate, self.regular,
        )


def predict(model, data_history, user_ids, top_k=topk):
    predictions = {}
    for uid in user_ids:
        last_basket = user_baskets(data_history[uid])[-1]
        scores = model.compute_scores(int(uid), last_basket)
        predictions[uid] = np.argsort(scores)[::-1][:top_k].tolist()
    return predictions


def train_model(data_history, data_future, keyset, n_factor, learn_rate, regular,
                n_neg, n_epoch, patience=5, top_k=topk):
    u_list, i_list, b_concat, offsets, lens = build_train_instances(data_history)
    n_user = max(int(u) for u in data_history) + 1
    n_item = keyset["item_num"]
    print(f"{datetime.now()}: Training instances={len(u_list)} users={n_user} items={n_item}")

    model = FPMC(n_user, n_item, n_factor, learn_rate, regular)
    best_ndcg = -1
    best_state = None
    patience_counter = 0

    for epoch in range(n_epoch):
        model.learn_epoch(u_list, i_list, b_concat, offsets, lens, n_neg)
        val_pred = predict(model, data_history, keyset["val"], top_k)
        evaluator = RecommendationEvaluator("temp", "temp", split="val")
        evaluator.predictions = val_pred
        evaluator.ground_truth = {uid: data_future[uid] for uid in keyset["val"]}
        _, avg_metrics = evaluator.evaluate()
        val_ndcg = avg_metrics["nDCG@5"]
        print(f"{datetime.now()}: Epoch {epoch} Validation nDCG@5: {val_ndcg:.4f}")
        sys.stdout.flush()

        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            patience_counter = 0
            best_state = (model.VUI.copy(), model.VIU.copy(), model.VIL.copy(), model.VLI.copy())
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")
            if patience_counter >= patience:
                print(f"{datetime.now()}: Early stopping at epoch {epoch}")
                break

    model.VUI, model.VIU, model.VIL, model.VLI = best_state
    return model


def main():
    dataset_name = sys.argv[1]
    default_params = load_default_parameters(dataset_name)

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name")
    parser.add_argument("keyset_index", type=int, nargs="?", default=0)
    parser.add_argument("--n_factor", type=int, default=default_params.get("n_factor", 32))
    parser.add_argument("--learn_rate", type=float, default=default_params.get("learn_rate", 0.01))
    parser.add_argument("--regular", type=float, default=default_params.get("regular", 0.001))
    parser.add_argument("--n_neg", type=int, default=default_params.get("n_neg", 10))
    parser.add_argument("--n_epoch", type=int, default=default_params.get("n_epoch", 20))
    args = parser.parse_args()

    print(f"Dataset: {args.dataset_name}")
    print(f"Keyset Index: {args.keyset_index}")
    print("Hyperparameters:")
    print(f"  n_factor: {args.n_factor}")
    print(f"  learn_rate: {args.learn_rate}")
    print(f"  regular: {args.regular}")
    print(f"  n_neg: {args.n_neg}")
    print(f"  n_epoch: {args.n_epoch}")

    random.seed(42)
    np.random.seed(42)

    data_history, data_future, keyset = load_data(args.dataset_name, args.keyset_index)
    model = train_model(
        data_history, data_future, keyset,
        args.n_factor, args.learn_rate, args.regular, args.n_neg, args.n_epoch,
    )
    pred_dict = {}
    pred_dict.update(predict(model, data_history, keyset["val"]))
    pred_dict.update(predict(model, data_history, keyset["test"]))
    save_predictions(pred_dict, args.dataset_name, args.keyset_index)
    print(f"Predictions saved for {len(pred_dict)} users")


if __name__ == "__main__":
    main()
