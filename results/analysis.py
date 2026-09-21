# read the results of the methods and make a table of dataset x method x metric

import pandas as pd
import os
import sys

METRICS = ["HR@5", "nDCG@5", "HR@10", "nDCG@10"]


def make_methods_metrics_table(dataset_name):
    dataset_dir = os.path.join(".", dataset_name)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    rows = {}
    for filename in os.listdir(dataset_dir):
        if not filename.endswith("_summary_metrics.csv"):
            continue

        method_name = filename[: -len("_summary_metrics.csv")]
        file_path = os.path.join(dataset_dir, filename)

        df = pd.read_csv(file_path)
        filtered = df[df["metric"].isin(METRICS)][["metric", "value"]]

        # Ensure consistent column order with missing metrics as NaN
        row_series = (
            filtered.set_index("metric")["value"].reindex(METRICS)
        )
        rows[method_name] = row_series

    if not rows:
        return pd.DataFrame(columns=METRICS)

    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "method"
    table.columns.name = "metric"

    desired_order = [
        "recency",
        "frequency",
        "racf",
        "tifuknn",
        "fpmc",
        "triple2vec",
        "sets2sets",
        "dnntsp",
        "cbp",
        "diffrec",
    ]

    present_in_order = [m for m in desired_order if m in table.index]
    others = [m for m in table.index if m not in desired_order]
    ordered_index = present_in_order + sorted(others)
    return table.reindex(ordered_index)

if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "dunnhumby"
    table = make_methods_metrics_table(dataset)
    table.to_csv(f"./{dataset}.csv", index=True)