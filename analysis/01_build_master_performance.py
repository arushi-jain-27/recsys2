from pathlib import Path
import json
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATASETS = [
    "dunnhumby",
    "instacart",
    "retailrocket",
    "sam",
    "taobao",
    "valuedshopper",
    "tmall",
    "tafeng"
]

METHODS = [
    "recency",
    "frequency",
    "racf",
    "tifuknn",
    "triple2vec",
    "sets2sets",
    "dnntsp",
]

KEYSETS = [0, 1, 2]


all_rows = []

for dataset in DATASETS:
    for keyset in KEYSETS:

        # Expected test users for this repeated split
        keyset_path = ROOT / "datasets" / dataset / f"keyset_{keyset}.json"
        with open(keyset_path) as f:
            split = json.load(f)

        test_users = set(split["test"])

        merged = None

        for method in METHODS:
            path = (
                ROOT
                / "results"
                / dataset
                / f"{method}_keyset{keyset}_detailed_results.csv"
            )

            df = pd.read_csv(path, dtype={"user": str})

            # The detailed result file should now contain test users only
            assert set(df["user"]) == test_users, (
                f"{dataset} keyset {keyset} {method}: "
                f"{len(df)} result users vs {len(test_users)} test users"
            )

            method_df = df[["user", "nDCG@5"]].rename(
                columns={"nDCG@5": method}
            )

            if merged is None:
                merged = method_df
            else:
                merged = merged.merge(method_df, on="user", how="inner")

        assert len(merged) == len(test_users)

        merged.insert(0, "keyset", keyset)
        merged.insert(0, "dataset", dataset)

        all_rows.append(merged)


master = pd.concat(all_rows, ignore_index=True)

assert not master.duplicated(["dataset", "keyset", "user"]).any()

output_dir = ROOT / "analysis" / "data"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "master_performance.csv"
master.to_csv(output_path, index=False)


print("\nMaster table shape:")
print(master.shape)

print("\nRows by dataset/keyset:")
print(
    master.groupby(["dataset", "keyset"])
    .size()
    .unstack()
)

print("\nMean nDCG@5:")
print(
    master.groupby("dataset")[METHODS]
    .mean()
    .round(4)
)

print(f"\nSaved to {output_path}")