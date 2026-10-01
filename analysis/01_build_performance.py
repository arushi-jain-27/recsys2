from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "analysis" / "data"

DATASETS = ["dunnhumby", "instacart", "retailrocket", "sam", "taobao", "valuedshopper", "tmall", "tafeng"]
METHODS = ["recency", "frequency", "tifuknn", "fpmc", "triple2vec", "dnntsp", "cbp", "diffrec"]
KEYSETS = [0, 1, 2]

DATA_DIR.mkdir(parents=True, exist_ok=True)


def build_performance(metric, output_file):
    rows = []

    for dataset in DATASETS:
        for keyset in KEYSETS:
            with open(ROOT / "datasets" / dataset / f"keyset_{keyset}.json") as f:
                test_users = set(map(str, json.load(f)["test"]))

            merged = None

            for method in METHODS:
                path = ROOT / "results" / dataset / f"{method}_keyset{keyset}_detailed_results.csv"
                df = pd.read_csv(path, dtype={"user": str})

                assert not df["user"].duplicated().any(), (
                    f"{dataset} keyset {keyset} {method}: duplicate users"
                )
                assert set(df["user"]) == test_users, (
                    f"{dataset} keyset {keyset} {method}: "
                    f"{len(df)} result users vs {len(test_users)} expected test users"
                )

                method_df = df[["user", metric]].rename(columns={metric: method})
                merged = (
                    method_df
                    if merged is None
                    else merged.merge(method_df, on="user", how="inner", validate="one_to_one")
                )

            assert len(merged) == len(test_users)
            merged.insert(0, "keyset", keyset)
            merged.insert(0, "dataset", dataset)
            rows.append(merged)

    performance = pd.concat(rows, ignore_index=True)
    assert not performance.duplicated(["dataset", "keyset", "user"]).any()

    output_path = DATA_DIR / output_file
    performance.to_csv(output_path, index=False)

    print(f"\n{metric} performance table shape:", performance.shape)
    print("\nRows by dataset/keyset:")
    print(performance.groupby(["dataset", "keyset"]).size().unstack())
    print(f"\nMean {metric}:")
    print(performance.groupby("dataset")[METHODS].mean().round(4))
    print(f"\nSaved to {output_path}")


build_performance("nDCG@5", "performance.csv")
build_performance("recall@5", "performance_recall.csv")