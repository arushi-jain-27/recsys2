from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "analysis" / "data"


performance = pd.read_csv(
    DATA_DIR / "master_performance.csv",
    dtype={"user": str},
)

intrinsic = pd.read_csv(
    DATA_DIR / "user_history_features.csv",
    dtype={"user": str},
)

population = pd.read_csv(
    DATA_DIR / "user_population_features.csv",
    dtype={"user": str},
)


# One feature row per dataset-user
features = intrinsic.merge(
    population,
    on=["dataset", "user"],
    how="inner",
)

# Join fixed user features onto each test appearance
master = performance.merge(
    features,
    on=["dataset", "user"],
    how="left",
)


output_path = DATA_DIR / "master_analysis.csv"
master.to_csv(output_path, index=False)


print("\nShapes:")
print("performance:", performance.shape)
print("intrinsic:", intrinsic.shape)
print("population:", population.shape)
print("features:", features.shape)
print("master:", master.shape)

print("\nRows by dataset/keyset:")
print(
    master.groupby(["dataset", "keyset"])
    .size()
    .unstack()
)

print("\nMissing values in master:")
print(
    master.isna()
    .sum()
    .sort_values(ascending=False)
)

print(f"\nSaved to {output_path}")