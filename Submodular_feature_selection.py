#!/usr/bin/env python3
# Assignment 2 - Task Z (Submodular feature selection)
# Roll number: 25CS60R41

from pathlib import Path
import csv
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
XY_DIR = BASE_DIR / "TaskXY" / "results"
Z_DIR = BASE_DIR / "TaskZ" / "results"
Z_DIR.mkdir(parents=True, exist_ok=True)

INTERPRETABLE_NAMES = [
    "age_bin",
    "balance_bin",
    "housing_bin",
    "edu_bin",
    "marital_bin",
    "loan_bin",
]


def coverage_value(W, selected):
    if not selected:
        return 0.0
    sq_norm = np.sum(W[:, selected] ** 2, axis=1)
    return float(np.sum(np.sqrt(sq_norm)))


def greedy_submodular_selection(W, k=3):
    selected = []
    current_value = 0.0

    for _ in range(k):
        best_feature = None
        best_gain = -1.0

        for feature_idx in range(W.shape[1]):
            if feature_idx in selected:
                continue

            trial_value = coverage_value(W, selected + [feature_idx])
            gain = trial_value - current_value
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx

        selected.append(best_feature)
        current_value += best_gain

    return selected, current_value


def print_instance_rankings(W, model_name, statuses):
    print("\nTop-3 features per selected instance for", model_name)
    for idx in range(W.shape[0]):
        top3 = np.argsort(W[idx])[::-1][:3]
        joined = ", ".join(
            "{}({:.6f})".format(INTERPRETABLE_NAMES[j], W[idx, j]) for j in top3
        )
        print("  Instance {} [{}]: {}".format(idx, statuses[idx], joined))


def save_top_features_csv(path, selected):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "feature_index", "feature_name"])
        for rank, feature_idx in enumerate(selected, start=1):
            writer.writerow([rank, feature_idx, INTERPRETABLE_NAMES[feature_idx]])


def main():
    print("=" * 60)
    print("Assignment 2 - Task Z (Submodular feature selection)")
    print("=" * 60)

    lime_coeffs_accurate = np.load(Z_DIR / "lime_coeffs_accurate.npy")
    lime_coeffs_fair = np.load(Z_DIR / "lime_coeffs_fair.npy")
    selected_indices = np.load(XY_DIR / "selected_indices.npy")

    W_accurate = np.abs(lime_coeffs_accurate[:, 1:])
    W_fair = np.abs(lime_coeffs_fair[:, 1:])
    statuses = ["DISAGREE", "DISAGREE", "AGREE", "AGREE", "AGREE"]

    print("Importance matrix shape (Model-Accurate):", W_accurate.shape)
    print("Importance matrix shape (Model-Fair):", W_fair.shape)

    print_instance_rankings(W_accurate, "Model-Accurate", statuses)
    print_instance_rankings(W_fair, "Model-Fair", statuses)

    top_accurate, cov_accurate = greedy_submodular_selection(W_accurate, k=3)
    top_fair, cov_fair = greedy_submodular_selection(W_fair, k=3)

    print("\nGreedy submodular top-3 features for Model-Accurate:")
    for rank, feature_idx in enumerate(top_accurate, start=1):
        print("  {}. {} (index {})".format(rank, INTERPRETABLE_NAMES[feature_idx], feature_idx))
    print("Total coverage:", "{:.6f}".format(cov_accurate))

    print("\nGreedy submodular top-3 features for Model-Fair:")
    for rank, feature_idx in enumerate(top_fair, start=1):
        print("  {}. {} (index {})".format(rank, INTERPRETABLE_NAMES[feature_idx], feature_idx))
    print("Total coverage:", "{:.6f}".format(cov_fair))

    np.save(Z_DIR / "submodular_top_accurate.npy", np.array(top_accurate, dtype=int))
    np.save(Z_DIR / "submodular_top_fair.npy", np.array(top_fair, dtype=int))

    save_top_features_csv(Z_DIR / "submodular_top_accurate.csv", top_accurate)
    save_top_features_csv(Z_DIR / "submodular_top_fair.csv", top_fair)

    with open(Z_DIR / "taskZ_submodular_summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "model_name",
            "coverage",
            "selected_global_indices",
            "top_feature_1",
            "top_feature_2",
            "top_feature_3",
        ])
        writer.writerow([
            "Model-Accurate",
            float(cov_accurate),
            " ".join(str(int(x)) for x in selected_indices),
            INTERPRETABLE_NAMES[top_accurate[0]],
            INTERPRETABLE_NAMES[top_accurate[1]],
            INTERPRETABLE_NAMES[top_accurate[2]],
        ])
        writer.writerow([
            "Model-Fair",
            float(cov_fair),
            " ".join(str(int(x)) for x in selected_indices),
            INTERPRETABLE_NAMES[top_fair[0]],
            INTERPRETABLE_NAMES[top_fair[1]],
            INTERPRETABLE_NAMES[top_fair[2]],
        ])

    print("\nSaved submodular outputs to:", Z_DIR)


if __name__ == "__main__":
    main()
