#!/usr/bin/env python3
# Assignment 2 - Task Z (LIME-style local explanations)
# Roll number: 25CS60R41

from pathlib import Path
import csv
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
XY_DIR = BASE_DIR / "TaskXY" / "results"
OUT_DIR = BASE_DIR / "TaskZ" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INTERPRETABLE_NAMES = [
    "age_bin",
    "balance_bin",
    "housing_bin",
    "edu_bin",
    "marital_bin",
    "loan_bin",
]


def hamming_distance_matrix(neighborhood, original):
    return np.sum(np.abs(neighborhood - original[None, :]), axis=1)


def kernel_weights(distances, sigma):
    return np.exp(-(distances ** 2) / (sigma ** 2))


def weighted_linear_regression(X, y, weights, ridge=1e-6):
    X_design = np.hstack([np.ones((X.shape[0], 1), dtype=float), X])
    sqrt_w = np.sqrt(weights)[:, None]
    Xw = X_design * sqrt_w
    yw = y * sqrt_w[:, 0]

    gram = Xw.T @ Xw
    gram = gram + ridge * np.eye(gram.shape[0], dtype=float)
    rhs = Xw.T @ yw

    coeffs = np.linalg.solve(gram, rhs)
    preds = X_design @ coeffs
    mse = float(np.mean((preds - y) ** 2))
    return coeffs, preds, mse


def explain_one_instance(neighborhood, probs, original_interp, sigma):
    distances = hamming_distance_matrix(neighborhood, original_interp)
    weights = kernel_weights(distances, sigma)
    coeffs, preds, mse = weighted_linear_regression(neighborhood, probs, weights)
    return {
        "distances": distances,
        "weights": weights,
        "coeffs": coeffs,
        "local_pred": float(coeffs[0] + np.dot(original_interp, coeffs[1:])),
        "fit_mse": mse,
        "surrogate_preds": preds,
    }


def save_explanation_csv(path, coeffs_matrix):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["instance_id", "bias"] + INTERPRETABLE_NAMES)
        for idx in range(coeffs_matrix.shape[0]):
            writer.writerow([idx] + list(coeffs_matrix[idx]))


def main():
    print("=" * 60)
    print("Assignment 2 - Task Z (LIME-style local explanations)")
    print("=" * 60)

    selected_x_interp = np.load(XY_DIR / "selected_x_interp.npy")
    perturbed_interp = np.load(XY_DIR / "perturbed_interp.npy")
    probs_accurate = np.load(XY_DIR / "probs_accurate.npy")
    probs_fair = np.load(XY_DIR / "probs_fair.npy")
    preds_accurate_sel = np.load(XY_DIR / "preds_accurate_sel.npy")
    preds_fair_sel = np.load(XY_DIR / "preds_fair_sel.npy")
    selected_y = np.load(XY_DIR / "selected_y.npy")
    selected_indices = np.load(XY_DIR / "selected_indices.npy")

    n_instances = selected_x_interp.shape[0]
    sigma = np.sqrt(selected_x_interp.shape[1])

    lime_coeffs_accurate = np.zeros((n_instances, selected_x_interp.shape[1] + 1), dtype=float)
    lime_coeffs_fair = np.zeros((n_instances, selected_x_interp.shape[1] + 1), dtype=float)
    hamming_all = np.zeros((n_instances, perturbed_interp.shape[1]), dtype=float)
    weights_all = np.zeros((n_instances, perturbed_interp.shape[1]), dtype=float)
    local_pred_accurate = np.zeros(n_instances, dtype=float)
    local_pred_fair = np.zeros(n_instances, dtype=float)
    fit_mse_accurate = np.zeros(n_instances, dtype=float)
    fit_mse_fair = np.zeros(n_instances, dtype=float)

    print("Using kernel width sigma = {:.6f}".format(sigma))

    for idx in range(n_instances):
        exp_acc = explain_one_instance(
            perturbed_interp[idx],
            probs_accurate[idx],
            selected_x_interp[idx],
            sigma,
        )
        exp_fair = explain_one_instance(
            perturbed_interp[idx],
            probs_fair[idx],
            selected_x_interp[idx],
            sigma,
        )

        lime_coeffs_accurate[idx] = exp_acc["coeffs"]
        lime_coeffs_fair[idx] = exp_fair["coeffs"]
        hamming_all[idx] = exp_acc["distances"]
        weights_all[idx] = exp_acc["weights"]
        local_pred_accurate[idx] = exp_acc["local_pred"]
        local_pred_fair[idx] = exp_fair["local_pred"]
        fit_mse_accurate[idx] = exp_acc["fit_mse"]
        fit_mse_fair[idx] = exp_fair["fit_mse"]

        status = "DISAGREE" if idx < 2 else "AGREE"
        print(
            "\nInstance {} [{}] global_idx={} true={} pred_acc={} pred_fair={}".format(
                idx,
                status,
                int(selected_indices[idx]),
                int(selected_y[idx]),
                int(preds_accurate_sel[idx]),
                int(preds_fair_sel[idx]),
            )
        )
        for feat_idx, feat_name in enumerate(INTERPRETABLE_NAMES):
            print(
                "  {} -> coeff_acc={:.6f}, coeff_fair={:.6f}".format(
                    feat_name,
                    lime_coeffs_accurate[idx, feat_idx + 1],
                    lime_coeffs_fair[idx, feat_idx + 1],
                )
            )
        print(
            "  local_pred_acc={:.6f}, local_pred_fair={:.6f}, mse_acc={:.6f}, mse_fair={:.6f}".format(
                local_pred_accurate[idx],
                local_pred_fair[idx],
                fit_mse_accurate[idx],
                fit_mse_fair[idx],
            )
        )

    np.save(OUT_DIR / "lime_coeffs_accurate.npy", lime_coeffs_accurate)
    np.save(OUT_DIR / "lime_coeffs_fair.npy", lime_coeffs_fair)
    np.save(OUT_DIR / "hamming_distances.npy", hamming_all)
    np.save(OUT_DIR / "kernel_weights.npy", weights_all)
    np.save(OUT_DIR / "local_pred_accurate.npy", local_pred_accurate)
    np.save(OUT_DIR / "local_pred_fair.npy", local_pred_fair)
    np.save(OUT_DIR / "fit_mse_accurate.npy", fit_mse_accurate)
    np.save(OUT_DIR / "fit_mse_fair.npy", fit_mse_fair)

    save_explanation_csv(OUT_DIR / "lime_coeffs_accurate.csv", lime_coeffs_accurate)
    save_explanation_csv(OUT_DIR / "lime_coeffs_fair.csv", lime_coeffs_fair)

    with open(OUT_DIR / "taskZ_lime_summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "instance_id",
            "global_index",
            "true_label",
            "pred_model_accurate",
            "pred_model_fair",
            "local_pred_accurate",
            "local_pred_fair",
            "fit_mse_accurate",
            "fit_mse_fair",
        ])
        for idx in range(n_instances):
            writer.writerow([
                idx,
                int(selected_indices[idx]),
                int(selected_y[idx]),
                int(preds_accurate_sel[idx]),
                int(preds_fair_sel[idx]),
                float(local_pred_accurate[idx]),
                float(local_pred_fair[idx]),
                float(fit_mse_accurate[idx]),
                float(fit_mse_fair[idx]),
            ])

    print("\nSaved LIME-style outputs to:", OUT_DIR)


if __name__ == "__main__":
    main()
