#!/usr/bin/env python3
# Assignment 2 - Task X and Task Y
# Roll number: 25CS60R41

from pathlib import Path
import csv
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "TaskXY" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INTERPRETABLE_NAMES = [
    "age_bin",
    "balance_bin",
    "housing_bin",
    "edu_bin",
    "marital_bin",
    "loan_bin",
]


def sigmoid(values):
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def add_bias(X):
    ones = np.ones((X.shape[0], 1), dtype=float)
    return np.hstack([ones, X])


def per_sample_loss(w, X, y):
    probs = sigmoid(X @ w)
    eps = 1e-12
    return -(y * np.log(probs + eps) + (1.0 - y) * np.log(1.0 - probs + eps))


def mean_loss_and_grad(w, X, y):
    probs = sigmoid(X @ w)
    eps = 1e-12
    losses = -(y * np.log(probs + eps) + (1.0 - y) * np.log(1.0 - probs + eps))
    grad = (X.T @ (probs - y)) / X.shape[0]
    return float(np.mean(losses)), grad


def train_basic_logreg(X, y, iters=2500, lr=0.1, reg=1e-4):
    w = np.zeros(X.shape[1], dtype=float)
    for _ in range(iters):
        _, grad = mean_loss_and_grad(w, X, y)
        w = w - lr * (grad + reg * w)
    return w


def project_covariance_constraint(w, g, c_limit):
    denom = float(np.dot(g, g))
    if denom < 1e-14:
        return w

    current = float(np.dot(g, w))
    if current > c_limit:
        return w - ((current - c_limit) / denom) * g
    if current < -c_limit:
        return w - ((current + c_limit) / denom) * g
    return w


def train_model_accurate(X, y, z, c_limit=0.8, iters=3000, lr=0.08, reg=1e-4):
    w = train_basic_logreg(X, y, iters=1000, lr=0.08, reg=reg)
    z_centered = z - np.mean(z)
    cov_grad = np.mean(z_centered[:, None] * X, axis=0)

    for _ in range(iters):
        _, grad = mean_loss_and_grad(w, X, y)
        w = w - lr * (grad + reg * w)
        w = project_covariance_constraint(w, cov_grad, c_limit)

    return w


def train_model_fair(X, y, z, gamma=0.0, iters=1500, lr_w=0.05, lr_lambda=0.1, reg=1e-4, lambda_cap=200.0):
    theta_star = train_basic_logreg(X, y, iters=2500, lr=0.1, reg=reg)
    loss_star = per_sample_loss(theta_star, X, y)
    rhs = (1.0 + gamma) * loss_star

    w = theta_star.copy()
    lambda_vec = np.zeros(X.shape[0], dtype=float)

    z_centered = z - np.mean(z)
    cov_grad = np.mean(z_centered[:, None] * X, axis=0)

    for _ in range(iters):
        scores = X @ w
        probs = sigmoid(scores)
        eps = 1e-12
        losses = -(y * np.log(probs + eps) + (1.0 - y) * np.log(1.0 - probs + eps))
        violations = losses - rhs

        cov_value = float(np.mean(z_centered * scores))
        fair_grad = (cov_value / np.sqrt(cov_value * cov_value + 1e-12)) * cov_grad
        loss_grads = (probs - y)[:, None] * X
        constraint_grad = np.mean(lambda_vec[:, None] * loss_grads, axis=0)

        w = w - lr_w * (fair_grad + constraint_grad + reg * w)
        lambda_vec = np.clip(lambda_vec + lr_lambda * violations, 0.0, lambda_cap)

    return w


def predict_proba(w, X):
    return sigmoid(X @ w)


def predict_label(w, X):
    return (predict_proba(w, X) >= 0.5).astype(int)


def accuracy_score(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def covariance_value(w, X, z):
    z_centered = z - np.mean(z)
    return float(np.mean(z_centered * (X @ w)))


def p_rule_value(preds, z):
    mask0 = z == 0
    mask1 = z == 1

    rate0 = float(preds[mask0].mean()) if np.any(mask0) else 0.0
    rate1 = float(preds[mask1].mean()) if np.any(mask1) else 0.0

    eps = 1e-12
    if rate0 <= eps and rate1 <= eps:
        return 100.0
    if rate0 <= eps or rate1 <= eps:
        return 0.0
    return float(min(rate1 / rate0, rate0 / rate1) * 100.0)


def load_bank_dataset(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter=";")
        next(reader)
        for row in reader:
            rows.append(row)

    job_cats = sorted([
        "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
        "retired", "self-employed", "services", "student", "technician",
        "unemployed", "unknown",
    ])
    contact_cats = sorted(["cellular", "telephone", "unknown"])
    month_cats = sorted(["apr", "aug", "dec", "feb", "jan", "jul", "jun", "mar", "may", "nov", "oct", "sep"])
    poutcome_cats = sorted(["failure", "other", "success", "unknown"])

    job_map = {name: idx for idx, name in enumerate(job_cats)}
    contact_map = {name: idx for idx, name in enumerate(contact_cats)}
    month_map = {name: idx for idx, name in enumerate(month_cats)}
    poutcome_map = {name: idx for idx, name in enumerate(poutcome_cats)}

    n_rows = len(rows)

    interpretable = np.zeros((n_rows, 6), dtype=float)
    non_interpretable = np.zeros((n_rows, 37), dtype=float)
    labels = np.zeros(n_rows, dtype=float)

    for idx, row in enumerate(rows):
        age = int(row[0])
        job = row[1].strip()
        marital = row[2].strip()
        education = row[3].strip()
        default = row[4].strip()
        balance = int(row[5])
        housing = row[6].strip()
        loan = row[7].strip()
        contact = row[8].strip()
        day = int(row[9])
        month = row[10].strip()
        duration = int(row[11])
        campaign = int(row[12])
        pdays = int(row[13])
        previous = int(row[14])
        poutcome = row[15].strip()
        target = row[16].strip()

        interpretable[idx, 0] = 1.0 if age >= 39 else 0.0
        interpretable[idx, 1] = 1.0 if balance > 448 else 0.0
        interpretable[idx, 2] = 1.0 if housing == "yes" else 0.0
        interpretable[idx, 3] = 1.0 if education in {"primary", "secondary"} else 0.0
        interpretable[idx, 4] = 1.0 if marital == "married" else 0.0
        interpretable[idx, 5] = 1.0 if loan == "yes" else 0.0

        cursor = 0
        non_interpretable[idx, cursor] = 1.0 if default == "yes" else 0.0
        cursor += 1

        non_interpretable[idx, cursor + job_map[job]] = 1.0
        cursor += len(job_cats)

        non_interpretable[idx, cursor + contact_map[contact]] = 1.0
        cursor += len(contact_cats)

        non_interpretable[idx, cursor] = float(day)
        cursor += 1

        non_interpretable[idx, cursor + month_map[month]] = 1.0
        cursor += len(month_cats)

        non_interpretable[idx, cursor] = float(duration)
        cursor += 1

        non_interpretable[idx, cursor] = float(campaign)
        cursor += 1

        non_interpretable[idx, cursor] = float(pdays)
        cursor += 1

        non_interpretable[idx, cursor] = float(previous)
        cursor += 1

        non_interpretable[idx, cursor + poutcome_map[poutcome]] = 1.0
        labels[idx] = 1.0 if target == "yes" else 0.0

    return interpretable, non_interpretable, labels


def train_test_split(num_rows, seed=42):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_rows)
    split = int(0.8 * num_rows)
    return perm[:split], perm[split:]


def zscore_fit_transform(train_part, full_part):
    mean = np.mean(train_part, axis=0)
    std = np.std(train_part, axis=0)
    std = np.where(std < 1e-10, 1.0, std)
    return (full_part - mean) / std, mean, std


def select_instances(pred_acc, pred_fair):
    disagree = np.where(pred_acc != pred_fair)[0]
    agree = np.where(pred_acc == pred_fair)[0]

    if len(disagree) < 2:
        raise RuntimeError("Need at least 2 disagreement instances in the test set.")
    if len(agree) < 3:
        raise RuntimeError("Need at least 3 agreement instances in the test set.")

    chosen = np.concatenate([disagree[:2], agree[:3]])
    status = np.array(["disagree", "disagree", "agree", "agree", "agree"])
    return chosen, status


def save_text_summary(path, metrics):
    with open(path, "w", encoding="utf-8") as handle:
        for key, value in metrics:
            handle.write(f"{key}: {value}\n")


def main():
    print("=" * 60)
    print("Assignment 2 - Task X and Task Y")
    print("=" * 60)

    csv_path = BASE_DIR / "bank+marketing" / "bank" / "bank-full.csv"
    interpretable, non_interpretable, y_all = load_bank_dataset(csv_path)

    train_idx, test_idx = train_test_split(len(y_all), seed=42)
    non_interpretable_norm, norm_mean, norm_std = zscore_fit_transform(
        non_interpretable[train_idx], non_interpretable
    )

    X_full = np.hstack([interpretable, non_interpretable_norm])
    X_train = X_full[train_idx]
    X_test = X_full[test_idx]
    y_train = y_all[train_idx]
    y_test = y_all[test_idx]

    z_train = interpretable[train_idx, 0]
    z_test = interpretable[test_idx, 0]

    X_train_b = add_bias(X_train)
    X_test_b = add_bias(X_test)

    print("Training Model-Accurate with c = 0.8")
    w_accurate = train_model_accurate(X_train_b, y_train, z_train, c_limit=0.8)

    print("Training Model-Fair with gamma = 0.0")
    w_fair = train_model_fair(X_train_b, y_train, z_train, gamma=0.0)

    pred_acc = predict_label(w_accurate, X_test_b)
    pred_fair = predict_label(w_fair, X_test_b)
    prob_acc = predict_proba(w_accurate, X_test_b)
    prob_fair = predict_proba(w_fair, X_test_b)

    metrics = [
        ("model_accurate_test_accuracy", f"{accuracy_score(y_test, pred_acc):.6f}"),
        ("model_fair_test_accuracy", f"{accuracy_score(y_test, pred_fair):.6f}"),
        ("model_accurate_test_covariance_abs", f"{abs(covariance_value(w_accurate, X_test_b, z_test)):.6f}"),
        ("model_fair_test_covariance_abs", f"{abs(covariance_value(w_fair, X_test_b, z_test)):.6f}"),
        ("model_accurate_test_p_rule", f"{p_rule_value(pred_acc, z_test):.6f}"),
        ("model_fair_test_p_rule", f"{p_rule_value(pred_fair, z_test):.6f}"),
    ]

    for key, value in metrics:
        print(f"{key}: {value}")

    selected_local, selected_status = select_instances(pred_acc, pred_fair)
    selected_global = test_idx[selected_local]

    selected_x_full = X_test_b[selected_local]
    selected_x_without_bias = X_test[selected_local]
    selected_x_interp = interpretable[test_idx][selected_local]
    selected_y = y_test[selected_local]
    selected_prob_acc = prob_acc[selected_local]
    selected_prob_fair = prob_fair[selected_local]
    selected_pred_acc = pred_acc[selected_local]
    selected_pred_fair = pred_fair[selected_local]

    print("\nSelected instances:")
    for idx in range(len(selected_local)):
        print(
            "  local_test_idx={} global_idx={} status={} true={} pred_acc={} pred_fair={}".format(
                int(selected_local[idx]),
                int(selected_global[idx]),
                selected_status[idx],
                int(selected_y[idx]),
                int(selected_pred_acc[idx]),
                int(selected_pred_fair[idx]),
            )
        )

    n_selected = selected_x_full.shape[0]
    n_perturb = 10
    n_interp = len(INTERPRETABLE_NAMES)

    rng = np.random.default_rng(123)
    perturbed_interp = rng.integers(0, 2, size=(n_selected, n_perturb, n_interp)).astype(float)

    reconstructed = np.repeat(selected_x_full[:, None, :], repeats=n_perturb, axis=1)
    reconstructed[:, :, 1:1 + n_interp] = perturbed_interp

    probs_accurate = sigmoid(reconstructed @ w_accurate)
    probs_fair = sigmoid(reconstructed @ w_fair)

    np.save(OUT_DIR / "model_accurate_w.npy", w_accurate)
    np.save(OUT_DIR / "model_fair_w.npy", w_fair)
    np.save(OUT_DIR / "train_indices.npy", train_idx)
    np.save(OUT_DIR / "test_indices.npy", test_idx)
    np.save(OUT_DIR / "norm_mean.npy", norm_mean)
    np.save(OUT_DIR / "norm_std.npy", norm_std)
    np.save(OUT_DIR / "X_test_b.npy", X_test_b)
    np.save(OUT_DIR / "y_test.npy", y_test)
    np.save(OUT_DIR / "z_test.npy", z_test)
    np.save(OUT_DIR / "selected_local_indices.npy", selected_local)
    np.save(OUT_DIR / "selected_indices.npy", selected_global)
    np.save(OUT_DIR / "selected_x_full.npy", selected_x_full)
    np.save(OUT_DIR / "selected_x_without_bias.npy", selected_x_without_bias)
    np.save(OUT_DIR / "selected_x_interp.npy", selected_x_interp)
    np.save(OUT_DIR / "selected_y.npy", selected_y)
    np.save(OUT_DIR / "preds_accurate_sel.npy", selected_pred_acc)
    np.save(OUT_DIR / "preds_fair_sel.npy", selected_pred_fair)
    np.save(OUT_DIR / "probs_accurate_sel.npy", selected_prob_acc)
    np.save(OUT_DIR / "probs_fair_sel.npy", selected_prob_fair)
    np.save(OUT_DIR / "perturbed_interp.npy", perturbed_interp)
    np.save(OUT_DIR / "reconstructed_full.npy", reconstructed)
    np.save(OUT_DIR / "probs_accurate.npy", probs_accurate)
    np.save(OUT_DIR / "probs_fair.npy", probs_fair)

    with open(OUT_DIR / "selected_instance_summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "picked_order",
            "selected_local_test_index",
            "selected_global_index",
            "selection_status",
            "true_label",
            "pred_model_accurate",
            "pred_model_fair",
            "prob_model_accurate",
            "prob_model_fair",
        ])
        for idx in range(n_selected):
            writer.writerow([
                idx,
                int(selected_local[idx]),
                int(selected_global[idx]),
                selected_status[idx],
                int(selected_y[idx]),
                int(selected_pred_acc[idx]),
                int(selected_pred_fair[idx]),
                float(selected_prob_acc[idx]),
                float(selected_prob_fair[idx]),
            ])

    with open(OUT_DIR / "interpretable_feature_names.txt", "w", encoding="utf-8") as handle:
        for name in INTERPRETABLE_NAMES:
            handle.write(name + "\n")

    save_text_summary(OUT_DIR / "taskXY_metrics.txt", metrics)

    print("\nTask Y neighborhoods generated:")
    print("  perturbed_interp shape:", perturbed_interp.shape)
    print("  probs_accurate shape  :", probs_accurate.shape)
    print("  probs_fair shape      :", probs_fair.shape)
    print("\nOutputs saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
