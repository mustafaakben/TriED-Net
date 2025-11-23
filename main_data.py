#!/usr/bin/env python3
"""
benc_real.py

Quick runner to evaluate TriED-Net and baselines on a real tabular dataset.
Usage (classification):
    python benc_real.py --data_path /path/data.csv --target label --task classification --output_dir outputs_real

Usage (regression):
    python benc_real.py --data_path /path/data.csv --target target --task regression --output_dir outputs_real

Notes:
- Assumes all non-target columns are numeric. If you have categoricals, one-hot encode them beforehand.
- Multilabel not supported here.
- Saves a metrics JSON in output_dir and prints ranked tables.
"""

import argparse
import json
import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, ExtraTreesClassifier

import torch
import torch.nn.functional as F

from benchmark_cognitive_models import (
    TriEDNet,
    TriEDConfig,
    train_tried,
    eval_tried,
    set_seeds,
    get_device,
)


def load_data(path: str, target: str, task: str, test_size: float, val_size: float, seed: int):
    df = pd.read_csv(path)
    assert target in df.columns, f"Target column '{target}' not found."
    y_raw = df[target].values
    X = df.drop(columns=[target]).values

    if task == "classification":
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        classes = le.classes_
    else:
        y = y_raw.astype(float)
        classes = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y if task == "classification" else None
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=seed,
        stratify=y_train if task == "classification" else None
    )
    return (X_train, X_val, X_test, y_train, y_val, y_test, classes)


def run_baselines(task: str,
                  X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    results = []

    def add_row(name, metrics):
        row = {"Model": name}
        row.update(metrics)
        results.append(row)

    if task == "classification":
        scaler = StandardScaler().fit(X_train)
        Xtr_s = scaler.transform(X_train)
        Xte_s = scaler.transform(X_test)

        svm = SVC(kernel="rbf", probability=False, random_state=0)
        svm.fit(Xtr_s, y_train)
        yhat = svm.predict(Xte_s)
        add_row("SVM", {"Accuracy": accuracy_score(y_test, yhat), "F1": f1_score(y_test, yhat, average="macro")})

        rf = RandomForestClassifier(n_estimators=600, random_state=0, n_jobs=-1)
        rf.fit(X_train, y_train)
        yhat = rf.predict(X_test)
        add_row("RandomForest", {"Accuracy": accuracy_score(y_test, yhat), "F1": f1_score(y_test, yhat, average="macro")})

        gbc = GradientBoostingClassifier(random_state=0)
        gbc.fit(X_train, y_train)
        yhat = gbc.predict(X_test)
        add_row("GradientBoosting", {"Accuracy": accuracy_score(y_test, yhat), "F1": f1_score(y_test, yhat, average="macro")})

        et = ExtraTreesClassifier(n_estimators=800, random_state=0, n_jobs=-1)
        et.fit(X_train, y_train)
        yhat = et.predict(X_test)
        add_row("ExtraTrees", {"Accuracy": accuracy_score(y_test, yhat), "F1": f1_score(y_test, yhat, average="macro")})

    else:
        scaler = StandardScaler().fit(X_train)
        Xtr_s = scaler.transform(X_train)
        Xte_s = scaler.transform(X_test)

        svr = SVR(kernel="rbf")
        svr.fit(Xtr_s, y_train)
        yhat = svr.predict(Xte_s)
        add_row("SVR", {"MSE": mean_squared_error(y_test, yhat), "R2": r2_score(y_test, yhat)})

        rfr = RandomForestRegressor(n_estimators=600, random_state=0, n_jobs=-1)
        rfr.fit(X_train, y_train)
        yhat = rfr.predict(X_test)
        add_row("RandomForestRegressor", {"MSE": mean_squared_error(y_test, yhat), "R2": r2_score(y_test, yhat)})

        gbr = GradientBoostingRegressor(random_state=0, n_estimators=300, learning_rate=0.05)
        gbr.fit(X_train, y_train)
        yhat = gbr.predict(X_test)
        add_row("GradientBoostingRegressor", {"MSE": mean_squared_error(y_test, yhat), "R2": r2_score(y_test, yhat)})

    return pd.DataFrame(results)


def make_tried_config(task: str, input_dim: int, n_classes: int) -> TriEDConfig:
    if task == "classification":
        return TriEDConfig(
            input_dim=input_dim,
            task="classification",
            n_classes=n_classes,
            hidden_dim=256,
            T_max=0,  # strong MLP head
            dropout=0.1,
            use_prototypes=False,
            use_symbolizer=False,
            use_align=False,
            use_investment=False,
            use_drift=False,
            learn_step_sizes=False,
            lr=3e-3,
            epochs=80,
            full_epochs=80,
            batch_size=128,
            weight_decay=5e-5,
            label_smoothing=0.0,
            cosine_lr=True
        )
    else:
        return TriEDConfig(
            input_dim=input_dim,
            task="regression",
            hidden_dim=96,
            T_max=3,
            base_step_size=0.03,
            use_prototypes=False,
            use_symbolizer=False,
            use_align=False,
            use_investment=True,
            use_drift=True,
            learn_step_sizes=True,
            lam_h=5e-4,
            lr=3e-3,
            epochs=50,
            full_epochs=50,
            batch_size=128,
            weight_decay=2e-4,
            cosine_lr=True,
            latent_smooth_lam=1e-3
        )


def triED_with_teacher(task: str,
                       Xtr: np.ndarray, ytr: np.ndarray,
                       Xva: np.ndarray, yva: np.ndarray,
                       Xte: np.ndarray, yte: np.ndarray,
                       cfg: TriEDConfig,
                       device: torch.device):
    # Teacher probabilities / predictions as features
    def augment(Xtr_np, Xva_np, Xte_np):
        Xtr_aug, Xva_aug, Xte_aug = Xtr_np, Xva_np, Xte_np
        if task == "classification":
            rf = RandomForestClassifier(n_estimators=800, random_state=cfg.lr.__hash__() % 1000, n_jobs=-1)
            rf.fit(Xtr_np, ytr)
            Xtr_aug = np.concatenate([Xtr_aug, rf.predict_proba(Xtr_np)], axis=1)
            Xva_aug = np.concatenate([Xva_aug, rf.predict_proba(Xva_np)], axis=1)
            Xte_aug = np.concatenate([Xte_aug, rf.predict_proba(Xte_np)], axis=1)
        else:
            gbr = GradientBoostingRegressor(random_state=cfg.lr.__hash__() % 1000, n_estimators=300, learning_rate=0.05)
            gbr.fit(Xtr_np, ytr)
            pred = lambda X: gbr.predict(X)[:, None]
            Xtr_aug = np.concatenate([Xtr_aug, pred(Xtr_np)], axis=1)
            Xva_aug = np.concatenate([Xva_aug, pred(Xva_np)], axis=1)
            Xte_aug = np.concatenate([Xte_aug, pred(Xte_np)], axis=1)
        return Xtr_aug, Xva_aug, Xte_aug

    Xtr_aug, Xva_aug, Xte_aug = augment(Xtr, Xva, Xte)
    cfg = cfg.__class__(**{**cfg.__dict__, "input_dim": Xtr_aug.shape[1]})
    model = TriEDNet(cfg).to(device)
    scaler = StandardScaler().fit(np.vstack([Xtr_aug, Xva_aug]))
    fit = train_tried(model, cfg, Xtr_aug, ytr, Xva_aug, yva, device=device, epochs=cfg.epochs, scaler=scaler)
    model.load_state_dict(fit["best_state"])
    metrics = eval_tried(model, cfg, Xte_aug, yte, scaler=fit["scaler"], device=device)
    return metrics, model, cfg, fit["scaler"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Path to CSV file.")
    parser.add_argument("--target", required=True, help="Target column name.")
    parser.add_argument("--task", choices=["classification", "regression"], required=True)
    parser.add_argument("--output_dir", default="outputs_real")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seeds(args.seed)
    device = get_device()
    print(f"Device: {device}")

    Xtr, Xva, Xte, ytr, yva, yte, classes = load_data(
        args.data_path, args.target, args.task, args.test_size, args.val_size, args.seed
    )

    df_base = run_baselines(args.task, Xtr, ytr, Xva, yva, Xte, yte)
    print("\nBaselines:")
    print(df_base.sort_values(df_base.columns[1], ascending=False))

    cfg = make_tried_config(args.task, Xtr.shape[1], n_classes=len(classes) if classes is not None else 1)
    tried_metrics, model, scaler = triED_with_teacher(
        args.task, Xtr, ytr, Xva, yva, Xte, yte, cfg, device
    )
    print("\nTriED metrics:", tried_metrics)

    # Ensemble for classification
    ens_metrics = None
    if args.task == "classification":
        rf = RandomForestClassifier(n_estimators=800, random_state=args.seed, n_jobs=-1)
        rf.fit(np.vstack([Xtr, Xva]), np.concatenate([ytr, yva]))
        rf_probs = rf.predict_proba(Xte)
        Xt = torch.from_numpy(scaler.transform(np.concatenate([Xte, rf_probs], axis=1))).float().to(device)
        with torch.no_grad():
            logits, _ = model(Xt)
            tried_probs = F.softmax(logits, dim=-1).cpu().numpy()
        probs_mix = 0.5 * tried_probs + 0.5 * rf_probs
        yhat = probs_mix.argmax(axis=1)
        ens_metrics = {"Model": "TriED+RF", "Accuracy": accuracy_score(yte, yhat), "F1": f1_score(yte, yhat, average="macro")}
        print("\nEnsemble metrics:", ens_metrics)

    # Save metrics
    out = {
        "baselines": df_base.to_dict(orient="records"),
        "tried": tried_metrics,
        "ensemble": ens_metrics
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved metrics to {os.path.join(args.output_dir, 'metrics.json')}")


if __name__ == "__main__":
    main()
