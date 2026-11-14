#!/usr/bin/env python3
"""Participant-held-out EEG-to-kinematics regression with condition-stratified scoring.

Input is one row per audited matched trial. Predictor columns must be named eeg_*;
outcome columns and condition labels are never admitted as predictors.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ID_COLUMNS = {"participant_id", "trial_id", "condition"}


def _metric(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)
    return {
        "n_trials": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def _bootstrap_mean_ci(values: Iterable[float], seed: int = 20261114,
                       n_boot: int = 5000) -> dict[str, float | int | None]:
    """Percentile CI over participant-level values, not trial-level pseudo-replicates."""
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n_participants": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    rng = np.random.default_rng(seed)
    draws = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return {
        "n_participants": int(len(x)),
        "mean": float(x.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
    }


def validate_table(frame: pd.DataFrame, target: str, feature_prefix: str = "eeg_") -> list[str]:
    required = ID_COLUMNS | {target}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    features = [c for c in frame.columns if c.startswith(feature_prefix)]
    if not features:
        raise ValueError(f"No predictor columns begin with {feature_prefix!r}.")
    if target in features or any(c in ID_COLUMNS for c in features):
        raise ValueError("Identifiers, condition and target must not be predictors.")
    if frame[list(required)].isna().any().any():
        raise ValueError("Participant, trial, condition and target columns must be complete.")
    if frame[features].isna().any().any():
        raise ValueError("EEG feature columns contain missing values; resolve them within the audit plan.")
    if not np.isfinite(frame[features].to_numpy(dtype=float)).all():
        raise ValueError("EEG features contain non-finite values.")
    if not np.isfinite(frame[target].to_numpy(dtype=float)).all():
        raise ValueError("Target contains non-finite values.")
    if frame.duplicated(["participant_id", "trial_id"]).any():
        examples = frame.loc[frame.duplicated(["participant_id", "trial_id"], keep=False),
                             ["participant_id", "trial_id"]].head().to_dict("records")
        raise ValueError(f"Duplicate participant/trial keys: {examples}")
    if frame["participant_id"].nunique() < 3:
        raise ValueError("At least three participants are required for LOPO plus grouped inner tuning.")
    return features


def run_lopo(frame: pd.DataFrame, target: str, output_dir: Path,
             feature_prefix: str = "eeg_", seed: int = 20261114) -> dict:
    features = validate_table(frame, target, feature_prefix)
    frame = frame.copy()
    frame["participant_id"] = frame["participant_id"].astype(str)
    frame["condition"] = frame["condition"].astype(str).str.strip().str.lower()
    participants = sorted(frame["participant_id"].unique())
    conditions = sorted(frame["condition"].unique())
    if len(conditions) < 2:
        raise ValueError("Condition-stratified analysis requires at least two verified conditions.")

    prediction_frames = []
    fold_rows = []
    alpha_grid = {"ridge__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}

    for held_out in participants:
        train = frame.loc[frame.participant_id != held_out].copy()
        test = frame.loc[frame.participant_id == held_out].copy()
        train_groups = train["participant_id"].to_numpy()
        n_train_groups = np.unique(train_groups).size
        if n_train_groups < 2:
            raise ValueError(f"Insufficient training participants in fold {held_out}.")
        cv = GroupKFold(n_splits=min(5, n_train_groups))
        estimator = Pipeline([
            ("scale", StandardScaler()),
            ("ridge", Ridge()),
        ])
        search = GridSearchCV(
            estimator, param_grid=alpha_grid, scoring="neg_mean_absolute_error",
            cv=cv, refit=True, n_jobs=1, error_score="raise",
        )
        X_train = train[features].to_numpy(dtype=float)
        y_train = train[target].to_numpy(dtype=float)
        X_test = test[features].to_numpy(dtype=float)
        search.fit(X_train, y_train, groups=train_groups)
        pred = search.predict(X_test)
        baseline = np.full(len(test), float(np.mean(y_train)))
        fold = test[["participant_id", "trial_id", "condition", target]].copy()
        fold["prediction_ridge"] = pred
        fold["prediction_training_mean"] = baseline
        fold["held_out_participant"] = held_out
        fold["selected_alpha"] = float(search.best_params_["ridge__alpha"])
        prediction_frames.append(fold)

        model_metrics = _metric(test[target], pred)
        baseline_metrics = _metric(test[target], baseline)
        fold_rows.append({
            "participant_id": held_out,
            "selected_alpha": float(search.best_params_["ridge__alpha"]),
            "ridge_mae": model_metrics["mae"],
            "ridge_rmse": model_metrics["rmse"],
            "mean_baseline_mae": baseline_metrics["mae"],
            "mean_baseline_rmse": baseline_metrics["rmse"],
            "n_trials": len(test),
        })

    predictions = pd.concat(prediction_frames, ignore_index=True)
    fold_metrics = pd.DataFrame(fold_rows)
    y = predictions[target].to_numpy(dtype=float)
    ridge = predictions["prediction_ridge"].to_numpy(dtype=float)
    baseline = predictions["prediction_training_mean"].to_numpy(dtype=float)
    overall = {
        "target": target,
        "n_participants": len(participants),
        "n_trials": len(predictions),
        "n_features": len(features),
        "ridge": _metric(y, ridge),
        "training_mean_baseline": _metric(y, baseline),
        "participant_macro_mae": _bootstrap_mean_ci(fold_metrics["ridge_mae"], seed),
        "baseline_participant_macro_mae": _bootstrap_mean_ci(fold_metrics["mean_baseline_mae"], seed + 1),
    }

    condition_rows = []
    participant_condition = []
    for condition in conditions:
        subset = predictions.loc[predictions.condition == condition]
        if subset.empty:
            continue
        overall_condition = _metric(subset[target], subset["prediction_ridge"])
        baseline_condition = _metric(subset[target], subset["prediction_training_mean"])
        participant_maes = []
        for participant, group in subset.groupby("participant_id"):
            m = _metric(group[target], group["prediction_ridge"])
            participant_maes.append(m["mae"])
            participant_condition.append({
                "participant_id": participant, "condition": condition,
                "mae": m["mae"], "rmse": m["rmse"], "n_trials": m["n_trials"],
            })
        condition_rows.append({
            "condition": condition,
            **{f"ridge_{k}": v for k, v in overall_condition.items()},
            "baseline_mae": baseline_condition["mae"],
            "participant_macro_mae": _bootstrap_mean_ci(participant_maes, seed + len(condition_rows) + 2),
        })

    pc = pd.DataFrame(participant_condition)
    condition_delta = None
    if len(conditions) == 2 and not pc.empty:
        pivot = pc.pivot(index="participant_id", columns="condition", values="mae").dropna()
        if len(pivot):
            # Positive delta means higher error in the second alphabetically sorted condition.
            delta = pivot[conditions[1]] - pivot[conditions[0]]
            condition_delta = {
                "contrast": f"{conditions[1]} minus {conditions[0]}",
                "interpretation": "positive values indicate higher participant-level MAE in the second condition",
                **_bootstrap_mean_ci(delta, seed + 10),
            }

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "lopo_predictions.csv", index=False)
    fold_metrics.to_csv(output_dir / "participant_fold_metrics.csv", index=False)
    pc.to_csv(output_dir / "condition_participant_metrics.csv", index=False)
    pd.DataFrame(condition_rows).to_csv(output_dir / "condition_metrics.csv", index=False)
    summary = {
        "status": "EXECUTED_ON_SUPPLIED_DERIVED_TABLE",
        "warning": "This script does not audit raw EEG or establish that input features are leakage-free. That provenance must be demonstrated separately.",
        "overall": overall,
        "conditions": condition_rows,
        "condition_error_contrast": condition_delta,
        "feature_columns": features,
        "participants": participants,
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "seed": seed,
        "validation": "Outer leave-one-participant-out; inner GroupKFold; scaler and ridge tuning fitted only on training participants.",
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, help="Audited one-row-per-trial feature/target table.")
    parser.add_argument("--target", default="movement_duration_ms")
    parser.add_argument("--output-dir", default="reports/lopo_experiment")
    parser.add_argument("--feature-prefix", default="eeg_")
    parser.add_argument("--seed", type=int, default=20261114)
    args = parser.parse_args()
    frame = pd.read_csv(args.input_csv)
    summary = run_lopo(frame, args.target, Path(args.output_dir), args.feature_prefix, args.seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
