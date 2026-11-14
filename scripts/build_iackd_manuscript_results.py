#!/usr/bin/env python3
"""Create manuscript-ready tables, figures and Results text from completed LOPO runs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TARGETS = {
    "lopo_duration": ("Movement duration", "ms"),
    "lopo_path_length": ("3-D path length", "mm"),
    "lopo_peak_speed": ("Peak 3-D speed", "mm/s"),
}


def load_target(root: Path, folder: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    base = root / folder
    summary = json.loads((base / "metrics.json").read_text(encoding="utf-8"))
    participants = pd.read_csv(base / "participant_fold_metrics.csv")
    conditions = pd.read_csv(base / "condition_metrics.csv")
    return summary, participants, conditions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    args = ap.parse_args()
    root = args.results_dir
    root.mkdir(parents=True, exist_ok=True)

    summaries = []
    condition_rows = []
    participant_rows = []
    for folder, (label, unit) in TARGETS.items():
        summary, participants, conditions = load_target(root, folder)
        o = summary["overall"]
        summaries.append({
            "outcome": label, "unit": unit,
            "ridge_mae": o["ridge"]["mae"], "ridge_rmse": o["ridge"]["rmse"],
            "training_mean_mae": o["training_mean_baseline"]["mae"],
            "training_mean_rmse": o["training_mean_baseline"]["rmse"],
            "n_participants": o["n_participants"], "n_trials": o["n_trials"],
        })
        for _, row in conditions.iterrows():
            condition_rows.append({
                "outcome": label, "condition": row["condition"],
                "ridge_mae": row["ridge_mae"], "ridge_rmse": row["ridge_rmse"],
                "baseline_mae": row["baseline_mae"], "n_trials": row["ridge_n_trials"],
            })
        participants["outcome"] = label
        participant_rows.append(participants)

    overall = pd.DataFrame(summaries)
    condition_table = pd.DataFrame(condition_rows)
    participant_table = pd.concat(participant_rows, ignore_index=True)
    overall.to_csv(root / "manuscript_overall_metrics.csv", index=False)
    condition_table.to_csv(root / "manuscript_condition_metrics.csv", index=False)
    participant_table.to_csv(root / "manuscript_participant_metrics.csv", index=False)

    # Figure 1: participant-held-out model versus training-mean baseline.
    x = np.arange(len(overall))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, overall.ridge_mae, width, label="Ridge")
    ax.bar(x + width / 2, overall.training_mean_mae, width, label="Training mean")
    ax.set_xticks(x, overall.outcome)
    ax.set_ylabel("Mean absolute error (outcome-specific units)")
    ax.set_title("Participant-held-out prediction error")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(root / "figure_1_overall_mae.png", dpi=300)
    plt.close(fig)

    # Figure 2: condition-stratified held-out error.
    fig, ax = plt.subplots(figsize=(9, 5))
    conditions = sorted(condition_table.condition.unique())
    width = 0.8 / max(1, len(conditions))
    for i, condition in enumerate(conditions):
        subset = condition_table[condition_table.condition == condition]
        positions = np.arange(len(overall)) - 0.4 + width / 2 + i * width
        vals = [
            subset.loc[subset.outcome == name, "ridge_mae"].iloc[0]
            if (subset.outcome == name).any() else np.nan
            for name in overall.outcome
        ]
        ax.bar(positions, vals, width, label=condition)
    ax.set_xticks(np.arange(len(overall)), overall.outcome)
    ax.set_ylabel("Mean absolute error (outcome-specific units)")
    ax.set_title("Held-out prediction error by visuomotor condition")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(root / "figure_2_condition_mae.png", dpi=300)
    plt.close(fig)

    # Figure 3: participant-level error distributions by outcome and condition.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (folder, (label, unit)) in zip(axes, TARGETS.items()):
        base = root / folder / "condition_participant_metrics.csv"
        frame = pd.read_csv(base)
        vals = [frame.loc[frame.condition == c, "mae"].dropna().to_numpy()
                for c in conditions]
        ax.boxplot(vals, tick_labels=conditions, showfliers=False)
        ax.set_title(label)
        ax.set_ylabel(f"Participant MAE ({unit})")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Participant-level error by condition")
    fig.tight_layout()
    fig.savefig(root / "figure_3_participant_condition_errors.png", dpi=300)
    plt.close(fig)

    lines = [
        "# Results (generated from completed IACKD LOPO runs)", "",
        "The following text is generated directly from the saved out-of-fold predictions and metric files. "
        "It should be reviewed alongside the audit summary before manuscript use.", "",
        "## Overall participant-held-out prediction", "",
        overall.to_markdown(index=False), "",
        "## Condition-stratified performance", "",
        condition_table.to_markdown(index=False), "",
        "## Reporting note", "",
        "All values above must be interpreted as participant-held-out performance from the causal-filtered "
        "EEG feature pipeline. The model was evaluated on all valid trials from one participant at a time; "
        "standardisation and Ridge hyperparameter selection were fitted within participant-grouped training folds. "
        "The training-mean baseline was estimated using training participants only. Condition comparisons are "
        "descriptive unless a prespecified participant-level paired analysis is added.", "",
    ]
    (root / "MANUSCRIPT_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("Generated manuscript tables, figures and MANUSCRIPT_RESULTS.md in", root)


if __name__ == "__main__":
    main()
