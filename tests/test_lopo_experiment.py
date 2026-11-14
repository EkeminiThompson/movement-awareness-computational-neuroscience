import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest

# The lightweight legacy CI job may not install optional analysis dependencies.
# The dedicated LOPO workflow installs requirements-analysis.txt and runs this module.
pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from scripts.run_lopo_experiment import run_lopo, validate_table


def make_fixture(seed=12):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(6):
        for c, condition in enumerate(("congruent", "incongruent")):
            for t in range(8):
                signal = rng.normal()
                # Shared signal-outcome relation; modest condition shift for test coverage.
                target = 300 + 15 * signal + 8 * c + rng.normal(0, 3)
                rows.append({
                    "participant_id": f"sub-{p:02d}",
                    "trial_id": f"{condition}-{t:02d}",
                    "condition": condition,
                    "movement_duration_ms": target,
                    "eeg_c3_mu": signal,
                    "eeg_cz_beta": 0.5 * signal + rng.normal(0, 0.2),
                })
    return pd.DataFrame(rows)


class TestLopoExperiment(unittest.TestCase):
    def test_lopo_runner_outputs_predictions_and_condition_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_lopo(make_fixture(), "movement_duration_ms", Path(tmp))
            self.assertEqual(result["overall"]["n_participants"], 6)
            self.assertEqual(result["overall"]["n_trials"], 96)
            self.assertEqual(len(result["conditions"]), 2)
            self.assertTrue((Path(tmp) / "lopo_predictions.csv").exists())
            predictions = pd.read_csv(Path(tmp) / "lopo_predictions.csv")
            self.assertEqual(predictions["held_out_participant"].nunique(), 6)
            for held_out, group in predictions.groupby("held_out_participant"):
                self.assertEqual(set(group["participant_id"]), {held_out})

    def test_only_eeg_prefixed_columns_are_selected(self):
        frame = make_fixture()
        features = validate_table(frame, "movement_duration_ms")
        self.assertEqual(features, ["eeg_c3_mu", "eeg_cz_beta"])
        self.assertNotIn("condition", features)

    def test_duplicate_trial_keys_are_rejected(self):
        frame = make_fixture()
        frame.loc[1, "trial_id"] = frame.loc[0, "trial_id"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_table(frame, "movement_duration_ms")

    def test_kinematic_columns_cannot_become_predictors(self):
        frame = make_fixture()
        frame["path_length_mm"] = frame["movement_duration_ms"] * 2
        features = validate_table(frame, "movement_duration_ms")
        self.assertNotIn("path_length_mm", features)


if __name__ == "__main__":
    unittest.main()
