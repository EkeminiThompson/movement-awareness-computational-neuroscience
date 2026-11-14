import unittest

import numpy as np
import pandas as pd

from scripts.audit_and_extract_iackd import (
    causal_filter, eeg_features, event_rows, extract_kinematics,
    map_out_index_to_trial_info,
)


class TestIackdRawAudit(unittest.TestCase):
    def test_event_rows_reconcile_premove_execution_hit(self):
        frame = pd.DataFrame([
            {"onset": 1.0, "duration": 1.0, "trial_type": "premove", "hand": "left",
             "congruency": "congruent", "direction": "right", "ball_color": "red"},
            {"onset": 2.0, "duration": 0.4, "trial_type": "execution", "hand": "left",
             "congruency": "congruent", "direction": "right", "ball_color": "red"},
            {"onset": 2.4, "duration": 0.0, "trial_type": "hit", "hand": "left",
             "congruency": "congruent", "direction": "right", "ball_color": "red"},
        ])
        result = event_rows(frame)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result.iloc[0].event14_s, 2.0)
        self.assertAlmostEqual(result.iloc[0].event_execution_duration_ms, 400.0)

    def test_event_rows_reject_timing_disagreement(self):
        frame = pd.DataFrame([
            {"onset": 1.0, "duration": 1.0, "trial_type": "premove", "hand": "left",
             "congruency": "congruent", "direction": "right", "ball_color": "red"},
            {"onset": 2.2, "duration": 0.4, "trial_type": "execution", "hand": "left",
             "congruency": "congruent", "direction": "right", "ball_color": "red"},
            {"onset": 2.6, "duration": 0.0, "trial_type": "hit", "hand": "left",
             "congruency": "congruent", "direction": "right", "ball_color": "red"},
        ])
        with self.assertRaisesRegex(ValueError, "No premove events"):
            event_rows(frame)

    def test_causal_filter_has_no_future_to_past_leakage(self):
        x = np.zeros((2, 4096), dtype=float)
        x[0, 2048:] = 1.0
        filtered = causal_filter(x, 1024.0)
        self.assertTrue(np.allclose(filtered[:, :2048], 0.0, atol=1e-12))
        self.assertGreater(float(np.max(np.abs(filtered[:, 2048:]))), 0.0)

    def test_eeg_features_are_prefixed_and_finite(self):
        fs = 200.0
        t = np.arange(200) / fs
        epoch = np.vstack([10 * np.sin(2 * np.pi * 10 * t), 5 * np.sin(2 * np.pi * 20 * t)])
        features = eeg_features(epoch, fs, ["C3", "C4"])
        self.assertTrue(features)
        self.assertTrue(all(k.startswith("eeg_") for k in features))
        self.assertTrue(all(np.isfinite(v) for v in features.values()))

    def test_kinematic_outcomes_from_aligned_matlab_arrays(self):
        out = {
            "t_ms": np.array([-200, 0, 10, 20, 30, 40], dtype=float),
            "x_mm": np.array([0, 0, 1, 2, 2, 3], dtype=float),
            "y_mm": np.zeros(6),
        }
        result = extract_kinematics(out)
        self.assertAlmostEqual(result["movement_duration_ms"], 40.0)
        self.assertAlmostEqual(result["path_length_mm"], 3.0)
        self.assertAlmostEqual(result["peak_speed_mm_s"], 100.0)

    def test_matlab_eeg_segment_index_maps_to_trial_info_and_event_time(self):
        info = [
            {"valid": 1, "t14_samp": 200},
            {"valid": 1, "t14_samp": 400},
        ]
        events = pd.DataFrame({"event14_s": [1.0, 2.0]})
        out = {"eeg_seg_index": 2}
        info_i, event_i, error = map_out_index_to_trial_info(out, info, 200.0, events)
        self.assertEqual(info_i, 1)
        self.assertEqual(event_i, 1)
        self.assertAlmostEqual(error, 0.0)

    def test_zero_duration_execution_uses_hit_event_fallback(self):
        frame = pd.DataFrame([
            {"onset": 1.0, "duration": 1.0, "trial_type": "premove", "hand": "left",
             "congruency": "incongruent", "direction": "right", "ball_color": "yellow"},
            {"onset": 2.0, "duration": 0.0, "trial_type": "execution", "hand": "left",
             "congruency": "incongruent", "direction": "right", "ball_color": "yellow"},
            {"onset": 8.2, "duration": 0.0, "trial_type": "hit", "hand": "left",
             "congruency": "incongruent", "direction": "right", "ball_color": "yellow"},
        ])
        result = event_rows(frame)
        self.assertEqual(result.iloc[0].event_duration_source,
                         "hit_onset_minus_execution_onset_fallback")
        self.assertAlmostEqual(result.iloc[0].event_execution_duration_ms, 6200.0)

    def test_orphan_execution_is_audited_not_mispaired(self):
        frame = pd.DataFrame([
            {"onset": 1.0, "duration": 1.0, "trial_type": "premove", "hand": "left",
             "congruency": "congruent", "direction": "right", "ball_color": "red"},
            {"onset": 2.0, "duration": 0.5, "trial_type": "execution", "hand": "left",
             "congruency": "congruent", "direction": "right", "ball_color": "red"},
            {"onset": 2.5, "duration": 0.0, "trial_type": "hit", "hand": "left",
             "congruency": "congruent", "direction": "right", "ball_color": "red"},
            {"onset": 9.0, "duration": 0.0, "trial_type": "execution", "hand": "left",
             "congruency": "incongruent", "direction": "left", "ball_color": "yellow"},
        ])
        result = event_rows(frame)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.attrs["event_audit"]["n_execution_unmatched"], 1)


if __name__ == "__main__":
    unittest.main()
