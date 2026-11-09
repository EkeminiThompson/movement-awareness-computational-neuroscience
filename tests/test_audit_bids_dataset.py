import tempfile
import unittest
from pathlib import Path

from src.audit_bids_dataset import count_tsv_rows


class TestAuditHelpers(unittest.TestCase):
    def test_counts_tsv_data_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sub-01_events.tsv"
            path.write_text("onset\tduration\ttrial_type\n0\t1\tA\n2\t1\tB\n", encoding="utf-8")
            rows, error = count_tsv_rows(path)
            self.assertEqual(rows, 2)
            self.assertIsNone(error)

    def test_empty_tsv_has_zero_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.tsv"
            path.write_text("onset\tduration\n", encoding="utf-8")
            rows, error = count_tsv_rows(path)
            self.assertEqual(rows, 0)
            self.assertIsNone(error)


if __name__ == "__main__":
    unittest.main()
