import unittest

from src.lopo_preprocess import iter_lopo_folds


class TestLOPOFolds(unittest.TestCase):
    def test_holds_out_each_subject_once(self):
        folds = list(iter_lopo_folds(["sub-01", "sub-02", "sub-03"]))
        self.assertEqual([held_out for _, held_out in folds], ["sub-01", "sub-02", "sub-03"])
        for train, held_out in folds:
            self.assertNotIn(held_out, train)
            self.assertEqual(len(train), 2)

    def test_deduplicates_subject_ids(self):
        self.assertEqual(list(iter_lopo_folds(["a", "a", "b"])),
                         [(["b"], "a"), (["a"], "b")])

    def test_requires_two_distinct_subjects(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            list(iter_lopo_folds(["sub-01"]))


if __name__ == "__main__":
    unittest.main()
