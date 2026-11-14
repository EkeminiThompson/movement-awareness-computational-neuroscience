import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.download_iackd_manifest import digest_file, safe_path


class TestNEMARManifestDownloader(unittest.TestCase):
    def test_sha256_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "sample.bin"
            p.write_bytes(b"iackd-test")
            self.assertEqual(digest_file(p, "sha256"), hashlib.sha256(b"iackd-test").hexdigest())

    def test_git_blob_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "sample.bin"
            payload = b"iackd-test"
            p.write_bytes(payload)
            expected = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
            self.assertEqual(digest_file(p, "git"), expected)

    def test_manifest_path_must_stay_inside_dataset_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "Unsafe manifest path"):
                safe_path(root, "../escape.bin")
            target = safe_path(root, "sub-01/eeg/sample.vhdr")
            self.assertTrue(target.is_relative_to(root))


if __name__ == "__main__":
    unittest.main()
