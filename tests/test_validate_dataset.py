import unittest

from scripts.validate_dataset import validate_counts


class ValidateDatasetCountsTest(unittest.TestCase):
    def test_accepts_last_index_metadata_when_row_total_matches(self):
        warning = validate_counts(
            mode="test",
            file_count=49,
            metadata_shards=48,
            actual_rows=22_979_465,
            metadata_rows=22_979_465,
        )

        self.assertIn("last shard index", warning)

    def test_rejects_row_total_mismatch(self):
        with self.assertRaisesRegex(ValueError, "row count mismatch"):
            validate_counts(
                mode="test",
                file_count=49,
                metadata_shards=48,
                actual_rows=22_979_464,
                metadata_rows=22_979_465,
            )


if __name__ == "__main__":
    unittest.main()
