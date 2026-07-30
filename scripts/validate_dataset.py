import argparse
import json
from pathlib import Path


def validate_counts(mode, file_count, metadata_shards, actual_rows, metadata_rows):
    if actual_rows != metadata_rows:
        raise ValueError(
            f"{mode} row count mismatch: files={actual_rows} "
            f"metadata={metadata_rows}"
        )
    if file_count == metadata_shards:
        return None
    if file_count == metadata_shards + 1:
        return (
            f"{mode}: metadata num_shards={metadata_shards} is the last shard "
            f"index; discovered {file_count} files with an exact row-total match"
        )
    raise ValueError(
        f"{mode} shard count mismatch: files={file_count} "
        f"metadata={metadata_shards}"
    )


def validate_split(root, mode):
    import pyarrow.parquet as pq

    split_dir = root / mode
    shards = sorted(split_dir.glob(f"{mode}-shard-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No {mode} shards found in {split_dir}")

    metadata_path = split_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    row_counts = [pq.ParquetFile(path).metadata.num_rows for path in shards]
    actual_rows = sum(row_counts)
    print(
        f"{mode}: files={len(shards)} metadata_shards={metadata['num_shards']} "
        f"rows={actual_rows} tail_rows={row_counts[-1]}"
    )
    warning = validate_counts(
        mode=mode,
        file_count=len(shards),
        metadata_shards=metadata["num_shards"],
        actual_rows=actual_rows,
        metadata_rows=metadata["total_rows"],
    )
    if warning:
        print(f"WARNING: {warning}")


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a TAOBAO-MM download")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("taobao-mm"),
        help="Dataset root (default: ./taobao-mm)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    import pyarrow.parquet as pq

    dataset_root = args.root.resolve()
    validate_split(dataset_root, "train")
    validate_split(dataset_root, "test")
    required_maps = [
        "150_2_180_sorted_map_p90.npy",
        "scl_emb_int8_p90_keys.npy",
        "scl_emb_int8_p90_values.npy",
    ]
    for name in required_maps:
        path = dataset_root / "feature_map" / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing feature map: {path}")
        print(f"feature_map/{name}: exists=True size={path.stat().st_size}")

    first_shard = sorted((dataset_root / "train").glob("train-shard-*.parquet"))[0]
    sample = next(pq.ParquetFile(first_shard).iter_batches(batch_size=2))
    history = sample.column(sample.schema.get_field_index("150_2_180"))
    print(
        f"sample: rows={sample.num_rows} columns={sample.num_columns} "
        f"history_lengths={[len(value) for value in history.to_pylist()]}"
    )
