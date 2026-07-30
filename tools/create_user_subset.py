import argparse
import json
from pathlib import Path


def create_subset(input_dir, output_dir, mode, modulus, keep_buckets):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted(input_dir.glob(f"{mode}-shard-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No {mode} shards found in {input_dir}")

    selected_rows = 0
    source_rows = 0
    for index, shard in enumerate(shards, start=1):
        output_path = output_dir / shard.name
        source_count = pq.ParquetFile(shard).metadata.num_rows
        source_rows += source_count
        if output_path.exists():
            kept = pq.ParquetFile(output_path).metadata.num_rows
        else:
            table = pq.read_table(shard)
            uid = table.column("129_1").combine_chunks().to_numpy()
            bucket = (uid & np.int64(0x7FFFFFFFFFFFFFFF)) % modulus
            filtered = table.filter(pa.array(bucket < keep_buckets))
            pq.write_table(filtered, output_path, compression="zstd")
            kept = filtered.num_rows
        selected_rows += kept
        print(
            f"[{index}/{len(shards)}] {shard.name}: "
            f"source={source_count} selected={kept}",
            flush=True,
        )

    metadata = {
        "mode": mode,
        "source_rows": source_rows,
        "total_rows": selected_rows,
        "num_shards": len(shards),
        "selection_rule": {
            "field": "129_1",
            "modulus": modulus,
            "keep_buckets": keep_buckets,
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Create a deterministic user subset")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["train", "test"], required=True)
    parser.add_argument("--modulus", type=int, default=100)
    parser.add_argument("--keep-buckets", type=int, default=1)
    args = parser.parse_args()
    if args.modulus <= 0:
        raise ValueError("modulus must be positive")
    if not 0 < args.keep_buckets <= args.modulus:
        raise ValueError("keep_buckets must be in [1, modulus]")
    create_subset(
        args.input_dir,
        args.output_dir,
        args.mode,
        args.modulus,
        args.keep_buckets,
    )


if __name__ == "__main__":
    main()
