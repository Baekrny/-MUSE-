import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Download the TAOBAO-MM dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("taobao-mm"),
        help="Local dataset directory (default: ./taobao-mm)",
    )
    parser.add_argument(
        "--repo-id",
        default="TaoBao-MM/Taobao-MM",
        help="Hugging Face dataset repository",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(args.output_dir.resolve()),
    )
