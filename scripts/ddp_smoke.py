import os

import torch
import torch.distributed as dist


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    value = torch.tensor([rank], device=f"cuda:{local_rank}")
    dist.all_reduce(value)
    print(
        f"rank={rank} world_size={world_size} "
        f"gpu={torch.cuda.get_device_name(local_rank)} reduced={value.item()}",
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
