import torch
from torch import nn
from torch.nn import functional as F


def should_time_retrieval(batch_index, warmup_steps):
    return batch_index >= warmup_steps


class RandomProjectionHash(nn.Module):
    def __init__(self, input_dim, bits, seed):
        super().__init__()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        projection = torch.randn(input_dim, bits, generator=generator)
        self.register_buffer("projection", projection)

    def forward(self, x):
        return (x @ self.projection) >= 0


def hamming_topk(query_code, history_code, valid_mask, keep_top):
    if query_code.dim() == 3:
        query_code = query_code.squeeze(1)
    valid_mask = valid_mask.to(device=history_code.device, dtype=torch.bool)
    distance = torch.logical_xor(
        query_code.unsqueeze(1), history_code
    ).sum(dim=-1)
    bits = history_code.shape[-1]
    distance = distance.masked_fill(~valid_mask, bits + 1)
    ordered = torch.argsort(distance, dim=1, stable=True)

    available = min(keep_top, history_code.shape[1])
    indices = ordered[:, :available]
    invalid = ~valid_mask.gather(1, indices)
    if available < keep_top:
        pad = torch.zeros(
            history_code.shape[0],
            keep_top - available,
            dtype=torch.long,
            device=history_code.device,
        )
        indices = torch.cat([indices, pad], dim=1)
        invalid = torch.cat([invalid, torch.ones_like(pad, dtype=torch.bool)], dim=1)
    return indices, invalid


def target_history_cosine(target_content, history_content, indices=None):
    if target_content.dim() == 2:
        target_content = target_content.unsqueeze(1)
    if indices is not None:
        batch_indices = torch.arange(
            history_content.shape[0], device=history_content.device
        ).unsqueeze(1).expand_as(indices)
        history_content = history_content[batch_indices, indices]

    target_normalized = F.normalize(target_content, dim=-1)
    history_normalized = F.normalize(history_content, dim=-1)
    return torch.bmm(
        target_normalized, history_normalized.transpose(-1, -2)
    ).squeeze(1)


def topk_recall(exact, approx, exact_invalid):
    exact_invalid = exact_invalid.to(device=exact.device, dtype=torch.bool)
    found = (exact.unsqueeze(-1) == approx.unsqueeze(1)).any(dim=-1)
    valid = ~exact_invalid
    recalled = (found & valid).sum(dim=1, dtype=torch.float32)
    denominator = valid.sum(dim=1).clamp_min(1).to(torch.float32)
    return (recalled / denominator).mean()
