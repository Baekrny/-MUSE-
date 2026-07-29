import torch


def build_recent_mask(valid_mask, short_window):
    if valid_mask.ndim != 2:
        raise ValueError("valid_mask must have shape (batch, sequence)")
    if short_window <= 0:
        raise ValueError("short_window must be positive")
    sequence_length = valid_mask.shape[1]
    positions = torch.arange(sequence_length, device=valid_mask.device)
    first_recent = max(sequence_length - short_window, 0)
    return valid_mask & (positions >= first_recent).unsqueeze(0)


def build_eligible_mask(valid_mask, short_window, policy):
    if policy == "overlap":
        return valid_mask
    if policy == "disjoint":
        return valid_mask & ~build_recent_mask(valid_mask, short_window)
    raise ValueError(f"Unknown long_history_policy: {policy}")


def overlap_summary(top_k_indices, valid_mask, recent_window, invalid_mask=None):
    recent_mask = build_recent_mask(valid_mask, recent_window)
    selected_recent = recent_mask.gather(1, top_k_indices)
    if invalid_mask is not None:
        selected_recent = selected_recent & ~invalid_mask
        selected_per_sample = (~invalid_mask).sum(dim=1).float()
    else:
        selected_per_sample = torch.full(
            (top_k_indices.shape[0],),
            top_k_indices.shape[1],
            device=top_k_indices.device,
            dtype=torch.float32,
        )
    selected_count = selected_per_sample.sum().item()
    recent_count = selected_recent.sum().item()
    overlap_rate = recent_count / max(selected_count, 1)
    valid_length = valid_mask.sum(dim=1).clamp_min(1).float()
    random_rate_per_sample = torch.minimum(
        torch.full_like(valid_length, recent_window, dtype=torch.float32),
        valid_length,
    ).div(valid_length)
    expected_recent_count = (random_rate_per_sample * selected_per_sample).sum().item()
    return {
        "overlap_rate": overlap_rate,
        "enrichment": recent_count / max(expected_recent_count, 1e-12),
        "recent_count": recent_count,
        "selected_count": selected_count,
        "expected_recent_count": expected_recent_count,
    }
