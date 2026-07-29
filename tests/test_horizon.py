import torch

from utils.horizon import build_recent_mask, build_eligible_mask
from utils.utils import sim_mm_top_k


def test_recent_mask_keeps_only_rightmost_valid_items():
    valid = torch.tensor([[False, False, True, True, True]])
    assert build_recent_mask(valid, short_window=2).tolist() == [
        [False, False, False, True, True]
    ]


def test_disjoint_policy_removes_recent_valid_items():
    valid = torch.tensor([[False, True, True, True, True]])
    eligible = build_eligible_mask(valid, short_window=2, policy="disjoint")
    assert eligible.tolist() == [[False, True, True, False, False]]


def test_overlap_policy_preserves_all_valid_items():
    valid = torch.tensor([[False, True, True, True, True]])
    eligible = build_eligible_mask(valid, short_window=2, policy="overlap")
    assert eligible.equal(valid)


def test_masked_topk_marks_unavailable_slots():
    target = torch.tensor([[[1.0, 0.0]]])
    sequence = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]])
    eligible = torch.tensor([[True, False, False]])

    indices, invalid = sim_mm_top_k(
        target,
        sequence,
        keep_top=2,
        eligible_mask=eligible,
        return_invalid_mask=True,
    )

    assert indices.shape == (1, 2)
    assert invalid.tolist() == [[False, True]]
