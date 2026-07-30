import torch

from utils.hash_retrieval import (
    RandomProjectionHash,
    hamming_topk,
    target_history_cosine,
    topk_recall,
)


def test_seeded_hash_and_padding_safe_retrieval_recall():
    x = torch.tensor(
        [
            [1.0, -2.0, 3.0, 0.5],
            [-1.0, 0.0, 2.0, 4.0],
        ]
    )
    first_code = RandomProjectionHash(input_dim=4, bits=6, seed=7)(x)
    second_code = RandomProjectionHash(input_dim=4, bits=6, seed=7)(x)

    assert first_code.dtype == torch.bool
    assert first_code.shape == (2, 6)
    torch.testing.assert_close(first_code, second_code)

    query_code = torch.tensor([[False, False, False, False]])
    history_code = torch.tensor(
        [
            [
                [True, False, False, False],
                [False, False, False, False],
                [True, True, False, False],
            ]
        ]
    )
    valid_mask = torch.tensor([[True, False, True]])

    indices, invalid = hamming_topk(
        query_code, history_code, valid_mask, keep_top=3
    )

    assert indices.shape == invalid.shape == (1, 3)
    assert invalid.dtype == torch.bool
    assert invalid.sum().item() == 1
    assert valid_mask.gather(1, indices)[~invalid].all()
    assert 1 not in indices[~invalid].tolist()

    exact = torch.tensor([[0, 2, 99]])
    exact_invalid = torch.tensor([[False, False, True]])
    recall = topk_recall(exact, indices, exact_invalid)

    torch.testing.assert_close(recall, torch.tensor(1.0))


def test_candidate_cosine_matches_gathered_full_history_cosine():
    target_content = torch.tensor(
        [
            [1.0, 2.0, -1.0],
            [0.5, -2.0, 3.0],
        ]
    )
    history_content = torch.tensor(
        [
            [
                [1.0, 0.0, 2.0],
                [-1.0, 3.0, 0.5],
                [2.0, 2.0, -2.0],
                [0.5, -1.0, 4.0],
            ],
            [
                [2.0, -1.0, 0.0],
                [1.0, 1.0, 1.0],
                [-2.0, 0.5, 3.0],
                [4.0, -3.0, 2.0],
            ],
        ]
    )
    indices = torch.tensor([[3, 1], [0, 2]])

    full_cosine = target_history_cosine(target_content, history_content)
    candidate_cosine = target_history_cosine(
        target_content, history_content, indices=indices
    )
    expected = full_cosine.gather(1, indices)

    assert full_cosine.shape == (2, 4)
    assert candidate_cosine.shape == (2, 2)
    torch.testing.assert_close(candidate_cosine, expected)
