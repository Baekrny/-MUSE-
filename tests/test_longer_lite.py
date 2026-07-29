import torch

from model.base_model.longer_lite import GroupPoolTA, InnerTransTA


def test_group_pool_is_mask_aware_and_zeroes_all_padding_groups():
    model = GroupPoolTA(input_dim=4, model_dim=4, max_len=8, group_size=4)
    with torch.no_grad():
        model.token_projection.weight.copy_(torch.eye(4))
        model.token_projection.bias.zero_()
        model.position_embedding.weight.zero_()

    history = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [100.0, 100.0, 100.0, 100.0],
                [200.0, 200.0, 200.0, 200.0],
                [300.0, 300.0, 300.0, 300.0],
                [400.0, 400.0, 400.0, 400.0],
            ]
        ]
    )
    valid = torch.tensor([[True, True, False, False, False, False]])

    merged, merged_valid = model.merge(history, valid)

    assert merged.shape == (1, 2, 4)
    assert merged_valid.tolist() == [[True, False]]
    torch.testing.assert_close(merged[0, 0], history[0, :2].mean(dim=0))
    torch.testing.assert_close(merged[0, 1], torch.zeros(4))


def test_zero_residual_scale_preserves_base_interest_and_is_trainable():
    model = GroupPoolTA(input_dim=8, model_dim=4, max_len=8, group_size=4)
    base_interest = torch.randn(2, 4)
    full_history_interest = torch.randn(2, 4)

    combined = model.combine_interest(base_interest, full_history_interest)

    torch.testing.assert_close(combined, base_interest)
    assert isinstance(model.residual_scale, torch.nn.Parameter)
    assert model.residual_scale.requires_grad


def test_inner_trans_merges_independent_groups_and_zeroes_padding():
    model = InnerTransTA(
        input_dim=4,
        model_dim=4,
        max_len=8,
        group_size=4,
        num_heads=2,
        transform_chunk_size=2,
    )
    model.eval()
    transform_batch_sizes = []

    def record_transform_batch_size(_, inputs):
        transform_batch_sizes.append(inputs[0].shape[0])

    model.inner_transformer.register_forward_pre_hook(record_transform_batch_size)
    history = torch.randn(2, 8, 4)
    valid = torch.tensor(
        [
            [True, True, True, True, False, False, False, False],
            [True, True, True, True, True, True, True, True],
        ]
    )

    merged, merged_valid = model.merge(history, valid)
    perturbed_history = history.clone()
    perturbed_history[1, :4] += 1_000.0
    perturbed, _ = model.merge(perturbed_history, valid)

    assert merged.shape == (2, 2, 4)
    assert merged_valid.tolist() == [[True, False], [True, True]]
    torch.testing.assert_close(merged[0, 1], torch.zeros(4))
    torch.testing.assert_close(merged[1, 1], perturbed[1, 1])
    assert torch.isfinite(merged).all()
    assert len(transform_batch_sizes) >= 2
    assert max(transform_batch_sizes) <= 2
