import torch

from model.base_model.longer_lite import GroupPoolTA


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
