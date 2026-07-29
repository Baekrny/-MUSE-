from unittest.mock import patch

import torch

from model.base_model.layers import MultiHeadAttV2


def test_calc_attn_score_matches_shared_multimodal_logits():
    torch.manual_seed(0)
    layer = MultiHeadAttV2(4, 4, [4], [4], attn_score_cross=True)
    layer.reset_parameters()
    query = torch.randn(2, 1, 4)
    fact = torch.randn(2, 5, 4)
    mm = torch.randn(2, 5)
    mask = torch.tensor(
        [[True, True, True, False, False], [True, False, True, True, False]]
    )

    raw = layer.raw_relevance_logits(query, fact, mask=mask, mm_cosine=[mm])

    expected_query = layer.query_layers[0](query)
    expected_query = expected_query * torch.sigmoid(expected_query)
    expected_fact = layer.fact_layers[0](fact)
    expected_fact = expected_fact * torch.sigmoid(expected_fact)
    expected_raw = torch.matmul(expected_fact, expected_query.transpose(-1, -2))
    mm_bias = mm / layer.cosine_tau1[0, 0].view(1, -1)
    mm_bias = mm_bias.view(-1, expected_raw.shape[1], 1)
    b_1 = layer.cosine_tau2[0, 0].view(1, 1, 1)
    b_2 = layer.cosine_tau2[1, 0].view(1, 1, 1)
    b_3 = layer.cosine_tau2[2, 0].view(1, 1, 1)
    expected_raw = (
        b_1 * expected_raw
        + b_2 * mm_bias
        + b_3 * expected_raw * mm_bias
    )
    expected_raw = expected_raw.masked_fill(
        ~mask.unsqueeze(-1).bool(), float("-inf")
    )

    assert raw.shape == (2, layer.heads, 5, 1)
    expanded_mask = mask[:, None, :, None].expand_as(raw)
    assert torch.isneginf(raw.masked_select(~expanded_mask)).all()
    torch.testing.assert_close(raw[:, 0], expected_raw)
    expected = torch.softmax(expected_raw.unsqueeze(1), dim=2)
    expected = expected.squeeze(-1).mean(dim=1)
    expected = expected.masked_fill(~mask, 0)
    with patch.object(
        layer, "raw_relevance_logits", wraps=layer.raw_relevance_logits
    ) as shared:
        actual = layer.calc_attn_score(query, fact, mask=mask, mm_cosine=[mm])

    shared.assert_called_once()
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual).all()


def test_forward_calls_the_shared_raw_relevance_api():
    torch.manual_seed(0)
    layer = MultiHeadAttV2(4, 4, [4], [4], attn_score_cross=True)
    layer.reset_parameters()
    query = torch.randn(2, 1, 4)
    fact = torch.randn(2, 5, 4)
    mm = torch.randn(2, 5)
    mask = torch.tensor(
        [[True, True, True, False, False], [False, False, False, False, False]]
    )

    with patch.object(
        layer, "raw_relevance_logits", wraps=layer.raw_relevance_logits
    ) as shared:
        output = layer(query, fact, mask=mask, mm_cosine=[mm])

    shared.assert_called_once()
    assert output.shape == (2, 4)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output[1], torch.zeros_like(output[1]))
    output.sum().backward()
    for parameter in layer.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
