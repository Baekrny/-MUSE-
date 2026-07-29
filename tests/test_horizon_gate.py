import torch

from model.base_model.horizon_gate import ScalarHorizonGate


def test_scalar_gate_returns_complementary_weights():
    gate = ScalarHorizonGate(interest_dim=8, hidden_dim=4)
    gate.reset_parameters()
    target = torch.randn(3, 8)
    short = torch.randn(3, 8)
    long = torch.randn(3, 8)

    short_out, long_out, weight = gate(target, short, long)

    assert short_out.shape == short.shape
    assert long_out.shape == long.shape
    assert weight.shape == (3, 1)
    assert torch.all((weight >= 0) & (weight <= 1))
    assert torch.allclose(short_out, short * weight)
    assert torch.allclose(long_out, long * (1 - weight))
