import torch

from trainer import StagedLongerController


class TinyDenseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        self.longer_lite = torch.nn.Linear(2, 2)


def build_controller(warmup_steps=2):
    dense = TinyDenseModel()
    sparse = torch.nn.Linear(2, 2)
    dense.backbone.bias.requires_grad_(False)
    dense_opt = torch.optim.SGD(dense.parameters(), lr=0.2)
    sparse_opt = torch.optim.SGD(sparse.parameters(), lr=0.3)
    controller = StagedLongerController(
        dense_model=dense,
        sparse_model=sparse,
        dense_opt=dense_opt,
        sparse_opt=sparse_opt,
        warmup_steps=warmup_steps,
        joint_dense_lr=0.05,
        joint_sparse_lr=0.01,
    )
    return controller, dense, sparse, dense_opt, sparse_opt


def test_stage_one_trains_only_longer_branch():
    controller, dense, sparse, _, _ = build_controller()

    controller.start()

    assert all(parameter.requires_grad for parameter in dense.longer_lite.parameters())
    assert not any(parameter.requires_grad for parameter in dense.backbone.parameters())
    assert not any(parameter.requires_grad for parameter in sparse.parameters())


def test_stage_two_restores_flags_and_switches_learning_rates_once():
    controller, dense, sparse, dense_opt, sparse_opt = build_controller()
    controller.start()

    assert not controller.maybe_transition(step=1)
    assert controller.maybe_transition(step=2)
    assert dense.backbone.weight.requires_grad
    assert not dense.backbone.bias.requires_grad
    assert all(parameter.requires_grad for parameter in sparse.parameters())
    assert dense_opt.param_groups[0]["lr"] == 0.05
    assert sparse_opt.param_groups[0]["lr"] == 0.01
    assert not controller.maybe_transition(step=3)
