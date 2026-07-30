import torch

from model.muse import MUSE_DIN
from trainer import StagedLongerController


def run_variant(use_short_sa_ta, use_horizon_gate):
    batch_size = 2
    dim = 4
    short_window = 4
    long_window = 5
    args = {
        "method": "muse",
        "use_aux_loss": False,
        "use_short_sa_ta": use_short_sa_ta,
        "use_horizon_gate": use_horizon_gate,
    }
    model = MUSE_DIN(
        args=args,
        D=dim,
        RT_STEPS=short_window,
        UNI_STEPS=long_window,
    )
    model.fc_tower.register_forward_pre_hook(
        lambda module, inputs: print(
            f"tower_input={tuple(inputs[0].shape)} expected={module.linears[0].in_features}"
        )
    )
    user_embs = [
        torch.randn(batch_size, dim),
        torch.randn(batch_size, dim),
        torch.randn(batch_size, dim),
        torch.randn(batch_size, dim // 2),
        torch.randn(batch_size, dim // 2),
        torch.randn(batch_size, dim // 2),
    ]
    ad_embs = [torch.randn(batch_size, dim) for _ in range(4)]
    ad_embs.append(torch.randn(batch_size, 128))
    uni_seq_embs = [
        torch.randn(batch_size, long_window * dim),
        torch.randn(batch_size, long_window * dim),
        torch.randn(batch_size, long_window, 128),
    ]
    short_seq_fn = [
        torch.randn(batch_size, short_window * dim),
        torch.randn(batch_size, short_window * dim),
        torch.randn(batch_size, short_window, 128),
    ]
    label = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    loss, prop = model(user_embs, ad_embs, uni_seq_embs, short_seq_fn, label)
    loss.backward()
    assert torch.isfinite(loss)
    assert prop.shape == (batch_size, 2)
    print(
        f"short_sa_ta={use_short_sa_ta} gate={use_horizon_gate} "
        f"loss={loss.item():.6f} prop_shape={tuple(prop.shape)}"
    )


def run_longer_variant(longer_variant):
    batch_size = 2
    dim = 16
    short_window = 4
    long_window = 5
    full_window = 8
    args = {
        "method": "muse",
        "use_aux_loss": False,
        "use_short_sa_ta": False,
        "use_horizon_gate": False,
        "longer_variant": longer_variant,
        "longer_group_size": 4,
        "longer_model_dim": 2 * dim,
        "longer_num_heads": 1,
        "longer_recent_queries": 2,
        "longer_transform_chunk_size": 2,
        "longer_residual_init": 0.05 if longer_variant == "global-token" else 0.0,
    }
    model = MUSE_DIN(
        args=args,
        D=dim,
        RT_STEPS=short_window,
        UNI_STEPS=long_window,
    )
    user_embs = [
        torch.randn(batch_size, dim),
        torch.randn(batch_size, dim),
        torch.randn(batch_size, dim),
        torch.randn(batch_size, dim // 2),
        torch.randn(batch_size, dim // 2),
        torch.randn(batch_size, dim // 2),
    ]
    ad_embs = [torch.randn(batch_size, dim) for _ in range(4)]
    ad_embs.append(torch.randn(batch_size, 128))
    uni_seq_embs = [
        torch.randn(batch_size, long_window * dim),
        torch.randn(batch_size, long_window * dim),
        torch.randn(batch_size, long_window, 128),
    ]
    short_seq_fn = [
        torch.randn(batch_size, short_window * dim),
        torch.randn(batch_size, short_window * dim),
        torch.randn(batch_size, short_window, 128),
    ]
    full_seq_embs = {
        "target_item": torch.randn(batch_size, dim),
        "target_category": torch.randn(batch_size, dim),
        "target_scl": torch.randn(batch_size, 128),
        "history_item": torch.randn(batch_size, full_window, dim),
        "history_category": torch.randn(batch_size, full_window, dim),
        "history_scl": torch.randn(batch_size, full_window, 128),
        "valid_mask": torch.tensor(
            [
                [False, False, True, True, True, True, True, True],
                [False, False, False, False, True, True, True, True],
            ]
        ),
    }
    label = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    loss, prop = model(
        user_embs,
        ad_embs,
        uni_seq_embs,
        short_seq_fn,
        label,
        full_seq_embs=full_seq_embs,
    )
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert torch.isfinite(loss)
    assert prop.shape == (batch_size, 2)
    assert torch.isfinite(prop).all()
    assert gradients and all(torch.isfinite(grad).all() for grad in gradients)
    if longer_variant == "global-token":
        torch.testing.assert_close(
            model.longer_lite.residual_scale.detach(), torch.tensor(0.05)
        )
    print(
        f"longer_variant={longer_variant} loss={loss.item():.6f} "
        f"prop_shape={tuple(prop.shape)} finite_backward=True"
    )


def run_staged_controller_smoke():
    class TinyDense(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2)
            self.longer_lite = torch.nn.Linear(2, 2)

    dense = TinyDense()
    sparse = torch.nn.Linear(2, 2)
    dense_opt = torch.optim.SGD(dense.parameters(), lr=0.2)
    sparse_opt = torch.optim.SGD(sparse.parameters(), lr=0.3)
    controller = StagedLongerController(
        dense,
        sparse,
        dense_opt,
        sparse_opt,
        warmup_steps=1,
        joint_dense_lr=0.05,
        joint_sparse_lr=0.01,
    )
    controller.start()
    assert all(parameter.requires_grad for parameter in dense.longer_lite.parameters())
    assert not any(parameter.requires_grad for parameter in dense.backbone.parameters())
    assert controller.maybe_transition(step=1)
    assert all(parameter.requires_grad for parameter in dense.parameters())
    assert all(parameter.requires_grad for parameter in sparse.parameters())
    print("staged_controller=branch-only->joint transition=True")


if __name__ == "__main__":
    run_variant(False, False)
    run_variant(True, True)
    for variant in ("group-pool", "inner-trans", "global-token"):
        run_longer_variant(variant)
    run_staged_controller_smoke()
