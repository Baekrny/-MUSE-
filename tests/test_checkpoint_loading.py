from types import SimpleNamespace
from unittest.mock import patch

import pytest

from main import (
    build_arg_parser,
    destroy_process_group_if_initialized,
    load_eval_checkpoint_models,
    load_warm_start_models,
    load_warm_start_or_cleanup,
    run_job,
    validate_warm_start_paths,
)


def test_validate_warm_start_paths_requires_both_files(tmp_path):
    dense_path = tmp_path / "dense.pt"
    sparse_path = tmp_path / "sparse.pt"
    dense_path.touch()
    args = SimpleNamespace(
        warm_start_dense_ckpt=str(dense_path),
        warm_start_sparse_ckpt=str(sparse_path),
    )

    with pytest.raises(FileNotFoundError, match="sparse warm-start checkpoint"):
        validate_warm_start_paths(args)


def test_validate_warm_start_paths_rejects_one_sided_configuration(tmp_path):
    dense_path = tmp_path / "dense.pt"
    dense_path.touch()
    args = SimpleNamespace(warm_start_dense_ckpt=str(dense_path))

    with pytest.raises(ValueError, match="must be provided together"):
        validate_warm_start_paths(args)


def test_validate_warm_start_paths_returns_existing_pair(tmp_path):
    dense_path = tmp_path / "dense.pt"
    sparse_path = tmp_path / "sparse.pt"
    dense_path.touch()
    sparse_path.touch()
    args = SimpleNamespace(
        warm_start_dense_ckpt=str(dense_path),
        warm_start_sparse_ckpt=str(sparse_path),
    )

    assert validate_warm_start_paths(args) == (str(dense_path), str(sparse_path))


def test_use_ddp_cli_default_does_not_override_config():
    parser = build_arg_parser()

    namespace, remaining = parser.parse_known_args([])

    assert namespace.use_ddp is None


def test_use_ddp_cli_flag_is_explicit_true():
    parser = build_arg_parser()

    namespace, remaining = parser.parse_known_args(["--use_ddp"])

    assert namespace.use_ddp is True


def test_hash_shortlist_cli_override_is_integer():
    parser = build_arg_parser()

    namespace, remaining = parser.parse_known_args(["--hash_shortlist", "400"])

    assert namespace.hash_shortlist == 400
    assert remaining == []


def test_load_warm_start_models_disables_internal_barriers():
    class RecordingModel:
        def __init__(self):
            self.calls = []

        def load_ckpt(self, path, synchronize=True):
            self.calls.append((path, synchronize))

    dense = RecordingModel()
    sparse = RecordingModel()

    load_warm_start_models(dense, sparse, ("dense.ckpt", "sparse.ckpt"))

    assert dense.calls == [("dense.ckpt", False)]
    assert sparse.calls == [("sparse.ckpt", False)]


def test_destroy_process_group_is_noop_when_uninitialized():
    with (
        patch("main.dist.is_available", return_value=True),
        patch("main.dist.is_initialized", return_value=False),
        patch("main.dist.destroy_process_group") as destroy,
    ):
        destroy_process_group_if_initialized()

    destroy.assert_not_called()


def test_destroy_process_group_runs_when_initialized():
    with (
        patch("main.dist.is_available", return_value=True),
        patch("main.dist.is_initialized", return_value=True),
        patch("main.dist.destroy_process_group") as destroy,
    ):
        destroy_process_group_if_initialized()

    destroy.assert_called_once_with()


def test_warm_start_load_failure_cleans_up_process_group():
    class FailingDenseModel:
        def load_ckpt(self, path, synchronize=True):
            raise RuntimeError("broken checkpoint")

    class RecordingSparseModel:
        def __init__(self):
            self.calls = []

        def load_ckpt(self, path, synchronize=True):
            self.calls.append((path, synchronize))

    dense = FailingDenseModel()
    sparse = RecordingSparseModel()

    with patch("main.destroy_process_group_if_initialized") as cleanup:
        with pytest.raises(RuntimeError, match="broken checkpoint"):
            load_warm_start_or_cleanup(
                dense,
                sparse,
                ("dense.ckpt", "sparse.ckpt"),
            )

    cleanup.assert_called_once_with()
    assert sparse.calls == []


def test_load_eval_checkpoint_models_disables_internal_barriers():
    class RecordingModel:
        def __init__(self):
            self.calls = []

        def load_ckpt(self, path, synchronize=True):
            self.calls.append((path, synchronize))

    class Wrapper:
        def __init__(self, module):
            self.module = module

    wrapped_dense_model = RecordingModel()
    wrapped_sparse_model = RecordingModel()
    dense = Wrapper(wrapped_dense_model)
    sparse = Wrapper(wrapped_sparse_model)

    load_eval_checkpoint_models(
        dense,
        sparse,
        ("dense.ckpt", "sparse.ckpt"),
        use_ddp=True,
    )

    assert wrapped_dense_model.calls == [("dense.ckpt", False)]
    assert wrapped_sparse_model.calls == [("sparse.ckpt", False)]

    unwrapped_dense = RecordingModel()
    unwrapped_sparse = RecordingModel()

    load_eval_checkpoint_models(
        unwrapped_dense,
        unwrapped_sparse,
        ("dense.ckpt", "sparse.ckpt"),
        use_ddp=False,
    )

    assert unwrapped_dense.calls == [("dense.ckpt", False)]
    assert unwrapped_sparse.calls == [("sparse.ckpt", False)]


def test_run_job_cleans_up_after_training_setup_failure():
    config = {"job_type": "train", "use_ddp": True}

    with (
        patch(
            "main.train_and_eval_ddp",
            side_effect=RuntimeError("setup failed"),
        ) as train,
        patch("main.destroy_process_group_if_initialized") as cleanup,
    ):
        with pytest.raises(RuntimeError, match="setup failed"):
            run_job(config)

    train.assert_called_once_with(args=config, use_ddp=True)
    cleanup.assert_called_once_with()


def test_run_job_cleans_up_after_eval_setup_failure():
    config = {"job_type": "eval", "use_ddp": True}

    with (
        patch(
            "main.eval_ddp",
            side_effect=RuntimeError("setup failed"),
        ) as evaluate,
        patch("main.destroy_process_group_if_initialized") as cleanup,
    ):
        with pytest.raises(RuntimeError, match="setup failed"):
            run_job(config)

    evaluate.assert_called_once_with(args=config, use_ddp=True)
    cleanup.assert_called_once_with()
