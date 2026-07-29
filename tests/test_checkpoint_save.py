from unittest.mock import patch

from trainer import prepare_checkpoint_dir, synchronize_after_checkpoint


def test_prepare_checkpoint_dir_creates_missing_directory_idempotently(tmp_path):
    checkpoint_dir = tmp_path / "nested" / "ckpt"
    assert not checkpoint_dir.exists()

    prepare_checkpoint_dir(checkpoint_dir)

    assert checkpoint_dir.is_dir()

    prepare_checkpoint_dir(checkpoint_dir)

    assert checkpoint_dir.is_dir()


def test_synchronize_after_checkpoint_barriers_when_initialized():
    with (
        patch("trainer.dist.is_available", return_value=True),
        patch("trainer.dist.is_initialized", return_value=True),
        patch("trainer.dist.barrier") as barrier,
    ):
        synchronize_after_checkpoint()

    barrier.assert_called_once_with()


def test_synchronize_after_checkpoint_is_noop_when_uninitialized():
    with (
        patch("trainer.dist.is_available", return_value=True),
        patch("trainer.dist.is_initialized", return_value=False),
        patch("trainer.dist.barrier") as barrier,
    ):
        synchronize_after_checkpoint()

    barrier.assert_not_called()
