"""Tests pour le tracking TensorBoard (TDD — tests d'abord)."""

import pytest
import numpy as np
import torch

from train import (
    TBLogger,
    create_model,
    parse_args,
)


class TestTBLoggerDisabled:
    """Quand logdir est None, le logger est un no-op silencieux."""

    def test_creates_without_error(self):
        tb = TBLogger(None)
        assert tb.writer is None

    def test_log_scalars_noop(self):
        tb = TBLogger(None)
        tb.log_scalars({"loss": 0.5}, step=0)

    def test_log_hparams_noop(self):
        tb = TBLogger(None)
        tb.log_hparams({"lr": 1e-3}, {"val_acc": 0.8})

    def test_log_image_noop(self):
        tb = TBLogger(None)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        tb.log_image("test", img, step=0)

    def test_close_noop(self):
        tb = TBLogger(None)
        tb.close()

    def test_context_manager_noop(self):
        with TBLogger(None) as tb:
            tb.log_scalars({"x": 1.0}, step=0)


class TestTBLoggerEnabled:
    """Quand logdir est fourni, le logger écrit des événements TensorBoard."""

    def test_creates_writer(self, tmp_path):
        tb = TBLogger(str(tmp_path / "runs"))
        assert tb.writer is not None
        tb.close()

    def test_creates_logdir(self, tmp_path):
        logdir = tmp_path / "runs" / "test"
        tb = TBLogger(str(logdir))
        tb.log_scalars({"loss": 0.5}, step=0)
        tb.close()
        assert logdir.exists()

    def test_log_scalars_creates_events(self, tmp_path):
        logdir = tmp_path / "runs"
        tb = TBLogger(str(logdir))
        tb.log_scalars({"train/loss": 0.5, "train/acc": 0.8}, step=1)
        tb.close()
        event_files = list(logdir.glob("events.out.tfevents.*"))
        assert len(event_files) > 0

    def test_log_hparams_creates_events(self, tmp_path):
        logdir = tmp_path / "runs"
        tb = TBLogger(str(logdir))
        tb.log_hparams(
            {"model": "mobilenetv2", "lr": 1e-3, "epochs": 80},
            {"val_acc": 0.75, "val_loss": 1.2},
        )
        tb.close()
        event_files = list(logdir.glob("events.out.tfevents.*"))
        assert len(event_files) > 0

    def test_log_image(self, tmp_path):
        logdir = tmp_path / "runs"
        tb = TBLogger(str(logdir))
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        tb.log_image("confusion_matrix", img, step=0)
        tb.close()
        event_files = list(logdir.glob("events.out.tfevents.*"))
        assert len(event_files) > 0

    def test_context_manager_closes(self, tmp_path):
        logdir = tmp_path / "runs"
        with TBLogger(str(logdir)) as tb:
            tb.log_scalars({"x": 1.0}, step=0)
        event_files = list(logdir.glob("events.out.tfevents.*"))
        assert len(event_files) > 0

    def test_log_lr_from_optimizer(self, tmp_path):
        logdir = tmp_path / "runs"
        model, _, _ = create_model("mobilenetv2", num_classes=5, pretrained=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        tb = TBLogger(str(logdir))
        tb.log_lr(optimizer, step=0)
        tb.close()
        event_files = list(logdir.glob("events.out.tfevents.*"))
        assert len(event_files) > 0


class TestParseArgsTensorBoard:
    """Le flag --logdir est reconnu par le CLI."""

    def test_logdir_default_none(self):
        args = parse_args(["--model", "mobilenetv2"])
        assert args.logdir is None

    def test_logdir_custom(self):
        args = parse_args(["--logdir", "runs/exp1"])
        assert str(args.logdir) == "runs/exp1"

    def test_logdir_auto(self):
        args = parse_args(["--logdir", "auto"])
        assert str(args.logdir) == "auto"


class TestTrainMainTensorBoard:
    """train.py main() crée des logs TensorBoard quand --logdir est fourni."""

    def test_creates_tensorboard_events(self, mini_dataset, tmp_path):
        dataset_dir, label_map, _ = mini_dataset
        logdir = tmp_path / "tb_logs"

        from train import main
        main([
            "--model", "mobilenetv2",
            "--dataset", str(dataset_dir),
            "--epochs", "2",
            "--batch-size", "4",
            "--workers", "0",
            "--output", str(tmp_path / "output"),
            "--logdir", str(logdir),
            "--no-ema",
            "--no-mixup",
            "--patience", "0",
            "--freeze-backbone-epochs", "0",
        ])

        run_dirs = [d for d in logdir.iterdir() if d.is_dir()]
        assert len(run_dirs) == 1
        event_files = list(run_dirs[0].glob("events.out.tfevents.*"))
        assert len(event_files) > 0

    def test_no_tensorboard_without_flag(self, mini_dataset, tmp_path):
        dataset_dir, label_map, _ = mini_dataset

        from train import main
        main([
            "--model", "mobilenetv2",
            "--dataset", str(dataset_dir),
            "--epochs", "1",
            "--batch-size", "4",
            "--workers", "0",
            "--output", str(tmp_path / "output"),
            "--no-ema",
            "--no-mixup",
            "--patience", "0",
            "--freeze-backbone-epochs", "0",
        ])

        runs_dir = tmp_path / "output" / "runs"
        assert not runs_dir.exists()

    def test_logdir_auto_creates_named_dir(self, mini_dataset, tmp_path):
        dataset_dir, label_map, _ = mini_dataset

        from train import main
        main([
            "--model", "mobilenetv2",
            "--dataset", str(dataset_dir),
            "--epochs", "1",
            "--batch-size", "4",
            "--workers", "0",
            "--output", str(tmp_path / "output"),
            "--logdir", "auto",
            "--no-ema",
            "--no-mixup",
            "--patience", "0",
            "--freeze-backbone-epochs", "0",
        ])

        runs_dir = tmp_path / "output" / "runs"
        assert runs_dir.exists()
        run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
        assert len(run_dirs) == 1
        assert "mobilenetv2" in run_dirs[0].name
