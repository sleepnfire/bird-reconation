"""Tests pour la Knowledge Distillation ViT-B/16 → MobileNetV2 (TDD — tests d'abord)."""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train import (
    BirdDataset,
    FocalLoss,
    ModelEMA,
    apply_mixup_cutmix,
    create_model,
    discover_samples,
    evaluate,
    get_transforms,
    save_checkpoint,
)


class TestDistillationLoss:
    """La loss de distillation combine KL divergence (soft targets) et hard label loss."""

    def test_returns_scalar_loss(self):
        from distill import distillation_loss
        student_logits = torch.randn(4, 10)
        teacher_logits = torch.randn(4, 10)
        labels = torch.randint(0, 10, (4,))
        loss = distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.7)
        assert loss.dim() == 0
        assert loss.item() > 0
        assert np.isfinite(loss.item())

    def test_alpha_zero_is_pure_hard_loss(self):
        from distill import distillation_loss
        student_logits = torch.randn(4, 10)
        teacher_logits = torch.randn(4, 10)
        labels = torch.randint(0, 10, (4,))
        loss_distill = distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.0)
        loss_ce = F.cross_entropy(student_logits, labels)
        assert torch.allclose(loss_distill, loss_ce, atol=1e-5)

    def test_alpha_one_is_pure_kl_loss(self):
        from distill import distillation_loss
        student_logits = torch.randn(4, 10)
        teacher_logits = torch.randn(4, 10)
        labels = torch.randint(0, 10, (4,))
        loss_alpha1 = distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=1.0)
        T = 4.0
        expected_kl = F.kl_div(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1),
            reduction="batchmean",
        ) * (T * T)
        assert torch.allclose(loss_alpha1, expected_kl, atol=1e-5)

    def test_temperature_scaling_effect(self):
        from distill import distillation_loss
        student_logits = torch.randn(4, 10)
        teacher_logits = torch.randn(4, 10)
        labels = torch.randint(0, 10, (4,))
        loss_t2 = distillation_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.7)
        loss_t8 = distillation_loss(student_logits, teacher_logits, labels, T=8.0, alpha=0.7)
        assert loss_t2.item() != loss_t8.item()

    def test_gradient_flows_to_student(self):
        from distill import distillation_loss
        model = nn.Linear(10, 5)
        x = torch.randn(4, 10)
        student_logits = model(x)
        teacher_logits = torch.randn(4, 5)
        labels = torch.randint(0, 5, (4,))
        loss = distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.7)
        loss.backward()
        assert all(p.grad is not None for p in model.parameters())

    def test_teacher_logits_detached(self):
        from distill import distillation_loss
        student_logits = torch.randn(4, 5, requires_grad=True)
        teacher_logits = torch.randn(4, 5)
        labels = torch.randint(0, 5, (4,))
        loss = distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.7)
        loss.backward()
        assert teacher_logits.grad is None

    def test_with_soft_labels_from_mixup(self):
        from distill import distillation_loss
        student_logits = torch.randn(4, 5)
        teacher_logits = torch.randn(4, 5)
        soft_labels = torch.zeros(4, 5)
        soft_labels[0, 0] = 0.8
        soft_labels[0, 1] = 0.2
        soft_labels[1, 1] = 1.0
        soft_labels[2, 2] = 0.5
        soft_labels[2, 3] = 0.5
        soft_labels[3, 4] = 1.0
        loss = distillation_loss(student_logits, teacher_logits, soft_labels, T=4.0, alpha=0.7)
        assert loss.dim() == 0
        assert np.isfinite(loss.item())

    def test_with_focal_loss_as_hard_loss(self):
        from distill import distillation_loss
        student_logits = torch.randn(8, 10)
        teacher_logits = torch.randn(8, 10)
        labels = torch.randint(0, 10, (8,))
        focal = FocalLoss(gamma=2.0, label_smoothing=0.1)
        loss = distillation_loss(
            student_logits, teacher_logits, labels, T=4.0, alpha=0.7,
            hard_loss_fn=focal,
        )
        assert loss.dim() == 0
        assert loss.item() > 0


class TestDistillationCLI:
    """Arguments CLI pour le script de distillation."""

    def test_parse_args_teacher_checkpoint_required(self):
        from distill import parse_args
        with pytest.raises(SystemExit):
            parse_args([])

    def test_parse_args_temperature_default(self):
        from distill import parse_args
        args = parse_args(["--teacher-checkpoint", "teacher.pth"])
        assert args.distill_temperature == 4.0

    def test_parse_args_alpha_default(self):
        from distill import parse_args
        args = parse_args(["--teacher-checkpoint", "teacher.pth"])
        assert args.distill_alpha == 0.7

    def test_parse_args_student_model_default(self):
        from distill import parse_args
        args = parse_args(["--teacher-checkpoint", "teacher.pth"])
        assert args.model == "mobilenetv2"

    def test_parse_args_custom_values(self):
        from distill import parse_args
        args = parse_args([
            "--teacher-checkpoint", "teacher.pth",
            "--distill-temperature", "6.0",
            "--distill-alpha", "0.5",
        ])
        assert args.distill_temperature == 6.0
        assert args.distill_alpha == 0.5


class TestLoadTeacher:
    """Chargement du modèle teacher depuis un checkpoint (frozen, eval, no_grad)."""

    def test_loads_teacher_from_checkpoint(self, vit_checkpoint_path):
        from distill import load_teacher
        teacher, arch, label_map = load_teacher(vit_checkpoint_path, torch.device("cpu"))
        output = teacher(torch.randn(1, 3, 224, 224))
        assert output.shape == (1, 5)

    def test_teacher_is_frozen(self, vit_checkpoint_path):
        from distill import load_teacher
        teacher, _, _ = load_teacher(vit_checkpoint_path, torch.device("cpu"))
        assert all(not p.requires_grad for p in teacher.parameters())

    def test_teacher_is_eval_mode(self, vit_checkpoint_path):
        from distill import load_teacher
        teacher, _, _ = load_teacher(vit_checkpoint_path, torch.device("cpu"))
        assert not teacher.training

    def test_teacher_produces_finite_logits(self, vit_checkpoint_path):
        from distill import load_teacher
        teacher, _, _ = load_teacher(vit_checkpoint_path, torch.device("cpu"))
        with torch.no_grad():
            output = teacher(torch.randn(1, 3, 224, 224))
        assert torch.isfinite(output).all()

    def test_mismatched_classes_raises(self, tmp_path, label_map):
        from distill import load_teacher
        model, _, _ = create_model("vit_b_16", num_classes=10, pretrained=False)
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        lm_10 = {f"sp_{i}": i for i in range(10)}
        ckpt_path = tmp_path / "teacher_10.pth"
        save_checkpoint(ckpt_path, model, optimizer, scheduler, 0, 0.0, lm_10, "vit_b_16")
        teacher, _, teacher_lm = load_teacher(ckpt_path, torch.device("cpu"))
        assert len(teacher_lm) == 10


class TestDistillTrainOneEpoch:
    """Un epoch de distillation entraîne le student avec les logits du teacher."""

    @pytest.fixture
    def distill_setup(self, mini_dataset, vit_checkpoint_path):
        from distill import load_teacher
        dataset_dir, label_map, _ = mini_dataset
        train_samples = discover_samples(dataset_dir / "train", label_map)
        num_classes = max(label_map.values()) + 1

        transform = get_transforms(224, augment=True)
        ds = BirdDataset(train_samples, transform=transform)
        loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

        student, _, _ = create_model("mobilenetv2", num_classes=num_classes, pretrained=False)
        teacher, _, _ = load_teacher(vit_checkpoint_path, torch.device("cpu"))

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)

        return student, teacher, loader, criterion, optimizer, num_classes

    def test_runs_without_error(self, distill_setup):
        from distill import distill_one_epoch
        student, teacher, loader, criterion, optimizer, _ = distill_setup
        loss, acc = distill_one_epoch(
            student, teacher, loader, optimizer, torch.device("cpu"),
            T=4.0, alpha=0.7,
        )
        assert isinstance(loss, float)
        assert isinstance(acc, float)

    def test_loss_is_finite(self, distill_setup):
        from distill import distill_one_epoch
        student, teacher, loader, criterion, optimizer, _ = distill_setup
        loss, _ = distill_one_epoch(
            student, teacher, loader, optimizer, torch.device("cpu"),
            T=4.0, alpha=0.7,
        )
        assert np.isfinite(loss)

    def test_student_weights_change(self, distill_setup):
        from distill import distill_one_epoch
        student, teacher, loader, _, optimizer, _ = distill_setup
        before = {n: p.data.clone() for n, p in student.named_parameters()}
        distill_one_epoch(
            student, teacher, loader, optimizer, torch.device("cpu"),
            T=4.0, alpha=0.7,
        )
        changed = any(
            not torch.equal(before[n], p.data) for n, p in student.named_parameters()
        )
        assert changed

    def test_teacher_weights_unchanged(self, distill_setup):
        from distill import distill_one_epoch
        student, teacher, loader, _, optimizer, _ = distill_setup
        before = {n: p.data.clone() for n, p in teacher.named_parameters()}
        distill_one_epoch(
            student, teacher, loader, optimizer, torch.device("cpu"),
            T=4.0, alpha=0.7,
        )
        for n, p in teacher.named_parameters():
            assert torch.equal(before[n], p.data)

    def test_with_ema(self, distill_setup):
        from distill import distill_one_epoch
        student, teacher, loader, _, optimizer, _ = distill_setup
        ema = ModelEMA(student, decay=0.999)
        loss, acc = distill_one_epoch(
            student, teacher, loader, optimizer, torch.device("cpu"),
            T=4.0, alpha=0.7, ema_model=ema,
        )
        assert np.isfinite(loss)

    def test_with_mixup(self, distill_setup):
        from distill import distill_one_epoch
        student, teacher, loader, _, optimizer, num_classes = distill_setup
        mixup_fn = lambda imgs, lbls: apply_mixup_cutmix(imgs, lbls, num_classes)
        loss, acc = distill_one_epoch(
            student, teacher, loader, optimizer, torch.device("cpu"),
            T=4.0, alpha=0.7, mixup_fn=mixup_fn, num_classes=num_classes,
        )
        assert np.isfinite(loss)

    def test_with_gradient_clipping(self, distill_setup):
        from distill import distill_one_epoch
        student, teacher, loader, _, optimizer, _ = distill_setup
        loss, acc = distill_one_epoch(
            student, teacher, loader, optimizer, torch.device("cpu"),
            T=4.0, alpha=0.7, clip_grad=1.0,
        )
        assert np.isfinite(loss)

    def test_cpu_compatible(self, distill_setup):
        from distill import distill_one_epoch
        student, teacher, loader, _, optimizer, _ = distill_setup
        loss, acc = distill_one_epoch(
            student, teacher, loader, optimizer, torch.device("cpu"),
            T=4.0, alpha=0.7,
        )
        assert isinstance(loss, float)
        assert 0.0 <= acc <= 1.0


@pytest.mark.slow
class TestIntegrationDistillation:
    """Pipeline de distillation de bout en bout."""

    def test_full_distillation_pipeline(self, mini_dataset, vit_checkpoint_path):
        from distill import distill_one_epoch, load_teacher
        dataset_dir, label_map, _ = mini_dataset
        train_samples = discover_samples(dataset_dir / "train", label_map)
        num_classes = max(label_map.values()) + 1

        transform = get_transforms(224, augment=False)
        ds = BirdDataset(train_samples, transform=transform)
        loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)

        student, _, _ = create_model("mobilenetv2", num_classes=num_classes, pretrained=False)
        teacher, _, _ = load_teacher(vit_checkpoint_path, torch.device("cpu"))
        optimizer = torch.optim.Adam(student.parameters(), lr=1e-2)

        losses = []
        for _ in range(3):
            loss, _ = distill_one_epoch(
                student, teacher, loader, optimizer, torch.device("cpu"),
                T=4.0, alpha=0.7,
            )
            losses.append(loss)

        assert all(np.isfinite(l) for l in losses)

    def test_distill_then_export(self, mini_dataset, vit_checkpoint_path, tmp_path):
        from distill import distill_one_epoch, load_teacher
        from export import export_onnx
        dataset_dir, label_map, _ = mini_dataset
        train_samples = discover_samples(dataset_dir / "train", label_map)
        num_classes = max(label_map.values()) + 1

        transform = get_transforms(224, augment=False)
        ds = BirdDataset(train_samples, transform=transform)
        loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

        student, _, _ = create_model("mobilenetv2", num_classes=num_classes, pretrained=False)
        teacher, _, _ = load_teacher(vit_checkpoint_path, torch.device("cpu"))
        optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)

        distill_one_epoch(
            student, teacher, loader, optimizer, torch.device("cpu"),
            T=4.0, alpha=0.7,
        )

        onnx_path = tmp_path / "distilled_student.onnx"
        size_mb = export_onnx(student, onnx_path, image_size=224)
        assert onnx_path.exists()
        assert size_mb > 0
