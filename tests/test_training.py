import json

import torch
import torch.nn as nn
import numpy as np
import pytest
from PIL import Image
from torchvision import transforms

from train import (
    BirdDataset,
    EarlyStopping,
    FocalLoss,
    ModelEMA,
    apply_mixup_cutmix,
    build_optimizer_groups,
    create_model,
    create_scheduler,
    create_weighted_sampler,
    discover_samples,
    evaluate,
    find_problematic_images,
    freeze_backbone,
    get_transforms,
    parse_args,
    save_checkpoint,
    seed_everything,
    train_one_epoch,
)
from torch.utils.data import DataLoader


class TestDiscoverSamples:
    """discover_samples charge les images et les bbox depuis annotations.json."""

    def test_returns_triples_with_bbox(self, mini_dataset):
        dataset_dir, label_map, _ = mini_dataset
        samples = discover_samples(dataset_dir / "train", label_map)
        assert len(samples) > 0
        for path, label, bbox in samples:
            assert isinstance(path, str)
            assert isinstance(label, int)
            assert bbox is not None
            assert len(bbox) == 4

    def test_missing_annotations_gives_none_bbox(self, tmp_path, label_map):
        train_dir = tmp_path / "train"
        for slug in label_map:
            sp_dir = train_dir / slug
            sp_dir.mkdir(parents=True)
            Image.new("RGB", (64, 64)).save(sp_dir / "test.jpg")
        samples = discover_samples(train_dir, label_map)
        for _, _, bbox in samples:
            assert bbox is None

    def test_null_detection_gives_none_bbox(self, tmp_path, label_map):
        train_dir = tmp_path / "train"
        slug = list(label_map.keys())[0]
        sp_dir = train_dir / slug
        sp_dir.mkdir(parents=True)
        Image.new("RGB", (64, 64)).save(sp_dir / "test.jpg")
        with open(sp_dir / "annotations.json", "w") as f:
            json.dump({"test.jpg": None}, f)
        samples = discover_samples(train_dir, label_map)
        assert len(samples) == 1
        assert samples[0][2] is None


class TestBirdDatasetBboxCrop:
    """BirdDataset recadre l'image selon la bbox des annotations."""

    def test_crop_changes_dimensions(self, tmp_path):
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        img.save(tmp_path / "test.jpg")
        samples_crop = [(str(tmp_path / "test.jpg"), 0, [20, 20, 60, 60])]
        samples_full = [(str(tmp_path / "test.jpg"), 0, None)]
        to_tensor = transforms.ToTensor()
        ds_crop = BirdDataset(samples_crop, transform=to_tensor)
        ds_full = BirdDataset(samples_full, transform=to_tensor)
        img_crop, _ = ds_crop[0]
        img_full, _ = ds_full[0]
        assert img_crop.shape == (3, 60, 60)
        assert img_full.shape == (3, 100, 100)

    def test_no_bbox_uses_full_image(self, tmp_path):
        img = Image.new("RGB", (80, 80))
        img.save(tmp_path / "test.jpg")
        samples = [(str(tmp_path / "test.jpg"), 0, None)]
        to_tensor = transforms.ToTensor()
        ds = BirdDataset(samples, transform=to_tensor)
        out, _ = ds[0]
        assert out.shape == (3, 80, 80)


class TestWeightedSampler:
    """Gestion du déséquilibre de classes via WeightedRandomSampler."""

    def test_balances_classes(self, mini_dataset_imbalanced):
        dataset_dir, label_map, image_counts = mini_dataset_imbalanced
        samples = discover_samples(dataset_dir / "train", label_map)
        num_classes = max(label_map.values()) + 1
        sampler = create_weighted_sampler(samples, num_classes)

        transform = get_transforms(64, augment=False)
        ds = BirdDataset(samples, transform=transform)
        loader = DataLoader(ds, batch_size=16, sampler=sampler)

        label_counts = np.zeros(num_classes)
        for _, labels in loader:
            for l in labels.numpy():
                label_counts[l] += 1

        rare_ratio = label_counts[0] / label_counts.sum()
        assert rare_ratio > 0.15, f"Espèce rare sous-représentée : {rare_ratio:.2%}"

    def test_handles_empty_classes(self):
        samples = [("a.jpg", 0, None), ("b.jpg", 0, None), ("c.jpg", 2, None)]
        sampler = create_weighted_sampler(samples, num_classes=10)
        assert len(sampler) == 3


class TestTrainOneEpoch:
    """Un epoch d'entraînement complet tourne sans erreur et produit des métriques."""

    @pytest.fixture
    def training_setup(self, mini_dataset):
        dataset_dir, label_map, _ = mini_dataset
        train_samples = discover_samples(dataset_dir / "train", label_map)
        num_classes = max(label_map.values()) + 1

        transform = get_transforms(64, augment=True)
        ds = BirdDataset(train_samples, transform=transform)
        loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

        model, _, _ = create_model("mobilenetv2", num_classes=num_classes, pretrained=False)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        return model, loader, criterion, optimizer

    def test_runs_without_error(self, training_setup):
        model, loader, criterion, optimizer = training_setup
        loss, acc = train_one_epoch(model, loader, criterion, optimizer, torch.device("cpu"))
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss > 0
        assert 0.0 <= acc <= 1.0

    def test_loss_is_finite(self, training_setup):
        model, loader, criterion, optimizer = training_setup
        loss, _ = train_one_epoch(model, loader, criterion, optimizer, torch.device("cpu"))
        assert np.isfinite(loss)


class TestEvaluate:
    """Évaluation du modèle sur un jeu de validation."""

    @pytest.fixture
    def eval_setup(self, mini_dataset):
        dataset_dir, label_map, _ = mini_dataset
        val_samples = discover_samples(dataset_dir / "validation", label_map)
        num_classes = max(label_map.values()) + 1

        transform = get_transforms(64, augment=False)
        ds = BirdDataset(val_samples, transform=transform)
        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

        model, _, _ = create_model("mobilenetv2", num_classes=num_classes, pretrained=False)
        criterion = nn.CrossEntropyLoss()

        return model, loader, criterion

    def test_returns_predictions_and_labels(self, eval_setup):
        model, loader, criterion = eval_setup
        loss, acc, preds, labels = evaluate(model, loader, criterion, torch.device("cpu"))
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert isinstance(preds, np.ndarray)
        assert isinstance(labels, np.ndarray)
        assert len(preds) == len(labels)

    def test_predictions_are_valid_class_indices(self, eval_setup):
        model, loader, criterion = eval_setup
        _, _, preds, _ = evaluate(model, loader, criterion, torch.device("cpu"))
        assert all(0 <= p < 558 for p in preds)


class TestSaveCheckpoint:
    """Sauvegarde et restauration de checkpoints."""

    def test_checkpoint_contains_required_keys(self, tmp_path):
        model, _, _ = create_model("mobilenetv2", num_classes=5, pretrained=False)
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        label_map = {"a": 0, "b": 1}

        path = tmp_path / "checkpoint.pth"
        save_checkpoint(path, model, optimizer, scheduler, epoch=5, best_acc=0.85,
                        label_map=label_map, arch="mobilenetv2")

        ckpt = torch.load(path, weights_only=False)
        assert "model_state_dict" in ckpt
        assert "optimizer_state_dict" in ckpt
        assert "scheduler_state_dict" in ckpt
        assert ckpt["epoch"] == 5
        assert ckpt["best_acc"] == 0.85
        assert ckpt["label_map"] == label_map
        assert ckpt["arch"] == "mobilenetv2"

    def test_restored_model_same_output(self, tmp_path):
        model, _, _ = create_model("mobilenetv2", num_classes=5, pretrained=False)
        model.eval()
        dummy = torch.randn(1, 3, 224, 224)
        original_output = model(dummy).detach()

        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        path = tmp_path / "ckpt.pth"
        save_checkpoint(path, model, optimizer, scheduler, 0, 0.0, {}, "mobilenetv2")

        restored, _, _ = create_model("mobilenetv2", num_classes=5, pretrained=False)
        ckpt = torch.load(path, weights_only=False)
        restored.load_state_dict(ckpt["model_state_dict"])
        restored.eval()
        restored_output = restored(dummy).detach()

        assert torch.allclose(original_output, restored_output)


@pytest.mark.slow
class TestIntegrationTraining:
    """Mini-entraînement de bout en bout sur un subset pour vérifier le pipeline complet."""

    def test_full_pipeline_runs(self, mini_dataset):
        dataset_dir, label_map, _ = mini_dataset
        train_samples = discover_samples(dataset_dir / "train", label_map)
        val_samples = discover_samples(dataset_dir / "validation", label_map)
        num_classes = max(label_map.values()) + 1

        train_ds = BirdDataset(train_samples, get_transforms(64, augment=True))
        val_ds = BirdDataset(val_samples, get_transforms(64, augment=False))

        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

        model, _, _ = create_model("mobilenetv2", num_classes=num_classes, pretrained=False)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        device = torch.device("cpu")

        losses = []
        for epoch in range(3):
            train_loss, _ = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
            losses.append(train_loss)

        assert all(np.isfinite(l) for l in losses)

    def test_loss_decreases(self, mini_dataset):
        dataset_dir, label_map, _ = mini_dataset
        train_samples = discover_samples(dataset_dir / "train", label_map)
        num_classes = max(label_map.values()) + 1

        ds = BirdDataset(train_samples, get_transforms(64, augment=False))
        loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)

        model, _, _ = create_model("mobilenetv2", num_classes=num_classes, pretrained=False)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        device = torch.device("cpu")

        losses = []
        for _ in range(5):
            loss, _ = train_one_epoch(model, loader, criterion, optimizer, device)
            losses.append(loss)

        assert losses[-1] < losses[0], f"La loss n'a pas diminué : {losses[0]:.4f} → {losses[-1]:.4f}"

    def test_not_single_class_prediction(self, mini_dataset):
        seed_everything(42)
        dataset_dir, label_map, _ = mini_dataset
        train_samples = discover_samples(dataset_dir / "train", label_map)
        val_samples = discover_samples(dataset_dir / "validation", label_map)
        num_classes = max(label_map.values()) + 1

        train_ds = BirdDataset(train_samples, get_transforms(64, augment=False))
        val_ds = BirdDataset(val_samples, get_transforms(64, augment=False))
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

        model, _, _ = create_model("mobilenetv2", num_classes=num_classes, pretrained=False)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        device = torch.device("cpu")

        for _ in range(15):
            train_one_epoch(model, train_loader, criterion, optimizer, device)

        _, _, preds, _ = evaluate(model, val_loader, criterion, device)
        unique_preds = set(preds)
        assert len(unique_preds) > 1, "Le modèle prédit toujours la même classe"


class TestLabelSmoothing:
    """Label smoothing réduit l'overfitting en adoucissant les cibles."""

    def test_label_smoothing_changes_loss(self, mini_dataset):
        dataset_dir, label_map, _ = mini_dataset
        samples = discover_samples(dataset_dir / "train", label_map)
        num_classes = max(label_map.values()) + 1
        ds = BirdDataset(samples, get_transforms(64, augment=False))
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

        model, _, _ = create_model("mobilenetv2", num_classes=num_classes, pretrained=False)
        model.eval()
        device = torch.device("cpu")

        criterion_hard = nn.CrossEntropyLoss(label_smoothing=0.0)
        criterion_smooth = nn.CrossEntropyLoss(label_smoothing=0.1)

        images, labels = next(iter(loader))
        with torch.no_grad():
            outputs = model(images)
        loss_hard = criterion_hard(outputs, labels).item()
        loss_smooth = criterion_smooth(outputs, labels).item()
        assert loss_smooth != loss_hard

    def test_parse_args_label_smoothing(self):
        from train import parse_args
        args = parse_args(["--label-smoothing", "0.2"])
        assert args.label_smoothing == 0.2

    def test_parse_args_label_smoothing_default(self):
        from train import parse_args
        args = parse_args([])
        assert args.label_smoothing == 0.1


class TestDropout:
    """Dropout configurable sur la tête de classification."""

    def test_create_model_with_dropout(self):
        model, _, _ = create_model("mobilenetv2", num_classes=10, pretrained=False, dropout=0.5)
        has_dropout = any(
            isinstance(m, nn.Dropout) and m.p == 0.5
            for m in model.classifier.modules()
        )
        assert has_dropout

    def test_create_model_efficientnet_with_dropout(self):
        model, _, _ = create_model("efficientnet_b0", num_classes=10, pretrained=False, dropout=0.4)
        has_dropout = any(
            isinstance(m, nn.Dropout) and m.p == 0.4
            for m in model.classifier.modules()
        )
        assert has_dropout

    def test_output_shape_unchanged_with_dropout(self):
        model, _, _ = create_model("mobilenetv2", num_classes=558, pretrained=False, dropout=0.5)
        dummy = torch.randn(2, 3, 224, 224)
        output = model(dummy)
        assert output.shape == (2, 558)

    def test_parse_args_dropout(self):
        from train import parse_args
        args = parse_args(["--dropout", "0.4"])
        assert args.dropout == 0.4


class TestAugmentation:
    """Augmentation renforcée avec RandAugment et RandomErasing."""

    def test_augmented_transform_includes_randaugment(self):
        t = get_transforms(224, augment=True)
        transform_types = [type(tr).__name__ for tr in t.transforms]
        assert "RandAugment" in transform_types

    def test_augmented_transform_includes_erasing(self):
        t = get_transforms(224, augment=True)
        transform_types = [type(tr).__name__ for tr in t.transforms]
        assert "RandomErasing" in transform_types

    def test_augmented_output_shape(self):
        t = get_transforms(224, augment=True)
        img = Image.new("RGB", (300, 400))
        tensor = t(img)
        assert tensor.shape == (3, 224, 224)

    def test_eval_transform_unchanged(self):
        t = get_transforms(224, augment=False)
        transform_types = [type(tr).__name__ for tr in t.transforms]
        assert "RandAugment" not in transform_types
        assert "RandomErasing" not in transform_types


class TestFreezeBackbone:
    """Gel du backbone les premières epochs."""

    def test_freeze_disables_gradients(self):
        model, backbone_params, head_params = create_model(
            "mobilenetv2", num_classes=10, pretrained=False
        )
        freeze_backbone(model, "mobilenetv2", freeze=True)
        assert all(not p.requires_grad for p in model.features.parameters())
        assert all(p.requires_grad for p in model.classifier.parameters())

    def test_unfreeze_enables_gradients(self):
        model, _, _ = create_model("mobilenetv2", num_classes=10, pretrained=False)
        freeze_backbone(model, "mobilenetv2", freeze=True)
        freeze_backbone(model, "mobilenetv2", freeze=False)
        assert all(p.requires_grad for p in model.features.parameters())

    def test_parse_args_freeze_epochs(self):
        from train import parse_args
        args = parse_args(["--freeze-backbone-epochs", "5"])
        assert args.freeze_backbone_epochs == 5

    def test_parse_args_freeze_default(self):
        from train import parse_args
        args = parse_args([])
        assert args.freeze_backbone_epochs == 3


class TestEarlyStopping:
    """Arrêt anticipé quand val_acc ne s'améliore plus."""

    def test_no_stop_when_improving(self):
        es = EarlyStopping(patience=3)
        for epoch, acc in enumerate([0.1, 0.2, 0.3, 0.4, 0.5]):
            assert es.step(acc, epoch) is False

    def test_stops_after_patience_stagnation(self):
        es = EarlyStopping(patience=3)
        es.step(0.8, 0)
        es.step(0.7, 1)
        es.step(0.6, 2)
        assert es.step(0.5, 3) is True

    def test_tiny_improvement_resets_counter(self):
        es = EarlyStopping(patience=3)
        es.step(0.8000, 0)
        es.step(0.7, 1)
        es.step(0.7, 2)
        assert es.step(0.8001, 3) is False
        assert es.counter == 0

    def test_patience_zero_stops_immediately(self):
        es = EarlyStopping(patience=0)
        es.step(0.5, 0)
        assert es.step(0.4, 1) is True

    def test_best_epoch_and_acc_accessible(self):
        es = EarlyStopping(patience=2)
        es.step(0.5, 0)
        es.step(0.9, 1)
        es.step(0.8, 2)
        es.step(0.7, 3)
        assert es.best_epoch == 1
        assert es.best_acc == 0.9

    def test_counter_resets_on_improvement(self):
        es = EarlyStopping(patience=3)
        es.step(0.5, 0)
        es.step(0.4, 1)
        es.step(0.3, 2)
        assert es.counter == 2
        es.step(0.6, 3)
        assert es.counter == 0


class TestLRWarmup:
    """Warmup linéaire du learning rate avant cosine decay."""

    @pytest.fixture
    def optimizer(self):
        model, _, _ = create_model("mobilenetv2", num_classes=5, pretrained=False)
        return torch.optim.AdamW(model.parameters(), lr=1e-3)

    def test_lr_starts_low_with_warmup(self, optimizer):
        scheduler = create_scheduler(optimizer, epochs=30, warmup_epochs=3)
        initial_lr = optimizer.param_groups[0]["lr"]
        assert initial_lr == pytest.approx(1e-3 * 0.04, rel=0.01)

    def test_lr_reaches_target_after_warmup(self, optimizer):
        scheduler = create_scheduler(optimizer, epochs=30, warmup_epochs=3)
        for _ in range(3):
            scheduler.step()
        lr_after_warmup = optimizer.param_groups[0]["lr"]
        assert lr_after_warmup == pytest.approx(1e-3, rel=0.01)

    def test_lr_decays_after_warmup(self, optimizer):
        scheduler = create_scheduler(optimizer, epochs=30, warmup_epochs=3)
        for _ in range(3):
            scheduler.step()
        lr_at_warmup_end = optimizer.param_groups[0]["lr"]
        for _ in range(10):
            scheduler.step()
        lr_later = optimizer.param_groups[0]["lr"]
        assert lr_later < lr_at_warmup_end

    def test_no_warmup_when_zero(self, optimizer):
        scheduler = create_scheduler(optimizer, epochs=30, warmup_epochs=0)
        initial_lr = optimizer.param_groups[0]["lr"]
        assert initial_lr == pytest.approx(1e-3)

    def test_scheduler_state_dict_round_trip(self, optimizer):
        scheduler = create_scheduler(optimizer, epochs=30, warmup_epochs=3)
        for _ in range(5):
            scheduler.step()
        state = scheduler.state_dict()
        scheduler2 = create_scheduler(optimizer, epochs=30, warmup_epochs=3)
        scheduler2.load_state_dict(state)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(
            optimizer.param_groups[0]["lr"]
        )


class TestReproducibility:
    """Reproductibilité des résultats via seed_everything."""

    def test_same_seed_same_random(self):
        seed_everything(42)
        t1 = torch.randn(3, 3)
        seed_everything(42)
        t2 = torch.randn(3, 3)
        assert torch.equal(t1, t2)

    def test_works_without_cuda(self):
        seed_everything(123)


# === Étape 1 — Defaults CLI (weight decay, backbone LR, dropout, epochs, patience) ===


class TestWeightDecayCLI:
    """Weight decay exposé en CLI au lieu du hardcoded 1e-4."""

    def test_parse_args_weight_decay_default(self):
        args = parse_args([])
        assert args.weight_decay == 1e-2

    def test_parse_args_weight_decay_custom(self):
        args = parse_args(["--weight-decay", "0.05"])
        assert args.weight_decay == 0.05


class TestDefaultsChanged:
    """Nouveaux defaults pour backbone-lr-factor, dropout, epochs, patience."""

    def test_backbone_lr_factor_default(self):
        args = parse_args([])
        assert args.backbone_lr_factor == 0.01

    def test_dropout_default(self):
        args = parse_args([])
        assert args.dropout == 0.5

    def test_epochs_default(self):
        args = parse_args([])
        assert args.epochs == 80

    def test_patience_default(self):
        args = parse_args([])
        assert args.patience == 15


# === Étape 2 — ColorJitter ===


class TestColorJitter:
    """ColorJitter ajouté dans les augmentations d'entraînement."""

    def test_augmented_transform_includes_colorjitter(self):
        t = get_transforms(224, augment=True)
        transform_types = [type(tr).__name__ for tr in t.transforms]
        assert "ColorJitter" in transform_types

    def test_colorjitter_not_in_eval(self):
        t = get_transforms(224, augment=False)
        transform_types = [type(tr).__name__ for tr in t.transforms]
        assert "ColorJitter" not in transform_types

    def test_colorjitter_before_randaugment(self):
        t = get_transforms(224, augment=True)
        transform_types = [type(tr).__name__ for tr in t.transforms]
        cj_idx = transform_types.index("ColorJitter")
        ra_idx = transform_types.index("RandAugment")
        assert cj_idx < ra_idx


# === Étape 3 — RandAugment CLI ===


class TestRandAugmentCLI:
    """RandAugment configurable via CLI."""

    def test_parse_args_randaugment_ops_default(self):
        args = parse_args([])
        assert args.randaugment_ops == 3

    def test_parse_args_randaugment_magnitude_default(self):
        args = parse_args([])
        assert args.randaugment_magnitude == 12

    def test_parse_args_randaugment_custom(self):
        args = parse_args(["--randaugment-ops", "4", "--randaugment-magnitude", "10"])
        assert args.randaugment_ops == 4
        assert args.randaugment_magnitude == 10

    def test_transforms_use_custom_randaugment_params(self):
        t = get_transforms(224, augment=True, randaugment_ops=4, randaugment_magnitude=10)
        for tr in t.transforms:
            if type(tr).__name__ == "RandAugment":
                assert tr.num_ops == 4
                assert tr.magnitude == 10
                break
        else:
            pytest.fail("RandAugment non trouvé dans les transforms")


# === Étape 4 — Gradient clipping ===


class TestGradientClipping:
    """Gradient clipping dans la boucle d'entraînement."""

    def test_parse_args_clip_grad_default(self):
        args = parse_args([])
        assert args.clip_grad == 1.0

    def test_parse_args_clip_grad_custom(self):
        args = parse_args(["--clip-grad", "5.0"])
        assert args.clip_grad == 5.0

    def test_parse_args_clip_grad_disabled(self):
        args = parse_args(["--clip-grad", "0"])
        assert args.clip_grad == 0.0

    def test_train_one_epoch_with_clip_grad(self, mini_dataset):
        dataset_dir, label_map, _ = mini_dataset
        train_samples = discover_samples(dataset_dir / "train", label_map)
        num_classes = max(label_map.values()) + 1
        transform = get_transforms(64, augment=False)
        ds = BirdDataset(train_samples, transform=transform)
        loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
        model, _, _ = create_model("mobilenetv2", num_classes=num_classes, pretrained=False)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss, acc = train_one_epoch(
            model, loader, criterion, optimizer, torch.device("cpu"), clip_grad=1.0
        )
        assert isinstance(loss, float)
        assert np.isfinite(loss)

    def test_gradients_are_clipped(self):
        model = nn.Linear(10, 5)
        x = torch.randn(2, 10) * 1000
        y = torch.tensor([0, 1])
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        for p in model.parameters():
            if p.grad is not None:
                assert p.grad.norm().item() <= 1.0 + 1e-6


# === Étape 5 — EarlyStopping mode min/max ===


class TestEarlyStoppingMinMode:
    """EarlyStopping en mode 'min' pour surveiller val_loss."""

    def test_min_mode_no_stop_when_decreasing(self):
        es = EarlyStopping(patience=3, mode="min")
        for epoch, loss in enumerate([1.0, 0.9, 0.8, 0.7]):
            assert es.step(loss, epoch) is False

    def test_min_mode_stops_after_patience(self):
        es = EarlyStopping(patience=3, mode="min")
        es.step(0.5, 0)
        es.step(0.6, 1)
        es.step(0.7, 2)
        assert es.step(0.8, 3) is True

    def test_min_mode_best_value_tracked(self):
        es = EarlyStopping(patience=2, mode="min")
        es.step(1.0, 0)
        es.step(0.3, 1)
        es.step(0.5, 2)
        assert es.best_value == 0.3
        assert es.best_epoch == 1

    def test_min_mode_tiny_improvement_resets(self):
        es = EarlyStopping(patience=3, mode="min")
        es.step(0.5, 0)
        es.step(0.6, 1)
        es.step(0.7, 2)
        assert es.step(0.4999, 3) is False
        assert es.counter == 0

    def test_max_mode_backward_compat(self):
        es = EarlyStopping(patience=3, mode="max")
        for epoch, acc in enumerate([0.1, 0.2, 0.3, 0.4]):
            assert es.step(acc, epoch) is False

    def test_best_acc_alias(self):
        es = EarlyStopping(patience=2, mode="max")
        es.step(0.5, 0)
        es.step(0.9, 1)
        assert es.best_acc == 0.9
        assert es.best_value == 0.9

    def test_default_mode_is_max(self):
        es = EarlyStopping(patience=3)
        es.step(0.5, 0)
        es.step(0.4, 1)
        assert es.counter == 1


# === Étape 6 — MixUp + CutMix ===


class TestMixUpCutMix:
    """MixUp et CutMix au niveau batch pour régularisation."""

    def test_parse_args_mixup_defaults(self):
        args = parse_args([])
        assert args.mixup_alpha == 0.2
        assert args.cutmix_alpha == 1.0
        assert args.no_mixup is False

    def test_parse_args_no_mixup(self):
        args = parse_args(["--no-mixup"])
        assert args.no_mixup is True

    def test_returns_soft_labels(self):
        images = torch.randn(8, 3, 64, 64)
        labels = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])
        mixed_images, soft_labels = apply_mixup_cutmix(
            images, labels, num_classes=5, mixup_alpha=0.2, cutmix_alpha=1.0
        )
        assert mixed_images.shape == images.shape
        assert soft_labels.shape == (8, 5)
        assert torch.allclose(soft_labels.sum(dim=1), torch.ones(8), atol=1e-5)

    def test_soft_labels_are_probabilities(self):
        images = torch.randn(4, 3, 32, 32)
        labels = torch.tensor([0, 1, 2, 3])
        _, soft_labels = apply_mixup_cutmix(images, labels, num_classes=5)
        assert (soft_labels >= 0).all()
        assert (soft_labels <= 1).all()

    def test_loss_works_with_soft_labels(self):
        images = torch.randn(4, 3, 32, 32)
        labels = torch.tensor([0, 1, 2, 3])
        _, soft_labels = apply_mixup_cutmix(images, labels, num_classes=5)
        logits = torch.randn(4, 5)
        import torch.nn.functional as F
        loss = F.cross_entropy(logits, soft_labels)
        assert loss.dim() == 0
        assert np.isfinite(loss.item())

    def test_train_one_epoch_with_mixup(self, mini_dataset):
        dataset_dir, label_map, _ = mini_dataset
        train_samples = discover_samples(dataset_dir / "train", label_map)
        num_classes = max(label_map.values()) + 1
        transform = get_transforms(64, augment=False)
        ds = BirdDataset(train_samples, transform=transform)
        loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
        model, _, _ = create_model("mobilenetv2", num_classes=num_classes, pretrained=False)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        mixup_fn = lambda imgs, lbls: apply_mixup_cutmix(imgs, lbls, num_classes)
        loss, acc = train_one_epoch(
            model, loader, criterion, optimizer, torch.device("cpu"),
            mixup_fn=mixup_fn, num_classes=num_classes,
        )
        assert isinstance(loss, float)
        assert np.isfinite(loss)


# === Étape 7 — ModelEMA ===


class TestModelEMA:
    """Exponential Moving Average des poids du modèle."""

    def test_parse_args_ema_defaults(self):
        args = parse_args([])
        assert args.ema_decay == 0.9999
        assert args.no_ema is False

    def test_parse_args_no_ema(self):
        args = parse_args(["--no-ema"])
        assert args.no_ema is True

    def test_ema_initialization(self):
        model, _, _ = create_model("mobilenetv2", num_classes=5, pretrained=False)
        ema = ModelEMA(model, decay=0.9999)
        for ema_p, p in zip(ema.module.parameters(), model.parameters()):
            assert torch.equal(ema_p.data, p.data)

    def test_ema_update_moves_shadow(self):
        model = nn.Linear(10, 5)
        ema = ModelEMA(model, decay=0.9)
        original_weight = ema.module.weight.data.clone()
        model.weight.data += 1.0
        ema.update(model)
        assert not torch.equal(ema.module.weight.data, original_weight)
        assert not torch.equal(ema.module.weight.data, model.weight.data)

    def test_ema_eval_produces_output(self):
        model, _, _ = create_model("mobilenetv2", num_classes=5, pretrained=False)
        ema = ModelEMA(model, decay=0.9999)
        ema.module.eval()
        dummy = torch.randn(1, 3, 224, 224)
        output = ema.module(dummy)
        assert output.shape == (1, 5)


# === Étape 8 — EfficientNetV2-B2 (timm) ===


class TestEfficientNetV2B2:
    """Support de EfficientNetV2-B2 via timm."""

    def test_parse_args_model_choice(self):
        args = parse_args(["--model", "efficientnetv2_b2"])
        assert args.model == "efficientnetv2_b2"

    def test_output_shape(self):
        model, _, _ = create_model("efficientnetv2_b2", num_classes=10, pretrained=False)
        output = model(torch.randn(1, 3, 260, 260))
        assert output.shape == (1, 10)

    def test_freeze_backbone(self):
        model, _, _ = create_model("efficientnetv2_b2", num_classes=10, pretrained=False)
        freeze_backbone(model, "efficientnetv2_b2", freeze=True)
        backbone_frozen = all(
            not p.requires_grad for n, p in model.named_parameters()
            if not n.startswith("classifier")
        )
        assert backbone_frozen
        assert all(p.requires_grad for p in model.classifier.parameters())

    def test_build_optimizer_groups(self):
        model, _, _ = create_model("efficientnetv2_b2", num_classes=10, pretrained=False)
        groups = build_optimizer_groups(
            model, "efficientnetv2_b2", lr=1e-3, backbone_lr_factor=0.1, weight_decay=1e-4
        )
        assert len(groups) == 4
        total_model = sum(p.numel() for p in model.parameters())
        total_groups = sum(p.numel() for g in groups for p in g["params"])
        assert total_groups == total_model

    def test_dropout(self):
        model, _, _ = create_model("efficientnetv2_b2", num_classes=10, pretrained=False, dropout=0.4)
        has_dropout = any(
            isinstance(m, nn.Dropout) and m.p == 0.4
            for m in model.classifier.modules()
        )
        assert has_dropout


# === Étape 9 — ViT-B/16 ===


class TestViTB16:
    """Support de Vision Transformer ViT-B/16."""

    def test_parse_args_model_choice(self):
        args = parse_args(["--model", "vit_b_16"])
        assert args.model == "vit_b_16"

    def test_output_shape(self):
        model, _, _ = create_model("vit_b_16", num_classes=10, pretrained=False)
        output = model(torch.randn(1, 3, 224, 224))
        assert output.shape == (1, 10)

    def test_param_groups_disjoint(self):
        model, backbone, head = create_model("vit_b_16", num_classes=10, pretrained=False)
        backbone_ids = {id(p) for p in backbone}
        head_ids = {id(p) for p in head}
        assert backbone_ids.isdisjoint(head_ids)

    def test_param_groups_cover_all(self):
        model, backbone, head = create_model("vit_b_16", num_classes=10, pretrained=False)
        total = sum(p.numel() for p in model.parameters())
        covered = sum(p.numel() for p in backbone) + sum(p.numel() for p in head)
        assert covered == total

    def test_freeze_backbone(self):
        model, _, _ = create_model("vit_b_16", num_classes=10, pretrained=False)
        freeze_backbone(model, "vit_b_16", freeze=True)
        assert all(not p.requires_grad for p in model.encoder.parameters())
        assert all(not p.requires_grad for p in model.conv_proj.parameters())
        assert all(p.requires_grad for p in model.heads.parameters())

    def test_unfreeze(self):
        model, _, _ = create_model("vit_b_16", num_classes=10, pretrained=False)
        freeze_backbone(model, "vit_b_16", freeze=True)
        freeze_backbone(model, "vit_b_16", freeze=False)
        assert all(p.requires_grad for p in model.encoder.parameters())

    def test_dropout(self):
        model, _, _ = create_model("vit_b_16", num_classes=10, pretrained=False, dropout=0.5)
        has_dropout = any(
            isinstance(m, nn.Dropout) and m.p == 0.5
            for m in model.heads.modules()
        )
        assert has_dropout

    def test_build_optimizer_groups(self):
        model, _, _ = create_model("vit_b_16", num_classes=10, pretrained=False)
        groups = build_optimizer_groups(
            model, "vit_b_16", lr=1e-3, backbone_lr_factor=0.1, weight_decay=1e-4
        )
        assert len(groups) == 4
        total_model = sum(p.numel() for p in model.parameters())
        total_groups = sum(p.numel() for g in groups for p in g["params"])
        assert total_groups == total_model


# === Étape 10 — FocalLoss ===


class TestFocalLoss:
    """Focal Loss pour gérer le déséquilibre de classes."""

    def test_parse_args_focal_defaults(self):
        args = parse_args([])
        assert args.focal_loss is False
        assert args.focal_gamma == 2.0

    def test_parse_args_focal_enabled(self):
        args = parse_args(["--focal-loss", "--focal-gamma", "3.0"])
        assert args.focal_loss is True
        assert args.focal_gamma == 3.0

    def test_with_hard_labels(self):
        criterion = FocalLoss(gamma=2.0, label_smoothing=0.0)
        logits = torch.randn(8, 10)
        labels = torch.randint(0, 10, (8,))
        loss = criterion(logits, labels)
        assert loss.dim() == 0
        assert loss.item() > 0
        assert np.isfinite(loss.item())

    def test_with_soft_labels(self):
        criterion = FocalLoss(gamma=2.0, label_smoothing=0.0)
        logits = torch.randn(4, 5)
        soft_labels = torch.zeros(4, 5)
        soft_labels[0, 0] = 0.8
        soft_labels[0, 1] = 0.2
        soft_labels[1, 1] = 1.0
        soft_labels[2, 2] = 0.5
        soft_labels[2, 3] = 0.5
        soft_labels[3, 4] = 1.0
        loss = criterion(logits, soft_labels)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_with_label_smoothing(self):
        criterion_smooth = FocalLoss(gamma=2.0, label_smoothing=0.1)
        criterion_hard = FocalLoss(gamma=2.0, label_smoothing=0.0)
        logits = torch.randn(8, 10)
        labels = torch.randint(0, 10, (8,))
        loss_smooth = criterion_smooth(logits, labels)
        loss_hard = criterion_hard(logits, labels)
        assert loss_smooth.item() != loss_hard.item()

    def test_reduces_easy_example_weight(self):
        focal = FocalLoss(gamma=2.0, label_smoothing=0.0)
        ce = FocalLoss(gamma=0.0, label_smoothing=0.0)
        logits = torch.tensor([[3.0, -1.0, -1.0]])
        labels = torch.tensor([0])
        focal_loss = focal(logits, labels)
        ce_loss = ce(logits, labels)
        assert focal_loss.item() < ce_loss.item()

    def test_gradient_flows(self):
        criterion = FocalLoss(gamma=2.0, label_smoothing=0.1)
        model = nn.Linear(10, 5)
        x = torch.randn(4, 10)
        labels = torch.randint(0, 5, (4,))
        loss = criterion(model(x), labels)
        loss.backward()
        assert all(p.grad is not None for p in model.parameters())


# === Étape 11 — QAT (Quantization-Aware Training) ===


class TestQATCLI:
    """Arguments CLI pour Quantization-Aware Training."""

    def test_parse_args_qat_default_false(self):
        args = parse_args([])
        assert args.qat is False

    def test_parse_args_qat_enabled(self):
        args = parse_args(["--qat"])
        assert args.qat is True

    def test_parse_args_qat_epochs_default(self):
        args = parse_args([])
        assert args.qat_epochs == 5

    def test_parse_args_qat_epochs_custom(self):
        args = parse_args(["--qat-epochs", "10"])
        assert args.qat_epochs == 10


class TestPrepareQAT:
    """Insertion de fake quantizers pour QAT via FX graph mode."""

    def test_prepare_qat_returns_model(self):
        from train import prepare_qat
        model, _, _ = create_model("mobilenetv2", num_classes=10, pretrained=False)
        qat_model = prepare_qat(model, "mobilenetv2")
        assert qat_model is not None

    def test_prepared_model_forward_works(self):
        from train import prepare_qat
        model, _, _ = create_model("mobilenetv2", num_classes=10, pretrained=False)
        qat_model = prepare_qat(model, "mobilenetv2")
        dummy = torch.randn(2, 3, 224, 224)
        output = qat_model(dummy)
        assert output.shape == (2, 10)

    def test_prepared_model_gradients_flow(self):
        from train import prepare_qat
        model, _, _ = create_model("mobilenetv2", num_classes=10, pretrained=False)
        qat_model = prepare_qat(model, "mobilenetv2")
        dummy = torch.randn(2, 3, 224, 224)
        output = qat_model(dummy)
        loss = output.sum()
        loss.backward()
        has_grad = any(
            p.grad is not None for p in qat_model.parameters() if p.requires_grad
        )
        assert has_grad

    def test_only_cnn_supported(self):
        from train import prepare_qat
        model, _, _ = create_model("vit_b_16", num_classes=10, pretrained=False)
        with pytest.raises(ValueError, match="QAT"):
            prepare_qat(model, "vit_b_16")


class TestConvertQAT:
    """Conversion du modèle QAT vers quantifié statique."""

    def test_convert_produces_model(self):
        from train import convert_qat, prepare_qat
        model, _, _ = create_model("mobilenetv2", num_classes=10, pretrained=False)
        qat_model = prepare_qat(model, "mobilenetv2")
        dummy = torch.randn(2, 3, 224, 224)
        qat_model(dummy)
        converted = convert_qat(qat_model)
        assert converted is not None

    def test_converted_model_forward_works(self):
        from train import convert_qat, prepare_qat
        model, _, _ = create_model("mobilenetv2", num_classes=10, pretrained=False)
        qat_model = prepare_qat(model, "mobilenetv2")
        dummy = torch.randn(2, 3, 224, 224)
        qat_model(dummy)
        converted = convert_qat(qat_model)
        output = converted(dummy)
        assert output.shape == (2, 10)


@pytest.mark.slow
class TestIntegrationQAT:
    """Pipeline QAT de bout en bout : train → QAT epochs → convert → export."""

    def test_qat_after_training(self, mini_dataset):
        from train import prepare_qat
        dataset_dir, label_map, _ = mini_dataset
        train_samples = discover_samples(dataset_dir / "train", label_map)
        num_classes = max(label_map.values()) + 1

        ds = BirdDataset(train_samples, get_transforms(224, augment=False))
        loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

        model, _, _ = create_model("mobilenetv2", num_classes=num_classes, pretrained=False)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        device = torch.device("cpu")

        for _ in range(2):
            train_one_epoch(model, loader, criterion, optimizer, device)

        qat_model = prepare_qat(model, "mobilenetv2")
        qat_optimizer = torch.optim.Adam(
            [p for p in qat_model.parameters() if p.requires_grad], lr=1e-4
        )
        for _ in range(2):
            train_one_epoch(qat_model, loader, criterion, qat_optimizer, device)

    def test_qat_model_exports_to_onnx(self, mini_dataset):
        from export import export_onnx
        from train import convert_qat, prepare_qat
        dataset_dir, label_map, _ = mini_dataset
        train_samples = discover_samples(dataset_dir / "train", label_map)
        num_classes = max(label_map.values()) + 1

        ds = BirdDataset(train_samples, get_transforms(224, augment=False))
        loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

        model, _, _ = create_model("mobilenetv2", num_classes=num_classes, pretrained=False)
        qat_model = prepare_qat(model, "mobilenetv2")
        dummy = torch.randn(2, 3, 224, 224)
        qat_model(dummy)
        converted = convert_qat(qat_model)

        onnx_path = dataset_dir / "qat_model.onnx"
        size_mb = export_onnx(converted, onnx_path, image_size=224)
        assert onnx_path.exists()
        assert size_mb > 0


# === Détection d'images problématiques ===


class TestFindProblematicImages:
    """Analyse per-sample pour détecter les images problématiques."""

    @pytest.fixture
    def analysis_setup(self, mini_dataset):
        dataset_dir, label_map, _ = mini_dataset
        train_samples = discover_samples(dataset_dir / "train", label_map)
        num_classes = max(label_map.values()) + 1
        model, _, _ = create_model("mobilenetv2", num_classes=num_classes, pretrained=False)
        eval_transform = get_transforms(64, augment=False)
        device = torch.device("cpu")
        return model, train_samples, eval_transform, device, label_map

    def test_returns_dict_with_required_keys(self, analysis_setup):
        model, samples, transform, device, label_map = analysis_setup
        result = find_problematic_images(model, samples, transform, device, label_map)
        assert isinstance(result, dict)
        assert "summary" in result
        assert "images" in result

    def test_images_have_correct_fields(self, analysis_setup):
        model, samples, transform, device, label_map = analysis_setup
        result = find_problematic_images(model, samples, transform, device, label_map)
        required = {"path", "loss", "label", "predicted", "label_name",
                    "predicted_name", "confidence", "correct"}
        for entry in result["images"]:
            assert required.issubset(entry.keys())
            assert isinstance(entry["path"], str)
            assert isinstance(entry["loss"], float)
            assert isinstance(entry["label"], int)
            assert isinstance(entry["predicted"], int)
            assert isinstance(entry["confidence"], float)
            assert isinstance(entry["correct"], bool)

    def test_images_sorted_by_loss_descending(self, analysis_setup):
        model, samples, transform, device, label_map = analysis_setup
        result = find_problematic_images(model, samples, transform, device, label_map)
        losses = [img["loss"] for img in result["images"]]
        assert losses == sorted(losses, reverse=True)

    def test_all_samples_included(self, analysis_setup):
        model, samples, transform, device, label_map = analysis_setup
        result = find_problematic_images(model, samples, transform, device, label_map)
        assert len(result["images"]) == len(samples)

    def test_summary_contains_statistics(self, analysis_setup):
        model, samples, transform, device, label_map = analysis_setup
        result = find_problematic_images(model, samples, transform, device, label_map)
        summary = result["summary"]
        assert summary["total_images"] == len(samples)
        assert isinstance(summary["problematic_count"], int)
        assert isinstance(summary["threshold"], float)
        assert isinstance(summary["mean_loss"], float)
        assert isinstance(summary["std_loss"], float)
        assert summary["sigma_k"] == 3.0

    def test_problematic_count_matches_threshold(self, analysis_setup):
        model, samples, transform, device, label_map = analysis_setup
        result = find_problematic_images(model, samples, transform, device, label_map)
        threshold = result["summary"]["threshold"]
        count = sum(1 for img in result["images"] if img["loss"] > threshold)
        assert count == result["summary"]["problematic_count"]

    def test_correct_field_matches_prediction(self, analysis_setup):
        model, samples, transform, device, label_map = analysis_setup
        result = find_problematic_images(model, samples, transform, device, label_map)
        for entry in result["images"]:
            assert entry["correct"] == (entry["label"] == entry["predicted"])

    def test_confidence_between_zero_and_one(self, analysis_setup):
        model, samples, transform, device, label_map = analysis_setup
        result = find_problematic_images(model, samples, transform, device, label_map)
        for entry in result["images"]:
            assert 0.0 <= entry["confidence"] <= 1.0

    def test_paths_match_input_samples(self, analysis_setup):
        model, samples, transform, device, label_map = analysis_setup
        result = find_problematic_images(model, samples, transform, device, label_map)
        result_paths = {img["path"] for img in result["images"]}
        sample_paths = {s[0] for s in samples}
        assert result_paths == sample_paths
