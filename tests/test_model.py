import torch
import pytest

from train import build_optimizer_groups, create_model, get_device


class TestCreateModel:
    """Construction du modèle avec tête de classification adaptée à 558 classes."""

    # MobileNetV2 a bien 558 sorties (pas les 1000 d'ImageNet)
    def test_mobilenetv2_output_558_classes(self):
        model, _, _ = create_model("mobilenetv2", num_classes=558, pretrained=False)
        dummy = torch.randn(1, 3, 224, 224)
        output = model(dummy)
        assert output.shape == (1, 558)

    # EfficientNet-B0 a bien 558 sorties
    def test_efficientnet_b0_output_558_classes(self):
        model, _, _ = create_model("efficientnet_b0", num_classes=558, pretrained=False)
        dummy = torch.randn(1, 3, 224, 224)
        output = model(dummy)
        assert output.shape == (1, 558)

    # Un batch de N images produit N lignes de prédictions
    def test_batch_dimension(self):
        model, _, _ = create_model("mobilenetv2", num_classes=10, pretrained=False)
        dummy = torch.randn(8, 3, 224, 224)
        output = model(dummy)
        assert output.shape == (8, 10)

    # Les paramètres backbone et head sont bien séparés (pour le LR différentiel)
    def test_param_groups_are_disjoint(self):
        model, backbone_params, head_params = create_model("mobilenetv2", num_classes=10, pretrained=False)
        backbone_ids = {id(p) for p in backbone_params}
        head_ids = {id(p) for p in head_params}
        assert backbone_ids.isdisjoint(head_ids)

    # La somme backbone + head couvre tous les paramètres du modèle
    def test_param_groups_cover_all(self):
        model, backbone_params, head_params = create_model("mobilenetv2", num_classes=10, pretrained=False)
        total_model = sum(p.numel() for p in model.parameters())
        total_groups = sum(p.numel() for p in backbone_params) + sum(p.numel() for p in head_params)
        assert total_groups == total_model

    # Architecture inconnue → erreur claire
    def test_invalid_arch_raises(self):
        with pytest.raises(ValueError, match="inconnue"):
            create_model("resnet9000", num_classes=10)

    # Le nombre de classes peut varier (5 espèces de test, 558 en prod)
    def test_variable_num_classes(self):
        for n in [2, 5, 100, 558]:
            model, _, _ = create_model("mobilenetv2", num_classes=n, pretrained=False)
            output = model(torch.randn(1, 3, 224, 224))
            assert output.shape == (1, n)


class TestGetDevice:
    """Sélection automatique du device (CUDA > MPS > CPU)."""

    # Un device explicite est respecté
    def test_explicit_cpu(self):
        device = get_device("cpu")
        assert device == torch.device("cpu")

    # Sans argument, retourne un device valide
    def test_auto_returns_valid_device(self):
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ("cpu", "cuda", "mps")


class TestParamGroupsWeightDecay:
    """Séparation des paramètres pour exclure bias/BN du weight decay."""

    @pytest.fixture
    def groups(self):
        model, _, _ = create_model("mobilenetv2", num_classes=10, pretrained=False)
        return build_optimizer_groups(model, "mobilenetv2", lr=1e-3, backbone_lr_factor=0.1, weight_decay=1e-4)

    # Les groupes "decay" ne contiennent que des matrices de poids (dim >= 2)
    def test_decay_groups_only_weight_matrices(self, groups):
        for g in groups:
            if g["weight_decay"] > 0:
                assert all(p.dim() >= 2 for p in g["params"])

    # Les groupes "no_decay" ne contiennent que des bias/BN (dim < 2)
    def test_no_decay_groups_only_bias_bn(self, groups):
        for g in groups:
            if g["weight_decay"] == 0.0:
                assert all(p.dim() < 2 for p in g["params"])

    # Les 4 groupes couvrent tous les paramètres sans doublons
    def test_groups_cover_all_params_no_overlap(self, groups):
        model, _, _ = create_model("mobilenetv2", num_classes=10, pretrained=False)
        total_model = sum(p.numel() for p in model.parameters())
        total_groups = sum(p.numel() for g in groups for p in g["params"])
        assert total_groups == total_model
        all_ids = [id(p) for g in groups for p in g["params"]]
        assert len(all_ids) == len(set(all_ids))

    # Les groupes backbone ont le LR réduit (lr × 0.1)
    def test_backbone_lr_is_reduced(self, groups):
        backbone_lrs = [g["lr"] for g in groups if g["lr"] < 1e-3]
        assert len(backbone_lrs) == 2
        assert all(lr == pytest.approx(1e-4) for lr in backbone_lrs)

    # Les groupes no_decay ont weight_decay=0.0
    def test_no_decay_groups_have_zero_wd(self, groups):
        for g in groups:
            if any(p.dim() < 2 for p in g["params"]):
                assert g["weight_decay"] == 0.0
