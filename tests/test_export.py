"""Tests pour l'export ONNX et la quantification INT8 (TDD — tests d'abord)."""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from export import (
    IMX500_MAX_SIZE_MB,
    check_imx500_size,
    export_onnx,
    load_model_from_checkpoint,
    parse_args,
    quantize_onnx_int8,
)
from train import create_model, save_checkpoint


class TestLoadModelFromCheckpoint:
    """Chargement d'un modèle entraîné depuis un fichier checkpoint."""

    # Le modèle chargé a la bonne architecture et le bon nombre de classes
    def test_loads_correct_architecture(self, checkpoint_path):
        model, _, _ = load_model_from_checkpoint(checkpoint_path)
        output = model(torch.randn(1, 3, 224, 224))
        assert output.shape == (1, 5)

    # La fonction retourne aussi l'architecture et le label_map du checkpoint
    def test_returns_arch_and_label_map(self, checkpoint_path, label_map):
        _, arch, lm = load_model_from_checkpoint(checkpoint_path)
        assert arch == "mobilenetv2"
        assert lm == label_map

    # Le modèle chargé produit des sorties numériques valides (pas NaN/Inf)
    def test_model_produces_finite_output(self, checkpoint_path):
        model, _, _ = load_model_from_checkpoint(checkpoint_path)
        model.eval()
        with torch.no_grad():
            output = model(torch.randn(1, 3, 224, 224))
        assert torch.isfinite(output).all()

    # Fonctionne aussi avec un checkpoint EfficientNet-B0
    def test_efficientnet_checkpoint(self, tmp_path):
        lm = {"a": 0, "b": 1, "c": 2}
        model, _, _ = create_model("efficientnet_b0", num_classes=3, pretrained=False)
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        ckpt_path = tmp_path / "effnet.pth"
        save_checkpoint(ckpt_path, model, optimizer, scheduler, 0, 0.0, lm, "efficientnet_b0")

        loaded, arch, _ = load_model_from_checkpoint(ckpt_path)
        assert arch == "efficientnet_b0"
        assert loaded(torch.randn(1, 3, 224, 224)).shape == (1, 3)

    # Un chemin invalide lève une erreur claire
    def test_invalid_checkpoint_raises(self, tmp_path):
        with pytest.raises((FileNotFoundError, RuntimeError)):
            load_model_from_checkpoint(tmp_path / "nonexistent.pth")


class TestExportOnnx:
    """Export d'un modèle PyTorch au format ONNX float32."""

    # L'export crée un fichier ONNX sur disque
    def test_creates_onnx_file(self, checkpoint_path, tmp_path):
        model, _, _ = load_model_from_checkpoint(checkpoint_path)
        output_path = tmp_path / "model.onnx"
        export_onnx(model, output_path)
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    # Le fichier ONNX passe la validation onnx.checker
    def test_onnx_model_is_valid(self, checkpoint_path, tmp_path):
        model, _, _ = load_model_from_checkpoint(checkpoint_path)
        output_path = tmp_path / "model.onnx"
        export_onnx(model, output_path)
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)

    # La forme de sortie ONNX correspond au modèle PyTorch (1, num_classes)
    def test_output_shape_matches_pytorch(self, checkpoint_path, tmp_path):
        model, _, _ = load_model_from_checkpoint(checkpoint_path)
        output_path = tmp_path / "model.onnx"
        export_onnx(model, output_path)
        onnx_model = onnx.load(str(output_path))
        output_shape = [d.dim_value for d in onnx_model.graph.output[0].type.tensor_type.shape.dim]
        assert output_shape[1] == 5

    # La fonction retourne la taille du fichier en Mo
    def test_returns_size_mb(self, checkpoint_path, tmp_path):
        model, _, _ = load_model_from_checkpoint(checkpoint_path)
        output_path = tmp_path / "model.onnx"
        size_mb = export_onnx(model, output_path)
        assert isinstance(size_mb, float)
        assert size_mb > 0

    # L'export fonctionne avec une taille d'image différente de 224
    def test_custom_image_size(self, checkpoint_path, tmp_path):
        model, _, _ = load_model_from_checkpoint(checkpoint_path)
        output_path = tmp_path / "model_128.onnx"
        export_onnx(model, output_path, image_size=128)
        onnx_model = onnx.load(str(output_path))
        input_shape = [d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]
        assert input_shape[2] == 128 and input_shape[3] == 128


class TestQuantizeOnnxInt8:
    """Quantification INT8 d'un modèle ONNX via onnxruntime."""

    @pytest.fixture
    def onnx_path(self, checkpoint_path, tmp_path):
        model, _, _ = load_model_from_checkpoint(checkpoint_path)
        path = tmp_path / "model.onnx"
        export_onnx(model, path)
        return path

    # La quantification crée un fichier ONNX INT8
    def test_creates_quantized_file(self, onnx_path, tmp_path):
        output_path = tmp_path / "model_int8.onnx"
        quantize_onnx_int8(onnx_path, output_path)
        assert output_path.exists()

    # Le modèle INT8 est plus petit que le float32
    def test_quantized_smaller_than_float32(self, onnx_path, tmp_path):
        float32_size = onnx_path.stat().st_size
        output_path = tmp_path / "model_int8.onnx"
        quantize_onnx_int8(onnx_path, output_path)
        assert output_path.stat().st_size < float32_size

    # La fonction retourne la taille en Mo
    def test_returns_size_mb(self, onnx_path, tmp_path):
        output_path = tmp_path / "model_int8.onnx"
        size_mb = quantize_onnx_int8(onnx_path, output_path)
        assert isinstance(size_mb, float)
        assert size_mb > 0

    # Le modèle quantifié peut être chargé par onnxruntime pour inférence
    def test_quantized_model_runs_in_ort(self, onnx_path, tmp_path):
        output_path = tmp_path / "model_int8.onnx"
        quantize_onnx_int8(onnx_path, output_path)
        session = ort.InferenceSession(str(output_path))
        input_name = session.get_inputs()[0].name
        dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
        results = session.run(None, {input_name: dummy})
        assert results[0].shape == (1, 5)


class TestIMX500SizeCheck:
    """Vérification que le modèle respecte la contrainte de 8 Mo SRAM de l'IMX500."""

    # Un modèle de 3 Mo passe la vérification
    def test_under_limit_returns_true(self):
        assert check_imx500_size(3.0) is True

    # Un modèle de 10 Mo dépasse la limite
    def test_over_limit_returns_false(self):
        assert check_imx500_size(10.0) is False

    # La limite exacte de 8 Mo est acceptée (<=)
    def test_exact_limit_passes(self):
        assert check_imx500_size(IMX500_MAX_SIZE_MB) is True

    # MobileNetV2 quantifié INT8 passe la contrainte IMX500
    def test_mobilenetv2_int8_passes(self, checkpoint_path, tmp_path):
        model, _, _ = load_model_from_checkpoint(checkpoint_path)
        onnx_path = tmp_path / "model.onnx"
        export_onnx(model, onnx_path)
        int8_path = tmp_path / "model_int8.onnx"
        size_mb = quantize_onnx_int8(onnx_path, int8_path)
        assert check_imx500_size(size_mb) is True


@pytest.mark.slow
class TestIntegrationExport:
    """Pipeline complet : checkpoint → ONNX → INT8 → vérification taille → inférence."""

    # La chaîne complète fonctionne de bout en bout
    def test_full_pipeline_checkpoint_to_int8(self, tmp_path, label_map):
        num_classes = len(label_map)
        model, _, _ = create_model("mobilenetv2", num_classes=num_classes, pretrained=False)
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        ckpt_path = tmp_path / "best_mobilenetv2.pth"
        save_checkpoint(ckpt_path, model, optimizer, scheduler, 10, 0.85, label_map, "mobilenetv2")

        loaded_model, arch, lm = load_model_from_checkpoint(ckpt_path)
        assert arch == "mobilenetv2"

        onnx_path = tmp_path / "model.onnx"
        onnx_size = export_onnx(loaded_model, onnx_path)
        assert onnx_path.exists()

        int8_path = tmp_path / "model_int8.onnx"
        int8_size = quantize_onnx_int8(onnx_path, int8_path)
        assert int8_size < onnx_size

        assert check_imx500_size(int8_size) is True

        session = ort.InferenceSession(str(int8_path))
        input_name = session.get_inputs()[0].name
        dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
        results = session.run(None, {input_name: dummy})
        assert results[0].shape == (1, num_classes)


# === Pipeline multi-cible (onnx, hailo, imx500) ===


class TestExportTargetCLI:
    """CLI multi-cible : --target onnx|hailo|imx500."""

    def test_parse_args_target_default_onnx(self):
        args = parse_args(["--checkpoint", "model.pth"])
        assert args.target == "onnx"

    def test_parse_args_target_hailo(self):
        args = parse_args(["--checkpoint", "model.pth", "--target", "hailo"])
        assert args.target == "hailo"

    def test_parse_args_target_imx500(self):
        args = parse_args(["--checkpoint", "model.pth", "--target", "imx500"])
        assert args.target == "imx500"

    def test_parse_args_calibration_images_default(self):
        args = parse_args(["--checkpoint", "model.pth"])
        assert args.calibration_images == 200

    def test_parse_args_calibration_images_custom(self):
        args = parse_args(["--checkpoint", "model.pth", "--calibration-images", "500"])
        assert args.calibration_images == 500

    def test_backward_compat_no_target(self):
        args = parse_args(["--checkpoint", "model.pth"])
        assert args.target == "onnx"


class TestExportHailo:
    """Export ONNX float32 optimisé pour le Hailo Dataflow Compiler."""

    def test_creates_onnx_file(self, checkpoint_path, tmp_path):
        from export import export_hailo
        model, _, _ = load_model_from_checkpoint(checkpoint_path)
        output_path = tmp_path / "model_hailo.onnx"
        export_hailo(model, output_path)
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_fixed_input_shape(self, checkpoint_path, tmp_path):
        from export import export_hailo
        model, _, _ = load_model_from_checkpoint(checkpoint_path)
        output_path = tmp_path / "model_hailo.onnx"
        export_hailo(model, output_path)
        onnx_model = onnx.load(str(output_path))
        input_shape = [d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]
        assert input_shape == [1, 3, 224, 224]

    def test_opset_version_13(self, checkpoint_path, tmp_path):
        from export import export_hailo
        model, _, _ = load_model_from_checkpoint(checkpoint_path)
        output_path = tmp_path / "model_hailo.onnx"
        export_hailo(model, output_path)
        onnx_model = onnx.load(str(output_path))
        assert onnx_model.opset_import[0].version >= 13

    def test_returns_size_mb(self, checkpoint_path, tmp_path):
        from export import export_hailo
        model, _, _ = load_model_from_checkpoint(checkpoint_path)
        output_path = tmp_path / "model_hailo.onnx"
        size_mb = export_hailo(model, output_path)
        assert isinstance(size_mb, float)
        assert size_mb > 0


class TestExportIMX500:
    """Export quantifié INT8 pour Sony IMX500 via model_compression_toolkit."""

    @pytest.fixture(autouse=True)
    def _require_mct(self):
        pytest.importorskip("model_compression_toolkit")

    def test_creates_file(self, checkpoint_path, calibration_loader, tmp_path):
        from export import export_imx500
        model, _, _ = load_model_from_checkpoint(checkpoint_path)
        output_path = tmp_path / "model_imx500.onnx"
        export_imx500(model, output_path, calibration_loader)
        assert output_path.exists()

    def test_model_contains_quantization_nodes(self, checkpoint_path, calibration_loader, tmp_path):
        from export import export_imx500
        model, _, _ = load_model_from_checkpoint(checkpoint_path)
        quant_path = tmp_path / "model_imx500.onnx"
        export_imx500(model, quant_path, calibration_loader)
        onnx_model = onnx.load(str(quant_path))
        node_types = {n.op_type for n in onnx_model.graph.node}
        assert node_types & {"QuantizeLinear", "DequantizeLinear", "QLinearConv"}

    def test_returns_size_mb(self, checkpoint_path, calibration_loader, tmp_path):
        from export import export_imx500
        model, _, _ = load_model_from_checkpoint(checkpoint_path)
        output_path = tmp_path / "model_imx500.onnx"
        size_mb = export_imx500(model, output_path, calibration_loader)
        assert isinstance(size_mb, float)
        assert size_mb > 0

    def test_runs_in_ort(self, checkpoint_path, calibration_loader, tmp_path):
        from export import export_imx500
        model, _, _ = load_model_from_checkpoint(checkpoint_path)
        output_path = tmp_path / "model_imx500.onnx"
        export_imx500(model, output_path, calibration_loader)
        session = ort.InferenceSession(str(output_path))
        input_name = session.get_inputs()[0].name
        dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
        results = session.run(None, {input_name: dummy})
        assert results[0].shape == (1, 5)


@pytest.mark.slow
class TestIntegrationMultiTargetExport:
    """Pipeline multi-cible de bout en bout."""

    def test_onnx_target_full_pipeline(self, tmp_path, label_map):
        num_classes = len(label_map)
        model, _, _ = create_model("mobilenetv2", num_classes=num_classes, pretrained=False)
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        ckpt_path = tmp_path / "best_mobilenetv2.pth"
        save_checkpoint(ckpt_path, model, optimizer, scheduler, 10, 0.85, label_map, "mobilenetv2")

        loaded_model, arch, _ = load_model_from_checkpoint(ckpt_path)
        onnx_path = tmp_path / "model.onnx"
        onnx_size = export_onnx(loaded_model, onnx_path)
        assert onnx_path.exists()

        int8_path = tmp_path / "model_int8.onnx"
        int8_size = quantize_onnx_int8(onnx_path, int8_path)
        assert int8_size < onnx_size

    def test_hailo_target_full_pipeline(self, tmp_path, label_map):
        from export import export_hailo
        num_classes = len(label_map)
        model, _, _ = create_model("mobilenetv2", num_classes=num_classes, pretrained=False)
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        ckpt_path = tmp_path / "best_mobilenetv2.pth"
        save_checkpoint(ckpt_path, model, optimizer, scheduler, 10, 0.85, label_map, "mobilenetv2")

        loaded_model, arch, _ = load_model_from_checkpoint(ckpt_path)
        hailo_path = tmp_path / "model_hailo.onnx"
        size_mb = export_hailo(loaded_model, hailo_path)
        assert hailo_path.exists()
        assert size_mb > 0

        onnx_model = onnx.load(str(hailo_path))
        onnx.checker.check_model(onnx_model)
        assert onnx_model.opset_import[0].version >= 13
