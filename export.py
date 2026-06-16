"""Export multi-cible des modèles entraînés.

Pipelines supportés :
  --target onnx    : ONNX float32 + quantification dynamique INT8 (onnxruntime)
  --target hailo   : ONNX float32 propre (opset 13+, input fixe) pour Hailo DFC
  --target imx500  : Quantification statique INT8 via Sony MCT pour IMX500
"""

import argparse
import json
import logging
import random
from pathlib import Path

import onnx
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train import BirdDataset, create_model, discover_samples, get_transforms

logger = logging.getLogger(__name__)

IMX500_MAX_SIZE_MB = 8.0


def load_model_from_checkpoint(checkpoint_path, device="cpu"):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arch = ckpt["arch"]
    label_map = ckpt["label_map"]
    num_classes = max(label_map.values()) + 1

    model, _, _ = create_model(arch, num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, arch, label_map


def export_onnx(model, output_path, image_size=224):
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        model, (dummy,), str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
    )
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    return size_mb


def export_hailo(model, output_path, image_size=224):
    """Export ONNX float32 avec input shape fixe et opset >= 13 pour Hailo DFC."""
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        model, (dummy,), str(output_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        dynamo=False,
    )
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    return size_mb


def export_imx500(model, output_path, calibration_loader, image_size=224):
    """Quantification statique INT8 via Sony MCT pour IMX500."""
    import model_compression_toolkit as mct

    def representative_dataset_gen():
        for images, _ in calibration_loader:
            yield [images.numpy()]

    tpc = mct.get_target_platform_capabilities(
        tpc_version="6.0", device_type="imx500",
    )

    quantized_model, _ = mct.ptq.pytorch_post_training_quantization(
        model,
        representative_dataset_gen,
        target_platform_capabilities=tpc,
    )

    dummy = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        quantized_model, (dummy,), str(output_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        dynamo=False,
    )
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    return size_mb


def create_calibration_loader(dataset_dir, label_map, num_images=200, image_size=224):
    """Crée un DataLoader de calibration depuis le dataset d'entraînement."""
    samples = discover_samples(Path(dataset_dir) / "train", label_map)
    if len(samples) > num_images:
        rng = random.Random(42)
        samples = rng.sample(samples, num_images)
    transform = get_transforms(image_size, augment=False)
    ds = BirdDataset(samples, transform=transform)
    return DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)


def quantize_onnx_int8(onnx_path, output_path):
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        str(onnx_path),
        str(output_path),
        weight_type=QuantType.QInt8,
    )
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    return size_mb


def check_imx500_size(size_mb):
    return size_mb <= IMX500_MAX_SIZE_MB


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Export ONNX multi-cible")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("output"))
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--target", choices=["onnx", "hailo", "imx500"], default="onnx")
    p.add_argument("--calibration-images", type=int, default=200,
                   help="Nombre d'images de calibration pour l'IMX500")
    p.add_argument("--dataset", type=Path, default=Path("dataset/europe"),
                   help="Dataset pour la calibration IMX500")
    p.add_argument("--check-size", action="store_true",
                   help="Vérifie que MobileNetV2 quantifié < 8 Mo (contrainte IMX500)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print(f"Chargement du checkpoint : {args.checkpoint}")
    model, arch, label_map = load_model_from_checkpoint(args.checkpoint)
    print(f"Architecture : {arch}, {len(label_map)} classes")

    if args.target == "hailo":
        hailo_path = args.output / f"{arch}_hailo.onnx"
        hailo_size = export_hailo(model, hailo_path, args.image_size)
        print(f"ONNX Hailo : {hailo_path} ({hailo_size:.1f} Mo)")
        print(f"\nPour compiler avec le Hailo DFC :")
        print(f"  hailo parser onnx {hailo_path}")
        print(f"  hailo optimize --hw-arch hailo10h")
        print(f"  hailo compile")

    elif args.target == "imx500":
        cal_loader = create_calibration_loader(
            args.dataset, label_map,
            num_images=args.calibration_images,
            image_size=args.image_size,
        )
        imx_path = args.output / f"{arch}_imx500.onnx"
        imx_size = export_imx500(model, imx_path, cal_loader, args.image_size)
        print(f"ONNX IMX500 : {imx_path} ({imx_size:.1f} Mo)")
        if check_imx500_size(imx_size):
            print(f"OK : {imx_size:.1f} Mo <= {IMX500_MAX_SIZE_MB} Mo (compatible IMX500)")
        else:
            print(f"ATTENTION : {imx_size:.1f} Mo > {IMX500_MAX_SIZE_MB} Mo (dépasse la limite)")

    else:
        onnx_path = args.output / f"{arch}.onnx"
        onnx_size = export_onnx(model, onnx_path, args.image_size)
        print(f"ONNX exporté : {onnx_path} ({onnx_size:.1f} Mo)")

        int8_path = args.output / f"{arch}_int8.onnx"
        int8_size = quantize_onnx_int8(onnx_path, int8_path)
        print(f"ONNX INT8 : {int8_path} ({int8_size:.1f} Mo)")

        if args.check_size and arch == "mobilenetv2":
            if check_imx500_size(int8_size):
                print(f"OK : {int8_size:.1f} Mo <= {IMX500_MAX_SIZE_MB} Mo (compatible IMX500)")
            else:
                print(f"ATTENTION : {int8_size:.1f} Mo > {IMX500_MAX_SIZE_MB} Mo (dépasse la limite)")

    with open(args.output / f"{arch}_label_map.json", "w") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)

    print(f"\nFichiers exportés dans {args.output}/")


if __name__ == "__main__":
    main()
