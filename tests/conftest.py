import json
import random
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from train import BirdDataset, create_model, get_transforms, save_checkpoint


@pytest.fixture
def label_map():
    """Label map minimal avec 5 espèces pour les tests."""
    return {
        "parus_major": 0,
        "turdus_merula": 1,
        "erithacus_rubecula": 2,
        "passer_domesticus": 3,
        "fringilla_coelebs": 4,
    }


@pytest.fixture
def label_map_558():
    """Label map complet chargé depuis le vrai fichier."""
    path = Path("dataset/europe/label_map.json")
    if not path.exists():
        pytest.skip("label_map.json absent")
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def mini_dataset(tmp_path, label_map):
    """Crée un mini-dataset avec des images dans train/validation/test + annotations.json."""
    counts = {}

    for slug in label_map:
        n_train = random.randint(8, 15)
        n_val = random.randint(2, 4)
        n_test = random.randint(2, 4)
        counts[slug] = n_train

        for split_name, n in [("train", n_train), ("validation", n_val), ("test", n_test)]:
            species_dir = tmp_path / split_name / slug
            species_dir.mkdir(parents=True)
            annotations = {}
            for i in range(n):
                img = Image.new("RGB", (64, 64), color=(
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                ))
                img.save(species_dir / f"{i:04d}.jpg")
                annotations[f"{i:04d}.jpg"] = {
                    "bbox": [10, 10, 40, 40],
                    "score": 0.95,
                }
            with open(species_dir / "annotations.json", "w") as f:
                json.dump(annotations, f)

    with open(tmp_path / "label_map.json", "w") as f:
        json.dump(label_map, f)

    metadata = {
        slug: {
            "slug": slug,
            "scientific_name": slug.replace("_", " ").title(),
            "family": "Testidae",
            "english_name": f"Test {slug}",
            "french_name": f"Test {slug} FR",
        }
        for slug in label_map
    }
    with open(tmp_path / "metadata.json", "w") as f:
        json.dump(metadata, f)

    return tmp_path, label_map, counts


@pytest.fixture
def mini_dataset_imbalanced(tmp_path):
    """Dataset déséquilibré : 1 espèce a 50 images, les autres 5."""
    label_map = {"espece_rare": 0, "espece_commune": 1, "espece_moyenne": 2}
    train_dir = tmp_path / "train"
    image_counts = {"espece_rare": 5, "espece_commune": 50, "espece_moyenne": 10}

    for slug, count in image_counts.items():
        species_dir = train_dir / slug
        species_dir.mkdir(parents=True)
        annotations = {}
        for i in range(count):
            img = Image.new("RGB", (64, 64), color=(random.randint(0, 255), 0, 0))
            img.save(species_dir / f"{i:04d}.jpg")
            annotations[f"{i:04d}.jpg"] = {
                "bbox": [10, 10, 40, 40],
                "score": 0.95,
            }
        with open(species_dir / "annotations.json", "w") as f:
            json.dump(annotations, f)

    with open(tmp_path / "label_map.json", "w") as f:
        json.dump(label_map, f)

    return tmp_path, label_map, image_counts


@pytest.fixture
def checkpoint_path(tmp_path, label_map):
    """Crée un checkpoint MobileNetV2 avec 5 classes pour les tests."""
    model, _, _ = create_model("mobilenetv2", num_classes=5, pretrained=False)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    save_checkpoint(
        tmp_path / "test_checkpoint.pth",
        model, optimizer, scheduler,
        epoch=0, best_acc=0.0, label_map=label_map, arch="mobilenetv2",
    )
    return tmp_path / "test_checkpoint.pth"


@pytest.fixture
def vit_checkpoint_path(tmp_path, label_map):
    """Crée un checkpoint ViT-B/16 avec 5 classes (teacher pour la distillation)."""
    model, _, _ = create_model("vit_b_16", num_classes=5, pretrained=False)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    save_checkpoint(
        tmp_path / "vit_checkpoint.pth",
        model, optimizer, scheduler,
        epoch=0, best_acc=0.0, label_map=label_map, arch="vit_b_16",
    )
    return tmp_path / "vit_checkpoint.pth"


@pytest.fixture
def calibration_loader(mini_dataset):
    """DataLoader de calibration avec des images représentatives."""
    dataset_dir, label_map, _ = mini_dataset
    from train import discover_samples
    samples = discover_samples(dataset_dir / "train", label_map)
    transform = get_transforms(224, augment=False)
    ds = BirdDataset(samples, transform=transform)
    return DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
