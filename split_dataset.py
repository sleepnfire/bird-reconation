"""
Split du dataset train/ en train/validation/test (80/10/10).

Déplace les images depuis train/ vers validation/ et test/,
en préservant la structure par espèce. Déterministe avec seed.
"""

import random
import shutil
from pathlib import Path


def split_dataset(
    dataset_dir: str,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> dict:
    dataset_dir = Path(dataset_dir)
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "validation"
    test_dir = dataset_dir / "test"

    if val_dir.exists() or test_dir.exists():
        raise FileExistsError(
            f"Les dossiers validation/ ou test/ existent déjà dans {dataset_dir}"
        )

    species_dirs = sorted(d for d in train_dir.iterdir() if d.is_dir())
    rng = random.Random(seed)

    stats = {"train": 0, "validation": 0, "test": 0, "species": len(species_dirs)}

    for sp_dir in species_dirs:
        images = sorted(
            f for f in sp_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        rng.shuffle(images)

        n = len(images)
        n_val = max(1, round(n * val_ratio))
        n_test = max(1, round(n * (1 - train_ratio - val_ratio)))
        n_train = n - n_val - n_test

        if n_train < 1:
            n_val = max(1, (n - 1) // 2)
            n_test = max(1, n - 1 - n_val)
            n_train = n - n_val - n_test

        val_images = images[:n_val]
        test_images = images[n_val:n_val + n_test]

        val_sp = val_dir / sp_dir.name
        test_sp = test_dir / sp_dir.name
        val_sp.mkdir(parents=True, exist_ok=True)
        test_sp.mkdir(parents=True, exist_ok=True)

        for img in val_images:
            shutil.move(str(img), str(val_sp / img.name))
        for img in test_images:
            shutil.move(str(img), str(test_sp / img.name))

        stats["train"] += n_train
        stats["validation"] += n_val
        stats["test"] += n_test

    total = stats["train"] + stats["validation"] + stats["test"]
    print(f"Split terminé: {total} images réparties sur {stats['species']} espèces")
    print(f"  train:      {stats['train']} ({stats['train']/total*100:.0f}%)")
    print(f"  validation: {stats['validation']} ({stats['validation']/total*100:.0f}%)")
    print(f"  test:       {stats['test']} ({stats['test']/total*100:.0f}%)")

    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Split dataset train → train/validation/test")
    parser.add_argument("dataset_dir", nargs="?", default="dataset/europe")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split_dataset(args.dataset_dir, seed=args.seed)
