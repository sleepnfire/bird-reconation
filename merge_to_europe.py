#!/usr/bin/env python3
"""
Fusionne les images filtrées de europe_to_trait/ dans le dataset europe/.

Après le pipeline annotation + filtrage qualité sur europe_to_trait/,
déplace les images restantes (bonnes) vers europe/{train,val,test}/
en respectant le ratio 80/10/10 et fusionne les annotations.json.

Usage:
    python merge_to_europe.py --dry-run
    python merge_to_europe.py
"""

import argparse
import json
import random
import shutil
from pathlib import Path

DATASET_DIR = Path(__file__).parent / "dataset"
EUROPE_DIR = DATASET_DIR / "europe"
EUROPE_REJECTED_DIR = DATASET_DIR / "europe_rejected"
SOURCE_DIR = DATASET_DIR / "europe_to_trait"
SOURCE_REJECTED_DIR = DATASET_DIR / "europe_to_trait_rejected"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def list_species_with_images(source_dir: Path) -> list[Path]:
    species = []
    for sp_dir in sorted(source_dir.iterdir()):
        if not sp_dir.is_dir():
            continue
        has_images = any(f.suffix.lower() in IMAGE_EXTENSIONS
                        for f in sp_dir.iterdir() if f.is_file())
        if has_images:
            species.append(sp_dir)
    return species


def load_annotations(ann_path: Path) -> dict:
    if ann_path.exists():
        with open(ann_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_annotations(ann_path: Path, annotations: dict):
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2)


def split_images(images: list[Path], train_ratio: float, val_ratio: float,
                 seed: int) -> dict[str, list[Path]]:
    rng = random.Random(seed)
    shuffled = list(images)
    rng.shuffle(shuffled)

    n = len(shuffled)
    if n == 1:
        return {"train": shuffled, "validation": [], "test": []}
    if n == 2:
        return {"train": shuffled[:1], "validation": shuffled[1:], "test": []}

    n_val = max(1, round(n * val_ratio))
    n_test = max(1, round(n * (1 - train_ratio - val_ratio)))
    n_train = n - n_val - n_test

    if n_train < 1:
        n_val = max(1, (n - 1) // 2)
        n_test = max(1, n - 1 - n_val)
        n_train = n - n_val - n_test

    return {
        "train": shuffled[:n_train],
        "validation": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def merge_species(sp_dir: Path, europe_dir: Path, train_ratio: float,
                  val_ratio: float, seed: int, dry_run: bool) -> dict:
    slug = sp_dir.name
    images = sorted(f for f in sp_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)

    if not images:
        return {"slug": slug, "total": 0, "train": 0, "validation": 0, "test": 0}

    source_ann = load_annotations(sp_dir / "annotations.json")
    splits = split_images(images, train_ratio, val_ratio, seed)

    stats = {"slug": slug, "total": len(images)}

    for split_name, split_images_list in splits.items():
        stats[split_name] = len(split_images_list)

        if dry_run or not split_images_list:
            continue

        dest_dir = europe_dir / split_name / slug
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest_ann = load_annotations(dest_dir / "annotations.json")

        for img in split_images_list:
            shutil.move(str(img), str(dest_dir / img.name))
            if img.name in source_ann:
                dest_ann[img.name] = source_ann[img.name]

        save_annotations(dest_dir / "annotations.json", dest_ann)

    return stats


def merge_rejected(source_rejected: Path, europe_rejected: Path,
                    dry_run: bool) -> dict:
    if not source_rejected.exists():
        return {"total": 0}

    total = 0
    for sp_dir in sorted(source_rejected.iterdir()):
        if not sp_dir.is_dir():
            continue
        slug = sp_dir.name
        images = sorted(f for f in sp_dir.iterdir()
                        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            continue

        total += len(images)

        if dry_run:
            continue

        dest_dir = europe_rejected / "train" / slug
        dest_dir.mkdir(parents=True, exist_ok=True)

        source_ann = load_annotations(sp_dir / "annotations.json")
        dest_ann = load_annotations(dest_dir / "annotations.json")

        for img in images:
            shutil.move(str(img), str(dest_dir / img.name))
            if img.name in source_ann:
                dest_ann[img.name] = source_ann[img.name]

        save_annotations(dest_dir / "annotations.json", dest_ann)

    return {"total": total}


def cleanup_source(source_dir: Path):
    for sp_dir in sorted(source_dir.iterdir()):
        if not sp_dir.is_dir():
            continue
        remaining = [f for f in sp_dir.iterdir()
                     if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
        if not remaining:
            shutil.rmtree(sp_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Fusionne europe_to_trait/ dans europe/{train,val,test}/"
    )
    parser.add_argument("--dataset-dir", type=Path, default=EUROPE_DIR)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--rejected-source", type=Path, default=SOURCE_REJECTED_DIR)
    parser.add_argument("--rejected-dest", type=Path, default=EUROPE_REJECTED_DIR)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.source_dir.exists():
        print(f"Source introuvable : {args.source_dir}")
        return

    species = list_species_with_images(args.source_dir)
    if not species:
        print("Aucune espece avec des images dans la source.")
        return

    print(f"Fusion de {len(species)} especes depuis {args.source_dir}", flush=True)
    print(f"Destination : {args.dataset_dir}", flush=True)
    test_ratio = round(1 - args.train_ratio - args.val_ratio, 2)
    print(f"Ratio : train={args.train_ratio} / val={args.val_ratio} "
          f"/ test={test_ratio}", flush=True)

    if args.dry_run:
        print("(dry-run : pas de deplacement)\n", flush=True)

    total_stats = {"train": 0, "validation": 0, "test": 0, "total": 0}

    for sp_dir in species:
        stats = merge_species(sp_dir, args.dataset_dir, args.train_ratio,
                              args.val_ratio, args.seed, args.dry_run)
        for key in total_stats:
            total_stats[key] += stats[key]

        print(f"  {stats['slug']:<40} "
              f"train:{stats['train']:>4}  val:{stats['validation']:>3}  "
              f"test:{stats['test']:>3}  (total:{stats['total']})", flush=True)

    rej_stats = {"total": 0}
    if args.rejected_source.exists():
        rej_species = list_species_with_images(args.rejected_source)
        if rej_species:
            print(f"\nFusion des rejetes : {len(rej_species)} especes "
                  f"depuis {args.rejected_source}", flush=True)
            print(f"Destination : {args.rejected_dest}/train/", flush=True)
            if args.dry_run:
                print("(dry-run : pas de deplacement)\n", flush=True)
            rej_stats = merge_rejected(args.rejected_source, args.rejected_dest,
                                       args.dry_run)
            print(f"  Images rejetees deplacees : {rej_stats['total']}", flush=True)

    if not args.dry_run:
        cleanup_source(args.source_dir)
        if args.rejected_source.exists():
            cleanup_source(args.rejected_source)
        print(f"\nNettoyage : dossiers vides supprimes", flush=True)

    total = total_stats["total"]
    print(f"\n{'=' * 60}", flush=True)
    print("RESUME", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"Especes fusionnees : {len(species)}", flush=True)
    print(f"Images deplacees : {total}", flush=True)
    if total > 0:
        print(f"  train:      {total_stats['train']:>6} "
              f"({total_stats['train']/total*100:.0f}%)", flush=True)
        print(f"  validation: {total_stats['validation']:>6} "
              f"({total_stats['validation']/total*100:.0f}%)", flush=True)
        print(f"  test:       {total_stats['test']:>6} "
              f"({total_stats['test']/total*100:.0f}%)", flush=True)
    print(f"Images rejetees : {rej_stats['total']}", flush=True)


if __name__ == "__main__":
    main()
