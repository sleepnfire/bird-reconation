#!/usr/bin/env python3
"""
Calibration des datasets europe / europe_rejected.

Score composite par image basé sur 3 signaux :
  1. Confiance de détection (Grounding DINO) — déjà dans annotations.json
  2. Taille de bbox — % de l'image occupé par l'oiseau
  3. Netteté — variance du Laplacien dans la zone bbox (cv2.Laplacian)

Références :
  - Bai et al. (2024) — Multimodal Data Curation via Object Detection and Filter Ensembles
  - Zhou et al. (2025) — Autonomous Bird Feeder : confiance ≥ 0.7, bbox > 2%
  - Pech-Pacheco et al. (ICPR 2000) — variance du Laplacien pour la netteté
  - Setting BirdNET confidence thresholds (Springer 2025) — seuils espèce-spécifiques

Usage:
    python calibrate_datasets.py score --all --workers 4
    python calibrate_datasets.py sample --n 5 --mode borderline
    python calibrate_datasets.py review [--resume]
    python calibrate_datasets.py metrics
    python calibrate_datasets.py optimize
    python calibrate_datasets.py audit [--report]
    python calibrate_datasets.py apply [--dry-run]
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

DATASET_DIR = Path(__file__).parent / "dataset" / "europe"
REJECTED_DIR = Path(__file__).parent / "dataset" / "europe_rejected"
CALIBRATION_DIR = Path(__file__).parent / "calibration"

DEFAULT_WEIGHTS = {
    "detection_score": 0.3,
    "bbox_pct": 0.4,
    "sharpness": 0.3,
}

SHARPNESS_NORMALIZER = 500.0
BBOX_PCT_NORMALIZER = 30.0


class ImageScorer:
    def __init__(self, weights: dict | None = None):
        self.weights = weights or DEFAULT_WEIGHTS

    def score_image(self, image_path: Path, annotation: dict | None) -> dict:
        image_path = Path(image_path)

        if annotation is None:
            return {
                "detection_score": 0.0,
                "bbox_pct": 0.0,
                "sharpness": 0.0,
                "composite_score": 0.0,
            }

        if not image_path.exists():
            return {
                "detection_score": 0.0,
                "bbox_pct": 0.0,
                "sharpness": 0.0,
                "composite_score": 0.0,
            }

        try:
            img = cv2.imread(str(image_path))
            if img is None:
                raise OSError("cannot read image")
            img_h, img_w = img.shape[:2]
        except (OSError, cv2.error):
            return {
                "detection_score": 0.0,
                "bbox_pct": 0.0,
                "sharpness": 0.0,
                "composite_score": 0.0,
            }

        det_score = float(annotation.get("score", 0.0))

        x, y, bw, bh = annotation["bbox"]
        img_area = img_w * img_h
        bbox_area = bw * bh
        bbox_pct = (bbox_area / img_area * 100) if img_area > 0 else 0.0

        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(img_w, int(x + bw))
        y2 = min(img_h, int(y + bh))
        if x2 > x1 and y2 > y1:
            crop = img[y1:y2, x1:x2]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = float(laplacian.var())
        else:
            sharpness = 0.0

        norm_det = min(det_score, 1.0)
        norm_bbox = min(bbox_pct / BBOX_PCT_NORMALIZER, 1.0)
        norm_sharp = min(sharpness / SHARPNESS_NORMALIZER, 1.0)

        composite = (
            self.weights["detection_score"] * norm_det
            + self.weights["bbox_pct"] * norm_bbox
            + self.weights["sharpness"] * norm_sharp
        )

        return {
            "detection_score": round(det_score, 4),
            "bbox_pct": round(bbox_pct, 4),
            "sharpness": round(sharpness, 2),
            "composite_score": round(composite, 4),
        }

    def score_species(self, species_dir: Path) -> dict[str, dict]:
        species_dir = Path(species_dir)
        ann_path = species_dir / "annotations.json"
        annotations = {}
        if ann_path.exists():
            with open(ann_path) as f:
                annotations = json.load(f)

        results = {}
        for img_path in sorted(species_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            ann = annotations.get(img_path.name)
            results[img_path.name] = self.score_image(img_path, ann)

        return results


class StratifiedSampler:
    def __init__(self, kept_dir: Path, rejected_dir: Path, scorer: ImageScorer):
        self.kept_dir = Path(kept_dir)
        self.rejected_dir = Path(rejected_dir)
        self.scorer = scorer

    def _collect_images(self, base_dir: Path, source: str) -> list[dict]:
        items = []
        if not base_dir.exists():
            return items
        for sp_dir in sorted(base_dir.iterdir()):
            if not sp_dir.is_dir():
                continue
            scores = self.scorer.score_species(sp_dir)
            for name, score_dict in scores.items():
                items.append({
                    "path": str(sp_dir / name),
                    "species": sp_dir.name,
                    "source": source,
                    "name": name,
                    **score_dict,
                })
        return items

    def sample(self, n_per_species: int = 5, mode: str = "random",
               species: list[str] | None = None, seed: int = 42) -> list[dict]:
        kept_items = self._collect_images(self.kept_dir, "europe")
        rejected_items = self._collect_images(self.rejected_dir, "europe_rejected")
        all_items = kept_items + rejected_items

        if species:
            all_items = [i for i in all_items if i["species"] in species]

        by_species: dict[str, list[dict]] = {}
        for item in all_items:
            by_species.setdefault(item["species"], []).append(item)

        rng = random.Random(seed)
        samples = []

        for sp, items in sorted(by_species.items()):
            if mode == "random":
                rng.shuffle(items)
                samples.extend(items[:n_per_species])

            elif mode == "borderline":
                items.sort(key=lambda x: abs(x["composite_score"] - 0.3))
                samples.extend(items[:n_per_species])

            elif mode == "worst_kept":
                kept = [i for i in items if i["source"] == "europe"]
                kept.sort(key=lambda x: x["composite_score"])
                samples.extend(kept[:n_per_species])

            elif mode == "best_rejected":
                rej = [i for i in items if i["source"] == "europe_rejected"]
                rej.sort(key=lambda x: x["composite_score"], reverse=True)
                samples.extend(rej[:n_per_species])

        return samples


class GroundTruth:
    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> dict:
        if not self.path.exists():
            return {}
        with open(self.path) as f:
            return json.load(f)

    def save(self, labels: dict):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(labels, f, indent=2)

    def add_label(self, image_key: str, label: str, source: str, species: str):
        labels = self.load()
        labels[image_key] = {"label": label, "source": source, "species": species}
        self.save(labels)


def compute_metrics(labels: dict) -> dict:
    tp = fp = fn = tn = 0
    for info in labels.values():
        is_kept = info["source"] == "europe"
        is_good = info["label"] == "good"
        if is_kept and is_good:
            tp += 1
        elif is_kept and not is_good:
            fp += 1
        elif not is_kept and is_good:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tp) if (fp + tp) > 0 else 0.0
    fnr = fn / (fn + tn) if (fn + tn) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def compute_metrics_by_reason(labels: dict) -> dict:
    by_reason: dict[str, dict] = {}
    for info in labels.values():
        reason = info.get("reject_reason")
        if reason is None or info["source"] != "europe_rejected":
            continue
        if reason not in by_reason:
            by_reason[reason] = {"false_negative_count": 0, "true_negative_count": 0, "total": 0}
        by_reason[reason]["total"] += 1
        if info["label"] == "good":
            by_reason[reason]["false_negative_count"] += 1
        else:
            by_reason[reason]["true_negative_count"] += 1
    return by_reason


class ThresholdOptimizer:
    def __init__(self, scorer: ImageScorer, kept_dir: Path, rejected_dir: Path):
        self.scorer = scorer
        self.kept_dir = Path(kept_dir)
        self.rejected_dir = Path(rejected_dir)

    def _score_image_from_path(self, path_str: str) -> float:
        path = Path(path_str)
        sp_dir = path.parent
        ann_path = sp_dir / "annotations.json"
        ann = None
        if ann_path.exists():
            with open(ann_path) as f:
                anns = json.load(f)
            ann = anns.get(path.name)
        return self.scorer.score_image(path, ann)["composite_score"]

    def sweep(self, ground_truth: GroundTruth,
              min_composite_range: list[float] | None = None) -> list[dict]:
        if min_composite_range is None:
            min_composite_range = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]

        labels = ground_truth.load()
        if not labels:
            return []

        scored = {}
        for key, info in labels.items():
            scored[key] = self._score_image_from_path(key)

        results = []
        for threshold in min_composite_range:
            tp = fp = fn = tn = 0
            for key, info in labels.items():
                score = scored[key]
                predicted_keep = score >= threshold
                actual_good = info["label"] == "good"

                if predicted_keep and actual_good:
                    tp += 1
                elif predicted_keep and not actual_good:
                    fp += 1
                elif not predicted_keep and actual_good:
                    fn += 1
                else:
                    tn += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            results.append({
                "thresholds": {"min_composite": threshold},
                "metrics": {
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                },
            })

        results.sort(key=lambda r: r["metrics"]["f1"], reverse=True)
        return results

    def recommend(self, ground_truth: GroundTruth) -> dict:
        results = self.sweep(ground_truth)
        if not results:
            return {"thresholds": {"min_composite": 0.2}, "metrics": {"f1": 0.0}}
        return results[0]


class DatasetAuditor:
    def __init__(self, scorer: ImageScorer, dataset_dir: Path):
        self.scorer = scorer
        self.dataset_dir = Path(dataset_dir)
        self._flagged_kept: dict[str, list[dict]] = {}
        self._flagged_rejected: dict[str, list[dict]] = {}

    def audit_kept(self, min_composite: float = 0.25) -> dict[str, list[dict]]:
        self._flagged_kept = {}
        for sp_dir in sorted(self.dataset_dir.iterdir()):
            if not sp_dir.is_dir():
                continue
            scores = self.scorer.score_species(sp_dir)
            flagged = []
            for name, score_dict in scores.items():
                if score_dict["composite_score"] < min_composite:
                    flagged.append({"name": name, **score_dict})
            if flagged:
                self._flagged_kept[sp_dir.name] = flagged
        return self._flagged_kept

    def audit_rejected(self, min_composite: float = 0.25) -> dict[str, list[dict]]:
        self._flagged_rejected = {}
        for sp_dir in sorted(self.dataset_dir.iterdir()):
            if not sp_dir.is_dir():
                continue
            scores = self.scorer.score_species(sp_dir)
            flagged = []
            for name, score_dict in scores.items():
                if score_dict["composite_score"] >= min_composite:
                    flagged.append({"name": name, **score_dict})
            if flagged:
                self._flagged_rejected[sp_dir.name] = flagged
        return self._flagged_rejected

    def summary(self) -> dict:
        total_kept = sum(len(v) for v in self._flagged_kept.values())
        total_rejected = sum(len(v) for v in self._flagged_rejected.values())
        by_species = {}
        for sp, imgs in self._flagged_kept.items():
            by_species[sp] = {"flagged_kept": len(imgs)}
        for sp, imgs in self._flagged_rejected.items():
            by_species.setdefault(sp, {})["flagged_rejected"] = len(imgs)
        return {
            "total_flagged": total_kept + total_rejected,
            "flagged_kept": total_kept,
            "flagged_rejected": total_rejected,
            "by_species": by_species,
        }


def consolidate(europe_dir: Path, rejected_dir: Path, to_trait_dir: Path,
                dry_run: bool = False) -> dict:
    """Rassemble toutes les images de europe/ et europe_rejected/ dans europe_to_trait/."""
    europe_dir = Path(europe_dir)
    rejected_dir = Path(rejected_dir)
    to_trait_dir = Path(to_trait_dir)

    total_moved = 0
    species_set: set[str] = set()

    for ds_dir in (europe_dir, rejected_dir):
        if not ds_dir.exists():
            continue
        for split in ("train", "validation", "test"):
            split_dir = ds_dir / split
            if not split_dir.exists():
                continue
            for sp_dir in sorted(split_dir.iterdir()):
                if not sp_dir.is_dir():
                    continue
                slug = sp_dir.name
                species_set.add(slug)
                dest_sp = to_trait_dir / slug

                images = [f for f in sp_dir.iterdir()
                          if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
                if not images:
                    continue

                src_ann_path = sp_dir / "annotations.json"
                src_ann = {}
                if src_ann_path.exists():
                    with open(src_ann_path) as f:
                        src_ann = json.load(f)

                if not dry_run:
                    dest_sp.mkdir(parents=True, exist_ok=True)

                    dest_ann_path = dest_sp / "annotations.json"
                    dest_ann = {}
                    if dest_ann_path.exists():
                        with open(dest_ann_path) as f:
                            dest_ann = json.load(f)

                    for img in images:
                        shutil.move(str(img), str(dest_sp / img.name))
                        if img.name in src_ann:
                            dest_ann[img.name] = src_ann[img.name]

                    with open(dest_ann_path, "w") as f:
                        json.dump(dest_ann, f, indent=2)

                total_moved += len(images)

    return {"total_moved": total_moved, "species": len(species_set)}


def select_top(to_trait_dir: Path, rejected_dir: Path, scorer: ImageScorer,
               max_per_species: int = 500) -> dict:
    """Garde les top N images par espèce, déplace le reste dans rejected_dir."""
    to_trait_dir = Path(to_trait_dir)
    rejected_dir = Path(rejected_dir)
    total_kept = 0
    total_rejected = 0

    for sp_dir in sorted(to_trait_dir.iterdir()):
        if not sp_dir.is_dir():
            continue

        scores = scorer.score_species(sp_dir)
        scored_images = [
            (name, s) for name, s in scores.items()
        ]
        scored_images.sort(key=lambda x: x[1]["composite_score"], reverse=True)

        to_keep = []
        to_reject = []
        for name, s in scored_images:
            if s["composite_score"] == 0.0:
                to_reject.append(name)
            elif len(to_keep) < max_per_species:
                to_keep.append(name)
            else:
                to_reject.append(name)

        if not to_reject:
            total_kept += len(to_keep)
            continue

        rej_sp = rejected_dir / sp_dir.name
        rej_sp.mkdir(parents=True, exist_ok=True)

        ann_path = sp_dir / "annotations.json"
        ann = {}
        if ann_path.exists():
            with open(ann_path) as f:
                ann = json.load(f)

        rej_ann = {}
        for name in to_reject:
            src = sp_dir / name
            if src.exists():
                shutil.move(str(src), str(rej_sp / name))
            if name in ann:
                rej_ann[name] = ann.pop(name)

        with open(ann_path, "w") as f:
            json.dump(ann, f, indent=2)
        with open(rej_sp / "annotations.json", "w") as f:
            json.dump(rej_ann, f, indent=2)

        total_kept += len(to_keep)
        total_rejected += len(to_reject)

    return {"total_kept": total_kept, "total_rejected": total_rejected}


def apply_reclassification(moves: list[dict], dry_run: bool = False) -> dict:
    moved = 0
    for move in moves:
        src = Path(move["path"])
        species = move["species"]
        name = move["name"]
        from_dir = Path(move["from_dir"])
        to_dir = Path(move["to_dir"])

        dest_sp = to_dir / species
        dest_file = dest_sp / name

        if dry_run:
            continue

        dest_sp.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest_file))

        # MAJ annotations source
        src_ann_path = from_dir / species / "annotations.json"
        src_ann = {}
        if src_ann_path.exists():
            with open(src_ann_path) as f:
                src_ann = json.load(f)
        moved_ann = src_ann.pop(name, None)
        with open(src_ann_path, "w") as f:
            json.dump(src_ann, f, indent=2)

        # MAJ annotations destination
        dest_ann_path = dest_sp / "annotations.json"
        dest_ann = {}
        if dest_ann_path.exists():
            with open(dest_ann_path) as f:
                dest_ann = json.load(f)
        if moved_ann is not None:
            dest_ann[name] = moved_ann
        with open(dest_ann_path, "w") as f:
            json.dump(dest_ann, f, indent=2)

        moved += 1

    return {"moved": moved, "total": len(moves)}


def main():
    parser = argparse.ArgumentParser(description="Calibration des datasets europe / europe_rejected")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- score ---
    sp = subparsers.add_parser("score", help="Scorer les images")
    sp.add_argument("--split", default="train", choices=["train", "validation", "test"])
    sp.add_argument("--all", action="store_true")
    sp.add_argument("--species", nargs="*")
    sp.add_argument("--workers", type=int, default=1)

    # --- sample ---
    sa = subparsers.add_parser("sample", help="Échantillonner pour revue")
    sa.add_argument("--n", type=int, default=5)
    sa.add_argument("--mode", default="borderline", choices=["random", "borderline", "worst_kept", "best_rejected"])
    sa.add_argument("--seed", type=int, default=42)
    sa.add_argument("--species", nargs="*")

    # --- review ---
    rv = subparsers.add_parser("review", help="Revue interactive en grille")
    rv.add_argument("--resume", action="store_true")
    rv.add_argument("--page-size", type=int, default=12)
    rv.add_argument("--mode", default="borderline", choices=["random", "borderline", "worst_kept", "best_rejected"])
    rv.add_argument("--n", type=int, default=5)

    # --- metrics ---
    subparsers.add_parser("metrics", help="Métriques precision/recall")

    # --- optimize ---
    subparsers.add_parser("optimize", help="Optimiser les seuils")

    # --- audit ---
    au = subparsers.add_parser("audit", help="Auditer les datasets")
    au.add_argument("--report", action="store_true")
    au.add_argument("--min-composite", type=float, default=0.25)

    # --- consolidate ---
    co = subparsers.add_parser("consolidate", help="Rassembler toutes les images dans europe_to_trait")
    co.add_argument("--dry-run", action="store_true")

    # --- select-top ---
    st = subparsers.add_parser("select-top", help="Garder les top N par espèce dans europe_to_trait")
    st.add_argument("--max-per-species", type=int, default=500)
    st.add_argument("--dry-run", action="store_true")

    # --- apply ---
    ap = subparsers.add_parser("apply", help="Appliquer la reclassification")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--min-composite", type=float, default=0.25)

    args = parser.parse_args()
    scorer = ImageScorer()

    if args.command == "score":
        splits = ["train", "validation", "test"] if args.all else [args.split]
        for split in splits:
            print(f"\n=== SCORING {split.upper()} ===")
            for dataset_name, base_dir in [("europe", DATASET_DIR), ("europe_rejected", REJECTED_DIR)]:
                split_dir = base_dir / split
                if not split_dir.exists():
                    continue
                out_path = CALIBRATION_DIR / f"scores_{dataset_name}_{split}.json"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                all_scores = {}
                species_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
                if args.species:
                    species_dirs = [d for d in species_dirs if d.name in args.species]
                for sp_dir in species_dirs:
                    scores = scorer.score_species(sp_dir)
                    all_scores[sp_dir.name] = scores
                    n_zero = sum(1 for s in scores.values() if s["composite_score"] == 0.0)
                    print(f"  {sp_dir.name}: {len(scores)} images, {n_zero} score=0")
                with open(out_path, "w") as f:
                    json.dump(all_scores, f, indent=2)
                print(f"  → {out_path}")

    elif args.command == "sample":
        sampler = StratifiedSampler(DATASET_DIR / "train", REJECTED_DIR / "train", scorer)
        samples = sampler.sample(n_per_species=args.n, mode=args.mode,
                                 species=args.species, seed=args.seed)
        out_path = CALIBRATION_DIR / "samples.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(samples, f, indent=2)
        print(f"{len(samples)} images échantillonnées → {out_path}")

    elif args.command == "review":
        gt = GroundTruth(CALIBRATION_DIR / "ground_truth.json")
        sampler = StratifiedSampler(DATASET_DIR / "train", REJECTED_DIR / "train", scorer)
        samples = sampler.sample(n_per_species=args.n, mode=args.mode)

        if args.resume:
            existing = gt.load()
            samples = [s for s in samples if s["path"] not in existing]

        if not samples:
            print("Aucune image à revoir.")
            return

        _run_grid_review(samples, gt, page_size=args.page_size)

    elif args.command == "metrics":
        gt = GroundTruth(CALIBRATION_DIR / "ground_truth.json")
        labels = gt.load()
        if not labels:
            print("Aucun ground truth. Lancez 'review' d'abord.")
            return
        metrics = compute_metrics(labels)
        print("\n=== MÉTRIQUES DU FILTRE ACTUEL ===")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    elif args.command == "optimize":
        gt = GroundTruth(CALIBRATION_DIR / "ground_truth.json")
        optimizer = ThresholdOptimizer(scorer, DATASET_DIR / "train", REJECTED_DIR / "train")
        best = optimizer.recommend(gt)
        print("\n=== SEUIL OPTIMAL ===")
        print(f"  Seuil: {best['thresholds']}")
        print(f"  F1: {best['metrics']['f1']}")
        results = optimizer.sweep(gt)
        out_path = CALIBRATION_DIR / "optimization_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Détails → {out_path}")

    elif args.command == "audit":
        kept_auditor = DatasetAuditor(scorer, DATASET_DIR / "train")
        rej_auditor = DatasetAuditor(scorer, REJECTED_DIR / "train")
        flagged_kept = kept_auditor.audit_kept(min_composite=args.min_composite)
        flagged_rej = rej_auditor.audit_rejected(min_composite=args.min_composite)

        total_bad_kept = sum(len(v) for v in flagged_kept.values())
        total_good_rej = sum(len(v) for v in flagged_rej.values())
        print(f"\n=== AUDIT ===")
        print(f"  Images gardées suspectes (score < {args.min_composite}): {total_bad_kept}")
        print(f"  Images rejetées récupérables (score >= {args.min_composite}): {total_good_rej}")

        if args.report:
            report = {
                "flagged_kept": {sp: [d for d in imgs] for sp, imgs in flagged_kept.items()},
                "flagged_rejected": {sp: [d for d in imgs] for sp, imgs in flagged_rej.items()},
            }
            out_path = CALIBRATION_DIR / "audit_report.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(report, f, indent=2)
            print(f"  Rapport → {out_path}")

    elif args.command == "consolidate":
        to_trait = DATASET_DIR.parent / "europe_to_trait"
        stats = consolidate(DATASET_DIR, REJECTED_DIR, to_trait, dry_run=args.dry_run)
        prefix = "[DRY-RUN] " if args.dry_run else ""
        print(f"\n{prefix}=== CONSOLIDATION ===")
        print(f"  {stats['total_moved']} images rassemblées dans {to_trait}")
        print(f"  {stats['species']} espèces")

    elif args.command == "select-top":
        to_trait = DATASET_DIR.parent / "europe_to_trait"
        to_trait_rej = DATASET_DIR.parent / "europe_to_trait_rejected"
        if not to_trait.exists():
            print("europe_to_trait/ n'existe pas. Lancez 'consolidate' d'abord.")
            return
        stats = select_top(to_trait, to_trait_rej, scorer,
                           max_per_species=args.max_per_species)
        print(f"\n=== SÉLECTION TOP {args.max_per_species} ===")
        print(f"  Gardées : {stats['total_kept']}")
        print(f"  Rejetées : {stats['total_rejected']}")

    elif args.command == "apply":
        kept_auditor = DatasetAuditor(scorer, DATASET_DIR / "train")
        rej_auditor = DatasetAuditor(scorer, REJECTED_DIR / "train")

        flagged_kept = kept_auditor.audit_kept(min_composite=args.min_composite)
        flagged_rej = rej_auditor.audit_rejected(min_composite=args.min_composite)

        moves = []
        for sp, imgs in flagged_kept.items():
            for img in imgs:
                moves.append({
                    "path": str(DATASET_DIR / "train" / sp / img["name"]),
                    "from_dir": str(DATASET_DIR / "train"),
                    "to_dir": str(REJECTED_DIR / "train"),
                    "species": sp,
                    "name": img["name"],
                })
        for sp, imgs in flagged_rej.items():
            for img in imgs:
                moves.append({
                    "path": str(REJECTED_DIR / "train" / sp / img["name"]),
                    "from_dir": str(REJECTED_DIR / "train"),
                    "to_dir": str(DATASET_DIR / "train"),
                    "species": sp,
                    "name": img["name"],
                })

        result = apply_reclassification(moves, dry_run=args.dry_run)
        prefix = "[DRY-RUN] " if args.dry_run else ""
        print(f"\n{prefix}{result['moved']}/{result['total']} images déplacées")


def _run_grid_review(samples: list[dict], ground_truth: GroundTruth, page_size: int = 12):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    labels_state = {}
    total_pages = (len(samples) + page_size - 1) // page_size

    for page_idx in range(total_pages):
        page_start = page_idx * page_size
        page_end = min(page_start + page_size, len(samples))
        page_samples = samples[page_start:page_end]

        n = len(page_samples)
        cols = 4
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        if rows == 1:
            axes = [axes] if cols == 1 else [axes]
        axes_flat = [ax for row in axes for ax in (row if hasattr(row, '__len__') else [row])]

        for idx, sample in enumerate(page_samples):
            ax = axes_flat[idx]
            try:
                img = Image.open(sample["path"])
                ax.imshow(img)
            except (OSError, UnidentifiedImageError):
                ax.text(0.5, 0.5, "ERREUR", ha="center", va="center", transform=ax.transAxes)

            source_tag = "KEPT" if sample["source"] == "europe" else "REJ"
            ax.set_title(
                f"{sample['species']}\n{source_tag} | score={sample['composite_score']:.2f}",
                fontsize=9,
            )
            ax.set_xticks([])
            ax.set_yticks([])

            key = sample["path"]
            labels_state[key] = labels_state.get(key, "good")
            color = "green" if labels_state[key] == "good" else "red"
            for spine in ax.spines.values():
                spine.set_color(color)
                spine.set_linewidth(3)

        for idx in range(n, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        def make_on_click(page_s, ax_flat, fig_ref):
            def on_click(event):
                for i, ax in enumerate(ax_flat[:len(page_s)]):
                    if ax == event.inaxes:
                        key = page_s[i]["path"]
                        labels_state[key] = "bad" if labels_state.get(key) == "good" else "good"
                        color = "green" if labels_state[key] == "good" else "red"
                        for spine in ax.spines.values():
                            spine.set_color(color)
                            spine.set_linewidth(3)
                        fig_ref.canvas.draw()
                        break
            return on_click

        def make_on_key(page_s, ax_flat, fig_ref):
            def on_key(event):
                if event.key == "g":
                    for i, s in enumerate(page_s):
                        labels_state[s["path"]] = "good"
                        for spine in ax_flat[i].spines.values():
                            spine.set_color("green")
                            spine.set_linewidth(3)
                    fig_ref.canvas.draw()
                elif event.key == "b":
                    for i, s in enumerate(page_s):
                        labels_state[s["path"]] = "bad"
                        for spine in ax_flat[i].spines.values():
                            spine.set_color("red")
                            spine.set_linewidth(3)
                    fig_ref.canvas.draw()
                elif event.key in ("enter", "n", "q"):
                    plt.close(fig_ref)
            return on_key

        fig.canvas.mpl_connect("button_press_event", make_on_click(page_samples, axes_flat, fig))
        fig.canvas.mpl_connect("key_press_event", make_on_key(page_samples, axes_flat, fig))

        fig.suptitle(
            f"Page {page_idx + 1}/{total_pages} | "
            f"Clic: toggle good/bad | g: tout good | b: tout bad | n/Enter: suivant | q: quitter",
            fontsize=11,
        )
        plt.tight_layout()
        plt.show()

        for s in page_samples:
            key = s["path"]
            label = labels_state.get(key, "good")
            ground_truth.add_label(key, label, s["source"], s["species"])

        print(f"  Page {page_idx + 1}: {len(page_samples)} images labellées")


if __name__ == "__main__":
    main()
