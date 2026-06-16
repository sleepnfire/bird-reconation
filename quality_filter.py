#!/usr/bin/env python3
"""
Filtrage qualité des images via CLIP zero-shot + détection d'outliers par embeddings.

Pipeline :
  1. CLIP zero-shot — catégories : good, dead_specimen, illustration, screen_scan, not_bird, poor_quality
  2. Outlier par embedding — centroïde par espèce, flag > 1.5σ (LAION-5B / FiftyOne)
  3. Filtre bbox — trop petit < 5% de l'image
  4. apply_filter — supprime les images rejetées + met à jour annotations.json

Références :
  - Radford et al. 2021 (CLIP, ICML) — https://arxiv.org/abs/2103.00020
  - DataComp (Gadre et al., NeurIPS 2023) — https://arxiv.org/abs/2304.14108
  - LAION-5B (Schuhmann et al., NeurIPS 2022) — https://arxiv.org/abs/2210.08402
  - BioTrove (Yang et al., NeurIPS 2024) — https://arxiv.org/abs/2406.17720

Usage:
    # Générer les rapports qualité (4 workers, resume)
    python quality_filter.py report --split train --workers 4

    # Générer pour les 3 splits
    python quality_filter.py report --all --workers 4

    # Appliquer le filtre (supprime images + MAJ annotations/label_map/metadata)
    python quality_filter.py apply --split train

    # Appliquer sur les 3 splits
    python quality_filter.py apply --all

    # Dry-run : voir ce qui serait supprimé sans rien toucher
    python quality_filter.py apply --all --dry-run
"""

import argparse
import json
import multiprocessing
import random
import shutil
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

CATEGORIES = ["good", "dead_specimen", "illustration", "screen_scan", "not_bird", "poor_quality"]

PROMPTS = {
    "good": [
        "a sharp clear photograph of a wild bird perched on a branch in nature",
        "a photograph of a wild bird in its natural habitat, well lit and in focus",
        "a wildlife photograph of a bird standing on the ground outdoors",
        "a clear photo of a bird in flight against the sky",
        "a nature photograph of a bird swimming on a lake or river in the wild",
    ],
    "dead_specimen": [
        "a photograph of a dead bird lying on the ground",
        "a photograph of a taxidermy bird specimen in a museum display case",
        "a photograph of a dead bird on a road, roadkill",
        "a photograph of a preserved bird specimen on a table",
        "a photograph of a bird carcass held in someone's hand",
    ],
    "illustration": [
        "a painted illustration of a bird from a field guide book",
        "a watercolor drawing of a bird on white paper",
        "a scientific illustration or sketch of a bird species",
        "a cartoon or digital drawing of a bird",
        "an engraving or lithograph of a bird",
    ],
    "screen_scan": [
        "a photograph of a computer monitor showing a bird image with visible pixels and screen bezel",
        "a photograph of a printed book page with text and a bird picture, showing paper texture",
        "a photograph of a phone screen with visible UI elements showing a bird photo",
        "a scan of a printed photograph of a bird with visible moiré pattern and paper edges",
        "a photograph of a television screen showing a bird with visible scan lines and screen frame",
    ],
    "not_bird": [
        "a photograph of an empty landscape with no animals visible at all",
        "a photograph of a bird nest with eggs but no bird present anywhere",
        "a photograph of only feathers on the ground with no bird visible",
        "a photograph of an insect, mammal, or reptile, with no bird at all",
        "a photograph of plants, flowers, or trees with no bird anywhere in the image",
    ],
    "poor_quality": [
        "an extremely blurry out of focus photograph where the subject is completely unrecognizable",
        "a nearly black underexposed photograph where nothing is visible",
        "a completely overexposed washed out white photograph with no detail",
        "a heavily pixelated low resolution photograph with visible compression artifacts",
        "a photograph corrupted with visual glitches and digital artifacts",
    ],
}

DATASET_DIR = Path(__file__).parent / "dataset" / "europe"
REPORTS_DIR = Path(__file__).parent / "reports"

DEFAULT_QUALITY_WEIGHTS = {"detection_score": 0.3, "bbox_pct": 0.4, "sharpness": 0.3}
SHARPNESS_NORMALIZER = 500.0
BBOX_PCT_NORMALIZER = 30.0
CALIBRATION_DIR = Path(__file__).parent / "calibration"
REJECTED_DIR = DATASET_DIR.parent / "europe_rejected"


# ── Module-level worker functions (multiprocessing spawn) ────────────────

_worker_clf = None


def _init_quality_worker(model_name, pretrained, use_fp16, reject_margin):
    global _worker_clf
    _worker_clf = QualityClassifier(model_name=model_name, pretrained=pretrained, use_fp16=use_fp16, reject_margin=reject_margin)


def _process_species_quality(args):
    global _worker_clf
    sp_dir_str, out_dir_str, sigma_threshold, duplicate_threshold = args
    sp_dir = Path(sp_dir_str)
    out_path = Path(out_dir_str)

    classifications, embeddings = _worker_clf.classify_and_embed_batched(sp_dir_str)
    if not classifications:
        return {"species": sp_dir.name, "status": "empty", "images": 0}

    outliers_info = _worker_clf.detect_outliers(embeddings, sigma_threshold)

    summary = {"total": len(classifications)}
    for cat in CATEGORIES:
        summary[cat] = sum(1 for r in classifications.values() if r["category"] == cat)

    data = {
        "summary": summary,
        "images": classifications,
        "outliers": {name: info for name, info in outliers_info.items()},
    }

    if duplicate_threshold > 0:
        duplicates = _worker_clf.detect_near_duplicates(embeddings, classifications, duplicate_threshold)
        data["duplicates"] = duplicates
        summary["duplicates"] = len(duplicates)

    ann_path = sp_dir / "annotations.json"
    ann_data = {}
    if ann_path.exists():
        with open(ann_path) as f:
            ann_data = json.load(f)
    for img_name, img_report in data["images"].items():
        img_report["quality"] = score_image_quality(sp_dir / img_name, ann_data.get(img_name))

    torch.save(embeddings, out_path / f"{sp_dir.name}.pt")
    with open(out_path / f"{sp_dir.name}.json", "w") as f:
        json.dump(data, f, indent=2)

    n_outliers = sum(1 for v in outliers_info.values() if v["is_outlier"])
    pct_good = summary.get("good", 0) / summary["total"] * 100
    print(
        f"  {sp_dir.name}: {summary['total']} images, "
        f"{pct_good:.0f}% good, {n_outliers} outliers",
        flush=True,
    )
    return {"species": sp_dir.name, "status": "processed", "images": summary["total"]}


class QualityClassifier:
    def __init__(self, model_name: str = "ViT-L-14", pretrained: str = "openai", use_fp16: bool = False, device: str | None = None, batch_size: int = 64, reject_margin: float = 0.005):
        self.model_name = model_name
        self.pretrained = pretrained
        self.use_fp16 = use_fp16
        self.batch_size = batch_size
        self.reject_margin = reject_margin
        if device is not None:
            self.device = torch.device(device)
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        if use_fp16:
            self.model = self.model.half()
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self._build_text_features()

    def _build_text_features(self):
        cat_features = []
        for cat in CATEGORIES:
            prompts = PROMPTS[cat]
            tokens = self.tokenizer(prompts).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                avg = feats.mean(dim=0)
                avg = avg / avg.norm()
            cat_features.append(avg)
        self.text_features = torch.stack(cat_features)

    def _classify_from_features(self, img_features: torch.Tensor) -> dict:
        similarities = (img_features @ self.text_features.T).squeeze(0)
        probs = similarities.softmax(dim=0).cpu().tolist()
        good_idx = CATEGORIES.index("good")
        good_score = probs[good_idx]
        best_neg_idx = max(
            (i for i in range(len(probs)) if i != good_idx),
            key=lambda i: probs[i],
        )
        if probs[best_neg_idx] - good_score < self.reject_margin:
            chosen_idx = good_idx
        else:
            chosen_idx = best_neg_idx
        return {
            "category": CATEGORIES[chosen_idx],
            "confidence": round(probs[chosen_idx], 4),
            "scores": {cat: round(p, 4) for cat, p in zip(CATEGORIES, probs)},
        }

    def classify_image(self, image_path: str) -> dict:
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        if self.use_fp16:
            img_tensor = img_tensor.half()

        with torch.no_grad():
            img_features = self.model.encode_image(img_tensor)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)

        return self._classify_from_features(img_features)

    def classify_and_embed(self, species_dir: str) -> tuple[dict, dict]:
        """Encode chaque image une seule fois : retourne classifications + embeddings."""
        sp_dir = Path(species_dir)
        classifications = {}
        embeddings = {}

        for img_path in sorted(sp_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
                if self.use_fp16:
                    img_tensor = img_tensor.half()
                with torch.no_grad():
                    img_features = self.model.encode_image(img_tensor)
                    img_features = img_features / img_features.norm(dim=-1, keepdim=True)

                classifications[img_path.name] = self._classify_from_features(img_features)
                embeddings[img_path.name] = img_features.float().squeeze(0).cpu()
            except (OSError, UnidentifiedImageError, SyntaxError):
                classifications[img_path.name] = {"category": "corrupted", "confidence": 1.0, "scores": {}}

        return classifications, embeddings

    def classify_and_embed_batched(self, species_dir: str, batch_size: int = 64) -> tuple[dict, dict]:
        """Encode par batches sur GPU — beaucoup plus rapide que image par image."""
        sp_dir = Path(species_dir)
        classifications = {}
        embeddings = {}

        img_paths = sorted(
            p for p in sp_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not img_paths:
            return classifications, embeddings

        batch_tensors = []
        batch_names = []

        for img_path in img_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                tensor = self.preprocess(img)
                if self.use_fp16:
                    tensor = tensor.half()
                batch_tensors.append(tensor)
                batch_names.append(img_path.name)
            except (OSError, UnidentifiedImageError, SyntaxError):
                classifications[img_path.name] = {"category": "corrupted", "confidence": 1.0, "scores": {}}

        for i in range(0, len(batch_tensors), batch_size):
            chunk_tensors = torch.stack(batch_tensors[i:i + batch_size]).to(self.device)
            chunk_names = batch_names[i:i + batch_size]

            with torch.no_grad():
                features = self.model.encode_image(chunk_tensors)
                features = features / features.norm(dim=-1, keepdim=True)

            for j, name in enumerate(chunk_names):
                feat = features[j]
                classifications[name] = self._classify_from_features(feat.unsqueeze(0))
                embeddings[name] = feat.float().cpu()

        return classifications, embeddings

    def compute_embeddings(self, species_dir: str) -> dict:
        sp_dir = Path(species_dir)
        embeddings = {}
        for img_path in sorted(sp_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    feat = self.model.encode_image(img_tensor).squeeze(0).cpu()
                    feat = feat / feat.norm()
                embeddings[img_path.name] = feat
            except (OSError, UnidentifiedImageError, SyntaxError):
                continue
        return embeddings

    def detect_outliers(self, embeddings: dict, sigma_threshold: float = 1.5) -> dict:
        if len(embeddings) < 3:
            return {
                name: {"distance": 0.0, "is_outlier": False}
                for name in embeddings
            }

        names = list(embeddings.keys())
        vectors = torch.stack([embeddings[n] for n in names])
        vectors = vectors / vectors.norm(dim=-1, keepdim=True)

        centroid = vectors.mean(dim=0)
        centroid = centroid / centroid.norm()

        distances = 1.0 - (vectors @ centroid).numpy()
        mean_d = float(np.mean(distances))
        std_d = float(np.std(distances))
        threshold = mean_d + sigma_threshold * std_d

        return {
            name: {
                "distance": round(float(distances[i]), 4),
                "is_outlier": bool(distances[i] > threshold),
            }
            for i, name in enumerate(names)
        }

    def detect_near_duplicates(
        self,
        embeddings: dict,
        classifications: dict,
        threshold: float = 0.95,
    ) -> list[dict]:
        if len(embeddings) < 2:
            return []

        names = list(embeddings.keys())
        vectors = torch.stack([embeddings[n] for n in names])
        vectors = vectors / vectors.norm(dim=-1, keepdim=True)

        sim_matrix = (vectors @ vectors.T).numpy()

        pairs = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if sim_matrix[i, j] > threshold:
                    pairs.append((i, j, float(sim_matrix[i, j])))

        pairs.sort(key=lambda x: x[2], reverse=True)

        removed = set()
        result = []
        for i, j, sim in pairs:
            if names[i] in removed and names[j] in removed:
                continue
            if names[i] in removed:
                result.append({"kept": names[j], "removed": names[i], "similarity": round(sim, 4)})
                continue
            if names[j] in removed:
                result.append({"kept": names[i], "removed": names[j], "similarity": round(sim, 4)})
                continue

            score_i = classifications.get(names[i], {}).get("quality", {}).get("composite_score", 0.0)
            score_j = classifications.get(names[j], {}).get("quality", {}).get("composite_score", 0.0)

            if score_i > score_j:
                kept, rej = names[i], names[j]
            elif score_j > score_i:
                kept, rej = names[j], names[i]
            else:
                kept, rej = (names[i], names[j]) if names[i] < names[j] else (names[j], names[i])

            removed.add(rej)
            result.append({"kept": kept, "removed": rej, "similarity": round(sim, 4)})

        return result

    def classify_species(self, species_dir: str) -> dict:
        classifications, _ = self.classify_and_embed(species_dir)
        return classifications

    def _process_species_thread(self, sp_dir: Path, out_path: Path, sigma_threshold: float, duplicate_threshold: float = 0.0) -> dict:
        classifications, embeddings = self.classify_and_embed_batched(str(sp_dir), batch_size=self.batch_size)
        if not classifications:
            return {"status": "empty", "images": 0}

        outliers_info = self.detect_outliers(embeddings, sigma_threshold)

        summary = {"total": len(classifications)}
        for cat in CATEGORIES:
            summary[cat] = sum(1 for r in classifications.values() if r["category"] == cat)

        data = {
            "summary": summary,
            "images": classifications,
            "outliers": {name: info for name, info in outliers_info.items()},
        }

        if duplicate_threshold > 0:
            duplicates = self.detect_near_duplicates(embeddings, classifications, duplicate_threshold)
            data["duplicates"] = duplicates
            summary["duplicates"] = len(duplicates)

        ann_path = sp_dir / "annotations.json"
        ann_data = {}
        if ann_path.exists():
            with open(ann_path) as f:
                ann_data = json.load(f)
        for img_name, img_report in data["images"].items():
            img_report["quality"] = score_image_quality(sp_dir / img_name, ann_data.get(img_name))

        torch.save(embeddings, out_path / f"{sp_dir.name}.pt")
        with open(out_path / f"{sp_dir.name}.json", "w") as f:
            json.dump(data, f, indent=2)

        n_outliers = sum(1 for v in outliers_info.values() if v["is_outlier"])
        pct_good = summary.get("good", 0) / summary["total"] * 100
        print(
            f"  {sp_dir.name}: {summary['total']} images, "
            f"{pct_good:.0f}% good, {n_outliers} outliers",
            flush=True,
        )
        return {"status": "processed", "images": summary["total"]}

    def generate_quality_report(
        self,
        split_dir: str,
        output_dir: str,
        sigma_threshold: float = 1.5,
        workers: int = 1,
        resume: bool = False,
        duplicate_threshold: float = 0.0,
    ) -> dict:
        split_path = Path(split_dir)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        species_dirs = sorted(d for d in split_path.iterdir() if d.is_dir())
        stats = {"processed": 0, "skipped": 0, "total_images": 0}

        if workers <= 1:
            for sp_dir in species_dirs:
                if resume and (out_path / f"{sp_dir.name}.json").exists():
                    stats["skipped"] += 1
                    continue

                classifications, embeddings = self.classify_and_embed_batched(str(sp_dir), batch_size=self.batch_size)
                if not classifications:
                    continue

                outliers_info = self.detect_outliers(embeddings, sigma_threshold)

                summary = {"total": len(classifications)}
                for cat in CATEGORIES:
                    summary[cat] = sum(1 for r in classifications.values() if r["category"] == cat)

                data = {
                    "summary": summary,
                    "images": classifications,
                    "outliers": {name: info for name, info in outliers_info.items()},
                }

                if duplicate_threshold > 0:
                    duplicates = self.detect_near_duplicates(embeddings, classifications, duplicate_threshold)
                    data["duplicates"] = duplicates
                    summary["duplicates"] = len(duplicates)

                ann_path = sp_dir / "annotations.json"
                ann_data = {}
                if ann_path.exists():
                    with open(ann_path) as f:
                        ann_data = json.load(f)
                for img_name, img_report in data["images"].items():
                    img_report["quality"] = score_image_quality(sp_dir / img_name, ann_data.get(img_name))

                torch.save(embeddings, out_path / f"{sp_dir.name}.pt")
                with open(out_path / f"{sp_dir.name}.json", "w") as f:
                    json.dump(data, f, indent=2)

                n_outliers = sum(1 for v in outliers_info.values() if v["is_outlier"])
                pct_good = summary.get("good", 0) / summary["total"] * 100
                print(
                    f"  {sp_dir.name}: {summary['total']} images, "
                    f"{pct_good:.0f}% good, {n_outliers} outliers",
                    flush=True,
                )
                stats["processed"] += 1
                stats["total_images"] += summary["total"]
        elif self.device.type in ("mps", "cuda"):
            to_process = []
            for sp_dir in species_dirs:
                if resume and (out_path / f"{sp_dir.name}.json").exists():
                    stats["skipped"] += 1
                else:
                    to_process.append(sp_dir)

            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        self._process_species_thread, sp_dir, out_path, sigma_threshold, duplicate_threshold
                    ): sp_dir
                    for sp_dir in to_process
                }
                for future in as_completed(futures):
                    result = future.result()
                    if result["status"] == "processed":
                        stats["processed"] += 1
                        stats["total_images"] += result["images"]
        else:
            to_process = []
            for sp_dir in species_dirs:
                if resume and (out_path / f"{sp_dir.name}.json").exists():
                    stats["skipped"] += 1
                else:
                    to_process.append((str(sp_dir), str(out_path), sigma_threshold, duplicate_threshold))

            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(
                workers,
                initializer=_init_quality_worker,
                initargs=(self.model_name, self.pretrained, self.use_fp16, self.reject_margin),
            ) as pool:
                for result in pool.imap_unordered(_process_species_quality, to_process):
                    if result["status"] == "processed":
                        stats["processed"] += 1
                        stats["total_images"] += result["images"]

        return stats


def filter_small_bbox(species_dir: Path, min_area_pct: float = 5.0) -> list[str]:
    ann_path = species_dir / "annotations.json"
    if not ann_path.exists():
        return []

    with open(ann_path) as f:
        annotations = json.load(f)

    flagged = []
    for name, det in annotations.items():
        if det is None:
            continue
        img_path = species_dir / name
        if not img_path.exists():
            continue
        try:
            img = Image.open(img_path)
            img_w, img_h = img.size
        except (OSError, UnidentifiedImageError):
            continue

        _, _, bw, bh = det["bbox"]
        bbox_area = bw * bh
        img_area = img_w * img_h
        if img_area > 0 and (bbox_area / img_area * 100) < min_area_pct:
            flagged.append(name)

    return flagged


def score_image_quality(
    image_path: Path,
    annotation: dict | None,
    weights: dict | None = None,
) -> dict:
    import cv2

    weights = weights or DEFAULT_QUALITY_WEIGHTS
    image_path = Path(image_path)
    zero = {"detection_score": 0.0, "bbox_pct": 0.0, "sharpness": 0.0, "composite_score": 0.0}

    if annotation is None:
        return zero

    if not image_path.exists():
        return zero

    try:
        img = cv2.imread(str(image_path))
        if img is None:
            raise OSError("cannot read image")
        img_h, img_w = img.shape[:2]
    except (OSError, cv2.error):
        return zero

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
        weights["detection_score"] * norm_det
        + weights["bbox_pct"] * norm_bbox
        + weights["sharpness"] * norm_sharp
    )

    return {
        "detection_score": round(det_score, 4),
        "bbox_pct": round(bbox_pct, 4),
        "sharpness": round(sharpness, 2),
        "composite_score": round(composite, 4),
    }


def score_species_quality(species_dir: Path) -> dict[str, dict]:
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
        results[img_path.name] = score_image_quality(img_path, ann)

    return results


def detect_mislabeled(
    split_dir: str,
    reports_dir: str,
    margin: float = 0.1,
) -> dict:
    reports_path = Path(reports_dir)

    all_embeddings = {}
    for pt_file in sorted(reports_path.glob("*.pt")):
        species = pt_file.stem
        emb = torch.load(pt_file, weights_only=True)
        if emb:
            all_embeddings[species] = emb

    if len(all_embeddings) < 2:
        return {}

    centroids = {}
    for species, emb_dict in all_embeddings.items():
        vectors = torch.stack(list(emb_dict.values()))
        vectors = vectors / vectors.norm(dim=-1, keepdim=True)
        centroid = vectors.mean(dim=0)
        centroid = centroid / centroid.norm()
        centroids[species] = centroid

    species_list = sorted(centroids.keys())
    centroid_matrix = torch.stack([centroids[sp] for sp in species_list])

    result = {}
    for species in species_list:
        sp_idx = species_list.index(species)
        emb_dict = all_embeddings[species]
        sp_result = {}

        for name, embedding in emb_dict.items():
            vec = embedding / embedding.norm()

            own_distance = float(1.0 - (vec @ centroids[species]).item())

            distances_to_all = (1.0 - (centroid_matrix @ vec)).numpy()
            distances_to_all[sp_idx] = float("inf")
            nearest_idx = int(np.argmin(distances_to_all))
            nearest_distance = float(distances_to_all[nearest_idx])
            nearest_species = species_list[nearest_idx]

            sp_result[name] = {
                "own_distance": round(own_distance, 4),
                "nearest_species": nearest_species,
                "nearest_distance": round(nearest_distance, 4),
                "suspected": bool(own_distance - nearest_distance > margin),
            }

        result[species] = sp_result

    return result


def apply_filter(
    split_dir: str,
    reports_dir: str,
    reject_categories: list[str] | None = None,
    remove_outliers: bool = False,
    remove_duplicates: bool = False,
    remove_mislabeled: bool = False,
    mislabel_margin: float = 0.1,
    dataset_root: str | None = None,
    min_bbox_pct: float = 0.0,
    rejected_dir: str | None = None,
    max_per_species: int = 0,
) -> dict:
    """Déplace les images rejetées dans rejected_dir et met à jour annotations.json, label_map.json, metadata.json."""
    split_path = Path(split_dir)
    reports_path = Path(reports_dir)
    rej_path = Path(rejected_dir) if rejected_dir else None

    if reject_categories is None:
        reject_categories = ["dead_specimen", "illustration", "screen_scan", "not_bird", "poor_quality"]

    mislabeled_all = {}
    if remove_mislabeled:
        mislabeled_all = detect_mislabeled(split_dir, reports_dir, margin=mislabel_margin)

    total_removed = 0
    total_kept = 0
    emptied_species = []

    for sp_dir in sorted(split_path.iterdir()):
        if not sp_dir.is_dir():
            continue

        report_file = reports_path / f"{sp_dir.name}.json"
        if not report_file.exists():
            continue

        with open(report_file) as f:
            report = json.load(f)

        to_remove = set()

        for name, info in report.get("images", {}).items():
            if info["category"] in reject_categories:
                to_remove.add(name)

        if remove_outliers:
            for name, info in report.get("outliers", {}).items():
                if info["is_outlier"]:
                    to_remove.add(name)

        if remove_duplicates:
            for dup in report.get("duplicates", []):
                to_remove.add(dup["removed"])

        if remove_mislabeled:
            for name, info in mislabeled_all.get(sp_dir.name, {}).items():
                if info["suspected"]:
                    to_remove.add(name)

        if min_bbox_pct > 0:
            small_bbox = filter_small_bbox(sp_dir, min_area_pct=min_bbox_pct)
            to_remove.update(small_bbox)

        if max_per_species > 0:
            remaining = [n for n in report.get("images", {}) if n not in to_remove]
            if len(remaining) > max_per_species:
                scored = []
                for n in remaining:
                    cs = report["images"][n].get("quality", {}).get("composite_score", 0.0)
                    scored.append((n, cs))
                scored.sort(key=lambda x: x[1], reverse=True)
                for n, _ in scored[max_per_species:]:
                    to_remove.add(n)

        # Déplacer les images vers rejected_dir ou les supprimer
        rej_sp = None
        if rej_path and to_remove:
            rej_sp = rej_path / sp_dir.name
            rej_sp.mkdir(parents=True, exist_ok=True)

        # Lire annotations avant de déplacer
        ann_path = sp_dir / "annotations.json"
        annotations = {}
        if ann_path.exists():
            with open(ann_path) as f:
                annotations = json.load(f)

        rejected_annotations = {}
        for name in to_remove:
            img_path = sp_dir / name
            if img_path.exists():
                if rej_sp:
                    shutil.move(str(img_path), str(rej_sp / name))
                else:
                    img_path.unlink()
            if name in annotations:
                rejected_annotations[name] = annotations.pop(name)

        # MAJ annotations.json source
        if ann_path.exists():
            with open(ann_path, "w") as f:
                json.dump(annotations, f, indent=2)

        # Écrire annotations.json dans rejected
        if rej_sp and rejected_annotations:
            with open(rej_sp / "annotations.json", "w") as f:
                json.dump(rejected_annotations, f, indent=2)

        removed = len(to_remove)
        kept = report["summary"]["total"] - removed
        total_removed += removed
        total_kept += kept

        if removed > 0:
            print(f"  {sp_dir.name}: {removed} déplacées, {kept} conservées", flush=True)

        if kept == 0:
            emptied_species.append(sp_dir.name)
            shutil.rmtree(sp_dir)
            print(f"  {sp_dir.name}: espèce retirée du dataset (0 images restantes)", flush=True)

    if dataset_root and emptied_species:
        root = Path(dataset_root)
        _update_label_map(root, emptied_species)
        _update_metadata(root, emptied_species)

    return {"removed": total_removed, "kept": total_kept}


def apply_filter_dataset(
    dataset_root: str,
    reports_root: str,
    reject_categories: list[str] | None = None,
    remove_outliers: bool = False,
    remove_duplicates: bool = False,
    remove_mislabeled: bool = False,
    mislabel_margin: float = 0.1,
    min_bbox_pct: float = 0.0,
    rejected_root: str | None = None,
    max_per_species: int = 0,
) -> dict:
    """Applique le filtre sur les 3 splits (train, validation, test)."""
    total_removed = 0
    total_kept = 0

    for split in ("train", "validation", "test"):
        split_dir = Path(dataset_root) / split
        reports_dir = Path(reports_root) / split
        if not split_dir.exists() or not reports_dir.exists():
            continue

        rej_dir = str(Path(rejected_root) / split) if rejected_root else None

        print(f"\n=== {split.upper()} ===")
        stats = apply_filter(
            str(split_dir), str(reports_dir),
            reject_categories=reject_categories,
            remove_outliers=remove_outliers,
            remove_duplicates=remove_duplicates,
            remove_mislabeled=remove_mislabeled,
            mislabel_margin=mislabel_margin,
            dataset_root=dataset_root,
            min_bbox_pct=min_bbox_pct,
            rejected_dir=rej_dir,
            max_per_species=max_per_species,
        )
        total_removed += stats["removed"]
        total_kept += stats["kept"]

    print(f"\nTotal: {total_removed} déplacées, {total_kept} conservées")
    return {"removed": total_removed, "kept": total_kept}


def _update_label_map(dataset_root: Path, removed_species: list[str]):
    lm_path = dataset_root / "label_map.json"
    if not lm_path.exists():
        return
    with open(lm_path) as f:
        label_map = json.load(f)
    for sp in removed_species:
        label_map.pop(sp, None)
    sorted_species = sorted(label_map.keys())
    label_map = {sp: i for i, sp in enumerate(sorted_species)}
    with open(lm_path, "w") as f:
        json.dump(label_map, f, indent=2)


def _update_metadata(dataset_root: Path, removed_species: list[str]):
    md_path = dataset_root / "metadata.json"
    if not md_path.exists():
        return
    with open(md_path) as f:
        metadata = json.load(f)
    for sp in removed_species:
        metadata.pop(sp, None)
    with open(md_path, "w") as f:
        json.dump(metadata, f, indent=2)


class StratifiedSampler:
    def __init__(self, kept_dir: Path, rejected_dir: Path):
        self.kept_dir = Path(kept_dir)
        self.rejected_dir = Path(rejected_dir)

    def _collect_images(self, base_dir: Path, source: str) -> list[dict]:
        items = []
        if not base_dir.exists():
            return items
        for sp_dir in sorted(base_dir.iterdir()):
            if not sp_dir.is_dir():
                continue
            scores = score_species_quality(sp_dir)
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
    def __init__(self, kept_dir: Path, rejected_dir: Path):
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
        return score_image_quality(path, ann)["composite_score"]

    def sweep(self, ground_truth: GroundTruth,
              min_composite_range: list[float] | None = None) -> list[dict]:
        if min_composite_range is None:
            min_composite_range = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]

        labels = ground_truth.load()
        if not labels:
            return []

        scored = {}
        for key in labels:
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


def main():
    parser = argparse.ArgumentParser(description="Filtrage qualité des images via CLIP")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- report ---
    rp = subparsers.add_parser("report", help="Générer les rapports qualité CLIP")
    rp.add_argument("--split", default="train", choices=["train", "validation", "test"])
    rp.add_argument("--all", action="store_true", help="Traiter les 3 splits")
    rp.add_argument("--workers", type=int, default=1)
    rp.add_argument("--resume", action="store_true", help="Sauter les espèces déjà traitées")
    rp.add_argument("--sigma", type=float, default=1.5, help="Seuil outlier (défaut: 1.5)")
    rp.add_argument("--batch-size", type=int, default=64, help="Taille de batch GPU (défaut: 64)")
    rp.add_argument("--fp16", action="store_true", help="Utiliser float16 pour accélérer")
    rp.add_argument("--reject-margin", type=float, default=0.005,
                    help="Marge min entre score négatif et good pour rejeter (défaut: 0.005)")
    rp.add_argument("--duplicate-threshold", type=float, default=0.0,
                    help="Seuil similarité cosinus pour near-duplicates (0 = désactivé, défaut: 0)")

    # --- apply ---
    ap = subparsers.add_parser("apply", help="Déplacer les mauvaises images dans europe_rejected/")
    ap.add_argument("--split", default="train", choices=["train", "validation", "test"])
    ap.add_argument("--all", action="store_true", help="Filtrer les 3 splits")
    ap.add_argument("--remove-outliers", action="store_true")
    ap.add_argument("--remove-duplicates", action="store_true",
                    help="Retirer les near-duplicates (garder le meilleur score good)")
    ap.add_argument("--remove-mislabeled", action="store_true",
                    help="Retirer les images suspectées d'être mal étiquetées")
    ap.add_argument("--mislabel-margin", type=float, default=0.1,
                    help="Marge pour la détection de mislabels (défaut: 0.1)")
    ap.add_argument("--min-bbox-pct", type=float, default=5.0, help="Bbox minimale en %% de l'image (défaut: 5)")
    ap.add_argument("--rejected-dir", default=str(DATASET_DIR.parent / "europe_rejected"),
                    help="Dossier de destination pour les images rejetées")

    # --- mislabel ---
    mp = subparsers.add_parser("mislabel", help="Détecter les images potentiellement mal étiquetées")
    mp.add_argument("--split", default="train", choices=["train", "validation", "test"])
    mp.add_argument("--all", action="store_true", help="Analyser les 3 splits")
    mp.add_argument("--margin", type=float, default=0.1, help="Marge de détection (défaut: 0.1)")

    args = parser.parse_args()

    if args.command == "report":
        clf = QualityClassifier(use_fp16=args.fp16, batch_size=args.batch_size, reject_margin=args.reject_margin)
        splits = ["train", "validation", "test"] if args.all else [args.split]
        for split in splits:
            split_dir = DATASET_DIR / split
            if not split_dir.exists():
                print(f"Dossier {split_dir} introuvable, skip")
                continue
            out = REPORTS_DIR / split
            print(f"\n=== {split.upper()} ===")
            stats = clf.generate_quality_report(
                str(split_dir), str(out),
                sigma_threshold=args.sigma,
                workers=args.workers,
                resume=args.resume,
                duplicate_threshold=args.duplicate_threshold,
            )
            print(f"  → {stats['processed']} traitées, {stats['skipped']} sautées, {stats['total_images']} images")

    elif args.command == "apply":
        if args.all:
            stats = apply_filter_dataset(
                str(DATASET_DIR), str(REPORTS_DIR),
                remove_outliers=args.remove_outliers,
                remove_duplicates=args.remove_duplicates,
                remove_mislabeled=args.remove_mislabeled,
                mislabel_margin=args.mislabel_margin,
                min_bbox_pct=args.min_bbox_pct,
                rejected_root=args.rejected_dir,
            )
        else:
            split_dir = DATASET_DIR / args.split
            reports_dir = REPORTS_DIR / args.split
            rej_dir = str(Path(args.rejected_dir) / args.split) if args.rejected_dir else None
            stats = apply_filter(
                str(split_dir), str(reports_dir),
                remove_outliers=args.remove_outliers,
                remove_duplicates=args.remove_duplicates,
                remove_mislabeled=args.remove_mislabeled,
                mislabel_margin=args.mislabel_margin,
                dataset_root=str(DATASET_DIR),
                min_bbox_pct=args.min_bbox_pct,
                rejected_dir=rej_dir,
            )
        print(f"\nRésultat: {stats['removed']} déplacées, {stats['kept']} conservées")

    elif args.command == "mislabel":
        splits = ["train", "validation", "test"] if args.all else [args.split]
        for split in splits:
            split_dir = DATASET_DIR / split
            reports_dir = REPORTS_DIR / split
            if not split_dir.exists() or not reports_dir.exists():
                print(f"Dossier {split_dir} ou {reports_dir} introuvable, skip")
                continue
            print(f"\n=== {split.upper()} ===")
            results = detect_mislabeled(str(split_dir), str(reports_dir), margin=args.margin)
            total_suspected = 0
            for species, images in sorted(results.items()):
                suspected = [n for n, info in images.items() if info["suspected"]]
                if suspected:
                    total_suspected += len(suspected)
                    print(f"  {species}: {len(suspected)} suspected mislabels")
                    for name in suspected:
                        info = images[name]
                        print(f"    {name}: own={info['own_distance']:.4f}, "
                              f"nearest={info['nearest_species']} ({info['nearest_distance']:.4f})")
            print(f"  Total: {total_suspected} suspected mislabels")


if __name__ == "__main__":
    main()
