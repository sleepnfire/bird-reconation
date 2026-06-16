"""
Auto-annotation d'images d'oiseaux avec bounding boxes au format COCO.

Backend : Grounding DINO (Liu et al., ECCV 2024) — 52.5 AP COCO zero-shot
Détection zero-shot guidée par texte (prompt "a bird").
Thresholds oiseaux : arxiv 2603.00184v1 (box=0.30, text=0.25)

Variantes :
- grounding_dino_tiny (défaut) : IDEA-Research/grounding-dino-tiny (~341 Mo)
- grounding_dino_base : IDEA-Research/grounding-dino-base (~856 Mo)
"""

import json
import multiprocessing
from pathlib import Path

import torch
from PIL import Image, ImageDraw, UnidentifiedImageError

VALID_BACKENDS = {"grounding_dino_tiny", "grounding_dino_base"}
GDINO_TEXT_PROMPT = "a bird"
GDINO_MODEL_IDS = {
    "grounding_dino_tiny": "IDEA-Research/grounding-dino-tiny",
    "grounding_dino_base": "IDEA-Research/grounding-dino-base",
}


_worker_annotator = None


def _init_worker(threshold, min_area_ratio, device=None,
                 backend="grounding_dino_tiny", text_threshold=0.25):
    global _worker_annotator
    _worker_annotator = BirdAnnotator(threshold=threshold, min_area_ratio=min_area_ratio,
                                      device=device, backend=backend,
                                      text_threshold=text_threshold)


def _process_species(species_dir_str):
    global _worker_annotator
    sp_dir = Path(species_dir_str)
    ann_path = sp_dir / "annotations.json"

    annotations = _worker_annotator.annotate_species(species_dir_str)
    detected = sum(1 for v in annotations.values() if v is not None)
    total_img = len(annotations)

    with open(ann_path, "w") as f:
        json.dump(annotations, f, indent=2)

    pct = detected / total_img * 100 if total_img else 0
    print(f"  {sp_dir.name}: {detected}/{total_img} détections ({pct:.0f}%)", flush=True)
    return {"species": sp_dir.name, "status": "annotated", "images": total_img, "detected": detected}


class BirdAnnotator:

    def __init__(self, threshold: float = 0.3, min_area_ratio: float = 0.01,
                 device: str | None = None, backend: str = "grounding_dino_tiny",
                 text_threshold: float = 0.25):
        if backend not in VALID_BACKENDS:
            raise ValueError(f"backend doit être l'un de {VALID_BACKENDS}, reçu: {backend!r}")
        self.threshold = threshold
        self.min_area_ratio = min_area_ratio
        self.backend = backend
        self.text_threshold = text_threshold
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        model_id = GDINO_MODEL_IDS[backend]
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

    def annotate_image(
        self, image_path: str, threshold: float | None = None
    ) -> list[dict]:
        thresh = threshold if threshold is not None else self.threshold

        try:
            img = Image.open(image_path)
            img.load()
        except (OSError, UnidentifiedImageError, SyntaxError):
            return []

        img = img.convert("RGB")
        img_w, img_h = img.size
        img_area = img_w * img_h

        inputs = self.processor(
            images=img,
            text=[[GDINO_TEXT_PROMPT]],
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results_list = self.processor.post_process_grounded_object_detection(
            outputs,
            threshold=thresh,
            text_threshold=self.text_threshold,
            target_sizes=[(img_h, img_w)],
        )

        detections = []
        result = results_list[0]
        for box, score in zip(result["boxes"].cpu(), result["scores"].cpu()):
            x1, y1, x2, y2 = box.tolist()
            x, y = max(0.0, x1), max(0.0, y1)
            w = min(x2, float(img_w)) - x
            h = min(y2, float(img_h)) - y
            if w <= 0 or h <= 0:
                continue
            if img_area > 0 and (w * h) / img_area < self.min_area_ratio:
                continue
            detections.append({
                "bbox": [round(x), round(y), round(w), round(h)],
                "score": round(score.item(), 4),
            })
        return detections

    def annotate_batch(self, image_paths: list[str], threshold: float | None = None) -> dict[str, list[dict]]:
        """Annote un batch d'images (séquentiel, images corrompues exclues)."""
        results = {}
        for path in image_paths:
            try:
                img = Image.open(path)
                img.load()
            except (OSError, UnidentifiedImageError, SyntaxError):
                continue
            results[path] = self.annotate_image(path, threshold)
        return results

    def best_detection(
        self, image_path: str, threshold: float | None = None
    ) -> dict | None:
        """Retourne la détection la plus probable du sujet principal.
        Prend la plus grande bbox ≤ 50% de l'image. Si toutes > 50%, prend le meilleur score."""
        results = self.annotate_image(image_path, threshold)
        if not results:
            return None
        try:
            img = Image.open(image_path)
            img_area = img.size[0] * img.size[1]
        except (OSError, UnidentifiedImageError, SyntaxError):
            return max(results, key=lambda d: d["score"])

        reasonable = [d for d in results if d["bbox"][2] * d["bbox"][3] / img_area <= 0.50]
        if reasonable:
            return max(reasonable, key=lambda d: d["bbox"][2] * d["bbox"][3])
        return max(results, key=lambda d: d["score"])

    def build_coco_annotations(self, image_dir: str) -> dict:
        image_dir = Path(image_dir)
        image_files = sorted(
            f for f in image_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        )

        images = []
        annotations = []
        ann_id = 1

        for img_id, filepath in enumerate(image_files, start=1):
            try:
                img = Image.open(filepath)
                img_w, img_h = img.size
            except (OSError, UnidentifiedImageError, SyntaxError):
                img_w, img_h = 0, 0

            images.append({
                "id": img_id,
                "file_name": filepath.name,
                "width": img_w,
                "height": img_h,
            })

            detections = self.annotate_image(str(filepath))
            for det in detections:
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": det["bbox"],
                    "area": det["bbox"][2] * det["bbox"][3],
                    "score": det["score"],
                    "iscrowd": 0,
                })
                ann_id += 1

        categories = [{"id": 1, "name": "bird", "supercategory": "animal"}]

        return {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }

    def annotate_species(self, species_dir: str) -> dict:
        species_dir = Path(species_dir)
        image_files = sorted(
            f for f in species_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        annotations = {}
        for filepath in image_files:
            det = self.best_detection(str(filepath))
            annotations[filepath.name] = det
        return annotations

    def annotate_dataset(self, train_dir: str, resume: bool = True, workers: int = 1) -> dict:
        train_dir = Path(train_dir)
        species_dirs = sorted(d for d in train_dir.iterdir() if d.is_dir())
        total = len(species_dirs)

        stats = {"total_species": total, "annotated": 0, "skipped": 0, "images": 0, "detected": 0}

        if workers <= 1:
            for i, sp_dir in enumerate(species_dirs, 1):
                ann_path = sp_dir / "annotations.json"

                if resume and ann_path.exists():
                    stats["skipped"] += 1
                    continue

                annotations = self.annotate_species(str(sp_dir))

                detected = sum(1 for v in annotations.values() if v is not None)
                total_img = len(annotations)

                with open(ann_path, "w") as f:
                    json.dump(annotations, f, indent=2)

                stats["annotated"] += 1
                stats["images"] += total_img
                stats["detected"] += detected

                pct = detected / total_img * 100 if total_img else 0
                print(f"[{i}/{total}] {sp_dir.name}: {detected}/{total_img} détections ({pct:.0f}%)")
        else:
            to_process = []
            for sp_dir in species_dirs:
                if resume and (sp_dir / "annotations.json").exists():
                    stats["skipped"] += 1
                else:
                    to_process.append(str(sp_dir))

            print(f"Annotation parallèle: {len(to_process)} espèces sur {workers} workers "
                  f"({stats['skipped']} déjà faites)")

            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(workers, initializer=_init_worker,
                          initargs=(self.threshold, self.min_area_ratio, None, self.backend,
                                    self.text_threshold)) as pool:
                for result in pool.imap_unordered(_process_species, to_process):
                    if result["status"] == "annotated":
                        stats["annotated"] += 1
                        stats["images"] += result["images"]
                        stats["detected"] += result["detected"]

        print(f"\nTerminé: {stats['annotated']} annotées, {stats['skipped']} déjà faites, "
              f"{stats['detected']}/{stats['images']} détections")
        return stats

    def generate_samples(
        self, image_dir: str, output_dir: str, max_samples: int = 10
    ) -> list[str]:
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted(
            f for f in image_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        )[:max_samples]

        generated = []
        for filepath in image_files:
            try:
                img = Image.open(filepath).convert("RGB")
            except (OSError, UnidentifiedImageError, SyntaxError):
                continue

            det = self.best_detection(str(filepath))
            draw = ImageDraw.Draw(img)
            if det:
                x, y, w, h = det["bbox"]
                draw.rectangle([x, y, x + w, y + h], outline="lime", width=3)
                draw.text((x, max(0, y - 12)), f"{det['score']:.2f}", fill="lime")

            out_path = output_dir / filepath.name
            img.save(out_path)
            generated.append(str(out_path))

        return generated


def build_parser():
    import argparse
    parser = argparse.ArgumentParser(
        description="Auto-annotation d'images d'oiseaux avec Grounding DINO",
    )
    sub = parser.add_subparsers(dest="command")
    annotate = sub.add_parser("annotate", help="Annoter un dossier d'espèces")
    annotate.add_argument("directory", help="Dossier contenant les sous-dossiers espèces")
    annotate.add_argument("--workers", type=int, default=1,
                          help="Nombre de workers multiprocessing (défaut: 1)")
    annotate.add_argument("--force", action="store_true",
                          help="Ré-annoter même si annotations.json existe")
    annotate.add_argument("--backend", default="grounding_dino_tiny",
                          choices=["grounding_dino_tiny", "grounding_dino_base"],
                          help="Modèle de détection (défaut: grounding_dino_tiny)")
    annotate.add_argument("--threshold", type=float, default=0.3,
                          help="Seuil de confiance bbox (défaut: 0.3)")
    annotate.add_argument("--text-threshold", type=float, default=0.25,
                          help="Seuil de confiance texte (défaut: 0.25)")
    return parser


def main():
    import sys
    parser = build_parser()
    args = parser.parse_args()

    if args.command != "annotate":
        parser.print_help()
        sys.exit(1)

    directory = Path(args.directory)
    if not directory.exists():
        print(f"Erreur : dossier introuvable — {directory}")
        sys.exit(1)

    annotator = BirdAnnotator(
        backend=args.backend,
        threshold=args.threshold,
        text_threshold=args.text_threshold,
    )
    annotator.annotate_dataset(
        str(directory),
        resume=not args.force,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
