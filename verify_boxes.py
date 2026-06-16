#!/usr/bin/env python3
"""
Génère des images avec les bounding boxes dessinées pour vérification visuelle.
Liste aussi les images sans détection dans no_detections.txt.

Usage:
    python verify_boxes.py                                  # 3 espèces aléatoires, 10 images chacune
    python verify_boxes.py --species parus_major            # une espèce précise
    python verify_boxes.py --species parus_major turdus_merula --samples 5
    python verify_boxes.py --split validation               # vérifier le split validation
    python verify_boxes.py --all --samples 3                # toutes les espèces, 3 images chacune
"""

import argparse
import json
import multiprocessing
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, UnidentifiedImageError

from auto_annotate import BirdAnnotator

DATASET_DIR = Path(__file__).parent / "dataset" / "europe"
OUTPUT_DIR = Path(__file__).parent / "samples"


def collect_no_detections(split_dir: Path) -> dict:
    """Parcourt les annotations.json de chaque espèce et liste les images sans détection."""
    no_det = {}
    for sp_dir in sorted(split_dir.iterdir()):
        ann_path = sp_dir / "annotations.json"
        if not ann_path.exists():
            continue
        with open(ann_path) as f:
            annotations = json.load(f)
        missing = [name for name, det in annotations.items() if det is None]
        if missing:
            no_det[sp_dir.name] = missing
    return no_det


def _best_from_detections(detections: list[dict], img_path: Path) -> dict | None:
    """Sélectionne la meilleure détection : plus grande bbox <= 50% de l'image."""
    if not detections:
        return None
    try:
        img = Image.open(img_path)
        img_area = img.size[0] * img.size[1]
    except (OSError, UnidentifiedImageError, SyntaxError):
        return max(detections, key=lambda d: d["score"])
    reasonable = [d for d in detections if d["bbox"][2] * d["bbox"][3] / img_area <= 0.50]
    if reasonable:
        return max(reasonable, key=lambda d: d["bbox"][2] * d["bbox"][3])
    return max(detections, key=lambda d: d["score"])


_retry_annotator = None


def _init_retry_worker(threshold, backend="grounding_dino_tiny"):
    global _retry_annotator
    _retry_annotator = BirdAnnotator(threshold=threshold, backend=backend)


def _retry_species(args: tuple) -> dict:
    global _retry_annotator
    sp_dir_str, files, threshold = args
    sp_dir = Path(sp_dir_str)
    ann_path = sp_dir / "annotations.json"

    with open(ann_path) as f:
        annotations = json.load(f)

    recovered = 0
    for name in files:
        path = sp_dir / name
        if not path.exists():
            continue
        dets = _retry_annotator.annotate_image(str(path), threshold=threshold)
        best = _best_from_detections(dets, path)
        if best:
            best["retry_threshold"] = threshold
            annotations[name] = best
            recovered += 1

    with open(ann_path, "w") as f:
        json.dump(annotations, f, indent=2)

    return {"species": sp_dir.name, "retried": len(files), "recovered": recovered}


def retry_no_detections(split_dir: str, threshold: float = 0.3, workers: int = 1,
                        backend: str = "grounding_dino_tiny") -> dict:
    """Ré-annote les images sans détection avec un seuil plus bas.
    Met à jour les annotations.json en place. Ne touche pas aux détections existantes.
    workers>1 parallélise par espèce sur plusieurs cœurs CPU."""
    split_dir = Path(split_dir)
    no_det = collect_no_detections(split_dir)
    total_missing = sum(len(v) for v in no_det.values())

    if total_missing == 0:
        print("Aucune image sans détection.")
        return {"retried": 0, "recovered": 0, "still_missing": 0}

    print(f"Retry de {total_missing} images avec seuil={threshold} "
          f"(workers={workers}, backend={backend})...")

    work_items = [
        (str(split_dir / species), files, threshold)
        for species, files in no_det.items()
    ]

    recovered = 0

    if workers <= 1:
        _init_retry_worker(threshold, backend)
        for item in work_items:
            result = _retry_species(item)
            recovered += result["recovered"]
            print(f"  {result['species']}: {result['recovered']}/{result['retried']} récupérées", flush=True)
    else:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(workers, initializer=_init_retry_worker, initargs=(threshold, backend)) as pool:
            for result in pool.imap_unordered(_retry_species, work_items):
                recovered += result["recovered"]
                print(f"  {result['species']}: {result['recovered']}/{result['retried']} récupérées", flush=True)

    still_missing = total_missing - recovered
    print(f"Récupérées: {recovered}/{total_missing}, encore manquantes: {still_missing}")
    return {"retried": total_missing, "recovered": recovered, "still_missing": still_missing}


def generate_review(split_dir: Path, output_dir: Path, n_detected: int = 10, n_missing: int = 10):
    """Pour chaque espèce annotée, génère un échantillon de:
    - N images avec détection (bbox dessinée en vert)
    - N images sans détection (bordure rouge, label NO DETECTION)
    Écrase le dossier output existant."""
    if output_dir.exists():
        shutil.rmtree(output_dir)

    rng = random.Random(42)
    species_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())

    for sp_dir in species_dirs:
        ann_path = sp_dir / "annotations.json"
        if not ann_path.exists():
            continue

        with open(ann_path) as f:
            annotations = json.load(f)

        detected = [name for name, det in annotations.items() if det is not None]
        missing = [name for name, det in annotations.items() if det is None]

        if not detected and not missing:
            continue

        sp_out = output_dir / sp_dir.name
        det_out = sp_out / "detected"
        miss_out = sp_out / "no_detection"

        if detected:
            det_out.mkdir(parents=True, exist_ok=True)
            rng.shuffle(detected)
            for name in detected[:n_detected]:
                img_path = sp_dir / name
                if not img_path.exists():
                    continue
                try:
                    img = Image.open(img_path).convert("RGB")
                except (OSError, UnidentifiedImageError, SyntaxError):
                    continue
                det = annotations[name]
                draw = ImageDraw.Draw(img)
                x, y, w, h = det["bbox"]
                draw.rectangle([x, y, x + w, y + h], outline="lime", width=3)
                label = f"{det['score']:.2f}"
                if "retry_threshold" in det:
                    label += f" (retry@{det['retry_threshold']})"
                draw.text((x, max(0, y - 12)), label, fill="lime")
                img.save(det_out / name)

        if missing:
            miss_out.mkdir(parents=True, exist_ok=True)
            rng.shuffle(missing)
            for name in missing[:n_missing]:
                img_path = sp_dir / name
                if not img_path.exists():
                    continue
                try:
                    img = Image.open(img_path).convert("RGB")
                except (OSError, UnidentifiedImageError, SyntaxError):
                    continue
                draw = ImageDraw.Draw(img)
                iw, ih = img.size
                draw.rectangle([0, 0, iw - 1, ih - 1], outline="red", width=4)
                draw.text((5, 5), "NO DETECTION", fill="red")
                img.save(miss_out / name)

        n_det = len(list(det_out.iterdir())) if det_out.exists() else 0
        n_miss = len(list(miss_out.iterdir())) if miss_out.exists() else 0
        if n_miss == 0 and miss_out.exists():
            miss_out.rmdir()
        if n_det == 0 and det_out.exists():
            det_out.rmdir()

        print(f"  {sp_dir.name}: {n_det} détectées + {n_miss} manquantes")


def main():
    parser = argparse.ArgumentParser(description="Vérification visuelle des bounding boxes")
    parser.add_argument("--species", nargs="*", help="Espèce(s) à vérifier (slug)")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--samples", type=int, default=10, help="Nombre d'images par espèce")
    parser.add_argument("--all", action="store_true", help="Toutes les espèces")
    parser.add_argument("--random-species", type=int, default=3, help="Nombre d'espèces aléatoires (si --species non spécifié)")
    parser.add_argument("--no-detections", action="store_true", help="Lister les images sans détection (nécessite annotations.json)")
    parser.add_argument("--retry", action="store_true", help="Réessayer les non-détectées avec un seuil plus bas")
    parser.add_argument("--threshold", type=float, default=0.3, help="Seuil pour --retry (défaut: 0.3)")
    parser.add_argument("--workers", type=int, default=1, help="Nombre de workers CPU pour --retry")
    parser.add_argument("--backend", default="grounding_dino_tiny",
                        choices=["grounding_dino_tiny", "grounding_dino_base"],
                        help="Modèle de détection (défaut: grounding_dino_tiny)")
    parser.add_argument("--review", action="store_true", help="Échantillon: 10 détectées + 10 non-détectées par espèce")
    args = parser.parse_args()

    split_dir = DATASET_DIR / args.split
    if not split_dir.exists():
        print(f"Dossier {split_dir} introuvable")
        return

    if args.retry:
        stats = retry_no_detections(str(split_dir), threshold=args.threshold,
                                    workers=args.workers, backend=args.backend)
        if stats["still_missing"] > 0:
            out_path = OUTPUT_DIR / f"no_detections_{args.split}.txt"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            no_det = collect_no_detections(split_dir)
            with open(out_path, "w") as f:
                for species, files in no_det.items():
                    for name in files:
                        f.write(f"{species}/{name}\n")
            print(f"Images encore manquantes → {out_path}")
        return

    if args.review:
        out = OUTPUT_DIR / f"review_{args.split}"
        generate_review(split_dir, out, n_detected=args.samples, n_missing=args.samples)
        print(f"\nOuvrir le dossier : open {out}")
        return

    if args.no_detections:
        no_det = collect_no_detections(split_dir)
        total = sum(len(v) for v in no_det.values())
        out_path = OUTPUT_DIR / f"no_detections_{args.split}.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for species, files in no_det.items():
                for name in files:
                    f.write(f"{species}/{name}\n")
        print(f"{total} images sans détection sur {len(no_det)} espèces → {out_path}")
        return

    species_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())

    if args.species:
        species_dirs = [d for d in species_dirs if d.name in args.species]
        if not species_dirs:
            print(f"Espèce(s) non trouvée(s) : {args.species}")
            return
    elif not args.all:
        species_dirs = random.sample(species_dirs, min(args.random_species, len(species_dirs)))

    print("Chargement du modèle Grounding DINO...")
    annotator = BirdAnnotator()

    for sp_dir in species_dirs:
        out = OUTPUT_DIR / args.split / sp_dir.name
        annotator.generate_samples(str(sp_dir), str(out), max_samples=args.samples)
        n = len(list(out.iterdir()))
        print(f"  {sp_dir.name}: {n} images → {out}")

    print(f"\nOuvrir le dossier : open {OUTPUT_DIR / args.split}")


if __name__ == "__main__":
    main()
