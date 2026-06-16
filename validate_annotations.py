#!/usr/bin/env python3
"""
Outil de validation visuelle des auto-annotations avant entraînement.

Pour chaque espèce, échantillonne N images, affiche la bbox auto-détectée,
et permet à l'utilisateur de valider, corriger ou rejeter.

Commandes dans la fenêtre :
  Entrée / n  → Accepter la bbox verte et passer à l'image suivante
  Clic+glisser → Redessiner la bbox manuellement (remplace l'auto-détection)
  d           → Marquer "pas d'oiseau" (supprimer la bbox)
  z           → Annuler le dernier tracé manuel
  s           → Passer (skip) sans valider
  q           → Quitter et sauvegarder

Les corrections sont sauvegardées dans dataset/europe/annotations.json (format COCO).
Les images non validées utilisent l'auto-détection (meilleur score).

Usage:
    python validate_annotations.py [--samples-per-species 5] [--species "Parus major"]
"""

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector
from PIL import Image

from auto_annotate import BirdAnnotator

DATASET_DIR = Path(__file__).parent / "dataset" / "europe"
ANNOTATIONS_PATH = DATASET_DIR / "annotations.json"
METADATA_PATH = DATASET_DIR / "metadata.json"


def load_existing_annotations() -> dict:
    if ANNOTATIONS_PATH.exists():
        with open(ANNOTATIONS_PATH) as f:
            return json.load(f)
    return {"validated": {}, "stats": {"total_validated": 0, "total_corrected": 0, "total_rejected": 0}}


def save_annotations(data: dict):
    with open(ANNOTATIONS_PATH, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def validate_image(filepath: Path, auto_bbox: dict | None, species_info: dict | None = None) -> tuple[str, list | None]:
    """Affiche une image avec sa bbox auto-détectée et attend la validation.
    Retourne (action, bbox_corrigée) où action = 'accept'|'correct'|'reject'|'skip'|'quit'."""

    img = Image.open(filepath)
    w, h = img.size

    manual_boxes = []
    rect_patches = []

    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(img)

    slug = filepath.parent.name
    if species_info:
        fr = species_info.get("french_name", "")
        en = species_info.get("english_name", "")
        sci = species_info.get("scientific_name", slug)
        species_label = f"{fr}  ({en})  —  {sci}"
    else:
        species_label = slug.replace("_", " ").capitalize()

    score_text = f"  |  score: {auto_bbox['score']:.2f}" if auto_bbox else ""
    status = "AUTO-DÉTECTÉ" if auto_bbox else "AUCUNE DÉTECTION"

    ax.set_title(
        f"{species_label}\n"
        f"{status}{score_text}\n"
        "[Entrée] accepter  |  [clic+glisser] corriger  |  [d] rejeter  |  [s] passer  |  [q] quitter",
        fontsize=11,
    )

    if auto_bbox:
        x, y, bw, bh = auto_bbox["bbox"]
        auto_rect = patches.Rectangle(
            (x, y), bw, bh,
            linewidth=2, edgecolor="lime", facecolor="none"
        )
        ax.add_patch(auto_rect)

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)

    result = {"action": "accept", "bbox": auto_bbox["bbox"] if auto_bbox else None}

    def on_select(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        bx = max(0, min(x1, x2))
        by = max(0, min(y1, y2))
        bw = min(abs(x2 - x1), w - bx)
        bh = min(abs(y2 - y1), h - by)
        if bw < 5 or bh < 5:
            return
        box = [round(bx), round(by), round(bw), round(bh)]
        manual_boxes.append(box)
        rect = patches.Rectangle(
            (box[0], box[1]), box[2], box[3],
            linewidth=2, edgecolor="cyan", facecolor="none"
        )
        ax.add_patch(rect)
        rect_patches.append(rect)
        result["action"] = "correct"
        result["bbox"] = box
        ax.set_xlabel("Bbox manuelle tracée (cyan) — [Entrée] pour valider", fontsize=11, color="cyan")
        fig.canvas.draw()

    def on_key(event):
        if event.key in ("enter", "n"):
            plt.close(fig)
        elif event.key == "d":
            result["action"] = "reject"
            result["bbox"] = None
            plt.close(fig)
        elif event.key == "s":
            result["action"] = "skip"
            plt.close(fig)
        elif event.key == "q":
            result["action"] = "quit"
            plt.close(fig)
        elif event.key == "z" and manual_boxes:
            manual_boxes.pop()
            p = rect_patches.pop()
            p.remove()
            if manual_boxes:
                result["bbox"] = manual_boxes[-1]
            elif auto_bbox:
                result["action"] = "accept"
                result["bbox"] = auto_bbox["bbox"]
            else:
                result["bbox"] = None
            fig.canvas.draw()

    selector = RectangleSelector(
        ax, on_select,
        useblit=True, button=[1],
        minspanx=5, minspany=5,
        spancoords="pixels", interactive=False,
    )

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.tight_layout()
    plt.show()

    return result["action"], result["bbox"]


def main():
    parser = argparse.ArgumentParser(description="Validation visuelle des auto-annotations")
    parser.add_argument("--samples-per-species", type=int, default=5)
    parser.add_argument("--species", nargs="*")
    parser.add_argument("--shuffle", action="store_true", default=True)
    args = parser.parse_args()

    train_dir = DATASET_DIR / "train"
    if not train_dir.exists():
        print("Aucun dataset trouvé dans dataset/europe/train/")
        sys.exit(1)

    print("Chargement du modèle Grounding DINO...")
    annotator = BirdAnnotator()

    metadata = {}
    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            metadata = json.load(f)

    data = load_existing_annotations()
    validated = data["validated"]

    species_dirs = sorted(d for d in train_dir.iterdir() if d.is_dir())
    if args.species:
        filter_set = {s.lower().replace(" ", "_") for s in args.species}
        species_dirs = [d for d in species_dirs if d.name in filter_set]

    total_species = len(species_dirs)
    stats = {"accepted": 0, "corrected": 0, "rejected": 0, "skipped": 0}
    quit_requested = False

    print(f"\n{'='*60}")
    print(f"  Validation des annotations — {total_species} espèces")
    print(f"  {args.samples_per_species} images/espèce")
    print(f"  {len(validated)} images déjà validées")
    print(f"{'='*60}\n")

    for sp_idx, sp_dir in enumerate(species_dirs, 1):
        if quit_requested:
            break

        images = [
            f for f in sp_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        ]

        unvalidated = [f for f in images if f.name not in validated.get(sp_dir.name, {})]
        if not unvalidated:
            continue

        if args.shuffle:
            random.shuffle(unvalidated)
        sample = unvalidated[:args.samples_per_species]

        species_name = sp_dir.name.replace("_", " ").title()
        print(f"[{sp_idx}/{total_species}] {species_name} — {len(sample)} images à valider")

        for img_path in sample:
            auto_det = annotator.best_detection(str(img_path))
            sp_info = metadata.get(sp_dir.name)
            action, bbox = validate_image(img_path, auto_det, sp_info)

            if sp_dir.name not in validated:
                validated[sp_dir.name] = {}

            if action == "accept":
                validated[sp_dir.name][img_path.name] = {
                    "bbox": bbox, "status": "accepted"
                }
                stats["accepted"] += 1
            elif action == "correct":
                validated[sp_dir.name][img_path.name] = {
                    "bbox": bbox, "status": "corrected"
                }
                stats["corrected"] += 1
            elif action == "reject":
                validated[sp_dir.name][img_path.name] = {
                    "bbox": None, "status": "rejected"
                }
                stats["rejected"] += 1
            elif action == "skip":
                stats["skipped"] += 1
            elif action == "quit":
                quit_requested = True
                break

            data["validated"] = validated
            data["stats"]["total_validated"] = sum(
                len(v) for v in validated.values()
            )
            data["stats"]["total_corrected"] = sum(
                1 for sp in validated.values()
                for info in sp.values()
                if info.get("status") == "corrected"
            )
            data["stats"]["total_rejected"] = sum(
                1 for sp in validated.values()
                for info in sp.values()
                if info.get("status") == "rejected"
            )
            save_annotations(data)

    print(f"\n{'='*60}")
    print(f"  Résultat de la session")
    print(f"  Acceptées : {stats['accepted']}")
    print(f"  Corrigées : {stats['corrected']}")
    print(f"  Rejetées  : {stats['rejected']}")
    print(f"  Passées   : {stats['skipped']}")
    print(f"  Total validées (cumulé) : {data['stats']['total_validated']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
