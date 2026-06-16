#!/usr/bin/env python3
"""
Outil visuel pour annoter les images fixtures avec des bounding boxes.

Pour chaque image, une fenêtre s'ouvre :
  - Clique gauche + glisser → dessiner un rectangle
  - Tu peux dessiner PLUSIEURS rectangles (s'il y a plusieurs oiseaux)
  - Touche 'z' → annuler le dernier rectangle
  - Touche 'n' ou Entrée → passer à l'image suivante (valider)
  - Touche 'q' → quitter et sauvegarder tout ce qui a été fait

Les coordonnées sont sauvegardées automatiquement dans expected_annotations.json
au format COCO : [x, y, largeur, hauteur] en pixels.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector
from PIL import Image

FIXTURES_DIR = Path(__file__).parent / "images"
ANNOTATIONS_FILE = Path(__file__).parent / "expected_annotations.json"


def load_annotations():
    with open(ANNOTATIONS_FILE) as f:
        return json.load(f)


def save_annotations(data):
    with open(ANNOTATIONS_FILE, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  → Sauvegardé dans {ANNOTATIONS_FILE.name}")


def species_from_filename(filename: str) -> str:
    """Extrait le nom d'espèce lisible depuis le nom de fichier '{espece}__{photo_id}.ext'."""
    if "__" in filename:
        slug = filename.split("__")[0]
        return slug.replace("_", " ").capitalize()
    return ""


def annotate_image(filepath, existing_boxes=None):
    img = Image.open(filepath)
    w, h = img.size

    species = species_from_filename(filepath.name)
    species_label = f"  →  {species}" if species else ""

    boxes = []
    rect_patches = []

    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(img)
    ax.set_title(
        f"{filepath.name}  ({w}×{h}){species_label}\n"
        f"Dessine les rectangles autour de : {species or 'oiseau(x)'}\n"
        "[z] annuler  |  [n/Entrée] suivant  |  [q] quitter",
        fontsize=11,
    )
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)

    if existing_boxes and existing_boxes != ["A_REMPLIR: [x, y, width, height]"]:
        for box in existing_boxes:
            if isinstance(box, list) and len(box) == 4:
                x, y, bw, bh = box
                rect = patches.Rectangle(
                    (x, y), bw, bh,
                    linewidth=2, edgecolor="cyan", facecolor="none", linestyle="--"
                )
                ax.add_patch(rect)
        ax.set_title(
            f"{filepath.name}  ({w}×{h}){species_label}\n"
            f"{len(existing_boxes)} annotations existantes (cyan) — redessine ou [n] pour garder\n"
            "[z] annuler  |  [n/Entrée] suivant  |  [q] quitter",
            fontsize=11,
        )

    result = {"action": "next"}

    def on_select(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        x = max(0, min(x1, x2))
        y = max(0, min(y1, y2))
        bw = min(abs(x2 - x1), w - x)
        bh = min(abs(y2 - y1), h - y)
        if bw < 5 or bh < 5:
            return
        box = [round(x), round(y), round(bw), round(bh)]
        boxes.append(box)
        rect = patches.Rectangle(
            (box[0], box[1]), box[2], box[3],
            linewidth=2, edgecolor="lime", facecolor="none"
        )
        ax.add_patch(rect)
        rect_patches.append(rect)
        ax.set_xlabel(f"{len(boxes)} rectangle(s) dessiné(s)", fontsize=12, color="green")
        fig.canvas.draw()

    def on_key(event):
        if event.key == "z" and boxes:
            boxes.pop()
            p = rect_patches.pop()
            p.remove()
            ax.set_xlabel(f"{len(boxes)} rectangle(s) dessiné(s)", fontsize=12, color="green")
            fig.canvas.draw()
        elif event.key in ("n", "enter"):
            result["action"] = "next"
            plt.close(fig)
        elif event.key == "q":
            result["action"] = "quit"
            plt.close(fig)

    selector = RectangleSelector(
        ax, on_select,
        useblit=True,
        button=[1],
        minspanx=5, minspany=5,
        spancoords="pixels",
        interactive=False,
    )

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.tight_layout()
    plt.show()

    return boxes, result["action"]


def main():
    data = load_annotations()

    image_files = sorted(
        f for f in FIXTURES_DIR.iterdir()
        if f.name in data and f.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    already_done = sum(
        1 for name, info in data.items()
        if not name.startswith("_") and info.get("bird_count") != "A_REMPLIR"
    )
    remaining = [
        f for f in image_files
        if data[f.name].get("bird_count") == "A_REMPLIR"
    ]

    print(f"\n{'='*60}")
    print(f"  Annotation des images fixtures")
    print(f"  {already_done} déjà annotées, {len(remaining)} restantes")
    print(f"{'='*60}\n")

    if not remaining:
        print("Toutes les images sont déjà annotées !")
        redo = input("Veux-tu tout ré-annoter ? (o/n) : ").strip().lower()
        if redo == "o":
            remaining = image_files
        else:
            return

    for i, filepath in enumerate(remaining):
        print(f"\n[{i+1}/{len(remaining)}] {filepath.name}")
        existing = data[filepath.name].get("boxes")
        boxes, action = annotate_image(filepath, existing)

        if boxes:
            data[filepath.name]["boxes"] = boxes
            data[filepath.name]["bird_count"] = len(boxes)
            print(f"  ✓ {len(boxes)} oiseau(x) annoté(s)")
        elif existing and existing != ["A_REMPLIR: [x, y, width, height]"]:
            print(f"  → Annotations existantes conservées ({len(existing)} boxes)")
        else:
            data[filepath.name]["boxes"] = []
            data[filepath.name]["bird_count"] = 0
            print(f"  → Aucun rectangle (0 oiseau)")

        save_annotations(data)

        if action == "quit":
            print(f"\nArrêt. Progression sauvegardée ({i+1}/{len(remaining)}).")
            break

    done = sum(
        1 for name, info in data.items()
        if not name.startswith("_") and info.get("bird_count") != "A_REMPLIR"
    )
    print(f"\nTerminé : {done}/{len(image_files)} images annotées.")


if __name__ == "__main__":
    main()
