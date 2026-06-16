"""Démo webcam — classification d'oiseaux en temps réel.

Usage :
    python demo_webcam.py                  # meilleur modèle du dernier run
    python demo_webcam.py 3               # meilleur modèle du run 3
    python demo_webcam.py chemin.pth      # checkpoint spécifique

    python demo_webcam.py 3 --compare     # tous les modèles du run 3 côte à côte
    python demo_webcam.py --compare       # meilleur de chaque run côte à côte (2-4 derniers)
    python demo_webcam.py --compare 1 3   # meilleur du run 1 vs meilleur du run 3
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import models, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

METADATA = Path("dataset/europe/metadata.json")
IMAGE_SIZE = 224
TOP_K = 5

ARCH_RANK = ["vit_b_16", "efficientnetv2_b2", "efficientnet_b0", "mobilenetv2"]

ARCH_DISPLAY = {
    "mobilenetv2": "MobileNetV2",
    "efficientnet_b0": "EfficientNet-B0",
    "efficientnetv2_b2": "EfficientNetV2-B2",
    "vit_b_16": "ViT-B/16",
}


def create_model(arch, num_classes):
    if arch == "mobilenetv2":
        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.0),
            nn.Linear(in_features, num_classes),
        )
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.0),
            nn.Linear(in_features, num_classes),
        )
    elif arch == "efficientnetv2_b2":
        import timm
        model = timm.create_model(
            "tf_efficientnetv2_b2.in1k", pretrained=False, num_classes=num_classes,
        )
        in_features = model.num_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.0),
            nn.Linear(in_features, num_classes),
        )
    elif arch == "vit_b_16":
        model = models.vit_b_16(weights=None)
        model.heads = nn.Sequential(
            nn.Dropout(p=0.0),
            nn.Linear(model.hidden_dim, num_classes),
        )
    else:
        raise ValueError(f"Architecture inconnue : {arch}")
    return model


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    label_map = ckpt["label_map"]
    arch = ckpt["arch"]
    num_classes = len(label_map)

    model = create_model(arch, num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    idx_to_species = {v: k for k, v in label_map.items()}
    return model, idx_to_species, arch


def load_french_names(metadata_path):
    if not metadata_path.exists():
        return {}
    with open(metadata_path) as f:
        meta = json.load(f)
    return {slug: info.get("french_name", slug) for slug, info in meta.items()}


def get_inference_transform():
    return transforms.Compose([
        transforms.Resize(int(IMAGE_SIZE * 256 / 224)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def predict(model, frame, transform, device):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)

    top_probs, top_indices = probs.topk(TOP_K, dim=1)
    return top_probs[0].cpu().numpy(), top_indices[0].cpu().numpy()


def draw_panel(panel, top_probs, top_indices, idx_to_species, french_names,
               label):
    h, w = panel.shape[:2]

    overlay = panel.copy()
    cv2.rectangle(overlay, (5, 5), (w - 5, 35 + TOP_K * 30), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, panel, 0.4, 0, panel)

    cv2.putText(panel, label, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        species = idx_to_species[idx]
        fr_name = french_names.get(species, species)

        y = 52 + i * 30
        conf = prob * 100

        if i == 0 and conf > 30:
            color = (0, 255, 0) if conf > 60 else (0, 200, 255)
        else:
            color = (180, 180, 180)

        bar_width = max(1, int(prob * (w - 40)))
        cv2.rectangle(panel, (10, y - 4), (10 + bar_width, y + 16), color, -1)

        text = f"{fr_name} {conf:.0f}%"
        cv2.putText(panel, text, (12, y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

    return panel


def draw_single(frame, top_probs, top_indices, idx_to_species, french_names,
                arch, num_classes):
    h, w = frame.shape[:2]
    display_name = ARCH_DISPLAY.get(arch, arch)

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (w - 10, 40 + TOP_K * 40), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    title = f"Bird Detection — {display_name} ({num_classes} especes)"
    cv2.putText(frame, title, (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        species = idx_to_species[idx]
        fr_name = french_names.get(species, species)
        sci_name = species.replace("_", " ").title()

        y = 70 + i * 40
        conf = prob * 100

        if i == 0 and conf > 30:
            color = (0, 255, 0) if conf > 60 else (0, 200, 255)
        else:
            color = (180, 180, 180)

        bar_width = int(prob * (w - 300))
        cv2.rectangle(frame, (20, y - 5), (20 + bar_width, y + 20), color, -1)

        text = f"{fr_name} ({sci_name}) — {conf:.1f}%"
        cv2.putText(frame, text, (25, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(frame, "Q pour quitter",
                (w - 180, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    return frame


def find_checkpoints_in_dir(directory):
    """Cherche les best_*.pth dans un dossier, y compris dans un sous-dossier output/."""
    directory = Path(directory)
    candidates = list(directory.glob("best_*.pth"))
    if not candidates:
        sub = directory / "output"
        if sub.exists():
            candidates = list(sub.glob("best_*.pth"))
    return candidates


def find_best_in_run(run_dir):
    candidates = find_checkpoints_in_dir(run_dir)
    if not candidates:
        return None, []

    ranked = []
    for pth in candidates:
        ckpt = torch.load(pth, map_location="cpu", weights_only=False)
        arch = ckpt.get("arch", "")
        rank = ARCH_RANK.index(arch) if arch in ARCH_RANK else len(ARCH_RANK)
        display = ARCH_DISPLAY.get(arch, arch)
        ranked.append((rank, pth, display))

    ranked.sort()
    return ranked[0][1], ranked


def find_all_in_run(run_dir):
    _, ranked = find_best_in_run(run_dir)
    return [pth for _, pth, _ in ranked]


def list_runs():
    output_dir = Path("output")
    if not output_dir.exists():
        return []
    runs = []
    for entry in sorted(output_dir.iterdir()):
        if entry.is_dir() and find_checkpoints_in_dir(entry):
            runs.append(entry)
    return runs


def resolve_run_dir(arg):
    output_dir = Path("output")
    if arg is None:
        runs = list_runs()
        if not runs:
            print("Aucun run trouvé dans output/")
            sys.exit(1)
        return runs[-1]

    if arg.isdigit():
        run_dir = output_dir / arg
        if not run_dir.exists():
            print(f"Run {arg} introuvable (pas de dossier output/{arg}/)")
            print("\nRuns disponibles :")
            for r in list_runs():
                print(f"  {r.name}")
            sys.exit(1)
        return run_dir

    return None


def resolve_checkpoint(arg):
    if arg is None:
        run_dir = resolve_run_dir(None)
        best, ranked = find_best_in_run(run_dir)
        print(f"Run {run_dir.name} — modèles disponibles :")
        for _, pth, display in ranked:
            marker = " <--" if pth == best else ""
            print(f"  {display:20s}  {pth}{marker}")
        return best

    if arg.endswith(".pth"):
        path = Path(arg)
        if not path.exists():
            print(f"Checkpoint introuvable : {path}")
            sys.exit(1)
        return path

    if arg.isdigit():
        run_dir = resolve_run_dir(arg)
        best, ranked = find_best_in_run(run_dir)
        if best is None:
            print(f"Aucun checkpoint best_*.pth dans output/{arg}/")
            sys.exit(1)
        print(f"Run {arg} — modèles disponibles :")
        for _, pth, display in ranked:
            marker = " <--" if pth == best else ""
            print(f"  {display:20s}  {pth}{marker}")
        return best

    print(f"Argument non reconnu : {arg}")
    print("Usage : python demo_webcam.py [numéro_de_run | chemin.pth] [--compare]")
    sys.exit(1)


def run_single(checkpoint_path, device, french_names, transform):
    print("Chargement du modèle...")
    model, idx_to_species, arch = load_model(checkpoint_path, device)
    num_classes = len(idx_to_species)
    display_name = ARCH_DISPLAY.get(arch, arch)
    print(f"Modèle : {display_name} — {num_classes} espèces")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Impossible d'ouvrir la webcam")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("Webcam ouverte — Q pour quitter")

    frame_count = 0
    top_probs = np.zeros(TOP_K)
    top_indices = np.zeros(TOP_K, dtype=int)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 3 == 0:
            top_probs, top_indices = predict(model, frame, transform, device)
        frame = draw_single(frame, top_probs, top_indices,
                            idx_to_species, french_names, arch, num_classes)
        cv2.imshow("Bird Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


def run_compare(entries, device, french_names, transform):
    """entries: list of (checkpoint_path, label)"""
    n = len(entries)
    loaded = []
    for cp, label in entries:
        print(f"Chargement de {cp}...")
        model, idx_to_species, arch = load_model(cp, device)
        display_name = ARCH_DISPLAY.get(arch, arch)
        print(f"  {display_name} — {len(idx_to_species)} espèces")
        loaded.append({
            "model": model,
            "idx_to_species": idx_to_species,
            "arch": arch,
            "label": label,
            "top_probs": np.zeros(TOP_K),
            "top_indices": np.zeros(TOP_K, dtype=int),
        })

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Impossible d'ouvrir la webcam")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print(f"\nWebcam ouverte — {n} modèles côte à côte — Q pour quitter")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cam_h, cam_w = frame.shape[:2]
        panel_w = cam_w // n

        if frame_count % 3 == 0:
            for entry in loaded:
                entry["top_probs"], entry["top_indices"] = predict(
                    entry["model"], frame, transform, device)

        panels = []
        for i, entry in enumerate(loaded):
            panel = cv2.resize(frame, (panel_w, cam_h))
            panel = draw_panel(panel, entry["top_probs"], entry["top_indices"],
                               entry["idx_to_species"], french_names,
                               entry["label"])
            if i > 0:
                cv2.line(panel, (0, 0), (0, cam_h), (255, 255, 255), 2)
            panels.append(panel)

        combined = np.hstack(panels)
        cv2.putText(combined, "Q pour quitter",
                    (combined.shape[1] - 180, combined.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imshow("Bird Detection — Comparaison", combined)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    compare = "--compare" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--compare"]
    run_numbers = [a for a in args if a.isdigit()]
    pth_args = [a for a in args if a.endswith(".pth")]

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device : {device}")

    french_names = load_french_names(METADATA)
    transform = get_inference_transform()

    if not compare:
        arg = args[0] if args else None
        checkpoint_path = resolve_checkpoint(arg)
        print(f"\nCheckpoint : {checkpoint_path}")
        run_single(checkpoint_path, device, french_names, transform)
        return

    # --compare avec des .pth explicites
    if pth_args:
        entries = []
        for p in pth_args:
            path = Path(p)
            if not path.exists():
                print(f"Checkpoint introuvable : {path}")
                sys.exit(1)
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            arch = ckpt.get("arch", "?")
            display = ARCH_DISPLAY.get(arch, arch)
            entries.append((path, f"{display} ({path.parent.name})"))
        if len(entries) > 4:
            entries = entries[:4]
        run_compare(entries, device, french_names, transform)
        return

    # --compare avec des numéros de run : comparer le meilleur de chaque run
    if len(run_numbers) >= 2:
        entries = []
        for num in run_numbers[:4]:
            run_dir = resolve_run_dir(num)
            best, ranked = find_best_in_run(run_dir)
            if best is None:
                print(f"Aucun checkpoint dans le run {num}")
                sys.exit(1)
            arch = ranked[0][2]
            entries.append((best, f"Run {num} — {arch}"))
        run_compare(entries, device, french_names, transform)
        return

    # --compare avec un seul numéro : comparer les modèles dans ce run
    if len(run_numbers) == 1:
        run_dir = resolve_run_dir(run_numbers[0])
        checkpoints = find_all_in_run(run_dir)
        if len(checkpoints) < 2:
            print(f"Un seul modèle dans le run {run_numbers[0]}")
            run_single(checkpoints[0], device, french_names, transform)
            return
        entries = []
        for cp in checkpoints:
            ckpt = torch.load(cp, map_location="cpu", weights_only=False)
            arch = ckpt.get("arch", "?")
            display = ARCH_DISPLAY.get(arch, arch)
            entries.append((cp, f"{display}"))
        print(f"\nRun {run_numbers[0]} — comparaison de {len(entries)} modèles")
        run_compare(entries, device, french_names, transform)
        return

    # --compare sans argument : meilleur de chaque run (2 à 4 derniers)
    runs = list_runs()
    if len(runs) < 2:
        print("Il faut au moins 2 runs pour comparer")
        if runs:
            best, _ = find_best_in_run(runs[0])
            run_single(best, device, french_names, transform)
        sys.exit(1)

    runs = runs[-4:]
    entries = []
    for run_dir in runs:
        best, ranked = find_best_in_run(run_dir)
        if best is None:
            continue
        arch = ranked[0][2]
        entries.append((best, f"Run {run_dir.name} — {arch}"))

    print(f"\nComparaison des {len(entries)} derniers runs :")
    for cp, label in entries:
        print(f"  {label:30s}  {cp}")

    run_compare(entries, device, french_names, transform)


if __name__ == "__main__":
    main()
