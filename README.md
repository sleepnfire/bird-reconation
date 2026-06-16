# Bird Detection — Reconnaissance d'oiseaux par IA

Classification de 558 espèces d'oiseaux européens, déployable sur Raspberry Pi.

## Cibles matérielles

| Cible | Accélérateur | Modèle recommandé | Contrainte | Export |
|-------|-------------|-------------------|------------|--------|
| Pi Zero + AI Camera | Sony IMX500 | MobileNetV2 / EfficientNetV2-B2 | INT8 < 8 Mo SRAM | `--target imx500` (Sony MCT) |
| Pi 5 + AI HAT+ 2 | Hailo-10H (40 TOPS) | ViT-B/16 | 8 Go LPDDR4 | `--target hailo` (Hailo DFC) |

Le **MobileNetV2** (3.5M params) est le baseline cross-platform. Le **ViT-B/16** (86.6M params, 84.5% top-1) est le meilleur du Model Zoo Hailo. L'**EfficientNetV2-B2** (10.1M params, 77.7% top-1 quantifié) est le meilleur du Model Zoo IMX500.

## Guide rapide — Entraînement complet sur Windows

### Etape 1 — Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

> Adapter `cu124` à la version CUDA du driver (`cu121`, `cu118`…).
> Vérifier avec `nvidia-smi` la version CUDA supportée.

### Etape 2 — Entraînement des 3 modèles

Lancer dans cet ordre (le ViT en premier car il sert de teacher pour la distillation) :

```bash
# ViT-B/16 — teacher pour Hailo-10H + distillation (~le plus long)
python train.py --model vit_b_16 --epochs 80 --batch-size 64 --workers 8

# EfficientNetV2-B2 — meilleur modèle IMX500 (résolution 260)
python train.py --model efficientnetv2_b2 --image-size 260 --epochs 80 --batch-size 64 --workers 8

# MobileNetV2 — baseline cross-platform
python train.py --model mobilenetv2 --epochs 80 --batch-size 128 --workers 8
```

### Etape 3 — Knowledge Distillation (ViT → MobileNetV2)

Utilise le ViT-B/16 entraîné comme teacher pour améliorer le MobileNetV2 :

```bash
python distill.py --teacher-checkpoint output/best_vit_b_16.pth --model mobilenetv2 --epochs 80 --batch-size 128 --workers 8
```

Produit `output/best_mobilenetv2_distilled.pth`.

### Etape 4 — Export

```bash
# Hailo-10H (Pi 5) — ViT-B/16
python export.py --checkpoint output/best_vit_b_16.pth --target hailo

# IMX500 (Pi Zero) — MobileNetV2 distillé
python export.py --checkpoint output/best_mobilenetv2_distilled.pth --target onnx --check-size

# IMX500 (Pi Zero) — EfficientNetV2-B2
python export.py --checkpoint output/best_efficientnetv2_b2.pth --target onnx --image-size 260 --check-size
```

### Etape 5 — Tests (optionnel)

```bash
pytest -m "not slow"
```

---

## Installation détaillée

### Windows + NVIDIA GPU (CUDA)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### macOS (CPU / MPS)

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision
pip install -r requirements.txt
```

## Entraînement

Les images sont recadrées sur l'oiseau via les bounding boxes stockées dans `annotations.json` (par dossier espèce). Le pipeline utilise les splits existants `train/`, `validation/`, `test/`.

```bash
# MobileNetV2 — baseline cross-platform (IMX500 + Hailo)
python train.py --model mobilenetv2

# EfficientNetV2-B2 — meilleur modèle IMX500 (résolution 260 recommandée)
python train.py --model efficientnetv2_b2 --image-size 260

# ViT-B/16 — meilleur modèle Hailo-10H
python train.py --model vit_b_16

# EfficientNet-B0 — IMX500, alternative légère
python train.py --model efficientnet_b0
```

### Options CLI

| Option | Description | Défaut |
|--------|-------------|--------|
| `--model` | `mobilenetv2`, `efficientnet_b0`, `efficientnetv2_b2`, `vit_b_16` | `mobilenetv2` |
| `--dataset` | Chemin du dataset | `dataset/europe` |
| `--epochs` | Nombre d'epochs | `80` |
| `--batch-size` | Taille de batch | `128` |
| `--lr` | Learning rate de la tête | `1e-3` |
| `--backbone-lr-factor` | Multiplicateur LR pour le backbone | `0.01` |
| `--weight-decay` | Régularisation L2 | `1e-2` |
| `--dropout` | Dropout sur la tête de classification | `0.5` |
| `--warmup-epochs` | Warmup linéaire avant cosine decay | `3` |
| `--patience` | Early stopping sur val_loss (0 = désactivé) | `15` |
| `--image-size` | Taille d'entrée du modèle | `224` |
| `--freeze-backbone-epochs` | Epochs avec backbone gelé | `3` |
| `--clip-grad` | Gradient clipping max norm (0 = désactivé) | `1.0` |
| `--mixup-alpha` | Alpha MixUp (régularisation inter-batch) | `0.2` |
| `--cutmix-alpha` | Alpha CutMix | `1.0` |
| `--no-mixup` | Désactiver MixUp/CutMix | — |
| `--ema-decay` | Decay EMA (moyenne mobile exponentielle) | `0.9999` |
| `--no-ema` | Désactiver EMA | — |
| `--focal-loss` | Utiliser Focal Loss (classes déséquilibrées) | off |
| `--focal-gamma` | Gamma de la Focal Loss | `2.0` |
| `--label-smoothing` | Label smoothing | `0.1` |
| `--randaugment-ops` | Nombre d'opérations RandAugment | `3` |
| `--randaugment-magnitude` | Magnitude RandAugment | `12` |
| `--qat` | Quantization-Aware Training après entraînement (CNN uniquement) | off |
| `--qat-epochs` | Epochs QAT | `5` |
| `--logdir` | Dossier TensorBoard (`auto` = output/runs/\<model\>_\<timestamp\>) | off |
| `--device` | Forcer un device (`cpu`, `cuda`, `mps`) | Auto |
| `--workers` | Workers DataLoader | `8` |
| `--resume` | Reprendre depuis un checkpoint | — |
| `--output` | Dossier de sortie | `output/` |
| `--seed` | Graine de reproductibilité | `42` |

### Exemple Windows avec RTX 3090 Ti (24 Go VRAM)

```bash
python train.py --model vit_b_16 --epochs 80 --batch-size 64 --workers 8
python train.py --model mobilenetv2 --epochs 80 --batch-size 128 --workers 8
python train.py --model efficientnetv2_b2 --image-size 260 --batch-size 64 --workers 8
```

Le device CUDA est détecté automatiquement. Mixed precision (AMP FP16) activé sur CUDA.

## TensorBoard

Le tracking TensorBoard est optionnel, activé via `--logdir` :

```bash
# Auto : crée output/runs/<model>_<timestamp>/
python train.py --model mobilenetv2 --logdir auto

# Dossier personnalisé
python train.py --model mobilenetv2 --logdir runs/exp1

# Visualiser
tensorboard --logdir output/runs
```

Métriques loguées : loss/acc (train + val) par epoch, learning rates par groupe, hyperparamètres et résultats finaux. Fonctionne aussi avec `distill.py --logdir auto`.

## Knowledge Distillation

Le teacher (ViT-B/16 entraîné) améliore le student (MobileNetV2) via des soft targets.

```bash
# 1. Entraîner le teacher ViT-B/16
python train.py --model vit_b_16 --epochs 80

# 2. Distiller vers MobileNetV2
python distill.py \
  --teacher-checkpoint output/best_vit_b_16.pth \
  --model mobilenetv2 \
  --distill-temperature 4.0 \
  --distill-alpha 0.7 \
  --epochs 80
```

`distill.py` supporte les mêmes options que `train.py` (mixup, ema, focal-loss, etc.) plus :

| Option | Description | Défaut |
|--------|-------------|--------|
| `--teacher-checkpoint` | Checkpoint du teacher (requis) | — |
| `--distill-temperature` | Température de softmax | `4.0` |
| `--distill-alpha` | Poids de la KL loss (1-alpha = poids hard loss) | `0.7` |

## Export multi-cible

```bash
# ONNX générique + quantification dynamique INT8
python export.py --checkpoint output/best_mobilenetv2.pth --target onnx --check-size

# Hailo-10H — ONNX float32 opset 13+ pour le Hailo Dataflow Compiler
python export.py --checkpoint output/best_vit_b_16.pth --target hailo

# IMX500 — quantification statique INT8 via Sony MCT (venv Python 3.12 dédié)
.venv-imx500/bin/python export.py --checkpoint output/best_mobilenetv2.pth --target imx500
```

### Venv IMX500

Sony MCT exige `matplotlib<3.10`, incompatible avec Python 3.14. Un venv séparé Python 3.12 est fourni :

```bash
# Créer le venv (une seule fois)
bash setup-imx500-venv.sh

# Exporter
.venv-imx500/bin/python export.py --checkpoint output/best_mobilenetv2.pth --target imx500
.venv-imx500/bin/python export.py --checkpoint output/best_mobilenetv2_distilled.pth --target imx500
```

### Pipeline de déploiement

**IMX500 (Pi Zero + AI Camera)** :
```
train.py → export.py --target imx500 → IMX500 Converter → firmware .rpk
```

**Hailo-10H (Pi 5 + AI HAT+ 2)** :
```
train.py → export.py --target hailo → hailo parser → optimize → compile → .hef
```

## Tests

```bash
# Tous les tests (rapides)
pytest -m "not slow"

# Tous les tests y compris intégration
pytest

# Par module
pytest tests/test_model.py         # Architecture, param groups
pytest tests/test_training.py      # Entraînement, régularisation, QAT
pytest tests/test_export.py        # ONNX, Hailo, IMX500
pytest tests/test_distill.py       # Knowledge distillation
pytest tests/test_dataset.py       # Chargement, split, transforms
pytest tests/test_auto_annotate.py # Annotation bbox
pytest tests/test_verify_boxes.py  # Vérification bbox
pytest tests/test_quality_filter.py # Filtrage CLIP
```

## Structure du projet

```
bird-detection/
├── train.py                   # Entraînement (4 architectures, MixUp/CutMix, EMA, QAT…)
├── distill.py                 # Knowledge Distillation (teacher → student)
├── export.py                  # Export ONNX multi-cible (onnx, hailo, imx500)
├── auto_annotate.py           # Annotation bbox (FasterRCNN / YOLO11)
├── verify_boxes.py            # Vérification visuelle des bbox + retry
├── requirements.txt           # Dépendances (hors PyTorch)
├── dataset/europe/            # Dataset (558 espèces, ~124 911 images)
│   ├── train/{espèce}/        # Images + annotations.json (bbox)
│   ├── validation/{espèce}/
│   ├── test/{espèce}/
│   ├── label_map.json         # slug → index (558 classes)
│   └── metadata.json          # Métadonnées par espèce
├── documentation/
│   ├── audit-training-recommendations.md  # Audit des hyperparamètres
│   └── deployment-model-zoo.md            # Modèles compatibles par plateforme
├── tests/                     # ~185 tests pytest
└── output/                    # Checkpoints, métriques, modèles exportés
```
