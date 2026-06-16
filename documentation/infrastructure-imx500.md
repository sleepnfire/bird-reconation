# Infrastructure IMX500 — Pi Zero 2 W + AI Camera

Date : 2026-06-14

Guide opérationnel pour déployer un modèle de reconnaissance d'oiseaux sur Raspberry Pi Zero 2 W + AI Camera (Sony IMX500).

Pour les justifications des choix de modèles, voir [validation-choix-ia.md](validation-choix-ia.md).

---

## 1. Spécifications matérielles

### 1a. Raspberry Pi Zero 2 W

| Caractéristique | Valeur |
|---|---|
| SoC | Broadcom BCM2710A1 (Cortex-A53 quad-core 1.0 GHz) |
| RAM | **512 Mo** LPDDR2 |
| Connectique caméra | CSI-2 (nappe 22 broches vers 15 broches) |
| Connectivité | Wi-Fi 802.11 b/g/n, Bluetooth 4.2 |
| OS | Raspberry Pi OS Lite 64-bit (Bookworm+) |
| Alimentation | Micro-USB 5V/2.5A |

### 1b. AI Camera (Sony IMX500)

| Caractéristique | Valeur |
|---|---|
| Capteur | Sony IMX500 (12.3 Mpx, 1/2.3") |
| NPU intégré | DSP dédié, inference **on-sensor** |
| Mémoire modèle | **8 Mo SRAM** |
| Multi-modèle | **Non** — un seul modèle chargé à la fois |
| Rechargement modèle | Plusieurs secondes (écriture en SRAM) |
| Latence inférence | ~33 ms (MobileNet SSD), ~58 ms (YOLO11n) |
| FPS | ~17-30 selon le modèle |
| Format modèle | `.rpk` (firmware Sony) |

### 1c. Architecture on-sensor

L'IMX500 est fondamentalement différent du Hailo-10H : l'inférence se fait **dans le capteur lui-même**, pas sur un accélérateur externe.

```
┌─────────────────────────────────────────┐
│            Sony IMX500 (capteur)         │
│                                         │
│  Photodiodes → ISP → NPU → Tenseurs    │
│                       ▲                 │
│                  .rpk (8 Mo SRAM)       │
└────────────────────┬────────────────────┘
                     │ CSI-2 (tenseurs de sortie)
                     ▼
              ┌──────────────┐
              │ Pi Zero 2 W  │
              │ Post-traitement│
              │ (softmax, argmax)│
              └──────────────┘
```

Le Pi Zero ne reçoit **que les tenseurs de sortie** (pas les pixels bruts pendant l'inférence), ce qui minimise la charge CPU et la bande passante.

### 1d. Installation physique

1. Éteindre le Pi Zero et débrancher l'alimentation
2. Connecter l'AI Camera au port CSI via la **nappe adaptateur 22→15 broches** (incluse avec le Pi Zero)
3. Fixer la caméra (vis M2 ou boîtier imprimé 3D)

**Sources** : [Raspberry Pi AI Camera](https://www.raspberrypi.com/products/ai-camera/), [Documentation AI Camera](https://www.raspberrypi.com/documentation/accessories/ai-camera.html)

---

## 2. Architecture du pipeline

### 2a. Contrainte mono-modèle

Le capteur IMX500 ne charge qu'**un seul modèle** en SRAM à la fois. Le rechargement prend plusieurs secondes, excluant tout pipeline two-stage en temps réel. Cela dicte trois stratégies de déploiement (voir section 5).

### 2b. Pipeline principal : classification pure

Pour une caméra fixe (mangeoire, nichoir), la stratégie recommandée est la **classification pure** :

```
┌─────────────────────────────────────┐
│         Sony IMX500                  │
│                                      │
│  Image → [EfficientNetV2-B2 .rpk]  │
│           ou [MobileNetV2 .rpk]     │
│           558 classes                │
│                 │                    │
│          Tenseur (558 logits)        │
└─────────┬───────────────────────────┘
          │ CSI-2
          ▼
┌─────────────────────┐    ┌──────────┐
│ Pi Zero 2 W         │    │ Résultat │
│ softmax → argmax    │───▶│ Espèce   │
│ label_map.json      │    │ + score  │
└─────────────────────┘    └──────────┘
```

### 2c. Latence estimée

| Étape | Latence estimée | Source |
|---|---|---|
| Capture + inférence on-sensor | ~33-58 ms | IMX500 Model Zoo |
| Transfert tenseur (CSI-2) | ~1 ms | Négligeable |
| Post-traitement (softmax, argmax) | ~1-2 ms | CPU Pi Zero |
| **Total** | **~35-60 ms** | **~17-28 FPS** |

**Référence** : le Model Zoo IMX500 annonce ~33 ms pour MobileNet SSD et ~58.82 ms pour YOLO11n.
Source : [Ultralytics IMX500 Integration](https://docs.ultralytics.com/integrations/sony-imx500)

---

## 3. Installation logicielle

### 3a. Sur le Raspberry Pi Zero 2 W (cible de déploiement)

```bash
# Mettre à jour l'OS
sudo apt update && sudo apt full-upgrade -y

# Installer le meta-paquet IMX500 (firmware + outils)
sudo apt install imx500-all -y

# Redémarrer
sudo reboot

# Vérifier la détection de la caméra
rpicam-hello -t 5s --post-process-file /usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json
```

**Paquets Python (sur le Pi Zero)** :

```bash
pip install picamera2    # Pré-installé sur Pi OS
pip install numpy        # Post-traitement
```

### 3b. Sur la machine de développement (conversion des modèles)

La conversion PyTorch → `.rpk` nécessite un **venv dédié** (Python 3.12) car le Sony MCT exige `matplotlib<3.10`, incompatible avec Python 3.14.

```bash
# Créer le venv IMX500 automatiquement
bash setup-imx500-venv.sh
```

Ce script :
1. Cherche Python 3.12 ou 3.11 (`python3.12` / `python3.11`)
2. Crée `.venv-imx500/` avec PyTorch + dépendances MCT
3. Installe les paquets depuis `requirements-imx500.txt`

**Dépendances IMX500** (`requirements-imx500.txt`) :

```
pillow>=11.0,<13.0
numpy>=2.0
scikit-learn>=1.5
matplotlib>=3.9,<3.10
tqdm>=4.60
timm>=1.0
onnx>=1.15
onnxruntime>=1.18
model-compression-toolkit>=2.0
```

**Outils supplémentaires pour la conversion finale** :

```bash
# IMX500 Converter (ONNX INT8 → .rpk)
# Disponible via le SDK Sony AITRIOS
pip install imx500-converter
```

**Sources** : [Documentation AI Camera](https://www.raspberrypi.com/documentation/accessories/ai-camera.html), [Sony MCT (GitHub)](https://github.com/sony/model_optimization), [SDK AITRIOS](https://developer.aitrios.sony-semicon.com/en/raspberrypi-ai-camera/documentation/imx500-converter)

---

## 4. Export PyTorch → .rpk

### 4a. Vue d'ensemble du pipeline de conversion

```
PyTorch (.pth)
     │
     ▼
  export.py --target imx500
  (Sony MCT, quantification INT8)
     │
     ▼
  ONNX INT8 (.onnx)
     │
     ▼
  IMX500 Converter (imx500-convert)
     │
     ▼
  Firmware .rpk (≤ 8 Mo)
```

### 4b. Étape 1 — Quantification INT8 via Sony MCT

Le fichier `export.py` gère la quantification statique INT8 via le Model Compression Toolkit de Sony.

```bash
# EfficientNetV2-B2 (modèle principal IMX500)
.venv-imx500/bin/python export.py \
    --checkpoint output/best_efficientnetv2_b2.pth \
    --target imx500 \
    --dataset dataset/europe \
    --calibration-images 200

# MobileNetV2 (baseline cross-platform)
.venv-imx500/bin/python export.py \
    --checkpoint output/best_mobilenetv2.pth \
    --target imx500 \
    --dataset dataset/europe \
    --calibration-images 200

# MobileNetV2 distillé (knowledge distillation ViT → MobileNetV2)
.venv-imx500/bin/python export.py \
    --checkpoint output/best_mobilenetv2_distilled.pth \
    --target imx500 \
    --dataset dataset/europe \
    --calibration-images 200
```

Produit : `output/{arch}_imx500.onnx` (ONNX quantifié INT8, opset 13+)

**Ce que fait `export.py --target imx500`** :

1. Charge le checkpoint PyTorch
2. Crée un `DataLoader` de calibration (200 images du dataset `train/`, seed 42)
3. Applique la quantification statique INT8 via `mct.ptq.pytorch_post_training_quantization()`
4. Utilise le TPC (Target Platform Capabilities) `imx500` version `6.0`
5. Exporte en ONNX (opset 13)
6. Vérifie la taille (≤ 8 Mo si `--check-size`)

### 4c. QAT (Quantization-Aware Training) — recommandé

La QAT réduit significativement la perte de précision due à la quantification INT8 (< 1 % de perte vs plusieurs points sans QAT) ([Jacob et al., CVPR 2018](https://arxiv.org/abs/1712.05877)).

```bash
# Entraîner avec QAT (5 époques supplémentaires)
python train.py \
    --model mobilenetv2 \
    --qat \
    --qat-epochs 5 \
    --dataset dataset/europe

# Puis exporter le modèle QAT
.venv-imx500/bin/python export.py \
    --checkpoint output/best_mobilenetv2_qat.pth \
    --target imx500 \
    --dataset dataset/europe
```

### 4d. Étape 2 — Conversion ONNX INT8 → .rpk

```bash
# Convertir en firmware Sony .rpk
imx500-convert -i output/mobilenetv2_imx500.onnx -o output/mobilenetv2.rpk

# Vérifier la taille
ls -lh output/mobilenetv2.rpk
# Doit être ≤ 8 Mo
```

### 4e. Tailles de référence (Model Zoo IMX500)

| Modèle | Top-1 ImageNet (INT8) | Taille .rpk | Compatible 8 Mo |
|---|---|---|---|
| **EfficientNetV2-B2** | **77.7 %** | **6.51 Mo** | ✅ |
| EfficientNetV2-B1 | 77.0 % | 6.37 Mo | ✅ |
| EfficientNetV2-B0 | 76.7 % | 6.52 Mo | ✅ |
| EfficientNet-B0 | 72.1 % | 5.99 Mo | ✅ |
| **MobileNetV2** | **71.6 %** | **3.89 Mo** | ✅ |
| MNASNet 1.0 | 73.2 % | 4.84 Mo | ✅ |
| ShuffleNetV2 x1.5 | 72.2 % | 3.89 Mo | ✅ |
| SqueezeNet 1.0 | 57.6 % | 1.52 Mo | ✅ |

**Source** : [IMX500 Model Zoo](https://github.com/raspberrypi/imx500-models)

### 4f. Récapitulatif des commandes

```bash
# 1. Créer le venv (une seule fois)
bash setup-imx500-venv.sh

# 2. Quantifier via Sony MCT
.venv-imx500/bin/python export.py --checkpoint output/best_mobilenetv2.pth --target imx500

# 3. Convertir en .rpk
imx500-convert -i output/mobilenetv2_imx500.onnx -o output/mobilenetv2.rpk

# 4. Copier sur le Pi Zero
scp output/mobilenetv2.rpk pi@pizero.local:~/models/

# 5. Tester sur le Pi Zero
rpicam-hello -t 10s --post-process-file mobilenetv2_config.json
```

**Sources** : [IMX500 Converter](https://developer.aitrios.sony-semicon.com/en/raspberrypi-ai-camera/documentation/imx500-converter), [Sony MCT GitHub](https://github.com/sony/model_optimization)

---

## 5. Stratégies de déploiement

La contrainte mono-modèle de l'IMX500 impose de choisir **une seule** stratégie. Trois options sont évaluées.

### 5a. Option A — Classification pure (recommandée pour caméra fixe)

```
IMX500 [EfficientNetV2-B2 ou MobileNetV2, 558 classes] → Pi Zero (softmax → espèce)
```

| Critère | Valeur |
|---|---|
| Modèle | EfficientNetV2-B2 (6.51 Mo) ou MobileNetV2 (3.89 Mo) |
| Hypothèse | Caméra fixe, champ cadré sur mangeoire/nichoir |
| Détection | Aucune — l'oiseau est supposé présent dans le champ |
| FPS | ~17-28 |
| Accuracy estimée | ~70-77 % (ImageNet baseline, à mesurer sur 558 espèces) |
| Avantage | Simplicité, latence minimale, exploit maximal du NPU |
| Inconvénient | Pas de localisation, sensible aux images sans oiseau |

**Cas d'usage** : mangeoire filmée en continu, l'oiseau occupe une grande partie du champ.

### 5b. Option B — YOLO11n 558 classes (tout-en-un)

```
IMX500 [YOLO11n 558 classes, ~3-4 Mo] → Pi Zero (NMS → espèce + bbox)
```

| Critère | Valeur |
|---|---|
| Modèle | YOLO11n fine-tuné 558 classes |
| Détection + classification | Oui, en une seule passe |
| FPS | ~17 (58 ms/image) |
| Accuracy estimée | **30-50 %** (nano + 558 classes fine-grained) |
| Avantage | Localisation + identification en une passe |
| Inconvénient | Précision espèce très faible (YOLO nano sur 558 classes) |

**Attention** : Ultralytics précise que l'export IMX500 est conçu et benchmarké uniquement pour YOLOv8n et YOLO11n (nano). Les autres tailles ne sont pas supportées.

### 5c. Option C — YOLO11n détecteur + MobileNetV2 CPU

```
IMX500 [YOLO11n 1 classe "bird", ~2 Mo] → bbox → crop → Pi Zero CPU [MobileNetV2] → espèce
```

| Critère | Valeur |
|---|---|
| Détecteur | YOLO11n 1 classe (on-sensor, ~58 ms) |
| Classificateur | MobileNetV2 (CPU Pi Zero, ~500 ms - 1 s) |
| FPS effectif | **~1-2 FPS** |
| Accuracy estimée | ~62 % (MobileNetV2 mesuré, run 3) |
| Avantage | Meilleure précision que l'option B |
| Inconvénient | Latence CPU très élevée (Cortex-A53 1.0 GHz) |

### 5d. Tableau comparatif

| Critère | Option A (classif. pure) | Option B (YOLO 558) | Option C (YOLO + CPU) |
|---|---|---|---|
| **FPS** | **~17-28** | ~17 | ~1-2 |
| **Accuracy espèce** | ~70-77 % | ~30-50 % | ~62 % |
| **Détection bbox** | Non | Oui | Oui |
| **Complexité** | Simple | Simple | Moyenne |
| **Taille .rpk** | 3.89-6.51 Mo | ~3-4 Mo | ~2 Mo |
| **Cas d'usage** | Caméra fixe | Champ large | Champ large |

### 5e. Recommandation

| Scénario | Option recommandée |
|---|---|
| **Mangeoire / nichoir (caméra fixe)** | **Option A** — classification pure (EfficientNetV2-B2) |
| Champ large, besoin de localiser | Option C — YOLO détecteur + MobileNetV2 CPU |
| Prototype rapide, précision secondaire | Option B — YOLO 558 classes |

L'**option A** est recommandée pour le cas d'usage principal du projet (oiseaux de jardin sur mangeoire). L'option C est l'alternative si la localisation est nécessaire.

---

## 6. Pipeline d'inférence

### 6a. Classification pure (Option A)

```python
import json
import numpy as np
from picamera2 import Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics

# --- Chargement du modèle .rpk ---
imx500 = IMX500("models/mobilenetv2.rpk")
intrinsics = imx500.network_intrinsics or NetworkIntrinsics()
intrinsics.task = "classification"

# --- Chargement du label map ---
with open("dataset/europe/label_map.json") as f:
    label_map = json.load(f)
index_to_species = {v: k for k, v in label_map.items()}

# --- Chargement des métadonnées (noms français) ---
with open("dataset/europe/metadata.json") as f:
    metadata = json.load(f)

# --- Initialisation caméra ---
camera = Picamera2(imx500.camera_num)
config = camera.create_preview_configuration(
    controls={"FrameRate": intrinsics.inference_rate}
)
camera.configure(config)
camera.start()

# --- Boucle d'inférence ---
while True:
    # L'inférence se fait on-sensor, on récupère les tenseurs
    metadata_frame = camera.capture_metadata()
    output_tensor = imx500.get_output(metadata_frame)

    if output_tensor is not None:
        # Softmax sur les 558 logits
        logits = output_tensor[0]
        probs = softmax(logits)

        # Top 5 prédictions
        top5_idx = np.argsort(probs)[-5:][::-1]
        for idx in top5_idx:
            species = index_to_species[idx]
            meta = metadata[species]
            name_fr = meta["french_name"]
            confidence = probs[idx] * 100
            print(f"  {name_fr} ({meta['scientific_name']}): {confidence:.1f}%")

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()
```

### 6b. Filtrage des espèces de jardin

Pour affiner les résultats aux ~20 espèces de jardin (cas d'usage principal) :

```python
GARDEN_SPECIES = {
    "parus_major", "cyanistes_caeruleus", "erithacus_rubecula",
    "turdus_merula", "passer_domesticus", "fringilla_coelebs",
    "carduelis_carduelis", "chloris_chloris", "aegithalos_caudatus",
    "sitta_europaea", "certhia_brachydactyla", "dendrocopos_major",
    "columba_palumbus", "streptopelia_decaocto", "sturnus_vulgaris",
    "garrulus_glandarius", "pica_pica", "phoenicurus_ochruros",
    "troglodytes_troglodytes", "regulus_regulus",
}

def filter_garden(probs, index_to_species, top_k=5):
    """Filtre les prédictions aux espèces de jardin."""
    garden_indices = [i for i, sp in index_to_species.items() if sp in GARDEN_SPECIES]
    garden_probs = {i: probs[i] for i in garden_indices}
    top = sorted(garden_probs.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(index_to_species[i], p) for i, p in top]
```

### 6c. Configuration JSON pour rpicam-apps

Pour tester avec les outils natifs Raspberry Pi :

```json
{
    "imx500_classification": {
        "network_file": "/home/pi/models/mobilenetv2.rpk",
        "labels_file": "/home/pi/models/label_map.txt",
        "threshold": 0.3,
        "top_k": 5
    }
}
```

---

## 7. Performance attendue

### 7a. Latence et FPS

| Modèle | Latence on-sensor | FPS | Source |
|---|---|---|---|
| MobileNetV2 (.rpk) | ~33-40 ms | ~25-30 | Model Zoo (MobileNet SSD) |
| EfficientNetV2-B2 (.rpk) | ~40-55 ms | ~18-25 | Estimé (modèle plus large) |
| YOLO11n (.rpk) | ~58 ms | ~17 | Ultralytics benchmarks |

### 7b. Accuracy

| Modèle | Top-1 ImageNet (INT8) | Top-1 projet (558 esp.) estimé | Source |
|---|---|---|---|
| EfficientNetV2-B2 | 77.7 % | ~65-70 % | IMX500 Model Zoo |
| MobileNetV2 | 71.6 % | ~55-62 % | IMX500 Model Zoo + run 3 |
| MobileNetV2 (distillé) | ~72-73 % | ~58-64 % | Estimé (gain KD ~1-3 pts) |

### 7c. Consommation énergétique

| Composant | Consommation | Source |
|---|---|---|
| Pi Zero 2 W | ~0.5-1 W (idle), ~2 W (charge) | Estimé ([specs Pi Zero 2 W](https://www.raspberrypi.com/products/raspberry-pi-zero-2-w/)) |
| AI Camera (IMX500) | ~1-1.5 W (inférence) | Estimé (datasheet Sony non publique) |
| **Total** | **~2-3.5 W** | — |

Fonctionnement possible sur batterie ou panneau solaire.

### 7d. Comparaison avec Pi 5 + Hailo-10H

| | Pi Zero + IMX500 | Pi 5 + Hailo-10H |
|---|---|---|
| **Puissance IA** | <1 TOPS | 40 TOPS |
| **Modèle** | MobileNetV2 / EffNetV2-B2 | ViT-B/16 |
| **Accuracy** (558 esp.) | ~55-70 % | 79.52 % |
| **FPS** | ~17-28 | ~17-25 |
| **Détection** | Non (classif. pure) | Oui (two-stage) |
| **Consommation** | ~2-3.5 W | ~12-15 W |
| **Prix** | ~50 € | ~150 € |
| **Cas d'usage** | Mangeoire, monitoring léger | Observation complète |

---

## 8. État d'avancement et TODO

| Composant | Statut | Détail |
|---|---|---|
| Entraînement MobileNetV2 | ✅ Terminé | 61.98 % test accuracy, 558 espèces |
| Entraînement EfficientNet-B0 | ✅ Terminé | 558 espèces |
| Knowledge distillation ViT → MobileNetV2 | ✅ Terminé | `distill.py`, T=4.0, alpha=0.7 |
| Export `--target imx500` | ✅ Terminé | `export.py`, Sony MCT v2.0+ |
| Venv IMX500 | ✅ Terminé | `setup-imx500-venv.sh`, Python 3.12 |
| Tests export IMX500 | ✅ Terminé | `tests/test_export.py::TestExportIMX500` |
| Entraînement EfficientNetV2-B2 | ❌ À faire | Via `timm` (pas dans torchvision) |
| QAT MobileNetV2 | ❌ À faire | `train.py --model mobilenetv2 --qat` |
| Conversion ONNX → .rpk | ❌ À faire | `imx500-convert` (SDK AITRIOS) |
| Pipeline d'inférence Pi Zero | ❌ À faire | Script Python avec picamera2 |
| Tests end-to-end | ❌ À faire | Sur Pi Zero + AI Camera physique |
| Benchmark FPS/latence réel | ❌ À faire | Sur Pi Zero physique |

### Dépendances bloquantes

| Dépendance | Type | Statut |
|---|---|---|
| IMX500 Converter (SDK AITRIOS) | Logiciel (gratuit) | À installer |
| Pi Zero 2 W + AI Camera | Matériel | À acquérir |
| EfficientNetV2-B2 via timm | Code | À intégrer dans `train.py` |

### Prochaines étapes recommandées

1. **Intégrer EfficientNetV2-B2** dans `train.py` via `timm` (non disponible dans torchvision)
2. **Entraîner EfficientNetV2-B2** sur le dataset europe (558 espèces)
3. **Entraîner MobileNetV2 avec QAT** (`--qat --qat-epochs 5`)
4. **Installer le SDK AITRIOS** et convertir les modèles en `.rpk`
5. **Implémenter le pipeline d'inférence** (basé sur le pseudo-code section 6)
6. **Benchmarker** sur le matériel réel

---

## 9. Références

### Documentation officielle

- [Raspberry Pi AI Camera — Product Page](https://www.raspberrypi.com/products/ai-camera/)
- [Documentation AI Camera — Raspberry Pi](https://www.raspberrypi.com/documentation/accessories/ai-camera.html)
- [Sony IMX500 — AITRIOS Developer Portal](https://www.aitrios.sony-semicon.com/edge-ai-devices/raspberry-pi-ai-camera)

### Conversion et quantification

- [IMX500 Converter](https://developer.aitrios.sony-semicon.com/en/raspberrypi-ai-camera/documentation/imx500-converter) — ONNX → .rpk
- [Sony Model Compression Toolkit (MCT) — GitHub](https://github.com/sony/model_optimization) — Quantification INT8
- [IMX500 Model Zoo — GitHub](https://github.com/raspberrypi/imx500-models) — 15 modèles pré-convertis .rpk

### Intégrations

- [Ultralytics — Sony IMX500 Integration](https://docs.ultralytics.com/integrations/sony-imx500) — YOLO11n export, 58.82 ms
- [Sony IMX500 + YOLO — Ultralytics Blog](https://www.ultralytics.com/blog/empowering-edge-ai-with-sony-imx500-and-aitrios)

### Recherche

- [Jacob et al., CVPR 2018](https://arxiv.org/abs/1712.05877) — Quantization-Aware Training, perte INT8 < 1 % vs baseline float
- [Tan & Le, ICML 2021](https://arxiv.org/abs/2104.00298) — EfficientNetV2
- [Tam & Kay, arXiv 2407.00018](https://arxiv.org/abs/2407.00018) — Comparaison détection fine-grained vs coarse-grained en écologie (faune australienne, 14 classes YOLOv8)

### Documentation interne

- [validation-choix-ia.md](validation-choix-ia.md) — Justification des choix de modèles
- [comparaison-detection-approaches.md](comparaison-detection-approaches.md) — Analyse des stratégies IMX500
- [deployment-model-zoo.md](deployment-model-zoo.md) — 15 modèles IMX500 disponibles
