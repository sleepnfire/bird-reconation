# Infrastructure Hailo-10H — Pi 5 + AI HAT+ 2

Date : 2026-06-14

Guide opérationnel pour déployer le pipeline two-stage (détection + classification) de reconnaissance d'oiseaux sur Raspberry Pi 5 + AI HAT+ 2 (Hailo-10H).

Pour les justifications des choix de modèles, voir [validation-choix-ia.md](validation-choix-ia.md).

---

## 1. Spécifications matérielles

### 1a. Raspberry Pi 5

| Caractéristique | Valeur |
|---|---|
| SoC | Broadcom BCM2712 (Cortex-A76 quad-core 2.4 GHz) |
| RAM | 16 Go LPDDR4X |
| Connectique IA | PCIe Gen 2 ×1 (Gen 3 configurable via `config.txt`, non certifié) |
| OS | Raspberry Pi OS 64-bit (Bookworm+) |
| Alimentation | USB-C 5V/5A (27W recommandé avec HAT+) |

### 1b. AI HAT+ 2 (Hailo-10H)

| Caractéristique | Valeur |
|---|---|
| Puce | Hailo-10H |
| Performance | **40 TOPS** (INT4) |
| Mémoire dédiée | **8 Go LPDDR4X** (on-board, pour les modèles et activations) |
| Interface | PCIe Gen 3 |
| Multi-modèle | **Oui** — scheduling natif via HailoRT |
| Format modèle | `.hef` (Hailo Executable Format) |

### 1c. Installation physique

1. Éteindre le Pi 5 et débrancher l'alimentation
2. Connecter le HAT+ au connecteur PCIe du Pi 5 (nappe FPC fournie)
3. Fixer avec les entretoises
4. Utiliser une alimentation **27W** (USB-C 5V/5A) — le HAT+ consomme ~5-10W supplémentaires

**Sources** : [Raspberry Pi AI HAT+ 2](https://www.raspberrypi.com/products/ai-hat-plus-2/), [Documentation AI HATs](https://www.raspberrypi.com/documentation/accessories/ai-hat-plus.html)

---

## 2. Architecture du pipeline

### 2a. Pipeline two-stage

```
┌─────────┐    ┌──────────────────┐    ┌───────┐    ┌──────────────────────┐    ┌─────────┐
│ Caméra  │───▶│ YOLO11s (.hef)   │───▶│ Crop  │───▶│ ViT-B/16 (.hef)      │───▶│ Espèce  │
│ (CSI/   │    │ Détection 1 cl.  │    │ bbox  │    │ Classification 558   │    │ + score │
│  USB)   │    │ "bird"           │    │       │    │ espèces              │    │         │
└─────────┘    └──────────────────┘    └───────┘    └──────────────────────┘    └─────────┘
                    Hailo-10H                            Hailo-10H
                   (modèle 1)                           (modèle 2)
```

### 2b. Scheduling multi-modèle

Le **HailoRT** (runtime Hailo) gère le scheduling des modèles sur le NPU :

- Les deux modèles `.hef` sont chargés en mémoire (8 Go LPDDR4X disponibles)
- Le runtime exécute séquentiellement : détection → classification
- Pas de rechargement entre les inférences (contrairement à l'IMX500)
- Le CPU du Pi 5 orchestre le pipeline (crop, NMS, post-traitement)

### 2c. Latence estimée

| Étape | Latence estimée | Source |
|---|---|---|
| Capture frame | ~5 ms | picamera2 |
| YOLO11s (détection) | ~7 ms | [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO10H/HAILO10H_object_detection.rst) (142 FPS batch 1) |
| Crop + resize | ~1-2 ms | CPU (numpy/PIL) |
| ViT-B/16 (classification) | ~17-18 ms | [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO10H/HAILO10H_classification.rst) (57 FPS batch 1) |
| Post-traitement | ~1 ms | CPU (softmax, argmax) |
| **Total pipeline** | **~25-30 ms** | **~33-40 FPS** |

**Référence** : le projet [rpi5-birdcam-hailo-bioclip](https://github.com/ekstremedia/rpi5-birdcam-hailo-bioclip) atteint **22 FPS** avec YOLOv8 + BioCLIP sur Pi 5 + Hailo-8 (26 TOPS). Le Hailo-10H (40 TOPS) devrait être au moins aussi rapide.

---

## 3. Installation logicielle

### 3a. Sur le Raspberry Pi 5 (cible de déploiement)

```bash
# Mettre à jour l'OS
sudo apt update && sudo apt full-upgrade -y

# Installer le meta-paquet Hailo (runtime + firmware + outils)
sudo apt install hailo-all -y

# Redémarrer pour charger le firmware
sudo reboot

# Vérifier la détection du Hailo-10H
hailortcli fw-control identify
# Attendu : "Hailo-10H" avec firmware version

# Vérifier le device PCIe
lspci | grep Hailo
# Attendu : "Co-processor: Hailo Technologies Ltd."
```

**Paquets Python (sur le Pi 5)** :

```bash
pip install hailo-platform    # Bindings Python pour HailoRT
pip install picamera2         # Capture caméra (pré-installé sur Pi OS)
pip install numpy pillow      # Post-traitement
```

### 3b. Sur la machine de développement (compilation des modèles)

Le **Hailo Dataflow Compiler (DFC)** compile les modèles ONNX en `.hef`. Il tourne sur x86_64 (pas sur le Pi).

```bash
# Installer le DFC (nécessite une licence Hailo Developer Zone)
pip install hailo_dataflow_compiler

# Installer le Hailo Model Zoo (scripts et configurations)
pip install hailo_model_zoo

# Vérifier
hailo --version
```

**Prérequis DFC** :
- Linux x86_64 (Ubuntu 22.04/24.04 recommandé)
- Python 3.10-3.12
- Licence Hailo Developer Zone (gratuite pour les développeurs)

**Sources** : [Hailo Developer Zone](https://hailo.ai/developer-zone/), [Documentation AI HATs](https://www.raspberrypi.com/documentation/accessories/ai-hat-plus.html)

---

## 4. Export PyTorch → .hef

### 4a. Export du classificateur ViT-B/16

**Étape 1 — Export ONNX** (sur la machine de dev, dans le projet bird-detection) :

```bash
python export.py --checkpoint output/best_vit_b_16.pth --target hailo
```

Produit : `output/vit_b_16_hailo.onnx` (ONNX float32, opset 13+, input fixe `[1, 3, 224, 224]`)

**Étape 2 — Compilation via Hailo DFC** :

```bash
# 1. Parser le modèle ONNX
hailo parser onnx output/vit_b_16_hailo.onnx

# 2. Optimiser (quantification INT4/INT8 automatique)
hailo optimize --hw-arch hailo10h \
    --calib-set-path calibration_images/ \
    vit_b_16_hailo.har

# 3. Compiler en .hef
hailo compile vit_b_16_hailo_quantized.har \
    --hw-arch hailo10h \
    -o vit_b_16.hef
```

Le DFC gère la quantification automatiquement. Un **dataset de calibration** (100-500 images représentatives) est nécessaire pour `hailo optimize`.

### 4b. Export du détecteur YOLO11s

**Option 1 — Via Ultralytics** (recommandé) :

```bash
# Exporter en ONNX
yolo export model=yolo11s_bird.pt format=onnx opset=13

# Puis compiler via Hailo DFC
hailo parser onnx yolo11s_bird.onnx
hailo optimize --hw-arch hailo10h yolo11s_bird.har
hailo compile yolo11s_bird_quantized.har --hw-arch hailo10h -o yolo11s_bird.hef
```

**Option 2 — Modèle pré-compilé du Hailo Model Zoo** :

YOLO11s est disponible dans le [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO10H/HAILO10H_object_detection.rst) avec des `.hef` pré-compilés pour COCO (80 classes). Pour notre détecteur 1 classe "bird", un fine-tuning et une recompilation sont nécessaires.

### 4c. Vérification des .hef

```bash
# Vérifier les métadonnées du modèle
hailortcli parse-hef vit_b_16.hef

# Benchmark de performance (sur le Pi 5)
hailortcli benchmark vit_b_16.hef

# Test d'inférence simple
hailortcli run vit_b_16.hef --input test_image.bin
```

**Sources** : [Hailo DFC User Guide v3.27.0](https://mmmsk.ai.kr/Projects/Embedded-AI/files/hailo_dataflow_compiler_v3.27.0_user_guide.pdf) (miroir tiers — source officielle : [Hailo Developer Zone](https://hailo.ai/developer-zone/)), [Hailo Model Zoo — Getting Started](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/GETTING_STARTED.rst)

---

## 5. Pipeline d'inférence temps réel

### 5a. Architecture du code

```python
import numpy as np
from PIL import Image
from hailo_platform import HEF, VDevice, ConfigureParams, InferVStreams, InputVStreamParams, OutputVStreamParams
from picamera2 import Picamera2

# --- Chargement des modèles ---
detector_hef = HEF("yolo11s_bird.hef")
classifier_hef = HEF("vit_b_16.hef")

# --- Chargement du label map ---
import json
with open("dataset/europe/label_map.json") as f:
    label_map = json.load(f)
index_to_species = {v: k for k, v in label_map.items()}

# --- Chargement des métadonnées (noms français) ---
with open("dataset/europe/metadata.json") as f:
    metadata = json.load(f)

# --- Initialisation caméra ---
camera = Picamera2()
camera.configure(camera.create_preview_configuration(
    main={"size": (1920, 1080), "format": "RGB888"}
))
camera.start()

# --- Boucle d'inférence ---
with VDevice() as vdevice:
    # Configurer les deux modèles sur le device
    det_net = vdevice.configure(detector_hef)
    cls_net = vdevice.configure(classifier_hef)

    while True:
        frame = camera.capture_array()

        # --- Étape 1 : Détection YOLO ---
        det_input = preprocess_yolo(frame)  # resize 640×640, normalize
        det_output = det_net.infer(det_input)
        boxes = postprocess_yolo(det_output, conf_threshold=0.5)

        # --- Étape 2 : Classification par bbox ---
        for box in boxes:
            x1, y1, x2, y2, conf = box
            crop = frame[y1:y2, x1:x2]
            crop_resized = preprocess_vit(crop)  # resize 224×224, normalize

            cls_output = cls_net.infer(crop_resized)
            probs = softmax(cls_output)
            top5 = np.argsort(probs)[-5:][::-1]

            for idx in top5:
                species = index_to_species[idx]
                name_fr = metadata[species]["french_name"]
                print(f"  {name_fr}: {probs[idx]*100:.1f}%")
```

### 5b. Fonctions utilitaires

```python
def preprocess_yolo(frame, size=640):
    """Prépare l'image pour YOLO11s."""
    img = Image.fromarray(frame).resize((size, size))
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)[np.newaxis]  # [1, 3, 640, 640]

def preprocess_vit(crop, size=224):
    """Prépare le crop pour ViT-B/16 (normalisation ImageNet)."""
    img = Image.fromarray(crop).resize((size, size))
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std
    return arr.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

def postprocess_yolo(output, conf_threshold=0.5):
    """NMS et filtrage des détections YOLO."""
    # Extraire les boxes avec confidence > seuil
    # Appliquer Non-Maximum Suppression (IoU > 0.45)
    # Retourner [(x1, y1, x2, y2, conf), ...]
    ...

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()
```

### 5c. Référence open-source

Le projet [rpi5-birdcam-hailo-bioclip](https://github.com/ekstremedia/rpi5-birdcam-hailo-bioclip) implémente un pipeline similaire (YOLOv8 + BioCLIP) sur Pi 5 + Hailo pour 42 espèces norvégiennes à 22 FPS. Son architecture de code est une référence directe pour notre implémentation.

---

## 6. Performance attendue

### 6a. Latence et FPS

| Scénario | Modèles | Latence | FPS |
|---|---|---|---|
| Classification seule | ViT-B/16 | ~17-18 ms | ~55-59 |
| Détection seule | YOLO11s | ~7 ms | ~142 |
| **Pipeline two-stage** | **YOLO11s + ViT-B/16** | **~25-30 ms** | **~33-40** |
| Avec post-traitement | Pipeline complet | ~30-35 ms | ~28-33 |

### 6b. Accuracy

| Modèle | Accuracy ImageNet (Hailo) | Accuracy projet (558 esp.) | Source |
|---|---|---|---|
| ViT-B/16 | 84.5 % (float) / 83.6 % (quantifié Hailo-10H) | 79.52 % (mesuré, run 3) | [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO10H/HAILO10H_classification.rst), analyse-entrainement-3-vit.md |
| YOLO11s (1 classe "bird") | mAP > 90 % (estimé) | À mesurer | — |
| Pipeline end-to-end | — | Estimé ~75-79 % | détection × classification |

### 6c. Mémoire

| Modèle | Taille estimée (.hef) | Mémoire disponible | Marge |
|---|---|---|---|
| YOLO11s | ~20-40 Mo | 8 Go LPDDR4X | Largement suffisant |
| ViT-B/16 | ~200-400 Mo | 8 Go LPDDR4X | Largement suffisant |
| **Total** | **~250-450 Mo** | **8 Go** | **> 7 Go de marge** |

### 6d. Comparaison avec des projets similaires

| Projet | Hardware | Modèles | Espèces | FPS | Source |
|---|---|---|---|---|---|
| rpi5-birdcam-hailo-bioclip | Pi 5 + Hailo-8 (26 TOPS) | YOLOv8 + BioCLIP | 42 | **22** | [GitHub](https://github.com/ekstremedia/rpi5-birdcam-hailo-bioclip) |
| **Notre projet (estimé)** | **Pi 5 + Hailo-10H (40 TOPS)** | **YOLO11s + ViT-B/16** | **558** | **~28-33** | — |

Le Hailo-10H a **+54 %** de TOPS par rapport au Hailo-8, ce qui devrait compenser la taille supérieure de ViT-B/16 par rapport à BioCLIP.

---

## 7. État d'avancement et TODO

| Composant | Statut | Détail |
|---|---|---|
| Entraînement ViT-B/16 | ✅ Terminé | 79.52 % test accuracy, 558 espèces |
| Export ONNX `--target hailo` | ✅ Terminé | `export.py`, ONNX float32 opset 13+ |
| Knowledge distillation | ✅ Terminé | `distill.py`, ViT → MobileNetV2 |
| Entraînement YOLO11s (1 classe "bird") | ❌ À faire | 195 726 images annotées bbox disponibles |
| Compilation .hef (DFC) | ❌ À faire | Nécessite Hailo DFC + licence |
| Pipeline d'inférence Pi 5 | ❌ À faire | Script Python avec HailoRT |
| Tests end-to-end | ❌ À faire | mAP détection + accuracy classification |
| Benchmark FPS/latence réel | ❌ À faire | Sur Pi 5 + HAT+ physique |

### Dépendances bloquantes

| Dépendance | Type | Statut |
|---|---|---|
| Hailo Dataflow Compiler (DFC) | Logiciel (licence gratuite) | À installer |
| Pi 5 + AI HAT+ 2 | Matériel | À acquérir |
| Dataset bbox pour YOLO | Données | ✅ 195 726 images annotées |

### Prochaines étapes recommandées

1. **Entraîner YOLO11s** sur les 195 726 images avec bbox (1 classe "bird")
2. **Installer le Hailo DFC** et compiler les deux modèles en .hef
3. **Implémenter le pipeline d'inférence** (basé sur le pseudo-code section 5)
4. **Benchmarker** sur le matériel réel

---

## 8. Références

### Documentation officielle

- [Raspberry Pi AI HAT+ 2 — Product Page](https://www.raspberrypi.com/products/ai-hat-plus-2/)
- [Documentation AI HATs — Raspberry Pi](https://www.raspberrypi.com/documentation/accessories/ai-hat-plus.html)
- [Hailo-10H — On-Device GenAI](https://hailo.ai/blog/bringing-on-device-generative-ai-to-the-pi-when-and-why-youll-need-the-raspberry-pi-ai-hat-2/)
- [Hailo Developer Zone](https://hailo.ai/developer-zone/)

### Model Zoo et compilation

- [Hailo Model Zoo — Classification Hailo-10H](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO10H/HAILO10H_classification.rst)
- [Hailo Model Zoo — Object Detection Hailo-10H](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO10H/HAILO10H_object_detection.rst)
- [Hailo Model Zoo — Getting Started](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/GETTING_STARTED.rst)
- [Hailo DFC User Guide v3.27.0](https://mmmsk.ai.kr/Projects/Embedded-AI/files/hailo_dataflow_compiler_v3.27.0_user_guide.pdf) (miroir tiers — source officielle : [Hailo Developer Zone](https://hailo.ai/developer-zone/))

### Projets de référence

- [rpi5-birdcam-hailo-bioclip](https://github.com/ekstremedia/rpi5-birdcam-hailo-bioclip) — Pi 5 + Hailo, 42 espèces, 22 FPS
- [Hailo Object Detection Demo](https://hailo.ai/resources/industries/personal-compute/object-detection-demo-with-hailo-10h-ai-accelerator/) — YOLO11m temps réel 4K

### Documentation interne

- [validation-choix-ia.md](validation-choix-ia.md) — Justification des choix de modèles
- [comparaison-detection-approaches.md](comparaison-detection-approaches.md) — Analyse two-stage vs all-in-one
- [analyse-entrainement-3-vit.md](analyse-entrainement-3-vit.md) — Résultats ViT-B/16 run 3
- [deployment-model-zoo.md](deployment-model-zoo.md) — Modèles déployables par plateforme
