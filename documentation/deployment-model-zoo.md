# Recherche des modèles déployables — IMX500 & Hailo-10H

Date de recherche : 2026-06-12

## Objectif

Identifier les modèles de classification d'images **réellement déployables** sur les deux plateformes cibles du projet, en croisant les Model Zoos officiels avec la disponibilité PyTorch (torchvision / timm) pour l'entraînement.

---

## 1. Sony IMX500 (Pi Zero 2 + AI Camera)

### Contraintes

- **8 Mo de SRAM** pour le modèle (après quantification INT8)
- Conversion via **IMX500 Converter** (ONNX → firmware .rpk Sony)
- ~30 FPS max

### Modèles de classification disponibles (Model Zoo officiel)

15 modèles pré-convertis au format `.rpk` :

| Modèle | Format | Entraînable (torchvision) |
|--------|--------|:-------------------------:|
| mobilenet_v2 | .rpk | ✅ `models.mobilenet_v2` |
| efficientnet_bo | .rpk | ✅ `models.efficientnet_b0` |
| efficientnet_lite0 | .rpk | ❌ (timm uniquement) |
| efficientnetv2_b0 | .rpk | ⚠️ timm (`tf_efficientnetv2_b0`), pas d'équivalent torchvision exact |
| efficientnetv2_b1 | .rpk | ✅ |
| **efficientnetv2_b2** | .rpk | ✅ |
| mnasnet1.0 | .rpk | ✅ `models.mnasnet1_0` |
| mobilevit_xs | .rpk | ❌ (timm uniquement) |
| mobilevit_xxs | .rpk | ❌ (timm uniquement) |
| regnetx_002 | .rpk | ⚠️ pas d'équivalent exact (plus proche : `models.regnet_x_400mf`) |
| regnety_002 | .rpk | ⚠️ pas d'équivalent exact (plus proche : `models.regnet_y_400mf`) |
| regnety_004 | .rpk | ✅ `models.regnet_y_400mf` |
| resnet18 | .rpk | ✅ `models.resnet18` |
| shufflenet_v2 | .rpk | ✅ `models.shufflenet_v2_x1_5` |
| squeezenet1.0 | .rpk | ✅ `models.squeezenet1_0` |

### Modèles absents (NON déployables directement)

- **MobileNetV3-Large** — absent du Model Zoo IMX500
- **EfficientNet-B1/B2/B3/B4** (non-V2) — absents
- **ResNet50** — absent (trop gros pour 8 Mo)

### Recommandation IMX500

**EfficientNetV2-B2** — meilleur rapport précision/taille parmi les modèles disponibles, pré-converti .rpk existant, entraînable via torchvision.

### Sources

- Model Zoo officiel : https://github.com/raspberrypi/imx500-models
- IMX500 Converter : https://developer.aitrios.sony-semicon.com/en/raspberrypi-ai-camera/documentation/imx500-converter
- Documentation Raspberry Pi AI Camera : https://www.raspberrypi.com/documentation/accessories/ai-camera.html

---

## 2. Hailo-10H (Pi 5 + AI HAT+ 2)

### Contraintes

- **40 TOPS** (INT4), 8 Go LPDDR4 dédié
- Conversion via **Hailo Dataflow Compiler** (ONNX → .hef)
- PCIe Gen 3

### Modèles de classification disponibles — Top 20 (Model Zoo officiel)

53 modèles au total. Classés par accuracy quantifiée Hailo-10H (ImageNet Top-1) :

| # | Modèle | Top-1 (%) | Résolution | Entraînable (torchvision) |
|---|--------|-----------|------------|:-------------------------:|
| 1 | **vit_base** | **83.6** | 224×224 | ✅ `models.vit_b_16` |
| 2 | nextvit_base | 83.3 | 224×224 | ❌ (Microsoft, GitHub) |
| 3 | nextvit_small | 82.6 | 224×224 | ❌ |
| 4 | **vit_large** | **82.5** | 224×224 | ✅ `models.vit_l_16` |
| 5 | davit_tiny | 82.3 | 224×224 | ❌ (timm) |
| 6 | cas_vit_t | 81.6 | 384×384 | ❌ |
| 7 | cas_vit_m | 81.1 | 384×384 | ❌ |
| 8 | **vit_small** | **80.5** | 224×224 | ❌ (timm `vit_small_patch16_224`) |
| 9 | efficientnet_lite4 | 80.1 | 300×300 | ❌ (timm) |
| 10 | **swin_small** | **80.0** | 224×224 | ✅ `models.swin_s` |
| 11 | cas_vit_s | 79.8 | 384×384 | ❌ |
| 12 | deit_base | 79.8 | 224×224 | ❌ (torch.hub Facebook) |
| 13 | **swin_tiny** | **79.4** | 224×224 | ✅ `models.swin_t` |
| 14 | efficientnet_l | 79.3 | 300×300 | ❌ (timm) |
| 15 | levit256 | 79.3 | 224×224 | ❌ (timm) |
| 16 | levit384 | 79.2 | 224×224 | ❌ (timm) |
| 17 | vit_base_bn | 79.1 | 224×224 | ❌ (variante Hailo) |
| 18 | efficientnet_lite3 | 78.7 | 280×280 | ❌ (timm) |
| 19 | efficientnet_m | 78.5 | 240×240 | ❌ (timm) |
| 20 | **resnext50_32x4d** | **78.4** | 224×224 | ✅ `models.resnext50_32x4d` |

### Modèles absents (NON dans le Model Zoo Hailo-10H)

- **EfficientNet-B0/B1/B2/B3/B4** (nomenclature torchvision) — le Model Zoo utilise efficientnet_s/m/l et efficientnet_lite0-4
- **EfficientNetV2** — absent du Model Zoo Hailo-10H

### Recommandation Hailo-10H

**ViT-B/16 (vit_base)** — 83.6% Top-1 ImageNet, le plus performant du Model Zoo, disponible dans torchvision (`models.vit_b_16`). Alternative : **Swin-Small** (80.0%) si le ViT s'avère trop lourd après fine-tuning.

### Sources

- Model Zoo officiel Hailo-10H Classification : https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO10H/HAILO10H_classification.rst
- Documentation Hailo : https://hailo.ai/developer-zone/documentation/
- Hailo Model Zoo GitHub : https://github.com/hailo-ai/hailo_model_zoo

---

## 3. Tableau croisé — Meilleurs modèles entraînables + déployables

| Priorité | Modèle | IMX500 | Hailo-10H | torchvision | Top-1 ImageNet |
|----------|--------|:------:|:---------:|:-----------:|:--------------:|
| ⭐ IMX500 | **EfficientNetV2-B2** | ✅ | ❌ | ✅ | ~80% |
| ⭐ Hailo | **ViT-B/16** | ❌ | ✅ (83.6%) | ✅ | 84.5% |
| Alt. IMX500 | EfficientNet-B0 | ✅ | ❌ | ✅ | 77.1% |
| Alt. IMX500 | MobileNetV2 | ✅ | ✅ (71.0%) | ✅ | 72.0% |
| Alt. Hailo | Swin-Small | ❌ | ✅ (80.0%) | ✅ | 83.1% |
| Alt. Hailo | Swin-Tiny | ❌ | ✅ (79.4%) | ✅ | 81.3% |
| Les deux | MobileNetV2 | ✅ | ✅ | ✅ | 72.0% |

### Modèles à entraîner (4 architectures)

1. **MobileNetV2** — baseline, déployable sur les deux plateformes (déjà implémenté)
2. **EfficientNet-B0** — meilleur que MobileNetV2 pour IMX500 (déjà implémenté)
3. **EfficientNetV2-B2** — meilleur choix IMX500 (à ajouter)
4. **ViT-B/16** — meilleur choix Hailo-10H, 83.6% (à ajouter)

---

## 4. Pipeline de déploiement

### IMX500 (PyTorch → .rpk)

```
PyTorch (.pth) → ONNX (.onnx) → Quantification INT8 → IMX500 Converter → Firmware .rpk
```

Outils nécessaires :
- `torch.onnx.export()`
- Sony IMX500 Converter (SDK AITRIOS)
- Contrainte : modèle INT8 < 8 Mo

### Hailo-10H (PyTorch → .hef)

```
PyTorch (.pth) → ONNX (.onnx) → Hailo Dataflow Compiler → Quantification INT4/INT8 → .hef
```

Outils nécessaires :
- `torch.onnx.export()`
- Hailo Dataflow Compiler (DFC)
- Hailo Model Zoo (scripts de parsing pré-configurés)
