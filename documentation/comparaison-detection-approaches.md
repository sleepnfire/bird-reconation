# Comparaison des approches de détection d'oiseaux

Date : 2026-06-14

---

## 1. Contexte et problématique

### État actuel : classification pure

Les modèles entraînés (ViT-B/16, EfficientNet-B0, MobileNetV2) sont des **classificateurs** : ils prennent en entrée une image pré-cadrée sur un oiseau et prédisent l'espèce. Le meilleur modèle (ViT-B/16, run 3) atteint **79.52% accuracy** et **F1 macro 0.772** sur 558 espèces européennes.

Le pré-cadrage est réalisé en amont par `auto_annotate.py` (FasterRCNN ou YOLO11), qui produit des bounding boxes au format COCO `[x, y, w, h]`. Ce processus est **offline** — il ne fait pas partie du pipeline d'inférence.

### Ce qui manque : la détection en temps réel

En production, le système doit :

1. **Localiser** l'oiseau dans le champ de la caméra (bounding box)
2. **Identifier** l'espèce (classification)
3. Faire les deux **simultanément**, en temps réel, sur un edge device

### Pourquoi la validation actuelle est insuffisante

Les 343 tests (`test_train.py`) ne valident que la classification sur des images déjà cadrées. Ils ne mesurent pas :

- La qualité de la localisation (IoU des bounding boxes prédites vs ground truth)
- La performance du pipeline complet (image brute → espèce + position)
- La latence end-to-end par plateforme (Hailo, IMX500)

Toute approche choisie devra inclure une **nouvelle suite de tests** couvrant ces trois aspects.

### Contraintes matérielles

| Cible | Accélérateur | Performance | Mémoire modèle | Multi-modèle |
|-------|-------------|-------------|-----------------|:------------:|
| Pi 5 + AI HAT+ 2 | Hailo-10H | 40 TOPS (INT4) | 8 Go LPDDR4X | **Oui** |
| Pi Zero + AI Camera | Sony IMX500 | <1 TOPS | **8 Mo SRAM** | **Non** |

**Contrainte critique IMX500** : le capteur ne peut charger qu'**un seul modèle** à la fois. Le rechargement d'un modèle (écriture en SRAM) prend plusieurs secondes, excluant toute approche multi-modèle en temps réel.

---

## 2. Inventaire des ressources existantes

### Données d'entraînement

| Ressource | Quantité | Détails |
|-----------|----------|---------|
| Images totales | 218 190 | Dataset europe, 558 espèces |
| Images avec bounding box | 195 726 (89.7%) | Format COCO, score médian 0.908 |
| Images score > 0.8 | 158 059 (80.8%) | Bbox de haute qualité |
| Images sans bbox | 22 464 (10.3%) | À traiter séparément |

### Dataset de détection existant

Le dossier `dataset/detection/` contient un travail antérieur :

- **2 010 images** avec labels au format YOLO (classe 0 = "bird")
- **100 images de test**
- `bird_box_model.h5` (53 Mo) — modèle Keras, non intégré au pipeline actuel

### Code existant réutilisable

- `auto_annotate.py` : détection FasterRCNN et YOLO11n/s/m, sortie bbox COCO
- `train.py` : pipeline de classification complet, `BirdDataset` avec crop bbox
- `export.py` : export ONNX, Hailo (.hef), IMX500 (.rpk) — classification uniquement

---

## 3. Approche 1 — YOLO all-in-one (détection + classification unifiée)

### Principe

Un seul modèle YOLO entraîné à la fois détecter et classifier les oiseaux. Chaque espèce est une classe de détection. Le modèle produit directement des bounding boxes annotées avec l'espèce.

### Architectures disponibles

| Modèle | Params | mAP COCO | Taille INT8 (80 classes) | Hailo-10H | IMX500 |
|--------|--------|----------|--------------------------|:---------:|:------:|
| YOLOv8n | 3.2M | 37.3 | ~3.2 Mo | ✅ | ✅ |
| YOLOv8s | 11.2M | 44.9 | ~11 Mo | ✅ | ❌ |
| YOLOv8m | 25.9M | 50.2 | ~26 Mo | ✅ | ❌ |
| YOLO11n | 2.6M | 39.5 | ~2.2 Mo | ✅ | ✅ |
| YOLO11s | 9.4M | 47.0 | ~9 Mo | ✅ | ❌ |
| YOLO11m | 20.1M | 51.5 | ~20 Mo | ✅ | ❌ |

*Sources : [Hailo Model Zoo — Object Detection](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO10H/HAILO10H_object_detection.rst), [Ultralytics IMX500 Integration](https://docs.ultralytics.com/integrations/sony-imx500)*

**IMX500** : seuls YOLOv8n et YOLO11n (nano) sont supportés. Ultralytics précise : *"IMX export is designed and benchmarked for YOLOv8n and YOLO11n (nano). Other architectures and model scales are not supported."*

### Impact du nombre de classes sur la taille

La tête de détection YOLO scale linéairement avec le nombre de classes, mais le backbone (qui représente >90% des paramètres) reste identique. Pour YOLO11n :

| Classes | Params estimés | Taille INT8 estimée |
|---------|---------------|---------------------|
| 80 (COCO) | 2.6M | 2.2 Mo |
| 558 (europe) | ~3.1M | ~3-4 Mo |

Un YOLO11n à 558 classes **tient dans les 8 Mo de l'IMX500** en termes de taille.

### Le problème : précision fine-grained

La recherche montre que la performance de YOLO chute significativement avec le nombre de classes fine-grained :

> Le regroupement d'espèces morphologiquement similaires améliore significativement la détection : par exemple, les macropodes passent d'un mAP individuel de ~0.55 à 0.97 une fois regroupés en une seule classe. Cependant, les gains varient selon les taxons et ne sont pas systématiques.
> — [Comparing fine-grained and coarse-grained object detection for ecology, arXiv 2407.00018](https://arxiv.org/html/2407.00018v1)

Les espèces morphologiquement similaires souffrent particulièrement :

> *"Macropods showed substantial improvements when merged into a single class"* — mAP individuel de 0.555 → 0.968 en groupe.

Pour 558 espèces d'oiseaux européens (incluant des genres très proches comme *Acrocephalus*, *Larus*, *Ficedula*), un YOLO ne peut pas rivaliser avec un classificateur spécialisé comme ViT-B/16. Les architectures YOLO ont un backbone trop petit pour apprendre les différences subtiles entre espèces proches.

Le modèle YOLO_BD (modification de YOLOv8 avec modules spécialisés pour les oiseaux) atteint **75.2% mAP@0.5 avec 6.6M paramètres** sur un nombre limité d'espèces lacustres — soit plus du double des paramètres d'un YOLO11n. ([Source : Nature Scientific Reports](https://www.nature.com/articles/s41598-026-47900-0))

### Avantages

- Architecture simple : un seul modèle, un seul export
- Latence minimale : une seule passe d'inférence
- Compatible IMX500 (taille) et Hailo-10H

### Inconvénients

- Précision espèce nettement inférieure à un classificateur spécialisé
- YOLO nano insuffisant pour 558 espèces fine-grained
- Pas de recherche publiée démontrant YOLO sur >200 espèces d'oiseaux
- Ajouter une espèce = re-entraîner tout le modèle

---

## 4. Approche 2 — Pipeline two-stage (détecteur + classificateur)

### Principe

Deux modèles spécialisés enchaînés :

1. **Détecteur binaire** : localise les oiseaux dans l'image (1 classe "bird") → bounding boxes
2. **Classificateur** : identifie l'espèce sur chaque crop → le classificateur existant (ViT-B/16, EfficientNet-B0, MobileNetV2)

### Validation par la littérature

Cette approche est le **standard de l'industrie** pour la reconnaissance d'espèces à grande échelle :

**Merlin Bird ID (Cornell Lab of Ornithology)**
- CNN (architecture non divulguée) entraîné sur 6 millions de photos
- Identifie **6 900 espèces** avec **95% de précision moyenne**
- Les annotateurs humains dessinent des bounding boxes, le modèle apprend à localiser puis identifier
- Modèle packagé en ~50 Mo pour mobile
- *Source : [eBird — New Photo ID Model in Merlin](https://ebird.org/news/new-photo-id-model-in-merlin)*

**iNaturalist**
- Classification de **5 089 espèces** (tous taxons)
- Baselines Inception-ResNet V2, ResNet, MobileNet, entraînés sur 859 000 images
- Dataset benchmark : iNaturalist Species Classification and Detection Dataset
- *Source : [Van Horn et al., CVPR 2018](https://arxiv.org/pdf/1707.06642)*

**Two-stage Wildlife Event Classification (2026)**
- Pipeline YOLOv8 (détection) + EfficientNet (classification)
- Déployé en production continue depuis mai 2025
- *Source : [Sensors, Vol. 26, Issue 4](https://www.mdpi.com/1424-8220/26/4/1366)*

**Autonomous AI Bird Feeder (2025)**
- Pipeline two-stage pour monitoring de biodiversité en jardin
- Détection + classification sur edge device
- *Source : [arXiv 2508.09398](https://arxiv.org/pdf/2508.09398)*

### Spécifications du détecteur binaire

| Caractéristique | Valeur |
|-----------------|--------|
| Architecture | YOLO11n (2.6M params) |
| Classes | 1 ("bird") |
| Taille INT8 | ~2 Mo |
| Données | 195 726 images avec bbox + 2 010 images detection/ |
| Latence IMX500 | ~58 ms |
| mAP attendu | >90% (1 seule classe, sujet distinct du fond) |

### Spécifications du classificateur (existant)

| Cible | Modèle | Accuracy | Taille |
|-------|--------|----------|--------|
| Hailo-10H | ViT-B/16 | 79.52% | ~987 Mo FP32, ~250 Mo INT8 |
| IMX500 | MobileNetV2 | 61.98% | ~34 Mo FP32, ~8 Mo INT8 |
| IMX500 (alt.) | EfficientNet-B0 | (à mesurer) | ~55 Mo FP32, ~14 Mo INT8 |

### Avantages

- **Meilleure précision** : chaque modèle est spécialisé dans sa tâche
- **Scalable** : ajouter des espèces = re-entraîner uniquement le classificateur
- **Réutilisation** : les classificateurs existants (ViT-B/16 79.5%) sont directement utilisables
- **Validé par les leaders** : Merlin (6 900 espèces, 95%), iNaturalist (5 089 espèces)
- **Modulaire** : possibilité de mettre à jour détecteur ou classificateur indépendamment

### Inconvénients

- Latence additive (détection + classification = 2 passes d'inférence)
- Complexité accrue du pipeline (2 modèles à maintenir, à exporter, à déployer)
- **IMX500 : impossible en temps réel** (un seul modèle chargeable, rechargement = secondes)

---

## 5. Approche 3 — Stratégie hybride (adaptée par plateforme)

### Principe

Chaque plateforme utilise l'approche la mieux adaptée à ses contraintes :

- **Hailo-10H** : two-stage (sa capacité multi-modèle le permet)
- **IMX500** : approche monolithique adaptée à ses limitations

### Stratégie Hailo-10H : two-stage

Le Hailo-10H supporte nativement le **scheduling multi-modèle** via le Hailo Runtime. Le pipeline :

```
Caméra → [YOLO11n/s détecteur 1 classe] → bbox → crop → [ViT-B/16 classificateur 558 espèces] → espèce
                    Hailo-10H (modèle 1)                        Hailo-10H (modèle 2)
```

- YOLO11s (9.4M params, 47.0 mAP) offre un bon compromis détection/taille
- ViT-B/16 est le #1 du Model Zoo Hailo (83.6% ImageNet Top-1)
- Latence estimée : ~30-50 ms total (les deux modèles sur le NPU)

### Stratégie IMX500 : options à évaluer

L'IMX500 ne pouvant charger qu'un modèle, deux options existent :

**Option A — YOLO11n 558 classes (tout-en-un)**

```
Caméra → [YOLO11n 558 classes] → espèce + bbox
              IMX500 (~3-4 Mo)
```

- Simple, temps réel (~58 ms/image)
- Précision espèce **significativement inférieure** au ViT-B/16 (estimation : 30-50% vs 79.5%)
- Convient si la localisation est plus importante que la précision d'identification

**Option B — YOLO11n détecteur + classificateur CPU**

```
Caméra → [YOLO11n 1 classe] → bbox → crop → [MobileNetV2 CPU] → espèce
              IMX500 (~2 Mo)                     Pi Zero CPU
```

- Détection rapide sur IMX500, classification sur CPU
- Latence CPU Pi Zero : ~500 ms-1 s par inférence MobileNetV2 (ARM Cortex-A53)
- Précision supérieure à l'option A (MobileNetV2 62% > YOLO11n 558 classes estimé)
- ~1-2 FPS effectifs

**Dans les deux cas**, l'entraînement utilise le dataset europe (558 espèces). La réduction éventuelle du nombre de classes est une **optimisation de déploiement**, pas un choix de dataset d'entraînement.

---

## 6. Analyse comparative

### Tableau récapitulatif

| Critère | YOLO all-in-one | Two-stage | Hybride |
|---------|:---------------:|:---------:|:-------:|
| **Précision classification** | Faible (nano 558 classes) | **Élevée** (ViT 79.5%) | Élevée (Hailo) / Moyenne (IMX500) |
| **Précision détection** | Bonne | **Bonne** | Bonne |
| **Latence Hailo** | ~30 ms | ~50 ms | ~50 ms |
| **Latence IMX500** | ~58 ms | ❌ Impossible temps réel | 58 ms (option A) / ~1 s (option B) |
| **Complexité** | Simple | Moyenne | Élevée (2 stratégies) |
| **Scalabilité espèces** | Faible (re-entraîner tout) | **Élevée** (seul classificateur) | Élevée (Hailo) |
| **Taille modèle** | 3-4 Mo (nano) / 20 Mo (m) | 2 Mo + 250 Mo | Variable |
| **Validation littérature** | Limitée (>200 classes) | **Forte** (Merlin, iNat) | — |

### Faisabilité par plateforme

| | Pi 5 + Hailo-10H | Pi Zero + IMX500 |
|---|:---:|:---:|
| YOLO all-in-one | ✅ Possible (v8m/v11m) | ⚠️ Possible (nano) mais précision faible |
| Two-stage | ✅ **Optimal** | ❌ Impossible temps réel |
| Hybride | ✅ Two-stage | ⚠️ Compromis (option A ou B) |

---

## 7. Métriques de validation

### Détection

- **mAP@0.5** : proportion de détections correctes (IoU ≥ 0.5)
- **mAP@0.5:0.95** : standard COCO, moyenne sur IoU de 0.5 à 0.95
- **Recall** : proportion d'oiseaux réellement détectés

### Classification (sur crops détectés)

- **Top-1 accuracy** : l'espèce prédite est la bonne
- **Top-5 accuracy** : l'espèce correcte est dans les 5 premières prédictions
- **Macro F1** : moyenne du F1 par espèce (sensible aux classes rares)

### End-to-end

- **Précision combinée** = détection correcte (IoU ≥ 0.5) × classification correcte
- **FPS** : images par seconde en conditions réelles par plateforme
- **Latence P95** : temps de traitement au 95e percentile

### Tests à développer

Les tests actuels (`test_train.py`, 343 tests) ne couvrent que la classification. Il faut ajouter :

1. **Tests détection** : IoU des bbox prédites vs annotations ground truth
2. **Tests pipeline** : image brute → espèce identifiée + position → vérification
3. **Tests performance** : latence, mémoire, FPS par plateforme cible
4. **Tests de régression** : s'assurer que la classification ne se dégrade pas dans le pipeline

---

## 8. Recommandation finale

### Pi 5 + Hailo-10H : two-stage (recommandé)

**Architecture** : YOLO11s détecteur 1 classe (9.4M params) + ViT-B/16 classificateur 558 espèces

**Justification** :
- Le Hailo-10H supporte le multi-modèle natif — exploiter cette capacité
- Le ViT-B/16 à 79.5% est notre meilleur classificateur — le réutiliser
- Approche validée par Merlin (Cornell, 6 900 espèces, 95%), iNaturalist (5 089 espèces), et la recherche récente (Two-stage wildlife 2026)
- Scalable : ajouter des espèces ne touche que le classificateur
- YOLO11s offre un meilleur mAP que v11n (47.0 vs 39.5) tout en restant compact

**Alternative** : YOLO11m 558 classes (tout-en-un) si la simplicité prime sur la précision. À évaluer expérimentalement.

### Pi Zero + IMX500 : évaluation empirique nécessaire

La contrainte du modèle unique rend l'IMX500 fondamentalement limité pour la détection + classification fine-grained de 558 espèces. Deux options à **tester et comparer** :

**Option A — YOLO11n 558 classes**
- Avantage : temps réel, simple
- Risque : précision espèce probablement 30-50%

**Option B — YOLO11n détecteur + MobileNetV2 CPU**
- Avantage : meilleure précision (MobileNetV2 62%)
- Risque : latence ~1 s, 1-2 FPS

**Recommandation** : implémenter les deux, benchmarker sur le Pi Zero, choisir sur données.

---

## 9. Plan d'implémentation

### Phase 1 — Préparation des données de détection

Convertir les `annotations.json` (format COCO `[x, y, w, h]`) en format YOLO normalisé :
- Détecteur binaire : classe 0 = "bird" pour toutes les espèces
- YOLO 558 classes : une classe par espèce, mapping via `label_map.json`

### Phase 2 — Entraînement du détecteur binaire

YOLO11n et YOLO11s fine-tunés depuis COCO pretrained, 1 classe "bird", sur les 195 726 images annotées.

### Phase 3 — Entraînement YOLO 558 classes

YOLO11n fine-tuné sur 558 classes, pour comparaison avec le pipeline two-stage.

### Phase 4 — Pipeline two-stage

Intégrer détecteur + classificateur dans un script d'inférence unifié.

### Phase 5 — Tests de validation

Nouvelle suite de tests couvrant détection, classification, et pipeline end-to-end.

### Phase 6 — Export et benchmark

- Hailo : YOLO11s → ONNX → .hef + ViT-B/16 → ONNX → .hef
- IMX500 : YOLO11n → ONNX → .rpk
- Benchmark FPS, latence, précision sur chaque plateforme

---

## Sources

1. [Merlin Photo ID — eBird](https://ebird.org/news/new-photo-id-model-in-merlin) — 6 900 espèces, 95% accuracy, CNN (architecture non divulguée)
2. [Comparing fine-grained and coarse-grained object detection for ecology — arXiv 2407.00018](https://arxiv.org/html/2407.00018v1) — Impact nombre de classes sur mAP, YOLOv8s
3. [YOLO_BD: Fine-grained detection of Qinghai Lake birds — Nature Scientific Reports](https://www.nature.com/articles/s41598-026-47900-0) — 75.2% mAP, 6.6M params
4. [Ultralytics — Sony IMX500 Integration](https://docs.ultralytics.com/integrations/sony-imx500) — YOLOv8n/YOLO11n uniquement, 2.2 MB, 58.82ms
5. [Hailo Model Zoo — Object Detection Hailo-10H](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO10H/HAILO10H_object_detection.rst) — YOLOv8n→v11l
6. [The iNaturalist Species Classification and Detection Dataset — Van Horn et al., CVPR 2018](https://arxiv.org/pdf/1707.06642) — 5 000+ espèces
7. [Two-stage Wildlife Event Classification for Edge Deployment — Sensors 2026](https://www.mdpi.com/1424-8220/26/4/1366) — YOLOv8 + EfficientNet
8. [Autonomous AI Bird Feeder for Backyard Biodiversity Monitoring — arXiv 2025](https://arxiv.org/pdf/2508.09398) — Pipeline two-stage
9. [Hailo-10H AI Accelerator — Object Detection Demo](https://hailo.ai/resources/industries/personal-compute/object-detection-demo-with-hailo-10h-ai-accelerator/) — YOLO11m temps réel 4K
10. [Sony IMX500 + YOLO — Ultralytics Blog](https://www.ultralytics.com/blog/empowering-edge-ai-with-sony-imx500-and-aitrios) — Architecture on-sensor
