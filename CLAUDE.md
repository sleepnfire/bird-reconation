# Bird Detection — Reconnaissance d'oiseaux par IA

## Objectif du projet

Développer un système d'intelligence artificielle capable de reconnaître et identifier des espèces d'oiseaux, déployé sur deux plateformes Raspberry Pi :

- **Raspberry Pi 5** (16 Go RAM) + **AI HAT+ 2** (Hailo-10H) — **40 TOPS** (INT4), 8 Go LPDDR4 dédié, PCIe Gen 3
- **Raspberry Pi Zero** + **AI Camera** (Sony IMX500) — **8 Mo de SRAM** pour le modèle, ~30 FPS (MobileNet SSD), pas de TOPS officiel (estimé <1 TOPS)

### Contraintes par cible

| Cible | Accélérateur | Performance | Mémoire modèle | Architectures compatibles |
|-------|-------------|-------------|-----------------|---------------------------|
| Pi 5 + AI HAT+ 2 | Hailo-10H | 40 TOPS | 8 Go LPDDR4 | EfficientNet-B0/B2, ResNet, YOLOv8 |
| Pi Zero + AI Camera | Sony IMX500 | <1 TOPS | **8 Mo SRAM** | MobileNetV2, EfficientNet-Lite uniquement |

Les modèles pour l'IMX500 doivent être quantifiés INT8 et convertis via l'IMX500 Converter (format firmware Sony).

## Structure du projet

```
bird-detection/
├── main.py                    # Point d'entrée (test PyTorch)
├── quality_filter.py          # Filtrage qualité CLIP + score composite + calibration
├── create-aws-dataset/        # Préparation du dataset à partir d'iNaturalist
│   ├── iNaturalist.db         # Base SQLite (~177 Go) des observations iNaturalist
│   ├── observations.csv       # Export CSV des observations (~27 Go)
│   ├── photos.csv             # Export CSV des photos (~47 Go)
│   ├── taxa.csv               # Export CSV de la taxonomie (~185 Mo)
│   └── RIOC.mwb               # Schéma MySQL Workbench (modèle de données)
├── dataset/                   # Datasets organisés par périmètre géographique
│   ├── garden/                # Oiseaux de jardin (~20 espèces principales)
│   ├── france/                # Oiseaux de France
│   ├── europe/                # Oiseaux d'Europe
│   └── world/                 # Oiseaux du monde
│       └── metadata.json      # Format : slug, nom scientifique, famille, noms FR/EN
├── reports/                   # Rapports qualité JSON + embeddings .pt
├── calibration/               # Ground truth et résultats d'optimisation
├── docs/                      # Documentation (guide-filtrage-qualite.md)
├── A-trier/                   # Datasets bruts à trier (images, CSV)
└── .venv/                     # Environnement virtuel Python
```

Chaque sous-dossier de `dataset/` contient : `train/`, `test/`, `validation/`, `label_map.json`, `metadata.json`.

## Stack technique

- **Python 3.14** (virtualenv)
- **PyTorch 2.10** + torchvision 0.25
- **Base de données** : SQLite (iNaturalist.db), MySQL Workbench pour le schéma
- **Source de données** : iNaturalist (observations, photos, taxonomie)
- **IDE** : PyCharm

## Espèces cibles

Le périmètre principal est les oiseaux de jardin en France (~20 espèces : mésanges, rougegorge, merle, moineaux, pinsons, etc.). Des périmètres plus larges (France, Europe, monde) sont prévus.

## Stratégie d'entraînement (2 étapes, coarse-to-fine)

**Étape 1 — Fine-tuning sur oiseaux de France/Europe (~200-400 espèces)**
- MobileNetV2 pré-entraîné ImageNet (Pi Zero) / EfficientNet-B0 pré-entraîné (Pi 5)
- Fine-tuner sur des données iNaturalist d'oiseaux français/européens
- Apprentissage des features discriminantes entre familles d'oiseaux régionaux

**Étape 2 — Fine-tuning final sur ~20-30 espèces cibles**
- Spécialisation sur les espèces de jardin + espèces « confondantes »
- Classe « autre oiseau » pour rejeter les espèces hors périmètre
- Data augmentation intensive (rotation, flip, couleur, crop) — ~224 images/espèce (124 911 images, 558 espèces)

## Commandes

```bash
# Activer l'environnement virtuel
source .venv/bin/activate

# Lancer le script principal
python main.py

# Filtrage qualité (pipeline complet dans quality_filter.py)
python quality_filter.py report --all --workers 4 --duplicate-threshold 0.95
python quality_filter.py apply --all --remove-outliers --remove-duplicates --remove-mislabeled
python quality_filter.py review --split train --mode borderline
python quality_filter.py metrics
python quality_filter.py optimize

# Tests
pytest tests/test_quality_filter.py -v
```

## Conventions

- Les métadonnées d'espèces suivent le format : `slug` (nom_scientifique en snake_case), `scientific_name`, `family`, `english_name`, `french_name`
- Les datasets sont structurés en `train/test/validation` avec un `label_map.json` et `metadata.json`
- Les fichiers volumineux (CSV, DB) ne doivent pas être versionnés dans git
