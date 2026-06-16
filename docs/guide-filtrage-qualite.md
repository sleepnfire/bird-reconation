# Guide : filtrage qualité des images

Date : 2026-06-15

## Pourquoi filtrer

Le dataset iNaturalist contient ~124 911 images pour 558 espèces d'oiseaux européens. Ces images proviennent de contributeurs amateurs et incluent inévitablement du bruit : spécimens morts, illustrations scannées, captures d'écran, photos floues, images sans oiseau, doublons visuels et erreurs d'étiquetage. Entraîner un modèle sur ces données brutes dégrade les performances et introduit des biais.

Le pipeline de filtrage (`quality_filter.py`) nettoie le dataset en 6 étapes avant l'entraînement.

---

## Vue d'ensemble du pipeline

```
Images brutes (iNaturalist)
        │
        ▼
┌──────────────────────────┐
│ 1. Classification CLIP   │  Catégorise chaque image via zero-shot
│    zero-shot (ViT-L-14)  │  → good / dead_specimen / illustration /
│                          │    screen_scan / not_bird / poor_quality
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ 2. Détection d'outliers  │  Embedding centroïde par espèce,
│    par embedding         │  flag si distance > mean + 1.5σ
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ 3. Déduplication visuelle│  Similarité cosinus entre paires,
│    (near-duplicates)     │  supprime les doublons (seuil 0.95)
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ 4. Vérification croisée  │  Compare la distance au centroïde propre
│    des labels (mislabel) │  vs centroïde le plus proche d'une autre espèce
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ 5. Filtre bbox minimale  │  Rejette si la bbox de l'oiseau < 5%
│                          │  de la surface de l'image
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ 6. apply_filter          │  Déplace les images rejetées,
│                          │  MAJ annotations/label_map/metadata
└──────────────────────────┘
```

Les étapes 1 à 4 produisent des rapports (JSON + embeddings .pt). L'étape 6 consomme ces rapports pour effectuer le nettoyage réel. Cette séparation permet d'inspecter les rapports avant de supprimer quoi que ce soit.

---

## Étape 1 — Classification CLIP zero-shot

### Principe

Le modèle CLIP ViT-L-14 (pré-entraîné OpenAI) classe chaque image dans l'une des 6 catégories via la similarité cosinus entre l'embedding de l'image et des prompts textuels prédéfinis.

### Catégories

| Catégorie | Exemples de prompts | Ce qui est capturé |
|-----------|--------------------|--------------------|
| `good` | "a sharp clear photograph of a wild bird perched on a branch in nature" | Photos exploitables pour l'entraînement |
| `dead_specimen` | "a photograph of a dead bird lying on the ground", "a taxidermy bird specimen" | Spécimens morts, taxidermie, roadkill |
| `illustration` | "a painted illustration of a bird from a field guide book" | Dessins, aquarelles, gravures |
| `screen_scan` | "a photograph of a computer monitor showing a bird image" | Photos d'écran, scans de livres |
| `not_bird` | "a photograph of an empty landscape with no animals visible" | Paysages, nids vides, plumes, insectes |
| `poor_quality` | "an extremely blurry out of focus photograph" | Flou, sous-exposé, surexposé, pixelisé |

Chaque catégorie utilise 5 prompts dont les embeddings textuels sont moyennés. La classification est le softmax de la similarité image-texte sur les 6 catégories.

### Reject margin

Un mécanisme de marge protège les cas ambigus : si le meilleur score négatif dépasse le score `good` de moins de 0.005, l'image est quand même classée `good`. Cela évite de rejeter des photos exploitables sur la base d'une différence de score marginale.

### Fondements académiques

- Radford et al. 2021 (CLIP, ICML) — le zero-shot CLIP rivalise avec les classificateurs supervisés sur ImageNet
- DataComp (Gadre et al., NeurIPS 2023) — filtrage CLIP appliqué à l'échelle de LAION pour la curation de datasets
- BioTrove (Yang et al., NeurIPS 2024) — curation d'images iNaturalist via CLIP

---

## Étape 2 — Détection d'outliers par embedding

### Principe

Pour chaque espèce, on calcule le centroïde des embeddings CLIP de toutes ses images. Les images dont la distance cosinus au centroïde dépasse `mean + 1.5σ` sont flaggées comme outliers.

### Pourquoi

Un outlier dans l'espace d'embedding signifie que l'image est visuellement très différente des autres images de la même espèce. Cela peut indiquer :
- une photo prise dans des conditions inhabituelles (angle extrême, éclairage atypique)
- un oiseau juvénile ou en plumage non-nuptial très différent de l'adulte
- une erreur d'identification dans iNaturalist
- une image partiellement corrompue qui n'a pas été détectée par l'étape 1

### Paramètre

Le seuil `sigma` (défaut 1.5) contrôle la sensibilité. Un seuil plus bas (ex. 1.0) est plus strict et flagge davantage d'outliers. Les espèces avec moins de 3 images ne subissent pas de détection d'outliers.

### Fondements académiques

- LAION-5B (Schuhmann et al., NeurIPS 2022) — détection d'outliers par centroïde + distance cosinus
- FiftyOne / Voxel51 — outils de curation visuels utilisant les embeddings pour détecter les anomalies

---

## Étape 3 — Déduplication visuelle (near-duplicates)

### Principe

Les photos en rafale ou prises par plusieurs observateurs du même oiseau produisent des images quasi-identiques. Ces doublons introduisent un biais (le modèle voit la même scène plusieurs fois) et un risque de data leakage si un doublon se retrouve dans le train et l'autre dans le test.

La déduplication calcule la similarité cosinus entre toutes les paires d'images d'une même espèce. Les paires dont la similarité dépasse le seuil (défaut 0.95) sont considérées comme des near-duplicates.

### Résolution des doublons

Pour chaque paire dupliquée, l'algorithme garde l'image avec le meilleur score CLIP `good` (celle qui a la meilleure qualité photographique estimée). En cas d'égalité, l'image dont le nom vient en premier alphabétiquement est gardée.

Quand plusieurs images forment un cluster de doublons (A≈B, B≈C), la résolution est gloutonne par similarité décroissante : les paires les plus similaires sont traitées en premier, et une image déjà marquée pour suppression n'entraîne pas de suppression supplémentaire.

### Paramètres

- `--duplicate-threshold` (défaut 0.0, désactivé) : seuil de similarité cosinus. Valeur recommandée : 0.95

### Fondements académiques

- LAION-5B (Schuhmann et al., NeurIPS 2022) — recommande la déduplication par similarité d'embedding
- DataComp (Gadre et al., NeurIPS 2023) — montre que la déduplication améliore les performances downstream

---

## Étape 4 — Vérification croisée des labels (mislabel detection)

### Principe

Les labels iNaturalist sont "research-grade" (validés par au moins 2 identificateurs), mais des erreurs subsistent. NABirds (CVPR 2015) a mesuré ~4% d'erreur de label dans CUB-200.

La détection de mislabels compare, pour chaque image, sa distance au centroïde de sa propre espèce (`own_distance`) avec sa distance au centroïde de l'espèce la plus proche (`nearest_distance`). Si l'image est significativement plus proche d'une autre espèce que de la sienne, elle est suspecte :

```
suspected = (own_distance - nearest_distance) > margin
```

### Exemple concret

Une photo étiquetée "Mésange charbonnière" dont l'embedding est plus proche du centroïde "Mésange bleue" que du centroïde "Mésange charbonnière" est probablement mal étiquetée — ou montre un cas ambigu (hybride, juvénile).

### Paramètres

- `--margin` (défaut 0.1) : marge de tolérance. Une marge de 0 flagge toute image plus proche d'une autre espèce. Une marge de 0.1 ne flagge que les cas où l'écart est significatif.

### Prérequis

Les embeddings doivent être persistés en fichiers `.pt` (générés automatiquement par la commande `report`). La détection nécessite au moins 2 espèces.

### Fondements académiques

- Van Horn et al. (NABirds, CVPR 2015) — mesure de 4% d'erreur de label dans CUB-200
- Pebblous DataClinic — identification des "ambiguous class boundaries" par overlap d'embeddings inter-classes
- arXiv 2412.15844 — méthodes de détection d'erreurs de labels par analyse d'embeddings

---

## Étape 5 — Filtre bbox minimale

### Principe

Les annotations iNaturalist incluent des bounding boxes (détection automatique). Si la bbox de l'oiseau occupe moins de 5% de la surface de l'image, l'oiseau est trop petit pour être utile à l'entraînement — le modèle apprendrait surtout le fond.

### Paramètre

- `--min-bbox-pct` (défaut 5.0) : pourcentage minimum de l'image occupé par la bbox

---

## Étape 6 — Application du filtre (apply_filter)

### Ce qui se passe

1. Les images rejetées sont **déplacées** (pas supprimées) vers un dossier `europe_rejected/` qui reproduit la structure du dataset
2. Le fichier `annotations.json` de chaque espèce est mis à jour (les annotations des images rejetées sont transférées dans le dossier rejected)
3. Si toutes les images d'une espèce sont rejetées, l'espèce est retirée du `label_map.json` et du `metadata.json`
4. Le `label_map.json` est ré-indexé alphabétiquement après suppression

---

## Utilisation

### Workflow complet recommandé

```bash
# 1. Générer les rapports qualité (avec détection de duplicates)
python quality_filter.py report --split train --workers 4 --duplicate-threshold 0.95

# 2. Inspecter les rapports (optionnel)
#    → reports/train/*.json contient les classifications, outliers et duplicates
#    → reports/train/*.pt contient les embeddings pour la détection de mislabels

# 3. Détecter les mislabels (rapport console)
python quality_filter.py mislabel --split train --margin 0.1

# 4. Appliquer le filtre (déplacer les images rejetées)
python quality_filter.py apply --split train \
    --remove-outliers \
    --remove-duplicates \
    --remove-mislabeled \
    --mislabel-margin 0.1

# 5. Répéter pour validation et test
python quality_filter.py report --all --workers 4 --duplicate-threshold 0.95
python quality_filter.py apply --all --remove-outliers --remove-duplicates --remove-mislabeled
```

### Commandes détaillées

#### `report` — Générer les rapports qualité

```bash
python quality_filter.py report [options]
```

| Option | Défaut | Description |
|--------|--------|-------------|
| `--split` | `train` | Split à traiter (`train`, `validation`, `test`) |
| `--all` | — | Traiter les 3 splits |
| `--workers N` | 1 | Nombre de workers parallèles |
| `--resume` | — | Sauter les espèces déjà traitées |
| `--sigma FLOAT` | 1.5 | Seuil outlier en écarts-types |
| `--batch-size N` | 64 | Taille de batch GPU |
| `--fp16` | — | Utiliser float16 (plus rapide) |
| `--reject-margin FLOAT` | 0.005 | Marge de rejet CLIP |
| `--duplicate-threshold FLOAT` | 0.0 | Seuil de similarité cosinus pour les near-duplicates (0 = désactivé) |

**Sorties** : pour chaque espèce, un fichier JSON (rapport) et un fichier `.pt` (embeddings) dans le dossier `reports/{split}/`.

#### `apply` — Appliquer le filtre

```bash
python quality_filter.py apply [options]
```

| Option | Défaut | Description |
|--------|--------|-------------|
| `--split` | `train` | Split à filtrer |
| `--all` | — | Filtrer les 3 splits |
| `--remove-outliers` | — | Retirer les images outliers |
| `--remove-duplicates` | — | Retirer les near-duplicates |
| `--remove-mislabeled` | — | Retirer les images suspectées mal étiquetées |
| `--mislabel-margin FLOAT` | 0.1 | Marge pour la détection de mislabels |
| `--min-bbox-pct FLOAT` | 5.0 | Bbox minimale en % de l'image |
| `--rejected-dir PATH` | `dataset/europe_rejected` | Dossier de destination |

#### `mislabel` — Rapport de mislabels (consultation uniquement)

```bash
python quality_filter.py mislabel [options]
```

| Option | Défaut | Description |
|--------|--------|-------------|
| `--split` | `train` | Split à analyser |
| `--all` | — | Analyser les 3 splits |
| `--margin FLOAT` | 0.1 | Marge de détection |

---

## Structure des fichiers générés

### Rapport JSON (`reports/{split}/{espèce}.json`)

```json
{
  "summary": {
    "total": 150,
    "good": 120,
    "dead_specimen": 10,
    "illustration": 5,
    "screen_scan": 8,
    "not_bird": 4,
    "poor_quality": 3,
    "duplicates": 2
  },
  "images": {
    "photo_001.jpg": {
      "category": "good",
      "confidence": 0.8234,
      "scores": {
        "good": 0.8234,
        "dead_specimen": 0.0456,
        "illustration": 0.0123,
        "screen_scan": 0.0987,
        "not_bird": 0.0156,
        "poor_quality": 0.0044
      }
    }
  },
  "outliers": {
    "photo_001.jpg": {
      "distance": 0.1234,
      "is_outlier": false
    }
  },
  "duplicates": [
    {
      "kept": "photo_001.jpg",
      "removed": "photo_003.jpg",
      "similarity": 0.97
    }
  ]
}
```

La clé `duplicates` n'est présente que si `--duplicate-threshold > 0` lors du `report`.

### Embeddings (`.pt`)

Fichier PyTorch (`reports/{split}/{espèce}.pt`) contenant un dict `{nom_image: tensor}` avec des vecteurs float32 de dimension 768 (sortie de ViT-L-14). Utilisés par la détection de mislabels.

---

## Tests

Le pipeline est couvert par 56 tests (`tests/test_quality_filter.py`) :

| Plage | Domaine |
|-------|---------|
| UC-Q01 à UC-Q04 | Classification CLIP zero-shot |
| UC-Q05 à UC-Q06 | Filtre bbox |
| UC-Q07 à UC-Q09 | Embeddings et outliers |
| UC-Q10 à UC-Q14 | Rapports qualité JSON |
| UC-Q15 à UC-Q17 | Parallélisation |
| UC-Q18 à UC-Q28 | Nettoyage et apply_filter |
| UC-Q29 à UC-Q35 | Batch processing, fp16, device |
| UC-Q36 à UC-Q42 | Déduplication visuelle |
| UC-Q43 à UC-Q52 | Vérification croisée des labels et intégration |

```bash
# Lancer tous les tests
pytest tests/test_quality_filter.py -v

# Lancer uniquement les tests de déduplication
pytest tests/test_quality_filter.py -k "UCQ36 or UCQ37 or UCQ38 or UCQ39 or UCQ40 or UCQ41 or UCQ42"

# Lancer uniquement les tests de mislabel
pytest tests/test_quality_filter.py -k "UCQ43 or UCQ44 or UCQ45 or UCQ46 or UCQ47 or UCQ48 or UCQ49"
```
