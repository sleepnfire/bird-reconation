# Plan d'implémentation — Pipeline de détection two-stage

Date : 2026-06-14

---

## 0. État des lieux

### Ce qui existe

| Composant | Fichier | Rôle | Limitation |
|---|---|---|---|
| Auto-annotation | `auto_annotate.py` | YOLO détecte les oiseaux → `annotations.json` (bbox COCO) | **Offline uniquement** — préparation du dataset, pas d'inférence |
| Entraînement | `train.py` | Entraîne des **classificateurs** (4 architectures) | Crop sur bbox avant classification — ne produit pas de bbox |
| Distillation | `distill.py` | ViT-B/16 → MobileNetV2 | Classificateur uniquement |
| Export | `export.py` | ONNX / Hailo / IMX500 | Exporte des **classificateurs**, pas de détecteur YOLO |
| Démo | `demo_webcam.py` | Classification frame entière | **Aucune bounding box** — classifie tout le cadre |
| Tests | 11 fichiers, 366 tests | Couverture complète du pipeline classification | Aucun test de détection |

### Ce qui manque pour détecter espèce + position

1. **Pipeline d'inférence two-stage** : YOLO détecte l'oiseau (bbox) → crop → classificateur identifie l'espèce
2. **Préparation du dataset YOLO** : conversion COCO `annotations.json` → format YOLO normalisé
3. **Entraînement YOLO11s** : fine-tuning 1 classe "bird" depuis COCO pretrained
4. **Export YOLO pour Hailo** : `.onnx` compatible Hailo DFC
5. **Démo avec boxes** : `demo_webcam.py` affichant des rectangles + noms d'espèces

### Données disponibles

- **195 726 images** avec bounding boxes au format COCO `[x, y, w, h]` dans `dataset/europe/{train,validation,test}/*/annotations.json`
- **558 espèces** annotées
- Format d'annotation : `{"image.jpg": {"bbox": [x, y, w, h], "score": 0.95}}` ou `null` (pas de détection)

### Graphe de dépendances

```
Phase 1 (Données YOLO) ──→ Phase 2 (Entraînement YOLO) ──→ Phase 3 (Pipeline inférence) ──→ Phase 4 (Webcam)
                                                         └──→ Phase 5 (Export YOLO Hailo)
```

Les tests TDD sont écrits **avant** le code dans chaque phase.

---

## 1. Phase 1 — Préparation des données YOLO

### Objectif

Convertir les `annotations.json` COCO en fichiers `.txt` au format YOLO normalisé, prêts pour `yolo detect train`.

### Fichiers à créer

| Fichier | Rôle |
|---|---|
| `tests/test_prepare_yolo_dataset.py` | Tests TDD (écrits en premier) |
| `prepare_yolo_dataset.py` | Script de conversion |

### Conversion COCO → YOLO

Format COCO (actuel) : `[x, y, w, h]` en pixels absolus (coin supérieur-gauche + dimensions)

Format YOLO (cible) : `class cx cy nw nh` normalisé [0, 1] (centre + dimensions)

```
cx = (x + w/2) / img_w
cy = (y + h/2) / img_h
nw = w / img_w
nh = h / img_h
```

Les dimensions de l'image sont lues via `PIL.Image.open(path).size` — `annotations.json` ne les stocke pas.

Classe unique : `0` = bird (détecteur binaire, toutes espèces confondues).

### Tests TDD

```python
class TestCocoToYoloConversion:
    def test_converts_bbox_to_normalized_center(self, tmp_path):
        # COCO [100, 50, 200, 300] sur image 640×480
        # → YOLO "0 0.3125 0.4167 0.3125 0.6250"

    def test_class_is_always_zero_for_binary_detector(self, tmp_path):
        # Toutes les espèces → classe 0

    def test_skips_null_annotations(self, tmp_path):
        # annotations.json avec valeur null → pas de .txt créé

    def test_output_file_per_image(self, tmp_path):
        # image.jpg → image.txt (même stem, extension .txt)

    def test_coordinates_between_zero_and_one(self, tmp_path):
        # Toutes les valeurs de sortie ∈ [0.0, 1.0]


class TestPrepareYoloDataset:
    def test_creates_dataset_yaml(self, tmp_path):
        # Fichier dataset.yaml créé à la racine du dataset YOLO

    def test_dataset_yaml_has_required_keys(self, tmp_path):
        # Clés obligatoires : path, train, val, nc, names

    def test_nc_equals_one_and_names_bird(self, tmp_path):
        # nc: 1, names: ['bird']

    def test_preserves_existing_splits(self, tmp_path):
        # train/, validation/ (→ val/), test/ conservés

    def test_creates_labels_directory_parallel_to_images(self, tmp_path):
        # dataset/yolo_bird/train/labels/*.txt côte à côte avec images/

    def test_uses_symlinks_for_images(self, tmp_path):
        # Pas de copie — symlinks vers les originaux

    def test_stats_report(self, tmp_path):
        # Affiche le nombre d'images converties, skippées, sans annotation
```

### Fonctions clés

```python
def coco_bbox_to_yolo(bbox: list[int], img_w: int, img_h: int) -> str:
    """Convertit COCO [x, y, w, h] en YOLO normalisé '0 cx cy nw nh'."""

def convert_species_annotations(species_dir: Path, output_labels_dir: Path,
                                 output_images_dir: Path) -> dict:
    """Convertit un dossier espèce. Retourne {converted: int, skipped: int}."""

def prepare_dataset(dataset_dir: Path, output_dir: Path) -> None:
    """Point d'entrée : itère train/val/test, convertit tout, crée dataset.yaml."""

def create_dataset_yaml(output_dir: Path, nc: int = 1) -> Path:
    """Crée le fichier dataset.yaml pour Ultralytics."""
```

### Structure de sortie

```
dataset/yolo_bird/
├── dataset.yaml          # path, train, val, nc=1, names=['bird']
├── train/
│   ├── images/           # symlinks → dataset/europe/train/*/image.jpg
│   └── labels/           # fichiers .txt (un par image)
├── val/                  # mappé depuis validation/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

**Note** : Ultralytics attend `val/` et non `validation/`. Le script doit mapper le nom.

### Commande

```bash
python prepare_yolo_dataset.py --dataset dataset/europe --output dataset/yolo_bird
```

---

## 2. Phase 2 — Entraînement YOLO11s (1 classe "bird")

### Objectif

Fine-tuner YOLO11s depuis les poids COCO pretrained sur 1 classe "bird", en utilisant les ~195 726 images annotées.

### Approche

Pas de script custom — utiliser directement le CLI Ultralytics qui gère l'entraînement, la validation, le logging et le early stopping.

### Commande d'entraînement

```bash
yolo detect train \
    model=yolo11s.pt \
    data=dataset/yolo_bird/dataset.yaml \
    epochs=50 \
    imgsz=640 \
    batch=32 \
    patience=10 \
    freeze=10 \
    project=output/yolo_bird \
    name=yolo11s_bird \
    exist_ok=True
```

### Hyperparamètres

| Paramètre | Valeur | Justification |
|---|---|---|
| `model` | `yolo11s.pt` | 9.4M params, bon compromis taille/performance |
| `epochs` | 50 | COCO pretrained connaît déjà "bird" — convergence rapide |
| `imgsz` | 640 | Standard YOLO |
| `batch` | 32 | Ajuster selon VRAM GPU |
| `patience` | 10 | Early stopping si pas d'amélioration |
| `freeze` | 10 | Geler les 10 premières couches (transfer learning) |

### Performance attendue

mAP@0.5 > **0.90** — le modèle COCO pretrained détecte déjà les oiseaux (classe 14). Le fine-tuning spécialise sur nos images iNaturalist.

### Tests TDD

```python
class TestYoloDatasetYaml:
    def test_yaml_loads_and_has_required_keys(self, yolo_dataset):
        # path, train, val, nc, names

    def test_nc_equals_one(self, yolo_dataset):

    def test_names_contains_bird(self, yolo_dataset):

    def test_train_and_val_paths_exist(self, yolo_dataset):


@pytest.mark.slow
class TestYoloTrainingSmoke:
    def test_yolo11s_trains_one_epoch(self, yolo_mini_dataset):
        # from ultralytics import YOLO
        # model = YOLO("yolo11s.pt")
        # results = model.train(data="dataset.yaml", epochs=1, imgsz=640)

    def test_training_produces_best_pt(self, yolo_mini_dataset):
        # Vérifie que weights/best.pt existe après entraînement

    def test_best_pt_loadable(self, yolo_mini_dataset):
        # model = YOLO("best.pt") → charge sans erreur


@pytest.mark.slow
class TestYoloValidation:
    def test_map50_above_threshold(self, trained_yolo_model):
        # Après entraînement, mAP@0.5 ≥ 0.85

    def test_detects_birds_in_fixture_images(self, trained_yolo_model):
        # Inférence sur des images connues → au moins 1 détection
```

### Sortie

`output/yolo_bird/yolo11s_bird/weights/best.pt`

---

## 3. Phase 3 — Pipeline d'inférence two-stage

### Objectif

Créer un pipeline d'inférence unifié qui chaîne YOLO (détection) + classificateur (identification d'espèce) et retourne espèce + bounding box pour chaque oiseau détecté.

### Fichiers à créer

| Fichier | Rôle |
|---|---|
| `tests/test_inference.py` | Tests TDD (écrits en premier) |
| `inference.py` | Pipeline two-stage |

### Architecture

```python
from dataclasses import dataclass

@dataclass
class Detection:
    bbox: list[float]       # [x1, y1, x2, y2] en pixels
    confidence: float

@dataclass
class Prediction:
    species: str            # slug (ex: "parus_major")
    score: float

@dataclass
class BirdPrediction:
    bbox: list[float]       # [x1, y1, x2, y2] en pixels
    detection_confidence: float
    species: str
    species_score: float
    top_k: list[Prediction]


class BirdDetector:
    """Détecteur d'oiseaux basé sur YOLO. Supporte .pt (PyTorch) et .onnx."""

    def __init__(self, model_path: str, conf_threshold: float = 0.5,
                 device: str | None = None):
        ...

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Détecte les oiseaux dans une image.
        Retourne une liste de Detection (bbox + confidence)."""
        ...


class BirdClassifier:
    """Classificateur d'espèces. Charge un checkpoint train.py."""

    def __init__(self, model_path: str, label_map_path: str,
                 device: str | None = None):
        ...

    def classify(self, crop: np.ndarray, top_k: int = 5) -> list[Prediction]:
        """Classifie une image croppée d'oiseau.
        Retourne le top-k des espèces prédites."""
        ...


class TwoStagePipeline:
    """Chaîne détection + classification."""

    def __init__(self, detector: BirdDetector, classifier: BirdClassifier,
                 min_crop_size: int = 20):
        ...

    def predict(self, image: np.ndarray) -> list[BirdPrediction]:
        """Pipeline complet : détecte les oiseaux, classifie chaque crop.
        Retourne une liste de BirdPrediction (bbox + espèce + scores)."""
        ...
```

### Flux d'exécution

```
Image (numpy BGR)
  │
  ▼
BirdDetector.detect(image)
  │ → [Detection(bbox=[120,50,380,340], conf=0.92), ...]
  │
  ▼  Pour chaque détection :
Crop : image[y1:y2, x1:x2]
  │ → Vérifier min_crop_size (skip si trop petit)
  │ → Clip aux bordures de l'image
  │
  ▼
BirdClassifier.classify(crop)
  │ → [Prediction("parus_major", 0.85), Prediction("parus_caeruleus", 0.08), ...]
  │
  ▼
BirdPrediction(bbox, detection_conf, species, species_score, top_k)
```

### Réutilisation du code existant

| Code existant | Fichier | Réutilisé dans |
|---|---|---|
| `get_inference_transform()` | `demo_webcam.py:101-107` | `BirdClassifier` — preprocessing du crop |
| `create_model()` | `demo_webcam.py:42-75` | `BirdClassifier` — instanciation de l'architecture |
| `_annotate_image_yolo()` | `auto_annotate.py:130-159` | Pattern pour `BirdDetector` (mais version simplifiée) |
| `IMAGENET_MEAN/STD` | `demo_webcam.py:25-26` | `BirdClassifier` — normalisation |

### Cas limites

| Cas | Comportement |
|---|---|
| 0 oiseaux détectés | `predict()` retourne `[]` |
| Plusieurs oiseaux | Chaque crop classifié indépendamment |
| Crop < `min_crop_size` px | Skip — pas de classification |
| Bbox déborde de l'image | Clip aux dimensions de l'image |
| Image corrompue | Retourne `[]` avec warning |
| Score classification très bas | Inclus dans les résultats (le caller décide du seuil) |

### Tests TDD

```python
class TestBirdDetector:
    def test_loads_yolo_model(self, yolo_weights_path):
    def test_detects_birds_returns_boxes(self, detector, bird_image):
    def test_no_bird_returns_empty(self, detector, empty_image):
    def test_boxes_have_xyxy_and_confidence(self, detector, bird_image):
    def test_confidence_threshold_filters(self, detector, bird_image):


class TestBirdClassifier:
    def test_loads_classifier_from_checkpoint(self, checkpoint_path):
    def test_classifies_crop_returns_species(self, classifier, bird_crop):
    def test_returns_top_k_predictions(self, classifier, bird_crop):
    def test_predictions_scores_sum_to_one(self, classifier, bird_crop):


class TestTwoStagePipeline:
    def test_pipeline_returns_list_of_predictions(self, pipeline, bird_image):
    def test_each_prediction_has_bbox_and_species(self, pipeline, bird_image):
    def test_zero_birds_returns_empty_list(self, pipeline, empty_image):
    def test_multiple_birds_returns_multiple_predictions(self, pipeline, multi_bird_image):
    def test_tiny_crop_skipped(self, pipeline):
    def test_crop_clipped_to_image_boundaries(self, pipeline):


class TestEdgeCases:
    def test_corrupted_image_returns_empty(self, pipeline):
    def test_very_large_image_handled(self, pipeline):
    def test_grayscale_image_converted(self, pipeline):


@pytest.mark.slow
class TestIntegrationPipeline:
    def test_end_to_end_with_real_classifier(self, pipeline, bird_image):
    def test_pipeline_output_format(self, pipeline, bird_image):
```

### Utilisation cible

```python
from inference import BirdDetector, BirdClassifier, TwoStagePipeline

detector = BirdDetector("output/yolo_bird/yolo11s_bird/weights/best.pt")
classifier = BirdClassifier("output/best_vit_b_16.pth", "dataset/europe/label_map.json")
pipeline = TwoStagePipeline(detector, classifier)

results = pipeline.predict(image)
for bird in results:
    print(f"{bird.species} ({bird.species_score:.0%}) at {bird.bbox}")
    # parus_major (85%) at [120, 50, 380, 340]
```

---

## 4. Phase 4 — Mise à jour demo_webcam.py

### Objectif

Ajouter un mode détection à `demo_webcam.py` qui affiche des bounding boxes vertes avec le nom de l'espèce au-dessus de chaque oiseau détecté.

### Fichiers à modifier/créer

| Fichier | Action |
|---|---|
| `tests/test_demo_webcam.py` | Nouveau — tests TDD |
| `demo_webcam.py` | Modifier — ajouter mode détection |

### Changements dans demo_webcam.py

1. **Nouveau flag CLI** : `--detection` active le mode two-stage, `--yolo-weights <path>` spécifie les poids YOLO
2. **Nouvelle fonction `draw_detections()`** :
   - Rectangle vert autour de chaque oiseau
   - Nom français de l'espèce + score (%) au-dessus de la box
   - Barre de confiance colorée (vert > 70%, orange > 40%, rouge sinon)
3. **Modification de `run_single()`** :
   - Si `--detection` : créer `TwoStagePipeline` et appeler `pipeline.predict(frame)`
   - Sinon : comportement actuel (classification full-frame) — **rétrocompatibilité préservée**

### Comparaison avant/après

| | Mode actuel (classification) | Mode détection (two-stage) |
|---|---|---|
| Entrée | Frame entière | Frame entière |
| Traitement | `predict(model, frame)` | `pipeline.predict(frame)` |
| Sortie visuelle | Top-5 espèces en overlay texte | Box verte + nom espèce par oiseau |
| Bounding box | ❌ | ✅ |
| Multi-oiseaux | ❌ (1 prédiction pour toute la frame) | ✅ (1 box + espèce par oiseau) |

### Tests TDD

```python
class TestDemoWebcamDetectionMode:
    def test_parse_args_detection_flag(self):
        # --detection active le mode two-stage

    def test_detection_mode_requires_yolo_weights(self):
        # --detection sans --yolo-weights → erreur

    def test_fallback_to_classification_without_flag(self):
        # Sans --detection → mode legacy préservé


class TestDrawDetections:
    def test_draws_bbox_rectangle(self, sample_frame):
        # Vérifie qu'un rectangle est dessiné sur le frame

    def test_draws_species_label(self, sample_frame):
        # Vérifie que le nom de l'espèce apparaît

    def test_no_detections_shows_message(self, sample_frame):
        # "Aucun oiseau détecté" affiché si liste vide

    def test_multiple_detections_drawn(self, sample_frame):
        # Plusieurs boxes + labels si plusieurs oiseaux
```

### Commande

```bash
# Mode legacy (inchangé)
python demo_webcam.py 3

# Mode détection (nouveau)
python demo_webcam.py 3 --detection --yolo-weights output/yolo_bird/yolo11s_bird/weights/best.pt
```

---

## 5. Phase 5 — Export YOLO pour Hailo

### Objectif

Ajouter l'export du détecteur YOLO au format ONNX compatible Hailo DFC dans `export.py`.

### Fichiers à modifier

| Fichier | Action |
|---|---|
| `tests/test_export.py` | Ajouter tests TDD pour l'export YOLO |
| `export.py` | Ajouter target `hailo-yolo` |

### Changements dans export.py

1. **Nouveau choix** dans `parse_args()` : `--target hailo-yolo`
2. **Nouvelle fonction** :

```python
def export_yolo_hailo(yolo_path: Path, output_path: Path, imgsz: int = 640) -> float:
    """Exporte un modèle YOLO en ONNX pour Hailo DFC.
    Utilise l'API Ultralytics : model.export(format='onnx', opset=13).
    Retourne la taille en Mo."""
```

3. **Instructions DFC** affichées après l'export (comme pour le target `hailo` existant) :

```
Pour compiler avec le Hailo DFC :
  hailo parser onnx yolo11s_bird_hailo.onnx
  hailo optimize --hw-arch hailo10h
  hailo compile
```

### Tests TDD

```python
class TestExportYoloOnnx:
    def test_creates_onnx_file(self, yolo_weights_path, tmp_path):
    def test_onnx_input_shape_640(self, yolo_onnx_path):
        # [1, 3, 640, 640]
    def test_opset_at_least_13(self, yolo_onnx_path):
    def test_onnx_passes_validation(self, yolo_onnx_path):
    def test_returns_size_mb(self, yolo_weights_path, tmp_path):


class TestExportYoloCLI:
    def test_target_hailo_yolo_accepted(self):
    def test_existing_targets_still_work(self):
```

### Commande

```bash
python export.py --checkpoint output/yolo_bird/yolo11s_bird/weights/best.pt --target hailo-yolo
```

---

## 6. Fixtures de test communes

### Nouvelles fixtures dans `tests/conftest.py`

```python
@pytest.fixture
def yolo_mini_dataset(tmp_path, mini_dataset):
    """Crée un mini dataset YOLO depuis le mini dataset de classification.
    Convertit les annotations.json en labels .txt YOLO normalisés."""
    ...

@pytest.fixture
def yolo_weights_path(tmp_path):
    """Chemin vers des poids YOLO (mock pour tests unitaires,
    réel pour @pytest.mark.slow)."""
    ...

@pytest.fixture
def detector(yolo_weights_path):
    """Instance BirdDetector pour les tests."""
    from inference import BirdDetector
    return BirdDetector(str(yolo_weights_path), device="cpu")

@pytest.fixture
def classifier(checkpoint_path):
    """Instance BirdClassifier pour les tests."""
    from inference import BirdClassifier
    return BirdClassifier(str(checkpoint_path), ...)

@pytest.fixture
def pipeline(detector, classifier):
    """Instance TwoStagePipeline pour les tests."""
    from inference import TwoStagePipeline
    return TwoStagePipeline(detector, classifier)

@pytest.fixture
def sample_frame():
    """Frame webcam simulée (640×480 RGB numpy array)."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
```

---

## 7. Tableau récapitulatif

| Phase | Fichiers créés/modifiés | Tests d'abord | Dépend de | Livrable |
|---|---|---|---|---|
| **1** | `prepare_yolo_dataset.py` | `test_prepare_yolo_dataset.py` | — | `dataset/yolo_bird/` |
| **2** | CLI Ultralytics uniquement | `test_train_yolo.py` | Phase 1 | `best.pt` |
| **3** | `inference.py` | `test_inference.py` | Phase 2 + classifier existant | `TwoStagePipeline` |
| **4** | `demo_webcam.py` (modif) | `test_demo_webcam.py` | Phase 3 | Webcam avec boxes |
| **5** | `export.py` (modif) | `test_export.py` (extension) | Phase 2 | `.onnx` pour DFC |
| — | `tests/conftest.py` (extension) | — | Toutes | Fixtures partagées |

---

## 8. Note IMX500 — Contrainte mono-modèle

L'IMX500 ne peut charger qu'un seul modèle en SRAM → le pipeline two-stage n'est **pas applicable**.

Trois options sont documentées dans `infrastructure-imx500.md` :

| Option | Modèle IMX500 | Précision estimée | FPS |
|---|---|---|---|
| **A — Classification pure** (recommandée) | EfficientNetV2-B2 | ~77 % | ~17 |
| B — YOLO all-in-one | YOLO11n 558 classes | 30-50 % | ~17 |
| C — YOLO + CPU | YOLO11n + MobileNetV2 CPU | ~62 % | ~1-2 |

L'option A est recommandée pour les caméras fixes (mangeoire, nichoir) où l'oiseau est naturellement cadré.

---

## 9. Références internes

| Document | Sections pertinentes |
|---|---|
| `comparaison-detection-approaches.md` | §8 Recommandation finale, §9 Plan (haut niveau) |
| `infrastructure-hailo.md` | §5 Pseudo-code inférence HailoRT, §7 TODO |
| `infrastructure-imx500.md` | §5 Options IMX500 |
| `validation-choix-ia.md` | §5 Two-stage vs all-in-one |
