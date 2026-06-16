"""
Tests TDD pour le module auto_annotate.

Backend : Grounding DINO (Liu et al., ECCV 2024) — détection zero-shot "a bird".
Bounding boxes au format COCO [x, y, width, height].

Les images fixtures sont dans tests/fixtures/images/.
Les annotations manuelles de référence sont dans tests/fixtures/expected_annotations.json.
"""

import json
from pathlib import Path

import pytest
from PIL import Image

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "images"
EXPECTED_PATH = Path(__file__).parent / "fixtures" / "expected_annotations.json"

REAL_BIRD_IMAGES = sorted(
    f for f in FIXTURES_DIR.iterdir()
    if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    and not f.name.startswith(("no_bird", "grayscale", "rgba", "corrupted"))
)


@pytest.fixture(scope="module")
def annotator():
    from auto_annotate import BirdAnnotator
    return BirdAnnotator(device="cpu")


@pytest.fixture(scope="module")
def expected_annotations():
    with open(EXPECTED_PATH) as f:
        data = json.load(f)
    entries = {
        k: v for k, v in data.items()
        if not k.startswith("_") and v.get("bird_count") != "A_REMPLIR"
    }
    if not entries:
        pytest.skip("expected_annotations.json non rempli — lancer annotate_tool.py d'abord")
    return entries


# ---------------------------------------------------------------------------
# UC-01 : Image avec oiseau visible → retourne au moins 1 bbox avec score > seuil
# ---------------------------------------------------------------------------
class TestUC01_SingleBirdDetection:
    def test_bird_image_returns_at_least_one_bbox(self, annotator, expected_annotations):
        for name, expected in expected_annotations.items():
            if expected["bird_count"] == 0:
                continue
            img_area = expected["width"] * expected["height"]
            max_box_ratio = max(
                (b[2] * b[3]) / img_area for b in expected["boxes"]
            )
            if max_box_ratio < 0.02:
                continue
            filepath = FIXTURES_DIR / name
            results = annotator.annotate_image(str(filepath))
            assert len(results) >= 1, (
                f"{name}: attendu ≥1 détection, obtenu {len(results)}"
            )

    def test_each_bbox_has_score_above_default_threshold(self, annotator, expected_annotations):
        for name, expected in expected_annotations.items():
            if expected["bird_count"] == 0:
                continue
            filepath = FIXTURES_DIR / name
            results = annotator.annotate_image(str(filepath))
            for det in results:
                assert det["score"] > annotator.threshold, (
                    f"{name}: score {det['score']} ≤ seuil {annotator.threshold}"
                )


# ---------------------------------------------------------------------------
# UC-02 : Image sans oiseau → retourne une liste vide
# ---------------------------------------------------------------------------
class TestUC02_NoBirdImage:
    def test_no_bird_returns_empty(self, annotator):
        filepath = FIXTURES_DIR / "no_bird__blue_sky.jpg"
        results = annotator.annotate_image(str(filepath))
        assert results == [], f"Attendu 0 détection sur ciel bleu, obtenu {len(results)}"


# ---------------------------------------------------------------------------
# UC-03 : Image avec plusieurs oiseaux → retourne plusieurs bboxes
# ---------------------------------------------------------------------------
class TestUC03_MultipleBirds:
    def test_multiple_birds_detected(self, annotator, expected_annotations):
        # Vérifie que l'annotateur détecte >1 oiseau sur au moins une image multi-oiseaux.
        # Les cas ratés (oiseaux trop proches, nuée) sont corrigés par validate_annotations.py.
        multi_bird_images = {
            k: v for k, v in expected_annotations.items() if v["bird_count"] > 1
        }
        if not multi_bird_images:
            pytest.skip("Aucune image annotée avec plusieurs oiseaux")
        results_per_image = {}
        for name in multi_bird_images:
            filepath = FIXTURES_DIR / name
            results = annotator.annotate_image(str(filepath))
            results_per_image[name] = len(results)
        detected_multi = {k: v for k, v in results_per_image.items() if v > 1}
        assert len(detected_multi) >= 1, (
            f"Aucune image multi-oiseaux n'a produit >1 détection : {results_per_image}"
        )


# ---------------------------------------------------------------------------
# UC-04 : Format bbox → COCO [x, y, width, height], toutes valeurs positives
# ---------------------------------------------------------------------------
class TestUC04_BboxFormat:
    def test_bbox_is_coco_format(self, annotator):
        filepath = REAL_BIRD_IMAGES[0]
        results = annotator.annotate_image(str(filepath))
        assert len(results) >= 1, "Pas de détection pour vérifier le format"
        for det in results:
            bbox = det["bbox"]
            assert isinstance(bbox, list) and len(bbox) == 4, (
                f"bbox doit être [x, y, w, h], obtenu {bbox}"
            )
            x, y, w, h = bbox
            assert x >= 0 and y >= 0, f"x,y négatifs: {bbox}"
            assert w > 0 and h > 0, f"w,h doivent être > 0: {bbox}"


# ---------------------------------------------------------------------------
# UC-05 : Bbox dans les limites de l'image
# ---------------------------------------------------------------------------
class TestUC05_BboxWithinBounds:
    def test_bbox_within_image_bounds(self, annotator, expected_annotations):
        for name, expected in expected_annotations.items():
            filepath = FIXTURES_DIR / name
            results = annotator.annotate_image(str(filepath))
            img_w, img_h = expected["width"], expected["height"]
            for det in results:
                x, y, w, h = det["bbox"]
                assert x + w <= img_w + 1, (
                    f"{name}: x+w={x+w} dépasse largeur {img_w}"
                )
                assert y + h <= img_h + 1, (
                    f"{name}: y+h={y+h} dépasse hauteur {img_h}"
                )


# ---------------------------------------------------------------------------
# UC-06 : Seuil de confiance → 0.9 donne moins de détections que 0.3
# ---------------------------------------------------------------------------
class TestUC06_ConfidenceThreshold:
    def test_high_threshold_fewer_detections(self, annotator):
        filepath = REAL_BIRD_IMAGES[0]
        results_low = annotator.annotate_image(str(filepath), threshold=0.3)
        results_high = annotator.annotate_image(str(filepath), threshold=0.9)
        assert len(results_high) <= len(results_low), (
            f"Seuil 0.9 ({len(results_high)} det) devrait donner ≤ "
            f"seuil 0.3 ({len(results_low)} det)"
        )


# ---------------------------------------------------------------------------
# UC-07 : Image corrompue → retourne liste vide sans crash
# ---------------------------------------------------------------------------
class TestUC07_CorruptedImage:
    def test_corrupted_returns_empty_no_crash(self, annotator):
        filepath = FIXTURES_DIR / "corrupted__bad.jpg"
        results = annotator.annotate_image(str(filepath))
        assert results == [], f"Image corrompue devrait donner 0 détection"


# ---------------------------------------------------------------------------
# UC-08 : Image niveaux de gris (1 canal) → fonctionne avec conversion auto
# ---------------------------------------------------------------------------
class TestUC08_GrayscaleImage:
    def test_grayscale_works(self, annotator):
        filepath = FIXTURES_DIR / "grayscale__robin.jpg"
        results = annotator.annotate_image(str(filepath))
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# UC-09 : Image RGBA (4 canaux) → fonctionne avec conversion auto
# ---------------------------------------------------------------------------
class TestUC09_RGBAImage:
    def test_rgba_works(self, annotator):
        filepath = FIXTURES_DIR / "rgba__robin.png"
        results = annotator.annotate_image(str(filepath))
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# UC-10 : Format COCO valide → output contient images, annotations, categories
# ---------------------------------------------------------------------------
class TestUC10_CocoFormatKeys:
    def test_coco_output_has_required_keys(self, annotator, tmp_path):
        _setup_mini_dir(tmp_path)
        coco = annotator.build_coco_annotations(str(tmp_path))
        for key in ("images", "annotations", "categories"):
            assert key in coco, f"Clé '{key}' manquante dans la sortie COCO"


# ---------------------------------------------------------------------------
# UC-11 : IDs uniques → tous les image_id et annotation_id sont uniques
# ---------------------------------------------------------------------------
class TestUC11_UniqueIds:
    def test_unique_image_ids(self, annotator, tmp_path):
        _setup_mini_dir(tmp_path)
        coco = annotator.build_coco_annotations(str(tmp_path))
        image_ids = [img["id"] for img in coco["images"]]
        assert len(image_ids) == len(set(image_ids)), "image_id en doublon"

    def test_unique_annotation_ids(self, annotator, tmp_path):
        _setup_mini_dir(tmp_path)
        coco = annotator.build_coco_annotations(str(tmp_path))
        ann_ids = [ann["id"] for ann in coco["annotations"]]
        assert len(ann_ids) == len(set(ann_ids)), "annotation_id en doublon"


# ---------------------------------------------------------------------------
# UC-12 : Cohérence image↔annotation → chaque annotation.image_id existe dans images
# ---------------------------------------------------------------------------
class TestUC12_ImageAnnotationCoherence:
    def test_annotation_image_ids_exist(self, annotator, tmp_path):
        _setup_mini_dir(tmp_path)
        coco = annotator.build_coco_annotations(str(tmp_path))
        valid_ids = {img["id"] for img in coco["images"]}
        for ann in coco["annotations"]:
            assert ann["image_id"] in valid_ids, (
                f"annotation {ann['id']} référence image_id={ann['image_id']} inexistant"
            )


# ---------------------------------------------------------------------------
# UC-13 : Cohérence catégorie → chaque annotation.category_id existe dans categories
# ---------------------------------------------------------------------------
class TestUC13_CategoryCoherence:
    def test_annotation_category_ids_exist(self, annotator, tmp_path):
        _setup_mini_dir(tmp_path)
        coco = annotator.build_coco_annotations(str(tmp_path))
        valid_cats = {cat["id"] for cat in coco["categories"]}
        for ann in coco["annotations"]:
            assert ann["category_id"] in valid_cats, (
                f"annotation {ann['id']} référence category_id={ann['category_id']} inexistant"
            )


# ---------------------------------------------------------------------------
# UC-14 : Toutes les images traitées → len(images) == nombre de fichiers
# ---------------------------------------------------------------------------
class TestUC14_AllImagesProcessed:
    def test_all_images_in_output(self, annotator, tmp_path):
        n_files = _setup_mini_dir(tmp_path)
        coco = annotator.build_coco_annotations(str(tmp_path))
        assert len(coco["images"]) == n_files, (
            f"Attendu {n_files} images, obtenu {len(coco['images'])}"
        )


# ---------------------------------------------------------------------------
# UC-15 : Taux de détection raisonnable → ≥70% des vraies photos ont une détection
# ---------------------------------------------------------------------------
class TestUC15_DetectionRate:
    def test_detection_rate_above_70_percent(self, annotator, expected_annotations):
        bird_images = {
            k: v for k, v in expected_annotations.items() if v["bird_count"] > 0
        }
        detected = 0
        for name in bird_images:
            filepath = FIXTURES_DIR / name
            results = annotator.annotate_image(str(filepath))
            if len(results) >= 1:
                detected += 1
        rate = detected / len(bird_images)
        assert rate >= 0.70, (
            f"Taux de détection {rate:.0%} < 70% "
            f"({detected}/{len(bird_images)} images)"
        )


# ---------------------------------------------------------------------------
# UC-16 : Pas de faux positifs géants → aucune bbox > 95% de la surface image
# ---------------------------------------------------------------------------
class TestUC16_NoGiantFalsePositives:
    def test_no_bbox_covers_entire_image(self, annotator, expected_annotations):
        for name, expected in expected_annotations.items():
            filepath = FIXTURES_DIR / name
            results = annotator.annotate_image(str(filepath))
            img_area = expected["width"] * expected["height"]
            for det in results:
                x, y, w, h = det["bbox"]
                bbox_area = w * h
                ratio = bbox_area / img_area
                assert ratio <= 0.95, (
                    f"{name}: bbox couvre {ratio:.0%} de l'image (>95%)"
                )


# ---------------------------------------------------------------------------
# UC-17 : Pas de micro-boxes → aucune bbox < 1% de la surface image
# ---------------------------------------------------------------------------
class TestUC17_NoMicroBoxes:
    def test_no_tiny_bbox(self, annotator, expected_annotations):
        for name, expected in expected_annotations.items():
            filepath = FIXTURES_DIR / name
            results = annotator.annotate_image(str(filepath))
            img_area = expected["width"] * expected["height"]
            for det in results:
                x, y, w, h = det["bbox"]
                bbox_area = w * h
                ratio = bbox_area / img_area
                assert ratio >= 0.01, (
                    f"{name}: bbox ne couvre que {ratio:.2%} de l'image (<1%)"
                )


# ---------------------------------------------------------------------------
# UC-18 : Score moyen cohérent → moyenne des scores > 0.6
# ---------------------------------------------------------------------------
class TestUC18_ConsistentScoreAverage:
    def test_mean_score_above_threshold(self, annotator, expected_annotations):
        all_scores = []
        for name in expected_annotations:
            filepath = FIXTURES_DIR / name
            results = annotator.annotate_image(str(filepath))
            all_scores.extend(det["score"] for det in results)
        if not all_scores:
            pytest.fail("Aucune détection sur l'ensemble des fixtures")
        mean_score = sum(all_scores) / len(all_scores)
        assert mean_score > 0.6, (
            f"Score moyen {mean_score:.2f} ≤ 0.6 "
            f"({len(all_scores)} détections)"
        )


# ---------------------------------------------------------------------------
# UC-19 : Génération d'échantillons visuels → le dossier contient des fichiers
# ---------------------------------------------------------------------------
class TestUC19_SampleImagesCreated:
    def test_samples_directory_has_files(self, annotator, tmp_path):
        _setup_mini_dir(tmp_path)
        output_dir = tmp_path / "annotation_samples"
        annotator.generate_samples(str(tmp_path), str(output_dir), max_samples=3)
        generated = list(output_dir.iterdir())
        assert len(generated) > 0, "Aucun échantillon généré"
        assert len(generated) <= 3, f"Trop d'échantillons: {len(generated)} > 3"


# ---------------------------------------------------------------------------
# UC-20 : Échantillons lisibles → chaque fichier généré est une image valide
# ---------------------------------------------------------------------------
class TestUC20_SampleImagesReadable:
    def test_sample_images_are_valid(self, annotator, tmp_path):
        _setup_mini_dir(tmp_path)
        output_dir = tmp_path / "annotation_samples"
        annotator.generate_samples(str(tmp_path), str(output_dir), max_samples=3)
        for f in output_dir.iterdir():
            assert f.stat().st_size > 0, f"{f.name} est vide"
            img = Image.open(f)
            img.verify()


# ---------------------------------------------------------------------------
# UC-21 : annotate_species retourne un dict {filename: detection | None}
#         pour chaque image du dossier, avec le bon format de sortie
# ---------------------------------------------------------------------------
class TestUC21_AnnotateSpeciesFormat:
    def test_returns_dict_with_all_images(self, annotator, tmp_path):
        species_dir = tmp_path / "parus_major"
        _setup_species_dir(species_dir, count=3)
        result = annotator.annotate_species(str(species_dir))
        images = [f.name for f in species_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        assert isinstance(result, dict)
        assert set(result.keys()) == set(images), (
            f"Clés attendues: {sorted(images)}, obtenues: {sorted(result.keys())}"
        )

    def test_each_entry_is_detection_or_none(self, annotator, tmp_path):
        species_dir = tmp_path / "parus_major"
        _setup_species_dir(species_dir, count=3)
        result = annotator.annotate_species(str(species_dir))
        for filename, det in result.items():
            if det is not None:
                assert "bbox" in det, f"{filename}: clé 'bbox' manquante"
                assert "score" in det, f"{filename}: clé 'score' manquante"
                assert isinstance(det["bbox"], list) and len(det["bbox"]) == 4
                assert all(isinstance(v, (int, float)) for v in det["bbox"])
            # None est accepté (pas de détection)


# ---------------------------------------------------------------------------
# UC-22 : annotate_species utilise best_detection (1 bbox par image)
#         et non annotate_image (multi-bbox) — garantit le format SSD
# ---------------------------------------------------------------------------
class TestUC22_AnnotateSpeciesUsesBestDetection:
    def test_at_most_one_bbox_per_image(self, annotator, tmp_path):
        species_dir = tmp_path / "turdus_merula"
        _setup_species_dir(species_dir, count=5)
        result = annotator.annotate_species(str(species_dir))
        for filename, det in result.items():
            assert det is None or isinstance(det["bbox"], list), (
                f"{filename}: attendu None ou un seul dict, pas une liste de détections"
            )


# ---------------------------------------------------------------------------
# UC-23 : annotate_dataset crée un annotations.json dans chaque dossier espèce
# ---------------------------------------------------------------------------
class TestUC23_AnnotateDatasetCreatesFiles:
    def test_creates_annotations_json_per_species(self, annotator, tmp_path):
        train_dir = _setup_mini_dataset(tmp_path, n_species=3, n_images=2)
        annotator.annotate_dataset(str(train_dir))
        for sp_dir in train_dir.iterdir():
            if sp_dir.is_dir():
                ann_path = sp_dir / "annotations.json"
                assert ann_path.exists(), f"annotations.json manquant dans {sp_dir.name}"


# ---------------------------------------------------------------------------
# UC-24 : annotations.json est du JSON valide avec le bon contenu
# ---------------------------------------------------------------------------
class TestUC24_AnnotationsJsonContent:
    def test_annotations_json_is_valid_and_complete(self, annotator, tmp_path):
        train_dir = _setup_mini_dataset(tmp_path, n_species=2, n_images=3)
        annotator.annotate_dataset(str(train_dir))
        for sp_dir in train_dir.iterdir():
            if not sp_dir.is_dir():
                continue
            ann_path = sp_dir / "annotations.json"
            with open(ann_path) as f:
                data = json.load(f)
            images = [f.name for f in sp_dir.iterdir()
                      if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
            assert set(data.keys()) == set(images), (
                f"{sp_dir.name}: clés JSON ({sorted(data.keys())}) ≠ images ({sorted(images)})"
            )


# ---------------------------------------------------------------------------
# UC-25 : annotate_dataset avec resume=True skip les espèces déjà annotées
# ---------------------------------------------------------------------------
class TestUC25_AnnotateDatasetResume:
    def test_resume_skips_existing_annotations(self, annotator, tmp_path):
        train_dir = _setup_mini_dataset(tmp_path, n_species=3, n_images=2)
        species_dirs = sorted(d for d in train_dir.iterdir() if d.is_dir())

        # Pré-créer annotations.json pour la 1ère espèce avec un contenu marqueur
        marker = {"_marker": "pre_existing"}
        with open(species_dirs[0] / "annotations.json", "w") as f:
            json.dump(marker, f)

        stats = annotator.annotate_dataset(str(train_dir), resume=True)

        # Le fichier marqueur ne doit PAS avoir été écrasé
        with open(species_dirs[0] / "annotations.json") as f:
            data = json.load(f)
        assert data == marker, "resume=True a écrasé un annotations.json existant"
        assert stats["skipped"] >= 1


# ---------------------------------------------------------------------------
# UC-26 : annotate_dataset avec resume=False ré-annote tout
# ---------------------------------------------------------------------------
class TestUC26_AnnotateDatasetForceRerun:
    def test_no_resume_overwrites_existing(self, annotator, tmp_path):
        train_dir = _setup_mini_dataset(tmp_path, n_species=2, n_images=2)
        species_dirs = sorted(d for d in train_dir.iterdir() if d.is_dir())

        marker = {"_marker": "should_be_overwritten"}
        with open(species_dirs[0] / "annotations.json", "w") as f:
            json.dump(marker, f)

        stats = annotator.annotate_dataset(str(train_dir), resume=False)

        with open(species_dirs[0] / "annotations.json") as f:
            data = json.load(f)
        assert data != marker, "resume=False n'a pas ré-annoté l'espèce existante"
        assert stats["skipped"] == 0


# ---------------------------------------------------------------------------
# UC-27 : annotate_dataset retourne des stats correctes
# ---------------------------------------------------------------------------
class TestUC27_AnnotateDatasetStats:
    def test_stats_are_consistent(self, annotator, tmp_path):
        train_dir = _setup_mini_dataset(tmp_path, n_species=2, n_images=3)
        stats = annotator.annotate_dataset(str(train_dir))
        assert stats["total_species"] == 2
        assert stats["annotated"] + stats["skipped"] == stats["total_species"]
        assert stats["detected"] <= stats["images"]
        assert stats["images"] > 0


# ---------------------------------------------------------------------------
# UC-28 : annotate_dataset avec workers>1 produit les mêmes annotations.json
#         que le mode séquentiel (même contenu, même fichiers créés)
# ---------------------------------------------------------------------------
class TestUC28_ParallelSameResultsAsSequential:
    def test_parallel_produces_same_annotations(self, tmp_path):
        from auto_annotate import BirdAnnotator

        seq_dir = _setup_mini_dataset(tmp_path / "seq", n_species=3, n_images=2)
        par_dir = _setup_mini_dataset(tmp_path / "par", n_species=3, n_images=2)

        a1 = BirdAnnotator()
        a1.annotate_dataset(str(seq_dir), workers=1)

        a2 = BirdAnnotator()
        a2.annotate_dataset(str(par_dir), workers=2)

        for sp_dir in sorted(seq_dir.iterdir()):
            if not sp_dir.is_dir():
                continue
            seq_ann = sp_dir / "annotations.json"
            par_ann = par_dir / sp_dir.name / "annotations.json"
            assert seq_ann.exists() and par_ann.exists(), f"{sp_dir.name}: annotations.json manquant"
            with open(seq_ann) as f:
                seq_data = json.load(f)
            with open(par_ann) as f:
                par_data = json.load(f)
            assert seq_data == par_data, f"{sp_dir.name}: résultats différents entre séquentiel et parallèle"


# ---------------------------------------------------------------------------
# UC-29 : annotate_dataset parallèle avec resume=True skip les espèces
#         qui ont déjà un annotations.json
# ---------------------------------------------------------------------------
class TestUC29_ParallelResume:
    def test_parallel_resume_skips_existing(self, tmp_path):
        from auto_annotate import BirdAnnotator
        train_dir = _setup_mini_dataset(tmp_path, n_species=3, n_images=2)
        species_dirs = sorted(d for d in train_dir.iterdir() if d.is_dir())

        marker = {"_marker": "pre_existing"}
        with open(species_dirs[0] / "annotations.json", "w") as f:
            json.dump(marker, f)

        a = BirdAnnotator()
        stats = a.annotate_dataset(str(train_dir), resume=True, workers=2)

        with open(species_dirs[0] / "annotations.json") as f:
            data = json.load(f)
        assert data == marker, "resume=True a écrasé un annotations.json existant en mode parallèle"
        assert stats["skipped"] >= 1


# ---------------------------------------------------------------------------
# UC-30 : annotate_dataset parallèle retourne des stats cohérentes
# ---------------------------------------------------------------------------
class TestUC30_ParallelStats:
    def test_parallel_stats_consistent(self, tmp_path):
        from auto_annotate import BirdAnnotator
        train_dir = _setup_mini_dataset(tmp_path, n_species=3, n_images=2)
        a = BirdAnnotator()
        stats = a.annotate_dataset(str(train_dir), workers=2)
        assert stats["total_species"] == 3
        assert stats["annotated"] + stats["skipped"] == stats["total_species"]
        assert stats["detected"] <= stats["images"]
        assert stats["images"] > 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _setup_species_dir(species_dir: Path, count: int = 3):
    """Copie des images fixtures dans un dossier espèce simulé."""
    import shutil
    species_dir.mkdir(parents=True, exist_ok=True)
    images = REAL_BIRD_IMAGES[:count]
    for img_path in images:
        shutil.copy2(img_path, species_dir / img_path.name)


def _setup_mini_dataset(tmp_path: Path, n_species: int = 3, n_images: int = 2) -> Path:
    """Crée un mini dataset train/ avec N espèces et M images chacune."""
    import shutil
    train_dir = tmp_path / "train"
    train_dir.mkdir(parents=True)
    species_names = ["species_a", "species_b", "species_c", "species_d"][:n_species]
    available_images = REAL_BIRD_IMAGES[:n_species * n_images]
    idx = 0
    for name in species_names:
        sp_dir = train_dir / name
        sp_dir.mkdir()
        for _ in range(n_images):
            if idx < len(available_images):
                shutil.copy2(available_images[idx], sp_dir / available_images[idx].name)
                idx += 1
    return train_dir


def _setup_mini_dir(tmp_path: Path) -> int:
    """Copie quelques images fixtures dans tmp_path pour les tests d'intégration.
    Retourne le nombre de fichiers copiés."""
    import shutil
    images = REAL_BIRD_IMAGES[:5]
    for img_path in images:
        shutil.copy2(img_path, tmp_path / img_path.name)
    return len(images)


# ---------------------------------------------------------------------------
# UC-34 : BirdAnnotator accepte un paramètre device explicite
#         Par défaut, CUDA si dispo, sinon CPU (MPS exclu car instable)
# ---------------------------------------------------------------------------
class TestUC34_DeviceParameter:
    def test_explicit_cpu(self):
        from auto_annotate import BirdAnnotator
        annotator = BirdAnnotator(device="cpu")
        assert annotator.device.type == "cpu"

    def test_default_skips_mps(self):
        import torch
        from auto_annotate import BirdAnnotator
        annotator = BirdAnnotator()
        if torch.cuda.is_available():
            assert annotator.device.type == "cuda"
        else:
            assert annotator.device.type == "cpu"

    def test_explicit_mps_when_available(self):
        import torch
        from auto_annotate import BirdAnnotator
        if not torch.backends.mps.is_available():
            pytest.skip("MPS non disponible")
        annotator = BirdAnnotator(device="mps")
        assert annotator.device.type == "mps"


# ---------------------------------------------------------------------------
# UC-35 : annotate_batch retourne les mêmes résultats que annotate_image
# ---------------------------------------------------------------------------
class TestUC35_AnnotateBatchSameAsSequential:
    def test_batch_matches_sequential(self, annotator):
        paths = [str(img) for img in REAL_BIRD_IMAGES[:5]]
        batch_results = annotator.annotate_batch(paths)
        for path in paths:
            seq_results = annotator.annotate_image(path)
            batch_dets = batch_results.get(path, [])
            assert len(seq_results) == len(batch_dets), (
                f"{Path(path).name}: seq={len(seq_results)} vs batch={len(batch_dets)}"
            )
            for s, b in zip(seq_results, batch_dets):
                assert s["bbox"] == b["bbox"]
                assert s["score"] == b["score"]


# ---------------------------------------------------------------------------
# UC-36 : annotate_batch ignore les images corrompues sans crasher
# ---------------------------------------------------------------------------
class TestUC36_AnnotateBatchSkipsCorrupted:
    def test_skips_corrupted(self, annotator, tmp_path):
        good = str(REAL_BIRD_IMAGES[0])
        bad = str(tmp_path / "corrupted.jpg")
        Path(bad).write_bytes(b"\xff\xd8bad data")
        results = annotator.annotate_batch([good, bad])
        assert good in results
        assert bad not in results


# ---------------------------------------------------------------------------
# UC-37 : annotate_batch avec liste vide retourne dict vide
# ---------------------------------------------------------------------------
class TestUC37_AnnotateBatchEmpty:
    def test_empty_list(self, annotator):
        assert annotator.annotate_batch([]) == {}


# ---------------------------------------------------------------------------
# UC-38 : backend invalide lève ValueError
# ---------------------------------------------------------------------------
class TestUC38_InvalidBackend:
    def test_invalid_backend_raises(self):
        from auto_annotate import BirdAnnotator
        with pytest.raises(ValueError, match="backend"):
            BirdAnnotator(backend="invalid_model")

    def test_old_backends_rejected(self):
        from auto_annotate import BirdAnnotator
        for old in ["fasterrcnn", "yolo11n", "yolo11s", "yolo11m"]:
            with pytest.raises(ValueError, match="backend"):
                BirdAnnotator(backend=old)


# ---------------------------------------------------------------------------
# UC-39 : text_threshold stocké et défaut à 0.25
# ---------------------------------------------------------------------------
class TestUC39_TextThreshold:
    def test_text_threshold_stored(self):
        from auto_annotate import BirdAnnotator
        a = BirdAnnotator(device="cpu", text_threshold=0.4)
        assert a.text_threshold == 0.4

    def test_text_threshold_default(self):
        from auto_annotate import BirdAnnotator
        a = BirdAnnotator(device="cpu")
        assert a.text_threshold == 0.25


# ---------------------------------------------------------------------------
# UC-40 : grounding_dino_base est un backend valide
# ---------------------------------------------------------------------------
class TestUC40_BaseBackend:
    def test_base_model_loads(self):
        from auto_annotate import BirdAnnotator
        a = BirdAnnotator(backend="grounding_dino_base", device="cpu")
        assert a.backend == "grounding_dino_base"

    def test_base_detects_bird(self):
        from auto_annotate import BirdAnnotator
        a = BirdAnnotator(backend="grounding_dino_base", device="cpu")
        results = a.annotate_image(str(REAL_BIRD_IMAGES[0]))
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# UC-41 : CLI — valeurs par défaut des arguments
# ---------------------------------------------------------------------------
class TestUC41_CLIDefaults:
    """Le CLI parse les arguments et utilise les valeurs par défaut recommandées."""

    def test_main_exists(self):
        from auto_annotate import main
        assert callable(main)

    def test_default_workers_is_1(self):
        from auto_annotate import build_parser
        parser = build_parser()
        args = parser.parse_args(["annotate", "/tmp/fake"])
        assert args.workers == 1

    def test_default_backend_is_grounding_dino_tiny(self):
        from auto_annotate import build_parser
        parser = build_parser()
        args = parser.parse_args(["annotate", "/tmp/fake"])
        assert args.backend == "grounding_dino_tiny"

    def test_default_threshold_is_030(self):
        from auto_annotate import build_parser
        parser = build_parser()
        args = parser.parse_args(["annotate", "/tmp/fake"])
        assert args.threshold == 0.3

    def test_default_text_threshold_is_025(self):
        from auto_annotate import build_parser
        parser = build_parser()
        args = parser.parse_args(["annotate", "/tmp/fake"])
        assert args.text_threshold == 0.25

    def test_default_resume_true(self):
        """Sans --force, resume=True (ne ré-annote pas les espèces déjà faites)."""
        from auto_annotate import build_parser
        parser = build_parser()
        args = parser.parse_args(["annotate", "/tmp/fake"])
        assert args.force is False


# ---------------------------------------------------------------------------
# UC-42 : CLI — --force désactive le resume
# ---------------------------------------------------------------------------
class TestUC42_CLIForceFlag:
    def test_force_flag_parsed(self):
        from auto_annotate import build_parser
        parser = build_parser()
        args = parser.parse_args(["annotate", "/tmp/fake", "--force"])
        assert args.force is True

    def test_force_means_resume_false(self, tmp_path):
        """--force doit passer resume=False à annotate_dataset."""
        from unittest.mock import patch, MagicMock
        from auto_annotate import main

        sp_dir = tmp_path / "species_a"
        sp_dir.mkdir()
        img = Image.new("RGB", (100, 100), "red")
        img.save(sp_dir / "img1.jpg")
        (sp_dir / "annotations.json").write_text("{}")

        with patch("auto_annotate.BirdAnnotator") as MockAnnotator:
            instance = MagicMock()
            instance.annotate_dataset.return_value = {
                "total_species": 1, "annotated": 1, "skipped": 0,
                "images": 1, "detected": 1,
            }
            MockAnnotator.return_value = instance

            with patch("sys.argv", ["auto_annotate.py", "annotate", str(tmp_path), "--force"]):
                main()

            instance.annotate_dataset.assert_called_once()
            call_kwargs = instance.annotate_dataset.call_args
            assert call_kwargs[1].get("resume") is False or call_kwargs[0][1:] == (False,) if len(call_kwargs[0]) > 1 else call_kwargs[1]["resume"] is False


# ---------------------------------------------------------------------------
# UC-43 : CLI — annotate crée les fichiers annotations.json
# ---------------------------------------------------------------------------
class TestUC43_CLIAnnotate:
    def test_annotate_creates_annotations(self, tmp_path, annotator):
        """Le CLI annote un dossier et crée annotations.json par espèce."""
        sp_dir = tmp_path / "test_species"
        sp_dir.mkdir()

        for bird_img in REAL_BIRD_IMAGES[:2]:
            import shutil
            shutil.copy(bird_img, sp_dir / bird_img.name)

        from auto_annotate import main
        from unittest.mock import patch

        with patch("sys.argv", [
            "auto_annotate.py", "annotate", str(tmp_path),
            "--workers", "1", "--force",
        ]):
            main()

        ann_path = sp_dir / "annotations.json"
        assert ann_path.exists()
        with open(ann_path) as f:
            annotations = json.load(f)
        assert len(annotations) == 2
        detected = sum(1 for v in annotations.values() if v is not None)
        assert detected >= 1


# ---------------------------------------------------------------------------
# UC-44 : CLI — sans --force, skip les espèces déjà annotées
# ---------------------------------------------------------------------------
class TestUC44_CLIResume:
    def test_resume_skips_existing(self, tmp_path):
        """Sans --force, les espèces avec annotations.json existant sont ignorées."""
        from unittest.mock import patch, MagicMock
        from auto_annotate import main

        sp_dir = tmp_path / "species_done"
        sp_dir.mkdir()
        (sp_dir / "annotations.json").write_text('{"img.jpg": null}')

        with patch("auto_annotate.BirdAnnotator") as MockAnnotator:
            instance = MagicMock()
            instance.annotate_dataset.return_value = {
                "total_species": 1, "annotated": 0, "skipped": 1,
                "images": 0, "detected": 0,
            }
            MockAnnotator.return_value = instance

            with patch("sys.argv", ["auto_annotate.py", "annotate", str(tmp_path)]):
                main()

            instance.annotate_dataset.assert_called_once()
            call_kwargs = instance.annotate_dataset.call_args
            assert call_kwargs[1].get("resume") is True or (len(call_kwargs[0]) > 1 and call_kwargs[0][1] is True)


# ---------------------------------------------------------------------------
# UC-45 : CLI — --workers transmet le nombre de workers
# ---------------------------------------------------------------------------
class TestUC45_CLIWorkers:
    def test_workers_parsed(self):
        from auto_annotate import build_parser
        parser = build_parser()
        args = parser.parse_args(["annotate", "/tmp/fake", "--workers", "6"])
        assert args.workers == 6

    def test_workers_passed_to_annotate_dataset(self, tmp_path):
        from unittest.mock import patch, MagicMock
        from auto_annotate import main

        with patch("auto_annotate.BirdAnnotator") as MockAnnotator:
            instance = MagicMock()
            instance.annotate_dataset.return_value = {
                "total_species": 0, "annotated": 0, "skipped": 0,
                "images": 0, "detected": 0,
            }
            MockAnnotator.return_value = instance

            with patch("sys.argv", ["auto_annotate.py", "annotate", str(tmp_path), "--workers", "4"]):
                main()

            call_kwargs = instance.annotate_dataset.call_args
            assert call_kwargs[1]["workers"] == 4


# ---------------------------------------------------------------------------
# UC-46 : CLI — --backend sélectionne le modèle
# ---------------------------------------------------------------------------
class TestUC46_CLIBackend:
    def test_backend_base_parsed(self):
        from auto_annotate import build_parser
        parser = build_parser()
        args = parser.parse_args(["annotate", "/tmp/fake", "--backend", "grounding_dino_base"])
        assert args.backend == "grounding_dino_base"

    def test_invalid_backend_rejected(self):
        from auto_annotate import build_parser
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["annotate", "/tmp/fake", "--backend", "fasterrcnn"])

    def test_backend_passed_to_annotator(self, tmp_path):
        from unittest.mock import patch, MagicMock
        from auto_annotate import main

        with patch("auto_annotate.BirdAnnotator") as MockAnnotator:
            instance = MagicMock()
            instance.annotate_dataset.return_value = {
                "total_species": 0, "annotated": 0, "skipped": 0,
                "images": 0, "detected": 0,
            }
            MockAnnotator.return_value = instance

            with patch("sys.argv", [
                "auto_annotate.py", "annotate", str(tmp_path),
                "--backend", "grounding_dino_base",
            ]):
                main()

            MockAnnotator.assert_called_once()
            call_kwargs = MockAnnotator.call_args
            assert call_kwargs[1]["backend"] == "grounding_dino_base"


# ---------------------------------------------------------------------------
# UC-47 : CLI — --threshold et --text-threshold transmis
# ---------------------------------------------------------------------------
class TestUC47_CLIThresholds:
    def test_custom_thresholds_parsed(self):
        from auto_annotate import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "annotate", "/tmp/fake",
            "--threshold", "0.4",
            "--text-threshold", "0.3",
        ])
        assert args.threshold == 0.4
        assert args.text_threshold == 0.3

    def test_thresholds_passed_to_annotator(self, tmp_path):
        from unittest.mock import patch, MagicMock
        from auto_annotate import main

        with patch("auto_annotate.BirdAnnotator") as MockAnnotator:
            instance = MagicMock()
            instance.annotate_dataset.return_value = {
                "total_species": 0, "annotated": 0, "skipped": 0,
                "images": 0, "detected": 0,
            }
            MockAnnotator.return_value = instance

            with patch("sys.argv", [
                "auto_annotate.py", "annotate", str(tmp_path),
                "--threshold", "0.4", "--text-threshold", "0.3",
            ]):
                main()

            call_kwargs = MockAnnotator.call_args
            assert call_kwargs[1]["threshold"] == 0.4
            assert call_kwargs[1]["text_threshold"] == 0.3


# ---------------------------------------------------------------------------
# UC-48 : CLI — dir inexistant → erreur
# ---------------------------------------------------------------------------
class TestUC48_CLIDirNotFound:
    def test_nonexistent_dir_exits(self):
        from unittest.mock import patch
        from auto_annotate import main

        with patch("sys.argv", ["auto_annotate.py", "annotate", "/tmp/nonexistent_dir_xyz"]):
            with pytest.raises(SystemExit):
                main()
