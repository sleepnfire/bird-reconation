"""
Tests pour la calibration des datasets europe / europe_rejected.

Approche basée sur :
  - Bai et al. (2024) — curation multi-signal (détection + filtres)
  - Zhou et al. (2025) — Autonomous Bird Feeder : confiance ≥ 0.7, bbox > 2%
  - Pech-Pacheco et al. (ICPR 2000) — variance du Laplacien pour la netteté
  - Setting BirdNET confidence thresholds (Springer 2025) — seuils espèce-spécifiques
"""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image as PILImage, ImageDraw, ImageFilter


# ── helpers ──────────────────────────────────────────────────────────────


def _make_sharp_bird_image(path: Path, size=(200, 200), bbox_pct=25):
    """Crée une image nette avec des bords contrastés (haute variance Laplacien)."""
    img = np.zeros((*size, 3), dtype=np.uint8)
    img[:] = (34, 139, 34)  # fond vert forêt
    h, w = size
    bh = int(h * (bbox_pct / 100) ** 0.5)
    bw = int(w * (bbox_pct / 100) ** 0.5)
    cx, cy = w // 2, h // 2
    x1, y1 = cx - bw // 2, cy - bh // 2
    x2, y2 = x1 + bw, y1 + bh
    cv2.rectangle(img, (x1, y1), (x2, y2), (200, 150, 100), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)
    for i in range(5):
        cv2.line(img, (x1 + i * bw // 5, y1), (x1 + i * bw // 5, y2), (50, 50, 50), 1)
    cv2.imwrite(str(path), img)


def _make_blurry_image(path: Path, size=(200, 200)):
    """Crée une image très floue (basse variance Laplacien)."""
    img = np.random.randint(100, 160, (*size, 3), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (31, 31), 15)
    cv2.imwrite(str(path), img)


def _species_dir_with_annotations(base: Path, species: str, images_annotations: dict,
                                  make_images=True, image_size=(200, 200)) -> Path:
    sp_dir = base / species
    sp_dir.mkdir(parents=True, exist_ok=True)
    annotations = {}
    for name, ann in images_annotations.items():
        if make_images:
            img = PILImage.new("RGB", image_size, color=(100, 150, 100))
            img.save(sp_dir / name)
        annotations[name] = ann
    with open(sp_dir / "annotations.json", "w") as f:
        json.dump(annotations, f)
    return sp_dir


def _build_calibration_dataset(tmp_path: Path, n_species=3, n_kept=5, n_rejected=3):
    """Crée un dataset europe + europe_rejected pour les tests."""
    europe = tmp_path / "europe"
    rejected = tmp_path / "europe_rejected"

    for split in ("train",):
        for i in range(n_species):
            slug = f"species_{i}"

            # Images gardées (europe)
            sp_kept = europe / split / slug
            sp_kept.mkdir(parents=True, exist_ok=True)
            ann_kept = {}
            for j in range(n_kept):
                name = f"kept_{j:03d}.jpg"
                _make_sharp_bird_image(sp_kept / name, bbox_pct=15)
                ann_kept[name] = {"bbox": [30, 30, 70, 70], "score": 0.85}
            with open(sp_kept / "annotations.json", "w") as f:
                json.dump(ann_kept, f)

            # Images rejetées (europe_rejected)
            sp_rej = rejected / split / slug
            sp_rej.mkdir(parents=True, exist_ok=True)
            ann_rej = {}
            for j in range(n_rejected):
                name = f"rejected_{j:03d}.jpg"
                _make_blurry_image(sp_rej / name)
                ann_rej[name] = {"bbox": [90, 90, 10, 10], "score": 0.4}
            with open(sp_rej / "annotations.json", "w") as f:
                json.dump(ann_rej, f)

    return europe, rejected


# ═════════════════════════════════════════════════════════════════════════
# IMAGE SCORER — score composite par image
# (détection + bbox + netteté, cf. Bai et al. 2024, Zhou et al. 2025)
# ═════════════════════════════════════════════════════════════════════════

# ── UC-C01 : score_image retourne un dict avec les 4 clés attendues ─────

class TestUCC01_ScoreImageReturnsDict:
    def test_returns_expected_keys(self, tmp_path):
        from calibrate_datasets import ImageScorer

        scorer = ImageScorer()
        img_path = tmp_path / "bird.jpg"
        _make_sharp_bird_image(img_path)
        annotation = {"bbox": [30, 30, 70, 70], "score": 0.9}

        result = scorer.score_image(img_path, annotation)
        assert "detection_score" in result
        assert "bbox_pct" in result
        assert "sharpness" in result
        assert "composite_score" in result


# ── UC-C02 : score_image avec annotation None retourne composite_score=0

class TestUCC02_ScoreImageNoneAnnotation:
    def test_none_annotation_gives_zero_composite(self, tmp_path):
        from calibrate_datasets import ImageScorer

        scorer = ImageScorer()
        img_path = tmp_path / "bird.jpg"
        _make_sharp_bird_image(img_path)

        result = scorer.score_image(img_path, None)
        assert result["composite_score"] == 0.0
        assert result["detection_score"] == 0.0
        assert result["bbox_pct"] == 0.0


# ── UC-C03 : score_image gère l'absence de fichier annotations.json ─────

class TestUCC03_ScoreImageMissingFile:
    def test_missing_image_raises(self, tmp_path):
        from calibrate_datasets import ImageScorer

        scorer = ImageScorer()
        result = scorer.score_image(tmp_path / "nonexistent.jpg", None)
        assert result["composite_score"] == 0.0


# ── UC-C04 : score composite plus élevé pour images nettes + grande bbox

class TestUCC04_CompositeScoreOrdering:
    def test_sharp_large_bbox_scores_higher(self, tmp_path):
        from calibrate_datasets import ImageScorer

        scorer = ImageScorer()

        sharp_path = tmp_path / "sharp.jpg"
        _make_sharp_bird_image(sharp_path, bbox_pct=25)
        sharp_ann = {"bbox": [30, 30, 100, 100], "score": 0.95}

        blurry_path = tmp_path / "blurry.jpg"
        _make_blurry_image(blurry_path)
        blurry_ann = {"bbox": [90, 90, 10, 10], "score": 0.3}

        sharp_result = scorer.score_image(sharp_path, sharp_ann)
        blurry_result = scorer.score_image(blurry_path, blurry_ann)

        assert sharp_result["composite_score"] > blurry_result["composite_score"]


# ── UC-C05 : score composite proche de 0 pour images sans détection ─────

class TestUCC05_NoDetectionLowScore:
    def test_no_detection_near_zero(self, tmp_path):
        from calibrate_datasets import ImageScorer

        scorer = ImageScorer()
        img_path = tmp_path / "bird.jpg"
        _make_sharp_bird_image(img_path)

        result = scorer.score_image(img_path, None)
        assert result["composite_score"] == 0.0


# ── UC-C06 : score_species retourne un dict image_name -> score_dict ────

class TestUCC06_ScoreSpecies:
    def test_returns_dict_for_all_images(self, tmp_path):
        from calibrate_datasets import ImageScorer

        scorer = ImageScorer()
        sp_dir = _species_dir_with_annotations(
            tmp_path / "train", "parus_major",
            {
                "img_0.jpg": {"bbox": [10, 10, 50, 50], "score": 0.9},
                "img_1.jpg": {"bbox": [10, 10, 30, 30], "score": 0.7},
                "img_2.jpg": None,
            }
        )

        result = scorer.score_species(sp_dir)
        assert len(result) == 3
        assert "img_0.jpg" in result
        assert "img_1.jpg" in result
        assert "img_2.jpg" in result
        for name, scores in result.items():
            assert "composite_score" in scores


# ── UC-C07 : score_species ignore les fichiers non-image ────────────────

class TestUCC07_ScoreSpeciesIgnoresNonImages:
    def test_skips_json_files(self, tmp_path):
        from calibrate_datasets import ImageScorer

        scorer = ImageScorer()
        sp_dir = _species_dir_with_annotations(
            tmp_path / "train", "test_sp",
            {"photo.jpg": {"bbox": [10, 10, 50, 50], "score": 0.9}}
        )

        result = scorer.score_species(sp_dir)
        assert "annotations.json" not in result
        assert "photo.jpg" in result


# ═════════════════════════════════════════════════════════════════════════
# STRATIFIED SAMPLER — échantillonnage pour revue humaine
# ═════════════════════════════════════════════════════════════════════════

# ── UC-C08 : sample retourne N images par espèce de chaque dataset ──────

class TestUCC08_SampleReturnsNPerSpecies:
    def test_returns_n_per_species(self, tmp_path):
        from calibrate_datasets import ImageScorer, StratifiedSampler

        europe, rejected = _build_calibration_dataset(tmp_path, n_species=2, n_kept=5, n_rejected=3)
        scorer = ImageScorer()
        sampler = StratifiedSampler(europe / "train", rejected / "train", scorer)

        samples = sampler.sample(n_per_species=2, mode="random", seed=42)
        species = {s["species"] for s in samples}
        assert len(species) == 2
        for sp in species:
            sp_samples = [s for s in samples if s["species"] == sp]
            assert len(sp_samples) <= 4  # 2 kept + 2 rejected max


# ── UC-C09 : mode borderline surreprésente les images proches du seuil ──

class TestUCC09_BorderlineMode:
    def test_borderline_oversamples_near_threshold(self, tmp_path):
        from calibrate_datasets import ImageScorer, StratifiedSampler

        europe, rejected = _build_calibration_dataset(tmp_path, n_species=2, n_kept=10, n_rejected=5)
        scorer = ImageScorer()
        sampler = StratifiedSampler(europe / "train", rejected / "train", scorer)

        samples = sampler.sample(n_per_species=3, mode="borderline", seed=42)
        assert len(samples) > 0
        for s in samples:
            assert "composite_score" in s


# ── UC-C10 : sample est reproductible avec un seed ──────────────────────

class TestUCC10_SampleReproducible:
    def test_same_seed_same_samples(self, tmp_path):
        from calibrate_datasets import ImageScorer, StratifiedSampler

        europe, rejected = _build_calibration_dataset(tmp_path, n_species=2, n_kept=5, n_rejected=3)
        scorer = ImageScorer()
        sampler = StratifiedSampler(europe / "train", rejected / "train", scorer)

        s1 = sampler.sample(n_per_species=2, mode="random", seed=42)
        s2 = sampler.sample(n_per_species=2, mode="random", seed=42)
        paths1 = [s["path"] for s in s1]
        paths2 = [s["path"] for s in s2]
        assert paths1 == paths2


# ── UC-C11 : sample gère les espèces avec moins d'images que N ──────────

class TestUCC11_SampleFewImages:
    def test_handles_species_with_few_images(self, tmp_path):
        from calibrate_datasets import ImageScorer, StratifiedSampler

        europe, rejected = _build_calibration_dataset(tmp_path, n_species=1, n_kept=2, n_rejected=1)
        scorer = ImageScorer()
        sampler = StratifiedSampler(europe / "train", rejected / "train", scorer)

        samples = sampler.sample(n_per_species=10, mode="random", seed=42)
        assert len(samples) <= 3  # max 2 kept + 1 rejected


# ═════════════════════════════════════════════════════════════════════════
# GROUND TRUTH — stockage des labels humains
# ═════════════════════════════════════════════════════════════════════════

# ── UC-C12 : load retourne un dict vide si pas de fichier ───────────────

class TestUCC12_GroundTruthLoadEmpty:
    def test_empty_if_no_file(self, tmp_path):
        from calibrate_datasets import GroundTruth

        gt = GroundTruth(tmp_path / "ground_truth.json")
        labels = gt.load()
        assert labels == {}


# ── UC-C13 : save persiste les labels et load les retrouve ──────────────

class TestUCC13_GroundTruthSaveLoad:
    def test_save_and_load(self, tmp_path):
        from calibrate_datasets import GroundTruth

        gt = GroundTruth(tmp_path / "ground_truth.json")
        labels = {"img1.jpg": {"label": "good", "source": "europe", "species": "parus_major"}}
        gt.save(labels)

        loaded = gt.load()
        assert loaded == labels


# ── UC-C14 : add_label ajoute un label sans écraser les autres ──────────

class TestUCC14_GroundTruthAddLabel:
    def test_add_label(self, tmp_path):
        from calibrate_datasets import GroundTruth

        gt = GroundTruth(tmp_path / "ground_truth.json")
        gt.add_label("img1.jpg", "good", "europe", "parus_major")
        gt.add_label("img2.jpg", "bad", "europe_rejected", "turdus_merula")

        labels = gt.load()
        assert len(labels) == 2
        assert labels["img1.jpg"]["label"] == "good"
        assert labels["img2.jpg"]["label"] == "bad"


# ═════════════════════════════════════════════════════════════════════════
# COMPUTE METRICS — précision / rappel du filtre actuel
# ═════════════════════════════════════════════════════════════════════════

# ── UC-C15 : compute_metrics retourne precision, recall, f1, fpr, fnr ───

class TestUCC15_ComputeMetricsKeys:
    def test_returns_expected_keys(self):
        from calibrate_datasets import compute_metrics

        labels = {
            "kept_good.jpg": {"label": "good", "source": "europe"},
            "kept_bad.jpg": {"label": "bad", "source": "europe"},
            "rej_good.jpg": {"label": "good", "source": "europe_rejected"},
            "rej_bad.jpg": {"label": "bad", "source": "europe_rejected"},
        }
        metrics = compute_metrics(labels)
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "false_positive_rate" in metrics
        assert "false_negative_rate" in metrics


# ── UC-C16 : métriques parfaites si toutes les décisions sont correctes ─

class TestUCC16_PerfectMetrics:
    def test_perfect_predictions(self):
        from calibrate_datasets import compute_metrics

        labels = {
            "kept_1.jpg": {"label": "good", "source": "europe"},
            "kept_2.jpg": {"label": "good", "source": "europe"},
            "rej_1.jpg": {"label": "bad", "source": "europe_rejected"},
            "rej_2.jpg": {"label": "bad", "source": "europe_rejected"},
        }
        metrics = compute_metrics(labels)
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["false_positive_rate"] == 0.0
        assert metrics["false_negative_rate"] == 0.0


# ── UC-C17 : precision 0 si toutes les images gardées sont mauvaises ────

class TestUCC17_ZeroPrecision:
    def test_all_kept_are_bad(self):
        from calibrate_datasets import compute_metrics

        labels = {
            "kept_1.jpg": {"label": "bad", "source": "europe"},
            "kept_2.jpg": {"label": "bad", "source": "europe"},
            "rej_1.jpg": {"label": "bad", "source": "europe_rejected"},
        }
        metrics = compute_metrics(labels)
        assert metrics["precision"] == 0.0


# ── UC-C18 : compute_metrics_by_reason ventile par raison de rejet ──────

class TestUCC18_MetricsByReason:
    def test_breakdown_by_reason(self):
        from calibrate_datasets import compute_metrics_by_reason

        labels = {
            "img1.jpg": {"label": "good", "source": "europe_rejected", "reject_reason": "bbox_small"},
            "img2.jpg": {"label": "bad", "source": "europe_rejected", "reject_reason": "bbox_small"},
            "img3.jpg": {"label": "good", "source": "europe_rejected", "reject_reason": "no_detection"},
            "img4.jpg": {"label": "bad", "source": "europe_rejected", "reject_reason": "no_detection"},
        }
        result = compute_metrics_by_reason(labels)
        assert "bbox_small" in result
        assert "no_detection" in result
        assert result["bbox_small"]["false_negative_count"] == 1
        assert result["no_detection"]["false_negative_count"] == 1


# ═════════════════════════════════════════════════════════════════════════
# THRESHOLD OPTIMIZER — balayage de seuils
# (cf. Setting BirdNET confidence thresholds, Springer 2025)
# ═════════════════════════════════════════════════════════════════════════

# ── UC-C19 : sweep retourne une liste de (combo, metrics) triée par f1 ──

class TestUCC19_ThresholdSweep:
    def test_sweep_returns_sorted_results(self, tmp_path):
        from calibrate_datasets import ThresholdOptimizer, GroundTruth, ImageScorer

        europe, rejected = _build_calibration_dataset(tmp_path, n_species=1, n_kept=5, n_rejected=3)
        scorer = ImageScorer()
        gt = GroundTruth(tmp_path / "gt.json")
        # Labeller les images
        for sp_dir in (europe / "train").iterdir():
            if not sp_dir.is_dir():
                continue
            for img in sp_dir.iterdir():
                if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    gt.add_label(str(img), "good", "europe", sp_dir.name)
        for sp_dir in (rejected / "train").iterdir():
            if not sp_dir.is_dir():
                continue
            for img in sp_dir.iterdir():
                if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    gt.add_label(str(img), "bad", "europe_rejected", sp_dir.name)

        optimizer = ThresholdOptimizer(scorer, europe / "train", rejected / "train")
        results = optimizer.sweep(gt)
        assert len(results) > 0
        f1_scores = [r["metrics"]["f1"] for r in results]
        assert f1_scores == sorted(f1_scores, reverse=True)


# ── UC-C20 : sweep avec un seul ground truth fonctionne ─────────────────

class TestUCC20_SweepSingleSample:
    def test_single_sample_works(self, tmp_path):
        from calibrate_datasets import ThresholdOptimizer, GroundTruth, ImageScorer

        europe, rejected = _build_calibration_dataset(tmp_path, n_species=1, n_kept=1, n_rejected=0)
        scorer = ImageScorer()
        gt = GroundTruth(tmp_path / "gt.json")
        for img in (europe / "train" / "species_0").iterdir():
            if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                gt.add_label(str(img), "good", "europe", "species_0")
                break

        optimizer = ThresholdOptimizer(scorer, europe / "train", rejected / "train")
        results = optimizer.sweep(gt)
        assert len(results) > 0


# ── UC-C21 : recommend retourne la combinaison avec le meilleur f1 ──────

class TestUCC21_ThresholdRecommend:
    def test_recommend_returns_best_f1(self, tmp_path):
        from calibrate_datasets import ThresholdOptimizer, GroundTruth, ImageScorer

        europe, rejected = _build_calibration_dataset(tmp_path, n_species=1, n_kept=5, n_rejected=3)
        scorer = ImageScorer()
        gt = GroundTruth(tmp_path / "gt.json")
        for sp_dir in (europe / "train").iterdir():
            if not sp_dir.is_dir():
                continue
            for img in sp_dir.iterdir():
                if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    gt.add_label(str(img), "good", "europe", sp_dir.name)
        for sp_dir in (rejected / "train").iterdir():
            if not sp_dir.is_dir():
                continue
            for img in sp_dir.iterdir():
                if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    gt.add_label(str(img), "bad", "europe_rejected", sp_dir.name)

        optimizer = ThresholdOptimizer(scorer, europe / "train", rejected / "train")
        best = optimizer.recommend(gt)
        assert "thresholds" in best
        assert "metrics" in best
        assert best["metrics"]["f1"] >= 0.0


# ═════════════════════════════════════════════════════════════════════════
# DATASET AUDITOR — audit automatique des deux datasets
# ═════════════════════════════════════════════════════════════════════════

# ── UC-C22 : audit_kept flag les images gardées avec score < seuil ──────

class TestUCC22_AuditKept:
    def test_flags_low_score_kept_images(self, tmp_path):
        from calibrate_datasets import DatasetAuditor, ImageScorer

        europe, _ = _build_calibration_dataset(tmp_path, n_species=1, n_kept=3, n_rejected=0)
        # Ajouter une image sans détection (score 0)
        sp_dir = europe / "train" / "species_0"
        _make_blurry_image(sp_dir / "no_bird.jpg")
        with open(sp_dir / "annotations.json") as f:
            ann = json.load(f)
        ann["no_bird.jpg"] = None
        with open(sp_dir / "annotations.json", "w") as f:
            json.dump(ann, f)

        scorer = ImageScorer()
        auditor = DatasetAuditor(scorer, europe / "train")
        flagged = auditor.audit_kept(min_composite=0.1)
        all_flagged_names = [
            img["name"] for imgs in flagged.values() for img in imgs
        ]
        assert "no_bird.jpg" in all_flagged_names


# ── UC-C23 : audit_rejected flag les images rejetées avec score > seuil ─

class TestUCC23_AuditRejected:
    def test_flags_high_score_rejected_images(self, tmp_path):
        from calibrate_datasets import DatasetAuditor, ImageScorer

        _, rejected = _build_calibration_dataset(tmp_path, n_species=1, n_kept=0, n_rejected=3)
        # Ajouter une bonne image dans rejected
        sp_dir = rejected / "train" / "species_0"
        _make_sharp_bird_image(sp_dir / "good_bird.jpg", bbox_pct=25)
        with open(sp_dir / "annotations.json") as f:
            ann = json.load(f)
        ann["good_bird.jpg"] = {"bbox": [30, 30, 100, 100], "score": 0.95}
        with open(sp_dir / "annotations.json", "w") as f:
            json.dump(ann, f)

        scorer = ImageScorer()
        auditor = DatasetAuditor(scorer, rejected / "train")
        flagged = auditor.audit_rejected(min_composite=0.1)
        all_flagged_names = [
            img["name"] for imgs in flagged.values() for img in imgs
        ]
        assert "good_bird.jpg" in all_flagged_names


# ── UC-C24 : summary retourne les comptes d'images flagées ──────────────

class TestUCC24_AuditSummary:
    def test_summary_counts(self, tmp_path):
        from calibrate_datasets import DatasetAuditor, ImageScorer

        europe, _ = _build_calibration_dataset(tmp_path, n_species=2, n_kept=3, n_rejected=0)
        scorer = ImageScorer()
        auditor = DatasetAuditor(scorer, europe / "train")
        auditor.audit_kept(min_composite=0.1)
        summary = auditor.summary()
        assert "total_flagged" in summary
        assert "by_species" in summary
        assert isinstance(summary["total_flagged"], int)


# ═════════════════════════════════════════════════════════════════════════
# APPLY RECLASSIFICATION — déplacement d'images entre datasets
# ═════════════════════════════════════════════════════════════════════════

# ── UC-C25 : apply_reclassification déplace les images flagées ──────────

class TestUCC25_ApplyMovesImages:
    def test_moves_flagged_images(self, tmp_path):
        from calibrate_datasets import apply_reclassification

        europe, rejected = _build_calibration_dataset(tmp_path, n_species=1, n_kept=3, n_rejected=2)
        sp_kept = europe / "train" / "species_0"
        first_img = sorted(sp_kept.glob("*.jpg"))[0].name

        moves = [{
            "path": str(sp_kept / first_img),
            "from_dir": str(europe / "train"),
            "to_dir": str(rejected / "train"),
            "species": "species_0",
            "name": first_img,
        }]

        result = apply_reclassification(moves)
        assert not (sp_kept / first_img).exists()
        assert (rejected / "train" / "species_0" / first_img).exists()
        assert result["moved"] == 1


# ── UC-C26 : apply_reclassification met à jour annotations.json ─────────

class TestUCC26_ApplyUpdatesAnnotations:
    def test_updates_annotations(self, tmp_path):
        from calibrate_datasets import apply_reclassification

        europe, rejected = _build_calibration_dataset(tmp_path, n_species=1, n_kept=3, n_rejected=2)
        sp_kept = europe / "train" / "species_0"
        first_img = sorted(sp_kept.glob("*.jpg"))[0].name

        moves = [{
            "path": str(sp_kept / first_img),
            "from_dir": str(europe / "train"),
            "to_dir": str(rejected / "train"),
            "species": "species_0",
            "name": first_img,
        }]

        apply_reclassification(moves)

        with open(sp_kept / "annotations.json") as f:
            ann_kept = json.load(f)
        assert first_img not in ann_kept

        with open(rejected / "train" / "species_0" / "annotations.json") as f:
            ann_rej = json.load(f)
        assert first_img in ann_rej


# ── UC-C27 : dry_run ne déplace rien ────────────────────────────────────

class TestUCC27_DryRunNoChanges:
    def test_dry_run_moves_nothing(self, tmp_path):
        from calibrate_datasets import apply_reclassification

        europe, rejected = _build_calibration_dataset(tmp_path, n_species=1, n_kept=3, n_rejected=2)
        sp_kept = europe / "train" / "species_0"
        first_img = sorted(sp_kept.glob("*.jpg"))[0].name

        moves = [{
            "path": str(sp_kept / first_img),
            "from_dir": str(europe / "train"),
            "to_dir": str(rejected / "train"),
            "species": "species_0",
            "name": first_img,
        }]

        result = apply_reclassification(moves, dry_run=True)
        assert (sp_kept / first_img).exists()
        assert result["moved"] == 0


# ── UC-C28 : gère les espèces absentes du dataset destination ───────────

class TestUCC28_ApplyCreatesSpeciesDir:
    def test_creates_species_dir_in_destination(self, tmp_path):
        from calibrate_datasets import apply_reclassification

        europe = tmp_path / "europe"
        rejected = tmp_path / "europe_rejected"
        sp_dir = europe / "train" / "new_species"
        sp_dir.mkdir(parents=True)
        _make_sharp_bird_image(sp_dir / "bird.jpg")
        ann = {"bird.jpg": {"bbox": [10, 10, 50, 50], "score": 0.9}}
        with open(sp_dir / "annotations.json", "w") as f:
            json.dump(ann, f)

        moves = [{
            "path": str(sp_dir / "bird.jpg"),
            "from_dir": str(europe / "train"),
            "to_dir": str(rejected / "train"),
            "species": "new_species",
            "name": "bird.jpg",
        }]

        result = apply_reclassification(moves)
        assert (rejected / "train" / "new_species" / "bird.jpg").exists()
        assert result["moved"] == 1


# ═════════════════════════════════════════════════════════════════════════
# CONSOLIDATE — rassembler toutes les images dans europe_to_trait
# ═════════════════════════════════════════════════════════════════════════


def _build_split_dataset(tmp_path: Path, n_species=2, n_per_split=3):
    """Crée europe/{train,val,test} et europe_rejected/{train,val,test} avec images."""
    europe = tmp_path / "europe"
    rejected = tmp_path / "europe_rejected"

    for ds_dir, prefix, score in [(europe, "kept", 0.9), (rejected, "rej", 0.4)]:
        for split in ("train", "validation", "test"):
            for i in range(n_species):
                slug = f"species_{i}"
                sp_dir = ds_dir / split / slug
                sp_dir.mkdir(parents=True, exist_ok=True)
                ann = {}
                for j in range(n_per_split):
                    name = f"{prefix}_{split}_{j:03d}.jpg"
                    _make_sharp_bird_image(sp_dir / name, bbox_pct=15)
                    ann[name] = {"bbox": [30, 30, 70, 70], "score": score}
                with open(sp_dir / "annotations.json", "w") as f:
                    json.dump(ann, f)

    label_map = {f"species_{i}": i for i in range(n_species)}
    with open(europe / "label_map.json", "w") as f:
        json.dump(label_map, f)
    metadata = {f"species_{i}": {"slug": f"species_{i}"} for i in range(n_species)}
    with open(europe / "metadata.json", "w") as f:
        json.dump(metadata, f)

    return europe, rejected


# ── UC-C29 : consolidate rassemble toutes les images dans europe_to_trait

class TestUCC29_ConsolidateMovesAll:
    def test_all_images_in_to_trait(self, tmp_path):
        from calibrate_datasets import consolidate

        europe, rejected = _build_split_dataset(tmp_path, n_species=2, n_per_split=3)
        to_trait = tmp_path / "europe_to_trait"

        stats = consolidate(europe, rejected, to_trait)

        # 2 espèces × (3 kept + 3 rejected) × 3 splits = 36 images
        assert stats["total_moved"] == 36
        for i in range(2):
            sp_dir = to_trait / f"species_{i}"
            assert sp_dir.exists()
            images = [f for f in sp_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
            assert len(images) == 18  # 9 kept + 9 rejected


# ── UC-C30 : consolidate fusionne les annotations.json de tous les splits

class TestUCC30_ConsolidateMergesAnnotations:
    def test_annotations_merged(self, tmp_path):
        from calibrate_datasets import consolidate

        europe, rejected = _build_split_dataset(tmp_path, n_species=1, n_per_split=2)
        to_trait = tmp_path / "europe_to_trait"

        consolidate(europe, rejected, to_trait)

        ann_path = to_trait / "species_0" / "annotations.json"
        assert ann_path.exists()
        with open(ann_path) as f:
            ann = json.load(f)
        # 2 per split × 3 splits × 2 datasets = 12
        assert len(ann) == 12


# ── UC-C31 : select_top garde les top N par score, exclut score=0

class TestUCC31_SelectTopKeepsN:
    def test_keeps_top_n_per_species(self, tmp_path):
        from calibrate_datasets import ImageScorer, select_top

        to_trait = tmp_path / "europe_to_trait"
        to_trait_rej = tmp_path / "europe_to_trait_rejected"
        sp_dir = to_trait / "species_0"
        sp_dir.mkdir(parents=True)
        ann = {}
        for i in range(10):
            name = f"img_{i:03d}.jpg"
            _make_sharp_bird_image(sp_dir / name, bbox_pct=15)
            ann[name] = {"bbox": [30, 30, 70, 70], "score": 0.5 + i * 0.05}
        with open(sp_dir / "annotations.json", "w") as f:
            json.dump(ann, f)

        scorer = ImageScorer()
        stats = select_top(to_trait, to_trait_rej, scorer, max_per_species=5)

        kept = [f for f in sp_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        rej = [f for f in (to_trait_rej / "species_0").iterdir()
               if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        assert len(kept) == 5
        assert len(rej) == 5
        assert stats["total_kept"] == 5
        assert stats["total_rejected"] == 5


# ── UC-C32 : select_top exclut les images avec score=0 (no detection)

class TestUCC32_SelectTopExcludesZero:
    def test_excludes_no_detection(self, tmp_path):
        from calibrate_datasets import ImageScorer, select_top

        to_trait = tmp_path / "europe_to_trait"
        to_trait_rej = tmp_path / "europe_to_trait_rejected"
        sp_dir = to_trait / "species_0"
        sp_dir.mkdir(parents=True)
        ann = {}
        for i in range(5):
            name = f"good_{i:03d}.jpg"
            _make_sharp_bird_image(sp_dir / name, bbox_pct=15)
            ann[name] = {"bbox": [30, 30, 70, 70], "score": 0.9}
        for i in range(3):
            name = f"nodet_{i:03d}.jpg"
            _make_blurry_image(sp_dir / name)
            ann[name] = None
        with open(sp_dir / "annotations.json", "w") as f:
            json.dump(ann, f)

        scorer = ImageScorer()
        stats = select_top(to_trait, to_trait_rej, scorer, max_per_species=500)

        kept = [f for f in sp_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        assert len(kept) == 5  # seulement les 5 avec détection
        rej_dir = to_trait_rej / "species_0"
        rej = [f for f in rej_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        assert len(rej) == 3  # les 3 sans détection


# ── UC-C33 : select_top met à jour annotations.json des deux côtés

class TestUCC33_SelectTopUpdatesAnnotations:
    def test_annotations_updated_both_sides(self, tmp_path):
        from calibrate_datasets import ImageScorer, select_top

        to_trait = tmp_path / "europe_to_trait"
        to_trait_rej = tmp_path / "europe_to_trait_rejected"
        sp_dir = to_trait / "species_0"
        sp_dir.mkdir(parents=True)
        ann = {}
        for i in range(6):
            name = f"img_{i:03d}.jpg"
            _make_sharp_bird_image(sp_dir / name, bbox_pct=15)
            ann[name] = {"bbox": [30, 30, 70, 70], "score": 0.5 + i * 0.1}
        with open(sp_dir / "annotations.json", "w") as f:
            json.dump(ann, f)

        scorer = ImageScorer()
        select_top(to_trait, to_trait_rej, scorer, max_per_species=3)

        with open(sp_dir / "annotations.json") as f:
            kept_ann = json.load(f)
        assert len(kept_ann) == 3

        with open(to_trait_rej / "species_0" / "annotations.json") as f:
            rej_ann = json.load(f)
        assert len(rej_ann) == 3


# ── UC-C34 : dry_run consolidate ne déplace rien

class TestUCC34_ConsolidateDryRun:
    def test_dry_run_no_move(self, tmp_path):
        from calibrate_datasets import consolidate

        europe, rejected = _build_split_dataset(tmp_path, n_species=1, n_per_split=2)
        to_trait = tmp_path / "europe_to_trait"

        stats = consolidate(europe, rejected, to_trait, dry_run=True)

        assert stats["total_moved"] > 0
        assert not to_trait.exists() or not any(to_trait.iterdir())
