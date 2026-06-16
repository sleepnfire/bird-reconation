"""
Tests pour le filtrage qualité des images via CLIP (zero-shot classification)
et détection d'outliers par embeddings.

Pipeline basé sur :
  - Radford et al. 2021 (CLIP, ICML) — prompts domaine-spécifiques
  - DataComp (Gadre et al., NeurIPS 2023) — filtrage CLIP
  - LAION-5B (Schuhmann et al., NeurIPS 2022) — outliers par centroïde + distance
  - BioTrove (Yang et al., NeurIPS 2024) — curation iNaturalist via CLIP
"""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage, ImageDraw


FIXTURES = Path(__file__).parent / "fixtures" / "images"


# ── helpers ──────────────────────────────────────────────────────────────

def _make_illustration(path: Path):
    img = PILImage.new("RGB", (224, 224), "white")
    draw = ImageDraw.Draw(img)
    draw.ellipse([80, 60, 150, 120], outline="black", width=2)
    draw.polygon([(150, 90), (180, 85), (150, 95)], fill="orange")
    draw.line([(100, 120), (95, 160), (90, 180)], fill="black", width=2)
    draw.line([(120, 120), (125, 160), (130, 180)], fill="black", width=2)
    draw.text((60, 190), "Bird illustration", fill="black")
    img.save(path)


def _make_screen_photo(path: Path):
    img = PILImage.new("RGB", (224, 224), "black")
    inner = PILImage.new("RGB", (160, 120), (100, 150, 100))
    draw_inner = ImageDraw.Draw(inner)
    draw_inner.ellipse([50, 20, 110, 80], fill=(139, 90, 43))
    draw_inner.text((30, 90), "screen capture", fill="white")
    img.paste(inner, (32, 52))
    draw = ImageDraw.Draw(img)
    draw.line([(0, 30), (224, 50)], fill=(255, 255, 255, 80), width=1)
    img.save(path)


def _species_dir_with_annotations(base: Path, species: str, images_annotations: dict) -> Path:
    sp_dir = base / species
    sp_dir.mkdir(parents=True, exist_ok=True)
    annotations = {}
    for name, ann in images_annotations.items():
        img = PILImage.new("RGB", (200, 200), color=(100, 150, 100))
        img.save(sp_dir / name)
        annotations[name] = ann
    with open(sp_dir / "annotations.json", "w") as f:
        json.dump(annotations, f)
    return sp_dir


def _real_bird_fixtures(n: int = 5) -> list[Path]:
    return sorted(
        f for f in FIXTURES.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        and not f.name.startswith(("no_bird", "grayscale", "rgba", "corrupted"))
    )[:n]


# ═════════════════════════════════════════════════════════════════════════
# CLIP ZERO-SHOT — catégories : good, dead_specimen, illustration,
#                                screen_scan, not_bird, poor_quality
# ═════════════════════════════════════════════════════════════════════════

# ── UC-Q01 : classify_image retourne une catégorie valide + confidence ───

class TestUCQ01_ClassifyReturnsCategory:
    def test_returns_valid_category(self):
        from quality_filter import QualityClassifier, CATEGORIES

        clf = QualityClassifier()
        result = clf.classify_image(str(_real_bird_fixtures(1)[0]))
        assert result["category"] in CATEGORIES
        assert 0.0 <= result["confidence"] <= 1.0
        assert "scores" in result


# ── UC-Q02 : les vraies photos d'oiseaux sont classées good ─────────────

class TestUCQ02_LiveBirdDetected:
    def test_real_bird_photos_classified_as_good(self):
        from quality_filter import QualityClassifier

        clf = QualityClassifier()
        birds = _real_bird_fixtures(5)
        good_count = sum(
            1 for img in birds
            if clf.classify_image(str(img))["category"] == "good"
        )
        assert good_count >= 3, f"Seulement {good_count}/5 classées good"


# ── UC-Q03 : une illustration est classée illustration ───────────────────

class TestUCQ03_IllustrationDetected:
    def test_illustration_classified(self, tmp_path):
        from quality_filter import QualityClassifier

        clf = QualityClassifier()
        illus_path = tmp_path / "illustration.jpg"
        _make_illustration(illus_path)
        result = clf.classify_image(str(illus_path))
        assert result["category"] in ("illustration", "not_bird"), (
            f"Illustration classée comme {result['category']}"
        )


# ── UC-Q04 : image sans oiseau classée not_bird ─────────────────────────

class TestUCQ04_NotBirdDetected:
    def test_no_bird_image(self):
        from quality_filter import QualityClassifier

        clf = QualityClassifier(reject_margin=0.0)
        result = clf.classify_image(str(FIXTURES / "no_bird__blue_sky.jpg"))
        assert result["category"] == "not_bird", (
            f"Image sans oiseau classée comme {result['category']}"
        )

    def test_no_bird_with_margin_has_low_good_score(self):
        """Avec la marge par défaut, un ciel bleu peut rester good mais le score
        not_bird doit être supérieur ou égal au score good."""
        from quality_filter import QualityClassifier

        clf = QualityClassifier()
        result = clf.classify_image(str(FIXTURES / "no_bird__blue_sky.jpg"))
        assert result["scores"]["not_bird"] >= result["scores"]["good"]


# ═════════════════════════════════════════════════════════════════════════
# FILTRE BBOX — trop petite (< min_area_pct% de l'image)
# ═════════════════════════════════════════════════════════════════════════

# ── UC-Q05 : filter_small_bbox détecte les bbox trop petites ────────────

class TestUCQ05_SmallBboxFilter:
    def test_flags_small_bbox(self, tmp_path):
        from quality_filter import filter_small_bbox

        sp_dir = _species_dir_with_annotations(
            tmp_path / "train", "test_species",
            {
                "big_bird.jpg": {"bbox": [10, 10, 100, 100], "score": 0.9},
                "tiny_bird.jpg": {"bbox": [50, 50, 5, 5], "score": 0.8},
                "no_det.jpg": None,
            }
        )
        flagged = filter_small_bbox(sp_dir, min_area_pct=5.0)
        assert "tiny_bird.jpg" in flagged
        assert "big_bird.jpg" not in flagged
        assert "no_det.jpg" not in flagged


# ── UC-Q06 : filter_small_bbox utilise les dimensions réelles de l'image ─

class TestUCQ06_SmallBboxUsesImageSize:
    def test_uses_actual_image_dimensions(self, tmp_path):
        from quality_filter import filter_small_bbox

        sp_dir = tmp_path / "train" / "sp"
        sp_dir.mkdir(parents=True)
        big_img = PILImage.new("RGB", (1000, 1000))
        big_img.save(sp_dir / "photo.jpg")
        annotations = {"photo.jpg": {"bbox": [0, 0, 50, 50], "score": 0.9}}
        with open(sp_dir / "annotations.json", "w") as f:
            json.dump(annotations, f)

        flagged = filter_small_bbox(sp_dir, min_area_pct=5.0)
        # 50*50 = 2500, image = 1000*1000 = 1_000_000, ratio = 0.25% < 5%
        assert "photo.jpg" in flagged


# ═════════════════════════════════════════════════════════════════════════
# OUTLIER PAR EMBEDDING — centroïde par espèce, flag > 1.5σ
# (Schuhmann et al., NeurIPS 2022 / FiftyOne)
# ═════════════════════════════════════════════════════════════════════════

# ── UC-Q07 : compute_embeddings retourne un tensor par image ─────────────

class TestUCQ07_ComputeEmbeddings:
    def test_returns_embeddings_dict(self, tmp_path):
        from quality_filter import QualityClassifier

        clf = QualityClassifier()
        sp_dir = tmp_path / "sp"
        sp_dir.mkdir()
        for i in range(3):
            img = PILImage.new("RGB", (100, 100), color=(50 * i, 100, 50))
            img.save(sp_dir / f"photo_{i}.jpg")

        embeddings = clf.compute_embeddings(str(sp_dir))
        assert len(embeddings) == 3
        for name, emb in embeddings.items():
            assert emb.shape[0] > 0  # vecteur non vide


# ── UC-Q08 : detect_outliers flag les images loin du centroïde ───────────

class TestUCQ08_DetectOutliers:
    def test_outlier_flagged(self):
        from quality_filter import QualityClassifier

        clf = QualityClassifier()
        bird_images = _real_bird_fixtures(5)

        sp_dir = bird_images[0].parent
        embeddings = {}
        for img in bird_images:
            embs = clf.compute_embeddings(str(sp_dir))
            embeddings.update(embs)
            break

        # Avec seulement des images similaires, peu d'outliers
        outliers = clf.detect_outliers(embeddings, sigma_threshold=1.5)
        assert isinstance(outliers, dict)
        for name, info in outliers.items():
            assert "distance" in info
            assert "is_outlier" in info


# ── UC-Q09 : detect_outliers avec une image très différente la flag ──────

class TestUCQ09_OutlierWithDifferentImage:
    def test_different_image_is_outlier(self, tmp_path):
        import shutil
        from quality_filter import QualityClassifier

        clf = QualityClassifier()
        sp_dir = tmp_path / "mixed"
        sp_dir.mkdir()

        birds = _real_bird_fixtures(5)
        for img in birds:
            shutil.copy2(img, sp_dir / img.name)

        _make_illustration(sp_dir / "illustration.jpg")
        _make_screen_photo(sp_dir / "screen.jpg")

        embeddings = clf.compute_embeddings(str(sp_dir))
        outliers = clf.detect_outliers(embeddings, sigma_threshold=1.5)

        outlier_names = [n for n, info in outliers.items() if info["is_outlier"]]
        assert any(
            "illustration" in n or "screen" in n for n in outlier_names
        ), f"Ni illustration ni screen flagés, outliers: {outlier_names}"


# ═════════════════════════════════════════════════════════════════════════
# RAPPORT — JSON par espèce avec résumé
# ═════════════════════════════════════════════════════════════════════════

# ── UC-Q10 : classify_species retourne un rapport par image ──────────────

class TestUCQ10_ClassifySpeciesReport:
    def test_returns_per_image_report(self, tmp_path):
        from quality_filter import QualityClassifier

        clf = QualityClassifier()
        sp_dir = tmp_path / "train" / "test_sp"
        sp_dir.mkdir(parents=True)
        for i in range(3):
            img = PILImage.new("RGB", (100, 100), color=(50 * i, 100, 50))
            img.save(sp_dir / f"photo_{i}.jpg")

        report = clf.classify_species(str(sp_dir))
        assert len(report) == 3
        for name, info in report.items():
            assert "category" in info
            assert "confidence" in info


# ── UC-Q11 : classify_species ignore les fichiers non-image ──────────────

class TestUCQ11_IgnoresNonImages:
    def test_skips_json_files(self, tmp_path):
        from quality_filter import QualityClassifier

        clf = QualityClassifier()
        sp_dir = tmp_path / "train" / "test_sp"
        sp_dir.mkdir(parents=True)
        img = PILImage.new("RGB", (100, 100))
        img.save(sp_dir / "photo.jpg")
        (sp_dir / "annotations.json").write_text("{}")

        report = clf.classify_species(str(sp_dir))
        assert "annotations.json" not in report
        assert "photo.jpg" in report


# ── UC-Q12 : generate_quality_report produit un JSON par espèce ──────────

class TestUCQ12_QualityReportFormat:
    def test_generates_json_report(self, tmp_path):
        from quality_filter import QualityClassifier

        clf = QualityClassifier()
        train_dir = tmp_path / "train"
        sp_dir = train_dir / "parus_major"
        sp_dir.mkdir(parents=True)
        for i in range(3):
            img = PILImage.new("RGB", (100, 100), color=(50 * i, 100, 50))
            img.save(sp_dir / f"photo_{i}.jpg")

        clf.generate_quality_report(str(train_dir), str(tmp_path / "reports"))
        report_path = tmp_path / "reports" / "parus_major.json"
        assert report_path.exists()
        with open(report_path) as f:
            data = json.load(f)
        assert "summary" in data
        assert "images" in data
        assert "outliers" in data


# ── UC-Q13 : le rapport contient un résumé avec le compte par catégorie ──

class TestUCQ13_ReportSummary:
    def test_summary_counts_categories(self, tmp_path):
        from quality_filter import QualityClassifier

        clf = QualityClassifier()
        train_dir = tmp_path / "train"
        sp_dir = train_dir / "test_sp"
        sp_dir.mkdir(parents=True)
        for i in range(5):
            img = PILImage.new("RGB", (100, 100), color=(50 * i, 100, 50))
            img.save(sp_dir / f"photo_{i}.jpg")

        clf.generate_quality_report(str(train_dir), str(tmp_path / "reports"))
        with open(tmp_path / "reports" / "test_sp.json") as f:
            data = json.load(f)
        summary = data["summary"]
        assert "total" in summary
        assert summary["total"] == 5
        total_cats = sum(v for k, v in summary.items() if k != "total")
        assert total_cats == 5


# ── UC-Q14 : le rapport inclut les outliers avec distance au centroïde ───

class TestUCQ14_ReportIncludesOutliers:
    def test_outliers_section_in_report(self, tmp_path):
        from quality_filter import QualityClassifier

        clf = QualityClassifier()
        train_dir = tmp_path / "train"
        sp_dir = train_dir / "test_sp"
        sp_dir.mkdir(parents=True)
        for i in range(5):
            img = PILImage.new("RGB", (100, 100), color=(50 * i, 100, 50))
            img.save(sp_dir / f"photo_{i}.jpg")

        clf.generate_quality_report(str(train_dir), str(tmp_path / "reports"))
        with open(tmp_path / "reports" / "test_sp.json") as f:
            data = json.load(f)
        assert "outliers" in data
        for name, info in data["outliers"].items():
            assert "distance" in info
            assert "is_outlier" in info


# ═════════════════════════════════════════════════════════════════════════
# PARALLÉLISATION — multiprocessing spawn, même résultat qu'en séquentiel
# ═════════════════════════════════════════════════════════════════════════

def _build_mini_quality_dataset(tmp_path: Path, n_species: int = 3, n_images: int = 3) -> Path:
    train_dir = tmp_path / "train"
    for i in range(n_species):
        sp_dir = train_dir / f"species_{i}"
        sp_dir.mkdir(parents=True)
        for j in range(n_images):
            img = PILImage.new("RGB", (100, 100), color=((i * 40 + j * 20) % 256, 100, 50))
            img.save(sp_dir / f"photo_{j}.jpg")
    return train_dir


# ── UC-Q15 : parallel produit les mêmes rapports que séquentiel ─────────

class TestUCQ15_ParallelSameAsSequential:
    def test_parallel_same_reports(self, tmp_path):
        from quality_filter import QualityClassifier

        train_dir = _build_mini_quality_dataset(tmp_path / "seq", n_species=3, n_images=3)
        clf = QualityClassifier()

        out_seq = tmp_path / "reports_seq"
        clf.generate_quality_report(str(train_dir), str(out_seq), workers=1)

        out_par = tmp_path / "reports_par"
        clf.generate_quality_report(str(train_dir), str(out_par), workers=2)

        for json_file in sorted(out_seq.glob("*.json")):
            with open(json_file) as f:
                data_seq = json.load(f)
            with open(out_par / json_file.name) as f:
                data_par = json.load(f)
            assert data_seq["summary"] == data_par["summary"]
            for img_name in data_seq["images"]:
                assert data_seq["images"][img_name]["category"] == data_par["images"][img_name]["category"]


# ── UC-Q16 : parallel avec resume saute les espèces déjà traitées ───────

class TestUCQ16_ParallelResume:
    def test_parallel_skips_existing_reports(self, tmp_path):
        from quality_filter import QualityClassifier

        train_dir = _build_mini_quality_dataset(tmp_path / "data", n_species=3, n_images=2)
        out = tmp_path / "reports"
        out.mkdir(parents=True)

        # Pré-créer un rapport pour species_0
        existing = {"summary": {"total": 2, "good": 2}, "images": {}, "outliers": {}}
        with open(out / "species_0.json", "w") as f:
            json.dump(existing, f)

        clf = QualityClassifier()
        clf.generate_quality_report(str(train_dir), str(out), workers=2, resume=True)

        # species_0 ne doit pas avoir été écrasé
        with open(out / "species_0.json") as f:
            data = json.load(f)
        assert data == existing

        # Les autres doivent exister
        assert (out / "species_1.json").exists()
        assert (out / "species_2.json").exists()


# ── UC-Q17 : parallel retourne des stats globales ───────────────────────

class TestUCQ17_ParallelStats:
    def test_returns_global_stats(self, tmp_path):
        from quality_filter import QualityClassifier

        train_dir = _build_mini_quality_dataset(tmp_path / "data", n_species=2, n_images=3)
        clf = QualityClassifier()
        out = tmp_path / "reports"

        stats = clf.generate_quality_report(str(train_dir), str(out), workers=2)
        assert "processed" in stats
        assert "skipped" in stats
        assert stats["processed"] == 2
        assert stats["total_images"] == 6


# ═════════════════════════════════════════════════════════════════════════
# NETTOYAGE — suppression des images filtrées + mise à jour annotations.json
# ═════════════════════════════════════════════════════════════════════════

def _build_dataset_with_reports(tmp_path: Path, dataset_root: Path | None = None) -> tuple[Path, Path]:
    """Crée un dataset + rapports simulés pour tester le nettoyage."""
    train_dir = tmp_path / "train"
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True)

    sp_dir = train_dir / "parus_major"
    sp_dir.mkdir(parents=True)
    annotations = {}
    for i in range(5):
        name = f"photo_{i}.jpg"
        img = PILImage.new("RGB", (100, 100), color=(50 * i, 100, 50))
        img.save(sp_dir / name)
        annotations[name] = {"bbox": [10, 10, 50, 50], "score": 0.9}
    with open(sp_dir / "annotations.json", "w") as f:
        json.dump(annotations, f)

    report = {
        "summary": {"total": 5, "good": 3, "dead_specimen": 1, "illustration": 1,
                     "screen_scan": 0, "not_bird": 0, "poor_quality": 0},
        "images": {
            "photo_0.jpg": {"category": "good", "confidence": 0.9, "scores": {},
                            "quality": {"detection_score": 0.9, "bbox_pct": 25.0, "sharpness": 100.0, "composite_score": 0.7}},
            "photo_1.jpg": {"category": "good", "confidence": 0.85, "scores": {},
                            "quality": {"detection_score": 0.9, "bbox_pct": 25.0, "sharpness": 80.0, "composite_score": 0.65}},
            "photo_2.jpg": {"category": "dead_specimen", "confidence": 0.7, "scores": {},
                            "quality": {"detection_score": 0.9, "bbox_pct": 25.0, "sharpness": 60.0, "composite_score": 0.6}},
            "photo_3.jpg": {"category": "illustration", "confidence": 0.8, "scores": {},
                            "quality": {"detection_score": 0.9, "bbox_pct": 25.0, "sharpness": 40.0, "composite_score": 0.55}},
            "photo_4.jpg": {"category": "good", "confidence": 0.88, "scores": {},
                            "quality": {"detection_score": 0.9, "bbox_pct": 25.0, "sharpness": 120.0, "composite_score": 0.75}},
        },
        "outliers": {
            "photo_0.jpg": {"distance": 0.02, "is_outlier": False},
            "photo_1.jpg": {"distance": 0.03, "is_outlier": False},
            "photo_2.jpg": {"distance": 0.15, "is_outlier": True},
            "photo_3.jpg": {"distance": 0.20, "is_outlier": True},
            "photo_4.jpg": {"distance": 0.01, "is_outlier": False},
        },
    }
    with open(reports_dir / "parus_major.json", "w") as f:
        json.dump(report, f)

    # Créer label_map.json et metadata.json au niveau dataset_root si fourni
    if dataset_root is not None:
        dataset_root.mkdir(parents=True, exist_ok=True)
        label_map = {"parus_major": 0}
        with open(dataset_root / "label_map.json", "w") as f:
            json.dump(label_map, f)
        metadata = {
            "parus_major": {
                "slug": "parus_major",
                "scientific_name": "Parus major",
                "family": "Paridae",
                "english_name": "Great Tit",
                "french_name": "Mésange charbonnière",
            }
        }
        with open(dataset_root / "metadata.json", "w") as f:
            json.dump(metadata, f)

    return train_dir, reports_dir


def _build_multi_species_dataset(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Crée un dataset 2 espèces : une entièrement mauvaise, une bonne."""
    dataset_root = tmp_path / "europe"
    train_dir = dataset_root / "train"
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True)

    # Espèce 1 : toutes les images sont mauvaises
    sp1 = train_dir / "species_bad"
    sp1.mkdir(parents=True)
    ann1 = {}
    for i in range(3):
        name = f"photo_{i}.jpg"
        PILImage.new("RGB", (100, 100)).save(sp1 / name)
        ann1[name] = {"bbox": [10, 10, 50, 50], "score": 0.9}
    with open(sp1 / "annotations.json", "w") as f:
        json.dump(ann1, f)

    report1 = {
        "summary": {"total": 3, "good": 0, "dead_specimen": 3, "illustration": 0,
                     "screen_scan": 0, "not_bird": 0, "poor_quality": 0},
        "images": {
            f"photo_{i}.jpg": {"category": "dead_specimen", "confidence": 0.8, "scores": {}}
            for i in range(3)
        },
        "outliers": {
            f"photo_{i}.jpg": {"distance": 0.05, "is_outlier": False} for i in range(3)
        },
    }
    with open(reports_dir / "species_bad.json", "w") as f:
        json.dump(report1, f)

    # Espèce 2 : toutes les images sont bonnes
    sp2 = train_dir / "species_good"
    sp2.mkdir(parents=True)
    ann2 = {}
    for i in range(4):
        name = f"photo_{i}.jpg"
        PILImage.new("RGB", (100, 100)).save(sp2 / name)
        ann2[name] = {"bbox": [10, 10, 50, 50], "score": 0.9}
    with open(sp2 / "annotations.json", "w") as f:
        json.dump(ann2, f)

    report2 = {
        "summary": {"total": 4, "good": 4, "dead_specimen": 0, "illustration": 0,
                     "screen_scan": 0, "not_bird": 0, "poor_quality": 0},
        "images": {
            f"photo_{i}.jpg": {"category": "good", "confidence": 0.9, "scores": {}}
            for i in range(4)
        },
        "outliers": {
            f"photo_{i}.jpg": {"distance": 0.02, "is_outlier": False} for i in range(4)
        },
    }
    with open(reports_dir / "species_good.json", "w") as f:
        json.dump(report2, f)

    # label_map et metadata
    label_map = {"species_bad": 0, "species_good": 1}
    with open(dataset_root / "label_map.json", "w") as f:
        json.dump(label_map, f)

    metadata = {
        "species_bad": {"slug": "species_bad", "scientific_name": "Species bad",
                        "family": "Testidae", "english_name": "Bad bird", "french_name": "Mauvais oiseau"},
        "species_good": {"slug": "species_good", "scientific_name": "Species good",
                         "family": "Testidae", "english_name": "Good bird", "french_name": "Bon oiseau"},
    }
    with open(dataset_root / "metadata.json", "w") as f:
        json.dump(metadata, f)

    return train_dir, reports_dir, dataset_root


# ── UC-Q18 : apply_filter déplace les images rejetées dans rejected_dir ──

class TestUCQ18_ApplyFilterMovesImages:
    def test_moves_bad_category_images(self, tmp_path):
        from quality_filter import apply_filter

        train_dir, reports_dir = _build_dataset_with_reports(tmp_path)
        rejected = tmp_path / "rejected"
        reject = ["dead_specimen", "illustration", "screen_scan", "not_bird", "poor_quality"]
        stats = apply_filter(str(train_dir), str(reports_dir),
                             reject_categories=reject, rejected_dir=str(rejected))

        sp_dir = train_dir / "parus_major"
        remaining = [f.name for f in sp_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        assert "photo_0.jpg" in remaining
        assert "photo_1.jpg" in remaining
        assert "photo_4.jpg" in remaining
        assert "photo_2.jpg" not in remaining
        assert "photo_3.jpg" not in remaining

        rej_sp = rejected / "parus_major"
        assert (rej_sp / "photo_2.jpg").exists()
        assert (rej_sp / "photo_3.jpg").exists()


# ── UC-Q19 : apply_filter met à jour annotations.json (source et rejected)

class TestUCQ19_ApplyFilterUpdatesAnnotations:
    def test_annotations_updated(self, tmp_path):
        from quality_filter import apply_filter

        train_dir, reports_dir = _build_dataset_with_reports(tmp_path)
        rejected = tmp_path / "rejected"
        reject = ["dead_specimen", "illustration"]
        apply_filter(str(train_dir), str(reports_dir),
                     reject_categories=reject, rejected_dir=str(rejected))

        with open(train_dir / "parus_major" / "annotations.json") as f:
            annotations = json.load(f)
        assert "photo_2.jpg" not in annotations
        assert "photo_3.jpg" not in annotations
        assert "photo_0.jpg" in annotations
        assert len(annotations) == 3

        with open(rejected / "parus_major" / "annotations.json") as f:
            rej_ann = json.load(f)
        assert "photo_2.jpg" in rej_ann
        assert "photo_3.jpg" in rej_ann
        assert len(rej_ann) == 2


# ── UC-Q20 : apply_filter déplace aussi les outliers si demandé ──────────

class TestUCQ20_ApplyFilterMovesOutliers:
    def test_moves_outliers_when_requested(self, tmp_path):
        from quality_filter import apply_filter

        train_dir, reports_dir = _build_dataset_with_reports(tmp_path)
        rejected = tmp_path / "rejected"
        stats = apply_filter(
            str(train_dir), str(reports_dir),
            reject_categories=["dead_specimen", "illustration"],
            remove_outliers=True,
            rejected_dir=str(rejected),
        )

        sp_dir = train_dir / "parus_major"
        remaining = [f.name for f in sp_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        assert "photo_2.jpg" not in remaining
        assert "photo_3.jpg" not in remaining
        assert stats["removed"] >= 2

        rej_sp = rejected / "parus_major"
        assert (rej_sp / "photo_2.jpg").exists()
        assert (rej_sp / "photo_3.jpg").exists()


# ── UC-Q21 : apply_filter retourne des stats (removed, kept)

class TestUCQ21_ApplyFilterStats:
    def test_returns_stats(self, tmp_path):
        from quality_filter import apply_filter

        train_dir, reports_dir = _build_dataset_with_reports(tmp_path)
        rejected = tmp_path / "rejected"
        stats = apply_filter(
            str(train_dir), str(reports_dir),
            reject_categories=["dead_specimen", "illustration"],
            rejected_dir=str(rejected),
        )
        assert "removed" in stats
        assert "kept" in stats
        assert stats["removed"] == 2
        assert stats["kept"] == 3


# ═════════════════════════════════════════════════════════════════════════
# MISE À JOUR label_map.json / metadata.json
# ═════════════════════════════════════════════════════════════════════════

# ── UC-Q22 : si une espèce perd toutes ses images, elle est retirée de
#             label_map.json et metadata.json, et les IDs sont re-numérotés

class TestUCQ22_RemovesEmptySpeciesFromLabelMap:
    def test_empty_species_removed_from_label_map(self, tmp_path):
        from quality_filter import apply_filter

        train_dir, reports_dir, dataset_root = _build_multi_species_dataset(tmp_path)
        rejected = tmp_path / "rejected"
        apply_filter(
            str(train_dir), str(reports_dir),
            reject_categories=["dead_specimen"],
            dataset_root=str(dataset_root),
            rejected_dir=str(rejected),
        )

        with open(dataset_root / "label_map.json") as f:
            label_map = json.load(f)
        assert "species_bad" not in label_map
        assert "species_good" in label_map
        assert label_map["species_good"] == 0


# ── UC-Q23 : metadata.json est aussi nettoyé ────────────────────────────

class TestUCQ23_RemovesEmptySpeciesFromMetadata:
    def test_empty_species_removed_from_metadata(self, tmp_path):
        from quality_filter import apply_filter

        train_dir, reports_dir, dataset_root = _build_multi_species_dataset(tmp_path)
        rejected = tmp_path / "rejected"
        apply_filter(
            str(train_dir), str(reports_dir),
            reject_categories=["dead_specimen"],
            dataset_root=str(dataset_root),
            rejected_dir=str(rejected),
        )

        with open(dataset_root / "metadata.json") as f:
            metadata = json.load(f)
        assert "species_bad" not in metadata
        assert "species_good" in metadata


# ── UC-Q24 : espèce vidée → dossier source vide, images dans rejected ───

class TestUCQ24_EmptySpeciesMovedToRejected:
    def test_empty_species_moved_to_rejected(self, tmp_path):
        from quality_filter import apply_filter

        train_dir, reports_dir, dataset_root = _build_multi_species_dataset(tmp_path)
        rejected = tmp_path / "rejected"
        apply_filter(
            str(train_dir), str(reports_dir),
            reject_categories=["dead_specimen"],
            dataset_root=str(dataset_root),
            rejected_dir=str(rejected),
        )

        assert not (train_dir / "species_bad").exists()
        assert (train_dir / "species_good").exists()
        # Les images rejetées sont dans rejected/
        rej_sp = rejected / "species_bad"
        assert rej_sp.exists()
        assert len(list(rej_sp.glob("*.jpg"))) == 3


# ── UC-Q25 : si aucune espèce n'est vidée, label_map et metadata ne
#             changent pas ────────────────────────────────────────────────

class TestUCQ25_NoChangeIfNoSpeciesRemoved:
    def test_label_map_unchanged_when_no_species_emptied(self, tmp_path):
        from quality_filter import apply_filter

        train_dir, reports_dir = _build_dataset_with_reports(tmp_path, dataset_root=tmp_path / "europe")
        dataset_root = tmp_path / "europe"
        rejected = tmp_path / "rejected"

        with open(dataset_root / "label_map.json") as f:
            original_lm = json.load(f)
        with open(dataset_root / "metadata.json") as f:
            original_md = json.load(f)

        apply_filter(
            str(train_dir), str(reports_dir),
            reject_categories=["dead_specimen", "illustration"],
            dataset_root=str(dataset_root),
            rejected_dir=str(rejected),
        )

        with open(dataset_root / "label_map.json") as f:
            updated_lm = json.load(f)
        with open(dataset_root / "metadata.json") as f:
            updated_md = json.load(f)

        assert updated_lm == original_lm
        assert updated_md == original_md


# ═════════════════════════════════════════════════════════════════════════
# FIX 1 : filter_small_bbox intégré dans apply_filter
# ═════════════════════════════════════════════════════════════════════════

# ── UC-Q26 : apply_filter retire les images avec bbox trop petite ────────

class TestUCQ26_ApplyFilterSmallBbox:
    def test_removes_small_bbox_images(self, tmp_path):
        from quality_filter import apply_filter

        train_dir = tmp_path / "train"
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir(parents=True)

        sp_dir = train_dir / "parus_major"
        sp_dir.mkdir(parents=True)
        # photo_0 : bbox 100x100 sur image 200x200 = 25% → OK
        PILImage.new("RGB", (200, 200)).save(sp_dir / "photo_0.jpg")
        # photo_1 : bbox 5x5 sur image 200x200 = 0.0625% → trop petit
        PILImage.new("RGB", (200, 200)).save(sp_dir / "photo_1.jpg")

        annotations = {
            "photo_0.jpg": {"bbox": [10, 10, 100, 100], "score": 0.9},
            "photo_1.jpg": {"bbox": [50, 50, 5, 5], "score": 0.8},
        }
        with open(sp_dir / "annotations.json", "w") as f:
            json.dump(annotations, f)

        report = {
            "summary": {"total": 2, "good": 2},
            "images": {
                "photo_0.jpg": {"category": "good", "confidence": 0.9, "scores": {}},
                "photo_1.jpg": {"category": "good", "confidence": 0.85, "scores": {}},
            },
            "outliers": {
                "photo_0.jpg": {"distance": 0.02, "is_outlier": False},
                "photo_1.jpg": {"distance": 0.03, "is_outlier": False},
            },
        }
        with open(reports_dir / "parus_major.json", "w") as f:
            json.dump(report, f)

        rejected = tmp_path / "rejected"
        stats = apply_filter(
            str(train_dir), str(reports_dir),
            reject_categories=[],
            min_bbox_pct=5.0,
            rejected_dir=str(rejected),
        )

        remaining = [f.name for f in sp_dir.iterdir() if f.suffix.lower() in (".jpg",)]
        assert "photo_0.jpg" in remaining
        assert "photo_1.jpg" not in remaining
        assert stats["removed"] == 1
        assert (rejected / "parus_major" / "photo_1.jpg").exists()


# ═════════════════════════════════════════════════════════════════════════
# FIX 2 : compute_embeddings réutilise les features déjà calculées
# ═════════════════════════════════════════════════════════════════════════

# ── UC-Q27 : classify_and_embed retourne classifications + embeddings ────

class TestUCQ27_ClassifyAndEmbed:
    def test_returns_both(self, tmp_path):
        from quality_filter import QualityClassifier

        clf = QualityClassifier()
        sp_dir = tmp_path / "sp"
        sp_dir.mkdir()
        for i in range(3):
            PILImage.new("RGB", (100, 100), color=(50 * i, 100, 50)).save(sp_dir / f"photo_{i}.jpg")

        classifications, embeddings = clf.classify_and_embed(str(sp_dir))
        assert len(classifications) == 3
        assert len(embeddings) == 3
        for name in classifications:
            assert name in embeddings
            assert "category" in classifications[name]
            assert embeddings[name].shape[0] > 0


# ═════════════════════════════════════════════════════════════════════════
# FIX 4 : apply_filter_dataset traite les 3 splits
# ═════════════════════════════════════════════════════════════════════════

# ── UC-Q28 : apply_filter_dataset filtre train, validation et test ───────

class TestUCQ28_ApplyFilterDataset:
    def test_filters_all_splits(self, tmp_path):
        from quality_filter import apply_filter_dataset

        dataset_root = tmp_path / "europe"
        reports_root = tmp_path / "reports"

        for split in ("train", "validation", "test"):
            sp_dir = dataset_root / split / "parus_major"
            sp_dir.mkdir(parents=True)
            for i in range(3):
                PILImage.new("RGB", (100, 100)).save(sp_dir / f"photo_{i}.jpg")

            annotations = {f"photo_{i}.jpg": {"bbox": [10, 10, 50, 50], "score": 0.9} for i in range(3)}
            with open(sp_dir / "annotations.json", "w") as f:
                json.dump(annotations, f)

            rdir = reports_root / split
            rdir.mkdir(parents=True)
            report = {
                "summary": {"total": 3, "good": 2, "dead_specimen": 1,
                            "illustration": 0, "screen_scan": 0, "not_bird": 0, "poor_quality": 0},
                "images": {
                    "photo_0.jpg": {"category": "good", "confidence": 0.9, "scores": {}},
                    "photo_1.jpg": {"category": "dead_specimen", "confidence": 0.7, "scores": {}},
                    "photo_2.jpg": {"category": "good", "confidence": 0.85, "scores": {}},
                },
                "outliers": {f"photo_{i}.jpg": {"distance": 0.02, "is_outlier": False} for i in range(3)},
            }
            with open(rdir / "parus_major.json", "w") as f:
                json.dump(report, f)

        label_map = {"parus_major": 0}
        with open(dataset_root / "label_map.json", "w") as f:
            json.dump(label_map, f)
        metadata = {"parus_major": {"slug": "parus_major"}}
        with open(dataset_root / "metadata.json", "w") as f:
            json.dump(metadata, f)

        rejected_root = tmp_path / "europe_rejected"
        stats = apply_filter_dataset(
            str(dataset_root), str(reports_root),
            reject_categories=["dead_specimen"],
            rejected_root=str(rejected_root),
        )

        for split in ("train", "validation", "test"):
            sp = dataset_root / split / "parus_major"
            remaining = [f.name for f in sp.iterdir() if f.suffix.lower() in (".jpg",)]
            assert "photo_1.jpg" not in remaining, f"{split}: photo_1.jpg pas supprimée"
            assert len(remaining) == 2

            rej_sp = rejected_root / split / "parus_major"
            assert (rej_sp / "photo_1.jpg").exists(), f"{split}: photo_1.jpg pas dans rejected"

        assert stats["removed"] == 3
        assert stats["kept"] == 6


# ═════════════════════════════════════════════════════════════════════════
# BATCHING — classify_and_embed_batched produit le même résultat, plus vite
# ═════════════════════════════════════════════════════════════════════════

# ── UC-Q29 : classify_and_embed_batched retourne le même résultat ────────

class TestUCQ29_BatchedSameAsSequential:
    def test_batched_same_results(self, tmp_path):
        import shutil
        from quality_filter import QualityClassifier

        clf = QualityClassifier()
        sp_dir = tmp_path / "sp"
        sp_dir.mkdir()
        for img in _real_bird_fixtures(5):
            shutil.copy2(img, sp_dir / img.name)

        cls_seq, emb_seq = clf.classify_and_embed(str(sp_dir))
        cls_bat, emb_bat = clf.classify_and_embed_batched(str(sp_dir), batch_size=2)

        assert set(cls_seq.keys()) == set(cls_bat.keys())
        for name in cls_seq:
            assert cls_seq[name]["category"] == cls_bat[name]["category"], (
                f"{name}: {cls_seq[name]['category']} != {cls_bat[name]['category']}"
            )
        assert set(emb_seq.keys()) == set(emb_bat.keys())


# ── UC-Q30 : classify_and_embed_batched gère batch_size > n_images ───────

class TestUCQ30_BatchedLargerThanDataset:
    def test_batch_larger_than_images(self, tmp_path):
        from quality_filter import QualityClassifier

        clf = QualityClassifier()
        sp_dir = tmp_path / "sp"
        sp_dir.mkdir()
        for i in range(3):
            PILImage.new("RGB", (100, 100), color=(50 * i, 100, 50)).save(sp_dir / f"photo_{i}.jpg")

        cls, emb = clf.classify_and_embed_batched(str(sp_dir), batch_size=64)
        assert len(cls) == 3
        assert len(emb) == 3


# ── UC-Q31 : classify_and_embed_batched ignore les non-images ────────────

class TestUCQ31_BatchedIgnoresNonImages:
    def test_skips_json(self, tmp_path):
        from quality_filter import QualityClassifier

        clf = QualityClassifier()
        sp_dir = tmp_path / "sp"
        sp_dir.mkdir()
        PILImage.new("RGB", (100, 100)).save(sp_dir / "photo.jpg")
        (sp_dir / "annotations.json").write_text("{}")

        cls, emb = clf.classify_and_embed_batched(str(sp_dir), batch_size=8)
        assert "annotations.json" not in cls
        assert "photo.jpg" in cls


# ── UC-Q32 : float16 produit le même classement que float32 ─────────────

class TestUCQ32_Float16SameCategory:
    def test_fp16_same_classification(self):
        from quality_filter import QualityClassifier

        clf_fp32 = QualityClassifier()
        clf_fp16 = QualityClassifier(use_fp16=True)

        birds = _real_bird_fixtures(3)
        for img in birds:
            r32 = clf_fp32.classify_image(str(img))
            r16 = clf_fp16.classify_image(str(img))
            assert r32["category"] == r16["category"], (
                f"{img.name}: fp32={r32['category']} != fp16={r16['category']}"
            )


# ═════════════════════════════════════════════════════════════════════════
# THREAD WORKERS — parallélisation sûre sur GPU (MPS/CUDA)
# ═════════════════════════════════════════════════════════════════════════

# ── UC-Q33 : thread workers produisent les mêmes rapports que séquentiel ──

class TestUCQ33_ThreadWorkersSameAsSequential:
    def test_thread_workers_same_reports(self, tmp_path):
        from quality_filter import QualityClassifier

        train_dir = _build_mini_quality_dataset(tmp_path / "data", n_species=3, n_images=3)
        clf = QualityClassifier()

        out_seq = tmp_path / "reports_seq"
        clf.generate_quality_report(str(train_dir), str(out_seq), workers=1)

        out_thr = tmp_path / "reports_thr"
        clf.generate_quality_report(str(train_dir), str(out_thr), workers=3)

        for json_file in sorted(out_seq.glob("*.json")):
            with open(json_file) as f:
                data_seq = json.load(f)
            with open(out_thr / json_file.name) as f:
                data_thr = json.load(f)
            assert data_seq["summary"] == data_thr["summary"]
            for img_name in data_seq["images"]:
                assert data_seq["images"][img_name]["category"] == data_thr["images"][img_name]["category"]


# ── UC-Q34 : device explicite est respecté ────────────────────────────────

# ── UC-Q35 : batch_size du constructeur est propagé aux méthodes ──────────

class TestUCQ35_BatchSizePropagated:
    def test_constructor_batch_size_used(self, tmp_path):
        from quality_filter import QualityClassifier

        clf = QualityClassifier(batch_size=2)
        assert clf.batch_size == 2

        sp_dir = tmp_path / "sp"
        sp_dir.mkdir()
        for i in range(5):
            PILImage.new("RGB", (100, 100), color=(50 * i, 100, 50)).save(sp_dir / f"photo_{i}.jpg")

        out = tmp_path / "reports"
        stats = clf.generate_quality_report(str(tmp_path), str(out), workers=1)
        assert stats["processed"] == 1
        assert stats["total_images"] == 5


# ── UC-Q34 : device explicite est respecté ────────────────────────────────

class TestUCQ34_ExplicitDevice:
    def test_force_cpu_device(self):
        from quality_filter import QualityClassifier

        clf = QualityClassifier(device="cpu")
        assert clf.device.type == "cpu"

    def test_reject_margin_keeps_ambiguous_as_good(self):
        """Avec la marge par défaut (0.005), les vraies photos d'oiseaux
        doivent rester classées good — la marge protège les cas ambigus."""
        from quality_filter import QualityClassifier

        clf = QualityClassifier()
        birds = _real_bird_fixtures(5)
        results = [clf.classify_image(str(img)) for img in birds]
        good_count = sum(1 for r in results if r["category"] == "good")
        assert good_count >= 4, (
            f"Avec reject_margin=0.005, attendu >=4 good sur 5 vraies photos, got {good_count}"
        )

    def test_reject_margin_zero_is_argmax(self):
        """Avec reject_margin=0, le comportement est l'argmax pur (ancien comportement)."""
        from quality_filter import QualityClassifier

        clf = QualityClassifier(reject_margin=0.0)
        result = clf.classify_image(str(_real_bird_fixtures(1)[0]))
        assert result["category"] in ("good", "dead_specimen", "illustration",
                                       "screen_scan", "not_bird", "poor_quality")

    def test_default_device_is_auto(self):
        import torch
        from quality_filter import QualityClassifier

        clf = QualityClassifier()
        if torch.backends.mps.is_available():
            assert clf.device.type == "mps"
        elif torch.cuda.is_available():
            assert clf.device.type == "cuda"
        else:
            assert clf.device.type == "cpu"


# ═════════════════════════════════════════════════════════════════════════
# DÉDUPLICATION VISUELLE — near-duplicate detection par similarité cosinus
# ═════════════════════════════════════════════════════════════════════════

# ── UC-Q36 : images distinctes ne sont pas des duplicates ────────────────

class TestUCQ36_NoDuplicatesBelowThreshold:
    def test_distinct_images_no_duplicates(self, tmp_path):
        from quality_filter import QualityClassifier

        clf = QualityClassifier()
        sp_dir = tmp_path / "species_a"
        sp_dir.mkdir()
        for i in range(5):
            PILImage.new("RGB", (224, 224), color=(i * 50, 100, 200 - i * 30)).save(
                sp_dir / f"photo_{i}.jpg"
            )

        classifications, embeddings = clf.classify_and_embed_batched(str(sp_dir))
        duplicates = clf.detect_near_duplicates(embeddings, classifications, threshold=0.99)
        assert duplicates == []


# ── UC-Q37 : images identiques détectées comme duplicates ────────────────

class TestUCQ37_IdenticalImagesDetected:
    def test_identical_images_are_duplicates(self, tmp_path):
        import shutil
        from quality_filter import QualityClassifier

        clf = QualityClassifier()
        sp_dir = tmp_path / "species_a"
        sp_dir.mkdir()

        source = _real_bird_fixtures(1)[0]
        for i in range(3):
            shutil.copy(str(source), sp_dir / f"copy_{i}.jpg")

        classifications, embeddings = clf.classify_and_embed_batched(str(sp_dir))
        duplicates = clf.detect_near_duplicates(embeddings, classifications, threshold=0.95)

        assert len(duplicates) >= 1
        removed_names = {d["removed"] for d in duplicates}
        kept_names = {d["kept"] for d in duplicates}
        assert len(removed_names) == 2
        assert len(kept_names) == 1
        for d in duplicates:
            assert d["similarity"] > 0.95


# ── UC-Q38 : le duplicate avec le meilleur score good est gardé ──────────

class TestUCQ38_KeepsBestCompositeScore:
    def test_keeps_higher_composite_score(self):
        import torch
        from quality_filter import QualityClassifier

        clf = QualityClassifier()

        vec = torch.randn(768)
        vec = vec / vec.norm()
        embeddings = {
            "img_a.jpg": vec.clone(),
            "img_b.jpg": vec.clone(),
        }
        classifications = {
            "img_a.jpg": {"category": "good", "confidence": 0.7, "scores": {"good": 0.7}, "quality": {"composite_score": 0.3}},
            "img_b.jpg": {"category": "good", "confidence": 0.9, "scores": {"good": 0.9}, "quality": {"composite_score": 0.8}},
        }

        duplicates = clf.detect_near_duplicates(embeddings, classifications, threshold=0.95)
        assert len(duplicates) == 1
        assert duplicates[0]["kept"] == "img_b.jpg"
        assert duplicates[0]["removed"] == "img_a.jpg"


# ── UC-Q39 : une seule image, pas de duplicate ──────────────────────────

class TestUCQ39_SingleImageNoDuplicates:
    def test_single_image_returns_empty(self):
        import torch
        from quality_filter import QualityClassifier

        clf = QualityClassifier()
        embeddings = {"only.jpg": torch.randn(768)}
        classifications = {"only.jpg": {"category": "good", "confidence": 0.9, "scores": {"good": 0.9}}}

        duplicates = clf.detect_near_duplicates(embeddings, classifications)
        assert duplicates == []


# ── UC-Q40 : generate_quality_report inclut les duplicates avec seuil ────

class TestUCQ40_ReportIncludesDuplicates:
    def test_duplicates_in_report(self, tmp_path):
        import shutil
        from quality_filter import QualityClassifier

        clf = QualityClassifier()
        train_dir = tmp_path / "train"
        sp_dir = train_dir / "species_dup"
        sp_dir.mkdir(parents=True)

        source = _real_bird_fixtures(1)[0]
        for i in range(3):
            shutil.copy(str(source), sp_dir / f"copy_{i}.jpg")

        out = tmp_path / "reports"
        clf.generate_quality_report(str(train_dir), str(out), duplicate_threshold=0.90)

        with open(out / "species_dup.json") as f:
            report = json.load(f)
        assert "duplicates" in report
        assert len(report["duplicates"]) >= 1
        for d in report["duplicates"]:
            assert "kept" in d
            assert "removed" in d
            assert "similarity" in d


# ── UC-Q41 : pas de clé duplicates par défaut ───────────────────────────

class TestUCQ41_NoDuplicatesKeyByDefault:
    def test_no_duplicates_when_disabled(self, tmp_path):
        from quality_filter import QualityClassifier

        train_dir = _build_mini_quality_dataset(tmp_path / "data", n_species=1, n_images=3)
        clf = QualityClassifier()
        out = tmp_path / "reports"

        clf.generate_quality_report(str(train_dir), str(out))

        for json_file in out.glob("*.json"):
            with open(json_file) as f:
                report = json.load(f)
            assert "duplicates" not in report


# ── UC-Q42 : apply_filter supprime les duplicates ───────────────────────

class TestUCQ42_ApplyFilterRemovesDuplicates:
    def test_removes_duplicate_images(self, tmp_path):
        from quality_filter import apply_filter

        train_dir, reports_dir = _build_dataset_with_reports(tmp_path)
        # Ajouter une section duplicates au rapport
        report_file = reports_dir / "parus_major.json"
        with open(report_file) as f:
            report = json.load(f)
        report["duplicates"] = [
            {"kept": "photo_0.jpg", "removed": "photo_1.jpg", "similarity": 0.97}
        ]
        with open(report_file, "w") as f:
            json.dump(report, f)

        rejected = tmp_path / "rejected"
        stats = apply_filter(
            str(train_dir), str(reports_dir),
            reject_categories=[],
            remove_duplicates=True,
            rejected_dir=str(rejected),
        )

        sp_dir = train_dir / "parus_major"
        remaining = [f.name for f in sp_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        assert "photo_0.jpg" in remaining
        assert "photo_1.jpg" not in remaining
        assert (rejected / "parus_major" / "photo_1.jpg").exists()
        assert stats["removed"] == 1


# ═════════════════════════════════════════════════════════════════════════
# VÉRIFICATION CROISÉE DES LABELS — cross-species mislabel detection
# ═════════════════════════════════════════════════════════════════════════

# ── helpers mislabel ────────────────────────────────────────────────────

def _build_dataset_with_embeddings(tmp_path, n_species=2, n_images=5, mislabel_spec=None):
    """Crée un dataset avec des embeddings synthétiques en .pt.

    mislabel_spec = {"species": idx_espèce, "image": idx_image} place un
    embedding de la mauvaise espèce pour simuler un mislabel.
    """
    import torch

    split_dir = tmp_path / "train"
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True)

    dim = 768
    for sp_idx in range(n_species):
        sp_name = f"species_{sp_idx}"
        sp_dir = split_dir / sp_name
        sp_dir.mkdir(parents=True)

        center = torch.zeros(dim)
        center[sp_idx] = 1.0

        embeddings = {}
        classifications = {}
        for img_idx in range(n_images):
            name = f"photo_{img_idx}.jpg"
            PILImage.new("RGB", (100, 100), color=((sp_idx * 50 + img_idx * 20) % 256, 100, 50)).save(sp_dir / name)

            if mislabel_spec and mislabel_spec["species"] == sp_idx and mislabel_spec["image"] == img_idx:
                other_sp = (sp_idx + 1) % n_species
                vec = torch.zeros(dim)
                vec[other_sp] = 1.0
                vec = vec + torch.randn(dim) * 0.01
            else:
                vec = center + torch.randn(dim) * 0.05

            vec = vec / vec.norm()
            embeddings[name] = vec
            classifications[name] = {"category": "good", "confidence": 0.9, "scores": {"good": 0.9}}

        torch.save(embeddings, reports_dir / f"{sp_name}.pt")

        report = {
            "summary": {"total": n_images, "good": n_images},
            "images": classifications,
            "outliers": {f"photo_{i}.jpg": {"distance": 0.02, "is_outlier": False} for i in range(n_images)},
        }
        with open(reports_dir / f"{sp_name}.json", "w") as f:
            json.dump(report, f)

    return split_dir, reports_dir


# ── UC-Q43 : une seule espèce, pas de mislabel possible ─────────────────

class TestUCQ43_SingleSpeciesNoMislabel:
    def test_single_species_returns_empty(self, tmp_path):
        from quality_filter import detect_mislabeled

        split_dir, reports_dir = _build_dataset_with_embeddings(tmp_path, n_species=1, n_images=5)
        result = detect_mislabeled(str(split_dir), str(reports_dir))
        assert result == {}


# ── UC-Q44 : image mal étiquetée détectée ────────────────────────────────

class TestUCQ44_MislabelDetected:
    def test_detects_misplaced_image(self, tmp_path):
        from quality_filter import detect_mislabeled

        split_dir, reports_dir = _build_dataset_with_embeddings(
            tmp_path, n_species=2, n_images=5,
            mislabel_spec={"species": 0, "image": 0},
        )

        result = detect_mislabeled(str(split_dir), str(reports_dir), margin=0.1)
        assert "species_0" in result
        assert "photo_0.jpg" in result["species_0"]
        assert result["species_0"]["photo_0.jpg"]["suspected"] is True
        assert result["species_0"]["photo_0.jpg"]["nearest_species"] == "species_1"


# ── UC-Q45 : images correctement étiquetées non flaggées ────────────────

class TestUCQ45_CorrectLabelsNotFlagged:
    def test_correct_images_not_suspected(self, tmp_path):
        from quality_filter import detect_mislabeled

        split_dir, reports_dir = _build_dataset_with_embeddings(
            tmp_path, n_species=2, n_images=5,
            mislabel_spec={"species": 0, "image": 0},
        )

        result = detect_mislabeled(str(split_dir), str(reports_dir), margin=0.1)
        for img_idx in range(1, 5):
            name = f"photo_{img_idx}.jpg"
            assert result["species_0"][name]["suspected"] is False


# ── UC-Q46 : structure du résultat mislabel ──────────────────────────────

class TestUCQ46_MislabelResultStructure:
    def test_result_has_expected_keys(self, tmp_path):
        from quality_filter import detect_mislabeled

        split_dir, reports_dir = _build_dataset_with_embeddings(tmp_path, n_species=2, n_images=3)

        result = detect_mislabeled(str(split_dir), str(reports_dir))
        for species, images in result.items():
            for name, info in images.items():
                assert "own_distance" in info
                assert "nearest_species" in info
                assert "nearest_distance" in info
                assert "suspected" in info
                assert isinstance(info["own_distance"], float)
                assert isinstance(info["nearest_distance"], float)
                assert info["own_distance"] >= 0
                assert info["nearest_distance"] >= 0


# ── UC-Q47 : generate_quality_report sauvegarde les .pt ──────────────────

class TestUCQ47_ReportSavesEmbeddings:
    def test_pt_files_saved(self, tmp_path):
        import torch
        from quality_filter import QualityClassifier

        clf = QualityClassifier()
        train_dir = _build_mini_quality_dataset(tmp_path / "data", n_species=2, n_images=3)
        out = tmp_path / "reports"

        clf.generate_quality_report(str(train_dir), str(out))

        pt_files = list(out.glob("*.pt"))
        assert len(pt_files) == 2
        for pt_file in pt_files:
            data = torch.load(pt_file, weights_only=True)
            assert isinstance(data, dict)
            assert len(data) == 3
            for name, tensor in data.items():
                assert tensor.shape[0] == 768


# ── UC-Q48 : apply_filter supprime les images mislabeled ─────────────────

class TestUCQ48_ApplyFilterRemovesMislabeled:
    def test_removes_mislabeled_images(self, tmp_path):
        from quality_filter import apply_filter

        split_dir, reports_dir = _build_dataset_with_embeddings(
            tmp_path, n_species=2, n_images=5,
            mislabel_spec={"species": 0, "image": 0},
        )

        rejected = tmp_path / "rejected"
        stats = apply_filter(
            str(split_dir), str(reports_dir),
            reject_categories=[],
            remove_mislabeled=True,
            mislabel_margin=0.1,
            rejected_dir=str(rejected),
        )

        sp_dir = split_dir / "species_0"
        remaining = [f.name for f in sp_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        assert "photo_0.jpg" not in remaining
        assert stats["removed"] >= 1


# ── UC-Q49 : margin=0 flagge plus d'images que margin=0.5 ───────────────

class TestUCQ49_MarginZeroMoreSuspected:
    def test_lower_margin_flags_more(self, tmp_path):
        from quality_filter import detect_mislabeled

        split_dir, reports_dir = _build_dataset_with_embeddings(
            tmp_path, n_species=2, n_images=5,
            mislabel_spec={"species": 0, "image": 0},
        )

        result_strict = detect_mislabeled(str(split_dir), str(reports_dir), margin=0.5)
        result_loose = detect_mislabeled(str(split_dir), str(reports_dir), margin=0.0)

        def count_suspected(r):
            return sum(1 for sp in r.values() for info in sp.values() if info["suspected"])

        assert count_suspected(result_loose) >= count_suspected(result_strict)


# ═════════════════════════════════════════════════════════════════════════
# INTÉGRATION — filtres combinés et régression
# ═════════════════════════════════════════════════════════════════════════

# ── UC-Q50 : les deux filtres ensemble ───────────────────────────────────

class TestUCQ50_CombinedFilters:
    def test_both_filters_together(self, tmp_path):
        from quality_filter import apply_filter

        split_dir, reports_dir = _build_dataset_with_embeddings(
            tmp_path, n_species=2, n_images=5,
            mislabel_spec={"species": 0, "image": 0},
        )

        # Ajouter des duplicates dans le rapport de species_1
        report_file = reports_dir / "species_1.json"
        with open(report_file) as f:
            report = json.load(f)
        report["duplicates"] = [
            {"kept": "photo_0.jpg", "removed": "photo_1.jpg", "similarity": 0.98}
        ]
        with open(report_file, "w") as f:
            json.dump(report, f)

        rejected = tmp_path / "rejected"
        stats = apply_filter(
            str(split_dir), str(reports_dir),
            reject_categories=[],
            remove_duplicates=True,
            remove_mislabeled=True,
            mislabel_margin=0.1,
            rejected_dir=str(rejected),
        )

        sp0 = split_dir / "species_0"
        remaining_0 = [f.name for f in sp0.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        assert "photo_0.jpg" not in remaining_0

        sp1 = split_dir / "species_1"
        remaining_1 = [f.name for f in sp1.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        assert "photo_1.jpg" not in remaining_1

        assert stats["removed"] >= 2


# ── UC-Q51 : format de rapport existant préservé ────────────────────────

class TestUCQ51_ExistingReportFormatPreserved:
    def test_existing_keys_unchanged(self, tmp_path):
        from quality_filter import QualityClassifier

        clf = QualityClassifier()
        train_dir = _build_mini_quality_dataset(tmp_path / "data", n_species=1, n_images=3)
        out = tmp_path / "reports"

        clf.generate_quality_report(str(train_dir), str(out))

        for json_file in out.glob("*.json"):
            with open(json_file) as f:
                report = json.load(f)
            assert "summary" in report
            assert "images" in report
            assert "outliers" in report
            assert "total" in report["summary"]
            assert "good" in report["summary"]
            assert "duplicates" not in report


# ── UC-Q52 : photos réelles différentes ne sont pas des duplicates ──────

class TestUCQ52_RealBirdNoDuplicates:
    def test_real_different_birds_not_duplicates(self, tmp_path):
        import shutil
        from quality_filter import QualityClassifier

        clf = QualityClassifier()
        sp_dir = tmp_path / "species_mixed"
        sp_dir.mkdir()

        birds = _real_bird_fixtures(5)
        for bird in birds:
            shutil.copy(str(bird), sp_dir / bird.name)

        classifications, embeddings = clf.classify_and_embed_batched(str(sp_dir))
        duplicates = clf.detect_near_duplicates(embeddings, classifications, threshold=0.95)
        assert duplicates == []


# ── Helpers scoring qualité ───────────────────────────────────────────────


def _make_sharp_bird_image(path: Path, size=(200, 200), bbox_pct=25):
    import cv2

    h, w = size
    img = np.full((h, w, 3), (34, 139, 34), dtype=np.uint8)
    bh = int(h * (bbox_pct / 100) ** 0.5)
    bw = int(w * (bbox_pct / 100) ** 0.5)
    cx, cy = w // 2, h // 2
    x1, y1 = cx - bw // 2, cy - bh // 2
    x2, y2 = x1 + bw, y1 + bh
    cv2.rectangle(img, (x1, y1), (x2, y2), (139, 90, 43), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)
    for i in range(x1 + 5, x2, 10):
        cv2.line(img, (i, y1), (i, y2), (0, 0, 0), 1)
    cv2.imwrite(str(path), img)


def _make_blurry_image(path: Path, size=(200, 200)):
    import cv2

    h, w = size
    img = np.random.randint(100, 160, (h, w, 3), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (31, 31), 15)
    cv2.imwrite(str(path), img)


# ── UC-Q53 : score_image_quality retourne les 4 clés ─────────────────────


class TestUCQ53_ScoreImageReturnsDict:
    def test_returns_expected_keys(self, tmp_path):
        from quality_filter import score_image_quality

        img_path = tmp_path / "bird.jpg"
        _make_sharp_bird_image(img_path)
        ann = {"bbox": [30, 30, 70, 70], "score": 0.85}
        result = score_image_quality(img_path, ann)
        assert set(result.keys()) == {"detection_score", "bbox_pct", "sharpness", "composite_score"}
        assert result["composite_score"] > 0


# ── UC-Q54 : annotation None → composite_score = 0 ──────────────────────


class TestUCQ54_ScoreImageNoneAnnotation:
    def test_none_annotation_gives_zero_composite(self, tmp_path):
        from quality_filter import score_image_quality

        img_path = tmp_path / "bird.jpg"
        _make_sharp_bird_image(img_path)
        result = score_image_quality(img_path, None)
        assert result["composite_score"] == 0.0


# ── UC-Q55 : fichier inexistant → composite_score = 0 ───────────────────


class TestUCQ55_ScoreImageMissingFile:
    def test_missing_image_returns_zeros(self, tmp_path):
        from quality_filter import score_image_quality

        result = score_image_quality(tmp_path / "nope.jpg", {"bbox": [0, 0, 10, 10], "score": 0.5})
        assert result["composite_score"] == 0.0


# ── UC-Q56 : image nette + grande bbox > image floue + petite bbox ───────


class TestUCQ56_CompositeScoreOrdering:
    def test_sharp_large_bbox_scores_higher(self, tmp_path):
        from quality_filter import score_image_quality

        sharp = tmp_path / "sharp.jpg"
        blurry = tmp_path / "blurry.jpg"
        _make_sharp_bird_image(sharp, bbox_pct=25)
        _make_blurry_image(blurry)
        ann_sharp = {"bbox": [30, 30, 70, 70], "score": 0.9}
        ann_blurry = {"bbox": [90, 90, 10, 10], "score": 0.4}
        s1 = score_image_quality(sharp, ann_sharp)
        s2 = score_image_quality(blurry, ann_blurry)
        assert s1["composite_score"] > s2["composite_score"]


# ── UC-Q57 : pas de détection → score ≈ 0 ────────────────────────────────


class TestUCQ57_NoDetectionLowScore:
    def test_no_detection_near_zero(self, tmp_path):
        from quality_filter import score_image_quality

        img = tmp_path / "img.jpg"
        _make_sharp_bird_image(img)
        ann = {"bbox": [0, 0, 0, 0], "score": 0.0}
        result = score_image_quality(img, ann)
        assert result["composite_score"] < 0.01


# ── UC-Q58 : score_species_quality retourne un dict pour toutes les images


class TestUCQ58_ScoreSpeciesQuality:
    def test_returns_dict_for_all_images(self, tmp_path):
        from quality_filter import score_species_quality

        sp = tmp_path / "parus_major"
        sp.mkdir()
        for i in range(3):
            _make_sharp_bird_image(sp / f"img_{i}.jpg")
        ann = {f"img_{i}.jpg": {"bbox": [30, 30, 70, 70], "score": 0.85} for i in range(3)}
        with open(sp / "annotations.json", "w") as f:
            json.dump(ann, f)
        result = score_species_quality(sp)
        assert len(result) == 3
        for v in result.values():
            assert "composite_score" in v


# ── UC-Q59 : score_species_quality ignore les non-images ─────────────────


class TestUCQ59_ScoreSpeciesIgnoresNonImages:
    def test_skips_json_files(self, tmp_path):
        from quality_filter import score_species_quality

        sp = tmp_path / "parus_major"
        sp.mkdir()
        _make_sharp_bird_image(sp / "img.jpg")
        with open(sp / "annotations.json", "w") as f:
            json.dump({"img.jpg": {"bbox": [30, 30, 70, 70], "score": 0.85}}, f)
        result = score_species_quality(sp)
        assert "annotations.json" not in result
        assert "img.jpg" in result


# ── UC-Q77 : generate_quality_report inclut quality score par image ─────


class TestUCQ77_ReportIncludesQualityScore:
    def test_quality_key_in_report(self, tmp_path):
        """generate_quality_report inclut quality.composite_score pour chaque image."""
        from quality_filter import QualityClassifier

        clf = QualityClassifier()
        train_dir = tmp_path / "train"
        sp_dir = train_dir / "parus_major"
        sp_dir.mkdir(parents=True)

        for i in range(3):
            _make_sharp_bird_image(sp_dir / f"img_{i}.jpg")

        annotations = {
            f"img_{i}.jpg": {"bbox": [30, 30, 70, 70], "score": 0.85}
            for i in range(3)
        }
        with open(sp_dir / "annotations.json", "w") as f:
            json.dump(annotations, f)

        reports_dir = tmp_path / "reports"
        clf.generate_quality_report(str(train_dir), str(reports_dir), workers=1)

        with open(reports_dir / "parus_major.json") as f:
            data = json.load(f)

        expected_keys = {"detection_score", "bbox_pct", "sharpness", "composite_score"}
        for img_name, img_report in data["images"].items():
            assert "quality" in img_report, f"missing 'quality' key for {img_name}"
            assert set(img_report["quality"].keys()) == expected_keys
            assert img_report["quality"]["composite_score"] > 0


# ── UC-Q78 : detect_near_duplicates garde le meilleur composite_score ───

class TestUCQ78_KeepsBestCompositeScore:
    def test_keeps_best_composite_not_best_good(self):
        import torch
        from quality_filter import QualityClassifier

        clf = QualityClassifier()

        vec = torch.randn(768)
        vec = vec / vec.norm()
        embeddings = {
            "img_x.jpg": vec.clone(),
            "img_y.jpg": vec.clone(),
        }
        classifications = {
            "img_x.jpg": {"category": "good", "confidence": 0.9, "scores": {"good": 0.9}, "quality": {"composite_score": 0.4}},
            "img_y.jpg": {"category": "good", "confidence": 0.9, "scores": {"good": 0.9}, "quality": {"composite_score": 0.85}},
        }

        duplicates = clf.detect_near_duplicates(embeddings, classifications, threshold=0.95)
        assert len(duplicates) == 1
        assert duplicates[0]["kept"] == "img_y.jpg"
        assert duplicates[0]["removed"] == "img_x.jpg"


# ── Helpers pour UC-Q74→UC-Q76 ──────────────────────────────────────────

def _build_all_good_dataset(tmp_path: Path, scores: dict[str, float]) -> tuple[Path, Path]:
    """Crée un dataset où toutes les images sont 'good' avec des composite_score distincts."""
    train_dir = tmp_path / "train"
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True)

    sp_dir = train_dir / "parus_major"
    sp_dir.mkdir(parents=True)
    annotations = {}
    for name in scores:
        PILImage.new("RGB", (100, 100)).save(sp_dir / name)
        annotations[name] = {"bbox": [10, 10, 50, 50], "score": 0.9}
    with open(sp_dir / "annotations.json", "w") as f:
        json.dump(annotations, f)

    images = {}
    for name, cs in scores.items():
        images[name] = {
            "category": "good",
            "confidence": 0.9,
            "scores": {"good": 0.9},
            "quality": {"composite_score": cs},
        }
    report = {
        "summary": {"total": len(scores), "good": len(scores),
                     "dead_specimen": 0, "illustration": 0,
                     "screen_scan": 0, "not_bird": 0, "poor_quality": 0},
        "images": images,
        "outliers": {n: {"distance": 0.01, "is_outlier": False} for n in scores},
    }
    with open(reports_dir / "parus_major.json", "w") as f:
        json.dump(report, f)

    return train_dir, reports_dir


# ── UC-Q74 : apply_filter avec max_per_species=3 sur 5 images ───────────

class TestUCQ74_MaxPerSpeciesCaps:
    def test_caps_to_max_per_species(self, tmp_path):
        from quality_filter import apply_filter

        scores = {
            "photo_0.jpg": 0.9,
            "photo_1.jpg": 0.7,
            "photo_2.jpg": 0.5,
            "photo_3.jpg": 0.3,
            "photo_4.jpg": 0.1,
        }
        train_dir, reports_dir = _build_all_good_dataset(tmp_path, scores)
        rejected = tmp_path / "rejected"

        stats = apply_filter(
            str(train_dir), str(reports_dir),
            reject_categories=[],
            max_per_species=3,
            rejected_dir=str(rejected),
        )

        sp_dir = train_dir / "parus_major"
        remaining = [f.name for f in sp_dir.iterdir() if f.suffix.lower() in (".jpg",)]
        assert len(remaining) == 3
        assert stats["removed"] == 2
        assert stats["kept"] == 3

        rej_sp = rejected / "parus_major"
        rejected_files = [f.name for f in rej_sp.iterdir() if f.suffix.lower() in (".jpg",)]
        assert len(rejected_files) == 2


# ── UC-Q75 : les images gardées sont celles avec le meilleur composite_score ─

class TestUCQ75_KeepsBestCompositeScores:
    def test_keeps_top_scores(self, tmp_path):
        from quality_filter import apply_filter

        scores = {
            "photo_0.jpg": 0.9,
            "photo_1.jpg": 0.7,
            "photo_2.jpg": 0.5,
            "photo_3.jpg": 0.3,
            "photo_4.jpg": 0.1,
        }
        train_dir, reports_dir = _build_all_good_dataset(tmp_path, scores)
        rejected = tmp_path / "rejected"

        apply_filter(
            str(train_dir), str(reports_dir),
            reject_categories=[],
            max_per_species=3,
            rejected_dir=str(rejected),
        )

        sp_dir = train_dir / "parus_major"
        remaining = {f.name for f in sp_dir.iterdir() if f.suffix.lower() in (".jpg",)}
        assert remaining == {"photo_0.jpg", "photo_1.jpg", "photo_2.jpg"}

        rej_sp = rejected / "parus_major"
        rejected_names = {f.name for f in rej_sp.iterdir() if f.suffix.lower() in (".jpg",)}
        assert rejected_names == {"photo_3.jpg", "photo_4.jpg"}


# ── UC-Q76 : max_per_species=0 → pas de capping ────────────────────────

class TestUCQ76_NoCappingByDefault:
    def test_no_capping_when_zero(self, tmp_path):
        from quality_filter import apply_filter

        scores = {
            "photo_0.jpg": 0.9,
            "photo_1.jpg": 0.7,
            "photo_2.jpg": 0.5,
            "photo_3.jpg": 0.3,
            "photo_4.jpg": 0.1,
        }
        train_dir, reports_dir = _build_all_good_dataset(tmp_path, scores)

        stats = apply_filter(
            str(train_dir), str(reports_dir),
            reject_categories=[],
            max_per_species=0,
        )

        sp_dir = train_dir / "parus_major"
        remaining = [f.name for f in sp_dir.iterdir() if f.suffix.lower() in (".jpg",)]
        assert len(remaining) == 5
        assert stats["removed"] == 0
        assert stats["kept"] == 5


# ── Helper pour UC-Q60→UC-Q73 ──────────────────────────────────────────


def _build_calibration_dataset_qf(tmp_path: Path, n_species=3, n_kept=5, n_rejected=3):
    """Crée un dataset europe + europe_rejected pour les tests de calibration."""
    europe = tmp_path / "europe"
    rejected = tmp_path / "europe_rejected"

    for i in range(n_species):
        slug = f"species_{i}"

        sp_kept = europe / "train" / slug
        sp_kept.mkdir(parents=True, exist_ok=True)
        ann_kept = {}
        for j in range(n_kept):
            name = f"kept_{j:03d}.jpg"
            _make_sharp_bird_image(sp_kept / name, bbox_pct=15)
            ann_kept[name] = {"bbox": [30, 30, 70, 70], "score": 0.85}
        with open(sp_kept / "annotations.json", "w") as f:
            json.dump(ann_kept, f)

        sp_rej = rejected / "train" / slug
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
# STRATIFIED SAMPLER — échantillonnage stratifié pour calibration
# ═════════════════════════════════════════════════════════════════════════

# ── UC-Q60 : sample retourne N images par espèce de chaque dataset ──────


class TestUCQ60_SampleReturnsNPerSpecies:
    def test_returns_n_per_species(self, tmp_path):
        from quality_filter import StratifiedSampler

        europe, rejected = _build_calibration_dataset_qf(tmp_path, n_species=2, n_kept=5, n_rejected=3)
        sampler = StratifiedSampler(europe / "train", rejected / "train")

        samples = sampler.sample(n_per_species=2, mode="random", seed=42)
        species = {s["species"] for s in samples}
        assert len(species) == 2
        for sp in species:
            sp_samples = [s for s in samples if s["species"] == sp]
            assert len(sp_samples) <= 4


# ── UC-Q61 : mode borderline surreprésente les images proches du seuil ──


class TestUCQ61_BorderlineMode:
    def test_borderline_oversamples_near_threshold(self, tmp_path):
        from quality_filter import StratifiedSampler

        europe, rejected = _build_calibration_dataset_qf(tmp_path, n_species=2, n_kept=10, n_rejected=5)
        sampler = StratifiedSampler(europe / "train", rejected / "train")

        samples = sampler.sample(n_per_species=3, mode="borderline", seed=42)
        assert len(samples) > 0
        for s in samples:
            assert "composite_score" in s


# ── UC-Q62 : sample est reproductible avec un seed ──────────────────────


class TestUCQ62_SampleReproducible:
    def test_same_seed_same_samples(self, tmp_path):
        from quality_filter import StratifiedSampler

        europe, rejected = _build_calibration_dataset_qf(tmp_path, n_species=2, n_kept=5, n_rejected=3)
        sampler = StratifiedSampler(europe / "train", rejected / "train")

        s1 = sampler.sample(n_per_species=2, mode="random", seed=42)
        s2 = sampler.sample(n_per_species=2, mode="random", seed=42)
        paths1 = [s["path"] for s in s1]
        paths2 = [s["path"] for s in s2]
        assert paths1 == paths2


# ── UC-Q63 : sample gère les espèces avec moins d'images que N ──────────


class TestUCQ63_SampleFewImages:
    def test_handles_species_with_few_images(self, tmp_path):
        from quality_filter import StratifiedSampler

        europe, rejected = _build_calibration_dataset_qf(tmp_path, n_species=1, n_kept=2, n_rejected=1)
        sampler = StratifiedSampler(europe / "train", rejected / "train")

        samples = sampler.sample(n_per_species=10, mode="random", seed=42)
        assert len(samples) <= 3


# ═════════════════════════════════════════════════════════════════════════
# GROUND TRUTH — stockage des labels humains
# ═════════════════════════════════════════════════════════════════════════

# ── UC-Q64 : load retourne un dict vide si pas de fichier ───────────────


class TestUCQ64_GroundTruthLoadEmpty:
    def test_empty_if_no_file(self, tmp_path):
        from quality_filter import GroundTruth

        gt = GroundTruth(tmp_path / "ground_truth.json")
        labels = gt.load()
        assert labels == {}


# ── UC-Q65 : save persiste les labels et load les retrouve ──────────────


class TestUCQ65_GroundTruthSaveLoad:
    def test_save_and_load(self, tmp_path):
        from quality_filter import GroundTruth

        gt = GroundTruth(tmp_path / "ground_truth.json")
        labels = {"img1.jpg": {"label": "good", "source": "europe", "species": "parus_major"}}
        gt.save(labels)

        loaded = gt.load()
        assert loaded == labels


# ── UC-Q66 : add_label ajoute un label sans écraser les autres ──────────


class TestUCQ66_GroundTruthAddLabel:
    def test_add_label(self, tmp_path):
        from quality_filter import GroundTruth

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

# ── UC-Q67 : compute_metrics retourne precision, recall, f1, fpr, fnr ───


class TestUCQ67_ComputeMetricsKeys:
    def test_returns_expected_keys(self):
        from quality_filter import compute_metrics

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


# ── UC-Q68 : métriques parfaites si toutes les décisions sont correctes ─


class TestUCQ68_PerfectMetrics:
    def test_perfect_predictions(self):
        from quality_filter import compute_metrics

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


# ── UC-Q69 : precision 0 si toutes les images gardées sont mauvaises ────


class TestUCQ69_ZeroPrecision:
    def test_all_kept_are_bad(self):
        from quality_filter import compute_metrics

        labels = {
            "kept_1.jpg": {"label": "bad", "source": "europe"},
            "kept_2.jpg": {"label": "bad", "source": "europe"},
            "rej_1.jpg": {"label": "bad", "source": "europe_rejected"},
        }
        metrics = compute_metrics(labels)
        assert metrics["precision"] == 0.0


# ── UC-Q70 : compute_metrics_by_reason ventile par raison de rejet ──────


class TestUCQ70_MetricsByReason:
    def test_breakdown_by_reason(self):
        from quality_filter import compute_metrics_by_reason

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
# ═════════════════════════════════════════════════════════════════════════

# ── UC-Q71 : sweep retourne une liste triée par f1 ──────────────────────


class TestUCQ71_ThresholdSweep:
    def test_sweep_returns_sorted_results(self, tmp_path):
        from quality_filter import ThresholdOptimizer, GroundTruth

        europe, rejected = _build_calibration_dataset_qf(tmp_path, n_species=1, n_kept=5, n_rejected=3)
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

        optimizer = ThresholdOptimizer(europe / "train", rejected / "train")
        results = optimizer.sweep(gt)
        assert len(results) > 0
        f1_scores = [r["metrics"]["f1"] for r in results]
        assert f1_scores == sorted(f1_scores, reverse=True)


# ── UC-Q72 : sweep avec un seul ground truth fonctionne ─────────────────


class TestUCQ72_SweepSingleSample:
    def test_single_sample_works(self, tmp_path):
        from quality_filter import ThresholdOptimizer, GroundTruth

        europe, rejected = _build_calibration_dataset_qf(tmp_path, n_species=1, n_kept=1, n_rejected=0)
        gt = GroundTruth(tmp_path / "gt.json")
        for img in (europe / "train" / "species_0").iterdir():
            if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                gt.add_label(str(img), "good", "europe", "species_0")
                break

        optimizer = ThresholdOptimizer(europe / "train", rejected / "train")
        results = optimizer.sweep(gt)
        assert len(results) > 0


# ── UC-Q73 : recommend retourne la combinaison avec le meilleur f1 ──────


class TestUCQ73_ThresholdRecommend:
    def test_recommend_returns_best_f1(self, tmp_path):
        from quality_filter import ThresholdOptimizer, GroundTruth

        europe, rejected = _build_calibration_dataset_qf(tmp_path, n_species=1, n_kept=5, n_rejected=3)
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

        optimizer = ThresholdOptimizer(europe / "train", rejected / "train")
        best = optimizer.recommend(gt)
        assert "thresholds" in best
        assert "metrics" in best
        assert best["metrics"]["f1"] >= 0.0
