"""
Tests pour verify_boxes.py — outil de vérification et récupération des annotations.
"""

import json
from pathlib import Path

import pytest


def _create_fake_species(base_dir: Path, species: str, n_images: int, n_detected: int):
    """Crée un dossier espèce avec de vraies images PIL et annotations.json simulées."""
    from PIL import Image as PILImage
    sp_dir = base_dir / species
    sp_dir.mkdir(parents=True, exist_ok=True)
    annotations = {}
    for i in range(n_images):
        name = f"photo_{i:04d}.jpg"
        img = PILImage.new("RGB", (100, 100), color=(i * 20 % 256, 100, 50))
        img.save(sp_dir / name)
        if i < n_detected:
            annotations[name] = {"bbox": [10, 10, 50, 50], "score": 0.85}
        else:
            annotations[name] = None
    with open(sp_dir / "annotations.json", "w") as f:
        json.dump(annotations, f)
    return sp_dir


def _build_annotated_dataset(tmp_path: Path) -> Path:
    """Crée un dataset annoté avec des espèces ayant différents taux de détection."""
    train_dir = tmp_path / "europe" / "train"
    _create_fake_species(train_dir, "parus_major", n_images=10, n_detected=9)
    _create_fake_species(train_dir, "scolopax_rusticola", n_images=10, n_detected=5)
    _create_fake_species(train_dir, "rallus_aquaticus", n_images=10, n_detected=10)
    return tmp_path / "europe"


# ---------------------------------------------------------------------------
# UC-V01 : --no-detections liste les images sans détection avec le bon format
#          (espece/filename.jpg, une par ligne)
# ---------------------------------------------------------------------------
class TestUCV01_NoDetectionsFormat:
    def test_lists_undetected_images(self, tmp_path):
        from verify_boxes import collect_no_detections
        dataset = _build_annotated_dataset(tmp_path)
        no_det = collect_no_detections(dataset / "train")
        all_missing = [f"{sp}/{name}" for sp, names in no_det.items() for name in names]
        assert len(all_missing) == 6
        assert all("/" in entry for entry in all_missing)


# ---------------------------------------------------------------------------
# UC-V02 : --no-detections ne liste que les images à None, pas celles avec bbox
# ---------------------------------------------------------------------------
class TestUCV02_OnlyNullEntries:
    def test_excludes_detected_images(self, tmp_path):
        from verify_boxes import collect_no_detections
        dataset = _build_annotated_dataset(tmp_path)
        no_det = collect_no_detections(dataset / "train")
        assert "rallus_aquaticus" not in no_det
        assert "parus_major" in no_det
        assert len(no_det["parus_major"]) == 1
        assert "scolopax_rusticola" in no_det
        assert len(no_det["scolopax_rusticola"]) == 5


# ---------------------------------------------------------------------------
# UC-V03 : --no-detections ignore les espèces sans annotations.json
# ---------------------------------------------------------------------------
class TestUCV03_SkipsUnannotated:
    def test_skips_species_without_annotations(self, tmp_path):
        from verify_boxes import collect_no_detections
        dataset = _build_annotated_dataset(tmp_path)
        no_ann_dir = dataset / "train" / "species_no_ann"
        no_ann_dir.mkdir()
        (no_ann_dir / "photo.jpg").write_bytes(b"\xff\xd8" + b"\x00" * 50)
        no_det = collect_no_detections(dataset / "train")
        assert "species_no_ann" not in no_det


# ---------------------------------------------------------------------------
# UC-V04 : retry_with_lower_threshold ré-annote les images None avec un seuil
#          plus bas et met à jour annotations.json
# ---------------------------------------------------------------------------
class TestUCV04_RetryLowerThreshold:
    def test_retry_updates_annotations(self, tmp_path):
        from verify_boxes import retry_no_detections
        from auto_annotate import BirdAnnotator
        import shutil

        dataset = tmp_path / "europe" / "train"
        sp_dir = dataset / "parus_major"
        sp_dir.mkdir(parents=True)

        fixtures = Path(__file__).parent / "fixtures" / "images"
        real_images = sorted(
            f for f in fixtures.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
            and not f.name.startswith(("no_bird", "grayscale", "rgba", "corrupted"))
        )[:3]
        annotations = {}
        for img in real_images:
            shutil.copy2(img, sp_dir / img.name)
            annotations[img.name] = None

        with open(sp_dir / "annotations.json", "w") as f:
            json.dump(annotations, f)

        stats = retry_no_detections(str(tmp_path / "europe" / "train"), threshold=0.3)
        assert stats["retried"] > 0

        with open(sp_dir / "annotations.json") as f:
            updated = json.load(f)
        recovered = sum(1 for v in updated.values() if v is not None)
        assert recovered >= 1


# ---------------------------------------------------------------------------
# UC-V05 : retry_with_lower_threshold ne touche pas aux images déjà détectées
# ---------------------------------------------------------------------------
class TestUCV05_RetryPreservesExisting:
    def test_retry_keeps_existing_detections(self, tmp_path):
        from verify_boxes import retry_no_detections

        dataset = tmp_path / "europe" / "train"
        sp_dir = dataset / "parus_major"
        sp_dir.mkdir(parents=True)
        (sp_dir / "detected.jpg").write_bytes(b"\xff\xd8" + b"\x00" * 100)
        (sp_dir / "missing.jpg").write_bytes(b"\xff\xd8" + b"\x00" * 100)

        original_det = {"bbox": [10, 20, 100, 80], "score": 0.92}
        annotations = {"detected.jpg": original_det, "missing.jpg": None}
        with open(sp_dir / "annotations.json", "w") as f:
            json.dump(annotations, f)

        retry_no_detections(str(dataset), threshold=0.3)

        with open(sp_dir / "annotations.json") as f:
            updated = json.load(f)
        assert updated["detected.jpg"] == original_det


# ---------------------------------------------------------------------------
# UC-V06 : retry retourne des stats (retried, recovered, still_missing)
# ---------------------------------------------------------------------------
class TestUCV06_RetryStats:
    def test_retry_returns_stats(self, tmp_path):
        from verify_boxes import retry_no_detections
        dataset = _build_annotated_dataset(tmp_path)
        stats = retry_no_detections(str(dataset / "train"), threshold=0.3)
        assert "retried" in stats
        assert "recovered" in stats
        assert "still_missing" in stats
        assert stats["retried"] == 6
        assert stats["recovered"] + stats["still_missing"] == stats["retried"]


# ---------------------------------------------------------------------------
# UC-V07 : retry marque les détections récupérées avec retry_threshold
#          pour traçabilité du changement de seuil
# ---------------------------------------------------------------------------
class TestUCV07_RetryMarksThreshold:
    def test_retry_adds_threshold_field(self, tmp_path):
        from verify_boxes import retry_no_detections
        import shutil

        dataset = tmp_path / "europe" / "train"
        sp_dir = dataset / "parus_major"
        sp_dir.mkdir(parents=True)

        fixtures = Path(__file__).parent / "fixtures" / "images"
        real_images = sorted(
            f for f in fixtures.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
            and not f.name.startswith(("no_bird", "grayscale", "rgba", "corrupted"))
        )[:2]
        annotations = {}
        for img in real_images:
            shutil.copy2(img, sp_dir / img.name)
            annotations[img.name] = None
        with open(sp_dir / "annotations.json", "w") as f:
            json.dump(annotations, f)

        retry_no_detections(str(dataset), threshold=0.3)

        with open(sp_dir / "annotations.json") as f:
            updated = json.load(f)
        for name, det in updated.items():
            if det is not None:
                assert det["retry_threshold"] == 0.3, (
                    f"{name}: retry_threshold manquant ou incorrect"
                )


# ---------------------------------------------------------------------------
# UC-V08 : generate_review crée un dossier detected/ et no_detection/
#          par espèce annotée
# ---------------------------------------------------------------------------
class TestUCV08_ReviewCreatesStructure:
    def test_creates_detected_and_missing_folders(self, tmp_path):
        from verify_boxes import generate_review
        dataset = _build_annotated_dataset(tmp_path)
        out = tmp_path / "review"
        generate_review(dataset / "train", out, n_detected=3, n_missing=3)

        assert (out / "scolopax_rusticola" / "detected").is_dir()
        assert (out / "scolopax_rusticola" / "no_detection").is_dir()
        assert (out / "parus_major" / "detected").is_dir()
        assert (out / "parus_major" / "no_detection").is_dir()


# ---------------------------------------------------------------------------
# UC-V09 : generate_review met le bon nombre d'images dans chaque dossier
# ---------------------------------------------------------------------------
class TestUCV09_ReviewCorrectCounts:
    def test_correct_sample_counts(self, tmp_path):
        from verify_boxes import generate_review
        dataset = _build_annotated_dataset(tmp_path)
        out = tmp_path / "review"
        generate_review(dataset / "train", out, n_detected=3, n_missing=3)

        det_files = list((out / "scolopax_rusticola" / "detected").iterdir())
        miss_files = list((out / "scolopax_rusticola" / "no_detection").iterdir())
        assert len(det_files) == 3
        assert len(miss_files) == 3

        rallus_det = list((out / "rallus_aquaticus" / "detected").iterdir())
        assert len(rallus_det) == 3
        assert not (out / "rallus_aquaticus" / "no_detection").exists()


# ---------------------------------------------------------------------------
# UC-V10 : generate_review écrase le dossier samples/ existant
# ---------------------------------------------------------------------------
class TestUCV10_ReviewOverwritesExisting:
    def test_overwrites_existing_output(self, tmp_path):
        from verify_boxes import generate_review
        dataset = _build_annotated_dataset(tmp_path)
        out = tmp_path / "review"

        out.mkdir(parents=True)
        (out / "old_file.txt").write_text("stale")

        generate_review(dataset / "train", out, n_detected=2, n_missing=2)
        assert not (out / "old_file.txt").exists()
        assert (out / "parus_major").is_dir()


# ---------------------------------------------------------------------------
# UC-V11 : retry avec workers>1 produit les mêmes résultats que workers=1
# ---------------------------------------------------------------------------
class TestUCV11_RetryParallelSameResults:
    def test_parallel_retry_matches_sequential(self, tmp_path):
        from verify_boxes import retry_no_detections
        import shutil

        fixtures = Path(__file__).parent / "fixtures" / "images"
        real_images = sorted(
            f for f in fixtures.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
            and not f.name.startswith(("no_bird", "grayscale", "rgba", "corrupted"))
        )[:4]

        for label, base in [("seq", tmp_path / "seq"), ("par", tmp_path / "par")]:
            train = base / "train"
            for sp_name in ["sp_a", "sp_b"]:
                sp_dir = train / sp_name
                sp_dir.mkdir(parents=True)
                annotations = {}
                for img in real_images[:2]:
                    shutil.copy2(img, sp_dir / img.name)
                    annotations[img.name] = None
                with open(sp_dir / "annotations.json", "w") as f:
                    json.dump(annotations, f)

        stats_seq = retry_no_detections(str(tmp_path / "seq" / "train"), threshold=0.3, workers=1)
        stats_par = retry_no_detections(str(tmp_path / "par" / "train"), threshold=0.3, workers=2)

        assert stats_seq["recovered"] == stats_par["recovered"]

        for sp_name in ["sp_a", "sp_b"]:
            with open(tmp_path / "seq" / "train" / sp_name / "annotations.json") as f:
                seq_data = json.load(f)
            with open(tmp_path / "par" / "train" / sp_name / "annotations.json") as f:
                par_data = json.load(f)
            for name in seq_data:
                seq_det = seq_data[name]
                par_det = par_data[name]
                if seq_det is None:
                    assert par_det is None
                else:
                    assert par_det is not None
                    assert seq_det["bbox"] == par_det["bbox"]


# ---------------------------------------------------------------------------
# UC-V12 : retry avec workers>1 préserve les détections existantes
# ---------------------------------------------------------------------------
class TestUCV12_RetryParallelPreservesExisting:
    def test_parallel_preserves_existing(self, tmp_path):
        from verify_boxes import retry_no_detections

        dataset = tmp_path / "europe" / "train"
        sp_dir = dataset / "parus_major"
        sp_dir.mkdir(parents=True)
        (sp_dir / "detected.jpg").write_bytes(b"\xff\xd8" + b"\x00" * 100)
        (sp_dir / "missing.jpg").write_bytes(b"\xff\xd8" + b"\x00" * 100)

        original_det = {"bbox": [10, 20, 100, 80], "score": 0.92}
        annotations = {"detected.jpg": original_det, "missing.jpg": None}
        with open(sp_dir / "annotations.json", "w") as f:
            json.dump(annotations, f)

        retry_no_detections(str(dataset), threshold=0.3, workers=2)

        with open(sp_dir / "annotations.json") as f:
            updated = json.load(f)
        assert updated["detected.jpg"] == original_det


# ---------------------------------------------------------------------------
# UC-V13 : retry avec workers>1 retourne des stats cohérentes
# ---------------------------------------------------------------------------
class TestUCV13_RetryParallelStats:
    def test_parallel_stats_consistent(self, tmp_path):
        from verify_boxes import retry_no_detections
        dataset = _build_annotated_dataset(tmp_path)
        stats = retry_no_detections(str(dataset / "train"), threshold=0.3, workers=2)
        assert stats["retried"] == 6
        assert stats["recovered"] + stats["still_missing"] == stats["retried"]
