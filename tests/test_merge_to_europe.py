"""
Tests TDD pour merge_to_europe.

Fusionne les images filtrées de europe_to_trait/{slug}/ vers
europe/{train,val,test}/{slug}/ avec ratio 80/10/10,
fusion des annotations.json, et nettoyage de la source.
"""

import json
from pathlib import Path

import pytest

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _create_fake_images(directory: Path, count: int, prefix: str = "photo") -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    names = []
    for i in range(count):
        name = f"{prefix}_{i:04d}.jpg"
        (directory / name).write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        names.append(name)
    return names


def _create_annotations(directory: Path, image_names: list[str]):
    annotations = {}
    for name in image_names:
        annotations[name] = {"bbox": [10, 10, 40, 40], "score": 0.95}
    with open(directory / "annotations.json", "w") as f:
        json.dump(annotations, f)


def _count_images(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for f in directory.iterdir()
               if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)


def _list_image_names(directory: Path) -> set[str]:
    if not directory.exists():
        return set()
    return {f.name for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS}


@pytest.fixture
def merge_setup(tmp_path):
    """Crée un dataset europe/ existant et un europe_to_trait/ avec des images à fusionner."""
    europe = tmp_path / "europe"
    source = tmp_path / "europe_to_trait"

    existing_train_names = _create_fake_images(europe / "train" / "parus_major", 40, "existing")
    _create_annotations(europe / "train" / "parus_major", existing_train_names)
    existing_val_names = _create_fake_images(europe / "validation" / "parus_major", 5, "existing_v")
    _create_annotations(europe / "validation" / "parus_major", existing_val_names)
    existing_test_names = _create_fake_images(europe / "test" / "parus_major", 5, "existing_t")
    _create_annotations(europe / "test" / "parus_major", existing_test_names)

    new_names = _create_fake_images(source / "parus_major", 20, "new")
    _create_annotations(source / "parus_major", new_names)

    return europe, source


@pytest.fixture
def merge_multi_species(tmp_path):
    """Plusieurs espèces dans europe_to_trait/."""
    europe = tmp_path / "europe"
    source = tmp_path / "europe_to_trait"

    for slug in ("parus_major", "turdus_merula", "erithacus_rubecula"):
        for split in ("train", "validation", "test"):
            names = _create_fake_images(europe / split / slug, 10, f"ex_{split}")
            _create_annotations(europe / split / slug, names)

        new_names = _create_fake_images(source / slug, 30, "new")
        _create_annotations(source / slug, new_names)

    return europe, source


@pytest.fixture
def merge_no_annotations(tmp_path):
    """Images dans europe_to_trait/ sans annotations.json."""
    europe = tmp_path / "europe"
    source = tmp_path / "europe_to_trait"

    _create_fake_images(europe / "train" / "parus_major", 10, "existing")
    _create_fake_images(source / "parus_major", 15, "new")

    return europe, source


@pytest.fixture
def merge_with_rejected(tmp_path):
    """Setup avec europe_to_trait/ (bonnes) et europe_to_trait_rejected/ (rejetées)."""
    europe = tmp_path / "europe"
    europe_rejected = tmp_path / "europe_rejected"
    source = tmp_path / "europe_to_trait"
    source_rejected = tmp_path / "europe_to_trait_rejected"

    for slug in ("parus_major", "turdus_merula"):
        existing_names = _create_fake_images(europe / "train" / slug, 10, "existing")
        _create_annotations(europe / "train" / slug, existing_names)

        good_names = _create_fake_images(source / slug, 20, "good")
        _create_annotations(source / slug, good_names)

        rej_names = _create_fake_images(source_rejected / slug, 5, "rejected")
        _create_annotations(source_rejected / slug, rej_names)

    return europe, europe_rejected, source, source_rejected


# ---------------------------------------------------------------------------
# UC-M01 : Les images sont déplacées de source vers europe/{train,val,test}
# ---------------------------------------------------------------------------
class TestUCM01_ImagesMovedToSplits:
    def test_images_moved_to_all_three_splits(self, merge_setup):
        from merge_to_europe import merge_species
        europe, source = merge_setup

        merge_species(source / "parus_major", europe, 0.80, 0.10, 42, dry_run=False)

        for split in ("train", "validation", "test"):
            new_count = len([f for f in (europe / split / "parus_major").iterdir()
                            if f.is_file() and f.name.startswith("new")])
            assert new_count > 0, f"Aucune nouvelle image dans {split}"


# ---------------------------------------------------------------------------
# UC-M02 : Aucune image perdue — total source = total ajouté dans les splits
# ---------------------------------------------------------------------------
class TestUCM02_NoImageLost:
    def test_all_images_accounted_for(self, merge_setup):
        from merge_to_europe import merge_species
        europe, source = merge_setup

        stats = merge_species(source / "parus_major", europe, 0.80, 0.10, 42, dry_run=False)

        assert stats["train"] + stats["validation"] + stats["test"] == 20
        assert stats["total"] == 20


# ---------------------------------------------------------------------------
# UC-M03 : Pas de doublon — les nouvelles images ne collisionnent pas
#           avec les existantes
# ---------------------------------------------------------------------------
class TestUCM03_NoDuplicates:
    def test_no_duplicate_filenames(self, merge_setup):
        from merge_to_europe import merge_species
        europe, source = merge_setup

        merge_species(source / "parus_major", europe, 0.80, 0.10, 42, dry_run=False)

        for split in ("train", "validation", "test"):
            sp_dir = europe / split / "parus_major"
            names = [f.name for f in sp_dir.iterdir()
                     if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
            assert len(names) == len(set(names)), f"Doublons dans {split}"


# ---------------------------------------------------------------------------
# UC-M04 : Proportions approximativement 80/10/10 (tolérance ±10%)
# ---------------------------------------------------------------------------
class TestUCM04_Proportions:
    def test_proportions_80_10_10(self, merge_multi_species):
        from merge_to_europe import merge_species
        europe, source = merge_multi_species

        for slug in ("parus_major", "turdus_merula", "erithacus_rubecula"):
            stats = merge_species(source / slug, europe, 0.80, 0.10, 42, dry_run=False)
            total = stats["total"]
            assert 0.70 <= stats["train"] / total <= 0.90, (
                f"{slug}: train={stats['train']}/{total}"
            )


# ---------------------------------------------------------------------------
# UC-M05 : Les annotations.json existants sont enrichis, pas écrasés
# ---------------------------------------------------------------------------
class TestUCM05_AnnotationsMerged:
    def test_existing_annotations_preserved(self, merge_setup):
        from merge_to_europe import merge_species
        europe, source = merge_setup

        merge_species(source / "parus_major", europe, 0.80, 0.10, 42, dry_run=False)

        ann_path = europe / "train" / "parus_major" / "annotations.json"
        with open(ann_path) as f:
            annotations = json.load(f)

        existing_keys = [k for k in annotations if k.startswith("existing")]
        new_keys = [k for k in annotations if k.startswith("new")]
        assert len(existing_keys) == 40, "Annotations existantes perdues"
        assert len(new_keys) > 0, "Nouvelles annotations non ajoutées"


# ---------------------------------------------------------------------------
# UC-M06 : Les nouvelles annotations sont ajoutées dans chaque split
# ---------------------------------------------------------------------------
class TestUCM06_AnnotationsInAllSplits:
    def test_new_annotations_in_all_splits(self, merge_setup):
        from merge_to_europe import merge_species
        europe, source = merge_setup

        merge_species(source / "parus_major", europe, 0.80, 0.10, 42, dry_run=False)

        for split in ("train", "validation", "test"):
            ann_path = europe / split / "parus_major" / "annotations.json"
            assert ann_path.exists(), f"annotations.json manquant dans {split}"
            with open(ann_path) as f:
                annotations = json.load(f)
            new_keys = [k for k in annotations if k.startswith("new")]
            images_in_split = _list_image_names(europe / split / "parus_major")
            new_images = {n for n in images_in_split if n.startswith("new")}
            assert set(new_keys) == new_images, (
                f"{split}: annotations et images désynchronisées"
            )


# ---------------------------------------------------------------------------
# UC-M07 : Dry-run ne déplace rien
# ---------------------------------------------------------------------------
class TestUCM07_DryRun:
    def test_dry_run_no_changes(self, merge_setup):
        from merge_to_europe import merge_species
        europe, source = merge_setup

        before_train = _count_images(europe / "train" / "parus_major")
        before_source = _count_images(source / "parus_major")

        stats = merge_species(source / "parus_major", europe, 0.80, 0.10, 42, dry_run=True)

        after_train = _count_images(europe / "train" / "parus_major")
        after_source = _count_images(source / "parus_major")

        assert after_train == before_train, "Dry-run a modifié train/"
        assert after_source == before_source, "Dry-run a modifié la source"
        assert stats["total"] == 20, "Stats incorrectes en dry-run"


# ---------------------------------------------------------------------------
# UC-M08 : Source nettoyée après fusion — dossiers espèce vides supprimés
# ---------------------------------------------------------------------------
class TestUCM08_SourceCleanedUp:
    def test_source_species_removed_after_merge(self, merge_setup):
        from merge_to_europe import merge_species, cleanup_source
        europe, source = merge_setup

        merge_species(source / "parus_major", europe, 0.80, 0.10, 42, dry_run=False)
        cleanup_source(source)

        assert not (source / "parus_major").exists(), "Dossier source non supprimé"


# ---------------------------------------------------------------------------
# UC-M09 : Déterministe — même seed → même répartition
# ---------------------------------------------------------------------------
class TestUCM09_Deterministic:
    def test_same_seed_same_split(self, tmp_path):
        from merge_to_europe import split_images

        images = [Path(f"img_{i:04d}.jpg") for i in range(30)]

        split1 = split_images(images, 0.80, 0.10, seed=42)
        split2 = split_images(images, 0.80, 0.10, seed=42)

        assert [p.name for p in split1["train"]] == [p.name for p in split2["train"]]
        assert [p.name for p in split1["validation"]] == [p.name for p in split2["validation"]]
        assert [p.name for p in split1["test"]] == [p.name for p in split2["test"]]


# ---------------------------------------------------------------------------
# UC-M10 : Fonctionne sans annotations.json dans la source
# ---------------------------------------------------------------------------
class TestUCM10_NoSourceAnnotations:
    def test_merge_without_annotations(self, merge_no_annotations):
        from merge_to_europe import merge_species
        europe, source = merge_no_annotations

        stats = merge_species(source / "parus_major", europe, 0.80, 0.10, 42, dry_run=False)

        assert stats["total"] == 15
        total_new = 0
        for split in ("train", "validation", "test"):
            total_new += len([f for f in (europe / split / "parus_major").iterdir()
                             if f.is_file() and f.name.startswith("new")])
        assert total_new == 15


# ---------------------------------------------------------------------------
# UC-M11 : Espèce avec 1 seule image — va dans train uniquement
# ---------------------------------------------------------------------------
class TestUCM11_SingleImage:
    def test_single_image_goes_to_train(self, tmp_path):
        from merge_to_europe import merge_species
        europe = tmp_path / "europe"
        source = tmp_path / "europe_to_trait"

        _create_fake_images(europe / "train" / "rare_bird", 5, "existing")
        _create_fake_images(source / "rare_bird", 1, "new")

        stats = merge_species(source / "rare_bird", europe, 0.80, 0.10, 42, dry_run=False)

        assert stats["train"] == 1
        assert stats["validation"] == 0
        assert stats["test"] == 0


# ---------------------------------------------------------------------------
# UC-M12 : Espèce avec 2 images — 1 train + 1 val, 0 test
# ---------------------------------------------------------------------------
class TestUCM12_TwoImages:
    def test_two_images_split(self, tmp_path):
        from merge_to_europe import merge_species
        europe = tmp_path / "europe"
        source = tmp_path / "europe_to_trait"

        _create_fake_images(europe / "train" / "rare_bird", 5, "existing")
        _create_fake_images(source / "rare_bird", 2, "new")

        stats = merge_species(source / "rare_bird", europe, 0.80, 0.10, 42, dry_run=False)

        assert stats["total"] == 2
        assert stats["train"] + stats["validation"] + stats["test"] == 2
        assert stats["train"] >= 1


# ---------------------------------------------------------------------------
# UC-M13 : Les images rejetées sont déplacées vers europe_rejected/train/{slug}
# ---------------------------------------------------------------------------
class TestUCM13_RejectedImagesMoved:
    def test_rejected_moved_to_europe_rejected(self, merge_with_rejected):
        from merge_to_europe import merge_rejected
        europe, europe_rejected, source, source_rejected = merge_with_rejected

        merge_rejected(source_rejected, europe_rejected, dry_run=False)

        for slug in ("parus_major", "turdus_merula"):
            dest = europe_rejected / "train" / slug
            assert dest.exists(), f"{slug}: dossier rejeté non créé"
            rej_count = _count_images(dest)
            assert rej_count == 5, f"{slug}: attendu 5, trouvé {rej_count}"


# ---------------------------------------------------------------------------
# UC-M14 : Aucune image rejetée perdue
# ---------------------------------------------------------------------------
class TestUCM14_NoRejectedLost:
    def test_all_rejected_accounted(self, merge_with_rejected):
        from merge_to_europe import merge_rejected
        europe, europe_rejected, source, source_rejected = merge_with_rejected

        stats = merge_rejected(source_rejected, europe_rejected, dry_run=False)

        assert stats["total"] == 10
        assert _count_images(europe_rejected / "train" / "parus_major") == 5
        assert _count_images(europe_rejected / "train" / "turdus_merula") == 5


# ---------------------------------------------------------------------------
# UC-M15 : Les annotations.json des rejetés sont fusionnés
# ---------------------------------------------------------------------------
class TestUCM15_RejectedAnnotationsMerged:
    def test_rejected_annotations_preserved(self, merge_with_rejected):
        from merge_to_europe import merge_rejected
        europe, europe_rejected, source, source_rejected = merge_with_rejected

        _create_fake_images(europe_rejected / "train" / "parus_major", 3, "old_rej")
        _create_annotations(europe_rejected / "train" / "parus_major",
                           [f"old_rej_{i:04d}.jpg" for i in range(3)])

        merge_rejected(source_rejected, europe_rejected, dry_run=False)

        ann_path = europe_rejected / "train" / "parus_major" / "annotations.json"
        with open(ann_path) as f:
            annotations = json.load(f)
        old_keys = [k for k in annotations if k.startswith("old_rej")]
        new_keys = [k for k in annotations if k.startswith("rejected")]
        assert len(old_keys) == 3, "Annotations existantes perdues"
        assert len(new_keys) == 5, "Nouvelles annotations rejetées manquantes"


# ---------------------------------------------------------------------------
# UC-M16 : Dry-run ne déplace pas les images rejetées
# ---------------------------------------------------------------------------
class TestUCM16_DryRunRejected:
    def test_dry_run_no_rejected_moved(self, merge_with_rejected):
        from merge_to_europe import merge_rejected
        europe, europe_rejected, source, source_rejected = merge_with_rejected

        before = _count_images(source_rejected / "parus_major")

        stats = merge_rejected(source_rejected, europe_rejected, dry_run=True)

        after = _count_images(source_rejected / "parus_major")
        assert after == before, "Dry-run a déplacé des images rejetées"
        assert stats["total"] == 10


# ---------------------------------------------------------------------------
# UC-M17 : Source rejetée nettoyée après fusion
# ---------------------------------------------------------------------------
class TestUCM17_RejectedSourceCleaned:
    def test_rejected_source_cleaned(self, merge_with_rejected):
        from merge_to_europe import merge_rejected, cleanup_source
        europe, europe_rejected, source, source_rejected = merge_with_rejected

        merge_rejected(source_rejected, europe_rejected, dry_run=False)
        cleanup_source(source_rejected)

        assert not (source_rejected / "parus_major").exists()
        assert not (source_rejected / "turdus_merula").exists()


# ---------------------------------------------------------------------------
# UC-M18 : Source rejetée inexistante — pas d'erreur
# ---------------------------------------------------------------------------
class TestUCM18_NoRejectedSource:
    def test_no_rejected_source(self, tmp_path):
        from merge_to_europe import merge_rejected

        stats = merge_rejected(
            tmp_path / "nonexistent", tmp_path / "europe_rejected", dry_run=False
        )
        assert stats["total"] == 0
