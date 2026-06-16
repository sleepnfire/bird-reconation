"""
Tests TDD pour le module split_dataset.

Découpe le dataset train/ en train/validation/test (80/10/10)
en déplaçant les images, sans chevauchement, avec seed reproductible.
"""

import json
from pathlib import Path

import pytest


def _create_fake_images(directory: Path, count: int):
    """Crée des faux fichiers .jpg dans un dossier."""
    directory.mkdir(parents=True, exist_ok=True)
    names = []
    for i in range(count):
        name = f"photo_{i:04d}.jpg"
        (directory / name).write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        names.append(name)
    return names


def _build_dataset(tmp_path, species_counts: dict) -> Path:
    """Crée un mini dataset avec N espèces et M images chacune.
    species_counts: {"parus_major": 50, "turdus_merula": 30, ...}
    Retourne le chemin du dossier parent (europe/)."""
    base = tmp_path / "europe"
    train_dir = base / "train"
    for species, count in species_counts.items():
        _create_fake_images(train_dir / species, count)
    return base


@pytest.fixture
def dataset_50(tmp_path):
    """Dataset avec 3 espèces de 50 images chacune."""
    return _build_dataset(tmp_path, {
        "parus_major": 50,
        "turdus_merula": 50,
        "erithacus_rubecula": 50,
    })


@pytest.fixture
def dataset_mixed(tmp_path):
    """Dataset avec des espèces de tailles variées (5, 20, 100)."""
    return _build_dataset(tmp_path, {
        "species_small": 5,
        "species_medium": 20,
        "species_large": 100,
    })


# ---------------------------------------------------------------------------
# UC-S01 : Le split crée les dossiers validation/ et test/ à côté de train/
# ---------------------------------------------------------------------------
class TestUCS01_CreatesDirectories:
    def test_creates_validation_and_test_dirs(self, dataset_50):
        from split_dataset import split_dataset
        split_dataset(str(dataset_50))
        assert (dataset_50 / "validation").is_dir()
        assert (dataset_50 / "test").is_dir()


# ---------------------------------------------------------------------------
# UC-S02 : Chaque espèce est présente dans les 3 splits
# ---------------------------------------------------------------------------
class TestUCS02_AllSpeciesInAllSplits:
    def test_species_present_in_all_splits(self, dataset_50):
        from split_dataset import split_dataset
        split_dataset(str(dataset_50))
        train_species = {d.name for d in (dataset_50 / "train").iterdir() if d.is_dir()}
        val_species = {d.name for d in (dataset_50 / "validation").iterdir() if d.is_dir()}
        test_species = {d.name for d in (dataset_50 / "test").iterdir() if d.is_dir()}
        assert train_species == val_species == test_species


# ---------------------------------------------------------------------------
# UC-S03 : Aucune image n'apparaît dans plusieurs splits (pas de doublon)
# ---------------------------------------------------------------------------
class TestUCS03_NoOverlap:
    def test_no_image_in_multiple_splits(self, dataset_50):
        from split_dataset import split_dataset
        split_dataset(str(dataset_50))
        for species_name in ("parus_major", "turdus_merula", "erithacus_rubecula"):
            train_imgs = {f.name for f in (dataset_50 / "train" / species_name).iterdir()}
            val_imgs = {f.name for f in (dataset_50 / "validation" / species_name).iterdir()}
            test_imgs = {f.name for f in (dataset_50 / "test" / species_name).iterdir()}
            assert train_imgs & val_imgs == set(), f"{species_name}: images en commun train/val"
            assert train_imgs & test_imgs == set(), f"{species_name}: images en commun train/test"
            assert val_imgs & test_imgs == set(), f"{species_name}: images en commun val/test"


# ---------------------------------------------------------------------------
# UC-S04 : Aucune image perdue — le total des 3 splits = total original
# ---------------------------------------------------------------------------
class TestUCS04_NoImageLost:
    def test_total_preserved(self, dataset_50):
        from split_dataset import split_dataset
        split_dataset(str(dataset_50))
        for species_name in ("parus_major", "turdus_merula", "erithacus_rubecula"):
            train_n = len(list((dataset_50 / "train" / species_name).iterdir()))
            val_n = len(list((dataset_50 / "validation" / species_name).iterdir()))
            test_n = len(list((dataset_50 / "test" / species_name).iterdir()))
            assert train_n + val_n + test_n == 50, (
                f"{species_name}: {train_n}+{val_n}+{test_n} ≠ 50"
            )


# ---------------------------------------------------------------------------
# UC-S05 : Proportions approximativement 80/10/10 (tolérance ±5%)
# ---------------------------------------------------------------------------
class TestUCS05_Proportions:
    def test_proportions_80_10_10(self, dataset_50):
        from split_dataset import split_dataset
        split_dataset(str(dataset_50))
        for species_name in ("parus_major", "turdus_merula", "erithacus_rubecula"):
            train_n = len(list((dataset_50 / "train" / species_name).iterdir()))
            val_n = len(list((dataset_50 / "validation" / species_name).iterdir()))
            test_n = len(list((dataset_50 / "test" / species_name).iterdir()))
            total = train_n + val_n + test_n
            assert 0.75 <= train_n / total <= 0.85, (
                f"{species_name}: train={train_n/total:.0%}, attendu ~80%"
            )
            assert 0.05 <= val_n / total <= 0.15, (
                f"{species_name}: val={val_n/total:.0%}, attendu ~10%"
            )
            assert 0.05 <= test_n / total <= 0.15, (
                f"{species_name}: test={test_n/total:.0%}, attendu ~10%"
            )


# ---------------------------------------------------------------------------
# UC-S06 : Déterministe — même seed → même résultat
# ---------------------------------------------------------------------------
class TestUCS06_Deterministic:
    def test_same_seed_same_split(self, tmp_path):
        from split_dataset import split_dataset

        base1 = _build_dataset(tmp_path / "run1", {"parus_major": 50})
        base2 = _build_dataset(tmp_path / "run2", {"parus_major": 50})

        split_dataset(str(base1), seed=42)
        split_dataset(str(base2), seed=42)

        val1 = sorted(f.name for f in (base1 / "validation" / "parus_major").iterdir())
        val2 = sorted(f.name for f in (base2 / "validation" / "parus_major").iterdir())
        assert val1 == val2, "Même seed devrait donner le même split"


# ---------------------------------------------------------------------------
# UC-S07 : Espèce avec peu d'images (5) — au moins 1 en val et 1 en test,
#           le reste en train. Pas de split vide.
# ---------------------------------------------------------------------------
class TestUCS07_SmallSpecies:
    def test_small_species_has_images_in_all_splits(self, dataset_mixed):
        from split_dataset import split_dataset
        split_dataset(str(dataset_mixed))
        for split_name in ("train", "validation", "test"):
            n = len(list((dataset_mixed / split_name / "species_small").iterdir()))
            assert n >= 1, f"species_small: 0 images dans {split_name}"


# ---------------------------------------------------------------------------
# UC-S08 : Espèce avec beaucoup d'images (100) — proportions respectées
# ---------------------------------------------------------------------------
class TestUCS08_LargeSpecies:
    def test_large_species_proportions(self, dataset_mixed):
        from split_dataset import split_dataset
        split_dataset(str(dataset_mixed))
        train_n = len(list((dataset_mixed / "train" / "species_large").iterdir()))
        val_n = len(list((dataset_mixed / "validation" / "species_large").iterdir()))
        test_n = len(list((dataset_mixed / "test" / "species_large").iterdir()))
        total = train_n + val_n + test_n
        assert total == 100
        assert 0.75 <= train_n / total <= 0.85


# ---------------------------------------------------------------------------
# UC-S09 : Ne split pas deux fois — si validation/ existe déjà, erreur ou skip
# ---------------------------------------------------------------------------
class TestUCS09_Idempotent:
    def test_raises_if_already_split(self, dataset_50):
        from split_dataset import split_dataset
        split_dataset(str(dataset_50))
        with pytest.raises(FileExistsError):
            split_dataset(str(dataset_50))


# ---------------------------------------------------------------------------
# UC-S10 : Retourne un résumé avec le compte par split
# ---------------------------------------------------------------------------
class TestUCS10_ReturnsSummary:
    def test_returns_stats(self, dataset_50):
        from split_dataset import split_dataset
        stats = split_dataset(str(dataset_50))
        assert "train" in stats
        assert "validation" in stats
        assert "test" in stats
        assert stats["train"] + stats["validation"] + stats["test"] == 150
        assert stats["species"] == 3
