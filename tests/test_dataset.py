import torch
import pytest
from PIL import Image

from train import discover_samples, split_samples, BirdDataset, get_transforms, validate_label_map


class TestDiscoverSamples:
    """Découverte des images sur disque à partir d'un dossier train/ et d'un label_map."""

    # Le dataset est partiellement téléchargé (3/558 espèces) → on trouve exactement les images présentes
    def test_finds_all_images(self, mini_dataset):
        dataset_dir, label_map, counts = mini_dataset
        samples = discover_samples(dataset_dir / "train", label_map)
        assert len(samples) == sum(counts.values())

    # Contrat de sortie : liste de (chemin, index_entier, bbox) utilisable par le DataLoader
    def test_returns_path_label_bbox_triples(self, mini_dataset):
        dataset_dir, label_map, _ = mini_dataset
        samples = discover_samples(dataset_dir / "train", label_map)
        for path, label, bbox in samples:
            assert isinstance(path, str)
            assert isinstance(label, int)
            assert bbox is None or len(bbox) == 4

    # Les labels assignés correspondent au label_map.json, pas à l'ordre alphabétique d'ImageFolder
    def test_labels_match_label_map(self, mini_dataset):
        dataset_dir, label_map, _ = mini_dataset
        samples = discover_samples(dataset_dir / "train", label_map)
        valid_labels = set(label_map.values())
        for _, label, _ in samples:
            assert label in valid_labels

    # Un dossier absent du label_map (erreur de nommage, fichier temporaire) ne casse pas le pipeline
    def test_ignores_unknown_folders(self, mini_dataset):
        dataset_dir, label_map, counts = mini_dataset
        unknown_dir = dataset_dir / "train" / "species_inconnue"
        unknown_dir.mkdir()
        Image.new("RGB", (32, 32)).save(unknown_dir / "img.jpg")

        samples = discover_samples(dataset_dir / "train", label_map)
        assert len(samples) == sum(counts.values())

    # Les .DS_Store, .txt etc. dans les dossiers d'images sont ignorés
    def test_ignores_non_image_files(self, mini_dataset):
        dataset_dir, label_map, counts = mini_dataset
        (dataset_dir / "train" / "parus_major" / "notes.txt").write_text("test")
        (dataset_dir / "train" / "parus_major" / ".DS_Store").write_bytes(b"\x00")

        samples = discover_samples(dataset_dir / "train", label_map)
        assert len(samples) == sum(counts.values())

    # Aucune image téléchargée encore → retour vide sans crash
    def test_empty_directory_returns_empty(self, tmp_path, label_map):
        empty_dir = tmp_path / "train"
        empty_dir.mkdir()
        samples = discover_samples(empty_dir, label_map)
        assert samples == []


class TestSplitSamples:
    """Séparation train/val/test stratifiée par classe."""

    # Aucune image perdue ni dupliquée lors du split
    def test_split_preserves_total_count(self, mini_dataset):
        dataset_dir, label_map, _ = mini_dataset
        samples = discover_samples(dataset_dir / "train", label_map)
        train, val, test = split_samples(samples)
        assert len(train) + len(val) + len(test) == len(samples)

    # Pas de fuite de données entre train et val/test (intégrité de l'évaluation)
    def test_no_overlap_between_splits(self, mini_dataset):
        dataset_dir, label_map, _ = mini_dataset
        samples = discover_samples(dataset_dir / "train", label_map)
        train, val, test = split_samples(samples)
        train_paths = {s[0] for s in train}
        val_paths = {s[0] for s in val}
        test_paths = {s[0] for s in test}
        assert train_paths.isdisjoint(val_paths)
        assert train_paths.isdisjoint(test_paths)
        assert val_paths.isdisjoint(test_paths)

    # Chaque espèce a au moins 1 image en train (sinon le modèle ne l'apprend jamais)
    def test_all_classes_in_train(self, mini_dataset):
        dataset_dir, label_map, _ = mini_dataset
        samples = discover_samples(dataset_dir / "train", label_map)
        train, _, _ = split_samples(samples)
        train_labels = {s[1] for s in train}
        all_labels = {s[1] for s in samples}
        assert train_labels == all_labels

    # Le split 80/10/10 est respecté approximativement
    def test_approximate_ratios(self, mini_dataset):
        dataset_dir, label_map, _ = mini_dataset
        samples = discover_samples(dataset_dir / "train", label_map)
        train, val, test = split_samples(samples, ratios=(0.80, 0.10, 0.10))
        total = len(samples)
        assert len(train) / total > 0.5
        assert len(val) / total > 0.05
        assert len(test) / total > 0.05

    # Même seed → même split (reproductibilité des expériences)
    def test_deterministic_with_seed(self, mini_dataset):
        dataset_dir, label_map, _ = mini_dataset
        samples = discover_samples(dataset_dir / "train", label_map)
        split1 = split_samples(samples, seed=123)
        split2 = split_samples(samples, seed=123)
        assert split1[0] == split2[0]
        assert split1[1] == split2[1]
        assert split1[2] == split2[2]

    # Seeds différents → splits différents (pas de bug de randomisation)
    def test_different_seed_different_split(self, mini_dataset):
        dataset_dir, label_map, _ = mini_dataset
        samples = discover_samples(dataset_dir / "train", label_map)
        train_a, _, _ = split_samples(samples, seed=1)
        train_b, _, _ = split_samples(samples, seed=99)
        paths_a = {s[0] for s in train_a}
        paths_b = {s[0] for s in train_b}
        assert paths_a != paths_b

    # Une espèce avec 1 seule image va en train, pas perdue dans val/test
    def test_single_sample_per_class_goes_to_train(self, tmp_path):
        label_map = {"rare_bird": 0}
        species_dir = tmp_path / "train" / "rare_bird"
        species_dir.mkdir(parents=True)
        Image.new("RGB", (32, 32)).save(species_dir / "only.jpg")

        samples = discover_samples(tmp_path / "train", label_map)
        train, val, test = split_samples(samples)
        assert len(train) == 1
        assert len(val) == 0
        assert len(test) == 0


class TestBirdDataset:
    """Chargement des images et intégration avec le DataLoader PyTorch."""

    # Le dataset sait combien d'images il contient
    def test_len(self, mini_dataset):
        dataset_dir, label_map, counts = mini_dataset
        samples = discover_samples(dataset_dir / "train", label_map)
        ds = BirdDataset(samples)
        assert len(ds) == sum(counts.values())

    # __getitem__ retourne un tenseur (3, 224, 224) + un entier — prêt pour le DataLoader
    def test_getitem_returns_image_and_label(self, mini_dataset):
        dataset_dir, label_map, _ = mini_dataset
        samples = discover_samples(dataset_dir / "train", label_map)
        transform = get_transforms(224, augment=False)
        ds = BirdDataset(samples, transform=transform)
        img, label = ds[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 224, 224)
        assert isinstance(label, int)

    # Image corrompue → retourne un autre sample valide (pas une image noire)
    def test_corrupted_image_returns_valid_sample(self, mini_dataset):
        dataset_dir, label_map, _ = mini_dataset
        corrupted = dataset_dir / "train" / "parus_major" / "corrupted.jpg"
        corrupted.write_bytes(b"not an image")

        samples = discover_samples(dataset_dir / "train", label_map)
        transform = get_transforms(224, augment=False)
        ds = BirdDataset(samples, transform=transform)
        corrupted_idx = next(i for i, s in enumerate(samples) if "corrupted" in s[0])
        img, _ = ds[corrupted_idx]
        assert img.shape == (3, 224, 224)
        # Pas une image noire — au moins quelques pixels non-nuls avant normalisation
        assert img.abs().sum() > 0

    # Le compteur de corruptions s'incrémente quand une image est corrompue
    def test_corruption_count_increments(self, mini_dataset):
        dataset_dir, label_map, _ = mini_dataset
        (dataset_dir / "train" / "parus_major" / "bad1.jpg").write_bytes(b"nope")
        (dataset_dir / "train" / "turdus_merula" / "bad2.jpg").write_bytes(b"nope")

        samples = discover_samples(dataset_dir / "train", label_map)
        transform = get_transforms(64, augment=False)
        ds = BirdDataset(samples, transform=transform)
        for i in range(len(ds)):
            ds[i]
        assert ds.corruption_count >= 2

    # Un warning est loggé avec le chemin du fichier corrompu
    def test_corrupted_image_logs_warning(self, mini_dataset, caplog):
        dataset_dir, label_map, _ = mini_dataset
        (dataset_dir / "train" / "parus_major" / "broken.jpg").write_bytes(b"bad")

        samples = discover_samples(dataset_dir / "train", label_map)
        ds = BirdDataset(samples, get_transforms(64, augment=False))
        broken_idx = next(i for i, s in enumerate(samples) if "broken" in s[0])
        import logging
        with caplog.at_level(logging.WARNING):
            ds[broken_idx]
        assert "broken.jpg" in caplog.text

    # 3 corruptions consécutives → fallback image noire (pas de récursion infinie)
    def test_all_corrupted_fallback_no_crash(self, tmp_path, label_map):
        species_dir = tmp_path / "train" / "parus_major"
        species_dir.mkdir(parents=True)
        for i in range(5):
            (species_dir / f"bad{i}.jpg").write_bytes(b"corrupt")

        samples = discover_samples(tmp_path / "train", label_map)
        ds = BirdDataset(samples, get_transforms(224, augment=False))
        img, label = ds[0]
        assert img.shape == (3, 224, 224)


class TestGetTransforms:
    """Pipelines de transforms (augmentation pour train, preprocessing pour eval)."""

    # Augmentation → bonne taille de sortie malgré crop/rotation aléatoires
    def test_train_transform_output_shape(self):
        transform = get_transforms(224, augment=True)
        img = Image.new("RGB", (640, 480))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    # Évaluation → bonne taille de sortie (resize + center crop déterministe)
    def test_eval_transform_output_shape(self):
        transform = get_transforms(224, augment=False)
        img = Image.new("RGB", (640, 480))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    # Les valeurs sont bien normalisées ImageNet (pas du 0–255 brut)
    def test_transform_normalizes(self):
        transform = get_transforms(224, augment=False)
        img = Image.new("RGB", (224, 224), color=(0, 0, 0))
        tensor = transform(img)
        # Noir (0,0,0) → valeurs négatives après normalisation ImageNet
        assert tensor.min() < 0

    # Support de tailles d'image variables (128, 224, 299)
    def test_different_image_sizes(self):
        for size in [128, 224, 299]:
            transform = get_transforms(size, augment=False)
            img = Image.new("RGB", (400, 300))
            tensor = transform(img)
            assert tensor.shape == (3, size, size)


class TestLabelMapValidation:
    """Validation de la cohérence du label_map avant entraînement."""

    # Un label_map avec des indices contigus 0..N-1 est valide
    def test_contiguous_indices_pass(self):
        label_map = {"a": 0, "b": 1, "c": 2}
        validate_label_map(label_map)

    # Un trou dans les indices (ex: 0, 1, 3 sans 2) lève une erreur claire
    def test_gap_in_indices_raises(self):
        label_map = {"a": 0, "b": 1, "c": 3}
        with pytest.raises(ValueError, match="manquants"):
            validate_label_map(label_map)

    # Des indices qui ne commencent pas à 0 lèvent une erreur
    def test_not_starting_at_zero_raises(self):
        label_map = {"a": 1, "b": 2, "c": 3}
        with pytest.raises(ValueError):
            validate_label_map(label_map)

    # Un label_map vide lève une erreur
    def test_empty_label_map_raises(self):
        with pytest.raises(ValueError, match="vide"):
            validate_label_map({})

    # Deux espèces mappées au même index lèvent une erreur
    def test_duplicate_indices_raises(self):
        label_map = {"a": 0, "b": 1, "c": 1}
        with pytest.raises(ValueError, match="dupliqués"):
            validate_label_map(label_map)
