"""Entraînement de modèles de classification d'oiseaux européens.

Supporte MobileNetV2 (cible Pi Zero + IMX500) et EfficientNet-B0 (cible Pi 5 + Hailo).
Fine-tuning depuis ImageNet avec LR différentiel backbone/tête.
"""

import argparse
import copy
import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from tqdm import tqdm

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TBLogger:
    """Logger TensorBoard optionnel — no-op silencieux quand logdir est None."""

    def __init__(self, logdir):
        if logdir is None:
            self.writer = None
            return
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=logdir)

    def log_scalars(self, scalars, step):
        if self.writer is None:
            return
        for tag, value in scalars.items():
            self.writer.add_scalar(tag, value, step)

    def log_hparams(self, hparam_dict, metric_dict):
        if self.writer is None:
            return
        self.writer.add_hparams(hparam_dict, metric_dict)

    def log_image(self, tag, img_array, step):
        if self.writer is None:
            return
        self.writer.add_image(tag, img_array, step, dataformats="HWC")

    def log_lr(self, optimizer, step):
        if self.writer is None:
            return
        for i, pg in enumerate(optimizer.param_groups):
            self.writer.add_scalar(f"lr/group_{i}", pg["lr"], step)

    def close(self):
        if self.writer is not None:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Entraînement classification oiseaux")
    p.add_argument("--model",
                   choices=["mobilenetv2", "efficientnet_b0", "efficientnetv2_b2", "vit_b_16"],
                   default="mobilenetv2")
    p.add_argument("--dataset", type=Path, default=Path("dataset/europe"))
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--backbone-lr-factor", type=float, default=0.01)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--patience", type=int, default=15,
                   help="Early stopping patience (0=désactivé)")
    p.add_argument("--output", type=Path, default=Path("output"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--freeze-backbone-epochs", type=int, default=3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--randaugment-ops", type=int, default=3)
    p.add_argument("--randaugment-magnitude", type=int, default=12)
    p.add_argument("--clip-grad", type=float, default=1.0,
                   help="Gradient clipping max norm (0=désactivé)")
    p.add_argument("--mixup-alpha", type=float, default=0.2)
    p.add_argument("--cutmix-alpha", type=float, default=1.0)
    p.add_argument("--no-mixup", action="store_true")
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--focal-loss", action="store_true")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--qat", action="store_true",
                   help="Quantization-Aware Training (MobileNetV2 uniquement)")
    p.add_argument("--qat-epochs", type=int, default=5,
                   help="Nombre d'epochs QAT après l'entraînement normal")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile() pour optimiser le modèle (GPU récents)")
    p.add_argument("--logdir", type=str, default=None,
                   help="Dossier TensorBoard ('auto' = output/runs/<model>_<timestamp>)")
    return p.parse_args(argv)


class BirdDataset(Dataset):
    """Dataset d'images d'oiseaux avec gestion des fichiers corrompus."""

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
        self.corruption_count = 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx, _depth=0):
        path, label, bbox = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except (OSError, IOError):
            self.corruption_count += 1
            logger.warning("Image corrompue ignorée: %s", path)
            if _depth < 3 and len(self.samples) > 1:
                alt_idx = random.randint(0, len(self.samples) - 1)
                return self.__getitem__(alt_idx, _depth + 1)
            img = Image.new("RGB", (224, 224))
            bbox = None
        if bbox is not None:
            x, y, w, h = bbox
            img = img.crop((x, y, x + w, y + h))
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(image_size, augment=False, randaugment_ops=3, randaugment_magnitude=12):
    if augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.RandAugment(num_ops=randaugment_ops, magnitude=randaugment_magnitude),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.25),
        ])
    return transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def validate_label_map(label_map):
    if not label_map:
        raise ValueError("label_map est vide")
    indices = sorted(label_map.values())
    if len(set(indices)) != len(indices):
        dupes = {v for v in indices if indices.count(v) > 1}
        raise ValueError(f"label_map contient des indices dupliqués : {dupes}")
    expected = list(range(len(label_map)))
    if indices != expected:
        missing = sorted(set(expected) - set(indices))
        raise ValueError(
            f"Indices non contigus 0..{len(label_map) - 1}. "
            f"Indices manquants : {missing}"
        )


def discover_samples(split_dir, label_map):
    """Découvre les images et charge les bbox depuis annotations.json.

    Retourne une liste de (path, label, bbox) où bbox est [x, y, w, h] ou None.
    """
    samples = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    for class_dir in sorted(Path(split_dir).iterdir()):
        if not class_dir.is_dir():
            continue
        slug = class_dir.name
        if slug not in label_map:
            logger.warning("Dossier '%s' absent du label_map, ignoré", slug)
            continue
        label = label_map[slug]

        annotations = {}
        ann_path = class_dir / "annotations.json"
        if ann_path.exists():
            with open(ann_path) as f:
                annotations = json.load(f)

        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in valid_extensions:
                bbox = None
                ann = annotations.get(img_path.name)
                if ann is not None:
                    bbox = ann["bbox"]
                samples.append((str(img_path), label, bbox))

    return samples


def split_samples(samples, ratios=(0.80, 0.10, 0.10), seed=42):
    """Split stratifié par classe, garantit au moins 1 sample en train par classe."""
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for sample in samples:
        by_class[sample[1]].append(sample)

    train, val, test = [], [], []
    for items in by_class.values():
        rng.shuffle(items)
        n = len(items)
        n_train = max(1, round(n * ratios[0]))
        n_val = max(0 if n < 2 else 1, round(n * ratios[1]))
        if n_train + n_val > n:
            n_val = max(0, n - n_train)
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    return train, val, test


def create_weighted_sampler(samples, num_classes):
    """WeightedRandomSampler pour compenser le déséquilibre de classes."""
    labels = [s[1] for s in samples]
    class_counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    class_weights = np.zeros_like(class_counts)
    np.divide(1.0, class_counts, out=class_weights, where=class_counts > 0)
    sample_weights = torch.tensor(
        [class_weights[s[1]] for s in samples], dtype=torch.float64
    )
    return WeightedRandomSampler(sample_weights, len(sample_weights))


def create_model(arch, num_classes, pretrained=True, dropout=0.2):
    """Crée le modèle avec tête de classification adaptée et retourne les groupes de paramètres."""
    if arch == "mobilenetv2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        backbone_params = list(model.features.parameters())
        head_params = list(model.classifier.parameters())
    elif arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        backbone_params = list(model.features.parameters())
        head_params = list(model.classifier.parameters())
    elif arch == "efficientnetv2_b2":
        import timm
        model = timm.create_model(
            "tf_efficientnetv2_b2.in1k", pretrained=pretrained, num_classes=num_classes,
        )
        in_features = model.num_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        backbone_params = [p for n, p in model.named_parameters() if not n.startswith("classifier")]
        head_params = list(model.classifier.parameters())
    elif arch == "vit_b_16":
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vit_b_16(weights=weights)
        model.heads = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(model.hidden_dim, num_classes),
        )
        backbone_params = (list(model.conv_proj.parameters())
                           + list(model.encoder.parameters())
                           + [model.class_token])
        head_params = list(model.heads.parameters())
    else:
        raise ValueError(f"Architecture inconnue : {arch}")

    return model, backbone_params, head_params


def freeze_backbone(model, arch, freeze=True):
    """Gèle ou dégèle le backbone (features) du modèle."""
    if arch in ("mobilenetv2", "efficientnet_b0"):
        for p in model.features.parameters():
            p.requires_grad = not freeze
    elif arch == "efficientnetv2_b2":
        for name, p in model.named_parameters():
            if not name.startswith("classifier"):
                p.requires_grad = not freeze
    elif arch == "vit_b_16":
        for p in model.conv_proj.parameters():
            p.requires_grad = not freeze
        for p in model.encoder.parameters():
            p.requires_grad = not freeze
        model.class_token.requires_grad = not freeze
    else:
        raise ValueError(f"Architecture inconnue : {arch}")


def build_optimizer_groups(model, arch, lr, backbone_lr_factor, weight_decay):
    if arch == "efficientnetv2_b2":
        backbone_decay, backbone_no_decay = [], []
        for name, p in model.named_parameters():
            if name.startswith("classifier"):
                continue
            (backbone_decay if p.dim() >= 2 else backbone_no_decay).append(p)
        head_decay, head_no_decay = [], []
        for p in model.classifier.parameters():
            (head_decay if p.dim() >= 2 else head_no_decay).append(p)
        return [
            {"params": backbone_decay, "lr": lr * backbone_lr_factor, "weight_decay": weight_decay},
            {"params": backbone_no_decay, "lr": lr * backbone_lr_factor, "weight_decay": 0.0},
            {"params": head_decay, "lr": lr, "weight_decay": weight_decay},
            {"params": head_no_decay, "lr": lr, "weight_decay": 0.0},
        ]
    elif arch in ("mobilenetv2", "efficientnet_b0"):
        backbone_modules = model.features
        head_modules = model.classifier
    elif arch == "vit_b_16":
        backbone_decay, backbone_no_decay = [], []
        for module in [model.conv_proj, model.encoder]:
            for p in module.parameters():
                (backbone_decay if p.dim() >= 2 else backbone_no_decay).append(p)
        backbone_decay.append(model.class_token)

        head_decay, head_no_decay = [], []
        for p in model.heads.parameters():
            (head_decay if p.dim() >= 2 else head_no_decay).append(p)

        return [
            {"params": backbone_decay, "lr": lr * backbone_lr_factor, "weight_decay": weight_decay},
            {"params": backbone_no_decay, "lr": lr * backbone_lr_factor, "weight_decay": 0.0},
            {"params": head_decay, "lr": lr, "weight_decay": weight_decay},
            {"params": head_no_decay, "lr": lr, "weight_decay": 0.0},
        ]
    else:
        raise ValueError(f"Architecture inconnue : {arch}")

    backbone_decay, backbone_no_decay = [], []
    for p in backbone_modules.parameters():
        (backbone_decay if p.dim() >= 2 else backbone_no_decay).append(p)

    head_decay, head_no_decay = [], []
    for p in head_modules.parameters():
        (head_decay if p.dim() >= 2 else head_no_decay).append(p)

    return [
        {"params": backbone_decay, "lr": lr * backbone_lr_factor, "weight_decay": weight_decay},
        {"params": backbone_no_decay, "lr": lr * backbone_lr_factor, "weight_decay": 0.0},
        {"params": head_decay, "lr": lr, "weight_decay": weight_decay},
        {"params": head_no_decay, "lr": lr, "weight_decay": 0.0},
    ]


def create_scheduler(optimizer, epochs, warmup_epochs):
    if warmup_epochs > 0:
        warmup = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.04, total_iters=warmup_epochs
        )
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs
        )
        return optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


def get_device(requested=None):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def apply_mixup_cutmix(images, labels, num_classes, mixup_alpha=0.2, cutmix_alpha=1.0):
    """MixUp ou CutMix (50/50) sur un batch. Retourne (mixed_images, soft_labels [B, C])."""
    batch_size = images.size(0)
    one_hot = torch.zeros(batch_size, num_classes, device=images.device)
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

    indices = torch.randperm(batch_size, device=images.device)
    use_cutmix = random.random() > 0.5

    if use_cutmix and cutmix_alpha > 0:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        _, _, H, W = images.shape
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)
        cy = np.random.randint(H)
        cx = np.random.randint(W)
        y1 = int(np.clip(cy - cut_h // 2, 0, H))
        y2 = int(np.clip(cy + cut_h // 2, 0, H))
        x1 = int(np.clip(cx - cut_w // 2, 0, W))
        x2 = int(np.clip(cx + cut_w // 2, 0, W))
        images[:, :, y1:y2, x1:x2] = images[indices, :, y1:y2, x1:x2]
        lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
    else:
        lam = np.random.beta(mixup_alpha, mixup_alpha) if mixup_alpha > 0 else 1.0
        images = lam * images + (1 - lam) * images[indices]

    soft_labels = lam * one_hot + (1 - lam) * one_hot[indices]
    return images, soft_labels


class ModelEMA:
    """Exponential Moving Average des poids du modèle."""

    def __init__(self, model, decay=0.9999):
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, p in zip(self.module.parameters(), model.parameters()):
            ema_p.data.lerp_(p.data, 1 - self.decay)


class FocalLoss(nn.Module):
    """Focal Loss : pondère les exemples difficiles plus fortement."""

    def __init__(self, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        if targets.dim() == 1:
            num_classes = logits.size(1)
            one_hot = torch.zeros_like(logits)
            one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
            targets = one_hot
        if self.label_smoothing > 0:
            num_classes = logits.size(1)
            targets = targets * (1 - self.label_smoothing) + self.label_smoothing / num_classes
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        p_t = (probs * targets).sum(dim=1)
        focal_weight = (1 - p_t) ** self.gamma
        ce = -(targets * log_probs).sum(dim=1)
        return (focal_weight * ce).mean()


def prepare_qat(model, arch):
    """Prépare le modèle pour le QAT via FX graph mode (fake quantizers)."""
    if arch not in ("mobilenetv2", "efficientnet_b0", "efficientnetv2_b2"):
        raise ValueError(
            f"QAT non supporté pour {arch} — uniquement CNN "
            f"(mobilenetv2, efficientnet_b0, efficientnetv2_b2)"
        )
    from torch.ao.quantization import get_default_qat_qconfig_mapping
    from torch.ao.quantization.quantize_fx import prepare_qat_fx

    backend = "qnnpack" if "qnnpack" in torch.backends.quantized.supported_engines else "x86"
    torch.backends.quantized.engine = backend

    model.train()
    qconfig_mapping = get_default_qat_qconfig_mapping(backend)
    example_input = (torch.randn(1, 3, 224, 224),)
    qat_model = prepare_qat_fx(model, qconfig_mapping, example_input)
    return qat_model


def convert_qat(model):
    """Convertit un modèle QAT en modèle quantifié statique INT8."""
    from torch.ao.quantization.quantize_fx import convert_fx

    model.eval()
    converted = convert_fx(model)
    return converted


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None,
                    *, mixup_fn=None, num_classes=0, clip_grad=0.0, ema_model=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        optimizer.zero_grad()

        if scaler and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        if ema_model is not None:
            ema_model.update(model)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        if labels.dim() == 2:
            true_labels = labels.argmax(dim=1)
        else:
            true_labels = labels
        total += images.size(0)
        correct += predicted.eq(true_labels).sum().item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100. * correct / total:.1f}%")

    return running_loss / total, correct / total


class EarlyStopping:
    def __init__(self, patience=7, mode="max"):
        self.patience = patience
        self.mode = mode
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.counter = 0
        self.best_epoch = 0

    @property
    def best_acc(self):
        return self.best_value

    @best_acc.setter
    def best_acc(self, value):
        self.best_value = value

    def _is_improvement(self, value):
        return value > self.best_value if self.mode == "max" else value < self.best_value

    def step(self, value, epoch):
        if self._is_improvement(value):
            self.best_value = value
            self.counter = 0
            self.best_epoch = epoch
            return False
        self.counter += 1
        return self.counter >= self.patience


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc="  Eval ", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_acc, label_map, arch):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_acc": best_acc,
        "label_map": label_map,
        "arch": arch,
    }, path)


def generate_report(preds, labels, label_map, output_dir):
    """Génère le rapport de classification, la matrice de confusion et les courbes."""
    idx_to_class = {v: k for k, v in label_map.items()}
    present_classes = sorted(set(labels) | set(preds))
    target_names = [idx_to_class.get(i, f"class_{i}") for i in present_classes]

    report = classification_report(
        labels, preds, labels=present_classes,
        target_names=target_names, output_dict=True, zero_division=0,
    )
    with open(output_dir / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(classification_report(
        labels, preds, labels=present_classes,
        target_names=target_names, zero_division=0,
    ))

    cm = confusion_matrix(labels, preds, labels=present_classes)
    np.save(output_dir / "confusion_matrix.npy", cm)

    if len(present_classes) <= 50:
        fig, ax = plt.subplots(
            figsize=(max(12, len(present_classes) * 0.4),
                     max(10, len(present_classes) * 0.35))
        )
        sns.heatmap(
            cm, annot=len(present_classes) <= 30, fmt="d",
            xticklabels=target_names, yticklabels=target_names,
            cmap="Blues", ax=ax,
        )
        ax.set_xlabel("Prédiction")
        ax.set_ylabel("Réalité")
        ax.set_title("Matrice de confusion")
        plt.tight_layout()
        fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
        plt.close(fig)


@torch.no_grad()
def find_problematic_images(model, samples, transform, device, label_map,
                            batch_size=64, num_workers=0, sigma_k=3.0):
    """Analyse per-sample pour détecter les images problématiques (mal labellisées, ambiguës)."""
    model.eval()
    ds = BirdDataset(samples, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    idx_to_class = {v: k for k, v in label_map.items()}
    all_losses, all_preds, all_confs = [], [], []

    for images, labels in tqdm(loader, desc="  Analyse", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        per_sample_loss = F.cross_entropy(logits, labels, reduction="none")
        probs = F.softmax(logits, dim=1)
        confs, preds = probs.max(dim=1)
        all_losses.append(per_sample_loss.cpu())
        all_preds.append(preds.cpu())
        all_confs.append(confs.cpu())

    all_losses = torch.cat(all_losses).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_confs = torch.cat(all_confs).numpy()

    mean_loss = float(all_losses.mean())
    std_loss = float(all_losses.std())
    threshold = mean_loss + sigma_k * std_loss

    image_entries = []
    for i, (path, label, _bbox) in enumerate(samples):
        pred = int(all_preds[i])
        image_entries.append({
            "path": path,
            "loss": float(all_losses[i]),
            "label": label,
            "predicted": pred,
            "label_name": idx_to_class.get(label, f"class_{label}"),
            "predicted_name": idx_to_class.get(pred, f"class_{pred}"),
            "confidence": float(all_confs[i]),
            "correct": label == pred,
        })

    image_entries.sort(key=lambda x: x["loss"], reverse=True)
    problematic_count = sum(1 for e in image_entries if e["loss"] > threshold)

    return {
        "summary": {
            "total_images": len(samples),
            "problematic_count": problematic_count,
            "threshold": threshold,
            "mean_loss": mean_loss,
            "std_loss": std_loss,
            "sigma_k": sigma_k,
        },
        "images": image_entries,
    }


def plot_training_history(history, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Évolution de la loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"], label="Validation")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Évolution de l'accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    seed_everything(args.seed)

    device = get_device(args.device)
    print(f"Device : {device}")

    with open(args.dataset / "label_map.json") as f:
        label_map = json.load(f)
    validate_label_map(label_map)
    num_classes = max(label_map.values()) + 1
    print(f"Classes dans label_map : {len(label_map)} (indices 0–{num_classes - 1})")

    train_samples = discover_samples(args.dataset / "train", label_map)
    val_samples = discover_samples(args.dataset / "validation", label_map)
    test_samples = discover_samples(args.dataset / "test", label_map)
    print(f"Images : train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")
    if not train_samples:
        print("Aucune image trouvée dans train/. Vérifiez le dataset.")
        return

    classes_found = len({s[1] for s in train_samples})
    print(f"Espèces avec images : {classes_found}/{len(label_map)}")

    train_transform = get_transforms(
        args.image_size, augment=True,
        randaugment_ops=args.randaugment_ops,
        randaugment_magnitude=args.randaugment_magnitude,
    )
    eval_transform = get_transforms(args.image_size, augment=False)

    train_ds = BirdDataset(train_samples, train_transform)
    val_ds = BirdDataset(val_samples, eval_transform)
    test_ds = BirdDataset(test_samples, eval_transform)

    sampler = create_weighted_sampler(train_samples, num_classes)
    pin = device.type != "cpu"
    persist = args.workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.workers, pin_memory=pin,
        persistent_workers=persist, prefetch_factor=4 if persist else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=pin,
        persistent_workers=persist, prefetch_factor=4 if persist else None,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=pin,
        persistent_workers=persist, prefetch_factor=4 if persist else None,
    )

    model, _, _ = create_model(args.model, num_classes, dropout=args.dropout)
    model = model.to(device)
    if getattr(args, "compile", False) and hasattr(torch, "compile"):
        model = torch.compile(model)
        print("torch.compile() activé")
    print(f"Modèle : {args.model} ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M paramètres)")
    print(f"Dropout : {args.dropout}, Label smoothing : {args.label_smoothing}")

    if args.freeze_backbone_epochs > 0:
        freeze_backbone(model, args.model, freeze=True)
        print(f"Backbone gelé pour les {args.freeze_backbone_epochs} premières epochs")

    param_groups = build_optimizer_groups(
        model, args.model, args.lr, args.backbone_lr_factor,
        weight_decay=args.weight_decay,
    )
    optimizer = optim.AdamW(param_groups)

    scheduler = create_scheduler(optimizer, args.epochs, args.warmup_epochs)

    if args.focal_loss:
        criterion = FocalLoss(gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    ema = ModelEMA(model, decay=args.ema_decay) if not args.no_ema else None

    if not args.no_mixup:
        mixup_fn = lambda imgs, lbls: apply_mixup_cutmix(
            imgs, lbls, num_classes,
            mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha,
        )
    else:
        mixup_fn = None

    start_epoch = 0
    best_loss = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        print(f"Reprise depuis epoch {start_epoch}, best_loss={best_loss:.4f}")

    args.output.mkdir(parents=True, exist_ok=True)

    logdir = None
    if args.logdir == "auto":
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        logdir = str(args.output / "runs" / f"{args.model}_{timestamp}")
    elif args.logdir is not None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        logdir = str(Path(args.logdir) / f"{args.model}_{timestamp}")
    tb = TBLogger(logdir)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    early_stopper = EarlyStopping(patience=args.patience, mode="min") if args.patience > 0 else None

    print(f"\nEntraînement {args.model} — {args.epochs} epochs")
    print(f"Backbone LR : {args.lr * args.backbone_lr_factor:.1e}, Head LR : {args.lr:.1e}")
    if early_stopper:
        print(f"Early stopping : patience={args.patience} (sur val_loss)")
    if logdir:
        print(f"TensorBoard : {logdir}")
    print()

    eval_model = ema.module if ema else model

    for epoch in range(start_epoch, args.epochs):
        if epoch == args.freeze_backbone_epochs and args.freeze_backbone_epochs > 0:
            freeze_backbone(model, args.model, freeze=False)
            print(f"\n  Backbone dégelé à l'epoch {epoch + 1}")

        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            mixup_fn=mixup_fn, num_classes=num_classes,
            clip_grad=args.clip_grad, ema_model=ema,
        )
        val_loss, val_acc, _, _ = evaluate(eval_model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        tb.log_scalars({
            "loss/train": train_loss,
            "loss/val": val_loss,
            "acc/train": train_acc,
            "acc/val": val_acc,
        }, step=epoch)
        tb.log_lr(optimizer, step=epoch)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch + 1:3d}/{args.epochs} — "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} — "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} — "
            f"{elapsed:.0f}s"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(
                args.output / f"best_{args.model}.pth",
                eval_model, optimizer, scheduler, epoch, best_loss, label_map, args.model,
            )
            print(f"  → Meilleur modèle sauvegardé (val_loss={best_loss:.4f})")

        if early_stopper and early_stopper.step(val_loss, epoch):
            print(
                f"\nEarly stopping à l'epoch {epoch + 1} "
                f"(pas d'amélioration depuis {args.patience} epochs, "
                f"best_loss={early_stopper.best_value:.4f})"
            )
            break

    save_checkpoint(
        args.output / f"last_{args.model}.pth",
        eval_model, optimizer, scheduler, args.epochs - 1, best_loss, label_map, args.model,
    )

    if args.qat:
        print(f"\nQAT — {args.qat_epochs} epochs supplémentaires")
        qat_model = prepare_qat(model, args.model)
        qat_optimizer = optim.AdamW(
            [p for p in qat_model.parameters() if p.requires_grad],
            lr=args.lr * 0.1, weight_decay=args.weight_decay,
        )
        qat_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        for qat_epoch in range(args.qat_epochs):
            t0 = time.time()
            qat_loss, qat_acc = train_one_epoch(
                qat_model, train_loader, qat_criterion, qat_optimizer, device,
            )
            qat_val_loss, qat_val_acc, _, _ = evaluate(
                qat_model, val_loader, qat_criterion, device,
            )
            elapsed = time.time() - t0
            print(
                f"QAT {qat_epoch + 1:3d}/{args.qat_epochs} — "
                f"train_loss={qat_loss:.4f} train_acc={qat_acc:.4f} — "
                f"val_loss={qat_val_loss:.4f} val_acc={qat_val_acc:.4f} — "
                f"{elapsed:.0f}s"
            )
        converted = convert_qat(qat_model)
        torch.save({
            "model_state_dict": converted.state_dict(),
            "label_map": label_map,
            "arch": args.model,
            "qat": True,
        }, args.output / f"best_{args.model}_qat.pth")
        print(f"Modèle QAT sauvegardé : {args.output}/best_{args.model}_qat.pth")

    with open(args.output / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    plot_training_history(history, args.output)

    print(f"\nÉvaluation finale sur le jeu de test ({len(test_ds)} images)...")
    test_loss, test_acc, test_preds, test_labels = evaluate(
        eval_model, test_loader, criterion, device
    )
    print(f"Test — loss={test_loss:.4f} acc={test_acc:.4f}")

    generate_report(test_preds, test_labels, label_map, args.output)

    print(f"\nAnalyse per-sample sur le jeu d'entraînement ({len(train_samples)} images)...")
    problematic = find_problematic_images(
        eval_model, train_samples, eval_transform, device, label_map,
        batch_size=args.batch_size, num_workers=args.workers,
    )
    with open(args.output / "problematic_images.json", "w") as f:
        json.dump(problematic, f, indent=2, ensure_ascii=False)
    ps = problematic["summary"]
    print(
        f"  {ps['problematic_count']} images problématiques "
        f"(loss > {ps['threshold']:.2f}, seuil = μ + {ps['sigma_k']:.0f}σ)"
    )

    tb.log_hparams(
        {
            "model": args.model,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "backbone_lr_factor": args.backbone_lr_factor,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "label_smoothing": args.label_smoothing,
            "image_size": args.image_size,
            "mixup": not args.no_mixup,
            "ema": not args.no_ema,
            "focal_loss": args.focal_loss,
        },
        {
            "hparam/test_acc": test_acc,
            "hparam/test_loss": test_loss,
            "hparam/best_val_loss": best_loss,
        },
    )
    tb.close()

    print(f"\nRésultats sauvegardés dans {args.output}/")


if __name__ == "__main__":
    main()
