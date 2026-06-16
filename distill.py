"""Knowledge Distillation : ViT-B/16 (teacher) → MobileNetV2 (student).

Le teacher (gelé) produit des soft targets via softmax(logits/T).
Loss combinée : alpha * KL_div + (1-alpha) * hard_label_loss.
"""

import argparse
import json
import logging
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from export import load_model_from_checkpoint
from train import (
    BirdDataset,
    EarlyStopping,
    FocalLoss,
    ModelEMA,
    TBLogger,
    apply_mixup_cutmix,
    build_optimizer_groups,
    create_model,
    create_scheduler,
    create_weighted_sampler,
    discover_samples,
    evaluate,
    find_problematic_images,
    freeze_backbone,
    generate_report,
    get_device,
    get_transforms,
    plot_training_history,
    save_checkpoint,
    seed_everything,
    validate_label_map,
)

logger = logging.getLogger(__name__)


def distillation_loss(student_logits, teacher_logits, labels, T, alpha, hard_loss_fn=None):
    """Loss combinée : KL divergence (soft targets) + hard label loss.

    La KL divergence est calculée sur log_softmax(student/T) vs softmax(teacher/T),
    multipliée par T² pour compenser l'effet de la température sur les gradients.
    """
    soft_student = F.log_softmax(student_logits / T, dim=1)
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T * T)

    if alpha >= 1.0:
        return soft_loss

    if hard_loss_fn is None:
        hard_loss = F.cross_entropy(student_logits, labels)
    else:
        hard_loss = hard_loss_fn(student_logits, labels)

    return alpha * soft_loss + (1 - alpha) * hard_loss


def load_teacher(checkpoint_path, device):
    """Charge le teacher depuis un checkpoint, le gèle et le met en eval."""
    model, arch, label_map = load_model_from_checkpoint(checkpoint_path, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model = model.to(device)
    return model, arch, label_map


def distill_one_epoch(student, teacher, loader, optimizer, device, T, alpha,
                      scaler=None, *, hard_loss_fn=None, mixup_fn=None,
                      num_classes=0, clip_grad=0.0, ema_model=None):
    """Un epoch de distillation : le student apprend des soft targets du teacher."""
    student.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="  Distill", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        with torch.no_grad():
            teacher_logits = teacher(images)

        optimizer.zero_grad()

        if scaler and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                student_logits = student(images)
                loss = distillation_loss(
                    student_logits, teacher_logits, labels, T, alpha, hard_loss_fn,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            student_logits = student(images)
            loss = distillation_loss(
                student_logits, teacher_logits, labels, T, alpha, hard_loss_fn,
            )
            loss.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), clip_grad)
            optimizer.step()

        if ema_model is not None:
            ema_model.update(student)

        running_loss += loss.item() * images.size(0)
        _, predicted = student_logits.max(1)
        if labels.dim() == 2:
            true_labels = labels.argmax(dim=1)
        else:
            true_labels = labels
        total += images.size(0)
        correct += predicted.eq(true_labels).sum().item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100. * correct / total:.1f}%")

    return running_loss / total, correct / total


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Knowledge Distillation oiseaux")
    p.add_argument("--teacher-checkpoint", type=Path, required=True,
                   help="Checkpoint du modèle teacher (ViT-B/16)")
    p.add_argument("--distill-temperature", type=float, default=4.0)
    p.add_argument("--distill-alpha", type=float, default=0.7)
    p.add_argument("--model",
                   choices=["mobilenetv2", "efficientnet_b0", "efficientnetv2_b2"],
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
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--mixup-alpha", type=float, default=0.2)
    p.add_argument("--cutmix-alpha", type=float, default=1.0)
    p.add_argument("--no-mixup", action="store_true")
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--focal-loss", action="store_true")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--logdir", type=str, default=None,
                   help="Dossier TensorBoard ('auto' = output/runs/<model>_<timestamp>)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    seed_everything(args.seed)

    device = get_device(args.device)
    print(f"Device : {device}")

    print(f"Chargement du teacher : {args.teacher_checkpoint}")
    teacher, teacher_arch, teacher_label_map = load_teacher(
        args.teacher_checkpoint, device
    )
    teacher_num_classes = max(teacher_label_map.values()) + 1
    print(f"Teacher : {teacher_arch} ({teacher_num_classes} classes)")

    with open(args.dataset / "label_map.json") as f:
        label_map = json.load(f)
    validate_label_map(label_map)
    num_classes = max(label_map.values()) + 1

    if num_classes != teacher_num_classes:
        raise ValueError(
            f"Le teacher a {teacher_num_classes} classes mais le dataset en a {num_classes}"
        )

    print(f"Classes : {len(label_map)} (indices 0–{num_classes - 1})")

    train_samples = discover_samples(args.dataset / "train", label_map)
    val_samples = discover_samples(args.dataset / "validation", label_map)
    test_samples = discover_samples(args.dataset / "test", label_map)
    print(f"Images : train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")
    if not train_samples:
        print("Aucune image trouvée dans train/. Vérifiez le dataset.")
        return

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

    student, _, _ = create_model(args.model, num_classes, dropout=args.dropout)
    student = student.to(device)
    print(
        f"Student : {args.model} "
        f"({sum(p.numel() for p in student.parameters()) / 1e6:.1f}M paramètres)"
    )
    print(f"Distillation : T={args.distill_temperature}, alpha={args.distill_alpha}")

    if args.freeze_backbone_epochs > 0:
        freeze_backbone(student, args.model, freeze=True)
        print(f"Backbone gelé pour les {args.freeze_backbone_epochs} premières epochs")

    param_groups = build_optimizer_groups(
        student, args.model, args.lr, args.backbone_lr_factor,
        weight_decay=args.weight_decay,
    )
    optimizer = optim.AdamW(param_groups)

    scheduler = create_scheduler(optimizer, args.epochs, args.warmup_epochs)

    if args.focal_loss:
        hard_loss_fn = FocalLoss(gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
    else:
        hard_loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    ema = ModelEMA(student, decay=args.ema_decay) if not args.no_ema else None

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
        student.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        print(f"Reprise depuis epoch {start_epoch}, best_loss={best_loss:.4f}")

    args.output.mkdir(parents=True, exist_ok=True)

    logdir = None
    if args.logdir == "auto":
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        logdir = str(args.output / "runs" / f"distill_{args.model}_{timestamp}")
    elif args.logdir is not None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        logdir = str(Path(args.logdir) / f"distill_{args.model}_{timestamp}")
    tb = TBLogger(logdir)

    # Critère d'évaluation (sans distillation)
    eval_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    early_stopper = EarlyStopping(patience=args.patience, mode="min") if args.patience > 0 else None

    eval_model = ema.module if ema else student

    print(f"\nDistillation {args.model} ← {teacher_arch} — {args.epochs} epochs")
    print(f"Backbone LR : {args.lr * args.backbone_lr_factor:.1e}, Head LR : {args.lr:.1e}")
    if early_stopper:
        print(f"Early stopping : patience={args.patience} (sur val_loss)")
    if logdir:
        print(f"TensorBoard : {logdir}")
    print()

    for epoch in range(start_epoch, args.epochs):
        if epoch == args.freeze_backbone_epochs and args.freeze_backbone_epochs > 0:
            freeze_backbone(student, args.model, freeze=False)
            print(f"\n  Backbone dégelé à l'epoch {epoch + 1}")

        t0 = time.time()

        train_loss, train_acc = distill_one_epoch(
            student, teacher, train_loader, optimizer, device,
            T=args.distill_temperature, alpha=args.distill_alpha,
            scaler=scaler, hard_loss_fn=hard_loss_fn,
            mixup_fn=mixup_fn, num_classes=num_classes,
            clip_grad=args.clip_grad, ema_model=ema,
        )
        val_loss, val_acc, _, _ = evaluate(eval_model, val_loader, eval_criterion, device)
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
                args.output / f"best_{args.model}_distilled.pth",
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
        args.output / f"last_{args.model}_distilled.pth",
        eval_model, optimizer, scheduler, args.epochs - 1, best_loss, label_map, args.model,
    )

    with open(args.output / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    plot_training_history(history, args.output)

    print(f"\nÉvaluation finale sur le jeu de test ({len(test_ds)} images)...")
    test_loss, test_acc, test_preds, test_labels = evaluate(
        eval_model, test_loader, eval_criterion, device
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
            "teacher": str(args.teacher_checkpoint),
            "temperature": args.distill_temperature,
            "alpha": args.distill_alpha,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "backbone_lr_factor": args.backbone_lr_factor,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "label_smoothing": args.label_smoothing,
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
