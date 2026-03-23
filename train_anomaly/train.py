"""
train.py
========
Gộp model definition + training loop cho ATMA-V anomaly detection.
Model: MobileNetV3-Small (CNN backbone) + GRU (temporal) → 2 classes (normal / anomaly)

Sử dụng:
  python train_anomaly/train.py                        # train với mặc định
  python train_anomaly/train.py --epochs 50 --batch 32
  python train_anomaly/train.py --resume train_anomaly/checkpoints/last.pt
  python train_anomaly/train.py --no_model             # extract full frame, không dùng YOLO

Nếu chưa có clips → tự động extract trước khi train.
Nếu đã có clips   → bỏ qua bước extract.

Output:
  train_anomaly/checkpoints/best.pt   - model tốt nhất (val accuracy)
  train_anomaly/checkpoints/last.pt   - checkpoint cuối epoch
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from train_anomaly.dataset import create_dataloaders, run_extract, CLIPS_DIR

CLASS_NAMES = ["normal", "anomaly"]
CKPT_DIR    = Path(__file__).parent / "checkpoints"


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

class ATMAnomalyModel(nn.Module):
    """
    MobileNetV3-Small + GRU cho temporal anomaly classification.

    Input : (B, T, C, H, W)  — batch of T-frame clips
    Output: (B, num_classes)  — logits

    Tốc độ ước tính: ~35 fps trên CPU, ~150+ fps trên GPU (T=16 clips)
    """

    def __init__(
        self,
        num_classes: int = 2,
        gru_hidden: int = 128,
        gru_layers: int = 1,
        dropout: float = 0.3,
        freeze_backbone_ratio: float = 0.6,
    ):
        super().__init__()
        self.num_classes = num_classes

        # CNN Backbone: MobileNetV3-Small (pretrained ImageNet 224×224)
        backbone = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.cnn_features = backbone.features
        self.cnn_avgpool  = backbone.avgpool
        feat_dim = 576  # MobileNetV3-Small output channels

        # Freeze phần đầu backbone (early layers → low-level features ổn định)
        layers   = list(self.cnn_features.children())
        n_freeze = int(len(layers) * freeze_backbone_ratio)
        for layer in layers[:n_freeze]:
            for p in layer.parameters():
                p.requires_grad = False

        # Feature projection: 576 → 256
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Temporal GRU
        self.gru = nn.GRU(
            input_size  = 256,
            hidden_size = gru_hidden,
            num_layers  = gru_layers,
            batch_first = True,
            dropout     = dropout if gru_layers > 1 else 0.0,
        )

        # Classifier head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_hidden, num_classes),
        )

    def _cnn(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B*T, C, H, W) → (B*T, feat_dim)"""
        return self.cnn_avgpool(self.cnn_features(x)).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C, H, W) → (B, num_classes)"""
        B, T, C, H, W = x.shape
        cnn_out   = self._cnn(x.view(B * T, C, H, W))       # (B*T, 576)
        projected = self.proj(cnn_out).view(B, T, 256)       # (B, T, 256)
        gru_out, _ = self.gru(projected)                     # (B, T, gru_hidden)
        return self.head(gru_out[:, -1, :])                  # (B, num_classes)

    @torch.no_grad()
    def predict_clip(self, clip: torch.Tensor) -> Tuple[int, float]:
        """
        Inference 1 clip.
        clip: (T, C, H, W) hoặc (1, T, C, H, W)
        Trả về (class_idx, confidence):  0=normal  1=anomaly
        """
        self.eval()
        if clip.dim() == 4:
            clip = clip.unsqueeze(0)
        probs = torch.softmax(self(clip), dim=-1)[0]
        cls   = int(probs.argmax().item())
        return cls, float(probs[cls].item())


def build_model(**kwargs) -> ATMAnomalyModel:
    return ATMAnomalyModel(**kwargs)


def load_model(checkpoint_path: Path, device: str = "cpu") -> ATMAnomalyModel:
    """Tải model từ checkpoint."""
    ckpt  = torch.load(checkpoint_path, map_location=device)
    cfg   = ckpt.get("model_cfg", {})
    model = build_model(**cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"[Model] Loaded from {checkpoint_path} | cfg={cfg}")
    return model


def count_params(model: nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total: {total/1e6:.2f}M | Trainable: {train/1e6:.2f}M"


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None) -> dict:
    model.train()
    total_loss = correct = total = 0

    for clips, labels in loader:
        clips  = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss = criterion(model(clips), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(clips), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        preds       = model(clips).detach().argmax(dim=1) if scaler is None else \
                      criterion.__class__   # reuse logits below
        # Re-forward để lấy preds (đơn giản, chi phí thấp vì eval nhỏ)
        with torch.no_grad():
            logits = model(clips)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += clips.size(0)
        total_loss += loss.item() * clips.size(0)

    return {"loss": total_loss / total, "acc": correct / total}


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes=2) -> dict:
    model.eval()
    total_loss = total = 0
    conf_mat = np.zeros((num_classes, num_classes), dtype=int)

    for clips, labels in loader:
        clips  = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(clips)
        loss   = criterion(logits, labels)
        preds  = logits.argmax(dim=1)

        total_loss += loss.item() * clips.size(0)
        total      += clips.size(0)
        for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
            conf_mat[t][p] += 1

    acc = conf_mat.diagonal().sum() / conf_mat.sum()
    per_class = {}
    for i, name in enumerate(CLASS_NAMES[:num_classes]):
        tp = conf_mat[i, i]
        fp = conf_mat[:, i].sum() - tp
        fn = conf_mat[i, :].sum() - tp
        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-6)
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1}

    return {"loss": total_loss / total, "acc": acc,
            "confusion_matrix": conf_mat, "per_class": per_class}


def print_val_report(metrics: dict, epoch: int):
    print(f"\n  ── Val Epoch {epoch} ──────────────────────────────")
    print(f"  Loss: {metrics['loss']:.4f}  |  Accuracy: {metrics['acc']*100:.2f}%")
    cm = metrics["confusion_matrix"]
    print(f"  Confusion Matrix:")
    print(f"    {'':20s}  Pred:normal  Pred:anomaly")
    for i, name in enumerate(CLASS_NAMES):
        row = "  ".join(f"{cm[i, j]:8d}" for j in range(cm.shape[1]))
        print(f"    True:{name[:14]:14s}  {row}")
    print(f"  Per-class metrics:")
    for name, vals in metrics["per_class"].items():
        print(f"    {name:12s}  P={vals['precision']:.3f}  R={vals['recall']:.3f}  F1={vals['f1']:.3f}")
    print()


def save_checkpoint(model: ATMAnomalyModel, path: Path, model_cfg: dict, extra: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "model_cfg": model_cfg, **extra}, path)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train ATMA-V Anomaly Detection Model")
    # Data
    parser.add_argument("--clips_dir",   type=str,   default=str(CLIPS_DIR))
    parser.add_argument("--no_model",    action="store_true",
                        help="Dùng full frame khi extract (không dùng YOLO)")
    parser.add_argument("--skip",        type=int,   default=2,
                        help="Xử lý mỗi N frame khi extract")
    # Training
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch",       type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--val_ratio",   type=float, default=0.05)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--clip_frames", type=int,   default=16)
    # Model
    parser.add_argument("--gru_hidden",  type=int,   default=128)
    parser.add_argument("--dropout",     type=float, default=0.3)
    # Misc
    parser.add_argument("--device",      type=str,   default="auto")
    parser.add_argument("--resume",      type=str,   default=None)
    parser.add_argument("--out_dir",     type=str,   default=str(CKPT_DIR))
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else args.device
    print(f"\n{'='*60}")
    print(f"  ATMAnomalyModel  (MobileNetV3-Small + GRU)")
    print(f"  Device: {device} | Epochs: {args.epochs} | Batch: {args.batch}")
    print(f"{'='*60}")

    # ── Auto-extract nếu chưa có clips ────────────────────────────────────────
    clips_dir = Path(args.clips_dir)
    run_extract(
        out_dir   = clips_dir,
        use_model = not args.no_model,
        skip      = args.skip,
    )

    # ── Dataloaders ───────────────────────────────────────────────────────────
    train_loader, val_loader = create_dataloaders(
        clips_dir   = clips_dir,
        batch_size  = args.batch,
        val_ratio   = args.val_ratio,
        num_workers = args.num_workers,
        clip_frames = args.clip_frames,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model_cfg = dict(num_classes=2, gru_hidden=args.gru_hidden, dropout=args.dropout)
    model     = build_model(**model_cfg).to(device)
    print(f"  Params: {count_params(model)}")

    start_epoch = 0
    if args.resume:
        ckpt        = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"  Resumed from epoch {start_epoch}")

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    class_weights = train_loader.dataset.get_class_weights().to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)
    scaler        = torch.cuda.amp.GradScaler() if device == "cuda" else None

    # ── Training loop ─────────────────────────────────────────────────────────
    best_acc = 0.0
    out_dir  = Path(args.out_dir)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0           = time.time()
        train_m      = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_m        = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch+1:03d}/{start_epoch+args.epochs:03d}  "
            f"train_loss={train_m['loss']:.4f}  train_acc={train_m['acc']*100:.2f}%  "
            f"val_acc={val_m['acc']*100:.2f}%  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  ({elapsed:.0f}s)"
        )

        if (epoch + 1) % 5 == 0:
            print_val_report(val_m, epoch + 1)

        extra = {"epoch": epoch, "val_acc": val_m["acc"]}
        save_checkpoint(model, out_dir / "last.pt", model_cfg, extra)

        if val_m["acc"] > best_acc:
            best_acc = val_m["acc"]
            save_checkpoint(model, out_dir / "best.pt", model_cfg, extra)
            print(f"  ✅ New best: {best_acc*100:.2f}% → {out_dir}/best.pt")

    print(f"\n{'='*60}")
    print(f"  Training xong! Best val acc: {best_acc*100:.2f}%")
    print(f"  Checkpoint: {out_dir}/best.pt")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
