"""
Phase 0 视频骨干微调：S3D（Kinetics-400 初始化）+ 临时 CORAL 序数头，只在第一个阶段的
物种（Red_tilapia_2 + Red_tilapia_3）上梯度训练，跟 AquaLytic 音频 Phase 0 的做法对齐：
  - 同样的数据划分（seed=25，物种 x 强度内 80/10/10），见 datasets/fish_av_incremental.py
  - 同样的 CORAL 头和损失、Adam lr=1e-4、按验证集挑最佳模型
之后骨干冻结，临时头丢弃，后续阶段用闭式解更新。

数据增强在 GPU 上做（随机水平翻转 + 轻度随机缩放裁剪），输入为预处理好的
(16, 224, 224, 3) uint8 张量。
"""
import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets.fish_av_incremental import build_phase_items, VideoClipDataset
from models.s3d import S3D, load_s3d_kinetics

CLASS_NAMES = ['None', 'Weak', 'Medium', 'Strong']


def coral_loss(logits, targets, num_classes=4):
    ordinal = torch.stack([(targets > k).float() for k in range(num_classes - 1)], dim=1)
    return F.binary_cross_entropy_with_logits(logits, ordinal)


def coral_predict(logits):
    return torch.sigmoid(logits).sum(dim=-1).round().long().clamp(0, 3)


def make_loss_and_predict(head):
    """head='coral': 跟 AquaLytic 一致的序数头；head='ce': 普通 4 类 softmax 交叉熵（对照）。"""
    if head == 'coral':
        return coral_loss, coral_predict
    return F.cross_entropy, lambda logits: logits.argmax(dim=-1)


class VideoPhase0Model(nn.Module):
    def __init__(self, s3d_ckpt, head='coral', dropout=0.5):
        super().__init__()
        self.backbone = load_s3d_kinetics(S3D(400), s3d_ckpt)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(1024, 3 if head == 'coral' else 4)  # CORAL: 3 个阈值；CE: 4 类

    def forward(self, x):
        emb = self.backbone(x)['embedding']
        return self.head(self.dropout(emb)), emb


def gpu_augment(x, scale=(0.8, 1.0)):
    """x: (B,T,H,W,3) uint8 on GPU -> 同形状 uint8。每个样本独立：随机水平翻转 + 随机裁剪后缩放回原尺寸。"""
    B, T, H, W, C = x.shape
    out = torch.empty_like(x)
    for i in range(B):
        clip = x[i]
        s = np.random.uniform(*scale)
        h, w = int(H * s), int(W * s)
        top, left = np.random.randint(0, H - h + 1), np.random.randint(0, W - w + 1)
        clip = clip[:, top:top + h, left:left + w, :]
        if np.random.rand() < 0.5:
            clip = clip.flip(2)
        clip = clip.permute(0, 3, 1, 2).float()
        clip = F.interpolate(clip, size=(H, W), mode='bilinear', align_corners=False)
        out[i] = clip.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
    return out


@torch.no_grad()
def evaluate(model, loader, device, predict=coral_predict):
    model.eval()
    preds, targets = [], []
    for clip, label, _ in loader:
        clip = clip.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(clip)
        preds.append(predict(logits.float()).cpu())
        targets.append(label)
    preds, targets = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    cm = np.zeros((4, 4), dtype=int)
    for t, p in zip(targets, preds):
        cm[t, p] += 1
    acc = float((preds == targets).mean())
    class_acc = {CLASS_NAMES[k]: float(cm[k, k] / max(cm[k].sum(), 1)) for k in range(4)}
    return {'acc': acc, 'class_acc': class_acc, 'cm': cm.tolist(), 'preds': preds, 'targets': targets}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp_name', default='S3D_phase0_redtilapia')
    p.add_argument('--workspace', default='workspace')
    p.add_argument('--s3d_ckpt', default='Pre_trained_model/S3D400.pt')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=24)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--num_workers', type=int, default=6)
    p.add_argument('--seed', type=int, default=0, help='训练随机种子（不影响数据划分）')
    p.add_argument('--no_augment', action='store_true')
    p.add_argument('--head', choices=['coral', 'ce'], default='coral')
    p.add_argument('--split_mode', choices=['random', 'session'], default='random')
    p.add_argument('--phase', type=int, default=0, help='在哪个物种（阶段）上适应编码器；默认 0 = Red Tilapia')
    args = p.parse_args()

    out_dir = os.path.join(args.workspace, args.exp_name)
    os.makedirs(out_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                        handlers=[logging.FileHandler(os.path.join(out_dir, f'train-{time.time():.0f}.log')),
                                  logging.StreamHandler()])
    log = logging.getLogger()
    log.info(f'args: {vars(args)}')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = 'cuda'

    train, val, test, dropped = build_phase_items(args.phase, args.split_mode)
    log.info(f'Phase 0 视频样本 train/val/test = {len(train)}/{len(val)}/{len(test)}，缺视频剔除 {dropped}')

    def loader(items, shuffle):
        return DataLoader(VideoClipDataset(items), batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=True, drop_last=shuffle,
                          persistent_workers=True, prefetch_factor=4)
    train_loader, val_loader, test_loader = loader(train, True), loader(val, False), loader(test, False)

    model = VideoPhase0Model(args.s3d_ckpt, head=args.head).to(device)
    loss_fn, predict = make_loss_and_predict(args.head)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))

    best_acc, best_path = -1.0, os.path.join(out_dir, 'phase0_best.pt')
    history = []
    for epoch in range(args.epochs):
        model.train()
        t0, tot_loss, n_correct, n_seen = time.time(), 0.0, 0, 0
        for step, (clip, label, _) in enumerate(train_loader):
            clip, label = clip.to(device, non_blocking=True), label.to(device, non_blocking=True)
            if not args.no_augment:
                clip = gpu_augment(clip)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits, _ = model(clip)
            loss = loss_fn(logits.float(), label)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            tot_loss += loss.item() * len(label)
            n_correct += (predict(logits.detach().float()) == label).sum().item()
            n_seen += len(label)
            if step % 100 == 0:
                log.info(f'  epoch {epoch} step {step}/{len(train_loader)} loss {loss.item():.4f}')
        val_stats = evaluate(model, val_loader, device, predict)
        rec = {'epoch': epoch, 'train_loss': tot_loss / n_seen, 'train_acc': n_correct / n_seen,
               'val_acc': val_stats['acc'], 'val_class_acc': val_stats['class_acc'],
               'time_min': (time.time() - t0) / 60}
        history.append(rec)
        log.info(f'Epoch {epoch}: train loss {rec["train_loss"]:.4f} acc {rec["train_acc"]:.4f} | '
                 f'val acc {val_stats["acc"]:.4f} {val_stats["class_acc"]} | {rec["time_min"]:.1f} min')
        if val_stats['acc'] > best_acc:
            best_acc = val_stats['acc']
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'val_acc': best_acc,
                        'args': vars(args)}, best_path)
            log.info(f'  saved best -> {best_path} (val acc {best_acc:.4f})')

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    test_stats = evaluate(model, test_loader, device, predict)
    by_species = {}
    for sp in sorted({it['species'] for it in test}):
        mask = np.array([it['species'] == sp for it in test])
        by_species[sp] = float((test_stats['preds'][mask] == test_stats['targets'][mask]).mean())
    log.info(f'== Phase 0 Test (best epoch {ckpt["epoch"]}, val {ckpt["val_acc"]:.4f}) ==')
    log.info(f'  overall acc {test_stats["acc"]:.4f}')
    log.info(f'  per-class {test_stats["class_acc"]}')
    log.info(f'  per-species {by_species}')
    log.info(f'  confusion matrix (行=真实, 列=预测, None/Weak/Medium/Strong):\n{np.array(test_stats["cm"])}')
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump({'history': history, 'best_epoch': ckpt['epoch'], 'best_val_acc': ckpt['val_acc'],
                   'test_acc': test_stats['acc'], 'test_class_acc': test_stats['class_acc'],
                   'test_by_species': by_species, 'test_cm': test_stats['cm']}, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
