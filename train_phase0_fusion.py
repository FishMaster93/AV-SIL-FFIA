"""
HAIL-FFIR Phase 0：在冻结编码器的缓存特征上训练双向跨模态融合模块（Eq. 1-2）+ 临时 CORAL 头，
只用 Phase 0 物种（Red_tilapia_2/3），按验证集挑最佳；之后冻结，为所有阶段导出
f^av（通用层输入）、f'^v、f'^a（物种专属层输入）三组特征，格式与 extract_features.py 相同：
  features/hail_av_<tag>, hail_v_<tag>, hail_a_<tag>
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analytic_cil import load_phase, FEAT_ROOT
from datasets.fish_av_incremental import INCREMENTAL_PHASES
from models.hail_fusion import BidirectionalCrossModalFusion, cosine_consistency
from train_phase0_video import coral_loss, coral_predict, CLASS_NAMES


class FusionWithHead(nn.Module):
    def __init__(self, d, h, dropout):
        super().__init__()
        self.fusion = BidirectionalCrossModalFusion(d, h)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d, 3)

    def forward(self, fv, fa):
        out = self.fusion(fv, fa)
        out['logits'] = self.head(self.drop(out['f_av']))
        return out


def load_split(phase, split, video_feat, audio_feat):
    """按 wav 路径对齐视频特征图和音频特征（不转 float，特征图保持 fp16 省内存）。"""
    v, a = load_phase(video_feat, phase)[split], load_phase(audio_feat, phase)[split]
    a_idx = {p: i for i, p in enumerate(a['paths'])}
    keep = [i for i, p in enumerate(v['paths']) if p in a_idx]
    paths = [v['paths'][i] for i in keep]
    fv = v['feat'][torch.tensor(keep)]
    fa = a['feat'][torch.tensor([a_idx[p] for p in paths])].float()
    return fv, fa, v['labels'][keep], [v['species'][i] for i in keep], paths


@torch.no_grad()
def run(model, fv, fa, device, bs=512):
    model.eval()
    outs = {'f_av': [], 'f_v': [], 'f_a': [], 'logits': []}
    for i in range(0, len(fv), bs):
        o = model(fv[i:i + bs].to(device), fa[i:i + bs].to(device))
        for k in outs:
            outs[k].append(o[k].float().cpu())
    return {k: torch.cat(v) for k, v in outs.items()}


def acc_of(logits, y):
    return float((coral_predict(logits) == y).float().mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--video_feat', default='video_s3d_ft_fmap')
    p.add_argument('--audio_feat', default='audio_cnn14mv2')
    p.add_argument('--tag', default='mv2_s3dft')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--h', type=int, default=256)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--lambda_cos', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no_export', action='store_true')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda'
    out_dir = os.path.join('workspace', f'HAIL_fusion_phase0_{args.tag}')
    os.makedirs(out_dir, exist_ok=True)

    fv_tr, fa_tr, y_tr, _, _ = load_split(0, 'train', args.video_feat, args.audio_feat)
    fv_va, fa_va, y_va, _, _ = load_split(0, 'val', args.video_feat, args.audio_feat)
    fv_te, fa_te, y_te, _, _ = load_split(0, 'test', args.video_feat, args.audio_feat)
    print(f'Phase 0 配对样本 train/val/test = {len(y_tr)}/{len(y_va)}/{len(y_te)}, fv {tuple(fv_tr.shape[1:])}, fa {tuple(fa_tr.shape[1:])}')
    fv_tr, fa_tr, y_tr = fv_tr.to(device), fa_tr.to(device), y_tr.to(device)

    model = FusionWithHead(fa_tr.shape[1], args.h, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * (len(y_tr) // args.batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    best, best_state, hist = -1, None, []
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(y_tr), device=device)
        tl, tc, tn = 0.0, 0.0, 0
        for i in range(0, len(perm) - args.batch_size + 1, args.batch_size):
            idx = perm[i:i + args.batch_size]
            o = model(fv_tr[idx], fa_tr[idx])
            l_cls = coral_loss(o['logits'], y_tr[idx])
            l_cos = cosine_consistency(o['z_v'], o['z_a'])
            loss = l_cls + args.lambda_cos * l_cos
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tl += l_cls.item() * len(idx)
            tc += l_cos.item() * len(idx)
            tn += len(idx)
        va = acc_of(run(model, fv_va, fa_va, device)['logits'], y_va)
        hist.append({'epoch': ep, 'cls_loss': tl / tn, 'cos_loss': tc / tn, 'val_acc': va})
        flag = ''
        if va > best:
            best, flag = va, ' *'
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f'epoch {ep:2d} cls {tl / tn:.4f} cos {tc / tn:.4f} val {va:.4f}{flag}', flush=True)

    model.load_state_dict(best_state)
    te = run(model, fv_te, fa_te, device)
    pred = coral_predict(te['logits'])
    cm = np.zeros((4, 4), dtype=int)
    for t, q in zip(y_te.numpy(), pred.numpy()):
        cm[t, q] += 1
    test_acc = float((pred == y_te).float().mean())
    print(f'== Phase 0 融合 test acc {test_acc:.4f} (best val {best:.4f})')
    print({CLASS_NAMES[k]: round(cm[k, k] / cm[k].sum(), 4) for k in range(4)})
    print(cm)
    torch.save({'model_state_dict': best_state, 'args': vars(args), 'val_acc': best, 'test_acc': test_acc},
               os.path.join(out_dir, 'fusion_best.pt'))
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump({'args': vars(args), 'history': hist, 'best_val': best, 'test_acc': test_acc,
                   'test_cm': cm.tolist()}, f, indent=2)

    if args.no_export:
        return
    # ---- 冻结后为所有阶段导出 f^av / f'^v / f'^a ----
    del fv_tr, fa_tr
    torch.cuda.empty_cache()
    for name in ('av', 'v', 'a'):
        os.makedirs(os.path.join(FEAT_ROOT, f'hail_{name}_{args.tag}'), exist_ok=True)
    for ph in INCREMENTAL_PHASES:
        res = {'av': {}, 'v': {}, 'a': {}}
        for split in ('train', 'val', 'test'):
            fv, fa, y, species, paths = load_split(ph, split, args.video_feat, args.audio_feat)
            o = run(model, fv, fa, device)
            for name, key in (('av', 'f_av'), ('v', 'f_v'), ('a', 'f_a')):
                res[name][split] = {'paths': paths, 'labels': y, 'species': species, 'feat': o[key]}
            del fv
        for name in res:
            res[name]['meta'] = {'phase': ph, 'species': INCREMENTAL_PHASES[ph], 'source': f'HAIL fusion {args.tag}',
                                 'fusion_ckpt': os.path.join(out_dir, 'fusion_best.pt')}
            torch.save(res[name], os.path.join(FEAT_ROOT, f'hail_{name}_{args.tag}', f'phase{ph}.pt'))
        print(f'导出 phase {ph} {INCREMENTAL_PHASES[ph]}', flush=True)


if __name__ == '__main__':
    main()
