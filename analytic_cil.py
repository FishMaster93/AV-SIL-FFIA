"""
基于预抽特征的闭式解物种增量学习（AquaLytic 机制的音视频版本）。

每个模态：原始特征 -> 固定随机投影 Linear(d, D) + ReLU（Kaiming normal，固定种子，永不训练）
-> CORAL 阈值的岭回归闭式解。每个阶段只把当前物种的 A = sum x x^T、B = sum x y^T 累加进来再解一次，
等价于在所有已见数据上联合求解，且最终结果与物种顺序无关。

融合方式 (--fusion)：
  audio / video : 单模态
  concat        : 两个模态各自随机投影后拼接，共用一个岭回归（早期融合）
  late          : 两个模态各自一个岭回归，阈值分数取平均（晚期融合，固定等权）

指标（与 AquaLytic 对齐，并补充标准定义）：
  acc_seen     : 第 k 阶段在已见物种测试集合并上的准确率（AquaLytic 的 Acc）
  forget_p0    : Phase 0 物种的准确率下降（AquaLytic 的 Forgetting）
  avg_forget   : 对每个旧阶段取 历史最高准确率 - 当前准确率 再平均（持续学习的标准定义）
  zero_shot    : 第 k 阶段更新前，模型在该阶段新物种测试集上的准确率（未见物种/场地的泛化）
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets.fish_av_incremental import INCREMENTAL_PHASES

MARGIN = 4.0          # 跟 AquaLytic 的 analytic_head.MARGIN 一致
FEAT_ROOT = 'features'


# ----------------------------------------------------------------------------- 数据

def load_phase(name, phase):
    return torch.load(os.path.join(FEAT_ROOT, name, f'phase{phase}.pt'), weights_only=False)


def align(phase_feats, split):
    """多个特征集按 wav 路径取交集对齐，返回 ([feat, ...], labels, species)。"""
    dicts = [pf[split] for pf in phase_feats]
    common = set(dicts[0]['paths'])
    for d in dicts[1:]:
        common &= set(d['paths'])
    order = [p for p in dicts[0]['paths'] if p in common]
    feats = []
    for d in dicts:
        idx = {p: i for i, p in enumerate(d['paths'])}
        sel = torch.tensor([idx[p] for p in order])
        feats.append(d['feat'][sel].float())
    ref = dicts[0]
    idx0 = {p: i for i, p in enumerate(ref['paths'])}
    sel0 = [idx0[p] for p in order]
    return feats, ref['labels'][sel0], [ref['species'][i] for i in sel0]


# ----------------------------------------------------------------------------- 模型

class RandomBuffer:
    """固定随机投影 + ReLU，复刻 AquaLytic 的两种初始化写法：
      layout='linear': 权重形状 (out, in)，跟 PANNS_Cnn10.random_proj (nn.Linear) 一样，fan_in=in
      layout='in_out': 权重形状 (in, out)，跟 analytic_trainer_cnn14mv2 一样，kaiming 的 fan_in 取到 out
    两者随机数和尺度都不同，为了逐数复现音频论文分别保留。"""

    def __init__(self, in_dim, out_dim, seed, device, layout='linear'):
        g = torch.Generator().manual_seed(seed)
        shape = (out_dim, in_dim) if layout == 'linear' else (in_dim, out_dim)
        w = torch.empty(*shape)
        torch.nn.init.kaiming_normal_(w, generator=g)
        self.w = (w if layout == 'in_out' else w.T).contiguous().to(device)   # 统一存成 (in, out)

    def __call__(self, x):
        return torch.relu(x @ self.w)


def coral_targets(y, margin=MARGIN):
    return torch.stack([torch.where(y > k, margin, -margin) for k in range(3)], dim=1).float()


def coral_decode(scores):
    return torch.sigmoid(scores).sum(dim=-1).round().long().clamp(0, 3)


class RidgeHead:
    """累加 A/B 的 CORAL 岭回归头（带 bias 列）。"""

    def __init__(self, dim, lam, device, dtype):
        self.A = torch.zeros(dim + 1, dim + 1, device=device, dtype=dtype)
        self.B = torch.zeros(dim + 1, 3, device=device, dtype=dtype)
        self.lam, self.beta = lam, None

    @staticmethod
    def aug(x):
        return torch.cat([x, torch.ones(len(x), 1, device=x.device, dtype=x.dtype)], dim=1)

    def update(self, x, y, chunk=8192):
        for i in range(0, len(x), chunk):
            xa = self.aug(x[i:i + chunk].to(self.A.dtype))
            self.A += xa.T @ xa
            self.B += xa.T @ coral_targets(y[i:i + chunk]).to(xa)
        eye = torch.eye(len(self.A), device=self.A.device, dtype=self.A.dtype)
        self.beta = torch.linalg.solve(self.A + self.lam * eye, self.B)

    def scores(self, x):
        return self.aug(x.to(self.A.dtype)) @ self.beta


class AnalyticModel:
    def __init__(self, fusion, in_dims, dim, lam, device, dtype, seed=42, layouts=None):
        self.fusion = fusion
        layouts = layouts or ['linear'] * len(in_dims)
        # 音频的第一个 buffer 用 seed=42，跟 AquaLytic 完全相同；其他模态顺延
        self.buffers = [RandomBuffer(d, dim, seed + i, device, lay)
                        for i, (d, lay) in enumerate(zip(in_dims, layouts))]
        n_heads = len(in_dims) if fusion == 'late' else 1
        head_dim = dim * len(in_dims) if fusion == 'concat' else dim
        self.heads = [RidgeHead(head_dim, lam, device, dtype) for _ in range(n_heads)]
        self.device = device

    def expand(self, feats):
        return [buf(f.to(self.device)) for buf, f in zip(self.buffers, feats)]

    def update(self, feats, y):
        h = self.expand(feats)
        y = y.to(self.device)
        if self.fusion == 'concat':
            self.heads[0].update(torch.cat(h, dim=1), y)
        else:
            for head, x in zip(self.heads, h):
                head.update(x, y)

    @torch.no_grad()
    def predict(self, feats, chunk=4096):
        preds = []
        n = len(feats[0])
        for i in range(0, n, chunk):
            h = self.expand([f[i:i + chunk] for f in feats])
            if self.fusion == 'concat':
                s = self.heads[0].scores(torch.cat(h, dim=1))
            else:
                s = torch.stack([head.scores(x) for head, x in zip(self.heads, h)]).mean(0)
            preds.append(coral_decode(s.float()).cpu())
        return torch.cat(preds)


# ----------------------------------------------------------------------------- 主流程

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fusion', choices=['audio', 'video', 'concat', 'late'], required=True)
    p.add_argument('--audio_feat', default='audio_cnn10')
    p.add_argument('--video_feat', default='video_s3d_ft')
    p.add_argument('--dim', type=int, default=8192, help='每个模态随机投影维度')
    p.add_argument('--lam', type=float, default=1.0)
    p.add_argument('--order', type=int, nargs='*', default=None,
                   help='Phase 1-6 的顺序（Phase 0 固定为第一个，因为骨干在它上面训练）')
    p.add_argument('--float64', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--eval_split', choices=['val', 'test'], default='test',
                   help='调超参（lam/dim）时用 val，最终报告用 test')
    p.add_argument('--out', default=None, help='结果 json 路径')
    args = p.parse_args()

    names = {'audio': [args.audio_feat], 'video': [args.video_feat],
             'concat': [args.audio_feat, args.video_feat], 'late': [args.audio_feat, args.video_feat]}[args.fusion]
    order = [0] + (args.order if args.order else list(range(1, len(INCREMENTAL_PHASES))))
    assert sorted(order) == list(range(len(INCREMENTAL_PHASES))), order
    device = 'cuda'
    dtype = torch.float64 if args.float64 else torch.float32

    data = {}
    for ph in order:
        pf = [load_phase(n, ph) for n in names]
        data[ph] = {'train': align(pf, 'train'), 'eval': align(pf, args.eval_split)}
    in_dims = [f.shape[1] for f in data[0]['train'][0]]
    # Cnn14_mobilev2 音频特征沿用 AquaLytic 对应实验的投影写法，其余用 nn.Linear 写法
    layouts = ['in_out' if 'mv2' in n else 'linear' for n in names]
    model = AnalyticModel(args.fusion, in_dims, args.dim, args.lam, device, dtype, args.seed, layouts)
    print(f'fusion={args.fusion} feats={names} in_dims={in_dims} D={args.dim} lam={args.lam} '
          f'order={order} eval={args.eval_split}')

    def acc_on(ph):
        feats, y, _ = data[ph]['eval']
        pred = model.predict(feats)
        return float((pred == y).float().mean()), int((pred == y).sum()), len(y)

    history, best_acc_per_phase, rows = [], {}, []
    for step, ph in enumerate(order):
        zero_shot = acc_on(ph)[0] if step > 0 else None
        feats, y, _ = data[ph]['train']
        model.update(feats, y)

        per_phase = {q: acc_on(q) for q in order[:step + 1]}
        n_corr = sum(v[1] for v in per_phase.values())
        n_tot = sum(v[2] for v in per_phase.values())
        acc_seen = n_corr / n_tot
        for q, v in per_phase.items():
            best_acc_per_phase[q] = max(best_acc_per_phase.get(q, 0.0), v[0])
        old = order[:step]
        avg_forget = float(np.mean([best_acc_per_phase[q] - per_phase[q][0] for q in old])) if old else 0.0
        forget_p0 = (history[0]['per_phase'][0] - per_phase[0][0]) if step > 0 else 0.0

        rec = {'step': step, 'phase': ph, 'species': INCREMENTAL_PHASES[ph], 'acc_seen': acc_seen,
               'avg_forget': avg_forget, 'forget_p0': forget_p0, 'zero_shot': zero_shot,
               'per_phase': {q: v[0] for q, v in per_phase.items()}}
        history.append(rec)
        zs = f'{zero_shot:.4f}' if zero_shot is not None else '  -   '
        print(f'step {step} phase {ph} {INCREMENTAL_PHASES[ph]!s:32s} acc_seen {acc_seen:.4f} | '
              f'forget_p0 {forget_p0:.4f} | avg_forget {avg_forget:.4f} | zero-shot {zs} | '
              f'per-phase ' + ' '.join(f'{q}:{v[0]:.3f}' for q, v in per_phase.items()))

    summary = {'final_acc': history[-1]['acc_seen'],
               'avg_inc_acc': float(np.mean([h['acc_seen'] for h in history])),
               'final_forget_p0': history[-1]['forget_p0'],
               'final_avg_forget': history[-1]['avg_forget']}
    print('SUMMARY ' + ' '.join(f'{k}={v:.4f}' for k, v in summary.items()))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump({'args': vars(args), 'summary': summary, 'history': history}, f, indent=2)


if __name__ == '__main__':
    main()
