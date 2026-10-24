"""
在同一套冻结编码器特征上重跑持续学习基线（论文 Table I）。

所有方法使用相同的特征（audio_cnn14mv2 / video_s3d_ft）、相同的数据划分和物种顺序，
输入可选 audio / video / av（两者拼接）。

梯度类方法共用同一个分类头：LayerNorm -> Linear(d, 512) -> ReLU -> Dropout(0.3) -> Linear(512, 4)
  finetune  : 顺序微调
  joint     : 每个阶段用所有已见物种的数据重新训练（上界，原稿的 Oracle）
  lwf       : Learning without Forgetting，旧模型输出蒸馏（T=2）
  ewc       : Online EWC，经验 Fisher 对角
  er        : 样本回放（固定总预算，按 物种x强度 均分，随机选样）
  icarl     : 回放（herding 选样）+ 输出蒸馏；分别报告 NME 和 FC 分类
  avcil     : AV-CIL [Pian et al.] 的核心机制在冻结特征上的复现：回放 + 输出蒸馏
              + 音视频相似度蒸馏（批内音频-视频跨模态相似度矩阵在新旧模型间保持一致）；仅 av 输入
解析类方法（固定随机投影 8192/模态 + ReLU，闭式解精确累加）：
  acil      : 单头 one-hot 岭回归 [Zhuang et al.]
  aqualytic : 单头 CORAL 序数岭回归 [Xiao et al.]
  dsal      : DS-AL [Zhuang et al.]：主流（ReLU）+ 补偿流（Tanh，拟合主流残差），预测 = 主流 + C·补偿流
  mmal      : MMAL [Yue et al.] 风格：融合头 + 两个模态专属补偿头，预测 = 融合头 + α·(音频头 + 视频头)；仅 av 输入
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analytic_cil import RandomBuffer
from datasets.fish_av_incremental import INCREMENTAL_PHASES
from sahl_cil import A_RAW, V_RAW, load_aligned

C = 4
DEV = 'cuda'
FEAT_DIM = {'audio': 1024, 'video': 1024}   # 其他数据集可在外部改写


# ----------------------------------------------------------------------------- 数据

def load_all(split):
    """{phase: (feat_audio, feat_video, y)}，按配对样本对齐。"""
    out = {}
    for ph in INCREMENTAL_PHASES:
        fd, y, _ = load_aligned([A_RAW, V_RAW], ph, split)
        out[ph] = (fd[A_RAW].to(DEV), fd[V_RAW].to(DEV), y.to(DEV))
    return out


def inp(modality, fa, fv):
    return {'audio': fa, 'video': fv, 'av': torch.cat([fa, fv], 1)}[modality]


# ----------------------------------------------------------------------------- 梯度类

class Head(nn.Module):
    def __init__(self, d_in, hidden=512, dropout=0.3):
        super().__init__()
        self.body = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.fc = nn.Linear(hidden, C)

    def forward(self, x, return_emb=False):
        e = self.body(x)
        return (self.fc(e), e) if return_emb else self.fc(e)


class AVHead(nn.Module):
    """AV-CIL 用的双分支头：音频、视频各一个 256 维分支，拼接后分类（总隐藏维度 512，与 Head 相同）。"""

    def __init__(self, d_a=None, d_v=None, hidden=256, dropout=0.3):
        d_a, d_v = d_a or FEAT_DIM['audio'], d_v or FEAT_DIM['video']
        super().__init__()
        self.d_a = d_a
        self.a = nn.Sequential(nn.LayerNorm(d_a), nn.Linear(d_a, hidden), nn.ReLU())
        self.v = nn.Sequential(nn.LayerNorm(d_v), nn.Linear(d_v, hidden), nn.ReLU())
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden, C)

    def forward(self, x, return_emb=False):
        e = torch.cat([self.a(x[:, :self.d_a]), self.v(x[:, self.d_a:])], 1)
        out = self.fc(self.drop(e))
        return (out, e) if return_emb else out


def kd_loss(logits, old_logits, T=2.0):
    return F.kl_div(F.log_softmax(logits / T, 1), F.softmax(old_logits / T, 1), reduction='batchmean') * T * T


def sim_distill(emb_new, emb_old, d_a):
    """AV-CIL 风格的音视频相似度蒸馏：AVHead 的隐藏表征前半为音频分支、后半为视频分支，
    要求新模型的批内音频-视频跨模态余弦相似度矩阵与旧模型一致。"""
    def cross(e):
        a, v = F.normalize(e[:, :d_a], dim=1), F.normalize(e[:, d_a:], dim=1)
        return a @ v.T
    return F.mse_loss(cross(emb_new), cross(emb_old))


class GradLearner:
    def __init__(self, method, d_in, hp, seed):
        torch.manual_seed(seed)
        self.method, self.hp, self.seed = method, hp, seed
        self.make = (lambda: AVHead()) if method == 'avcil' else (lambda: Head(d_in))
        self.model = self.make().to(DEV)
        self.old = None
        self.fisher, self.star = None, None
        self.mem_x, self.mem_y = None, None          # 回放记忆（特征 + 标签 + 物种）
        self.mem_sp = None
        self.n_species = 0
        self.seen_x, self.seen_y = [], []            # 仅 joint 使用

    def _train(self, x, y, epochs, lr):
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        g = torch.Generator(device=DEV).manual_seed(self.seed * 100 + self.n_species)
        bs = 256
        for _ in range(epochs):
            self.model.train()
            perm = torch.randperm(len(x), device=DEV, generator=g)
            for i in range(0, len(perm), bs):
                idx = perm[i:i + bs]
                xb, yb = x[idx], y[idx]
                logits, emb = self.model(xb, return_emb=True)
                loss = F.cross_entropy(logits, yb)
                if self.old is not None and self.method in ('lwf', 'icarl', 'avcil'):
                    with torch.no_grad():
                        old_logits, old_emb = self.old(xb, return_emb=True)
                    loss = loss + self.hp['kd'] * kd_loss(logits, old_logits)
                    if self.method == 'avcil':
                        loss = loss + self.hp['sim'] * sim_distill(emb, old_emb, emb.shape[1] // 2)
                if self.method == 'ewc' and self.fisher is not None:
                    pen = sum((f * (p - s) ** 2).sum() for p, f, s in
                              zip(self.model.parameters(), self.fisher, self.star))
                    loss = loss + self.hp['ewc'] / 2 * pen
                opt.zero_grad()
                loss.backward()
                opt.step()

    def learn(self, x, y):
        k = self.n_species
        epochs = self.hp['epochs']
        if self.method == 'joint':
            self.seen_x.append(x)
            self.seen_y.append(y)
            torch.manual_seed(self.seed + k)
            self.model = self.make().to(DEV)
            self._train(torch.cat(self.seen_x), torch.cat(self.seen_y), epochs, self.hp['lr'])
        else:
            tx, ty = x, y
            if self.method in ('er', 'icarl', 'avcil') and self.mem_x is not None:
                tx, ty = torch.cat([x, self.mem_x]), torch.cat([y, self.mem_y])
            self._train(tx, ty, epochs, self.hp['lr'])
        if self.method == 'ewc':
            self._update_fisher(x, y)
        if self.method in ('lwf', 'icarl', 'avcil'):
            self.old = copy.deepcopy(self.model).eval()
            for p_ in self.old.parameters():
                p_.requires_grad = False
        self.n_species += 1
        if self.method in ('er', 'icarl', 'avcil'):
            self._update_memory(x, y, k)

    def _update_fisher(self, x, y):
        self.model.eval()
        fis = [torch.zeros_like(p) for p in self.model.parameters()]
        n = 0
        for i in range(0, len(x), 256):
            self.model.zero_grad()
            loss = F.cross_entropy(self.model(x[i:i + 256]), y[i:i + 256], reduction='sum')
            loss.backward()
            for f, p in zip(fis, self.model.parameters()):
                f += p.grad.detach() ** 2 / len(x[i:i + 256])
            n += 1
        fis = [f / n for f in fis]
        self.fisher = fis if self.fisher is None else [a + b for a, b in zip(self.fisher, fis)]   # online EWC 累加
        self.star = [p.detach().clone() for p in self.model.parameters()]

    def _update_memory(self, x, y, k):
        """固定总预算 K，按 (物种, 强度) 均分；新物种进来时旧单元裁剪到新配额。"""
        K = self.hp['budget']
        # 按实际出现过的 (物种/任务, 类别) 单元均分预算：FFIR 中每个物种都有 4 类，等价于 K // (物种数 x 4)；
        # 类增量数据集（如 AVE）每个任务只含部分类别，不能把预算分给不存在的单元
        n_old = 0 if self.mem_x is None else len({(int(s), int(c)) for s, c in zip(self.mem_sp.tolist(), self.mem_y.tolist())})
        n_cur = int(torch.unique(y).numel())
        per = K // max(1, n_old + n_cur)
        xs, ys, sps = [], [], []
        if self.mem_x is not None:
            for sp in range(k):
                for c in range(C):
                    m = (self.mem_sp == sp) & (self.mem_y == c)
                    xs.append(self.mem_x[m][:per]); ys.append(self.mem_y[m][:per]); sps.append(self.mem_sp[m][:per])
        g = torch.Generator(device=DEV).manual_seed(self.seed * 1000 + k)
        self.model.eval()
        for c in range(C):
            xc = x[y == c]
            if self.method == 'er':
                sel = torch.randperm(len(xc), device=DEV, generator=g)[:per]
            else:   # iCaRL herding：逐个挑选使样本均值最接近类均值的样本
                with torch.no_grad():
                    e = F.normalize(self.model(xc, return_emb=True)[1], dim=1)
                mu = e.mean(0)
                sel, acc = [], torch.zeros_like(mu)
                avail = torch.ones(len(xc), dtype=torch.bool, device=DEV)
                for j in range(min(per, len(xc))):
                    d = ((mu[None] - (acc[None] + e) / (j + 1)) ** 2).sum(1)
                    d[~avail] = float('inf')
                    i = int(d.argmin())
                    sel.append(i); avail[i] = False; acc += e[i]
                sel = torch.tensor(sel, device=DEV, dtype=torch.long)
            xs.append(xc[sel]); ys.append(torch.full((len(sel),), c, device=DEV, dtype=torch.long))
            sps.append(torch.full((len(sel),), k, device=DEV, dtype=torch.long))
        self.mem_x, self.mem_y, self.mem_sp = torch.cat(xs), torch.cat(ys), torch.cat(sps)

    @torch.no_grad()
    def predict(self, x, nme=False):
        self.model.eval()
        logits, e = self.model(x, return_emb=True)
        if not nme:
            return logits.argmax(1)
        em = F.normalize(self.model(self.mem_x, return_emb=True)[1], dim=1)
        means = torch.stack([F.normalize(em[self.mem_y == c].mean(0), dim=0) for c in range(C)])
        return (F.normalize(e, dim=1) @ means.T).argmax(1)


# ----------------------------------------------------------------------------- 解析类

def aug(x):
    return torch.cat([x, torch.ones(len(x), 1, device=x.device)], 1)


class Ridge:
    def __init__(self, dim, lam, n_out):
        self.A = torch.zeros(dim, dim, device=DEV)
        self.B = torch.zeros(dim, n_out, device=DEV)
        self.lam, self.W = lam, None

    def add(self, X, T):
        self.A += X.T @ X
        self.B += X.T @ T
        self.W = torch.linalg.solve(self.A + self.lam * torch.eye(len(self.A), device=DEV), self.B)

    def __call__(self, X):
        return X @ self.W


def coral_targets(y, margin=4.0):
    return torch.stack([torch.where(y > k, margin, -margin) for k in range(C - 1)], 1).float()


class AnalyticLearner:
    def __init__(self, method, modality, hp, seed, D=8192):
        self.method, self.modality, self.hp = method, modality, hp
        # 音频 Cnn14-MobileNetV2 特征沿用 AquaLytic 的投影写法（in_out），视频用 nn.Linear 写法
        da, dv = FEAT_DIM['audio'], FEAT_DIM['video']
        self.buf = {'audio': RandomBuffer(da, D, seed, DEV, 'in_out'),
                    'video': RandomBuffer(dv, D, seed + 1, DEV, 'linear')}
        self.tanh = {'audio': RandomBuffer(da, D, seed + 2, DEV, 'linear'),
                     'video': RandomBuffer(dv, D, seed + 3, DEV, 'linear')}
        mods = ['audio', 'video'] if modality == 'av' else [modality]
        self.mods = mods
        dim = D * len(mods) + 1
        n_out = C - 1 if method == 'aqualytic' else C
        self.main = Ridge(dim, hp['lam'], n_out)
        if method == 'dsal':
            self.comp = Ridge(dim, hp['lam'], C)
        if method == 'mmal':
            self.uni = {m: Ridge(D + 1, hp['lam'], C) for m in mods}

    def phi(self, fa, fv, kind='relu', mods=None):
        src = {'audio': fa, 'video': fv}
        mods = mods or self.mods
        if kind == 'relu':
            parts = [self.buf[m](src[m]) for m in mods]
        else:   # DS-AL 补偿流：同样的随机投影但换成 Tanh 激活
            parts = [torch.tanh(src[m] @ self.tanh[m].w) for m in mods]
        return aug(torch.cat(parts, 1))

    def learn(self, fa, fv, y):
        for i in range(0, len(y), 8192):
            a, v, yy = fa[i:i + 8192], fv[i:i + 8192], y[i:i + 8192]
            X = self.phi(a, v)
            T = coral_targets(yy) if self.method == 'aqualytic' else F.one_hot(yy, C).float()
            self.main.A += X.T @ X
            self.main.B += X.T @ T
            if self.method == 'mmal':
                for m in self.mods:
                    Xm = self.phi(a, v, mods=[m])
                    self.uni[m].A += Xm.T @ Xm
                    self.uni[m].B += Xm.T @ T
        eye = torch.eye(len(self.main.A), device=DEV)
        self.main.W = torch.linalg.solve(self.main.A + self.main.lam * eye, self.main.B)
        if self.method == 'mmal':
            for m in self.mods:
                r = self.uni[m]
                r.W = torch.linalg.solve(r.A + r.lam * torch.eye(len(r.A), device=DEV), r.B)
        if self.method == 'dsal':   # 补偿流拟合当前主流在当前物种上的残差
            for i in range(0, len(y), 8192):
                a, v, yy = fa[i:i + 8192], fv[i:i + 8192], y[i:i + 8192]
                res = F.one_hot(yy, C).float() - self.main(self.phi(a, v))
                Xc = self.phi(a, v, 'tanh')
                self.comp.A += Xc.T @ Xc
                self.comp.B += Xc.T @ res
            self.comp.W = torch.linalg.solve(self.comp.A + self.comp.lam * eye, self.comp.B)

    @torch.no_grad()
    def predict(self, fa, fv):
        out = []
        for i in range(0, len(fa), 4096):
            a, v = fa[i:i + 4096], fv[i:i + 4096]
            s = self.main(self.phi(a, v))
            if self.method == 'aqualytic':
                out.append(torch.sigmoid(s).sum(1).round().long().clamp(0, C - 1))
                continue
            if self.method == 'dsal':
                s = s + self.hp['C'] * self.comp(self.phi(a, v, 'tanh'))
            if self.method == 'mmal':
                s = s + self.hp['alpha'] * sum(self.uni[m](self.phi(a, v, mods=[m])) for m in self.mods)
            out.append(s.argmax(1))
        return torch.cat(out)


# ----------------------------------------------------------------------------- 运行

ANALYTIC = {'acil', 'aqualytic', 'dsal', 'mmal'}


def run(method, modality, hp, seed, train, evals):
    order = list(INCREMENTAL_PHASES)
    if method in ANALYTIC:
        learner = AnalyticLearner(method, modality, hp, seed)
    else:
        learner = GradLearner(method, inp(modality, train[0][0][:1], train[0][1][:1]).shape[1], hp, seed)
    history, best = [], {}
    variants = ['nme', 'fc'] if method == 'icarl' else [None]
    hist = {v: [] for v in variants}
    bests = {v: {} for v in variants}
    for step, ph in enumerate(order):
        fa, fv, y = train[ph]
        if method in ANALYTIC:
            learner.learn(fa, fv, y)
        else:
            learner.learn(inp(modality, fa, fv), y)
        for v in variants:
            accs, corr, tot = {}, 0, 0
            for q in order[:step + 1]:
                ea, ev, ey = evals[q]
                pred = learner.predict(ea, ev) if method in ANALYTIC else learner.predict(inp(modality, ea, ev), nme=(v == 'nme'))
                c = int((pred == ey).sum())
                accs[q] = c / len(ey)
                corr += c
                tot += len(ey)
            for q, a in accs.items():
                bests[v][q] = max(bests[v].get(q, 0), a)
            fg = float(np.mean([bests[v][q] - accs[q] for q in order[:step]])) if step else 0.0
            hist[v].append({'acc_seen': corr / tot, 'avg_forget': fg, 'per_phase': accs})
    res = {}
    for v in variants:
        name = method if v is None else f'icarl_{v}'
        h = hist[v]
        res[name] = {'final_acc': h[-1]['acc_seen'], 'avg_inc_acc': float(np.mean([x['acc_seen'] for x in h])),
                     'final_avg_forget': h[-1]['avg_forget'], 'history': h}
    return res, learner


GRIDS = {
    'finetune': [{'epochs': e, 'lr': lr} for e in (10, 30) for lr in (1e-3, 3e-4)],
    'joint': [{'epochs': e, 'lr': lr} for e in (10, 30) for lr in (1e-3, 3e-4)],
    'lwf': [{'epochs': 10, 'lr': 1e-3, 'kd': kd} for kd in (0.5, 1, 2, 5)],
    'ewc': [{'epochs': 10, 'lr': 1e-3, 'ewc': w} for w in (1e2, 1e3, 1e4, 1e5)],
    'er': [{'epochs': e, 'lr': 1e-3, 'budget': 2000} for e in (10, 30)],
    'icarl': [{'epochs': 10, 'lr': 1e-3, 'kd': kd, 'budget': 2000} for kd in (0.5, 1, 2)],
    'avcil': [{'epochs': 10, 'lr': 1e-3, 'kd': kd, 'sim': sm, 'budget': 2000} for kd in (0.5, 1) for sm in (1, 10)],
    # 解析方法的 λ 覆盖到 1e5：统一选参标准含验证集损坏精度，强正则对鲁棒性有利，网格不能在边缘截断
    'acil': [{'lam': l} for l in (1, 10, 100, 1e3, 1e4, 1e5)],
    'aqualytic': [{'lam': l} for l in (1, 10, 100, 1e3, 1e4, 1e5)],
    'dsal': [{'lam': l, 'C': c} for l in (10, 100, 1e3, 1e4, 1e5) for c in (0.5, 1.0)],
    'mmal': [{'lam': l, 'alpha': a} for l in (10, 100, 1e3, 1e4, 1e5) for a in (0.25, 0.5, 1.0)],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--methods', nargs='*', default=list(GRIDS))
    p.add_argument('--modalities', nargs='*', default=['audio', 'video', 'av'])
    p.add_argument('--seeds', type=int, nargs='*', default=[42, 43, 44])
    p.add_argument('--out_dir', default='results/baselines')
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    train, val, test = load_all('train'), load_all('val'), load_all('test')

    for method in args.methods:
        for mod in args.modalities:
            if method in ('avcil', 'mmal') and mod != 'av':
                continue
            out_path = os.path.join(args.out_dir, f'{method}_{mod}.json')
            if os.path.exists(out_path):
                print(f'skip {out_path}')
                continue
            # 1) 超参取自统一选参（select_hp.py：½(验证集干净平均增量 + 验证集噪声/缺失损坏)）
            sj = json.load(open(f'results/selection/{method}_{mod}.json'))
            best_hp, sel = sj['best_hp'], [(r['S'], r['hp']) for r in sj['grid']]
            # 2) 测试集多种子
            per_seed = []
            for s in args.seeds:
                r, _ = run(method, mod, best_hp, s, train, test)
                per_seed.append({k: {kk: vv for kk, vv in v.items() if kk != 'history'} for k, v in r.items()})
            summary = {}
            for name in per_seed[0]:
                summary[name] = {m: (float(np.mean([ps[name][m] for ps in per_seed])),
                                     float(np.std([ps[name][m] for ps in per_seed])))
                                 for m in ('final_acc', 'avg_inc_acc', 'final_avg_forget')}
            with open(out_path, 'w') as f:
                json.dump({'method': method, 'modality': mod, 'best_hp': best_hp, 'val_grid': sel,
                           'summary': summary, 'per_seed': per_seed}, f, indent=2)
            for name, sm in summary.items():
                print(f'{name:10s} {mod:6s} hp={best_hp} | final {100 * sm["final_acc"][0]:.2f}±{100 * sm["final_acc"][1]:.2f} '
                      f'avg_inc {100 * sm["avg_inc_acc"][0]:.2f} forget {100 * sm["final_avg_forget"][0]:.2f}', flush=True)


if __name__ == '__main__':
    main()
