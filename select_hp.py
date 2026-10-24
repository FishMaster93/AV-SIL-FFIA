"""
统一的超参选择（AV-SIL-FFIR）：所有方法、所有模态列用同一个验证集标准
    S = ½ ( 验证集干净平均增量准确率  +  验证集"选参用损坏"上的平均准确率（最终模型） )
选参用损坏只有噪声和模态缺失（音视频列 8 种；单模态列只用本模态的 4 种）；
模糊、遮挡、浑浊（9 种）完全不参与选参，测试时作为留出损坏单独报告。
种子 42；每个配置的完整记录写入 results/selection/<method>_<modality>.json。

HAIL-FFIR 的 κ、π_av 只影响推理，对每组 (λ_s, λ_r) 只训练一次，在同一模型上评估全部 (κ, π_av)。
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baselines as BL
from datasets.fish_av_incremental import INCREMENTAL_PHASES
from hail_ffir import HAILFFIR
from sahl_cil import A_RAW, V_RAW, load_aligned

OUT = 'results/selection'
DEV = 'cuda'
SEL = {'av': ['a_noise_1', 'a_noise_2', 'a_noise_3', 'a_missing', 'v_noise_1', 'v_noise_2', 'v_noise_3', 'v_missing'],
       'audio': ['a_noise_1', 'a_noise_2', 'a_noise_3', 'a_missing'],
       'video': ['v_noise_1', 'v_noise_2', 'v_noise_3', 'v_missing']}
HELD_OUT = [f'v_{t}_{i}' for t in ('blur', 'occl', 'haze') for i in (1, 2, 3)]
HAIL_FIXED = dict(stats='proto', m=20, rank=8, s=500, s_deploy=500, tau=10.0)
HAIL_TRAIN_GRID = list(itertools.product((10.0, 100.0, 1e3, 1e4), (10.0, 30.0, 100.0)))      # (λ_s, λ_r)
HAIL_INFER_GRID = list(itertools.product((1.0, 2.0, 4.0), (0.33, 0.5, 0.6, 0.8)))         # (κ, π_av)，仅音视频列


def criterion(avg_inc, corr):
    return 0.5 * (avg_inc + float(np.mean(list(corr.values()))))


def load_split(split, cond=None):
    out = {}
    for ph in INCREMENTAL_PHASES:
        fd, y, _ = (load_aligned([A_RAW, V_RAW], ph, split) if cond is None
                    else load_aligned([A_RAW, V_RAW], ph, split, 'corrupt_val', cond))
        out[ph] = (fd[A_RAW].to(DEV), fd[V_RAW].to(DEV), y.to(DEV))
    return out


class Learner:
    def __init__(self, method, mod, hp, seed):
        self.method, self.mod = method, mod
        if method == 'hail':
            self.kind = 'hail'
            self.m = HAILFFIR(D=8192, lam_s=hp['lam_s'], lam_r=hp['lam_r'], kappa=2.0, prior_av=0.5, seed=seed,
                              modality=mod, **HAIL_FIXED)
        elif method in BL.ANALYTIC:
            self.kind = 'analytic'
            self.m = BL.AnalyticLearner(method, mod, hp, seed)
        else:
            self.kind = 'grad'
            d_in = {'audio': 1024, 'video': 1024, 'av': 2048}[mod]
            self.m = BL.GradLearner(method, d_in, hp, seed)

    def learn(self, fa, fv, y):
        if self.kind == 'hail':
            self.m.learn({A_RAW: fa, V_RAW: fv}, y)
        elif self.kind == 'analytic':
            self.m.learn(fa, fv, y)
        else:
            self.m.learn(BL.inp(self.mod, fa, fv), y)

    def predict(self, fa, fv):
        if self.kind == 'hail':
            return self.m.predict({A_RAW: fa, V_RAW: fv}).to(DEV)
        if self.kind == 'analytic':
            return self.m.predict(fa, fv)
        return self.m.predict(BL.inp(self.mod, fa, fv))      # iCaRL：FC 分类器（与表中 iCaRL-FC 一致）

    def set_infer(self, kappa, prior_av):
        self.m.kappa = kappa
        self.m.prior = {'av': prior_av, 'a': (1 - prior_av) / 2, 'v': (1 - prior_av) / 2}


def acc(learner, data, phases):
    c = t = 0
    for ph in phases:
        fa, fv, y = data[ph]
        c += int((learner.predict(fa, fv) == y).sum())
        t += len(y)
    return c / t


def evaluate_run(method, mod, hp, train, val, corr, infer_grid=None):
    """训练一遍；返回 [(推理超参, avg_inc, {cond: acc})]。infer_grid 为空时只有一个条目。"""
    L = Learner(method, mod, hp, 42)
    order = list(INCREMENTAL_PHASES)
    infer_grid = infer_grid or [None]
    seen_acc = {g: [] for g in infer_grid}
    for step, ph in enumerate(order):
        L.learn(*train[ph])
        for g in infer_grid:
            if g is not None:
                L.set_infer(*g)
            seen_acc[g].append(acc(L, val, order[:step + 1]))
    out = []
    for g in infer_grid:
        if g is not None:
            L.set_infer(*g)
        cacc = {c: acc(L, corr[c], order) for c in SEL[mod]}
        out.append((g, float(np.mean(seen_acc[g])), cacc))
    del L
    torch.cuda.empty_cache()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--methods', nargs='*', default=['hail'] + list(BL.GRIDS))
    p.add_argument('--modalities', nargs='*', default=['av', 'audio', 'video'])
    args = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    train, val = load_split('train'), load_split('val')
    corr = {c: load_split('val', c) for c in SEL['av']}
    for method in args.methods:
        for mod in args.modalities:
            if method in ('avcil', 'mmal') and mod != 'av':
                continue
            fn = f'{OUT}/{method}_{mod}.json'
            if os.path.exists(fn):
                continue
            t0, rows = time.time(), []
            if method == 'hail':
                for ls, lr in HAIL_TRAIN_GRID:
                    hp = {'lam_s': ls, 'lam_r': lr}
                    for g, avg, cacc in evaluate_run(method, mod, hp, train, val, corr,
                                                     HAIL_INFER_GRID if mod == 'av' else None):
                        h = dict(hp, **({'kappa': g[0], 'prior_av': g[1]} if g else {}))
                        rows.append({'hp': h, 'val_avg_inc': avg, 'val_corrupt': cacc, 'S': criterion(avg, cacc)})
                    print(f'  {method} {mod} λ_s={ls} λ_r={lr} done ({time.time() - t0:.0f}s)', flush=True)
            else:
                for hp in BL.GRIDS[method]:
                    (_, avg, cacc), = evaluate_run(method, mod, hp, train, val, corr)
                    rows.append({'hp': hp, 'val_avg_inc': avg, 'val_corrupt': cacc, 'S': criterion(avg, cacc)})
            best = max(rows, key=lambda r: r['S'])
            json.dump({'method': method, 'modality': mod,
                       'criterion': '0.5*(val clean avg_inc + mean val acc over noise/missing corruptions)',
                       'selection_corruptions': SEL[mod], 'held_out_corruptions': HELD_OUT,
                       'best_hp': best['hp'], 'grid': rows}, open(fn, 'w'), indent=1)
            print(f'{method:10s} {mod:6s} best {best["hp"]} S={100 * best["S"]:.2f} '
                  f'(clean {100 * best["val_avg_inc"]:.2f}, corr {100 * np.mean(list(best["val_corrupt"].values())):.2f}) '
                  f'| {len(rows)} configs, {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
