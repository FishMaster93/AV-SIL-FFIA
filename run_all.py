"""
统一实验驱动：同一个流程跑所有方法、所有实验设置，结果存为 results/all/<exp>/<method>_<tag>.json。

实验设置 (--exp)：
  main    : 片段级切分（与 AquaLytic 相同），视频特征 video_s3d_ft；测试集 + 17 种模态损坏
  k400    : 同上，但视频特征换成 Kinetics-400 冻结的 S3D（video_s3d_k400），回应 R3-1
  order   : 主设置下，Phase 1-6 用 --orders 指定的若干随机顺序（Phase 0 固定）
  session : 按天切分，编码器在按天切分的 Phase 0 训练集上重新训练（*_sess 特征）
每次运行记录：每一步的已见物种准确率 / 各物种准确率 / 平均遗忘，最终模型在测试集上的逐样本预测
（用于混淆矩阵和各物种分析），以及（main）各损坏条件下的准确率；HAIL-FFIR 另记录模态权重。
基线超参取自 results/baselines/<method>_av.json（验证集选出）。
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baselines as BL
from analytic_cil import FEAT_ROOT
from datasets.fish_av_incremental import INCREMENTAL_PHASES
from hail_ffir import HAILFFIR
from sahl_cil import A_RAW, V_RAW, load_aligned

R = 'results'
FEATS = {'main': ('audio_cnn14mv2', 'video_s3d_ft'), 'order': ('audio_cnn14mv2', 'video_s3d_ft'),
         'k400': ('audio_cnn14mv2', 'video_s3d_k400'), 'session': ('audio_cnn14mv2_sess', 'video_s3d_ft_sess')}
# 主配置：λ_s、λ_r、κ、π_av 取自统一选参（results/selection/hail_av.json），原型设置固定
_SEL = json.load(open('results/selection/hail_av.json'))['best_hp'] if os.path.exists(
    'results/selection/hail_av.json') else dict(lam_s=100.0, lam_r=30.0, kappa=2.0, prior_av=0.5)
HAIL_MAIN = dict(stats='proto', m=20, rank=8, s=500, s_deploy=500, kappa=_SEL['kappa'], prior_av=_SEL['prior_av'])
HAIL_LAM = dict(lam_s=_SEL['lam_s'], lam_r=_SEL['lam_r'])


def hail_lam(modality):
    """单模态版本用各自模态列选出的 λ_s、λ_r。"""
    if modality in ('av', 'av_only'):
        return HAIL_LAM
    hp = json.load(open(f'results/selection/hail_{modality}.json'))['best_hp']
    return dict(lam_s=hp['lam_s'], lam_r=hp['lam_r'])
# HAIL-FFIR 及其消融（全部基于主配置，只改一个开关）
HAIL_VARIANTS = {
    'hail':              {},
    'hail_exact':        dict(stats='exact'),
    'abl_shared_av':     dict(modality='av_only', experts=False, reliability=False),
    'abl_experts_av':    dict(modality='av_only', reliability=False),
    'abl_shared_rel':    dict(experts=False),
    'abl_fixed_weights': dict(reliability=False),
    'abl_route_av':      dict(route='av'),
    'proto_centroid_unw': dict(proto_mode='centroid_unw'),
    'proto_centroid':    dict(proto_mode='centroid'),
    'proto_diag':        dict(rank=0),
    # 联合训练对照：同一模型（共享头 + 物种专家 + 可靠性），每阶段用全部已见数据求解，专家锚定在最终共享头上
    'joint_hail':        dict(stats='exact'),
    'joint_hail_audio':  dict(stats='exact', modality='audio'),
    'joint_hail_video':  dict(stats='exact', modality='video'),
    # 单模态列（Table II）
    'hail_audio':        dict(modality='audio'),
    'hail_video':        dict(modality='video'),
}
GRAD = ['finetune', 'joint', 'lwf', 'ewc', 'er', 'icarl', 'avcil']
ANALYTIC = ['acil', 'dsal', 'aqualytic', 'mmal']


class Wrapper:
    """统一接口：learn(fd, y)；predict(fd) -> {变体名: preds}"""

    def __init__(self, method, seed):
        self.method = method
        self.seen = [] if method.startswith('joint_hail') else None
        if method.startswith(('hail', 'abl_', 'proto_', 'joint_hail')):
            self.kind = 'hail'
            cfg = dict(HAIL_MAIN, **HAIL_VARIANTS[method])
            self.m = HAILFFIR(tau=10.0, seed=seed, **dict(hail_lam(cfg.get('modality', 'av')), **cfg))
        else:
            hp = json.load(open(f'{R}/baselines/{method}_av.json'))['best_hp']
            if method in ANALYTIC:
                self.kind = 'analytic'
                self.m = BL.AnalyticLearner(method, 'av', hp, seed)
            else:
                self.kind = 'grad'
                self.m = BL.GradLearner(method, 2048, hp, seed)

    def learn(self, fd, y):
        fa, fv, y = fd[A_RAW].to(BL.DEV), fd[V_RAW].to(BL.DEV), y.to(BL.DEV)
        if self.kind == 'hail':
            self.m.learn(fd, y)
            if self.seen is not None:
                self.seen.append((fd, y))
                self.m.reanchor(self.seen)
        elif self.kind == 'analytic':
            self.m.learn(fa, fv, y)
        else:
            self.m.learn(torch.cat([fa, fv], 1), y)

    def predict(self, fd, aux=False):
        fa, fv = fd[A_RAW].to(BL.DEV), fd[V_RAW].to(BL.DEV)
        if self.kind == 'hail':
            if aux:
                p, r, w = self.m.predict(fd, return_aux=True)
                return {self.method: p.cpu()}, {'route': r, 'w': w}
            return {self.method: self.m.predict(fd).cpu()}
        if self.kind == 'analytic':
            out = {self.method: self.m.predict(fa, fv).cpu()}
        elif self.method == 'icarl':
            x = torch.cat([fa, fv], 1)
            out = {'icarl_nme': self.m.predict(x, nme=True).cpu(), 'icarl_fc': self.m.predict(x).cpu()}
        else:
            out = {self.method: self.m.predict(torch.cat([fa, fv], 1)).cpu()}
        return (out, None) if aux else out


def load(exp, ph, split, cond=None):
    af, vf = FEATS[exp]
    if cond is None:
        fd, y, paths = load_aligned([af, vf], ph, split)
    else:
        fd, y, paths = load_aligned([af, vf], ph, split, 'corrupt', cond)
    return {A_RAW: fd[af], V_RAW: fd[vf]}, y, paths


def run_one(method, exp, seed, order, corrupt):
    w = Wrapper(method, seed)
    test = {ph: load(exp, ph, 'test') for ph in order}
    hist, best, t_learn = {}, {}, 0.0
    for step, ph in enumerate(order):
        fd, y, _ = load(exp, ph, 'train')
        t0 = time.time()
        w.learn(fd, y)
        torch.cuda.synchronize()
        t_learn += time.time() - t0
        zero_shot = None
        per = {}
        for q in order[:step + 1]:
            preds = w.predict(test[q][0])
            for v, p in preds.items():
                per.setdefault(v, {})[q] = float((p == test[q][1]).float().mean())
        for v, accs in per.items():
            n = {q: len(test[q][1]) for q in accs}
            acc_seen = sum(accs[q] * n[q] for q in accs) / sum(n.values())
            b = best.setdefault(v, {})
            for q, a in accs.items():
                b[q] = max(b.get(q, 0), a)
            fg = float(np.mean([b[q] - accs[q] for q in order[:step]])) if step else 0.0
            hist.setdefault(v, []).append({'phase': ph, 'acc_seen': acc_seen, 'avg_forget': fg, 'per_phase': accs})
        # 下一阶段物种的零样本（学之前）
        if step + 1 < len(order):
            nxt = order[step + 1]
            nfd, ny, _ = load(exp, nxt, 'test')
            for v, p in w.predict(nfd).items():
                hist[v][-1]['zero_shot_next'] = float((p == ny).float().mean())
    # 最终模型：逐样本预测（混淆矩阵 / 各物种）
    final_preds, labels, species = {}, [], []
    aux_all = None
    for q in order:
        preds, aux = w.predict(test[q][0], aux=True)
        for v, p in preds.items():
            final_preds.setdefault(v, []).append(p)
        labels.append(test[q][1])
        species += [q] * len(test[q][1])
        if aux is not None:
            aux_all = aux_all or {'route': [], 'w': []}
            aux_all['route'].append(aux['route'])
            aux_all['w'].append(aux['w'])
    labels = torch.cat(labels)
    out = {}
    for v, ps in final_preds.items():
        p = torch.cat(ps)
        h = hist[v]
        res = {'final_acc': h[-1]['acc_seen'], 'avg_inc_acc': float(np.mean([x['acc_seen'] for x in h])),
               'final_avg_forget': h[-1]['avg_forget'], 'learn_time_s': t_learn, 'history': h,
               'preds': p.tolist(), 'labels': labels.tolist(), 'species': species, 'order': order}
        if aux_all is not None:
            res['route_acc'] = float((torch.cat(aux_all['route']) == torch.tensor(
                [order.index(s) for s in species])).float().mean())
            res['mean_w_clean'] = torch.cat(aux_all['w']).mean(0).tolist()
        if w.kind == 'hail':
            res['storage_MB'] = w.m.storage_bytes() / 1e6
        out[v] = res
    if corrupt:
        conds = sorted(os.listdir(os.path.join(FEAT_ROOT, 'corrupt')))
        for c in conds:
            corr, tot, ws = {}, 0, []
            for q in order:
                fd, y, _ = load(exp, q, 'test', c)
                preds, aux = w.predict(fd, aux=True)
                for v, p in preds.items():
                    corr[v] = corr.get(v, 0) + int((p == y).sum())
                tot += len(y)
                if aux is not None:
                    ws.append(aux['w'])
            for v in corr:
                out[v].setdefault('corrupt', {})[c] = corr[v] / tot
                if ws:
                    out[v].setdefault('corrupt_mean_w', {})[c] = torch.cat(ws).mean(0).tolist()
        for v in out:
            out[v]['corrupt_mean'] = float(np.mean(list(out[v]['corrupt'].values())))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp', choices=list(FEATS), required=True)
    p.add_argument('--methods', nargs='*', required=True)
    p.add_argument('--seeds', type=int, nargs='*', default=[42, 43, 44])
    p.add_argument('--orders', type=int, default=5, help='order 实验的随机顺序个数')
    args = p.parse_args()
    out_dir = f'{R}/all/{args.exp}'
    os.makedirs(out_dir, exist_ok=True)
    base = list(INCREMENTAL_PHASES)
    if args.exp == 'order':
        rs = np.random.RandomState(2026)
        runs = [(42, [0] + list(rs.permutation(base[1:]))) for _ in range(args.orders)]
    else:
        runs = [(s, base) for s in args.seeds]
    for method in args.methods:
        for i, (seed, order) in enumerate(runs):
            tag = f'order{i}' if args.exp == 'order' else f'seed{seed}'
            fn = f'{out_dir}/{method}_{tag}.json'
            if os.path.exists(fn):
                continue
            t0 = time.time()
            res = run_one(method, args.exp, seed, [int(o) for o in order], corrupt=(args.exp == 'main'))
            for v, r in res.items():
                with open(f'{out_dir}/{v}_{tag}.json', 'w') as f:
                    json.dump(r, f)
                extra = f" corrupt {100 * r['corrupt_mean']:.2f}" if 'corrupt_mean' in r else ''
                print(f'[{args.exp}] {v:20s} {tag:7s} order={order} final {100 * r["final_acc"]:.2f} '
                      f'avg_inc {100 * r["avg_inc_acc"]:.2f} forget {100 * r["final_avg_forget"]:.2f}{extra} '
                      f'| {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
