"""
标准音视频持续学习基准 AVE（回应 R2-1）：使用 AV-CIL [Pian et al., ICCV 2023] 发布的预抽特征和官方划分，
28 类、4 个任务（每任务 7 类，类别顺序同官方代码），回放预算 340（同 AV-CIL）。
HAIL-FFIR 中的"物种"在此对应"任务"：共享头覆盖已见的全部类别，任务专家以共享头为锚点，
任务由可靠性感知路由推断。所有方法在验证集上选超参，3 个种子；预测只在已见类别中取 argmax。
另做特征层面的模态损坏（拿不到原始音视频）：模态缺失（特征全零，对应无信号输入）和
特征高斯噪声（σ = 0.5/1/2 倍训练集逐维标准差）。

特征文件为第三方 pickle（.npy object 数组），只用 tools/safe_npy.py 的受限反序列化器加载。
"""
import argparse
import glob
import itertools
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tools'))
import baselines as BL
import hail_ffir as HF
import safe_npy
import stream_feats
from sahl_cil import A_RAW, V_RAW

# 各基准的官方设置（AV-CIL 的 run_incremental_ours.sh）：类别数、每任务类别数、回放预算
DATASETS = {
    'ave': dict(meta='AVE', feat='external/ave_features', n_cls=28, per_task=7, budget=340),
    'ks': dict(meta='kinetics-sounds', feat=os.environ.get('AVSIL_KS_ROOT', 'external/ks_features'), n_cls=30, per_task=6, budget=500),
}
DEV = 'cuda'


def configure(ds):
    global DS, META, FEAT_DIR, CACHE, OUT, N_CLS, PER_TASK, N_TASK, BUDGET
    c = DATASETS[ds]
    DS, FEAT_DIR, N_CLS, PER_TASK, BUDGET = ds, c['feat'], c['n_cls'], c['per_task'], c['budget']
    META = 'external/AV-CIL_ICCV2023/data/' + c['meta']
    CACHE = f'features/{ds}/pooled.pt'
    OUT = f'results/{ds}'
    N_TASK = N_CLS // PER_TASK
    for k in ('er', 'icarl', 'avcil'):
        GRIDS[k] = [dict(g, budget=BUDGET) for g in GRIDS[k]]




# ----------------------------------------------------------------------------- 数据

def find(name):
    for root, _, files in os.walk(FEAT_DIR):
        if name in files:
            return os.path.join(root, name)
    raise FileNotFoundError(name)


def build_cache():
    cat = safe_npy.load(f'{META}/all_id_category_dict.npy')
    enc = safe_npy.load(f'{META}/category_encode_dict.npy')
    out = {}
    for mod, fn in (('audio', 'audio_pretrained_feature_dict.npy'), ('visual', 'visual_pretrained_feature_dict.npy')):
        t0 = time.time()
        parts = sorted(glob.glob(f'{FEAT_DIR}/*.zip.a?'))
        if parts and mod == 'visual':   # K-S-CI 的视觉特征是 HDF5（每个视频一个数据集），需随机访问，先解压再逐个池化
            h5 = f'{FEAT_DIR}/visual_features.h5'
            if not os.path.exists(h5):
                stream_feats.extract_member(parts, 'visual_features.h5', h5)
            import h5py
            with h5py.File(h5, 'r') as f:
                out[mod] = {}
                for i, vid in enumerate(f.keys()):
                    a = np.asarray(f[vid][()], dtype=np.float32)
                    out[mod][vid] = a.reshape(-1, a.shape[-1]).mean(0)
                    if i % 2000 == 0:
                        print(f'  {i} visual', flush=True)
                shape = f[vid].shape
        elif parts:   # 分卷 zip 中的 npy：直接流式读取并池化，不解压
            out[mod], shape = stream_feats.load_pooled(stream_feats.open_member(parts, fn), progress=2000)
        else:
            d = safe_npy.load(find(fn))
            shape = np.asarray(next(iter(d.values()))).shape
            out[mod] = {vid: np.asarray(v, dtype=np.float32).reshape(-1, np.asarray(v).shape[-1]).mean(0) for vid, v in d.items()}
            del d
        print(f'{mod}: {len(out[mod])} clips, raw shape {shape} -> {len(next(iter(out[mod].values())))}-d, {time.time() - t0:.0f}s', flush=True)
    data = {}
    for split in ('train', 'val', 'test'):
        vids = [v for v in cat[split] if v in out['audio'] and v in out['visual']]
        data[split] = {'vids': vids,
                       'audio': torch.tensor(np.stack([out['audio'][v] for v in vids])),
                       'visual': torch.tensor(np.stack([out['visual'][v] for v in vids])),
                       'y': torch.tensor([enc[cat[split][v]] for v in vids])}
        print(split, len(vids), flush=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    torch.save(data, CACHE)
    return data


def task_of(y):
    return y // PER_TASK


# ----------------------------------------------------------------------------- 方法封装

class Learner:
    def __init__(self, method, hp, seed, dims):
        self.method, self.hp = method, hp
        if method.startswith('hail'):
            HF.C = N_CLS
            self.m = HF.HAILFFIR(D=8192, lam_s=hp['lam_s'], lam_r=hp['lam_r'], tau=10.0, kappa=hp.get('kappa', 2.0),
                                 prior_av=hp.get('prior_av', 0.5), stats=hp.get('stats', 'proto'), m=hp['m'], rank=8, s=10000 // hp['m'],
                                 s_deploy=10000 // hp['m'], seed=seed, feat_dims={A_RAW: dims[0], V_RAW: dims[1]},
                                 expert_mode=hp.get('expert_mode', 'snapshot'), experts=hp.get('experts', True),
                                 reliability=hp.get('reliability', True), modality=hp.get('modality', 'av'))
            self.kind = 'hail'
        elif method in ('acil', 'dsal', 'mmal'):
            self.m = BL.AnalyticLearner(method, 'av', hp, seed)
            self.kind = 'analytic'
        elif method == 'joint_ridge':   # 同一随机特征映射上，每步用全部已见数据一次性求岭回归（解析上界）
            self.seed, self.seen = seed, []
            self.kind = 'joint_ridge'
        else:
            self.m = BL.GradLearner(method, dims[0] + dims[1], hp, seed)
            self.kind = 'grad'

    def learn(self, a, v, y):
        if self.kind == 'hail':
            self.m.learn({A_RAW: a, V_RAW: v}, y)
        elif self.kind == 'joint_ridge':
            self.seen.append((a, v, y))
            self.m = BL.AnalyticLearner('acil', 'av', self.hp, self.seed)
            self.m.learn(*(torch.cat(z).to(DEV) for z in zip(*self.seen)))
        elif self.kind == 'analytic':
            self.m.learn(a.to(DEV), v.to(DEV), y.to(DEV))
        else:
            self.m.learn(torch.cat([a, v], 1).to(DEV), y.to(DEV))

    @torch.no_grad()
    def scores(self, a, v, seen):
        """返回 {变体: (N, 28) 得分}，未见类别置为 -inf。"""
        if self.kind == 'hail':
            out = {'hail': self.m.predict({A_RAW: a, V_RAW: v}, return_scores=True)}
        elif self.kind in ('analytic', 'joint_ridge'):
            m = self.m
            a, v = a.to(DEV), v.to(DEV)
            s = m.main(m.phi(a, v))
            if m.method == 'dsal':
                s = s + m.hp['C'] * m.comp(m.phi(a, v, 'tanh'))
            if m.method == 'mmal':
                s = s + m.hp['alpha'] * sum(m.uni[mm](m.phi(a, v, mods=[mm])) for mm in m.mods)
            out = {self.method: s.cpu()}  # joint_ridge 用 acil 的打分
        else:
            x = torch.cat([a, v], 1).to(DEV)
            self.m.model.eval()
            logits, emb = self.m.model(x, return_emb=True)
            out = {self.method if self.method != 'icarl' else 'icarl_fc': logits.cpu()}
            if self.method == 'icarl':   # NME：只在已见类别的样本均值之间比较
                em = F.normalize(self.m.model(self.m.mem_x, return_emb=True)[1], dim=1)
                means = torch.stack([F.normalize(em[self.m.mem_y == c].mean(0), dim=0) if (self.m.mem_y == c).any()
                                     else torch.zeros(em.shape[1], device=DEV) for c in range(N_CLS)])
                out['icarl_nme'] = (F.normalize(emb, dim=1) @ means.T).cpu()
        mask = torch.full((N_CLS,), float('-inf'))
        mask[seen] = 0
        return {k: s.float() + mask for k, s in out.items()}


# ----------------------------------------------------------------------------- 运行

def run(method, hp, seed, data, eval_split, corrupt=False, stats=None):
    dims = (data['train']['audio'].shape[1], data['train']['visual'].shape[1])
    L = Learner(method, hp, seed, dims)
    tr, ev = data['train'], data[eval_split]
    hist, best = {}, {}
    for t in range(N_TASK):
        idx = task_of(tr['y']) == t
        L.learn(tr['audio'][idx], tr['visual'][idx], tr['y'][idx])
        seen = list(range((t + 1) * PER_TASK))
        em = task_of(ev['y']) <= t
        sc = L.scores(ev['audio'][em], ev['visual'][em], seen)
        yt = ev['y'][em]
        for k, s in sc.items():
            pred = s.argmax(1)
            per = {j: float((pred[task_of(yt) == j] == yt[task_of(yt) == j]).float().mean()) for j in range(t + 1)}
            b = best.setdefault(k, {})
            for j, acc in per.items():
                b[j] = max(b.get(j, 0), acc)
            fg = float(np.mean([b[j] - per[j] for j in range(t)])) if t else 0.0
            hist.setdefault(k, []).append({'acc': float((pred == yt).float().mean()), 'forget': fg, 'per_task': per})
    res = {k: {'final_acc': h[-1]['acc'], 'avg_inc_acc': float(np.mean([x['acc'] for x in h])),
               'final_forget': h[-1]['forget'], 'history': h} for k, h in hist.items()}
    if corrupt:
        seen = list(range(N_CLS))
        conds = {}
        rng = torch.Generator().manual_seed(1234)
        for mod in ('audio', 'visual'):
            mu, sd = stats[mod]
            clean = ev[mod]
            conds[f'{mod}_missing'] = {mod: torch.zeros_like(clean)}     # 无信号：特征全零
            for k in (0.5, 1.0, 2.0):
                conds[f'{mod}_noise_{k}'] = {mod: clean + k * sd * torch.randn(clean.shape, generator=rng)}
        for c, rep in conds.items():
            a = rep.get('audio', ev['audio'])
            v = rep.get('visual', ev['visual'])
            for k, s in L.scores(a, v, seen).items():
                res[k].setdefault('corrupt', {})[c] = float((s.argmax(1) == ev['y']).float().mean())
        for k in res:
            res[k]['corrupt_mean'] = float(np.mean(list(res[k]['corrupt'].values())))
    return res


GRIDS = dict(BL.GRIDS)
# AVE 每个任务仅约 800 条训练样本（FFIR 每阶段上万条），FFIR 的 10/30 epoch 只对应几十到几百步更新，
# 梯度类方法明显欠拟合；这里把 epoch 网格扩展到 200（包含原取值），由验证集选择
EP = (10, 30, 100, 200)
GRIDS['finetune'] = [{'epochs': e, 'lr': lr} for e in EP for lr in (1e-3, 3e-4)]
GRIDS['joint'] = [{'epochs': e, 'lr': lr} for e in EP for lr in (1e-3, 3e-4)]
GRIDS['lwf'] = [{'epochs': e, 'lr': 1e-3, 'kd': kd} for e in EP for kd in (0.5, 1, 2, 5)]
GRIDS['ewc'] = [{'epochs': e, 'lr': 1e-3, 'ewc': w} for e in EP for w in (1e2, 1e3, 1e4, 1e5)]
GRIDS['er'] = [{'epochs': e, 'lr': 1e-3} for e in EP]
GRIDS['icarl'] = [{'epochs': e, 'lr': 1e-3, 'kd': kd} for e in EP for kd in (0.5, 1, 2)]
GRIDS['avcil'] = [{'epochs': e, 'lr': 1e-3, 'kd': kd, 'sim': sm} for e in EP for kd in (0.5, 1) for sm in (1, 10)]
# 类增量下各任务类别互不重叠、任务也不像 FFIR 的物种那样易于识别，是否启用任务专家同样交给验证集决定
GRIDS['joint_ridge'] = GRIDS['acil']
# 精确统计量对照（存 Gram 矩阵而非原型），同样由验证集选择
GRIDS['hail_exact'] = [dict(lam_s=ls, lam_r=30, m=20, stats='exact', experts=ex, prior_av=pa)
                       for ls, ex, pa in itertools.product((10, 100), (False, True), (0.5, 0.6))]
GRIDS['hail'] = ([dict(lam_s=ls, lam_r=lr, m=m) for ls, lr, m in itertools.product((10, 100), (10, 30, 100), (5, 20))]
                 + [dict(lam_s=ls, lam_r=30, m=m, experts=False) for ls, m in itertools.product((10, 100), (5, 20))])
# AV 头先验 π_av（Eq. 9）：类增量基准上单模态头较弱，扩展到 1.0（覆盖整个取值范围），同样由验证集选择
for k in ('hail', 'hail_exact'):
    st = {'stats': 'exact'} if k == 'hail_exact' else {}
    GRIDS[k] += [dict(lam_s=ls, lam_r=30, m=20, experts=False, prior_av=pa, **st)
                 for ls, pa in itertools.product((10, 100), (0.7, 0.8, 0.9, 0.95, 1.0))]
# 用于报告的固定变体（不参与选择）
VARIANTS = {'hail_experts': dict(lam_s=10, lam_r=100, m=20), 'hail_noexperts': dict(lam_s=10, lam_r=30, m=20, experts=False)}
METHODS = ['finetune', 'lwf', 'ewc', 'er', 'icarl', 'avcil', 'acil', 'dsal', 'mmal', 'joint', 'hail']

configure('ave')   # 默认（被其他脚本 import 时）


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--methods', nargs='*', default=METHODS)
    p.add_argument('--seeds', type=int, nargs='*', default=[42, 43, 44])
    p.add_argument('--dataset', default='ave', choices=list(DATASETS))
    args = p.parse_args()
    configure(args.dataset)
    data = torch.load(CACHE) if os.path.exists(CACHE) else build_cache()
    dims = (data['train']['audio'].shape[1], data['train']['visual'].shape[1])
    BL.C = N_CLS
    BL.FEAT_DIM = {'audio': dims[0], 'video': dims[1]}
    stats = {m: (data['train'][m].mean(0), data['train'][m].std(0)) for m in ('audio', 'visual')}
    os.makedirs(OUT, exist_ok=True)
    for method in args.methods:
        fn = f'{OUT}/{method}.json'
        if os.path.exists(fn):
            continue
        sel = []
        for hp in GRIDS[method]:
            # 统一选参标准（与 AV-SIL-FFIR 相同）：½(验证集干净平均增量 + 验证集特征损坏平均)
            r = run(method, hp, 42, data, 'val', corrupt=True, stats=stats)
            key = 'icarl_fc' if method == 'icarl' else 'hail' if method.startswith('hail') else method
            sel.append((0.5 * (r[key]['avg_inc_acc'] + r[key]['corrupt_mean']), hp))
        best_hp = max(sel, key=lambda x: x[0])[1]
        per_seed = [run(method, best_hp, s, data, 'test', corrupt=True, stats=stats) for s in args.seeds]
        summ = {}
        for k in per_seed[0]:
            summ[k] = {m: (float(np.mean([ps[k][m] for ps in per_seed])), float(np.std([ps[k][m] for ps in per_seed])))
                       for m in ('final_acc', 'avg_inc_acc', 'final_forget', 'corrupt_mean')}
            summ[k]['corrupt'] = {c: float(np.mean([ps[k]['corrupt'][c] for ps in per_seed])) for c in per_seed[0][k]['corrupt']}
        json.dump({'best_hp': best_hp, 'val_grid': sel, 'summary': summ}, open(fn, 'w'), indent=2)
        for k, s in summ.items():
            print(f'{k:10s} hp={best_hp} | final {100 * s["final_acc"][0]:.2f}±{100 * s["final_acc"][1]:.2f} '
                  f'avg_inc {100 * s["avg_inc_acc"][0]:.2f} forget {100 * s["final_forget"][0]:.2f} '
                  f'corrupt {100 * s["corrupt_mean"][0]:.2f}', flush=True)


if __name__ == '__main__':
    main()
