"""
HAIL-FFIR（论文最终方法）：基于缓存的冻结编码器特征做物种增量学习。

  头 h ∈ {av, a, v}：输入为固定随机投影 + ReLU 扩展后的特征（论文 Eq. 1）
  (1) 原型增强的闭式解训练（Eq. 2-4）
        每个物种、每个强度存 m 个高斯原型 (μ, σ, n)，位于编码器特征空间 z=[f^a; f^v]；
        学第 t 个物种时，旧物种的 A/B 由原型采样（每个原型 s 个伪特征，权重 n/s）重建，
        当前物种用真实数据；共享头 W_s = (A + λ_s I)^{-1} B
  (2) 物种条件闭式专家（Eq. 5-7）
        W_t = W_s(t) + (X_t^T X_t + λ_r I)^{-1} (X_t^T Y_t - X_t^T X_t W_s(t))，学完冻结；
        物种分类头与强度头共用 A
  (3) 可靠性感知的动态模态平衡（Eq. 8-11）
        u_h(x) = φ^T (A + λ_s I)^{-1} φ / ū_h，w_h ∝ π_h u_h^{-κ}，r_k = Σ_h w_h p_h(k|x)，
        ŷ = argmax Σ_h w_h Σ_k r_k φ_h^T W_k^h

  部署统计量（--stats）：
    proto ：每个阶段结束后只用所有物种的原型重建 A、B_sp（推理用的 A^{-1}、物种分类器、ū 都由它算），
            设备上持久保存的只有原型 + 专家 + 物种分类器权重（论文设定）
    exact ：精确累加 A/B 并保存（对照：存储大、精度上界）
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analytic_cil import FEAT_ROOT, RandomBuffer
from datasets.fish_av_incremental import INCREMENTAL_PHASES
from hail_cil import kmeans
from sahl_cil import A_RAW, V_RAW, load_aligned

C = 4
HEADS = {'av': [A_RAW, V_RAW], 'a': [A_RAW], 'v': [V_RAW]}
LAYOUT = {A_RAW: 'in_out', V_RAW: 'linear'}   # 与 AquaLytic / 前面实验一致的随机投影写法


def aug(x):
    return torch.cat([x, torch.ones(len(x), 1, device=x.device)], dim=1)


class HAILFFIR:
    def __init__(self, D=8192, lam_s=100.0, lam_r=30.0, tau=10.0, kappa=2.0, prior_av=0.6,
                 stats='proto', m=5, s=20, s_deploy=20, seed=42, device='cuda', rank=0, modality='av',
                 experts=True, reliability=True, route='rel', proto_mode='gauss', feat_dims=None,
                 expert_mode='snapshot', proto_scale='none'):
        """消融开关：experts=False 用共享头预测；reliability=False 用固定先验权重；route='av' 只用 AV 头判断物种；
        proto_mode：gauss（高斯原型，rank>0 时为低秩+对角）、centroid（只存中心，按样本数加权）、
        centroid_unw（只存中心、不加权，即原稿做法）。modality='av_only' 只有 AV 头。"""
        self.use_experts, self.use_rel, self.route, self.proto_mode = experts, reliability, route, proto_mode
        # snapshot：专家 = 学该物种时的共享头 + 修正量，整体冻结（物种增量默认）；
        # residual：预测时用最新共享头 + 各专家的修正量 ΔW_k（类增量时新类别只在最新共享头里）
        self.expert_mode = expert_mode
        self.deltas = {}
        self.shared = {}
        self.dev, self.lam_s, self.lam_r, self.tau, self.kappa = device, lam_s, lam_r, tau, kappa
        self.stats, self.m, self.s, self.s_deploy, self.seed = stats, m, s, s_deploy, seed
        self.rank = rank                         # 每个原型额外保存的主方向数（低秩协方差），0 = 只存对角方差
        # proto_scale='modality'：建原型（k-means、簇内 PCA）前把各模态分别除以其整体标准差，使音频、视频在
        # 分簇和主方向中分量相当；存储时换回原尺度（逐维线性变换，高斯参数精确换算），采样过程不变
        self.proto_scale = proto_scale
        # 单模态时只有该模态的头（无模态平衡），原型也只存该模态的编码器特征
        self.heads = {'av': HEADS, 'av_only': {'av': HEADS['av']}, 'audio': {'a': HEADS['a']},
                      'video': {'v': HEADS['v']}}[modality]
        self.feats = {'av': [A_RAW, V_RAW], 'av_only': [A_RAW, V_RAW], 'audio': [A_RAW], 'video': [V_RAW]}[modality]
        self.prior = ({'av': prior_av, 'a': (1 - prior_av) / 2, 'v': (1 - prior_av) / 2} if modality == 'av'
                      else {h: 1.0 for h in self.heads})
        self.fdim = feat_dims or {A_RAW: 1024, V_RAW: 1024}     # 各模态编码器特征维度（其他数据集可不同）
        self.bufs = {}
        for h, names in self.heads.items():
            i = list(HEADS).index(h)      # 与音视频版本使用相同的随机投影种子
            self.bufs[h] = [RandomBuffer(self.fdim[n], D, seed + 10 * i + j, device, LAYOUT[n]) for j, n in enumerate(names)]
        self.dim = {h: D * len(n) + 1 for h, n in self.heads.items()}
        self.protos = []                         # 每个物种: list of (mu, std, n, class, L)；L 为 (2d, r) 低秩因子或 None
        self.experts = {h: [] for h in self.heads}    # 冻结的物种专家
        self.exact = {h: [torch.zeros(self.dim[h], self.dim[h], device=device),
                          torch.zeros(self.dim[h], C, device=device),
                          torch.zeros(self.dim[h], 0, device=device)] for h in self.heads} if stats == 'exact' else None
        self.deploy = {}                         # 推理用：A^{-1}、物种分类器权重、ū

    # ---- 特征 ----
    def phi(self, h, z):
        """z: (N, 2048) = [f^a; f^v]"""
        offs = np.cumsum([0] + [self.fdim[n] for n in self.feats])
        parts = {n: z[:, offs[i]:offs[i + 1]] for i, n in enumerate(self.feats)}
        return aug(torch.cat([b(parts[n]) for b, n in zip(self.bufs[h], self.heads[h])], dim=1))

    def _pseudo(self, species_ids, s):
        """从指定物种的高斯原型采样伪特征，返回 (z, weight, class, species)。采样用固定种子，可复现。"""
        zs, ws, cs, ks = [], [], [], []
        for k in species_ids:
            g = torch.Generator(device=self.dev).manual_seed(self.seed * 1000 + k)
            for mu, std, n, c, L in self.protos[k]:
                z = mu[None] + std[None] * torch.randn(s, mu.numel(), device=self.dev, generator=g)
                if L is not None:   # 低秩协方差方向：z += L ε_r
                    z = z + torch.randn(s, L.shape[1], device=self.dev, generator=g) @ L.T
                zs.append(z)
                ws.append(torch.full((s,), n / s, device=self.dev))
                cs.append(torch.full((s,), c, device=self.dev, dtype=torch.long))
                ks.append(torch.full((s,), k, device=self.dev, dtype=torch.long))
        if not zs:
            return None
        return torch.cat(zs), torch.cat(ws), torch.cat(cs), torch.cat(ks)

    def _stats(self, h, z, w, y, sp, n_species, chunk=8192):
        """加权统计量 A = Σ w φφ^T, B = Σ w φ e_y^T, B_sp = Σ w φ e_sp^T"""
        D = self.dim[h]
        A = torch.zeros(D, D, device=self.dev)
        B = torch.zeros(D, C, device=self.dev)
        Bsp = torch.zeros(D, n_species, device=self.dev)
        for i in range(0, len(z), chunk):
            X = self.phi(h, z[i:i + chunk])
            Xw = X * w[i:i + chunk, None]
            A += Xw.T @ X
            B += Xw.T @ F.one_hot(y[i:i + chunk], C).float()
            Bsp += Xw.T @ F.one_hot(sp[i:i + chunk], n_species).float()
        return A, B, Bsp

    # ---- 学习一个新物种 ----
    def learn(self, fd, y):
        t = len(self.protos)
        z = torch.cat([fd[n] for n in self.feats], dim=1).to(self.dev).float()
        y = y.to(self.dev)
        ones = torch.ones(len(z), device=self.dev)
        spk = torch.full((len(z),), t, device=self.dev, dtype=torch.long)
        old = self._pseudo(range(t), self.s)
        for h in self.heads:
            # 当前物种真实数据的统计量
            At, Bt, Bspt = self._stats(h, z, ones, y, spk, t + 1)
            if self.stats == 'exact':
                ex = self.exact[h]
                ex[0] += At
                ex[1] += Bt
                ex[2] = torch.cat([ex[2], Bspt[:, t:t + 1]], dim=1)
                A, B = ex[0], ex[1]
            else:   # 旧物种由原型重建 + 当前物种真实数据（Eq. 3）
                A, B = At.clone(), Bt.clone()
                if old is not None:
                    Ao, Bo, _ = self._stats(h, *old, t + 1)
                    A += Ao
                    B += Bo
            eye = torch.eye(self.dim[h], device=self.dev)
            Ws = torch.linalg.solve(A + self.lam_s * eye, B)                           # 共享头 (Eq. 2)
            delta = torch.linalg.solve(At + self.lam_r * eye, Bt - At @ Ws)
            self.experts[h].append(Ws + delta)                                         # 专家 (Eq. 6)
            self.deltas.setdefault(h, []).append(delta)
            self.shared[h] = Ws
            del A, B, At, Bt, Bspt
        # 当前物种的高斯原型（Eq. 3 的记忆）
        cells = []
        sc = torch.ones(z.shape[1], device=self.dev)
        if self.proto_scale == 'modality':
            offs = np.cumsum([0] + [self.fdim[n] for n in self.feats])
            for i in range(len(self.feats)):
                sc[offs[i]:offs[i + 1]] = z[:, offs[i]:offs[i + 1]].std().clamp_min(1e-6)
        zs = z / sc
        for c in range(C):
            zc = zs[y == c]
            if len(zc) == 0:          # 类增量设定下当前任务不含该类
                continue
            ctr, asg = kmeans(zc, self.m, seed=c)
            for j in range(len(ctr)):
                mem = zc[asg == j]
                L = None
                if self.proto_mode != 'gauss':        # 只存中心（消融 / 原稿做法）
                    cells.append((ctr[j] * sc, torch.zeros_like(ctr[j]),
                                  float(len(mem)) if self.proto_mode == 'centroid' else 1.0, c, None))
                    continue
                if len(mem) > 1:
                    var = mem.var(0)
                    if self.rank and len(mem) > self.rank + 1:
                        # 簇内 PCA：前 r 个主方向存成低秩因子，其余方差留在对角项里
                        _, S_, Vh = torch.linalg.svd(mem - mem.mean(0), full_matrices=False)
                        L = Vh[:self.rank].T * (S_[:self.rank] / np.sqrt(len(mem) - 1))
                        var = (var - (L ** 2).sum(1)).clamp_min(0)
                    std = var.sqrt()
                else:
                    std = torch.zeros_like(ctr[j])
                cells.append((ctr[j] * sc, std * sc, float(len(mem)), c, None if L is None else L * sc[:, None]))
        self.protos.append(cells)
        self._build_deploy(real=(z, y))

    def reanchor(self, data):
        """联合训练对照（Joint training, HAIL-FFIR 模型）：所有专家都以当前（用全部已见数据求得的）共享头为锚点重新求解，
        data 为已学全部物种的 [(fd, y), ...]。需配合 stats='exact' 使用，此时共享头与可靠性即为全部数据上的闭式解。"""
        for h in self.heads:
            Ws, eye = self.shared[h], torch.eye(self.dim[h], device=self.dev)
            for k, (fd, y) in enumerate(data):
                z = torch.cat([fd[n] for n in self.feats], dim=1).to(self.dev).float()
                y = y.to(self.dev)
                At, Bt, _ = self._stats(h, z, torch.ones(len(z), device=self.dev), y,
                                        torch.zeros(len(z), device=self.dev, dtype=torch.long), 1)
                delta = torch.linalg.solve(At + self.lam_r * eye, Bt - At @ Ws)
                self.experts[h][k], self.deltas[h][k] = Ws + delta, delta
                del At, Bt

    def _build_deploy(self, real):
        """阶段结束：构建推理用的 A^{-1}、物种分类器和可靠性参照 ū（proto 模式只用原型）。"""
        S = len(self.protos)
        if self.stats == 'proto':
            pz, pw, pc, pk = self._pseudo(range(S), self.s_deploy)
        for h in self.heads:
            if self.stats == 'exact':
                A, Bsp = self.exact[h][0], self.exact[h][2]
            else:
                A, _, Bsp = self._stats(h, pz, pw, pc, pk, S)
            Ainv = torch.linalg.inv(A + self.lam_s * torch.eye(self.dim[h], device=self.dev))
            Xr = self.phi(h, real[0][:4096])
            ref = ((Xr @ Ainv) * Xr).sum(1).median()
            refs = self.deploy.get(h, {}).get('refs', []) + [ref]
            self.deploy[h] = {'Ainv': Ainv, 'Wsp': Ainv @ Bsp, 'refs': refs}
            del A

    # ---- 推理 ----
    @torch.no_grad()
    def predict(self, fd, chunk=2048, return_aux=False, return_scores=False):
        """return_scores=True 时返回最终得分矩阵 (N, 4)（校准分析用）。"""
        z_all = torch.cat([fd[n] for n in self.feats], dim=1).float()
        preds, routes, ws, scores = [], [], [], []
        hs = list(self.heads)
        for i in range(0, len(z_all), chunk):
            z = z_all[i:i + chunk].to(self.dev)
            X = {h: self.phi(h, z) for h in hs}
            lw = []
            for h in hs:
                d = self.deploy[h]
                u = ((X[h] @ d['Ainv']) * X[h]).sum(1) / torch.stack(d['refs']).mean()        # Eq. 8
                lw.append(np.log(self.prior[h]) - self.kappa * torch.log(u.clamp_min(1e-8)))
            w = torch.softmax(torch.stack(lw, 1), 1)                                           # Eq. 9
            if not self.use_rel:      # 消融：固定先验权重
                w = torch.tensor([self.prior[h] for h in hs], device=self.dev)[None].expand(len(z), -1)
            if self.route == 'av':    # 消融：只用 AV 头判断物种
                r = torch.softmax(self.tau * (X['av'] @ self.deploy['av']['Wsp']), 1)
            else:
                r = sum(w[:, j:j + 1] * torch.softmax(self.tau * (X[h] @ self.deploy[h]['Wsp']), 1)
                        for j, h in enumerate(hs))                                             # Eq. 10
            if self.use_experts and self.expert_mode == 'residual':
                score = sum(w[:, j:j + 1] * (X[h] @ self.shared[h] + sum(r[:, k:k + 1] * (X[h] @ dl)
                                                                          for k, dl in enumerate(self.deltas[h])))
                            for j, h in enumerate(hs))
            elif self.use_experts:
                score = sum(w[:, j:j + 1] * sum(r[:, k:k + 1] * (X[h] @ E) for k, E in enumerate(self.experts[h]))
                            for j, h in enumerate(hs))                                         # Eq. 11
            else:                     # 消融：不用物种专家，只用共享头
                score = sum(w[:, j:j + 1] * (X[h] @ self.shared[h]) for j, h in enumerate(hs))
            preds.append(score.argmax(1).cpu())
            scores.append(score.float().cpu())
            routes.append(r.argmax(1).cpu())
            ws.append(w.cpu())
        if return_scores:
            return torch.cat(scores)
        out = torch.cat(preds)
        return (out, torch.cat(routes), torch.cat(ws)) if return_aux else out

    def storage_bytes(self):
        """持久保存的字节数（float32）。proto：原型 + 专家 + 物种分类器；exact 另加 A、B。"""
        n_proto = sum(len(c) for c in self.protos)
        zdim = sum(self.fdim[n] for n in self.feats)
        # 每个原型：中心 + 样本数 + 类别；高斯原型另存逐维方差（只存中心的消融变体不存）
        per = zdim + 2 if self.proto_mode != 'gauss' else 2 * zdim + 2
        b = n_proto * per * 4
        b += sum(L.numel() for cells in self.protos for *_, L in cells if L is not None) * 4
        b += sum(sum(E.numel() for E in self.experts[h]) for h in self.heads) * 4
        b += sum(self.deploy[h]['Wsp'].numel() for h in self.heads) * 4
        if self.stats == 'exact':
            b += sum(self.exact[h][0].numel() + self.exact[h][1].numel() for h in self.heads) * 4
        return b


# ----------------------------------------------------------------------------- 实验

def evaluate(model, order, split, cond=None):
    names = [A_RAW, V_RAW]
    cdir = 'corrupt' if split == 'test' else 'corrupt_val'
    res = {}
    for ph in order:
        fd, y, _ = load_aligned(names, ph, split) if cond is None else load_aligned(names, ph, split, cdir, cond)
        pred = model.predict(fd)
        res[ph] = (int((pred == y).sum()), len(y))
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stats', choices=['proto', 'exact'], default='proto')
    p.add_argument('--m', type=int, default=5)
    p.add_argument('--s', type=int, default=20)
    p.add_argument('--s_deploy', type=int, default=20)
    p.add_argument('--modality', choices=['av', 'audio', 'video'], default='av')
    p.add_argument('--rank', type=int, default=0, help='每个原型保存的主方向数（低秩协方差）')
    p.add_argument('--dim', type=int, default=8192)
    p.add_argument('--lam_s', type=float, default=100.0)
    p.add_argument('--lam_r', type=float, default=30.0)
    p.add_argument('--kappa', type=float, default=2.0)
    p.add_argument('--prior_av', type=float, default=0.6)
    p.add_argument('--tau', type=float, default=10.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--order', type=int, nargs='*', default=None)
    p.add_argument('--eval_split', choices=['val', 'test'], default='val')
    p.add_argument('--corrupt', action='store_true')
    p.add_argument('--out', default=None)
    args = p.parse_args()

    order = [0] + (args.order if args.order else list(range(1, len(INCREMENTAL_PHASES))))
    model = HAILFFIR(args.dim, args.lam_s, args.lam_r, args.tau, args.kappa, args.prior_av, args.stats,
                     args.m, args.s, args.s_deploy, args.seed, rank=args.rank,
                     modality=args.modality)
    t0 = time.time()
    history, best = [], {}
    learn_time = 0.0
    for step, ph in enumerate(order):
        fd, y, _ = load_aligned([A_RAW, V_RAW], ph, 'train')
        ts = time.time()
        model.learn(fd, y)
        torch.cuda.synchronize()
        learn_time += time.time() - ts
        res = evaluate(model, order[:step + 1], args.eval_split)
        accs = {q: c / n for q, (c, n) in res.items()}
        acc = sum(c for c, _ in res.values()) / sum(n for _, n in res.values())
        for q, a in accs.items():
            best[q] = max(best.get(q, 0), a)
        fg = float(np.mean([best[q] - accs[q] for q in order[:step]])) if step else 0.0
        history.append({'step': step, 'phase': ph, 'acc_seen': acc, 'avg_forget': fg, 'per_phase': accs})
        print(f'step {step} phase {ph} {INCREMENTAL_PHASES[ph]!s:32s} acc_seen {acc:.4f} avg_forget {fg:.4f} | '
              + ' '.join(f'{q}:{a:.3f}' for q, a in accs.items()), flush=True)
    summary = {'final_acc': history[-1]['acc_seen'], 'avg_inc_acc': float(np.mean([h['acc_seen'] for h in history])),
               'final_avg_forget': history[-1]['avg_forget'], 'storage_MB': model.storage_bytes() / 1e6,
               'learn_time_s': learn_time}
    if args.corrupt:
        cdir = os.path.join(FEAT_ROOT, 'corrupt' if args.eval_split == 'test' else 'corrupt_val')
        summary['corrupt'] = {}
        for c in sorted(os.listdir(cdir)):
            r = evaluate(model, order, args.eval_split, c)
            summary['corrupt'][c] = sum(a for a, _ in r.values()) / sum(n for _, n in r.values())
        summary['corrupt_mean'] = float(np.mean(list(summary['corrupt'].values())))
    print('SUMMARY ' + ' '.join(f'{k}={v:.4f}' for k, v in summary.items() if not isinstance(v, dict))
          + f' | total {time.time() - t0:.0f}s')
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump({'args': vars(args), 'summary': summary, 'history': history}, f, indent=2)


if __name__ == '__main__':
    main()
