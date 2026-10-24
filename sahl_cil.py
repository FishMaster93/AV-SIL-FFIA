"""
HAIL-FFIR v2：物种感知的层级闭式解增量学习（Species-Aware Hierarchical Analytic Learning）

核心问题：同样的音视频表现，在不同物种上对应的投喂强度不同（P(y|x, species) 随物种变化），
一个所有物种共用的强度头会把这些相互冲突的映射"平均掉"。

结构（三个头 h ∈ {av, a, v}，结构相同；av 为主，a/v 为模态受损时的后备）：
  共享强度头   W_s^h：所有已见物种的 A/B 精确累加后求解（通用强度知识；与顺序无关）
  物种残差专家 ΔW_k^h：以共享头为先验，只拟合物种 k 偏离通用规律的部分
               ΔW_k = argmin ||X_k (W_s + Δ) - Y_k||^2 + λ_r ||Δ||^2 = (A_k + λ_r I)^{-1} (B_k - A_k W_s)
               学完物种 k 后即冻结；每个物种只存 ΔW_k（D x 4），不存 A_k
  物种路由     r(x)：推理时物种未知。analytic = 与 av 头共用 A 的闭式解物种分类头（只多存 B_sp）；
               proto = 物种 x 强度原型库最近邻；oracle = 真实物种（仅分析用）
  模态可靠性   u_h(x) = φ_h^T A_h^{-1} φ_h / ref_h（岭回归后验方差，按训练数据典型值归一化）；
               w_h ∝ π_h · u_h^{-κ}，干净时以 av 头为主，某个模态受损时自动转向可靠的单模态头
  预测         ŷ = Σ_h w_h(x) · [W_s^h φ_h + Σ_k r_k(x) ΔW_k^h φ_h]

目标为 one-hot（论文 Eq. 4），预测取 argmax。全程无梯度、不存原始样本。
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analytic_cil import FEAT_ROOT, RandomBuffer, load_phase
from datasets.fish_av_incremental import INCREMENTAL_PHASES
from hail_cil import kmeans

C = 4
A_RAW, V_RAW = 'audio_cnn14mv2', 'video_s3d_ft'


def aug(x):
    return torch.cat([x, torch.ones(len(x), 1, device=x.device)], dim=1)


class Head:
    """一个强度头：共享闭式解 + 每物种残差专家。"""

    def __init__(self, in_dims, layouts, D, lam_s, lam_r, device, seed):
        self.bufs = [RandomBuffer(d, D, seed + i, device, lay) for i, (d, lay) in enumerate(zip(in_dims, layouts))]
        self.dim = D * len(in_dims) + 1
        self.A = torch.zeros(self.dim, self.dim, device=device)
        self.B = torch.zeros(self.dim, C, device=device)
        self.lam_s, self.lam_r, self.device = lam_s, lam_r, device
        self.Ws = self.A_inv = None
        self.dW = []            # 每个物种的残差 ΔW_k
        self.Wsnap = []         # 学完物种 k 时的 W_s（snapshot 模式用）
        self.ref = []           # 每个物种训练数据的后验方差中位数

    def phi(self, feats):
        return aug(torch.cat([b(f.to(self.device)) for b, f in zip(self.bufs, feats)], dim=1))

    def learn(self, feats, y, chunk=8192):
        Y = F.one_hot(y.to(self.device), C).float()
        X = self.phi(feats)
        Ak = torch.zeros_like(self.A)
        Bk = torch.zeros_like(self.B)
        for i in range(0, len(X), chunk):
            Ak += X[i:i + chunk].T @ X[i:i + chunk]
            Bk += X[i:i + chunk].T @ Y[i:i + chunk]
        self.A += Ak
        self.B += Bk
        eye = torch.eye(self.dim, device=self.device)
        self.A_inv = torch.linalg.inv(self.A + self.lam_s * eye)
        self.Ws = self.A_inv @ self.B
        # 物种残差：以当前共享头为先验
        self.dW.append(torch.linalg.solve(Ak + self.lam_r * eye, Bk - Ak @ self.Ws))
        self.Wsnap.append(self.Ws.clone())
        self.ref.append(self.variance_aug(X[:4096]).median())
        del Ak, Bk, X

    def variance_aug(self, Xa):
        return ((Xa @ self.A_inv) * Xa).sum(dim=1)

    def scores(self, Xa, r, mode):
        """r: (N, n_species) 路由权重。"""
        if mode == 'shared':
            return Xa @ self.Ws
        if mode == 'snapshot':
            return sum(r[:, k:k + 1] * (Xa @ (self.Wsnap[k] + self.dW[k])) for k in range(len(self.dW)))
        out = Xa @ self.Ws
        for k in range(len(self.dW)):
            out = out + r[:, k:k + 1] * (Xa @ self.dW[k])
        return out


class SAHL:
    def __init__(self, heads_cfg, D, lam_s, lam_r, device, router='analytic', tau=10.0, m=5,
                 kappa=2.0, prior_av=0.8, reliability=True, residual_mode='residual', seed=42):
        self.heads = {}
        for i, (h, (names, in_dims)) in enumerate(heads_cfg.items()):
            layouts = ['in_out' if n == A_RAW else 'linear' for n in names]
            self.heads[h] = Head(in_dims, layouts, D, lam_s, lam_r, device, seed + 10 * i)
        self.names = {h: names for h, (names, _) in heads_cfg.items()}
        self.router, self.tau, self.m, self.kappa = router, tau, m, kappa
        self.prior = {h: (prior_av if h == 'av' else (1 - prior_av) / max(1, len(heads_cfg) - 1))
                      for h in heads_cfg}
        self.reliability, self.mode, self.device = reliability, residual_mode, device
        self.reliability_route = True
        self.B_sp = {}          # 闭式解物种头：每个强度头各一个，共用该头的 A，只存 B_sp
        self.protos, self.proto_sp = [], []
        self.n_species = 0

    def learn(self, feat_dict, y):
        """feat_dict: {feature_name: tensor}"""
        for h, head in self.heads.items():
            head.learn([feat_dict[n] for n in self.names[h]], y)
        k = self.n_species
        self.n_species += 1
        # 闭式解物种头（每个强度头一个，共用该头的 A）：B_sp 新增一列 X_k^T 1
        for h, head in self.heads.items():
            col = head.phi([feat_dict[n] for n in self.names[h]]).sum(0, keepdim=True).T
            self.B_sp[h] = col if h not in self.B_sp else torch.cat([self.B_sp[h], col], dim=1)
        # 原型库：物种 x 强度，归一化原始特征空间
        z = self._proto_space(feat_dict)
        yy = y.to(self.device)
        for c in range(C):
            if (yy == c).any():
                ctr, _ = kmeans(z[yy == c], self.m, seed=c)
                self.protos.append(ctr)
                self.proto_sp += [k] * len(ctr)

    def _proto_space(self, feat_dict):
        return torch.cat([F.normalize(feat_dict[n].to(self.device).float(), dim=1) for n in (A_RAW, V_RAW)], dim=1)

    def route(self, feat_dict, Xs, w, species_true=None):
        """w: (N, n_heads) 模态可靠性权重；analytic 路由按可靠性合并各头的物种判断。"""
        n = len(Xs['av'])
        if self.router == 'oracle':
            return F.one_hot(species_true.to(self.device), self.n_species).float()
        if self.router == 'none':
            return torch.zeros(n, self.n_species, device=self.device)
        if self.router == 'analytic':
            r = 0
            for j, (h, head) in enumerate(self.heads.items()):
                wh = w[:, j:j + 1] if self.reliability_route else (1.0 if h == 'av' else 0.0)
                r = r + wh * torch.softmax(self.tau * (Xs[h] @ (head.A_inv @ self.B_sp[h])), dim=1)
            return r
        else:
            sim = self._proto_space(feat_dict) @ torch.cat(self.protos).T
            sp = torch.tensor(self.proto_sp, device=self.device)
            s = torch.stack([sim[:, sp == k].max(1).values for k in range(self.n_species)], dim=1)
        return torch.softmax(self.tau * s, dim=1)

    @torch.no_grad()
    def predict(self, feat_dict, species_true=None, chunk=2048):
        n = len(next(iter(feat_dict.values())))
        preds, rpred, waux = [], [], []
        for i in range(0, n, chunk):
            fd = {k: v[i:i + chunk] for k, v in feat_dict.items()}
            st = species_true[i:i + chunk] if species_true is not None else None
            Xs = {h: head.phi([fd[nm] for nm in self.names[h]]) for h, head in self.heads.items()}
            if self.reliability and len(self.heads) > 1:
                logw = []
                for h, head in self.heads.items():
                    u = head.variance_aug(Xs[h]) / torch.stack(head.ref).mean()
                    logw.append(np.log(self.prior[h]) - self.kappa * torch.log(u.clamp_min(1e-8)))
                w = torch.softmax(torch.stack(logw, dim=1), dim=1)          # (N, n_heads)
            else:
                w = torch.zeros(len(Xs["av"]), len(self.heads), device=self.device)
                w[:, list(self.heads).index('av')] = 1
            r = self.route(fd, Xs, w, st)
            scores = {h: head.scores(Xs[h], r, self.mode) for h, head in self.heads.items()}
            total = sum(w[:, j:j + 1] * scores[h] for j, h in enumerate(self.heads))
            preds.append(total.argmax(1).cpu())
            rpred.append(r.argmax(1).cpu() if r.sum() > 0 else torch.full((len(r),), -1))
            waux.append(w.cpu())
        return torch.cat(preds), torch.cat(rpred), torch.cat(waux)


# ----------------------------------------------------------------------------- 数据

def load_aligned(names, phase, split, corrupt_dir=None, cond=None):
    """返回 ({name: feat}, labels, species_idx_placeholder, paths)，按第一个特征集的路径对齐取交集。"""
    if cond is None or cond == 'clean':
        ds = {n: load_phase(n, phase)[split] for n in names}
    else:
        d = torch.load(os.path.join(FEAT_ROOT, corrupt_dir, cond, f'phase{phase}.pt'), weights_only=False)
        ds = {n: d[n] for n in names}
    common = set.intersection(*[set(d['paths']) for d in ds.values()])
    order = [p for p in ds[names[0]]['paths'] if p in common]
    out = {}
    for n, d in ds.items():
        idx = {p: i for i, p in enumerate(d['paths'])}
        out[n] = d['feat'][torch.tensor([idx[p] for p in order])].float()
    d0 = ds[names[0]]
    idx0 = {p: i for i, p in enumerate(d0['paths'])}
    labels = d0['labels'][torch.tensor([idx0[p] for p in order])]
    return out, labels, order


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--av_feats', nargs='*', default=[A_RAW, V_RAW], help='av 头的输入特征集')
    p.add_argument('--no_unimodal', action='store_true', help='只用 av 头（不加单模态后备头）')
    p.add_argument('--dim', type=int, default=8192)
    p.add_argument('--lam_s', type=float, default=100.0)
    p.add_argument('--lam_r', type=float, default=100.0)
    p.add_argument('--router', choices=['analytic', 'proto', 'oracle', 'none'], default='analytic')
    p.add_argument('--tau', type=float, default=10.0)
    p.add_argument('--m', type=int, default=5)
    p.add_argument('--kappa', type=float, default=2.0)
    p.add_argument('--prior_av', type=float, default=0.8)
    p.add_argument('--no_reliability', action='store_true')
    p.add_argument('--av_route_only', action='store_true', help='路由只用 av 头（不按模态可靠性合并）')
    p.add_argument('--residual_mode', choices=['residual', 'snapshot', 'shared'], default='residual')
    p.add_argument('--eval_split', choices=['val', 'test'], default='val')
    p.add_argument('--corrupt', action='store_true', help='最后在损坏的评估集上测')
    p.add_argument('--order', type=int, nargs='*', default=None)
    p.add_argument('--out', default=None)
    args = p.parse_args()
    device = 'cuda'

    heads_cfg = {'av': args.av_feats}
    if not args.no_unimodal:
        heads_cfg.update({'a': [A_RAW], 'v': [V_RAW]})
    all_names = sorted({n for v in heads_cfg.values() for n in v} | {A_RAW, V_RAW})
    order = [0] + (args.order if args.order else list(range(1, len(INCREMENTAL_PHASES))))

    train, evals = {}, {}
    for ph in order:
        train[ph] = load_aligned(all_names, ph, 'train')
        evals[ph] = load_aligned(all_names, ph, args.eval_split)
    dims = {n: train[0][0][n].shape[1] for n in all_names}
    model = SAHL({h: (ns, [dims[n] for n in ns]) for h, ns in heads_cfg.items()}, args.dim, args.lam_s,
                 args.lam_r, device, router=args.router, tau=args.tau, m=args.m, kappa=args.kappa,
                 prior_av=args.prior_av, reliability=not args.no_reliability, residual_mode=args.residual_mode)
    model.reliability_route = not args.av_route_only
    print(f'heads={heads_cfg} router={args.router} mode={args.residual_mode} lam_s={args.lam_s} '
          f'lam_r={args.lam_r} reliability={not args.no_reliability} kappa={args.kappa} prior_av={args.prior_av} '
          f'eval={args.eval_split}')

    sp_index = {ph: i for i, ph in enumerate(order)}   # 路由的物种编号 = 学习顺序

    def eval_phase(ph, cond=None):
        if cond is None:
            fd, y, _ = evals[ph]
        else:
            fd, y, _ = load_aligned(all_names, ph, args.eval_split,
                                    'corrupt' if args.eval_split == 'test' else 'corrupt_val', cond)
        st = torch.full((len(y),), sp_index[ph]) if sp_index[ph] < model.n_species else None
        if st is None and args.router == 'oracle':
            return None
        pred, rp, w = model.predict(fd, st)
        route_acc = float((rp == sp_index[ph]).float().mean()) if sp_index[ph] < model.n_species else None
        return int((pred == y).sum()), len(y), route_acc, w.mean(0)

    history, best = [], {}
    for step, ph in enumerate(order):
        zs = eval_phase(ph) if step > 0 and args.router != 'oracle' else None
        fd, y, _ = train[ph]
        model.learn(fd, y)
        per = {q: eval_phase(q) for q in order[:step + 1]}
        acc_seen = sum(v[0] for v in per.values()) / sum(v[1] for v in per.values())
        accs = {q: v[0] / v[1] for q, v in per.items()}
        for q, a in accs.items():
            best[q] = max(best.get(q, 0), a)
        avg_f = float(np.mean([best[q] - accs[q] for q in order[:step]])) if step else 0.0
        racc = np.mean([v[2] for v in per.values()])
        history.append({'step': step, 'phase': ph, 'acc_seen': acc_seen, 'avg_forget': avg_f,
                        'zero_shot': zs[0] / zs[1] if zs else None, 'route_acc': float(racc), 'per_phase': accs})
        print(f'step {step} phase {ph} {INCREMENTAL_PHASES[ph]!s:32s} acc_seen {acc_seen:.4f} | avg_forget {avg_f:.4f} '
              f'| route_acc {racc:.3f} | ' + ' '.join(f'{q}:{a:.3f}' for q, a in accs.items()), flush=True)

    summary = {'final_acc': history[-1]['acc_seen'], 'avg_inc_acc': float(np.mean([h['acc_seen'] for h in history])),
               'final_avg_forget': history[-1]['avg_forget']}
    w_clean = torch.stack([eval_phase(q)[3] for q in order]).mean(0)
    print('SUMMARY ' + ' '.join(f'{k}={v:.4f}' for k, v in summary.items())
          + ' | 干净数据平均头权重 ' + ' '.join(f'{h}:{w:.2f}' for h, w in zip(model.heads, w_clean.tolist())))

    if args.corrupt:
        cdir = os.path.join(FEAT_ROOT, 'corrupt' if args.eval_split == 'test' else 'corrupt_val')
        conds = sorted(os.listdir(cdir))
        summary['corrupt'] = {}
        for c in conds:
            res = [eval_phase(q, c) for q in order]
            a = sum(r[0] for r in res) / sum(r[1] for r in res)
            summary['corrupt'][c] = a
        summary['corrupt_mean'] = float(np.mean(list(summary['corrupt'].values())))
        print('CORRUPT mean={:.4f} '.format(summary['corrupt_mean'])
              + ' '.join(f'{c}:{a:.3f}' for c, a in summary['corrupt'].items()))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump({'args': vars(args), 'summary': summary, 'history': history}, f, indent=2)


if __name__ == '__main__':
    main()
