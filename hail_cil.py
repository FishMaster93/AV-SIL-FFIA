"""
HAIL-FFIR 的层级闭式解增量学习（论文 Sec. III-B2 ~ III-C），输入为 train_phase0_fusion.py
导出的冻结融合特征 f^av / f'^v / f'^a。两个版本共用一套代码：

--version original （尽量照原稿实现）
  通用层   Eq. 9-10: 只用当前物种数据 + λp·P^av（m=5 个/类的通用原型）解岭回归，不累加历史统计量；
           λp = max(0.2, cos(当前物种特征均值, 原型均值))
  原型     Eq. 7: Phase 0 在 F^av 上 k-means 初始化；Eq. 11: 每个阶段后 EMA 更新（α=0.7），
           新特征按最近的旧原型分组求均值
  物种层   Eq. 5: 每个物种单独的音频/视频岭回归 W^a_k, W^v_k
  预测     Eq. 12-14: 物种层用当前阶段 k 的物种头（推理时没有物种 ID，原文没说怎么选，取字面意思）；
           β_{k,i} 依赖待预测的强度 i，无法计算，退化为 β=0.5；γ_k 按 Eq. 14 随阶段线性下降

--version revised （修订版，保留层级/原型/动态模态平衡结构，修正不可实现或有缺陷之处）
  通用层   A/B 精确累加（等价于在所有已见数据上联合求解，与顺序无关）
  物种层   同 Eq. 5
  物种原型 P^sp（Eq. 8，在扩展前的 f'^a、f'^v 空间，每物种每强度 m 个）用于软路由：
           r_k(x) = softmax_k(τ · s_k(x))，s_k = 两模态到物种 k 原型的最大余弦相似度的平均
  模态权重 β(x)：逐样本、与类别无关。uncert = 岭回归后验方差 q_m(x)=x^T A_m^{-1} x（按该物种训练集
           中位数归一化）越小权重越大；也可选 entropy / fixed
  γ(x)     = γmax - (γmax-γmin)·max_k r_k(x)：路由越确定越依赖物种层（"基于置信度"）
  预测     ŷ = softmax(γ ŷ^av + (1-γ) Σ_k r_k (β^a W^a_k F^a + β^v W^v_k F^v))

目标均为 one-hot（论文 Eq. 4），预测取 argmax。
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analytic_cil import RandomBuffer, align, load_phase
from datasets.fish_av_incremental import INCREMENTAL_PHASES

NUM_CLASSES = 4


# ----------------------------------------------------------------------------- 基本组件

def onehot(y):
    return torch.nn.functional.one_hot(y.long(), NUM_CLASSES).float()


def kmeans(x, k, iters=50, seed=0):
    """GPU k-means（k-means++ 初始化），返回 (centers, assign)。"""
    g = torch.Generator(device=x.device).manual_seed(seed)
    n = len(x)
    k = min(k, n)
    centers = x[torch.randint(n, (1,), generator=g, device=x.device)]
    for _ in range(1, k):
        d = torch.cdist(x, centers).min(dim=1).values ** 2
        idx = torch.multinomial(d / d.sum(), 1, generator=g)
        centers = torch.cat([centers, x[idx]])
    for _ in range(iters):
        assign = torch.cdist(x, centers).argmin(dim=1)
        new = torch.stack([x[assign == j].mean(0) if (assign == j).any() else centers[j] for j in range(k)])
        if torch.allclose(new, centers):
            break
        centers = new
    return centers, torch.cdist(x, centers).argmin(dim=1)


class Ridge:
    """带 bias 列的岭回归，可选累加统计量。"""

    def __init__(self, dim, lam, device, keep_inv=False):
        self.A = torch.zeros(dim + 1, dim + 1, device=device)
        self.B = torch.zeros(dim + 1, NUM_CLASSES, device=device)
        self.lam, self.keep_inv, self.W, self.A_inv = lam, keep_inv, None, None

    @staticmethod
    def aug(x):
        return torch.cat([x, torch.ones(len(x), 1, device=x.device)], dim=1)

    def add(self, x, Y, chunk=8192):
        for i in range(0, len(x), chunk):
            xa = self.aug(x[i:i + chunk])
            self.A += xa.T @ xa
            self.B += xa.T @ Y[i:i + chunk]

    def solve(self):
        eye = torch.eye(len(self.A), device=self.A.device)
        if self.keep_inv:
            self.A_inv = torch.linalg.inv(self.A + self.lam * eye)
            self.W = self.A_inv @ self.B
        else:
            self.W = torch.linalg.solve(self.A + self.lam * eye, self.B)

    def scores(self, x):
        return self.aug(x) @ self.W

    def variance(self, x):
        xa = self.aug(x)
        return ((xa @ self.A_inv) * xa).sum(dim=1)


# ----------------------------------------------------------------------------- HAIL-FFIR

class HAIL:
    def __init__(self, version, in_dims, D, lam_gen, lam_sp, device, m=5, alpha=0.7,
                 gamma_max=0.8, gamma_min=0.3, beta_mode='uncert', tau=20.0, n_stages=7, seed=42):
        self.v, self.device, self.m, self.alpha = version, device, m, alpha
        self.gmax, self.gmin, self.beta_mode, self.tau, self.K = gamma_max, gamma_min, beta_mode, tau, n_stages
        d_av, d_v, d_a = in_dims
        self.buf = {'av': RandomBuffer(d_av, D, seed, device),       # FE (Eq. 3)，三路各自固定随机投影
                    'v': RandomBuffer(d_v, D, seed + 1, device),
                    'a': RandomBuffer(d_a, D, seed + 2, device)}
        self.D, self.lam_gen, self.lam_sp = D, lam_gen, lam_sp
        self.general = Ridge(D, lam_gen, device)
        self.sp_heads = []          # [(Ridge_a, Ridge_v, q_ref_a, q_ref_v)]，每个物种一组
        self.P_av, self.P_av_lab = None, None     # 通用原型（扩展空间，原稿版用）
        self.P_sp = []              # 每个物种 (proto_a, proto_v)，扩展前空间，修订版路由用
        self.step = -1
        self.lam_p_hist = []

    def expand(self, feats):
        f_av, f_v, f_a = (f.to(self.device) for f in feats)
        return self.buf['av'](f_av), self.buf['v'](f_v), self.buf['a'](f_a), f_v, f_a

    def _class_kmeans(self, x, y):
        cs, labs = [], []
        for c in range(NUM_CLASSES):
            xc = x[y == c]
            if len(xc) == 0:
                continue
            ctr, _ = kmeans(xc, self.m, seed=c)
            cs.append(ctr)
            labs.append(torch.full((len(ctr),), c, device=x.device))
        return torch.cat(cs), torch.cat(labs)

    def learn(self, feats, y):
        self.step += 1
        y = y.to(self.device)
        Fav, Fv, Fa, fv, fa = self.expand(feats)
        Y = onehot(y)

        # ---- 通用强度层 ----
        if self.v == 'revised':
            self.general.add(Fav, Y)
            self.general.solve()
        else:
            g = Ridge(self.D, self.lam_gen, self.device)
            if self.P_av is None:                       # Phase 0：没有历史原型
                g.add(Fav, Y)
                self.lam_p_hist.append(None)
            else:                                       # Eq. 9：当前数据 + λp·原型
                sim = torch.nn.functional.cosine_similarity(Fav.mean(0), self.P_av.mean(0), dim=0).item()
                lam_p = max(0.2, sim)
                self.lam_p_hist.append(lam_p)
                g.add(torch.cat([Fav, lam_p * self.P_av]), torch.cat([Y, onehot(self.P_av_lab)]))
            g.solve()                                   # Eq. 10
            self.general = g
            # 原型：Phase 0 用 k-means 初始化（Eq. 7），之后 EMA 演化（Eq. 11）
            if self.P_av is None:
                self.P_av, self.P_av_lab = self._class_kmeans(Fav, y)
            else:
                for c in range(NUM_CLASSES):
                    pidx = (self.P_av_lab == c).nonzero().squeeze(1)
                    xc = Fav[y == c]
                    if len(xc) == 0:
                        continue
                    near = torch.cdist(xc, self.P_av[pidx]).argmin(dim=1)
                    for j_local, j in enumerate(pidx.tolist()):
                        sel = xc[near == j_local]
                        if len(sel):
                            self.P_av[j] = self.alpha * self.P_av[j] + (1 - self.alpha) * sel.mean(0)

        # ---- 物种专属层（Eq. 5），两个版本相同 ----
        keep_inv = self.v == 'revised' and self.beta_mode == 'uncert'
        ra, rv = Ridge(self.D, self.lam_sp, self.device, keep_inv), Ridge(self.D, self.lam_sp, self.device, keep_inv)
        ra.add(Fa, Y); ra.solve()
        rv.add(Fv, Y); rv.solve()
        qa = qv = None
        if keep_inv:   # 后验方差的参考尺度：该物种训练数据上的中位数
            qa = ra.variance(Fa[:4096]).median()
            qv = rv.variance(Fv[:4096]).median()
            ra.A = rv.A = None  # A 不再需要，只留 A_inv
        self.sp_heads.append((ra, rv, qa, qv))

        # ---- 物种原型（Eq. 8），扩展前空间 ----
        pa, _ = self._class_kmeans(torch.nn.functional.normalize(fa, dim=1), y)
        pv, _ = self._class_kmeans(torch.nn.functional.normalize(fv, dim=1), y)
        # 该物种训练数据到自己原型的典型距离（中位数），作为模态可靠性的参照尺度
        da_ref = (1 - (torch.nn.functional.normalize(fa, dim=1) @ pa.T).max(1).values).median()
        dv_ref = (1 - (torch.nn.functional.normalize(fv, dim=1) @ pv.T).max(1).values).median()
        self.P_sp.append((pa, pv, da_ref, dv_ref))

    # ---- 预测 ----
    def proto_sims(self, fv, fa):
        """每个模态到每个物种原型的最大余弦相似度，(N, n_species) x 2。"""
        fa_n = torch.nn.functional.normalize(fa, dim=1)
        fv_n = torch.nn.functional.normalize(fv, dim=1)
        sa = torch.stack([(fa_n @ pa.T).max(1).values for pa, _, _, _ in self.P_sp], dim=1)
        sv = torch.stack([(fv_n @ pv.T).max(1).values for _, pv, _, _ in self.P_sp], dim=1)
        return sa, sv

    def routing(self, fv, fa):
        sa, sv = self.proto_sims(fv, fa)
        return torch.softmax(self.tau * 0.5 * (sa + sv), dim=1)

    @torch.no_grad()
    def predict(self, feats, chunk=2048, return_aux=False):
        preds, aux = [], {'gamma': [], 'beta_a': [], 'route_max': []}
        n = len(feats[0])
        for i in range(0, n, chunk):
            Fav, Fv, Fa, fv, fa = self.expand([f[i:i + chunk] for f in feats])
            y_av = self.general.scores(Fav)
            if self.v == 'original':
                ra, rv, _, _ = self.sp_heads[-1]                        # 字面：当前阶段 k 的物种头
                y_sp = 0.5 * ra.scores(Fa) + 0.5 * rv.scores(Fv)        # β 无法按原文计算 -> 0.5
                gamma = self.gmax - (self.gmax - self.gmin) * (self.step + 1) / self.K   # Eq. 14
                gamma = torch.full((len(Fav), 1), gamma, device=self.device)
                beta_a = torch.full((len(Fav),), 0.5, device=self.device)
                rmax = torch.ones(len(Fav), device=self.device)
            else:
                sim_a, sim_v = self.proto_sims(fv, fa)
                r = torch.softmax(self.tau * 0.5 * (sim_a + sim_v), dim=1)   # 物种软路由 (N, S)
                y_sp = torch.zeros_like(y_av)
                beta_a = torch.zeros(len(Fav), device=self.device)
                for k, (ra, rv, qa, qv) in enumerate(self.sp_heads):
                    sa, sv = ra.scores(Fa), rv.scores(Fv)
                    if self.beta_mode == 'proto':
                        # 原型可靠性：离该物种原型越远（相对训练时的典型距离），该模态越不可靠
                        _, _, da_ref, dv_ref = self.P_sp[k]
                        ua = (1 - sim_a[:, k]).clamp_min(1e-6) / da_ref
                        uv = (1 - sim_v[:, k]).clamp_min(1e-6) / dv_ref
                        ba = uv / (ua + uv)
                    elif self.beta_mode == 'uncert':
                        ua, uv = ra.variance(Fa) / qa, rv.variance(Fv) / qv
                        ba = uv / (ua + uv)                              # 音频不确定性越小，音频权重越大
                    elif self.beta_mode == 'entropy':
                        ea = -(torch.softmax(sa, 1) * torch.log_softmax(sa, 1)).sum(1)
                        ev = -(torch.softmax(sv, 1) * torch.log_softmax(sv, 1)).sum(1)
                        ba = ev / (ea + ev + 1e-8)
                    else:
                        ba = torch.full((len(Fav),), 0.5, device=self.device)
                    y_sp += r[:, k:k + 1] * (ba[:, None] * sa + (1 - ba[:, None]) * sv)
                    beta_a += r[:, k] * ba
                rmax = r.max(1).values
                gamma = (self.gmax - (self.gmax - self.gmin) * rmax)[:, None]
            prob = torch.softmax(gamma * y_av + (1 - gamma) * y_sp, dim=1)
            preds.append(prob.argmax(1).cpu())
            if return_aux:
                aux['gamma'].append(gamma.squeeze(1).cpu())
                aux['beta_a'].append(beta_a.cpu())
                aux['route_max'].append(rmax.cpu())
        pred = torch.cat(preds)
        if return_aux:
            return pred, {k: torch.cat(v) for k, v in aux.items()}
        return pred


# ----------------------------------------------------------------------------- 主流程

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--version', choices=['original', 'revised'], required=True)
    p.add_argument('--tag', default='mv2_s3dft')
    p.add_argument('--feat_av', default=None, help='通用层输入，默认 hail_av_<tag>')
    p.add_argument('--feat_v', default=None, help='视频物种层输入，默认 hail_v_<tag>；可换成原始 video_s3d_ft')
    p.add_argument('--feat_a', default=None, help='音频物种层输入，默认 hail_a_<tag>；可换成原始 audio_cnn14mv2')
    p.add_argument('--dim', type=int, default=8192)
    p.add_argument('--lam_gen', type=float, default=1.0)
    p.add_argument('--lam_sp', type=float, default=1.0)
    p.add_argument('--beta', choices=['proto', 'uncert', 'entropy', 'fixed'], default='uncert')
    p.add_argument('--tau', type=float, default=20.0)
    p.add_argument('--gamma_max', type=float, default=0.8)
    p.add_argument('--gamma_min', type=float, default=0.3)
    p.add_argument('--m', type=int, default=5, help='每个强度的原型数')
    p.add_argument('--alpha', type=float, default=0.7)
    p.add_argument('--order', type=int, nargs='*', default=None)
    p.add_argument('--eval_split', choices=['val', 'test'], default='test')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default=None)
    args = p.parse_args()

    names = [args.feat_av or f'hail_av_{args.tag}', args.feat_v or f'hail_v_{args.tag}',
             args.feat_a or f'hail_a_{args.tag}']
    order = [0] + (args.order if args.order else list(range(1, len(INCREMENTAL_PHASES))))
    data = {}
    for ph in order:
        pf = [load_phase(n, ph) for n in names]
        data[ph] = {'train': align(pf, 'train'), 'eval': align(pf, args.eval_split)}
    in_dims = [f.shape[1] for f in data[0]['train'][0]]
    device = 'cuda'
    model = HAIL(args.version, in_dims, args.dim, args.lam_gen, args.lam_sp, device, m=args.m,
                 alpha=args.alpha, gamma_max=args.gamma_max, gamma_min=args.gamma_min,
                 beta_mode=args.beta, tau=args.tau, n_stages=len(order), seed=args.seed)
    print(f'HAIL-FFIR {args.version} | feats={names} | beta={args.beta} tau={args.tau} D={args.dim} '
          f'lam_gen={args.lam_gen} lam_sp={args.lam_sp} order={order} eval={args.eval_split}')

    def acc_on(ph):
        feats, y, _ = data[ph]['eval']
        pred = model.predict(feats)
        return float((pred == y).float().mean()), int((pred == y).sum()), len(y)

    history, best = [], {}
    for step, ph in enumerate(order):
        zero_shot = acc_on(ph)[0] if step > 0 else None
        feats, y, _ = data[ph]['train']
        model.learn(feats, y)
        per = {q: acc_on(q) for q in order[:step + 1]}
        acc_seen = sum(v[1] for v in per.values()) / sum(v[2] for v in per.values())
        for q, v in per.items():
            best[q] = max(best.get(q, 0.0), v[0])
        old = order[:step]
        avg_forget = float(np.mean([best[q] - per[q][0] for q in old])) if old else 0.0
        forget_p0 = history[0]['per_phase'][0] - per[0][0] if step > 0 else 0.0
        history.append({'step': step, 'phase': ph, 'acc_seen': acc_seen, 'avg_forget': avg_forget,
                        'forget_p0': forget_p0, 'zero_shot': zero_shot,
                        'per_phase': {q: v[0] for q, v in per.items()}})
        zs = f'{zero_shot:.4f}' if zero_shot is not None else '  -   '
        lp = model.lam_p_hist[-1] if model.lam_p_hist else None
        print(f'step {step} phase {ph} {INCREMENTAL_PHASES[ph]!s:32s} acc_seen {acc_seen:.4f} | '
              f'forget_p0 {forget_p0:.4f} | avg_forget {avg_forget:.4f} | zero-shot {zs} | '
              + (f'λp {lp:.3f} | ' if lp is not None else '')
              + 'per-phase ' + ' '.join(f'{q}:{v[0]:.3f}' for q, v in per.items()), flush=True)

    # 最终模型在全部已见测试集上的路由/权重统计
    allf = [torch.cat([data[q]['eval'][0][j] for q in order]) for j in range(3)]
    ally = torch.cat([data[q]['eval'][1] for q in order])
    _, aux = model.predict(allf, return_aux=True)
    summary = {'final_acc': history[-1]['acc_seen'],
               'avg_inc_acc': float(np.mean([h['acc_seen'] for h in history])),
               'final_forget_p0': history[-1]['forget_p0'],
               'final_avg_forget': history[-1]['avg_forget'],
               'mean_gamma': float(aux['gamma'].mean()), 'mean_beta_audio': float(aux['beta_a'].mean()),
               'mean_route_max': float(aux['route_max'].mean())}
    print('SUMMARY ' + ' '.join(f'{k}={v:.4f}' for k, v in summary.items()))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump({'args': vars(args), 'summary': summary, 'history': history}, f, indent=2)


if __name__ == '__main__':
    main()
