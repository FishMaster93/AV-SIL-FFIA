"""
校准分析（回应 R1-m4）。主设置（片段级切分，音视频），各方法学完 7 个阶段后：
  - 取最终模型在验证集和测试集上的得分 / logits；
  - 统一做温度缩放：在验证集上用最小化 NLL 拟合一个温度 T（所有方法同样处理，闭式解方法的输出是回归得分，
    必须先校准才是概率；AquaLytic 的 CORAL 阈值先换算为类别概率再作为 log 概率缩放）；
  - 测试集报告 ECE（15 个等宽分箱）、NLL、Brier；梯度类方法额外报告未校准的 ECE。
另外画 HAIL-FFIR / DS-AL / Joint 的可靠性图。
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_all import Wrapper, load
from datasets.fish_av_incremental import INCREMENTAL_PHASES
from sahl_cil import A_RAW, V_RAW

R = 'results'
FIG = 'outputs/figures'
METHODS = ['finetune', 'lwf', 'ewc', 'er', 'icarl', 'avcil', 'acil', 'dsal', 'aqualytic', 'mmal', 'joint', 'hail']


@torch.no_grad()
def scores(w, fd):
    """返回 (N, 4) 的 logits / log 概率。"""
    fa, fv = fd[A_RAW].cuda(), fd[V_RAW].cuda()
    if w.kind == 'hail':
        return w.m.predict(fd, return_scores=True)
    if w.kind == 'grad':
        w.m.model.eval()
        return w.m.model(torch.cat([fa, fv], 1)).float().cpu()
    m = w.m
    out = []
    for i in range(0, len(fa), 4096):
        a, v = fa[i:i + 4096], fv[i:i + 4096]
        s = m.main(m.phi(a, v))
        if m.method == 'aqualytic':      # CORAL：P(y>k) = σ(s_k) -> 类别概率 -> log
            p_gt = torch.sigmoid(s)
            p = torch.cat([1 - p_gt[:, :1], p_gt[:, :-1] - p_gt[:, 1:], p_gt[:, -1:]], 1).clamp_min(1e-6)
            s = torch.log(p / p.sum(1, keepdim=True))
        elif m.method == 'dsal':
            s = s + m.hp['C'] * m.comp(m.phi(a, v, 'tanh'))
        elif m.method == 'mmal':
            s = s + m.hp['alpha'] * sum(m.uni[mm](m.phi(a, v, mods=[mm])) for mm in m.mods)
        out.append(s.float().cpu())
    return torch.cat(out)


def fit_temperature(logits, y):
    """在验证集上最小化 NLL 拟合温度（对 log T 做网格 + 局部细化）。"""
    grid = np.exp(np.linspace(np.log(1e-3), np.log(1e3), 241))
    nll = [F.cross_entropy(logits / t, y).item() for t in grid]
    t0 = grid[int(np.argmin(nll))]
    fine = t0 * np.exp(np.linspace(-0.1, 0.1, 81))
    nll = [F.cross_entropy(logits / t, y).item() for t in fine]
    return float(fine[int(np.argmin(nll))])


def metrics(prob, y, n_bins=15, pred=None):
    """pred 给定时（AquaLytic：CORAL 官方解码 round(Σσ)，与主表一致）置信度取该预测类别的概率，否则取 argmax。"""
    if pred is None:
        conf, pred = prob.max(1)
    else:
        conf = prob.gather(1, pred[:, None]).squeeze(1)
    acc = (pred == y).float()
    ece = 0.0
    edges = torch.linspace(0, 1, n_bins + 1)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.float().mean().item() * abs(acc[m].mean().item() - conf[m].mean().item())
            bins.append((float((lo + hi) / 2), acc[m].mean().item(), conf[m].mean().item(), int(m.sum())))
    nll = F.nll_loss(torch.log(prob.clamp_min(1e-12)), y).item()
    brier = ((prob - F.one_hot(y, 4).float()) ** 2).sum(1).mean().item()
    return {'acc': acc.mean().item(), 'ece': ece, 'nll': nll, 'brier': brier, 'bins': bins}


def main():
    import sys
    only = sys.argv[1:]          # 只重算指定方法，其余沿用已有结果
    out = json.load(open(f'{R}/calibration.json')) if only else {}
    order = list(INCREMENTAL_PHASES)
    val = {ph: load('main', ph, 'val') for ph in order}
    test = {ph: load('main', ph, 'test') for ph in order}
    cat = lambda d: ({k: torch.cat([d[p][0][k] for p in order]) for k in (A_RAW, V_RAW)},
                     torch.cat([d[p][1] for p in order]))
    vfd, vy = cat(val)
    tfd, ty = cat(test)
    for method in (only or METHODS):
        res = []
        for seed in (42, 43, 44):
            w = Wrapper(method, seed)
            for ph in order:
                fd, y, _ = load('main', ph, 'train')
                w.learn(fd, y)
            lv, lt = scores(w, vfd), scores(w, tfd)
            T = fit_temperature(lv, vy)
            pred = (w.m.predict(tfd[A_RAW].cuda(), tfd[V_RAW].cuda()).cpu() if method == 'aqualytic' else None)
            r = {'T': T, 'cal': metrics(torch.softmax(lt / T, 1), ty, pred=pred)}
            if w.kind == 'grad':
                r['raw'] = metrics(torch.softmax(lt, 1), ty)
            res.append(r)
            del w
            torch.cuda.empty_cache()
        name = 'icarl_fc' if method == 'icarl' else method
        out[name] = res
        c = lambda k: 100 * np.mean([r['cal'][k] for r in res]) if k != 'nll' else np.mean([r['cal'][k] for r in res])
        raw = f" | raw ECE {100 * np.mean([r['raw']['ece'] for r in res]):.2f}" if 'raw' in res[0] else ''
        print(f'{name:10s} acc {c("acc"):.2f}  ECE {c("ece"):.2f}  NLL {c("nll"):.3f}  Brier {np.mean([r["cal"]["brier"] for r in res]):.4f}'
              f'  T={np.mean([r["T"] for r in res]):.3g}{raw}', flush=True)
    json.dump(out, open(f'{R}/calibration.json', 'w'))

    # 可靠性图
    show = [('dsal', 'DS-AL'), ('joint', 'Joint training'), ('hail', 'HAIL-FFIR')]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
    for ax, (k, t) in zip(axes, show):
        b = out[k][0]['cal']['bins']
        x = [v[0] for v in b]
        ax.bar(x, [v[1] for v in b], width=1 / 15, edgecolor='black', lw=0.4, color='#4C72B0', alpha=0.8, label='Accuracy')
        ax.plot([0, 1], [0, 1], ls='--', color='gray', lw=0.8)
        ax.set_title(f"{t} (ECE {100 * np.mean([r['cal']['ece'] for r in out[k]]):.2f}%)", fontsize=8)
        ax.set_xlabel('Confidence', fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    axes[0].set_ylabel('Accuracy', fontsize=8)
    fig.tight_layout()
    fig.savefig(f'{FIG}/reliability.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
