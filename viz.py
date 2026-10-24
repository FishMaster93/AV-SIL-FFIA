"""
论文可视化：
  (1) t-SNE：测试集编码器特征（音频+视频拼接），分别按物种和投喂强度着色
  (2) 物种条件的强度语义偏移：用所有物种共享的一个 AV 闭式解头给每个测试样本打"期望强度"分数，
      按 物种 x 真实强度 画箱线图；同一真实强度在不同物种上的分数位置不同，说明共享头无法同时拟合各物种
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analytic_cil import RandomBuffer
from sahl_cil import A_RAW, V_RAW, aug, load_aligned

FIG = 'outputs/figures'
SPECIES = ['Red Tilapia', 'Tilapia', 'Jade Perch', 'Largemouth Bass', 'Lotus Carp', 'Sunfish', 'O. punctatus']
CLASSES = ['None', 'Weak', 'Medium', 'Strong']
DEV = 'cuda'


def main():
    tr = {p: load_aligned([A_RAW, V_RAW], p, 'train') for p in range(7)}
    te = {p: load_aligned([A_RAW, V_RAW], p, 'test') for p in range(7)}

    # ---- (1) t-SNE ----
    rs = np.random.RandomState(0)
    Z, S, Y = [], [], []
    for p in range(7):
        fd, y, _ = te[p]
        idx = rs.choice(len(y), min(500, len(y)), replace=False)
        z = torch.cat([F.normalize(fd[A_RAW][idx], dim=1), F.normalize(fd[V_RAW][idx], dim=1)], 1)
        Z.append(z.numpy()); S += [p] * len(idx); Y += y[idx].tolist()
    Z = PCA(50, random_state=0).fit_transform(np.concatenate(Z))
    E = TSNE(2, perplexity=30, init='pca', random_state=0).fit_transform(Z)
    S, Y = np.array(S), np.array(Y)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6))
    cmap = plt.get_cmap('tab10')
    for p in range(7):
        axes[0].scatter(*E[S == p].T, s=2, color=cmap(p), label=SPECIES[p])
    for c, col in zip(range(4), ['#bdbdbd', '#9ecae1', '#3182bd', '#08519c']):
        axes[1].scatter(*E[Y == c].T, s=2, color=col, label=CLASSES[c])
    for ax, t in zip(axes, ('(a) coloured by species', '(b) coloured by feeding intensity')):
        ax.set_title(t, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        # 图例放在子图下方，避免遮住散点
        ax.legend(fontsize=6, markerscale=3, frameon=False, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout()
    fig.savefig(f'{FIG}/tsne.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ---- (2) 物种条件的强度语义偏移 ----
    bufs = [RandomBuffer(1024, 8192, 42, DEV, 'in_out'), RandomBuffer(1024, 8192, 43, DEV, 'linear')]
    phi = lambda fd: aug(torch.cat([bufs[0](fd[A_RAW].to(DEV)), bufs[1](fd[V_RAW].to(DEV))], 1))
    D = 16385
    A = torch.zeros(D, D, device=DEV); B = torch.zeros(D, 4, device=DEV)
    for p in range(7):
        fd, y, _ = tr[p]
        for i in range(0, len(y), 8192):
            X = phi({k: v[i:i + 8192] for k, v in fd.items()})
            A += X.T @ X; B += X.T @ F.one_hot(y[i:i + 8192].to(DEV), 4).float()
    W = torch.linalg.solve(A + 100 * torch.eye(D, device=DEV), B)
    data = {c: [] for c in range(4)}
    for p in range(7):
        fd, y, _ = te[p]
        prob = torch.softmax(10 * (phi(fd) @ W), 1).cpu()
        score = (prob * torch.arange(4)).sum(1).numpy()           # 期望强度 E[y]（0=None ... 3=Strong）
        for c in range(4):
            data[c].append(score[(y == c).numpy()])
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    width = 0.11
    for c, col in zip(range(4), ['#bdbdbd', '#9ecae1', '#3182bd', '#08519c']):
        pos = np.arange(7) + (c - 1.5) * (width * 1.6)
        bp = ax.boxplot(data[c], positions=pos, widths=width, patch_artist=True, showfliers=False,
                        medianprops=dict(color='black', lw=0.8))
        for b in bp['boxes']:
            b.set_facecolor(col)
        ax.plot([], [], color=col, lw=6, label=CLASSES[c])
    ax.set_xticks(range(7)); ax.set_xticklabels(SPECIES, fontsize=7)
    ax.set_ylabel('Shared-head intensity score')
    ax.set_yticks(range(4)); ax.set_yticklabels(CLASSES, fontsize=7)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(ncol=4, fontsize=7, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.2))
    fig.tight_layout()
    fig.savefig(f'{FIG}/species_shift.pdf')
    plt.close(fig)
    med = np.array([[np.median(data[c][p]) for c in range(4)] for p in range(7)])
    print('共享头期望强度的中位数（行=物种，列=真实强度 None/Weak/Medium/Strong）：')
    for p in range(7):
        print(f'  {SPECIES[p]:16s} ' + ' '.join(f'{v:.2f}' for v in med[p]))

    # ---- (3) 跨物种迁移：只用物种 i 训练，测物种 j 的准确率与带符号的强度偏差 E[pred - true] ----
    acc = np.zeros((7, 7))
    bias = np.zeros((7, 7))
    test_X = {p: phi(te[p][0]) for p in range(7)}
    for i in range(7):
        fd, y, _ = tr[i]
        Ai = torch.zeros(D, D, device=DEV); Bi = torch.zeros(D, 4, device=DEV)
        for k in range(0, len(y), 8192):
            X = phi({kk: v[k:k + 8192] for kk, v in fd.items()})
            Ai += X.T @ X; Bi += X.T @ F.one_hot(y[k:k + 8192].to(DEV), 4).float()
        Wi = torch.linalg.solve(Ai + 100 * torch.eye(D, device=DEV), Bi)
        for j in range(7):
            pred = (test_X[j] @ Wi).argmax(1).cpu()
            yt = te[j][1]
            acc[i, j] = 100 * (pred == yt).float().mean().item()
            bias[i, j] = (pred - yt).float().mean().item()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    short = ['RT', 'TI', 'JP', 'LB', 'LC', 'SF', 'OP']
    for ax, M, cmap, t, fmt_, vmin, vmax in (
            (axes[0], acc, 'Blues', '(a) Accuracy (%)', '{:.0f}', 0, 100),
            (axes[1], bias, 'RdBu_r', '(b) Mean intensity bias (pred $-$ true)', '{:+.2f}', -1, 1)):
        ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
        for a in range(7):
            for b in range(7):
                ax.text(b, a, fmt_.format(M[a, b]), ha='center', va='center', fontsize=6)
        ax.set_xticks(range(7)); ax.set_yticks(range(7))
        ax.set_xticklabels(short, fontsize=7); ax.set_yticklabels(short, fontsize=7)
        ax.set_xlabel('Test species', fontsize=8); ax.set_ylabel('Training species', fontsize=8)
        ax.set_title(t, fontsize=8)
    fig.tight_layout()
    fig.savefig(f'{FIG}/cross_species.pdf')
    plt.close(fig)
    np.save('results/cross_species_acc.npy', acc)
    np.save('results/cross_species_bias.npy', bias)
    print('跨物种准确率（行=训练物种，列=测试物种）：')
    for i in range(7):
        print(f'  {short[i]} ' + ' '.join(f'{v:5.1f}' for v in acc[i]))
    print('带符号强度偏差 E[pred-true]：')
    for i in range(7):
        print(f'  {short[i]} ' + ' '.join(f'{v:+.2f}' for v in bias[i]))



def shared_vs_expert():
    """R2-3：共享知识与物种特定知识。同一批测试样本，分别取"只用共享头"和"共享头 + 物种专家"（完整模型）的
    强度得分向量（|C| 维），做 t-SNE 并计算轮廓系数（按强度 / 按物种）。若专家把各物种映射到统一的强度语义，
    加入专家后按强度的轮廓系数上升、按物种的下降。结果：Latex/figures/tsne_heads.pdf、results/tsne_heads.json。"""
    import json
    from sklearn.metrics import silhouette_score
    from run_all import Wrapper, load
    w = Wrapper('hail', 42)
    for ph in range(7):
        fd, y, _ = load('main', ph, 'train')
        w.learn(fd, y)
    rs = np.random.RandomState(0)
    X = {A_RAW: [], V_RAW: []}
    S, Y = [], []
    for ph in range(7):
        fd, y, _ = load('main', ph, 'test')
        idx = rs.choice(len(y), min(400, len(y)), replace=False)
        for k in X:
            X[k].append(fd[k][idx])
        S += [ph] * len(idx)
        Y += y[idx].tolist()
    X = {k: torch.cat(v) for k, v in X.items()}
    S, Y = np.array(S), np.array(Y)
    m = w.m
    full = m.predict(X, return_scores=True).numpy()
    m.use_experts = False
    shared = m.predict(X, return_scores=True).numpy()
    m.use_experts = True
    out, emb = {}, {}
    for name, Z in (('Shared heads only', shared), ('Shared heads + species experts', full)):
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)
        out[name] = {'silhouette_intensity': float(silhouette_score(Z, Y)),
                     'silhouette_species': float(silhouette_score(Z, S))}
        E = TSNE(2, perplexity=30, init='pca', random_state=0).fit_transform(Z)
        emb[name] = (E - E.mean(0)) / np.abs(E - E.mean(0)).max()        # 两个子图使用相同的坐标比例
    np.savez('results/tsne_heads_emb.npz', shared=emb['Shared heads only'],
             experts=emb['Shared heads + species experts'], intensity=Y, species=S)
    cols_i = ['#bdbdbd', '#9ecae1', '#3182bd', '#08519c']
    cmap = plt.get_cmap('Dark2')      # 物种配色不含蓝色，避免与强度的蓝色系混淆

    def panel(color_by, fn, figsize):
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        for j, (name, tag) in enumerate((('Shared heads only', 'Shared heads'), ('Shared heads + species experts', 'With species experts'))):
            E, ax = emb[name], axes[j]
            if color_by == 'intensity':
                for c in range(4):
                    ax.scatter(*E[Y == c].T, s=1.2, color=cols_i[c], label=CLASSES[c], rasterized=True)
                sil = out[name]['silhouette_intensity']
            else:
                for p_ in range(7):
                    ax.scatter(*E[S == p_].T, s=1.2, color=cmap(p_), label=SPECIES[p_], rasterized=True)
                sil = out[name]['silhouette_species']
            ax.set_title(f"({'ab'[j]}) {tag}\nsilhouette {sil:.2f}", fontsize=7.5)
            ax.set_xlim(-1.08, 1.08); ax.set_ylim(-1.08, 1.08); ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, fontsize=6.5, markerscale=4, frameon=False, ncol=4 if color_by == 'intensity' else 4,
                   loc='lower center', bbox_to_anchor=(0.5, -0.01), columnspacing=0.8, handletextpad=0.2)
        fig.tight_layout(rect=(0, 0.1 if color_by == 'intensity' else 0.16, 1, 1))
        fig.savefig(f'{FIG}/{fn}.pdf', dpi=300, bbox_inches='tight')
        fig.savefig(f'outputs/figures/{fn}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    panel('intensity', 'tsne_heads', (3.5, 2.15))            # 正文：单栏，按强度着色
    panel('species', 'tsne_heads_species', (3.5, 2.35))      # 补充材料：按物种着色
    json.dump(out, open('results/tsne_heads.json', 'w'), indent=1)
    print('共享头 vs 专家 轮廓系数：', out)


if __name__ == '__main__':
    main()
    shared_vs_expert()
