"""正文 VI-B 的 t-SNE 图（单栏）：强度得分的 t-SNE，(a) 只用共享头，(b) 加物种专家。
背景散点按强度着色并淡化；大标记为每个物种（形状）、每个强度（颜色）的中心点；同色中心点越集中，说明各物种越被对齐。
数据 results/tsne_heads_emb.npz（viz.shared_vs_expert 生成）。输出 Latex/figures/tsne_heads.{pdf,svg,png}。"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

SPECIES = ['Red Tilapia', 'Tilapia', 'Jade Perch', 'Largemouth Bass', 'Lotus Carp', 'Sunfish', 'O. punctatus']
LEVELS = ['None', 'Weak', 'Medium', 'Strong']
COL_I = ['#bdbdbd', '#9ecae1', '#3182bd', '#08519c']
MARK = ['o', 's', '^', 'D', 'v', 'P', 'X']
e = np.load('results/tsne_heads_emb.npz')
Y, S = e['intensity'], e['species']
sil = json.load(open('results/tsne_heads.json'))


def spread(E):
    """同一强度下各物种中心点到该强度平均中心的平均距离（归一化坐标），对 4 个强度取平均。"""
    out = []
    for c in range(4):
        cs = np.array([E[(Y == c) & (S == k)].mean(0) for k in range(7) if ((Y == c) & (S == k)).any()])
        out.append(np.linalg.norm(cs - cs.mean(0), axis=1).mean())
    return float(np.mean(out))


plt.rcParams.update({'font.size': 7, 'axes.linewidth': 0.6})
fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.5))
for ax, (m, t, key) in zip(axes, (('shared', '(a) Shared heads', 'Shared heads only'),
                                  ('experts', '(b) With species experts', 'Shared heads + species experts'))):
    E = e[m]
    for c in range(4):
        ax.scatter(*E[Y == c].T, s=0.7, color=COL_I[c], alpha=0.28, linewidths=0, rasterized=True)
    for c in range(4):
        for k in range(7):
            sel = (Y == c) & (S == k)
            if sel.any():
                ax.scatter(*E[sel].mean(0), s=17, marker=MARK[k], color=COL_I[c], edgecolors='#1a1a1a',
                           linewidths=0.45, zorder=5)
    ax.set_xlim(-1.08, 1.08); ax.set_ylim(-1.08, 1.08); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5); sp.set_color('#808080')
    ax.set_title(f"{t}\nsilhouette {sil[key]['silhouette_intensity']:.2f} · spread {spread(E):.2f}", fontsize=6.8,
                 pad=4)
hi = [Line2D([], [], ls='none', marker='o', ms=4.2, mfc=COL_I[c], mec='#1a1a1a', mew=0.4, label=LEVELS[c])
      for c in range(4)]
hs = [Line2D([], [], ls='none', marker=MARK[k], ms=4.0, mfc='white', mec='#1a1a1a', mew=0.6, label=SPECIES[k])
      for k in range(7)]
leg1 = fig.legend(handles=hi, loc='lower center', bbox_to_anchor=(0.5, 0.12), ncol=4, fontsize=5.8, frameon=False,
                  handletextpad=0.15, columnspacing=0.9)
fig.legend(handles=hs, loc='lower center', bbox_to_anchor=(0.5, -0.01), ncol=4, fontsize=5.6, frameon=False,
           handletextpad=0.1, columnspacing=0.6)
fig.tight_layout(rect=(0, 0.19, 1, 1), pad=0.3)
OUT = 'outputs/figures/tsne_heads'
fig.savefig(OUT + '.pdf', bbox_inches='tight', dpi=300)
with plt.rc_context({'svg.fonttype': 'none'}):
    fig.savefig(OUT + '.svg', bbox_inches='tight', dpi=300)
fig.savefig(OUT + '.png', dpi=300, bbox_inches='tight')
fig.savefig('outputs/figures/tsne_heads.png', dpi=200, bbox_inches='tight')
print('spread', {m: round(spread(e[m]), 3) for m in ('shared', 'experts')})
