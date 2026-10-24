"""物种条件偏差图（替换正文 VI-B 的 t-SNE）：物种 × 真实强度 的平均带符号误差 E[ŷ − y]（单位：强度等级），
左：只用共享头（reliability 加权的三个头，无专家），右：完整模型（含物种专家）；3 个种子平均。
数据来自 results/species_bias.json（由 bias_probe 计算）。输出 Latex/figures/species_bias.{pdf,svg,png}。"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

SPECIES = ['Red Tilapia', 'Tilapia', 'Jade Perch', 'Largemouth Bass', 'Lotus Carp', 'Sunfish', 'O. punctatus']
LEVELS = ['None', 'Weak', 'Medium', 'Strong']
d = json.load(open('results/species_bias.json'))
B = {m: np.array([[d[f'{m}|{k}|{c}'][0] for c in range(4)] for k in range(7)]) for m in ('shared', 'experts')}

plt.rcParams.update({'font.size': 7, 'axes.linewidth': 0.6})
fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.3), sharey=True)
norm = TwoSlopeNorm(vmin=-0.7, vcenter=0.0, vmax=0.7)
for ax, (m, title) in zip(axes, (('shared', '(a) Shared heads'), ('experts', '(b) With species experts'))):
    M = B[m]
    im = ax.imshow(M, cmap='RdBu_r', norm=norm, aspect='auto')
    for k in range(7):
        for c in range(4):
            v = M[k, c]
            ax.text(c, k, f'{v:+.2f}'.replace('+0.00', '0.00').replace('-0.00', '0.00'), ha='center', va='center',
                    fontsize=5.6, color='white' if abs(v) > 0.38 else '#1a1a1a')
    ax.set_xticks(range(4)); ax.set_xticklabels(LEVELS, fontsize=5.7)
    ax.xaxis.tick_top(); ax.tick_params(length=0, pad=2)
    ax.set_yticks(range(7)); ax.set_yticklabels(SPECIES, fontsize=6.3)
    # 白色细线分隔格子
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True); ax.set_yticks(np.arange(-0.5, 7, 1), minor=True)
    ax.grid(which='minor', color='white', lw=1.2); ax.tick_params(which='minor', length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=7, pad=12)
cb = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.06, pad=0.04, aspect=40)
cb.set_ticks([-0.6, -0.3, 0, 0.3, 0.6]); cb.ax.tick_params(labelsize=6, length=2)
cb.outline.set_linewidth(0.4)
OUT = 'outputs/figures/species_bias'
fig.savefig(OUT + '.pdf', bbox_inches='tight')
with plt.rc_context({'svg.fonttype': 'none'}):
    fig.savefig(OUT + '.svg', bbox_inches='tight')
fig.savefig(OUT + '.png', dpi=300, bbox_inches='tight')
fig.savefig('outputs/figures/species_bias.png', dpi=200, bbox_inches='tight')
print('mean |error|', {m: round(float(np.abs(B[m]).mean()), 3) for m in B}, 'max', {m: round(float(np.abs(B[m]).max()), 2) for m in B})
