"""补充材料图：类增量基准上 π_av 的精度-鲁棒性取舍（Latex/figures/prior_sweep.pdf）。
读 results/{ave,ks}/prior_sweep.json；实线 = 精确统计，虚线 = 原型；蓝 = 干净平均增量精度，橙 = 特征损坏平均精度；空心圈 = 验证集选出的 π_av。"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE, ORANGE, INK, MUTED = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e'
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=False)
for ax, (ds, title) in zip(axes, (('ave', 'AVE-CI'), ('ks', 'K-S-CI'))):
    d = json.load(open(f'results/{ds}/prior_sweep.json'))
    sel = {st: json.load(open(f'results/{ds}/{f}.json'))['best_hp'].get('prior_av')
           for st, f in (('exact', 'hail_exact'), ('proto', 'hail'))}
    for st, ls, mk in (('exact', '-', 'o'), ('proto', '--', 's')):
        rows = sorted((v for v in d.values() if v['stats'] == st), key=lambda v: v['prior_av'])
        x = [v['prior_av'] for v in rows]
        for key, col in (('avg', BLUE), ('corrupt', ORANGE)):
            y = [v[key] for v in rows]
            ax.plot(x, y, ls=ls, color=col, lw=1.6, marker=mk, ms=4, mec='white', mew=0.8, zorder=3)
            # 验证集选出的点：空心大圈
            if sel[st] in x:
                i = x.index(sel[st])
                ax.plot(x[i], y[i], marker='o', ms=9, mfc='none', mec=INK, mew=1.0, zorder=4)
    ax.set_title(title, fontsize=9, color=INK)
    ax.set_xlabel(r'Prior weight of the audio-visual head $\pi_{av}$')
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.grid(axis='y', color='#e4e3df', lw=0.6)
    ax.set_axisbelow(True)
axes[0].set_ylabel('Accuracy (%)')
# 图例：颜色 = 指标，线型 = 统计量，空心圈 = 验证集选择
from matplotlib.lines import Line2D
handles = [Line2D([], [], color=BLUE, lw=1.6, label='Clean (avg. incremental)'),
           Line2D([], [], color=ORANGE, lw=1.6, label='Corrupted (mean)'),
           Line2D([], [], color=MUTED, lw=1.6, ls='-', marker='o', ms=4, label='Exact statistics'),
           Line2D([], [], color=MUTED, lw=1.6, ls='--', marker='s', ms=4, label='Prototype memory'),
           Line2D([], [], ls='none', marker='o', ms=9, mfc='none', mec=INK, label='Selected on validation')]
fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.savefig('outputs/figures/prior_sweep.pdf', bbox_inches='tight')
fig.savefig('outputs/figures/prior_sweep.png', dpi=150, bbox_inches='tight')
print('saved')
