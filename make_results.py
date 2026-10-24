"""
汇总 results/ 下的所有实验，生成论文表格（results/tables/*.tex + 终端 Markdown）和图（Latex/figures/*.pdf）。
可以在实验未全部完成时运行，缺失的部分会跳过。
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = 'results'
FIG = 'outputs/figures'
TAB = f'{R}/tables'
os.makedirs(TAB, exist_ok=True)
SPECIES = ['Red Tilapia', 'Tilapia', 'Jade Perch', 'Largemouth Bass', 'Lotus Carp', 'Sunfish', 'O. punctatus']
CLASSES = ['None', 'Weak', 'Medium', 'Strong']
NAMES = {'finetune': 'Fine-tuning', 'joint': 'Joint (shared head)', 'joint_hail': 'Joint (HAIL-FFIR model)', 'lwf': 'LwF', 'ewc': 'EWC', 'er': 'ER',
         'icarl_nme': 'iCaRL-NME', 'icarl_fc': 'iCaRL-FC', 'avcil': 'AV-CIL*', 'acil': 'ACIL', 'dsal': 'DS-AL',
         'aqualytic': 'AquaLytic', 'mmal': 'MMAL*', 'hail': 'HAIL-FFIR (ours)', 'hail_exact': 'HAIL-FFIR (exact stats)'}
CITE = {'lwf': 'li2017learning', 'ewc': 'Kirkpatrick2017overcoming', 'icarl_nme': 'rebuffi2017icarl',
        'icarl_fc': 'rebuffi2017icarl', 'avcil': 'pian2023audio', 'acil': 'zhuang2022acil', 'dsal': 'zhuang2024ds',
        'aqualytic': 'xiao2026aqualytic', 'mmal': 'yue2024mmal'}
EXEMPLAR = {'joint', 'joint_hail', 'er', 'icarl_nme', 'icarl_fc', 'avcil'}
GROUPS = {'Clean': ['clean'], 'A-noise': ['a_noise_1', 'a_noise_2', 'a_noise_3'], 'A-missing': ['a_missing'],
          'V-noise': ['v_noise_1', 'v_noise_2', 'v_noise_3'], 'V-blur': ['v_blur_1', 'v_blur_2', 'v_blur_3'],
          'V-occlusion': ['v_occl_1', 'v_occl_2', 'v_occl_3'], 'V-turbidity': ['v_haze_1', 'v_haze_2', 'v_haze_3'],
          'V-missing': ['v_missing']}


def runs(exp):
    """{method: [run dict, ...]}"""
    out = defaultdict(list)
    for f in sorted(glob.glob(f'{R}/all/{exp}/*.json')):
        m = re.match(r'(.+)_(seed\d+|order\d+)\.json', os.path.basename(f)).group(1)
        out[m].append(json.load(open(f)))
    return out


def ms(vals, scale=100):
    v = np.array(vals) * scale
    return v.mean(), v.std()


def fmt(m, s=None, bold=False):
    t = f'{m:.2f}' + (f'$\\pm${s:.2f}' if s is not None else '')
    return f'\\textbf{{{t}}}' if bold else t


def label(k):
    n = NAMES.get(k, k)
    return f'{n} \\cite{{{CITE[k]}}}' if k in CITE else n


# ----------------------------------------------------------------------------- Table I

def table1():
    """单模态列来自 results/baselines + results/final（HAIL 单模态）；音视频列来自 results/all/main。"""
    main = runs('main')
    rows = ['finetune', 'lwf', 'ewc', 'er', 'icarl_nme', 'icarl_fc', 'avcil', 'acil', 'dsal', 'aqualytic', 'mmal',
            'hail', 'joint', 'joint_hail']
    uni = {}
    for f in glob.glob(f'{R}/baselines/*.json'):
        d = json.load(open(f))
        for name, sm in d['summary'].items():
            uni[(name, d['modality'])] = {k: (100 * v[0], 100 * v[1]) for k, v in sm.items()}
    for mod in ('audio', 'video'):
        rs = main.get(f'hail_{mod}', [])       # 单模态 HAIL：run_all 的 hail_audio / hail_video 变体
        if rs:
            uni[('hail', mod)] = {k: ms([r[k] for r in rs]) for k in ('final_acc', 'avg_inc_acc', 'final_avg_forget')}
        rs = main.get(f'joint_hail_{mod}', [])
        if rs:
            uni[('joint_hail', mod)] = {k: ms([r[k] for r in rs]) for k in ('final_acc', 'avg_inc_acc', 'final_avg_forget')}
    av = {m: {k: ms([r[k] for r in rr]) for k in ('final_acc', 'avg_inc_acc', 'final_avg_forget')}
          for m, rr in main.items()}
    best = {}
    for mod in ('audio', 'video', 'av'):
        src = av if mod == 'av' else {k[0]: v for k, v in uni.items() if k[1] == mod}
        cand = {m: src[m] for m in rows if m in src and not m.startswith('joint')}
        if cand:
            best[mod] = (max(cand, key=lambda m: cand[m]['final_acc'][0]),
                         max(cand, key=lambda m: cand[m]['avg_inc_acc'][0]),
                         min(cand, key=lambda m: cand[m]['final_avg_forget'][0]))
    lines, md = [], []
    for m in rows:
        cells = []
        for mod in ('audio', 'video', 'av'):
            src = av if mod == 'av' else {k[0]: v for k, v in uni.items() if k[1] == mod}
            if m not in src:
                cells += ['--'] * 3
                continue
            r = src[m]
            b = best.get(mod, (None,) * 3)
            cells += [fmt(*r['final_acc'], bold=(m == b[0])), fmt(r['avg_inc_acc'][0], bold=(m == b[1])),
                      fmt(r['final_avg_forget'][0], bold=(m == b[2]))]
        ef = '\\XSolidBrush' if m in EXEMPLAR else '\\Checkmark'
        lines.append(f'{label(m)} & {ef} & ' + ' & '.join(cells) + ' \\\\')
        md.append(f'| {NAMES.get(m, m)} | ' + ' | '.join(re.sub(r'\\textbf\{(.*?)\}', r'**\1**', c).replace('$\\pm$', '±') for c in cells) + ' |')
    open(f'{TAB}/table1_rows.tex', 'w').write('\n'.join(lines) + '\n')
    print('\n## Table I（最终 Acc / 平均增量 Acc / 遗忘；音频 | 视频 | 音视频）')
    print('\n'.join(md))


# ----------------------------------------------------------------------------- 鲁棒性

def robustness():
    main = runs('main')
    order = ['finetune', 'lwf', 'ewc', 'er', 'icarl_fc', 'avcil', 'acil', 'dsal', 'aqualytic', 'mmal', 'joint',
             'hail_exact', 'hail']
    lines, md = [], []
    for m in order:
        if m not in main or 'corrupt' not in main[m][0]:
            continue
        cells = []
        for g, conds in GROUPS.items():
            vals = [np.mean([r['final_acc'] if c == 'clean' else r['corrupt'][c] for c in conds]) for r in main[m]]
            cells.append(ms(vals)[0])
        allc = [c for g, cs in GROUPS.items() if g != 'Clean' for c in cs]
        mean = ms([np.mean([r['corrupt'][c] for c in allc]) for r in main[m]])
        held = [c for c in allc if c.split('_')[1] in ('blur', 'occl', 'haze')]   # 未参与选参的损坏类型
        cells.append(ms([np.mean([r['corrupt'][c] for c in held]) for r in main[m]])[0])
        cells.append(mean[0])
        lines.append(f'{label(m)} & ' + ' & '.join(f'{c:.2f}' for c in cells) + ' \\\\')
        md.append(f'| {NAMES.get(m, m)} | ' + ' | '.join(f'{c:.2f}' for c in cells) + ' |')
    open(f'{TAB}/robustness_rows.tex', 'w').write('\n'.join(lines) + '\n')
    print('\n## 鲁棒性（' + ' / '.join(GROUPS) + ' / 损坏平均）')
    print('\n'.join(md))
    # 模态权重随损坏变化（HAIL-FFIR）
    if 'hail' in main and 'corrupt_mean_w' in main['hail'][0]:
        conds = ['clean'] + [c for g, cs in GROUPS.items() if g != 'Clean' for c in cs]
        W = []
        for c in conds:
            ws = [r['mean_w_clean'] if c == 'clean' else r['corrupt_mean_w'][c] for r in main['hail']]
            W.append(np.mean(ws, axis=0))
        W = np.array(W)
        # 单栏：同类损坏的 3 个强度相邻，类之间留空隙；配色与框架图一致（AV 绿、音频蓝、视频橙）
        groups = [('Clean', ['clean'])] + [(g, cs) for g, cs in GROUPS.items() if g != 'Clean']
        GLAB = {'Clean': 'Clean', 'A-noise': 'Audio\nnoise', 'A-missing': 'Audio\nmissing', 'V-noise': 'Video\nnoise',
                'V-blur': 'Blur', 'V-occlusion': 'Occlusion', 'V-turbidity': 'Turbidity', 'V-missing': 'Video\nmissing'}
        xs, gx, pos = [], [], 0.0
        for g, cs in groups:
            start = pos
            for _ in cs:
                xs.append(pos); pos += 1.0
            gx.append((g, (start + pos - 1) / 2, len(cs)))
            pos += 0.6
        xs = np.array(xs)
        plt.rcParams.update({'font.size': 7, 'axes.linewidth': 0.6})
        fig, ax = plt.subplots(figsize=(3.5, 2.05))
        bottom = np.zeros(len(conds))
        for j, (h, col) in enumerate(zip(['Audio-visual head', 'Audio head', 'Visual head'],
                                         ['#5BAA62', '#4A86C5', '#E8894A'])):
            ax.bar(xs, W[:, j], bottom=bottom, color=col, label=h, width=0.86, edgecolor='white', linewidth=0.4)
            bottom += W[:, j]
        ax.set_xlim(xs[0] - 0.7, xs[-1] + 0.7); ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(['0', '0.25', '0.50', '0.75', '1.00'], fontsize=6)
        ax.set_ylabel('Mean weight $w_h$', fontsize=6.8)
        sev = [str(i + 1) if g != 'Clean' and n == 3 else '' for g, cs in groups for i, n in
               [(k, len(cs)) for k in range(len(cs))]]
        ax.set_xticks(xs); ax.set_xticklabels(sev, fontsize=5)
        ax.tick_params(axis='x', length=0, pad=1.5); ax.tick_params(axis='y', length=2)
        for g, cx, n in gx:
            ax.text(cx, -0.105, GLAB[g], ha='center', va='top', fontsize=5.6, transform=ax.get_xaxis_transform(),
                    linespacing=0.95)
        # 说明柱子下方的数字是损坏强度（具体参数见图注）
        ax.text(xs[0], -0.012, 'severity', ha='center', va='top', fontsize=4.8, style='italic', color='#555555',
                transform=ax.get_xaxis_transform())
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.legend(ncol=3, fontsize=6, loc='lower center', bbox_to_anchor=(0.5, 1.0), frameon=False,
                  handlelength=1.2, handletextpad=0.4, columnspacing=1.0)
        fig.tight_layout(pad=0.3)
        fig.savefig(f'{FIG}/modality_weights.pdf', bbox_inches='tight')
        with plt.rc_context({'svg.fonttype': 'none'}):
            fig.savefig(f'{FIG}/modality_weights.svg', bbox_inches='tight')
        fig.savefig(f'{FIG}/modality_weights.png', dpi=300, bbox_inches='tight')
        fig.savefig('outputs/figures/modality_weights.png', dpi=200, bbox_inches='tight')
        plt.rcParams.update(plt.rcParamsDefault); matplotlib.use('Agg')
        plt.close(fig)


# ----------------------------------------------------------------------------- 消融

def ablation():
    main = runs('main')
    comp = [('abl_shared_av', 'Shared AV head only', '', '', ''),
            ('abl_experts_av', '+ species experts (AV head only)', '\\Checkmark', '', ''),
            ('abl_shared_rel', 'Shared heads + reliability weighting', '', '\\Checkmark', '\\Checkmark'),
            ('abl_fixed_weights', 'Experts + three heads, fixed weights', '\\Checkmark', '', '\\Checkmark'),
            ('abl_route_av', 'Experts + reliability, AV-only routing', '\\Checkmark', '\\Checkmark', ''),
            ('hail', 'Full HAIL-FFIR', '\\Checkmark', '\\Checkmark', '\\Checkmark')]
    print('\n## 组件消融（最终 Acc / 平均 Acc / 遗忘 / 损坏平均）')
    lines = []
    for k, name, e, rw, rr in comp:
        if k not in main:
            continue
        rs = main[k]
        a, b, c, d = (ms([r['final_acc'] for r in rs]), ms([r['avg_inc_acc'] for r in rs]),
                      ms([r['final_avg_forget'] for r in rs]), ms([r['corrupt_mean'] for r in rs]))
        lines.append(f'{name} & {e} & {rw} & {rr} & {fmt(*a)} & {fmt(b[0])} & {fmt(c[0])} & {fmt(*d)} \\\\')
        print(f'| {name} | {a[0]:.2f}±{a[1]:.2f} | {b[0]:.2f} | {c[0]:.2f} | {d[0]:.2f}±{d[1]:.2f} |')
    open(f'{TAB}/ablation_rows.tex', 'w').write('\n'.join(lines) + '\n')

    proto = [('proto_centroid_unw', 'Centroids, unweighted (original)'), ('proto_centroid', 'Centroids, count-weighted'),
             ('proto_diag', 'Gaussian, diagonal'), ('hail', 'Gaussian, low-rank + diagonal (ours)'),
             ('hail_exact', 'Exact statistics (no prototypes)')]
    print('\n## 原型消融（最终 Acc / 损坏平均 / 存储 MB）')
    # 存储按各变体实际需要保存的内容重算（结果文件里只存中心的变体多算了全零方差）：
    # 共同部分（专家 + 物种分类器）= HAIL 主配置存储 - 其原型部分；原型部分按每个原型的浮点数计
    n_proto, zdim, rank = 7 * 4 * 20, 2048, 8
    per_proto = {'proto_centroid_unw': zdim + 2, 'proto_centroid': zdim + 2, 'proto_diag': 2 * zdim + 2,
                 'hail': 2 * zdim + 2 + zdim * rank}
    common = main['hail'][0]['storage_MB'] - n_proto * per_proto['hail'] * 4 / 1e6 if 'hail' in main else 0
    lines = []
    for k, name in proto:
        if k not in main:
            continue
        rs = main[k]
        a, d = ms([r['final_acc'] for r in rs]), ms([r['corrupt_mean'] for r in rs])
        st = common + n_proto * per_proto[k] * 4 / 1e6 if k in per_proto else rs[0].get('storage_MB', float('nan'))
        lines.append(f'{name} & {fmt(*a)} & {fmt(*d)} & {st:.1f} \\\\')
        print(f'| {name} | {a[0]:.2f}±{a[1]:.2f} | {d[0]:.2f}±{d[1]:.2f} | {st:.1f} |')
    open(f'{TAB}/proto_ablation_rows.tex', 'w').write('\n'.join(lines) + '\n')


# ----------------------------------------------------------------------------- 其他设置

def other_settings():
    for exp, title in (('k400', 'Kinetics 冻结视频骨干'), ('order', '5 种物种顺序'), ('session', '按天切分')):
        rr = runs(exp)
        if not rr:
            continue
        print(f'\n## {title}（最终 Acc / 平均增量 Acc / 遗忘，均值±标准差）')
        lines = []
        for m in ['finetune', 'lwf', 'ewc', 'er', 'icarl_fc', 'avcil', 'acil', 'dsal', 'aqualytic', 'mmal', 'joint', 'hail']:
            if m not in rr:
                continue
            a, b, c = (ms([r['final_acc'] for r in rr[m]]), ms([r['avg_inc_acc'] for r in rr[m]]),
                       ms([r['final_avg_forget'] for r in rr[m]]))
            lines.append(f'{label(m)} & {fmt(*a)} & {fmt(*b)} & {fmt(*c)} \\\\')
            print(f'| {NAMES.get(m, m)} | {a[0]:.2f}±{a[1]:.2f} | {b[0]:.2f}±{b[1]:.2f} | {c[0]:.2f}±{c[1]:.2f} |')
        open(f'{TAB}/{exp}_rows.tex', 'w').write('\n'.join(lines) + '\n')


# ----------------------------------------------------------------------------- 各物种 / 混淆矩阵 / 曲线

def per_species_and_curves():
    main = runs('main')
    show = [m for m in ['er', 'icarl_fc', 'avcil', 'acil', 'dsal', 'mmal', 'joint', 'hail'] if m in main]
    if not show:
        return
    print('\n## 各物种最终准确率（音视频，3 个种子平均）')
    print('| 方法 | ' + ' | '.join(SPECIES) + ' |')
    lines = []
    for m in show:
        accs = []
        for sp in range(7):
            v = [np.mean([p == l for p, l, s in zip(r['preds'], r['labels'], r['species']) if s == sp]) for r in main[m]]
            accs.append(100 * np.mean(v))
        print(f'| {NAMES[m]} | ' + ' | '.join(f'{a:.2f}' for a in accs) + ' |')
        lines.append(f'{label(m)} & ' + ' & '.join(f'{a:.2f}' for a in accs) + ' \\\\')
    open(f'{TAB}/per_species_rows.tex', 'w').write('\n'.join(lines) + '\n')

    print('\n## 各强度准确率')
    for m in show:
        accs = []
        for c in range(4):
            v = [np.mean([p == l for p, l in zip(r['preds'], r['labels']) if l == c]) for r in main[m]]
            accs.append(100 * np.mean(v))
        print(f'| {NAMES[m]} | ' + ' | '.join(f'{a:.2f}' for a in accs) + ' |')

    # 混淆矩阵：HAIL-FFIR vs 最强基线 DS-AL vs Joint
    cms = [m for m in ['dsal', 'joint', 'hail'] if m in main]
    fig, axes = plt.subplots(1, len(cms), figsize=(2.6 * len(cms), 2.5))
    for ax, m in zip(np.atleast_1d(axes), cms):
        cm = np.zeros((4, 4))
        for r in main[m]:
            for p, l in zip(r['preds'], r['labels']):
                cm[l, p] += 1
        cm = cm / cm.sum(1, keepdims=True)
        ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f'{100 * cm[i, j]:.1f}', ha='center', va='center', fontsize=7,
                        color='white' if cm[i, j] > 0.5 else 'black')
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(CLASSES, fontsize=7); ax.set_yticklabels(CLASSES, fontsize=7)
        ax.set_title(NAMES[m].replace(' (ours)', ''), fontsize=8)
        ax.set_xlabel('Predicted', fontsize=7)
    np.atleast_1d(axes)[0].set_ylabel('True', fontsize=7)
    fig.tight_layout()
    fig.savefig(f'{FIG}/confusion.pdf')
    plt.close(fig)

    # 每一步的准确率曲线：每个方法颜色、标记、线型各不相同（黑白打印也可区分），本方法加粗；图例两列放在图内左下角
    STY = {'finetune': ('#1f77b4', 'o', '-'), 'lwf': ('#ff7f0e', 's', '--'), 'ewc': ('#2ca02c', '^', '-.'),
           'er': ('#d62728', 'D', ':'), 'icarl_fc': ('#9467bd', 'v', '-.'), 'avcil': ('#7f7f7f', 'p', ':'),
           'acil': ('#8c564b', '<', '-.'), 'dsal': ('#e377c2', 'X', ':'), 'aqualytic': ('#bcbd22', '*', '-.'),
           'mmal': ('#ff9896', 'h', '--'), 'joint': ('#000000', None, '--'), 'hail': ('#00b8c8', 'P', '-.')}
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for m in ['finetune', 'lwf', 'ewc', 'er', 'icarl_fc', 'avcil', 'acil', 'dsal', 'aqualytic', 'mmal', 'joint', 'hail']:
        if m not in main:
            continue
        curve = np.mean([[h['acc_seen'] for h in r['history']] for r in main[m]], axis=0) * 100
        col, mk, ls = STY[m]
        ours = m == 'hail'
        ax.plot(range(1, 8), curve, color=col, marker=mk, ls=ls, lw=3.0 if ours else 1.6, ms=11 if ours else 7,
                mec='white' if not ours else col, mew=0.6, zorder=5 if ours else 3,
                label='Joint training' if m == 'joint' else NAMES[m])
    ax.set_xlabel('Incremental stage', fontsize=13)
    ax.set_ylabel('Accuracy on seen species (%)', fontsize=13)
    ax.set_xticks(range(1, 8)); ax.set_xlim(0.8, 7.2)
    ax.set_ylim(35, 96); ax.set_yticks(range(35, 96, 5))
    ax.tick_params(labelsize=11)
    ax.grid(ls='-.', color='#cfcfcf', lw=0.7)
    # 图例放在图内底部（4 列 3 行）：纵轴下限取 35，所有折线（最低 54.6%）都在图例上方，不被遮挡
    ax.legend(fontsize=8.6, ncol=4, loc='lower center', frameon=True, framealpha=0.95, edgecolor='#bfbfbf',
              handlelength=2.6, columnspacing=0.9, borderpad=0.5)
    fig.tight_layout()
    fig.savefig(f'{FIG}/accuracy_step.pdf', bbox_inches='tight')
    with plt.rc_context({'svg.fonttype': 'none'}):           # SVG 中文字保持为可编辑文本
        fig.savefig(f'{FIG}/accuracy_step.svg', bbox_inches='tight')
    fig.savefig(f'{FIG}/accuracy_step.png', dpi=300, bbox_inches='tight')
    fig.savefig('outputs/figures/accuracy_step.png', dpi=150, bbox_inches='tight')
    # 画图数据（各阶段平均准确率，3 个种子）
    rows = []
    for m in ['finetune', 'lwf', 'ewc', 'er', 'icarl_fc', 'avcil', 'acil', 'dsal', 'aqualytic', 'mmal', 'joint', 'hail']:
        if m in main:
            c = np.mean([[h['acc_seen'] for h in r['history']] for r in main[m]], axis=0) * 100
            rows.append(('Joint training' if m == 'joint' else NAMES[m]) + ',' + ','.join(f'{v:.2f}' for v in c))
    open(f'{FIG}/accuracy_step_data.csv', 'w').write('method,' + ','.join(f'stage{i}' for i in range(1, 8)) + '\n'
                                                     + '\n'.join(rows) + '\n')
    plt.close(fig)

    # 跨场地零样本：学 O. punctatus 之前在它上面的准确率
    print('\n## 跨场地零样本（学 O. punctatus 之前 → 之后）')
    for m in show:
        zs = [r['history'][5].get('zero_shot_next') for r in main[m]]
        after = [r['history'][6]['per_phase']['6'] if '6' in r['history'][6]['per_phase'] else r['history'][6]['per_phase'][6]
                 for r in main[m]]
        if zs[0] is not None:
            print(f'| {NAMES[m]} | {100 * np.mean(zs):.2f} → {100 * np.mean(after):.2f} |')


# ----------------------------------------------------------------------------- 原型曲线 / 敏感性

def proto_curve():
    fs = glob.glob(f'{R}/proto_curve/*.json')
    if not fs:
        return
    data = defaultdict(dict)
    for f in fs:
        d = json.load(open(f))
        data[d['args']['rank']][d['args']['m']] = d['summary']
    exact = runs('main').get('hail_exact')
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    print('\n## 原型数量曲线（m / rank / 存储MB / 干净 / 损坏）')
    for rank, col, name in ((0, '#DD8452', 'diagonal'), (8, '#4C72B0', 'low-rank ($r$=8)')):
        if rank not in data:
            continue
        ms_ = sorted(data[rank])
        st = [data[rank][m]['storage_MB'] for m in ms_]
        cl = [100 * data[rank][m]['final_acc'] for m in ms_]
        co = [100 * data[rank][m]['corrupt_mean'] for m in ms_]
        for m, a, b, c in zip(ms_, st, cl, co):
            print(f'| {m} | {rank} | {a:.1f} | {b:.2f} | {c:.2f} |')
        axes[0].plot(st, cl, marker='o', color=col, label=name)
        axes[1].plot(st, co, marker='o', color=col, label=name)
        # 低秩的标注在点上方、对角的在点下方，避免两条曲线的 m 标签互相重叠
        for m, x, y in zip(ms_, st, cl):
            axes[0].annotate(f'{m}', (x, y), fontsize=6, color=col, ha='center' if rank else 'left',
                             xytext=(0, 6) if rank else (5, -9), textcoords='offset points')
    if exact:
        ex_st = exact[0]['storage_MB'] if 'storage_MB' in exact[0] else None
        for ax, key in ((axes[0], 'final_acc'), (axes[1], 'corrupt_mean')):
            ax.axhline(100 * np.mean([r[key] for r in exact]), color='gray', ls='--',
                       label=f'exact statistics ({ex_st:.0f} MB)' if ex_st else 'exact statistics')
    for ax, t in zip(axes, ('Clean accuracy (%)', 'Accuracy under corruption (%)')):
        ax.set_xscale('log')
        ax.set_xlabel('Persistent memory (MB)')
        ax.set_ylabel(t)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, loc='lower right')
    fig.tight_layout()
    fig.savefig(f'{FIG}/prototype_count_analysis.pdf')
    plt.close(fig)


def sensitivity():
    fs = glob.glob(f'{R}/sensitivity/*.json')
    if not fs:
        return
    groups = defaultdict(list)
    for f in fs:
        name = os.path.basename(f)[:-5]
        key, val = name.rsplit('_', 1)
        s = json.load(open(f))['summary']
        groups[key].append((float(val), 100 * s['final_acc'], 100 * s['corrupt_mean']))
    titles = {'lam_s': r'$\lambda_s$', 'lam_r': r'$\lambda_r$', 'kappa': r'$\kappa$', 'prior': r'$\pi_{av}$',
              'dim': r'$D$', 'tau': r'$\tau$'}
    keys = [k for k in titles if k in groups]
    fig, axes = plt.subplots(1, len(keys), figsize=(2.0 * len(keys), 2.3))
    print('\n## 超参敏感性（值 / 干净 / 损坏）')
    for ax, k in zip(np.atleast_1d(axes), keys):
        g = sorted(groups[k])
        x = [v for v, _, _ in g]
        ax.plot(range(len(x)), [a for _, a, _ in g], marker='o', label='clean')
        ax.plot(range(len(x)), [c for _, _, c in g], marker='s', label='corrupted')
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels([f'{v:g}' for v in x], fontsize=6, rotation=45)
        ax.set_xlabel(titles[k])
        ax.grid(alpha=0.3)
        print(f'{k}: ' + ', '.join(f'{v:g}: {a:.2f}/{c:.2f}' for v, a, c in g))
    np.atleast_1d(axes)[0].set_ylabel('Accuracy (%)')
    np.atleast_1d(axes)[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(f'{FIG}/sensitivity.pdf')
    plt.close(fig)


if __name__ == '__main__':
    for fn in (table1, robustness, ablation, other_settings, per_species_and_curves, proto_curve, sensitivity):
        try:
            fn()
        except Exception as e:   # 部分结果缺失时继续生成其余内容
            print(f'[{fn.__name__}] 跳过: {type(e).__name__}: {e}', file=sys.stderr)
