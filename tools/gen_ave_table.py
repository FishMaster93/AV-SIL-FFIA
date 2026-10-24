"""AVE-CI 补充实验的表格：Latex/tables/ave.tex（补充材料）和 results/ave/summary.json（总览页面 HTML）。

三部分：(a) 原文报告值（Pian 等，ICCV 2023，表 1；只有平均增量精度和遗忘）；
(b) 同一特征上重跑的基线（提供最终精度和特征损坏结果，以及原文没有的解析方法）；(c) HAIL-FFIR。
"""
import json

R = 'results/ave'
CITE = '\\cite{pian2023audio}'
# (a) 原文表 1，AVE-CI 列：Mean Acc.（= 平均增量精度）、Ave. Forget.
reported = [('Fine-tuning', True, '42.40', '70.99'), ('LwF \\cite{li2017learning}', True, '58.07', '26.90'),
            ('iCaRL-NME \\cite{rebuffi2017icarl}', False, '56.15', '11.71'),
            ('iCaRL-FC \\cite{rebuffi2017icarl}', False, '65.88', '26.08'), ('SS-IL', False, '61.94', '22.49'),
            ('AFC-NME', False, '68.46', '14.18'), ('AFC-LSC', False, '65.21', '28.11'),
            ('AV-CIL ' + CITE, False, '74.04', '7.63'), ('Oracle (upper bound)', False, '76.85', '--')]
# (b) 重跑
rerun = [('finetune', 'Fine-tuning', True), ('lwf', 'LwF \\cite{li2017learning}', True),
         ('ewc', 'EWC \\cite{Kirkpatrick2017overcoming}', True), ('er', 'ER', False),
         ('icarl_nme', 'iCaRL-NME \\cite{rebuffi2017icarl}', False), ('icarl_fc', 'iCaRL-FC \\cite{rebuffi2017icarl}', False),
         ('avcil', 'AV-CIL$^{*}$ ' + CITE, False), ('acil', 'ACIL \\cite{zhuang2022acil}', True),
         ('dsal', 'DS-AL \\cite{zhuang2024ds}', True), ('mmal', 'MMAL$^{*}$ \\cite{yue2024mmal}', True),
         ('joint', 'Joint training (MLP)', False), ('joint_ridge', 'Joint training (ridge)', False)]
files = {'icarl_nme': 'icarl', 'icarl_fc': 'icarl'}
lines, html = [], []
fmt = lambda m, s=None: f'{100 * m:.2f}' + (f'$\\pm${100 * s:.2f}' if s is not None else '')
hfmt = lambda m, s=None: f'{100 * m:.2f}' + (f'<span class="sd">±{100 * s:.2f}</span>' if s is not None else '')


def plain(name):
    return name.split(' \\cite')[0].replace('$^{*}$', '*')


def header(tex, text):
    lines.append('\\midrule\n\\multicolumn{6}{@{}l}{\\textit{' + tex + '}} \\\\')
    html.append(f'<tr class="sep"><td colspan="99"><i>{text}</i></td></tr>')


def row(name, ef, nums, hnums, ours=False):
    label = '\\textbf{' + name + '}' if ours else name
    lines.append(' & '.join([label, '\\Checkmark' if ef else '\\XSolidBrush'] + nums) + ' \\\\')
    tr = '<tr class="ours">' if ours else '<tr>'
    html.append(tr + f'<td>{plain(name)}</td><td>{"✓" if ef else "✗"}</td>'
                + ''.join(f'<td class="n">{h}</td>' for h in hnums) + '</tr>')


def add(name, ef, s, ours=False):
    keys = ('final_acc', 'avg_inc_acc', 'final_forget', 'corrupt_mean')
    row(name, ef, [fmt(*s[keys[0]])] + [fmt(s[k][0]) for k in keys[1:]],
        [hfmt(*s[keys[0]])] + [hfmt(s[k][0]) for k in keys[1:]], ours)


header('Reported in ' + CITE, 'Reported in [AV-CIL]')
for name, ef, avg, fgt in reported:
    row(name, ef, ['--', avg, fgt, '--'], ['–', avg, fgt.replace('--', '–'), '–'])
header('Re-run on the same features (three seeds)', 'Re-run on the same features (three seeds)')
for k, name, ef in rerun:
    add(name, ef, json.load(open(f'{R}/{files.get(k, k)}.json'))['summary'][k])
header('Ours', 'Ours')
add('HAIL-FFIR (prototype memory)', True, json.load(open(f'{R}/hail.json'))['summary']['hail'], ours=True)
add('HAIL-FFIR (exact statistics)', True, json.load(open(f'{R}/hail_exact.json'))['summary']['hail'], ours=True)

cap = ('Transfer to the class-incremental AVE benchmark of ' + CITE + ' (28 classes in four tasks of seven, official '
       'pre-extracted features, splits and class order, exemplar budget 340). Final: accuracy on all classes after the last task; '
       'Avg: average incremental accuracy (``Mean Acc.\'\' in ' + CITE + '); Fgt: forgetting; Corr.: mean accuracy under '
       'feature-level corruptions (each modality zeroed, or Gaussian noise of 0.5/1/2 times the feature standard deviation). '
       'The reported rows are copied from ' + CITE + ', whose models train fusion and attention modules on token-level features '
       'of the same frozen encoders; their forgetting is averaged over incremental steps, ours is measured after the last task. '
       'The re-run rows use token-averaged features with hyperparameters, including the number of epochs, selected on the '
       'validation set with one criterion for all methods (mean of clean and corrupted validation accuracy); the attention of AV-CIL cannot operate on averaged features, which explains the gap of AV-CIL$^{*}$ to '
       'its reported value. Joint training (MLP) retrains the head of the gradient-based methods on all data seen so far; '
       'Joint training (ridge) solves the ridge regression of the analytic methods on all data at once and coincides with ACIL, '
       'confirming that the recursive solution equals joint training. For HAIL-FFIR the validation set disabled the task experts, '
       'because tasks have disjoint classes; the exact-statistics variant stores Gram matrices instead of prototypes; both variants, including the prior weight $\\pi_{av}$ (0.95), are selected on the validation set (Fig.~\\ref{fig:prior_sweep}). '
       '$^{*}$Re-implemented.')
tex = ('\\begin{table}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{3pt}\n\\caption{' + cap + '}\n\\label{tab:ave}\n'
       '\\begin{tabular}{@{}lccccc@{}}\n\\toprule\nMethod & Ex.-free & Final & Avg & Fgt & Corr. \\\\\n'
       + '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')
open('outputs/tables/ave.tex', 'w').write(tex)
plain_cap = cap.replace(CITE, '[AV-CIL]').replace('$^{*}$', '*').replace('``', '"').replace("''", '"')
page = (
    '<p class="zh">补充材料用（AVE-CI 与 K-S-CI，下面两张表和 π_av 取舍图）。统一选参标准：½(验证集干净平均增量 + 验证集特征损坏)，'
    '所有方法相同。类增量下任务类别互不重叠，验证集关闭任务专家；两个基准都选 π_av=0.95。'
    '平均增量：HAIL-FFIR 精确统计 79.2 / 79.4（原文 AV-CIL 74.0 / 73.1，ACIL 74.7 / 77.1）；最终精度 76.2 / 70.5 为重跑方法最高。'
    '特征损坏：61.3 / 56.2（ACIL 48.2 / 41.0，MMAL 52.6 / 44.5）。解析基线在新标准下选 λ=100（干净标准下为 10），'
    '用更强正则换鲁棒性；HAIL 的鲁棒性来自可靠性加权，保持 λ_s=10。原型相对精确统计低约 3 个点。</p>'
    '<figure class="ptab"><div class="tw"><table><thead><tr><th>Method</th><th>Ex.-free</th><th>Final</th><th>Avg</th>'
    '<th>Fgt</th><th>Corr.</th></tr></thead><tbody>' + ''.join(html) + '</tbody></table></div>'
    f'<figcaption><b>Supplementary table. </b>{plain_cap}</figcaption></figure>')
json.dump({'html': page}, open(f'{R}/summary.json', 'w'))
print('\n'.join(lines))
