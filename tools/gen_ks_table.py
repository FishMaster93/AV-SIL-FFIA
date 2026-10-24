"""K-S-CI 补充实验的表格：Latex/tables/ks.tex 和 results/ks/summary.json（总览页面 HTML）。

(a) 原文报告值（Pian 等，ICCV 2023，表 1，K-S-CI 列）；(b) 同一特征上重跑的解析基线（提供最终精度和特征损坏结果）；
(c) HAIL-FFIR。梯度类基线未重跑，其干净精度以原文报告值为准。
"""
import json
import os

R = 'results/ks'
CITE = '\\cite{pian2023audio}'
reported = [('Fine-tuning', True, '41.18', '89.62'), ('LwF \\cite{li2017learning}', True, '65.54', '16.55'),
            ('iCaRL-NME \\cite{rebuffi2017icarl}', False, '64.51', '18.70'),
            ('iCaRL-FC \\cite{rebuffi2017icarl}', False, '65.54', '40.57'), ('SS-IL', False, '69.71', '10.53'),
            ('AFC-NME', False, '69.13', '27.30'), ('AFC-LSC', False, '67.02', '33.56'),
            ('AV-CIL ' + CITE, False, '73.06', '6.48'), ('Oracle (upper bound)', False, '80.43', '--')]
rerun = [('acil', 'ACIL \\cite{zhuang2022acil}', True), ('dsal', 'DS-AL \\cite{zhuang2024ds}', True),
         ('mmal', 'MMAL$^{*}$ \\cite{yue2024mmal}', True)]
lines, html = [], []
fmt = lambda m, s=None: f'{100 * m:.2f}' + (f'$\\pm${100 * s:.2f}' if s is not None else '')
hfmt = lambda m, s=None: f'{100 * m:.2f}' + (f'<span class="sd">±{100 * s:.2f}</span>' if s is not None else '')


def plain(name):
    return name.split(' \\cite')[0].replace('$^{*}$', '*')


def header(tex):
    lines.append('\\midrule\n\\multicolumn{6}{@{}l}{\\textit{' + tex + '}} \\\\')
    html.append(f'<tr class="sep"><td colspan="99"><i>{tex.replace(CITE, "[AV-CIL]")}</i></td></tr>')


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


header('Reported in ' + CITE)
for name, ef, avg, fgt in reported:
    row(name, ef, ['--', avg, fgt, '--'], ['–', avg, fgt.replace('--', '–'), '–'])
header('Re-run on the same features (three seeds)')
for k, name, ef in rerun:
    if os.path.exists(f'{R}/{k}.json'):
        add(name, ef, json.load(open(f'{R}/{k}.json'))['summary'][k])
header('Ours')
hp = {}
for f, name in (('hail', 'HAIL-FFIR (prototype memory)'), ('hail_exact', 'HAIL-FFIR (exact statistics)')):
    if os.path.exists(f'{R}/{f}.json'):
        d = json.load(open(f'{R}/{f}.json'))
        hp[f] = d['best_hp']
        add(name, True, d['summary']['hail'], ours=True)

cap = ('Transfer to the class-incremental Kinetics-Sounds benchmark of ' + CITE + ' (K-S-CI; 30 classes in five tasks of six, '
       'official pre-extracted features, splits and class order). Columns as in Table~\\ref{tab:ave}. The reported rows are '
       'copied from ' + CITE + '; the analytic baselines, which ' + CITE + ' does not evaluate, and HAIL-FFIR are run on the '
       'token-averaged features with hyperparameters selected on the validation set with the criterion of Section~\\ref{sec:supp_ave} (three seeds); both HAIL-FFIR variants select $\\pi_{av}=0.95$ (Fig.~\\ref{fig:prior_sweep}). $^{*}$Re-implemented.')
tex = ('\\begin{table}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{3pt}\n\\caption{' + cap + '}\n\\label{tab:ks}\n'
       '\\begin{tabular}{@{}lccccc@{}}\n\\toprule\nMethod & Ex.-free & Final & Avg & Fgt & Corr. \\\\\n'
       + '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')
open('outputs/tables/ks.tex', 'w').write(tex)
plain_cap = cap.replace(CITE, '[AV-CIL]').replace('$^{*}$', '*').replace('Table~\\ref{tab:ave}', 'the AVE-CI table')
page = ('<figure class="ptab"><div class="tw"><table><thead><tr><th>Method</th><th>Ex.-free</th><th>Final</th><th>Avg</th>'
        '<th>Fgt</th><th>Corr.</th></tr></thead><tbody>' + ''.join(html) + '</tbody></table></div>'
        f'<figcaption><b>Supplementary table (K-S-CI). </b>{plain_cap}</figcaption></figure>')
json.dump({'html': page, 'best_hp': hp}, open(f'{R}/summary.json', 'w'))
print('\n'.join(lines))
print(hp)
