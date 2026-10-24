"""仿 Pian 等 [ICCV 2023] 表 1 的对比表：基线全部取原文报告值（Mean Acc. / Ave. Forget.），最下方为 HAIL-FFIR。
HAIL-FFIR 的 Ave. Forget. 按原文定义（逐步遗忘再对步取平均），数据 results/avks_meanacc.json。
加粗：全表最好（不含 Oracle）；下划线：基线中最好。输出 Latex/tables/ave_ks_reported.tex。"""
import json

d = json.load(open('results/avks_meanacc.json'))
BASE = [('Fine-tuning', 42.40, 70.99, 41.18, 89.62), ('LwF \\cite{li2017learning}', 58.07, 26.90, 65.54, 16.55),
        ('iCaRL-NME \\cite{rebuffi2017icarl}', 56.15, 11.71, 64.51, 18.70),
        ('iCaRL-FC \\cite{rebuffi2017icarl}', 65.88, 26.08, 65.54, 40.57),
        ('SS-IL \\cite{ahn2021ssil}', 61.94, 22.49, 69.71, 10.53), ('AFC-NME \\cite{kang2022class}', 68.46, 14.18, 69.13, 27.30),
        ('AFC-LSC \\cite{kang2022class}', 65.21, 28.11, 67.02, 33.56), ('AV-CIL \\cite{pian2023audio}', 74.04, 7.63, 73.06, 6.48)]
# MMAL [Yue 等, ACM MM 2024] 原文报告值：拿到后填入 (AVE Mean Acc, AVE Ave.Forget, KS Mean Acc, KS Ave.Forget)，None 表示不加这一行
MMAL_REPORTED = (76.71, None, 78.98, None)   # MMAL 表中 Acc 即平均增量精度；其遗忘指标 PD 与 Ave. Forget. 定义不同，不填
if MMAL_REPORTED:
    BASE.append(('MMAL \\cite{yue2024mmal}',) + tuple(MMAL_REPORTED))
OURS = [('\\textbf{HAIL-FFIR (ours)}', 'hail')]          # 本方法 = 原型记忆版本（精确统计量仅为对照，不放入此表）
ours = [(n, [d[f'ave|{k}']['mean_acc'][0], d[f'ave|{k}']['ave_forget'][0], d[f'ks|{k}']['mean_acc'][0],
             d[f'ks|{k}']['ave_forget'][0]]) for n, k in OURS]
base = [(n, list(v)) for n, *v in BASE]
pick = [max, min, max, min]
best_all = [f([r[1][j] for r in base + ours if r[1][j] is not None]) for j, f in enumerate(pick)]
best_base = [f([r[1][j] for r in base if r[1][j] is not None]) for j, f in enumerate(pick)]


def cell(v, j, is_base):
    if v is None:
        return '--'
    s = f'{v:.2f}'
    if abs(v - best_all[j]) < 1e-9:
        s = f'\\textbf{{{s}}}'
    if is_base and abs(v - best_base[j]) < 1e-9:
        s = f'\\underline{{{s}}}'
    return s


lines = [n + ' & ' + ' & '.join(cell(v, j, True) for j, v in enumerate(vals)) + ' \\\\' for n, vals in base]
lines.append('\\midrule')
lines += [n + ' & ' + ' & '.join(cell(v, j, False) for j, v in enumerate(vals)) + ' \\\\' for n, vals in ours]
cap = ('Comparison on the class-incremental AVE-CI and K-S-CI benchmarks of \\cite{pian2023audio}. All results except '
       'HAIL-FFIR and MMAL are those reported in \\cite{pian2023audio}. HAIL-FFIR follows the same splits, class order and exemplar-free '
       'setting and uses the released pre-extracted features of the same frozen encoders, averaged over tokens; its forgetting is '
       'computed as in \\cite{pian2023audio} (average over incremental steps), and results are averaged over three seeds. MMAL '
       'results are those reported in \\cite{yue2024mmal}, which measures forgetting with a different metric (performance drop), '
       'hence the missing entries. HAIL-FFIR stores Gaussian prototypes and its shared heads, 15.6\\,MB on AVE-CI and 37.8\\,MB on '
       'K-S-CI, instead of the 1.62\\,GB of Gram matrices required by exact statistics. Bold: best overall; underlined: best of the reported methods.')
tex = ('\\begin{table}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{3.5pt}\n\\caption{' + cap + '}\n\\label{tab:avebench_reported}\n'
       '\\begin{tabular}{@{}l cc cc@{}}\n\\toprule\n'
       '\\multirow{2}{*}{Method} & \\multicolumn{2}{c}{AVE-CI} & \\multicolumn{2}{c}{K-S-CI} \\\\\n'
       '\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}\n'
       ' & Mean Acc. & Ave.~Forget. & Mean Acc. & Ave.~Forget. \\\\\n\\midrule\n'
       + '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')
open('outputs/tables/ave_ks_reported.tex', 'w').write(tex)
print('\n'.join(lines))
