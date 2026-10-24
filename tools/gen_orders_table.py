"""各物种顺序的结果表（补充材料）：Latex/tables/orders.tex。默认顺序 + 5 种随机顺序（第 1 个物种固定为 Red Tilapia，
因为编码器在其上适应），均为 seed 42；最后一行为 5 种随机顺序的均值±标准差。"""
import json

import numpy as np

AB = ['RT', 'TI', 'JP', 'LB', 'LC', 'SF', 'OP']
BASE = [('dsal', 'DS-AL'), ('acil', 'ACIL'), ('mmal', 'MMAL$^{*}$'), ('er', 'ER'), ('icarl_fc', 'iCaRL-FC')]
R = 'results/all'


def load(m, tag, exp):
    return json.load(open(f'{R}/{exp}/{m}_{tag}.json'))


rows = [('Default', load('hail', 'seed42', 'main'), {m: load(m, 'seed42', 'main') for m, _ in BASE})]
for i in range(5):
    rows.append((f'Random {i + 1}', load('hail', f'order{i}', 'order'), {m: load(m, f'order{i}', 'order') for m, _ in BASE}))
lines, rand = [], []
for name, h, b in rows:
    seq = '$\\to$'.join(AB[int(k)] for k in h['order'])
    v = [100 * h['final_acc'], 100 * h['avg_inc_acc'], 100 * h['final_avg_forget']] + [100 * b[m]['final_acc'] for m, _ in BASE]
    if name != 'Default':
        rand.append(v)
    lines.append(f'{name} & {seq} & ' + ' & '.join(f'{x:.2f}' for x in v) + ' \\\\')
    if name == 'Default':
        lines.append('\\midrule')
        lines.append('\\multicolumn{10}{@{}l}{\\textit{Random orders of stages 2--7 (first species fixed, same encoders)}} \\\\')
R_ = np.array(rand)
lines.append('\\midrule')
lines.append('Random (mean$\\pm$std) & -- & ' + ' & '.join(f'{m:.2f}$\\pm${s:.2f}' for m, s in zip(R_.mean(0), R_.std(0))) + ' \\\\')
# 第二部分：完全随机顺序（第一个物种也随机，编码器在该物种上重新适应），results/all/fullorder
FO = json.load(open(f'{R}/fullorder/orders.json'))
lines.append('\\midrule')
lines.append('\\multicolumn{10}{@{}l}{\\textit{Fully random orders: the first species is also randomised and the encoders are re-adapted on it}} \\\\')
full = []
for i, order in enumerate(FO):
    h = load('hail', f'order{i}', 'fullorder')
    b = {m: load(m, f'order{i}', 'fullorder') for m, _ in BASE}
    v = [100 * h['final_acc'], 100 * h['avg_inc_acc'], 100 * h['final_avg_forget']] + [100 * b[m]['final_acc'] for m, _ in BASE]
    full.append(v)
    seq = '$\\to$'.join(AB[k] for k in order)
    lines.append(f'Full {i + 1} & {seq} & ' + ' & '.join(f'{x:.2f}' for x in v) + ' \\\\')
F_ = np.array(full)
lines.append('\\midrule')
lines.append('Full (mean$\\pm$std) & -- & ' + ' & '.join(f'{m:.2f}$\\pm${s:.2f}' for m, s in zip(F_.mean(0), F_.std(0))) + ' \\\\')
cap = ('Results for each species order (audio-visual, seed 42). Top: the first species is fixed to Red Tilapia, on which the '
       'encoders are adapted, and stages 2--7 are permuted. Bottom: all seven positions are permuted, and both encoders are '
       're-adapted on the respective first species before the incremental stages, so that the variation also includes the '
       'effect of the encoders. RT: Red Tilapia; TI: Tilapia; JP: Jade Perch; LB: Largemouth Bass; LC: Lotus '
       'Carp; SF: Sunfish; OP: O.~punctatus. HAIL-FFIR: final accuracy, average incremental accuracy and forgetting; baselines: '
       'final accuracy. With fixed encoders, ACIL and MMAL are exactly order-invariant because their statistics are sums over '
       'species; DS-AL varies slightly through its compensation stream.')
tex = ('\\begin{table*}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n\\caption{' + cap + '}\n\\label{tab:orders}\n'
       '\\begin{tabular}{@{}ll ccc ccccc@{}}\n\\toprule\n'
       '\\multirow{2}{*}{Order} & \\multirow{2}{*}{Species sequence} & \\multicolumn{3}{c}{HAIL-FFIR} & \\multicolumn{5}{c}{Final accuracy of baselines} \\\\\n'
       '\\cmidrule(lr){3-5} \\cmidrule(lr){6-10}\n'
       ' & & Final & Avg & Fgt & ' + ' & '.join(n for _, n in BASE) + ' \\\\\n\\midrule\n'
       + '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n')
open('outputs/tables/orders.tex', 'w').write(tex)
print('\n'.join(lines))
