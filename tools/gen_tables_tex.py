"""把 results/ 中的结果写成可直接 \\input 的 LaTeX 表格文件（Latex/tables/*.tex），避免手抄数字。"""
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, '.')
from make_results import CITE, NAMES, runs, ms, fmt  # noqa: E402

R = 'results'
OUT = 'outputs/tables'
os.makedirs(OUT, exist_ok=True)


def label(k):
    n = NAMES.get(k, k).replace(' (ours)', '')
    if k == 'hail':
        return '\\textbf{HAIL-FFIR (ours)}'
    return f'{n} \\cite{{{CITE[k]}}}' if k in CITE else n


def rows_file(name):
    return open(f'{R}/tables/{name}_rows.tex').read().strip()


def table1():
    body = rows_file('table1')
    tex = r'''\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{3.5pt}
\caption{Species-incremental FFIR on AV-SIL-FFIR (clip-level split, seven stages). Final: accuracy on all species after the last stage; Avg: average incremental accuracy; Fgt: average forgetting (all in \%). Mean$\pm$std over three seeds; best exemplar-free or exemplar-based result in bold (Joint training excluded). Joint training (shared head) retrains a single classifier on all data seen so far and is the upper bound of the shared-head methods; Joint training (HAIL-FFIR model) fits the same model as HAIL-FFIR (shared heads, species experts anchored on the final shared heads, reliability weighting) on all data seen so far. AV-CIL and MMAL are audio-visual methods: with a single modality MMAL reduces to ACIL and the cross-modal distillation of AV-CIL is undefined (--). $^{*}$Re-implemented on our frozen features.}
\label{tab:AV result}
\begin{tabular}{@{}lc ccc ccc ccc@{}}
\toprule
\multirow{2}{*}{Method} & \multirow{2}{*}{Ex.-free} & \multicolumn{3}{c}{Audio} & \multicolumn{3}{c}{Visual} & \multicolumn{3}{c}{Audio-Visual} \\
\cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11}
 & & Final$\uparrow$ & Avg$\uparrow$ & Fgt$\downarrow$ & Final$\uparrow$ & Avg$\uparrow$ & Fgt$\downarrow$ & Final$\uparrow$ & Avg$\uparrow$ & Fgt$\downarrow$ \\
\midrule
''' + body.replace('\\textbf{HAIL-FFIR (Ours)}', 'HAIL-FFIR (ours)') + r'''
\bottomrule
\end{tabular}
\end{table*}
'''
    # Joint 行前加分隔线
    tex = tex.replace('Joint (shared head)', '\\midrule\nJoint training (shared head)', 1)
    tex = tex.replace('Joint (HAIL-FFIR model)', 'Joint training (HAIL-FFIR model)', 1)
    open(f'{OUT}/table1.tex', 'w').write(tex)


def robustness():
    body = rows_file('robustness').replace('HAIL-FFIR (ours)', '\\textbf{HAIL-FFIR (ours)}')
    tex = r'''\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{3.5pt}
\caption{Robustness to modality corruption (audio-visual input, final accuracy in \% after all seven stages, mean over three seeds). Corruptions are applied to the test inputs only. Each noise, blur, occlusion and turbidity column averages three severity levels (audio SNRs of 10, 0 and $-10$\,dB; video noise $\sigma$ of 20, 40 and 80; blur $\sigma$ of 2, 4 and 8 pixels; occlusion of 25, 50 and 75\%; turbidity ratios of 0.5, 0.7 and 0.9); a missing modality has a single level. Hyperparameters of all methods were selected with noise and missing modalities on the validation set; blur, occlusion and turbidity (``Held-out'', nine conditions) were never used for selection. The last column averages all 17 corrupted conditions.}
\label{tab:robust}
\begin{tabular}{@{}l c cc cccc c c c@{}}
\toprule
Method & Clean & \begin{tabular}{@{}c@{}}Audio\\noise\end{tabular} & \begin{tabular}{@{}c@{}}Audio\\missing\end{tabular} & \begin{tabular}{@{}c@{}}Video\\noise\end{tabular} & Blur & Occlusion & Turbidity & \begin{tabular}{@{}c@{}}Video\\missing\end{tabular} & \begin{tabular}{@{}c@{}}Held-out\\avg.\end{tabular} & \begin{tabular}{@{}c@{}}Corrupted\\avg.\end{tabular} \\
\midrule
''' + body + r'''
\bottomrule
\end{tabular}
\end{table*}
'''
    open(f'{OUT}/robustness.tex', 'w').write(tex)


def ablation():
    tex = r'''\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{2.5pt}
\caption{Ablation of the components of HAIL-FFIR (audio-visual, \%). SE: species-conditional experts; RW: reliability weighting of the three heads; RR: reliability-aware routing. ``Corr.'' is the mean accuracy over the 17 corrupted conditions.}
\label{tab:ablation}
\begin{tabular}{@{}lccc cccc@{}}
\toprule
Variant & SE & RW & RR & Final & Avg & Fgt & Corr. \\
\midrule
''' + rows_file('ablation') + r'''
\bottomrule
\end{tabular}
\end{table*}
'''
    open(f'{OUT}/ablation.tex', 'w').write(tex)
    tex = r'''\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\caption{Effect of the prototype model on prototype-enhanced analytic training ($m=20$ prototypes per species and intensity). Memory is the persistent storage of the whole model.}
\label{tab:proto}
\begin{tabular}{@{}lccc@{}}
\toprule
Statistics of earlier species & Clean & Corr. & Memory (MB) \\
\midrule
''' + rows_file('proto_ablation') + r'''
\bottomrule
\end{tabular}
\end{table}
'''
    open(f'{OUT}/proto.tex', 'w').write(tex)


def protocols():
    exps = {e: runs(e) for e in ('k400', 'session', 'order')}
    methods = ['finetune', 'lwf', 'ewc', 'er', 'icarl_fc', 'avcil', 'acil', 'dsal', 'aqualytic', 'mmal', 'joint', 'hail']
    lines = []
    for m in methods:
        cells = []
        for e, keys in (('k400', ('final_acc', 'final_avg_forget')), ('session', ('final_acc', 'avg_inc_acc', 'final_avg_forget')),
                        ('order', ('final_acc', 'avg_inc_acc'))):
            rr = exps[e].get(m)
            for k in keys:
                if rr is None:
                    cells.append('--')
                else:
                    a = ms([r[k] for r in rr])
                    cells.append(fmt(*a) if k != 'final_avg_forget' else fmt(a[0]))
        lines.append(f'{label(m)} & ' + ' & '.join(cells) + ' \\\\')
    tex = r'''\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\caption{Additional protocols (audio-visual, \%). Frozen Kinetics S3D: the visual encoder is not adapted to fish at all. Session-disjoint: all clips of a recording day belong to the same subset and the encoders are re-trained on the session-disjoint training set of the first species. Random orders: mean$\pm$std over five random orders of stages 2--7.}
\label{tab:protocols}
\begin{tabular}{@{}l cc ccc cc@{}}
\toprule
\multirow{2}{*}{Method} & \multicolumn{2}{c}{Frozen Kinetics S3D} & \multicolumn{3}{c}{Session-disjoint split} & \multicolumn{2}{c}{Random species orders} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-6} \cmidrule(lr){7-8}
 & Final & Fgt & Final & Avg & Fgt & Final & Avg \\
\midrule
''' + '\n'.join(lines) + r'''
\bottomrule
\end{tabular}
\end{table*}
'''
    open(f'{OUT}/protocols.tex', 'w').write(tex)


def per_species():
    tex = r'''\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{5pt}
\caption{Final accuracy (\%) of the audio-visual models on each species after all seven stages (mean over three seeds).}
\label{tab:species}
\begin{tabular}{@{}l ccccccc@{}}
\toprule
Method & Red Tilapia & Tilapia & Jade Perch & Largemouth Bass & Lotus Carp & Sunfish & O. punctatus \\
\midrule
''' + rows_file('per_species').replace('HAIL-FFIR (ours)', '\\textbf{HAIL-FFIR (ours)}') + r'''
\bottomrule
\end{tabular}
\end{table*}
'''
    open(f'{OUT}/species.tex', 'w').write(tex)


def cost():
    c = json.load(open(f'{R}/cost.json'))
    h = c['heads']
    rows = [('ER', 'er', '16.4$^{\\dagger}$'), ('iCaRL', 'icarl', '16.4$^{\\dagger}$'), ('AV-CIL$^{*}$', 'avcil', '16.4$^{\\dagger}$'),
            ('ACIL', 'acil', None), ('DS-AL', 'dsal', None), ('MMAL$^{*}$', 'mmal', None),
            ('HAIL-FFIR, exact statistics', 'hail_exact', None), ('\\textbf{HAIL-FFIR (ours)}', 'hail', None)]
    exact_mem = {'acil': (16385 ** 2 + 16385 * 4) * 4 / 1e6, 'dsal': 2 * (16385 ** 2 + 16385 * 4) * 4 / 1e6,
                 'mmal': (16385 ** 2 + 2 * 8193 ** 2 + 3 * 8193 * 4) * 4 / 1e6}
    lines = []
    for name, k, mem in rows:
        v = h[k]
        if mem is None:
            mem = f"{v['storage_MB']:.1f}" if 'storage_MB' in v else f'{exact_mem[k]:.1f}'
        lines.append(f"{name} & {v['learn_s_per_stage_mean']:.2f} & {v['head_gpu_bs1_ms']:.2f} & {mem} \\\\")
    tex = r'''\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{3pt}
\caption{Computational cost. Encoders (shared by all methods): S3D %.1fM parameters / %.1f GFLOPs per clip, PANNs MobileNetV2 %.1fM / %.2f GFLOPs; %.1f\,ms per clip on GPU at batch size 1. The table lists the time to learn one new species (s), the latency of the incremental classifier per clip (ms, batch size 1) and the persistent memory (MB). $^{\dagger}$2,000 stored feature vectors; storing 2,000 raw clips would require about %.1f\,GB.}
\label{tab:cost}
\begin{tabular}{@{}lccc@{}}
\toprule
Method & Learn / species & Latency & Memory \\
\midrule
''' % (c['params_M']['S3D'], c['GFLOPs_per_clip']['S3D'], c['params_M']['Cnn14-MobileNetV2'],
       c['GFLOPs_per_clip']['Cnn14-MobileNetV2'], c['latency']['encoders_gpu_bs1_ms_per_clip'],
       c['storage_reference_MB']['exemplar_2000_raw_clips_npy_video_plus_wav'] / 1000) + '\n'.join(lines) + r'''
\bottomrule
\end{tabular}
\end{table}
'''
    open(f'{OUT}/cost.tex', 'w').write(tex)


if __name__ == '__main__':
    for fn in (table1, robustness, ablation, protocols, per_species, cost):
        fn()
    print('written:', sorted(os.listdir(OUT)))


def calibration():
    c = json.load(open(f'{R}/calibration.json'))
    order = ['finetune', 'lwf', 'ewc', 'er', 'icarl_fc', 'avcil', 'acil', 'dsal', 'aqualytic', 'mmal', 'joint', 'hail']
    lines = []
    for m in order:
        rs = c[m]
        g = lambda k: np.mean([r['cal'][k] for r in rs])
        raw = f"{100 * np.mean([r['raw']['ece'] for r in rs]):.2f}" if 'raw' in rs[0] else '--'
        lines.append(f"{label(m)} & {100 * g('acc'):.2f} & {raw} & {100 * g('ece'):.2f} & {g('nll'):.3f} & {g('brier'):.3f} \\\\")
    tex = r'''\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{3pt}
\caption{Calibration on all seven species after the last stage (audio-visual). All outputs are temperature-scaled on the validation set; ECE (\%, 15 bins), NLL and Brier score are computed on the test set. ``Raw'' is the ECE of the uncalibrated softmax for gradient-trained methods. Accuracy equals the final accuracy in Table~\ref{tab:AV result}; for AquaLytic the prediction follows its CORAL decoding (number of exceeded thresholds) and the confidence is the probability of the predicted level.}
\label{tab:calib}
\begin{tabular}{@{}lccccc@{}}
\toprule
Method & Acc & ECE (raw) & ECE & NLL & Brier \\
\midrule
''' + '\n'.join(lines) + r'''
\bottomrule
\end{tabular}
\end{table}
'''
    open(f'{OUT}/calibration.tex', 'w').write(tex.replace('AV-CIL* ', 'AV-CIL$^{*}$ ').replace('MMAL* ', 'MMAL$^{*}$ '))


if __name__ == '__main__':
    calibration()
