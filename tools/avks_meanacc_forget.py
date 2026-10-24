"""AVE-CI / K-S-CI 上按 AV-CIL [Pian 等, ICCV 2023] 的口径计算 Mean Acc. 与 Ave. Forget.：
  Mean Acc.   = 各增量步准确率的平均（即平均增量精度）；
  Ave. Forget.= (1/(T-1)) Σ_{t=2..T} F_t，F_t = 已学旧任务上"历史最高 − 当前"的平均（逐步遗忘再对步取平均）。
超参取各方法统一选参的结果（results/{ave,ks}/<method>.json 的 best_hp），测试集 3 个种子。结果写入 results/avks_meanacc.json。"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
import ave_cil as A
import baselines as BL

METHODS = [('acil', 'acil', 'acil'), ('dsal', 'dsal', 'dsal'), ('mmal', 'mmal', 'mmal'),
           ('hail', 'hail', 'hail'), ('hail_exact', 'hail_exact', 'hail')]
out = {}
for ds in ('ave', 'ks'):
    A.configure(ds)
    data = torch.load(A.CACHE)
    BL.C = A.N_CLS
    BL.FEAT_DIM = {'audio': data['train']['audio'].shape[1], 'video': data['train']['visual'].shape[1]}
    for name, fn, key in METHODS:
        hp = json.load(open(f'{A.OUT}/{fn}.json'))['best_hp']
        ma, af = [], []
        for seed in (42, 43, 44):
            h = A.run(name, hp, seed, data, 'test')[key]['history']
            ma.append(np.mean([x['acc'] for x in h]))
            af.append(np.mean([x['forget'] for x in h[1:]]))
        out[f'{ds}|{name}'] = {'mean_acc': [100 * float(np.mean(ma)), 100 * float(np.std(ma))],
                               'ave_forget': [100 * float(np.mean(af)), 100 * float(np.std(af))], 'hp': hp}
        print(ds, name, f"Mean Acc {out[f'{ds}|{name}']['mean_acc'][0]:.2f}  Ave. Forget {out[f'{ds}|{name}']['ave_forget'][0]:.2f}",
              flush=True)
json.dump(out, open('results/avks_meanacc.json', 'w'), indent=1)
