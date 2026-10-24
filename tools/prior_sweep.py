"""π_av 取舍曲线（类增量基准）：固定其他超参（λ_s=10, λ_r=30, m=20, 无专家），π_av 取 0.5–1.0，
记录验证集干净精度与测试集干净/损坏精度（3 个种子）。结果写入 results/<ds>/prior_sweep.json。"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
import ave_cil as A
import baselines as BL

for ds in sys.argv[1:] or ['ave', 'ks']:
    A.configure(ds)
    data = torch.load(A.CACHE)
    BL.C = A.N_CLS
    BL.FEAT_DIM = {'audio': data['train']['audio'].shape[1], 'video': data['train']['visual'].shape[1]}
    stats = {m: (data['train'][m].mean(0), data['train'][m].std(0)) for m in ('audio', 'visual')}
    out = {}
    for st in ('proto', 'exact'):
        # 其余超参固定为该基准上统一选参选出的值，只改 π_av
        base = json.load(open(f'{A.OUT}/{"hail_exact" if st == "exact" else "hail"}.json'))['best_hp']
        for pa in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0):
            hp = dict(base, prior_av=pa, stats=st)
            val = A.run('hail', hp, 42, data, 'val')['hail']['avg_inc_acc']
            rs = [A.run('hail', hp, s, data, 'test', corrupt=True, stats=stats)['hail'] for s in (42, 43, 44)]
            g = lambda k: float(100 * np.mean([r[k] for r in rs]))
            out[f'{st}_{pa}'] = dict(stats=st, prior_av=pa, base=base, val_avg=100 * val, final=g('final_acc'),
                                     avg=g('avg_inc_acc'), corrupt=g('corrupt_mean'))
            print(ds, out[f'{st}_{pa}'], flush=True)
    json.dump(out, open(f'{A.OUT}/prior_sweep.json', 'w'), indent=1)
