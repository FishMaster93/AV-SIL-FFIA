"""AVE-CI / K-S-CI 上 HAIL-FFIR 的实际持久存储（float32），逐项统计：原型、共享头、专家、物种分类器、精确统计量（Gram 矩阵 A 与 B）。
超参取统一选参结果（专家在两个基准上都被验证集关闭，预测只用共享头）。seed 42。结果写入 results/avks_storage.json。"""
import json
import sys

import torch

sys.path.insert(0, '.')
import ave_cil as A
import baselines as BL
import hail_ffir as HF

out = {}
for ds in ('ave', 'ks'):
    A.configure(ds)
    data = torch.load(A.CACHE)
    BL.C = A.N_CLS
    dims = (data['train']['audio'].shape[1], data['train']['visual'].shape[1])
    for fn in ('hail', 'hail_exact'):
        hp = json.load(open(f'{A.OUT}/{fn}.json'))['best_hp']
        L = A.Learner('hail', hp, 42, dims)
        for t in range(A.N_TASK):
            idx = A.task_of(data['train']['y']) == t
            L.learn(data['train']['audio'][idx], data['train']['visual'][idx], data['train']['y'][idx])
        m = L.m
        B = 4 / 1e6
        protos = sum(c[0].numel() + c[1].numel() + 1 + (c[4].numel() if c[4] is not None else 0) for cells in m.protos for c in cells) * B
        shared = sum(W.numel() for W in m.shared.values()) * B
        experts = sum(E.numel() for h in m.heads for E in m.experts[h]) * B
        wsp = sum(m.deploy[h]['Wsp'].numel() for h in m.heads) * B
        exact = sum(m.exact[h][0].numel() + m.exact[h][1].numel() + m.exact[h][2].numel() for h in m.heads) * B if m.stats == 'exact' else 0.0
        used = shared + (experts + wsp if m.use_experts else 0.0)
        persistent = (protos if m.stats == 'proto' else exact) + used
        r = dict(stats=m.stats, experts_used=m.use_experts, n_prototypes=sum(len(c) for c in m.protos), prototypes_MB=protos,
                 shared_heads_MB=shared, experts_MB=experts, species_classifier_MB=wsp, exact_stats_MB=exact,
                 persistent_MB=persistent, storage_bytes_fn_MB=m.storage_bytes() / 1e6)
        out[f'{ds}|{fn}'] = r
        print(ds, fn, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        del L, m
        torch.cuda.empty_cache()
json.dump(out, open('results/avks_storage.json', 'w'), indent=1)
