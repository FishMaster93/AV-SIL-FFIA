"""物种 × 真实强度 的平均带符号误差 E[y_hat − y]：共享头（无专家）vs 完整模型，3 个种子，写入 results/species_bias.json。"""
import sys, numpy as np, torch
sys.path.insert(0, '.')
from run_all import Wrapper, load
res = {}
for seed in (42, 43, 44):
    w = Wrapper('hail', seed)
    for ph in range(7):
        fd, y, _ = load('main', ph, 'train'); w.learn(fd, y)
    m = w.m
    for mode in ('shared', 'experts'):
        m.use_experts = (mode == 'experts')
        for ph in range(7):
            fd, y, _ = load('main', ph, 'test')
            p = m.predict(fd).numpy(); y = y.numpy()
            for c in range(4):
                sel = y == c
                res.setdefault((mode, ph, c), []).append((float((p[sel] - c).mean()), float((p[sel] == c).mean()), int(sel.sum())))
    m.use_experts = True
import json
out = {f'{k[0]}|{k[1]}|{k[2]}': [float(np.mean([v[0] for v in vs])), float(np.mean([v[1] for v in vs])), vs[0][2]] for k, vs in res.items()}
json.dump(out, open('results/species_bias.json', 'w'), indent=1)
for mode in ('shared', 'experts'):
    print(mode)
    for ph in range(7):
        print('  ', ph, ' '.join(f'{out[f"{mode}|{ph}|{c}"][0]:+.2f}({100*out[f"{mode}|{ph}|{c}"][1]:.0f})' for c in range(4)))
    allb = [abs(out[f'{mode}|{ph}|{c}'][0]) for ph in range(7) for c in range(4)]
    print('   mean |bias|', np.mean(allb), 'max', np.max(allb))
