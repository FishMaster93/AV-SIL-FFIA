"""完全随机的物种顺序（第一个物种也随机）：每种顺序用在其第一个物种上重新适应的编码器所提取的特征。
5 种顺序由 RandomState(2027) 生成；方法：HAIL-FFIR 与 ACIL、DS-AL、MMAL、ER、iCaRL（seed 42，超参同主实验）。
结果写入 results/all/fullorder/<method>_order<i>.json。"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, '.')
import run_all as RA

OUT = 'results/all/fullorder'
os.makedirs(OUT, exist_ok=True)
rs = np.random.RandomState(2027)
orders = [[int(k) for k in rs.permutation(7)] for _ in range(5)]
json.dump(orders, open(f'{OUT}/orders.json', 'w'))
for i, order in enumerate(orders):
    F = order[0]
    exp = f'first{F}'
    RA.FEATS[exp] = (f'audio_cnn14mv2_first{F}', f'video_s3d_first{F}')
    for method in ('hail', 'acil', 'dsal', 'mmal', 'er', 'icarl'):
        names = ['icarl_nme', 'icarl_fc'] if method == 'icarl' else [method]
        if all(os.path.exists(f'{OUT}/{n}_order{i}.json') for n in names):
            continue
        t0 = time.time()
        res = RA.run_one(method, exp, 42, order, corrupt=False)
        for v, r in res.items():
            json.dump(r, open(f'{OUT}/{v}_order{i}.json', 'w'))
            print(f'order{i} {order} {v:10s} final {100 * r["final_acc"]:.2f} avg {100 * r["avg_inc_acc"]:.2f} '
                  f'fgt {100 * r["final_avg_forget"]:.2f} | {time.time() - t0:.0f}s', flush=True)
