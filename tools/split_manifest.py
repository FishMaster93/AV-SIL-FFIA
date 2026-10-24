"""
给 manifest.csv 加切分列：
  split_s0 .. split_s{K-1}: 按"天"整体切分（同一天的 AM/PM 场次一起分），所有物种共用
      同一组测试日，避免同一次投喂的相邻 2 秒片段同时出现在训练和测试里。每个种子在
      大量随机的天分配里，挑每个物种的片段比例最接近 70/10/20、且每个切分都包含
      全部 4 个强度等级的方案。广州和 Opleg 分别切（两个场地日期不重叠）。
  split_random: 片段级随机 70/10/20（按物种×强度分层），只用于和旧协议对照、量化泄漏。
"""
import argparse

import numpy as np
import pandas as pd

RATIOS = {'train': 0.7, 'val': 0.1, 'test': 0.2}


def day_of(session):
    return session[:-2] if session.endswith(('AM', 'PM')) else session


def assign_days(df, rng, n_trials):
    """在 n_trials 个随机天分配里挑最优的。df 只含参与切分的行。"""
    days = np.array(sorted(df['day'].unique()))
    # 每天每个(物种,强度)的片段数，用于快速打分
    # 注意：新版 pandas 的 pivot_table 在这里会丢掉 species 层，所以用 groupby+unstack
    counts = (df.groupby(['day', 'species', 'label']).size()
                .unstack(['species', 'label'], fill_value=0).reindex(days, fill_value=0))
    species_tot = df.groupby(['day', 'species']).size().unstack('species', fill_value=0).reindex(days, fill_value=0)
    n_val = max(1, round(len(days) * RATIOS['val']))
    n_test = max(1, round(len(days) * RATIOS['test']))

    best, best_score = None, np.inf
    for _ in range(n_trials):
        perm = rng.permutation(len(days))
        test_idx, val_idx = perm[:n_test], perm[n_test:n_test + n_val]
        train_idx = perm[n_test + n_val:]
        score = 0.0
        ok = True
        for name, idx in (('train', train_idx), ('val', val_idx), ('test', test_idx)):
            # 每个切分里每个(物种,强度)都要有样本
            if (counts.iloc[idx].sum(axis=0) == 0).any():
                ok = False
                break
            frac = species_tot.iloc[idx].sum(axis=0) / species_tot.sum(axis=0)
            score = max(score, float((frac - RATIOS[name]).abs().max()))
        if ok and score < best_score:
            best_score = score
            best = {**{d: 'train' for d in days[train_idx]},
                    **{d: 'val' for d in days[val_idx]},
                    **{d: 'test' for d in days[test_idx]}}
    if best is None:
        raise RuntimeError('没有找到每个切分都覆盖全部强度的天分配，增大 n_trials')
    return best, best_score


def random_split(df, seed):
    rng = np.random.RandomState(seed)
    out = pd.Series('', index=df.index)
    for _, g in df.groupby(['species', 'label']):
        idx = g.index.to_numpy().copy()
        rng.shuffle(idx)
        n_tr = int(len(idx) * RATIOS['train'])
        n_va = int(len(idx) * RATIOS['val'])
        out[idx[:n_tr]] = 'train'
        out[idx[n_tr:n_tr + n_va]] = 'val'
        out[idx[n_tr + n_va:]] = 'test'
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', default='data/manifest.csv')
    p.add_argument('--num_seeds', type=int, default=5)
    p.add_argument('--n_trials', type=int, default=20000)
    args = p.parse_args()

    m = pd.read_csv(args.manifest, keep_default_na=False)
    m = m[[c for c in m.columns if not c.startswith('split_')]]
    m['day'] = m['session'].map(day_of)
    # 天的分配只看音频（不看视频），这样切分只由日期决定，不随视频预处理进度或个别
    # 0 字节视频而变化；每一行（包括缺某个模态的行）都按所属的天打上切分标签，
    # 由数据加载时再按 has_audio/has_video 过滤。
    assignable = (m.has_audio == 1) & (m.excluded == 0)
    usable = assignable & (m.has_video == 1)

    for s in range(args.num_seeds):
        col = f'split_s{s}'
        m[col] = ''
        rng = np.random.RandomState(1000 + s)
        for site in ['gz', 'opleg']:
            sel = assignable & (m.site == site)
            mapping, score = assign_days(m[sel], rng, args.n_trials)
            rows = (m.site == site) & (m.excluded == 0)
            m.loc[rows, col] = m.loc[rows, 'day'].map(mapping).fillna('')
            test_days = sorted(str(d) for d, v in mapping.items() if v == 'test')
            val_days = sorted(str(d) for d, v in mapping.items() if v == 'val')
            print(f'[{col}] {site}: 最大比例偏差 {score:.3f} | val 天 {val_days} | test 天 {test_days}')

    m['split_random'] = ''
    m.loc[assignable, 'split_random'] = random_split(m[assignable], seed=42)
    m = m.drop(columns=['day'])
    m.to_csv(args.manifest, index=False)

    u = m[usable]
    print(f'\n可用音视频配对 {len(u)}（总行数 {len(m)}）')
    print(pd.crosstab([u.species], u.split_s0, margins=True).to_string())


if __name__ == '__main__':
    main()
