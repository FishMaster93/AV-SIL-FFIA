"""
扫描音频 wav 和预处理好的视频 npy，生成音视频配对清单 manifest.csv。

每行一个样本：sample_id, site, species, session, group, intensity, label, audio_path,
video_path, has_audio, has_video, excluded
  - site: gz（广州）/ opleg（Oplegnathus 独立场地）
  - session: 录制场次。广州是日期+AM/PM 目录（如 20231201AM），Opleg 是日期目录
  - group: Opleg 的 AM_40/PM_70 等子目录（广州为空）
  - excluded: Opleg 的 AM_5/PM_5/AM_15/PM_15 鱼太少，跟 AquaLytic 一样排除

视频源文件为 0 字节时预处理不会产出 npy，这里自然表现为 has_video=0。
切分（split 列）由 split_manifest.py 单独完成。
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, '.')
from datasets.fish_av_incremental import DATA_ROOT, GZ_AUDIO_BASE as GZ_AUDIO, GZ_VIDEO_BASE as GZ_VIDEO, \
    OPLEG_BASE_PATH as OPLEG_AUDIO, OPLEG_VIDEO_BASE as OPLEG_VIDEO  # noqa: E402
OPLEG_EXCLUDED = {'AM_5', 'PM_5', 'AM_15', 'PM_15'}

LABEL_MAP = {'none': 0, 'weak': 1, 'medium': 2, 'strong': 3}


def scan(root, ext):
    """返回 {(session, sub, intensity_dir, stem): path}"""
    out = {}
    if not os.path.isdir(root):
        return out
    for session in sorted(os.listdir(root)):
        p1 = os.path.join(root, session)
        if not os.path.isdir(p1):
            continue
        for sub in sorted(os.listdir(p1)):
            p2 = os.path.join(p1, sub)
            if not os.path.isdir(p2):
                continue
            for it in sorted(os.listdir(p2)):
                p3 = os.path.join(p2, it)
                if not os.path.isdir(p3):
                    continue
                for fn in os.listdir(p3):
                    if fn.endswith(ext) and not fn.endswith('.tmp' + ext):
                        out[(session, sub, it, fn[:-len(ext)])] = os.path.join(p3, fn)
    return out


def rows_for_site(site, audio_root, video_root):
    audio = scan(audio_root, '.wav')
    video = scan(video_root, '.npy')
    rows = []
    for key in sorted(set(audio) | set(video)):
        session, sub, it, stem = key
        if site == 'gz':
            species, group, excluded = sub, '', 0
        else:
            species, group = 'Oplegnathus_punctatus', sub
            excluded = int(sub in OPLEG_EXCLUDED)
        rows.append({
            'sample_id': f'{site}/{session}/{sub}/{it}/{stem}',
            'site': site,
            'species': species,
            'session': session,
            'group': group,
            'intensity': it.capitalize(),
            'label': LABEL_MAP[it.lower()],
            'audio_path': os.path.relpath(audio[key], DATA_ROOT) if key in audio else '',   # 相对 DATA_ROOT
            'video_path': os.path.relpath(video[key], DATA_ROOT) if key in video else '',
            'has_audio': int(key in audio),
            'has_video': int(key in video),
            'excluded': excluded,
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='data/manifest.csv')
    args = p.parse_args()

    rows = rows_for_site('gz', GZ_AUDIO, GZ_VIDEO) + rows_for_site('opleg', OPLEG_AUDIO, OPLEG_VIDEO)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'写出 {len(rows)} 行 -> {args.out}')


if __name__ == '__main__':
    main()
