"""
物种增量的音视频数据划分，跟 AquaLytic 音频实验 (dataset/fish_incremental_dataset.py)
完全一致：每个物种 x 强度内部，用 RandomState(seed) 打乱后 80/10/10 切分。

下面 INCREMENTAL_PHASES / get_wav_name_for_fish / build_opleg_phase_data / build_phase_data
逐行照搬音频代码，保证 train/val/test 的 wav 列表跟音频实验一模一样；
视频样本由 wav 路径一一映射到预处理好的 npy（目录结构和文件名跟音频对应），
因此音频、视频、音视频三种实验用的是同一批样本。
"""
import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

INCREMENTAL_PHASES = {
    0: ['Red_tilapia_2', 'Red_tilapia_3'],
    1: ['Tilapia'],
    2: ['Jade_perch'],
    3: ['Largemouth_bass'],
    4: ['Lotus_carp'],
    5: ['Sunfish'],
    6: ['Oplegnathus_punctatus'],
}

LABEL_MAP = {
    'None':   0,
    'Weak':   1,
    'Medium': 2,
    'Strong': 3,
}

# 原始数据根目录，可用环境变量 AVSIL_DATA 覆盖（目录结构见 README）
DATA_ROOT = os.environ.get('AVSIL_DATA', 'data/raw')
GZ_AUDIO_BASE = os.path.join(DATA_ROOT, 'gz/audio_dataset/')
GZ_VIDEO_BASE = os.path.join(DATA_ROOT, 'gz/video_npy/')
MANIFEST = os.environ.get('AVSIL_MANIFEST', 'data/manifest.csv')

# Oplegnathus_punctatus 使用独立数据集，结构不同
OPLEG_BASE_PATH = os.path.join(DATA_ROOT, 'opleg/audio_dataset/')
OPLEG_VIDEO_BASE = os.path.join(DATA_ROOT, 'opleg/video_npy/')
OPLEG_EXCLUDED  = {'AM_5', 'PM_5', 'AM_15', 'PM_15'}
OPLEG_LABEL_MAP = {'none': 0, 'weak': 1, 'medium': 2, 'strong': 3}


def get_wav_name_for_fish(fish_name, split,
                           base_path=GZ_AUDIO_BASE):
    audio = []
    if not os.path.exists(base_path):
        raise FileNotFoundError(f'路径不存在: {base_path}')
    for date_dir in sorted(os.listdir(base_path)):
        date_path = os.path.join(base_path, date_dir)
        if not os.path.isdir(date_path):
            continue
        wav_pattern = os.path.join(
            date_path, fish_name, split, '*.wav'
        )
        audio.extend(glob.glob(wav_pattern))
    return audio


def build_opleg_phase_data(seed=42):
    """
    Oplegnathus_punctatus 专用加载：
    结构为 {date}/{AM_40|PM_40|...}/{none|weak|medium|strong}/*.wav
    排除 AM_5, PM_5, AM_15, PM_15（鱼太少）
    """
    random_state = np.random.RandomState(seed)
    by_label = {0: [], 1: [], 2: [], 3: []}

    for date_dir in sorted(os.listdir(OPLEG_BASE_PATH)):
        date_path = os.path.join(OPLEG_BASE_PATH, date_dir)
        if not os.path.isdir(date_path):
            continue
        for subfolder in sorted(os.listdir(date_path)):
            if subfolder in OPLEG_EXCLUDED:
                continue
            sub_path = os.path.join(date_path, subfolder)
            if not os.path.isdir(sub_path):
                continue
            for intensity_name, label in OPLEG_LABEL_MAP.items():
                intensity_path = os.path.join(sub_path, intensity_name)
                if not os.path.isdir(intensity_path):
                    continue
                wavs = glob.glob(os.path.join(intensity_path, '*.wav'))
                by_label[label].extend(wavs)

    train_dict, val_dict, test_dict = [], [], []
    for label, wav_list in sorted(by_label.items()):
        intensity_name = {0:'none',1:'weak',2:'medium',3:'strong'}[label]
        random_state.shuffle(wav_list)
        total     = len(wav_list)
        train_end = int(total * 0.8)
        val_end   = train_end + int(total * 0.1)
        for wav in wav_list[:train_end]:
            train_dict.append([wav, label])
        for wav in wav_list[train_end:val_end]:
            val_dict.append([wav, label])
        for wav in wav_list[val_end:]:
            test_dict.append([wav, label])
        print(f'  Oplegnathus/{intensity_name} label={label}: '
              f'train={train_end}, val={int(total*0.1)}, test={total-val_end}')

    random_state.shuffle(train_dict)
    return train_dict, val_dict, test_dict


def build_phase_data(phase_id,
                     base_path=GZ_AUDIO_BASE,
                     seed=42):
    """
    增量鱼种版本：每个phase增加鱼种，但标签始终是4档强度。
    label 0-3: None, Weak, Medium, Strong
    Oplegnathus_punctatus(Phase 6) 使用独立数据集路径，自动分发。
    """
    if INCREMENTAL_PHASES[phase_id] == ['Oplegnathus_punctatus']:
        return build_opleg_phase_data(seed=seed)

    random_state = np.random.RandomState(seed)
    fish_list = INCREMENTAL_PHASES[phase_id]
    train_dict, val_dict, test_dict = [], [], []

    for fish in fish_list:
        for split_name, intensity in LABEL_MAP.items():
            wav_list = get_wav_name_for_fish(
                fish, split_name, base_path
            )
            if len(wav_list) == 0:
                print(f'[Warning] 未找到: {fish}/{split_name}')
                continue

            label = intensity

            random_state.shuffle(wav_list)
            total     = len(wav_list)
            train_end = int(total * 0.8)
            val_end   = train_end + int(total * 0.1)

            for wav in wav_list[:train_end]:
                train_dict.append([wav, label])
            for wav in wav_list[train_end:val_end]:
                val_dict.append([wav, label])
            for wav in wav_list[val_end:]:
                test_dict.append([wav, label])

            print(f'  {fish}/{split_name} label={label}: '
                  f'train={train_end}, '
                  f'val={int(total*0.1)}, '
                  f'test={total-val_end}')

    random_state.shuffle(train_dict)
    return train_dict, val_dict, test_dict




# ---------------------------------------------------------------------------
# 以下为音视频扩展：wav 路径 -> 视频 npy 路径，以及视频/音视频 Dataset
# ---------------------------------------------------------------------------

def wav_to_video_path(wav_path):
    if wav_path.startswith(OPLEG_BASE_PATH):
        rel = os.path.relpath(wav_path, OPLEG_BASE_PATH)
        return os.path.join(OPLEG_VIDEO_BASE, rel[:-4] + '.npy')
    rel = os.path.relpath(wav_path, GZ_AUDIO_BASE)
    return os.path.join(GZ_VIDEO_BASE, rel[:-4] + '.npy')


def species_of(wav_path):
    if wav_path.startswith(OPLEG_BASE_PATH):
        return 'Oplegnathus_punctatus'
    return os.path.relpath(wav_path, GZ_AUDIO_BASE).split(os.sep)[1]


def to_av_items(data_dict, require_video=True):
    """[[wav, label], ...] -> [{'audio','video','label','species'}, ...]；
    对应视频不存在（原始 mp4 为 0 字节）的样本在 require_video=True 时剔除。"""
    items, dropped = [], 0
    for wav, label in data_dict:
        vid = wav_to_video_path(wav)
        if require_video and not os.path.exists(vid):
            dropped += 1
            continue
        items.append({'audio': wav, 'video': vid, 'label': label, 'species': species_of(wav)})
    return items, dropped


# AquaLytic 所有实验的配置都是 Training.seed=25（不是 build_phase_data 的默认值 42）
SPLIT_SEED = 25


def build_av_phase_data(phase_id, seed=SPLIT_SEED, require_video=True):
    """返回 (train, val, test, dropped)，划分与 AquaLytic 音频实验完全相同。
    默认读 manifest 的 split_clip 列（论文实验实际使用的片段级切分）。glob 不排序，
    重新随机划分的结果依赖文件系统的列出顺序，因此只在没有 manifest 时才退回 build_av_phase_data_glob。"""
    if os.path.exists(MANIFEST):
        return build_av_phase_data_manifest(phase_id, 'split_clip', require_video)
    return build_av_phase_data_glob(phase_id, seed, require_video)


def build_av_phase_data_manifest(phase_id, split_col='split_clip', require_video=True, manifest=None):
    import csv
    species = set(INCREMENTAL_PHASES[phase_id])
    out, dropped = {'train': [], 'val': [], 'test': []}, {'train': 0, 'val': 0, 'test': 0}
    with open(manifest or MANIFEST, newline='') as f:
        for r in csv.DictReader(f):
            s = r[split_col]
            if r['species'] not in species or s not in out:
                continue
            if require_video and r['has_video'] != '1':
                dropped[s] += 1
                continue
            out[s].append({'audio': os.path.join(DATA_ROOT, r['audio_path']),
                           'video': os.path.join(DATA_ROOT, r['video_path']) if r['video_path'] else '',
                           'label': int(r['label']), 'species': r['species']})
    return out['train'], out['val'], out['test'], dropped


def build_av_phase_data_glob(phase_id, seed=SPLIT_SEED, require_video=True):
    """按 AquaLytic 的原始流程现场随机划分（结果依赖 glob 的文件顺序）。"""
    tr, va, te = build_phase_data(phase_id, base_path=GZ_AUDIO_BASE, seed=seed)
    out, dropped = [], {}
    for name, d in (('train', tr), ('val', va), ('test', te)):
        items, n = to_av_items(d, require_video)
        out.append(items)
        dropped[name] = n
    return out[0], out[1], out[2], dropped


class VideoClipDataset(Dataset):
    """读预处理好的 (16, 224, 224, 3) uint8 视频张量。增强放到 GPU 上做。"""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        clip = torch.from_numpy(np.load(it['video']))
        return clip, it['label'], i


# ---------------------------------------------------------------------------
# 按天切分（session-disjoint）：同一天的 AM/PM 场次整体分到同一个子集，
# 由 tools/split_manifest.py 生成的 manifest 列 split_s{k} 决定。只用音视频齐全的样本。
# ---------------------------------------------------------------------------


def build_av_phase_data_session(phase_id, split_col='split_s0', manifest=MANIFEST):
    import csv
    species = set(INCREMENTAL_PHASES[phase_id])
    out = {'train': [], 'val': [], 'test': []}
    with open(manifest, newline='') as f:
        for r in csv.DictReader(f):
            sp = r['species']
            if sp not in species or r['excluded'] == '1' or r['has_audio'] != '1' or r['has_video'] != '1':
                continue
            s = r[split_col]
            if s in out:
                out[s].append({'audio': os.path.join(DATA_ROOT, r['audio_path']), 'video': os.path.join(DATA_ROOT, r['video_path']), 'label': int(r['label']),
                               'species': sp})
    return out['train'], out['val'], out['test'], {'train': 0, 'val': 0, 'test': 0}


def build_phase_items(phase_id, split_mode='random', require_video=True):
    """统一入口：random = 与 AquaLytic 相同的片段级切分；session = 按天切分。"""
    if split_mode == 'session':
        return build_av_phase_data_session(phase_id)
    return build_av_phase_data(phase_id, require_video=require_video)
