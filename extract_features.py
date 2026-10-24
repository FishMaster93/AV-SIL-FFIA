"""
骨干冻结后，给 7 个阶段的全部样本一次性抽特征，后续闭式解实验只读特征文件。

  video:  S3D，--video_ckpt 为 Phase 0 微调后的 checkpoint（主实验），或 'k400' 表示
          Kinetics-400 预训练直接冻结（对照）。特征 = 最后一层特征图全局平均池化，1024 维。
  audio:  AquaLytic 的 Phase 0 音频骨干（--audio_backbone）：
            cnn10    -> PANNs-Cnn10 fc1(+ReLU) 的原始 512 维（extract_raw_feature）
            cnn14mv2 -> Cnn14-MobileNetV2 fc1(+LeakyReLU) 的 1024 维
          都不含随机投影；随机投影在闭式解阶段按 AquaLytic 同样的方式重建，保证跟音频论文等价。

划分与 AquaLytic 完全一致（datasets/fish_av_incremental.py）。音频用全部音频样本
（含少量无对应视频的），视频只有有视频的样本；两者用 wav 路径作为对齐的 key。

输出: <out_root>/<name>/phase{p}.pt = {split: {'paths', 'labels', 'species', 'feat'}}
"""
import argparse
import os
import sys
import time

import librosa
import numpy as np
import torch
from scipy.signal import resample
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets.fish_av_incremental import (DATA_ROOT, INCREMENTAL_PHASES, VideoClipDataset,
                                          build_phase_items)
from models.s3d import S3D, load_s3d_kinetics

AUDIO_FEATURES = dict(sample_rate=64000, window_size=2048, hop_size=1024,
                      mel_bins=64, fmin=1, fmax=128000)
AUDIO_PHASE0_CKPTS = {
    # AquaLytic 主结果（AudioSet 预训练 + Phase 0 微调），9.2M 参数
    'cnn10': 'workspace/PANNsCnn10_phase0/phase0_best.pt',
    # AquaLytic 的 backbone 对照（只在 Phase 0 上从零训练），6.2M 参数
    'cnn14mv2': 'workspace/Cnn14mv2_phase0/phase0_best.pt',
    # 按天切分：在按天切分的 Phase 0 训练集上重新训练（train_phase0_audio.py）
    'cnn14mv2_sess': 'workspace/Cnn14mv2_phase0_session/phase0_best.pt',
}


class AudioClipDataset(Dataset):
    """跟 AquaLytic 的 load_audio 一致：原采样率读入，再 resample 到 64k x 2s。"""

    def __init__(self, items, sr=64000):
        self.items, self.sr = items, sr

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        try:
            y, _ = librosa.load(it['audio'], sr=None)
            y = resample(y, num=self.sr * 2).astype(np.float32)
        except Exception as e:
            print(f'[Warning] 无法读取: {it["audio"]}, {e}')
            y = np.zeros(self.sr * 2, dtype=np.float32)
        return torch.from_numpy(y), it['label'], i


def build_video_model(ckpt, device):
    model = S3D(400)
    if ckpt == 'k400':
        load_s3d_kinetics(model)
    else:
        sd = torch.load(ckpt, map_location='cpu', weights_only=False)['model_state_dict']
        sd = {k[len('backbone.'):]: v for k, v in sd.items() if k.startswith('backbone.')}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        missing = [k for k in missing if not k.startswith('fc.')]
        assert not missing and not unexpected, (missing[:5], unexpected[:5])
    return model.to(device).eval()


def build_audio_model(device, backbone, ckpt_path=None):
    from models.audio_model import Audio_Frontend, AudioModel
    if backbone == 'cnn10':
        from models.panns_cnn10 import PANNS_Cnn10
        net = PANNS_Cnn10(classes_num=3)
    else:   # cnn14mv2 / cnn14mv2_sess
        from models.cnn14_mobilev2 import Cnn14_mobilev2
        net = Cnn14_mobilev2(classes_num=3)
    model = AudioModel(frontend=Audio_Frontend(**AUDIO_FEATURES), backbone=net)
    ckpt = torch.load(ckpt_path or AUDIO_PHASE0_CKPTS[backbone], map_location='cpu', weights_only=False)['model_state_dict']
    own = model.state_dict()
    loaded = {k: v for k, v in ckpt.items() if k in own and v.shape == own[k].shape}
    skipped = sorted(set(own) - set(loaded))
    # 只允许分类头 fc_audioset 没加载（形状/用途不同），骨干和前端必须全部对上
    assert all(k.startswith('backbone.fc_audioset') for k in skipped), skipped
    own.update(loaded)
    model.load_state_dict(own)
    return model.to(device).eval()


@torch.no_grad()
def extract(modality, model, items, device, batch_size, num_workers, save_fmap=False):
    ds = VideoClipDataset(items) if modality == 'video' else AudioClipDataset(items)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=True)
    feats = []
    for x, _, _ in loader:
        x = x.to(device, non_blocking=True)
        if modality == 'video':
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model(x)
            if save_fmap:
                # (B, 1024, T', H', W') -> (B, L=T', S=H'*W', d=1024)，给跨模态时空注意力用
                fm = out['fmap']
                feats.append(fm.flatten(3).permute(0, 2, 3, 1).half().cpu())
            else:
                feats.append(out['embedding'].float().cpu())
        elif hasattr(model.backbone, 'extract_raw_feature'):   # Cnn10: fc1+ReLU 的 512 维
            f = model.backbone.extract_raw_feature(model.frontend(x))
            feats.append(f.float().cpu())
        else:                                                  # Cnn14_mobilev2: fc1+LeakyReLU 的 1024 维
            feats.append(model(x)['embedding'].float().cpu())
    return torch.cat(feats)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--modality', choices=['video', 'audio'], required=True)
    p.add_argument('--audio_backbone', choices=list(AUDIO_PHASE0_CKPTS), default='cnn10')
    p.add_argument('--name', required=True, help='特征集名字，如 video_s3d_ft / video_s3d_k400 / audio_cnn10')
    p.add_argument('--video_ckpt', default='workspace/S3D_phase0_redtilapia/phase0_best.pt')
    p.add_argument('--audio_ckpt', default=None, help='覆盖 --audio_backbone 对应的默认检查点（完全随机顺序实验）')
    p.add_argument('--out_root', default='features')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=10)
    p.add_argument('--phases', type=int, nargs='*', default=list(INCREMENTAL_PHASES))
    p.add_argument('--split_mode', choices=['random', 'session'], default='random')
    p.add_argument('--save_fmap', action='store_true',
                   help='视频保存 (L, S, d) 时空特征图而不是池化向量（HAIL-FFIR 的跨模态注意力需要）')
    args = p.parse_args()

    device = 'cuda'
    model = build_video_model(args.video_ckpt, device) if args.modality == 'video' else build_audio_model(device, args.audio_backbone, args.audio_ckpt)
    out_dir = os.path.join(args.out_root, args.name)
    os.makedirs(out_dir, exist_ok=True)

    for ph in args.phases:
        out_path = os.path.join(out_dir, f'phase{ph}.pt')
        if os.path.exists(out_path):
            print(f'跳过已存在: {out_path}')
            continue
        t0 = time.time()
        tr, va, te, dropped = build_phase_items(ph, args.split_mode, require_video=(args.modality == 'video'))
        result = {}
        for split, items in (('train', tr), ('val', va), ('test', te)):
            feat = extract(args.modality, model, items, device, args.batch_size, args.num_workers,
                           args.save_fmap)
            assert len(feat) == len(items)
            result[split] = {
                'paths': [os.path.relpath(it['audio'], DATA_ROOT) for it in items],   # 样本 ID = 相对 DATA_ROOT 的音频路径
                'labels': torch.tensor([it['label'] for it in items]),
                'species': [it['species'] for it in items],
                'feat': feat.half() if args.modality == 'video' else feat,
            }
        result['meta'] = {'modality': args.modality, 'name': args.name, 'phase': ph,
                          'species': INCREMENTAL_PHASES[ph], 'dropped_no_video': dropped,
                          'video_ckpt': args.video_ckpt if args.modality == 'video' else None,
                          'fmap': bool(args.save_fmap),
                          'audio_ckpt': AUDIO_PHASE0_CKPTS[args.audio_backbone] if args.modality == 'audio' else None}
        torch.save(result, out_path)
        print(f'phase {ph} {INCREMENTAL_PHASES[ph]}: train/val/test = '
              f'{len(tr)}/{len(va)}/{len(te)}, shape {tuple(result["train"]["feat"].shape[1:])}, '
              f'{(time.time() - t0) / 60:.1f} min -> {out_path}', flush=True)


if __name__ == '__main__':
    main()
