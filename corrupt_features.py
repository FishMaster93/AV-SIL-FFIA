"""
模态损坏鲁棒性：在原始输入（视频帧 / 音频波形）上施加损坏，经冻结的编码器和 Phase 0 融合模块
重新计算测试集特征。训练数据保持干净（模拟"训练时环境正常、部署后水变浑/噪声变大"）。

视频损坏（uint8 帧上，每个样本用固定种子，可复现）：
  v_noise_{1,2,3}  高斯噪声 σ = 20 / 40 / 80（0-255 尺度）
  v_blur_{1,2,3}   高斯模糊 σ = 2 / 4 / 8 像素（失焦、水雾）
  v_occl_{1,2,3}   随机位置黑色方块遮挡 25% / 50% / 75% 面积（反光、遮挡物），各帧同一位置
  v_haze_{1,2,3}   与浑水颜色混合 t = 0.5 / 0.7 / 0.9（浑浊）
  v_missing        视频整段缺失（全黑）
音频损坏（波形上）：
  a_noise_{1,2,3}  白噪声 SNR = 10 / 0 / -10 dB
  a_missing        音频整段缺失（静音）

输出 features/corrupt/<cond>/phase{p}.pt = {特征集名: {'paths','labels','species','feat'}}（只有 test），
特征集与干净实验同名：video_s3d_ft, audio_cnn14mv2, hail_av_<tag>, hail_v_<tag>, hail_a_<tag>
"""
import argparse
import os
import sys
import time

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets.fish_av_incremental import DATA_ROOT, INCREMENTAL_PHASES, VideoClipDataset, build_av_phase_data
from extract_features import AudioClipDataset, build_audio_model, build_video_model
from analytic_cil import FEAT_ROOT, load_phase
from train_phase0_fusion import FusionWithHead

VIDEO_CONDS = {
    **{f'v_noise_{i + 1}': ('noise', s) for i, s in enumerate([20, 40, 80])},
    **{f'v_blur_{i + 1}': ('blur', s) for i, s in enumerate([2, 4, 8])},
    **{f'v_occl_{i + 1}': ('occl', s) for i, s in enumerate([0.25, 0.5, 0.75])},
    **{f'v_haze_{i + 1}': ('haze', s) for i, s in enumerate([0.5, 0.7, 0.9])},
    'v_missing': ('missing', None),
}
AUDIO_CONDS = {
    **{f'a_noise_{i + 1}': ('noise', s) for i, s in enumerate([10, 0, -10])},
    'a_missing': ('missing', None),
}
HAZE_COLOR = torch.tensor([85.0, 95.0, 60.0])   # 浑水的黄绿色


def corrupt_video(x, kind, s, idx):
    """x: (B,T,H,W,3) uint8 GPU；idx: 样本全局编号，用作随机种子。"""
    B, T, H, W, _ = x.shape
    if kind == 'missing':
        return torch.zeros_like(x)
    xf = x.float()
    if kind == 'noise':
        gens = [torch.Generator(device=x.device).manual_seed(int(i)) for i in idx]
        noise = torch.stack([torch.randn(x.shape[1:], generator=g, device=x.device) for g in gens])
        xf = xf + s * noise
    elif kind == 'blur':
        k = int(6 * s) | 1
        flat = xf.reshape(B * T, H, W, 3).permute(0, 3, 1, 2)
        flat = TF.gaussian_blur(flat, kernel_size=[k, k], sigma=[float(s), float(s)])
        xf = flat.permute(0, 2, 3, 1).reshape(B, T, H, W, 3)
    elif kind == 'occl':
        side_h, side_w = int(H * s ** 0.5), int(W * s ** 0.5)
        for b, i in enumerate(idx):
            g = torch.Generator().manual_seed(int(i))
            top = int(torch.randint(0, H - side_h + 1, (1,), generator=g))
            left = int(torch.randint(0, W - side_w + 1, (1,), generator=g))
            xf[b, :, top:top + side_h, left:left + side_w, :] = 0
    elif kind == 'haze':
        xf = (1 - s) * xf + s * HAZE_COLOR.to(x.device)
    return xf.round().clamp(0, 255).to(torch.uint8)


def corrupt_audio(w, kind, s, idx):
    if kind == 'missing':
        return torch.zeros_like(w)
    gens = [torch.Generator(device=w.device).manual_seed(int(i)) for i in idx]
    noise = torch.stack([torch.randn(w.shape[1:], generator=g, device=w.device) for g in gens])
    p_sig = w.pow(2).mean(dim=1, keepdim=True).clamp_min(1e-12)
    p_noise = p_sig / (10 ** (s / 10))
    return w + noise * p_noise.sqrt()


def lookup(name, phase, paths, split='test'):
    d = load_phase(name, phase)[split]
    idx = {p: i for i, p in enumerate(d['paths'])}
    return d['feat'][torch.tensor([idx[p] for p in paths])]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', default='mv2_s3dft')
    p.add_argument('--fusion_ckpt', default='workspace/HAIL_fusion_phase0_mv2_s3dft/fusion_best.pt')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--split', choices=['test', 'val'], default='test',
                   help='val 用来选可靠性门控的超参，test 只用来最终报告')
    args = p.parse_args()
    device = 'cuda'

    s3d = build_video_model('workspace/S3D_phase0_redtilapia/phase0_best.pt', device)
    audio = build_audio_model(device, 'cnn14mv2')
    ck = torch.load(args.fusion_ckpt, map_location='cpu', weights_only=False)
    fusion = FusionWithHead(1024, ck['args']['h'], ck['args']['dropout']).to(device).eval()
    fusion.load_state_dict(ck['model_state_dict'])
    out_root = os.path.join(FEAT_ROOT, 'corrupt' if args.split == 'test' else 'corrupt_val')

    for ph in INCREMENTAL_PHASES:
        t0 = time.time()
        _, val_items, test_items, _ = build_av_phase_data(ph)   # 只要有视频的配对样本
        test = test_items if args.split == 'test' else val_items
        paths = [os.path.relpath(it['audio'], DATA_ROOT) for it in test]   # 样本 ID 与 extract_features.py 相同
        labels = torch.tensor([it['label'] for it in test])
        species = [it['species'] for it in test]
        offset = ph * 100000 + (0 if args.split == 'test' else 50000)   # 各阶段/切分样本种子不重叠
        clean_a = lookup('audio_cnn14mv2', ph, paths, args.split).float()
        clean_fmap = lookup('video_s3d_ft_fmap', ph, paths, args.split)
        clean_vpool = lookup('video_s3d_ft', ph, paths, args.split).float()
        res = {c: {k: [] for k in ('video', 'audio', 'av', 'v', 'a')} for c in list(VIDEO_CONDS) + list(AUDIO_CONDS)}

        with torch.no_grad():
            # ---- 视频损坏：每批只读一次盘，13 种损坏一起算 ----
            loader = DataLoader(VideoClipDataset(test), batch_size=args.batch_size, num_workers=args.num_workers,
                                pin_memory=True)
            for clip, _, idx in loader:
                clip = clip.to(device, non_blocking=True)
                fa = clean_a[idx].to(device)
                for c, (kind, s) in VIDEO_CONDS.items():
                    xc = corrupt_video(clip, kind, s, idx + offset)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        o = s3d(xc)
                    fmap = o['fmap'].flatten(3).permute(0, 2, 3, 1)
                    fo = fusion(fmap, fa)
                    r = res[c]
                    r['video'].append(o['embedding'].float().cpu())
                    r['audio'].append(fa.cpu())
                    r['av'].append(fo['f_av'].float().cpu())
                    r['v'].append(fo['f_v'].float().cpu())
                    r['a'].append(fo['f_a'].float().cpu())
            # ---- 音频损坏 ----
            loader = DataLoader(AudioClipDataset(test), batch_size=128, num_workers=args.num_workers)
            for wav, _, idx in loader:
                wav = wav.to(device)
                fmap = clean_fmap[idx].to(device)
                for c, (kind, s) in AUDIO_CONDS.items():
                    fa = audio(corrupt_audio(wav, kind, s, idx + offset))['embedding'].float()
                    fo = fusion(fmap, fa)
                    r = res[c]
                    r['video'].append(clean_vpool[idx])
                    r['audio'].append(fa.cpu())
                    r['av'].append(fo['f_av'].float().cpu())
                    r['v'].append(fo['f_v'].float().cpu())
                    r['a'].append(fo['f_a'].float().cpu())

        names = {'video': 'video_s3d_ft', 'audio': 'audio_cnn14mv2', 'av': f'hail_av_{args.tag}',
                 'v': f'hail_v_{args.tag}', 'a': f'hail_a_{args.tag}'}
        for c, r in res.items():
            os.makedirs(os.path.join(out_root, c), exist_ok=True)
            torch.save({names[k]: {'paths': paths, 'labels': labels, 'species': species, 'feat': torch.cat(v)}
                        for k, v in r.items()}, os.path.join(out_root, c, f'phase{ph}.pt'))
        print(f'phase {ph} {INCREMENTAL_PHASES[ph]}: {len(test)} 个 {args.split} 样本, {len(res)} 种损坏, '
              f'{(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
