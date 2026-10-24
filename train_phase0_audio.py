"""
Phase 0 音频编码器训练：Cnn14-MobileNetV2 从零训练 + 临时 CORAL 头，只用 Phase 0 物种。
训练设置与 AquaLytic 的 Cnn14_mobilev2 Phase 0 一致（Adam lr=1e-3、batch 1024、200 epoch、按验证集挑最佳），
用于按天切分（--split_mode session）时重新得到一个没见过测试场次的音频编码器。
波形一次性读入内存（float16），每个 epoch 只做 GPU 计算。
"""
import argparse
import os
import sys
import time

import librosa
import numpy as np
import torch
from scipy.signal import resample
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets.fish_av_incremental import build_phase_items
from models.audio_model import Audio_Frontend, AudioModel
from models.cnn14_mobilev2 import Cnn14_mobilev2
from train_phase0_video import coral_loss, coral_predict

AUDIO_FEATURES = dict(sample_rate=64000, window_size=2048, hop_size=1024, mel_bins=64, fmin=1, fmax=128000)


def load_wav(path, sr=64000):
    y, _ = librosa.load(path, sr=None)
    return resample(y, num=sr * 2).astype(np.float16)


def load_split(items, workers=12):
    with ThreadPoolExecutor(workers) as ex:
        wavs = list(ex.map(lambda it: load_wav(it['audio']), items))
    return torch.from_numpy(np.stack(wavs)), torch.tensor([it['label'] for it in items])


@torch.no_grad()
def evaluate(model, x, y, bs=512):
    model.eval()
    preds = []
    for i in range(0, len(x), bs):
        preds.append(coral_predict(model(x[i:i + bs].cuda().float())['clipwise_output']).cpu())
    return float((torch.cat(preds) == y).float().mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split_mode', choices=['random', 'session'], default='session')
    p.add_argument('--phase', type=int, default=0, help='在哪个物种（阶段）上训练音频编码器；默认 0 = Red Tilapia')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=1024)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--eval_every', type=int, default=5)
    p.add_argument('--out', default='workspace/Cnn14mv2_phase0_session/phase0_best.pt')
    args = p.parse_args()
    torch.manual_seed(25)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    tr, va, te, _ = build_phase_items(args.phase, args.split_mode)
    t0 = time.time()
    xtr, ytr = load_split(tr)
    xva, yva = load_split(va)
    xte, yte = load_split(te)
    print(f'读入波形 train/val/test = {len(ytr)}/{len(yva)}/{len(yte)}，{(time.time() - t0) / 60:.1f} min', flush=True)

    model = AudioModel(frontend=Audio_Frontend(**AUDIO_FEATURES), backbone=Cnn14_mobilev2(classes_num=3)).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = -1
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(ytr))
        tot = 0.0
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            out = model(xtr[idx].cuda().float())['clipwise_output']
            loss = coral_loss(out, ytr[idx].cuda())
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if ep % args.eval_every == 0 or ep == args.epochs - 1:
            va_acc = evaluate(model, xva, yva)
            flag = ''
            if va_acc > best:
                best, flag = va_acc, ' *'
                torch.save({'model_state_dict': model.state_dict(), 'acc': va_acc, 'epoch': ep}, args.out)
            print(f'epoch {ep} loss {tot / len(ytr):.4f} val {va_acc:.4f}{flag} | {(time.time() - t0) / 60:.1f} min', flush=True)
    ck = torch.load(args.out, weights_only=False)
    model.load_state_dict(ck['model_state_dict'])
    print(f'== best epoch {ck["epoch"]} val {ck["acc"]:.4f} test {evaluate(model, xte, yte):.4f}')


if __name__ == '__main__':
    main()
