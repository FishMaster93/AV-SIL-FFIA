"""
计算开销（回应 R2-2 / R3-m7）：参数量、单片段 FLOPs、推理延迟（GPU batch=1 / 64，CPU batch=1）、
每阶段学习时间（GPU 空闲时测）、持久存储。
"""
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_features import AUDIO_FEATURES
from models.audio_model import Audio_Frontend, AudioModel
from models.cnn14_mobilev2 import Cnn14_mobilev2
from models.s3d import S3D
from run_all import Wrapper, load
from sahl_cil import A_RAW, V_RAW

R = 'results'


def flops(model, x):
    with FlopCounterMode(display=False) as fc:
        with torch.no_grad():
            model(x)
    return fc.get_total_flops()


def latency(fn, n=50, warm=10, cuda=True):
    for _ in range(warm):
        fn()
    if cuda:
        torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        fn()
    if cuda:
        torch.cuda.synchronize()
    return (time.time() - t) / n * 1000


def main():
    out = {}
    s3d = S3D(400).eval()
    aud = AudioModel(Audio_Frontend(**AUDIO_FEATURES), Cnn14_mobilev2(classes_num=3)).eval()
    out['params_M'] = {'S3D': sum(p.numel() for p in s3d.parameters()) / 1e6,
                       'Cnn14-MobileNetV2': sum(p.numel() for p in aud.parameters()) / 1e6}
    xv = torch.randint(0, 255, (1, 16, 224, 224, 3), dtype=torch.uint8)
    xa = torch.randn(1, 128000)
    out['GFLOPs_per_clip'] = {'S3D': flops(s3d, xv) / 1e9, 'Cnn14-MobileNetV2': flops(aud, xa) / 1e9}

    # 编码器延迟
    s3d, aud = s3d.cuda(), aud.cuda()
    with torch.no_grad():
        lat = {}
        for bs in (1, 64):
            v, a = xv.repeat(bs, 1, 1, 1, 1).cuda(), xa.repeat(bs, 1).cuda()
            def enc():
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    s3d(v)
                aud(a)
            lat[f'encoders_gpu_bs{bs}_ms_per_clip'] = latency(enc) / bs
        out['latency'] = lat

    # 头部（闭式解部分）的延迟与学习时间：在缓存特征上测
    methods = ['hail', 'hail_exact', 'acil', 'dsal', 'mmal', 'aqualytic', 'er', 'icarl', 'avcil', 'ewc', 'lwf', 'finetune']
    fd_test, _, _ = load('main', 0, 'test')
    learn = {}
    for m in methods:
        w = Wrapper(m, 42)
        t_stage = []
        for ph in range(7):
            fd, y, _ = load('main', ph, 'train')
            torch.cuda.synchronize(); t = time.time()
            w.learn(fd, y)
            torch.cuda.synchronize(); t_stage.append(time.time() - t)
        one = {k: v[:1] for k, v in fd_test.items()}
        b64 = {k: v[:64] for k, v in fd_test.items()}
        learn[m] = {'learn_s_per_stage_mean': float(np.mean(t_stage)), 'learn_s_stage1': t_stage[0],
                    'learn_s_total': float(np.sum(t_stage)), 'learn_s_per_stage': [float(x) for x in t_stage],
                    'head_gpu_bs1_ms': latency(lambda: w.predict(one), n=20, warm=3),
                    'head_gpu_bs64_ms_per_clip': latency(lambda: w.predict(b64), n=20, warm=3) / 64}
        if w.kind == 'hail':
            learn[m]['storage_MB'] = w.m.storage_bytes() / 1e6
        print(m, learn[m], flush=True)
        del w
        torch.cuda.empty_cache()
    out['heads'] = learn
    # 存储参照
    out['storage_reference_MB'] = {
        'exemplar_2000_features_fp32': 2000 * 2048 * 4 / 1e6,
        'exemplar_2000_raw_clips_npy_video_plus_wav': 2000 * (16 * 224 * 224 * 3 + 128000 * 2) / 1e6,
        'raw_dataset_GB_video_npy': 245.0,
    }
    json.dump(out, open(f'{R}/cost.json', 'w'), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != 'heads'}, indent=2))


if __name__ == '__main__':
    main()
