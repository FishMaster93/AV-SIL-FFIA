"""
把原始 mp4 一次性解码、均匀抽 16 帧、缩放到 224x224，存成 uint8 张量 (16, 224, 224, 3)
的 .npy 文件，目录结构和文件名跟音频数据集一一对应：

  广州 (--dataset gz):
    src: <src>/{date}/{中文物种名}/{强度}/{n}.mp4
    dst: <dst>/{date}/{英文物种名}/{强度}/{n}.npy      <-> /data/usb/audio_dataset/{date}/{英文物种名}/{强度}/{n}.wav
  Oplegnathus (--dataset opleg):
    src: <src>/{date}/{AM_40|PM_70|...}/{强度}/13_video_N.mp4
    dst: <dst>/{date}/{AM_40|PM_70|...}/{强度}/13_audio_N.npy  (文件名换成音频的名字，方便一一对应)

原始视频在 USB 机械盘上，瓶颈是读盘而不是解码：主进程按目录顺序串行读文件字节，
避免多进程随机读导致磁头来回寻道；解码和缩放交给进程池并行。支持断点续跑
（目标文件已存在就跳过，写入先写临时文件再 rename）。
"""
import argparse
import io
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait

import numpy as np
import av

NUM_FRAMES = 16
SIZE = 224

GZ_SPECIES = {
    '大口黑鲈': 'Largemouth_bass',
    '太阳鱼': 'Sunfish',
    '宝石鲈': 'Jade_perch',
    '彩虹鲷2': 'Red_tilapia_2',
    '彩虹鲷3': 'Red_tilapia_3',
    '罗非鱼': 'Tilapia',
    '荷花鲤': 'Lotus_carp',
}


def list_jobs(dataset, src, dst, species_filter=None, sessions=None):
    """按目录顺序列出 (src_mp4, dst_npy)，顺序读盘对机械硬盘最友好。"""
    jobs = []
    for date in sorted(os.listdir(src)):
        if sessions and date not in sessions:
            continue
        date_path = os.path.join(src, date)
        if not os.path.isdir(date_path):
            continue
        for sub in sorted(os.listdir(date_path)):
            if dataset == 'gz':
                out_sub = GZ_SPECIES.get(sub)
                if out_sub is None:
                    print(f'[Warning] 未知物种目录: {date}/{sub}')
                    continue
                if species_filter and out_sub not in species_filter:
                    continue
            else:
                out_sub = sub
            sub_path = os.path.join(date_path, sub)
            for intensity in sorted(os.listdir(sub_path)):
                in_dir = os.path.join(sub_path, intensity)
                if not os.path.isdir(in_dir):
                    continue
                for fn in sorted(os.listdir(in_dir)):
                    if not fn.endswith('.mp4'):
                        continue
                    stem = fn[:-4]
                    if dataset == 'opleg':
                        stem = stem.replace('_video_', '_audio_')
                    jobs.append((os.path.join(in_dir, fn),
                                 os.path.join(dst, date, out_sub, intensity, stem + '.npy')))
    return jobs


def decode_clip(data):
    """mp4 字节 -> (16, 224, 224, 3) uint8。MPEG-4 有帧间预测，必须顺序解完全部帧，
    但只对选中的 16 帧做缩放/转 RGB。"""
    container = av.open(io.BytesIO(data))
    stream = container.streams.video[0]
    stream.thread_type = 'AUTO'
    frames = [f for f in container.decode(video=0)]
    container.close()
    n = len(frames)
    if n == 0:
        raise RuntimeError('没有解出任何帧')
    idx = np.linspace(0, n - 1, NUM_FRAMES).round().astype(int)
    out = np.empty((NUM_FRAMES, SIZE, SIZE, 3), dtype=np.uint8)
    for k, i in enumerate(idx):
        out[k] = frames[i].reformat(width=SIZE, height=SIZE, format='rgb24',
                                    interpolation='AREA').to_ndarray()
    return out, n


def worker(args):
    src_path, dst_path, data = args
    try:
        clip, n = decode_clip(data)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        tmp = dst_path + '.tmp.npy'
        np.save(tmp, clip)
        os.replace(tmp, dst_path)
        return src_path, n, None
    except Exception as e:
        return src_path, 0, repr(e)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['gz', 'opleg'], required=True)
    p.add_argument('--src', required=True)
    p.add_argument('--dst', required=True)
    p.add_argument('--species', nargs='*', help='只处理这些英文物种名 (仅 gz)')
    p.add_argument('--sessions', nargs='*', help='只处理这些日期场次目录')
    p.add_argument('--workers', type=int, default=10)
    p.add_argument('--inflight', type=int, default=40, help='同时在内存里的视频数上限')
    p.add_argument('--limit', type=int, default=0, help='只处理前 N 个，试跑用')
    args = p.parse_args()

    jobs = list_jobs(args.dataset, args.src, args.dst, args.species, args.sessions)
    total = len(jobs)
    jobs = [j for j in jobs if not os.path.exists(j[1])]
    if args.limit:
        jobs = jobs[:args.limit]
    print(f'共 {total} 个视频，已完成 {total - len(jobs) if not args.limit else "-"}，本次处理 {len(jobs)}')

    os.makedirs(args.dst, exist_ok=True)
    fail_log = open(os.path.join(args.dst, '_failed.txt'), 'a')
    odd_log = open(os.path.join(args.dst, '_frame_count_not_50.txt'), 'a')

    t0 = time.time()
    done = n_bytes = n_fail = 0
    pending = set()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        it = iter(jobs)
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < args.inflight:
                try:
                    src_path, dst_path = next(it)
                except StopIteration:
                    exhausted = True
                    break
                with open(src_path, 'rb') as f:
                    data = f.read()
                n_bytes += len(data)
                pending.add(ex.submit(worker, (src_path, dst_path, data)))
            if not pending:
                break
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                src_path, n, err = fut.result()
                done += 1
                if err:
                    n_fail += 1
                    fail_log.write(f'{src_path}\t{err}\n')
                    fail_log.flush()
                elif n != 50:
                    odd_log.write(f'{src_path}\t{n}\n')
                if done % 200 == 0 or done == len(jobs):
                    el = time.time() - t0
                    rate = done / el
                    eta = (len(jobs) - done) / max(rate, 1e-9)
                    print(f'{done}/{len(jobs)} | {rate:.1f} clip/s | '
                          f'{n_bytes / el / 1e6:.0f} MB/s 读盘 | 失败 {n_fail} | '
                          f'剩余 {eta / 3600:.2f} h', flush=True)
    fail_log.close()
    odd_log.close()
    print(f'完成：{done} 个，失败 {n_fail} 个，用时 {(time.time() - t0) / 60:.1f} 分钟')


if __name__ == '__main__':
    main()
