#!/bin/bash
# 完全随机物种顺序实验：对每个不同的第一个物种重新适应两个编码器、提取全部特征，再跑各方法。断点续跑。
cd "$(dirname "$0")/.."
P=${PYTHON:-python}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date '+%F %T')] $*"; }
done_feat() { [ -f features/$1/phase6.pt ] && [ -f features/$1/phase0.pt ]; }
mkdir -p logs results/all/fullorder
for F in 6 2 4 1; do
  if [ ! -f workspace/S3D_first$F/phase0_best.pt ]; then
    $P train_phase0_video.py --phase $F --exp_name S3D_first$F > logs/fo_video_$F.log 2>&1 || { log "video $F failed"; exit 1; }
  fi
  log "first=$F video encoder: $(grep '== Phase 0 Test' -A1 logs/fo_video_$F.log | tail -1 | cut -c1-120)"
  if [ ! -f workspace/Cnn14mv2_first$F/phase0_best.pt ]; then
    $P train_phase0_audio.py --phase $F --split_mode random --out workspace/Cnn14mv2_first$F/phase0_best.pt > logs/fo_audio_$F.log 2>&1 || { log "audio $F failed"; exit 1; }
  fi
  log "first=$F audio encoder: $(tail -2 logs/fo_audio_$F.log | tr '\n' ' ' | cut -c1-160)"
  done_feat video_s3d_first$F || $P extract_features.py --modality video --name video_s3d_first$F \
      --video_ckpt workspace/S3D_first$F/phase0_best.pt --num_workers 8 > logs/fo_extract_v_$F.log 2>&1 || { log "extract v $F failed"; exit 1; }
  done_feat audio_cnn14mv2_first$F || $P extract_features.py --modality audio --audio_backbone cnn14mv2 \
      --audio_ckpt workspace/Cnn14mv2_first$F/phase0_best.pt --name audio_cnn14mv2_first$F --batch_size 128 \
      --num_workers 10 > logs/fo_extract_a_$F.log 2>&1 || { log "extract a $F failed"; exit 1; }
  log "first=$F features done"
done
$P tools/run_fullorder.py > logs/fo_runs.log 2>&1 || { log "runs failed"; exit 1; }
log "ALL DONE"; touch results/all/fullorder/DONE
