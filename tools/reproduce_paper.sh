#!/bin/bash
# 复现论文全部实验（AV-SIL-FFIR 主实验、协议、敏感性、原型曲线、AVE-CI / K-S-CI、校准、开销、图表）。
# 用法：nohup bash tools/reproduce_paper.sh > logs/reproduce.log 2>&1 &
cd "$(dirname "$0")/.."
P=${PYTHON:-python}
log() { echo "[$(date '+%F %T')] $*"; }

mkdir -p logs results/sensitivity results/proto_curve outputs/figures outputs/tables
# 超参数：默认直接用仓库里的 results/selection/*.json（论文所用）；设置 RESELECT=1 则在验证集上重新选参
if [ "${RESELECT:-0}" = "1" ]; then $P select_hp.py > logs/select_hp.log 2>&1; log "selection done"; fi

read LS LR K PI < <($P -c "import json;h=json.load(open('results/selection/hail_av.json'))['best_hp'];print(h['lam_s'],h['lam_r'],h['kappa'],h['prior_av'])")
log "HAIL selected: lam_s=$LS lam_r=$LR kappa=$K prior_av=$PI"

# ---------------- 流水线 A：AV-SIL-FFIR ----------------
pipeline_a() {
  $P baselines.py > logs/u_baselines.log 2>&1;                                      log "A: baselines (audio/video/av) done"
  BASE="finetune joint lwf ewc er icarl avcil acil dsal aqualytic mmal"
  $P run_all.py --exp main --methods $BASE hail hail_audio hail_video hail_exact abl_shared_av abl_experts_av \
     abl_shared_rel abl_fixed_weights abl_route_av proto_centroid_unw proto_centroid proto_diag \
     joint_hail joint_hail_audio joint_hail_video > logs/u_main.log 2>&1;                log "A: main done"
  $P run_all.py --exp k400 --methods hail acil dsal aqualytic mmal er icarl avcil finetune joint > logs/u_k400.log 2>&1
  log "A: k400 done"
  $P run_all.py --exp order --methods hail acil dsal aqualytic mmal er icarl avcil finetune lwf ewc > logs/u_order.log 2>&1
  log "A: order done"
  S="--eval_split test --corrupt --stats proto --m 20 --rank 8 --s 500 --s_deploy 500"
  D="--lam_s $LS --lam_r $LR --kappa $K --prior_av $PI"
  run() { name=$1; shift; $P hail_ffir.py $S $D "$@" --out results/sensitivity/$name.json 2>&1 | grep SUMMARY | sed "s/^/$name /"; }
  {
    for v in 1 10 100 1000 10000; do run lam_s_$v --lam_s $v; done
    for v in 1 3 10 30 100 300 1000; do run lam_r_$v --lam_r $v; done
    for v in 0 0.5 1 2 4 8; do run kappa_$v --kappa $v; done
    for v in 0.2 0.33 0.5 0.6 0.8 0.95; do run prior_$v --prior_av $v; done
    for v in 1024 2048 4096 8192; do run dim_$v --dim $v; done
    for v in 1 3 10 30; do run tau_$v --tau $v; done
  } > logs/u_sensitivity.log 2>&1
  log "A: sensitivity done"
}

# ---------------- 流水线 B：按天切分、原型曲线、AVE-CI / K-S-CI ----------------
pipeline_b() {
  # 按天切分实验需要基线的 best_hp（流水线 A 第一步写入 results/baselines）
  until [ "$(ls results/baselines/*_av.json 2>/dev/null | wc -l)" -ge 11 ]; do sleep 60; done
  $P run_all.py --exp session --methods hail acil dsal aqualytic mmal er icarl avcil finetune lwf ewc joint > logs/u_session.log 2>&1
  log "B: session done"
  for m in 1 2 5 10 20 50; do s=$((10000 / m)); for r in 0 8; do
    $P hail_ffir.py --eval_split test --corrupt --stats proto --m $m --rank $r --s $s --s_deploy $s \
       --lam_s $LS --lam_r $LR --kappa $K --prior_av $PI --out results/proto_curve/m${m}_r${r}.json 2>&1 | grep SUMMARY
  done; done > logs/u_proto_curve.log 2>&1
  log "B: prototype curve done"
  $P ave_cil.py --dataset ave --methods finetune lwf ewc er icarl avcil acil dsal mmal joint joint_ridge hail hail_exact \
     > logs/u_ave.log 2>&1;                                                           log "B: AVE-CI done"
  $P ave_cil.py --dataset ks --methods hail hail_exact acil dsal mmal > logs/u_ks.log 2>&1; log "B: K-S-CI done"
  $P tools/prior_sweep.py ave ks > logs/u_prior_sweep.log 2>&1;                       log "B: prior sweep done"
}

pipeline_a & PA=$!
pipeline_b & PB=$!
wait $PA $PB

# ---------------- 收尾：校准、开销（GPU 空闲时）、图表 ----------------
$P calibration.py > logs/u_calibration.log 2>&1;  log "calibration done"
$P cost.py > logs/u_cost.log 2>&1;                log "cost done"
$P viz.py > logs/u_viz.log 2>&1
$P make_results.py > results/summary.md 2> logs/u_make_results.err
$P tools/gen_tables_tex.py > /dev/null
$P tools/gen_ave_table.py > /dev/null
$P tools/gen_ks_table.py > /dev/null
$P tools/plot_prior_sweep.py > /dev/null
log "ALL DONE"
touch results/UNIFIED_DONE
