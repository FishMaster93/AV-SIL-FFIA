# AV-SIL-FFIR: Audio-Visual Species-Incremental Fish Feeding Intensity Recognition

Code, data splits and selected hyperparameters for

> **Reliability-Aware Audio-Visual Species-Incremental Learning for Fish Feeding Intensity Recognition**
> Meng Cui, Xianghu Yue, Xinyuan Qian, Jinzheng Zhao, Haohe Liu, Xubo Liu, Daoliang Li, Wenwu Wang.
> *IEEE Transactions on Emerging Topics in Computational Intelligence* (under review).

HAIL-FFIR learns fish feeding intensity (*None / Weak / Medium / Strong*) from audio and video while new
species arrive one at a time. It does not store historical recordings. It combines three parts:

1. **Prototype-enhanced analytic training.** Each earlier species is stored as Gaussian prototypes in the encoder feature
   space. Closed-form (ridge) heads are retrained from the reconstructed statistics.
2. **Species-conditional analytic experts.** Each species gets an expert anchored on the shared head. The expert is frozen
   once that species has been learned.
3. **Reliability-aware dynamic modality balancing.** The audio-visual, audio and visual heads are weighted by their
   posterior uncertainty. The same weights are used to route each sample to a species expert.

## Data and pre-computed features

| Resource | Link | Size |
|---|---|---|
| Six Guangzhou species: audio (`.wav`, hydrophone, 256 kHz) | **Zenodo: https://doi.org/10.5281/zenodo.21888149** | 43 GB |
| Six Guangzhou species: video (`.mp4`, 854×480, 25 fps) | **Zenodo: https://doi.org/10.5281/zenodo.23623291** | 28 GB |
| *O. punctatus* (second farm): audio + video | **Zenodo: https://doi.org/10.5281/zenodo.12759731**, plus Parts 2 ([11060370](https://zenodo.org/records/11060370)) and 3 ([11060195](https://zenodo.org/records/11060195)) | ~80 GB |
| Pre-extracted features + encoder checkpoints | **Zenodo: https://doi.org/10.5281/zenodo.23619403** | 7 GB |
| Data splits (clip-level and session-disjoint) | [`data/manifest.csv`](data/manifest.csv) (in this repository) | 24 MB |
| Hyperparameters selected on the validation set | [`results/selection/`](results/selection) (in this repository) | – |

The dataset covers seven species across seven incremental stages. Each sample is a 2-second clip.

| Stage | Species | Site |
|---|---|---|
| 1 | Red Tilapia | Guangzhou |
| 2 | Tilapia | Guangzhou |
| 3 | Jade Perch | Guangzhou |
| 4 | Largemouth Bass | Guangzhou |
| 5 | Lotus Carp | Guangzhou |
| 6 | Sunfish | Guangzhou |
| 7 | *Oplegnathus punctatus* | second farm |

The stage order is defined in
[`datasets/fish_av_incremental.py`](datasets/fish_av_incremental.py) (`INCREMENTAL_PHASES`).

**Most users only need the features.** All incremental-learning experiments, every table in the paper and most
figures run on the cached encoder features. They take minutes on a single GPU. The raw data is needed only to
re-train the encoders or to re-generate the corrupted test sets.

## Repository structure

```
hail_ffir.py              HAIL-FFIR (the proposed method)
baselines.py              Fine-tuning, Joint, LwF, EWC, ER, iCaRL, AV-CIL*, ACIL, DS-AL, AquaLytic, MMAL* on the same features
run_all.py                Runs all methods under one protocol (main / k400 / order / session), 3 seeds
select_hp.py              Unified hyperparameter selection on the validation set (all methods)
analytic_cil.py           Random projection buffer, recursive ridge regression, feature loading
hail_cil.py, sahl_cil.py  Building blocks imported by hail_ffir.py (heads, reliability, aligned loading)
ave_cil.py                Class-incremental benchmarks AVE-CI and K-S-CI (AV-CIL features)
calibration.py            Temperature scaling, ECE / NLL / Brier, reliability diagrams
cost.py                   Learning time, memory and FLOPs
viz.py, make_results.py   t-SNE, figures, results summary (results/summary.md)
train_phase0_video.py     Adapts S3D (Kinetics-400) on the first species (CORAL ordinal head)
train_phase0_audio.py     Trains the Cnn14-MobileNetV2 audio encoder on the first species (CORAL ordinal head)
extract_features.py       Extracts frozen encoder features for all stages
corrupt_features.py       Builds the corrupted test/validation sets (noise, missing modality, blur, occlusion, turbidity)
train_phase0_fusion.py    Auxiliary fusion module, needed only by corrupt_features.py
datasets/                 Splits and data loaders
models/                   S3D, Cnn14-MobileNetV2, PANNs Cnn10, audio front-end
tools/                    Data preparation, figure/table scripts, end-to-end shell scripts
data/manifest.csv         One row per clip: species, session, label, relative paths, split columns
results/selection/        Selected hyperparameters for every method and modality
```

## Installation

Tested with Python 3.11, PyTorch 2.11 (CUDA 12.8) and a single 24 GB GPU.

```bash
git clone https://github.com/FishMaster93/AV-SIL-FFIA.git
cd AV-SIL-FFIA
conda create -n avsil python=3.11 -y && conda activate avsil
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # pick the build for your CUDA
pip install -r requirements.txt
```

All scripts are run from the repository root.

## Quick start: run from the released features

1. Download the four `.tar` files from [Zenodo](https://doi.org/10.5281/zenodo.23619403) and unpack them in the repository
   root (`for f in AV-SIL-FFIR_*.tar; do tar -xf "$f"; done`) so that you get `features/` and `workspace/`:

   ```
   features/
     audio_cnn14mv2/phase{0..6}.pt        audio features (main protocol)
     video_s3d_ft/phase{0..6}.pt          video features (S3D adapted on Red Tilapia)
     corrupt/<condition>/phase{0..6}.pt   corrupted test sets (17 conditions)
     corrupt_val/<condition>/...          corrupted validation sets (used for hyperparameter selection)
     video_s3d_k400/                      frozen Kinetics-400 S3D (additional protocol)
     audio_cnn14mv2_sess/, video_s3d_ft_sess/                  session-disjoint split
     audio_cnn14mv2_first{1,2,4,6}/, video_s3d_first{1,2,4,6}/ encoders adapted on other first species
   workspace/                             encoder checkpoints (needed only for feature extraction)
   ```

   Each `phase{k}.pt` is a dictionary `{'train'|'val'|'test': {'paths', 'labels', 'species', 'feat'}, 'meta': ...}`.
   Here `paths` are clip IDs (the audio path relative to the dataset root, as in `data/manifest.csv`), `labels` are
   intensity levels 0–3 and `feat` is an `N × 1024` tensor.

2. Run HAIL-FFIR with the selected configuration (audio-visual, seed 42):

   ```bash
   python hail_ffir.py --eval_split test --corrupt \
       --stats proto --m 20 --rank 8 --s 500 --s_deploy 500 \
       --lam_s 100 --lam_r 30 --kappa 2 --prior_av 0.33 --tau 10 --dim 8192 \
       --seed 42 --out results/hail_seed42.json
   ```

   Expected output (about 40 s without `--corrupt`, a few minutes with it):

   ```
   step 6 phase 6 ['Oplegnathus_punctatus']  acc_seen 0.9012 ...
   SUMMARY final_acc=0.9012 avg_inc_acc=0.9193 final_avg_forget=0.0013 storage_MB=50.4021 ... corrupt_mean=...
   ```

   Useful options:

   | Option | Effect |
   |---|---|
   | `--modality audio` / `--modality video` | Single-modality variants |
   | `--stats exact` | Stores exact Gram matrices instead of prototypes (the "exact statistics" control) |
   | `--order 3 1 5 2 6 4` | A different order of stages 2–7 (phase ids 1–6; Red Tilapia stays first) |

## Reproducing the paper

### Everything with one script

```bash
nohup bash tools/reproduce_paper.sh > logs/reproduce.log 2>&1 &
```

The script runs two pipelines in parallel and then finishes with calibration, cost and figures.

- **Pipeline A:**
  1. Baselines (`baselines.py`).
  2. Main comparison and ablations (`run_all.py --exp main`).
  3. Frozen Kinetics S3D (`--exp k400`).
  4. Random species orders (`--exp order`).
  5. Hyperparameter sensitivity.
- **Pipeline B:**
  1. Session-disjoint split (`--exp session`).
  2. Prototype-count curve.
  3. AVE-CI and K-S-CI.
  4. π<sub>av</sub> sweep.
- **Afterwards:**
  - Calibration, cost and t-SNE.
  - `results/summary.md`, LaTeX tables in `outputs/tables/` and figures in `outputs/figures/`.

By default the script uses the hyperparameters in `results/selection/`, which are the ones used in the paper. Set
`RESELECT=1` to re-run the validation-set selection first; this takes several hours. Set `PYTHON=/path/to/python`
if `python` is not your environment's interpreter. The full run takes roughly one day on one GPU, and most of
that time goes to the gradient-based baselines and K-S-CI.

### Individual experiments

| Paper content | Command |
|---|---|
| Hyperparameter selection (all methods, validation set) | `python select_hp.py` → `results/selection/*.json` |
| Baselines, Table I (audio / video / audio-visual) | `python baselines.py` → `results/baselines/` |
| Main comparison, robustness and per-species results | `python run_all.py --exp main --methods finetune joint lwf ewc er icarl avcil acil dsal aqualytic mmal hail hail_audio hail_video hail_exact joint_hail` |
| Ablation | `python run_all.py --exp main --methods abl_shared_av abl_experts_av abl_shared_rel abl_fixed_weights abl_route_av` |
| Prototype variants | `python run_all.py --exp main --methods proto_centroid_unw proto_centroid proto_diag` |
| Frozen Kinetics-400 S3D | `python run_all.py --exp k400 --methods hail acil dsal aqualytic mmal er icarl avcil finetune joint` |
| Random orders of stages 2–7 | `python run_all.py --exp order --methods hail acil dsal aqualytic mmal er icarl avcil finetune lwf ewc` |
| Fully random orders (first species also random) | `bash tools/run_fullorder.sh` (re-adapts the encoders; needs raw data, or the released `*_first*` features with `python tools/run_fullorder.py`) |
| Session-disjoint split | `python run_all.py --exp session --methods hail acil dsal aqualytic mmal er icarl avcil finetune lwf ewc joint` |
| Calibration | `python calibration.py` |
| Cost (time, memory, FLOPs) | `python cost.py` |
| Species bias heatmap / t-SNE of the heads | `python tools/species_bias_probe.py && python tools/plot_species_bias.py`; `python viz.py && python tools/plot_tsne_heads.py` |
| Summary, tables and figures | `python make_results.py > results/summary.md && python tools/gen_tables_tex.py && python tools/gen_orders_table.py` |

`run_all.py` writes one JSON per method and seed to `results/all/<exp>/`. Each file holds:

- the per-stage accuracy and forgetting;
- the per-sample test predictions;
- the accuracy under every corruption (`main` only);
- the mean reliability weights (HAIL-FFIR only).

### Evaluation protocol

- **Splits.** The clip-level split is stored in the column `split_clip` of `data/manifest.csv`: 80/10/10 per
  species × intensity, the same split as AquaLytic. The session-disjoint split is stored in `split_s0`: all clips of
  one recording day go to the same subset.
- **Hyperparameter selection.** Every method uses the same criterion on the validation set: S = ½ (clean average
  incremental accuracy + accuracy under audio/video noise and missing modalities). Blur, occlusion and turbidity are
  never used for selection and are reported as held-out corruptions.
- **Metrics.** Final accuracy on all species after the last stage; average incremental accuracy; average forgetting.
  All are reported as mean ± std over seeds 42, 43 and 44.

## Full pipeline from raw data

Download the dataset from Zenodo and arrange it as below. You can also point the environment variable `AVSIL_DATA`
to another directory.

| Source | Destination |
|---|---|
| Guangzhou audio archive | `data/raw/gz/audio_dataset/` |
| Guangzhou video (`video_dataset.tar`) | `data/raw/gz/video_dataset/`; convert it to `.npy` with step 0 below |
| *O. punctatus* audio | `data/raw/opleg/audio_dataset/` |
| *O. punctatus* video | convert to `data/raw/opleg/video_npy/` with step 0 below |

The released Guangzhou videos are 480p transcodes of the original 2560×1440 clips. Features extracted from them can
differ slightly from the released features, so use the released features to reproduce the paper's numbers exactly.

```
data/raw/
  gz/audio_dataset/<session>/<Species>/<Intensity>/<n>.wav
  gz/video_npy/<session>/<Species>/<Intensity>/<n>.npy          (16 × 224 × 224 × 3, uint8)
  opleg/audio_dataset/<date>/<group>/<intensity>/<name>.wav
  opleg/video_npy/<date>/<group>/<intensity>/<name>.npy
```

All paths in `data/manifest.csv` are relative to `AVSIL_DATA`.

```bash
# 0) (only if you start from .mp4) decode 16 uniformly sampled frames per clip at 224×224
python tools/preprocess_video.py --dataset gz    --src data/raw/gz/video_dataset --dst data/raw/gz/video_npy
python tools/preprocess_video.py --dataset opleg --src <mp4_root_opleg> --dst data/raw/opleg/video_npy
#    (optional) rebuild the manifest; the released one already contains the paper's splits
#    python tools/build_manifest.py && python tools/split_manifest.py

# 1) adapt the encoders on the first species (Red Tilapia) with a temporary CORAL ordinal head
#    S3D Kinetics-400 weights: place S3D_kinetics400.pt at Pre_trained_model/S3D400.pt
#    (from https://github.com/kylemin/S3D)
python train_phase0_video.py --exp_name S3D_phase0_redtilapia
python train_phase0_audio.py --split_mode random --out workspace/Cnn14mv2_phase0/phase0_best.pt

# 2) extract frozen features for all stages
python extract_features.py --modality video --name video_s3d_ft \
       --video_ckpt workspace/S3D_phase0_redtilapia/phase0_best.pt
python extract_features.py --modality audio --audio_backbone cnn14mv2 --name audio_cnn14mv2
python extract_features.py --modality video --name video_s3d_k400 --video_ckpt k400          # frozen S3D protocol

# 3) corrupted test / validation sets (needs video feature maps and the auxiliary fusion module)
python extract_features.py --modality video --name video_s3d_ft_fmap --save_fmap \
       --video_ckpt workspace/S3D_phase0_redtilapia/phase0_best.pt
python train_phase0_fusion.py
python corrupt_features.py --split test
python corrupt_features.py --split val

# 4) continue with "Reproducing the paper"
```

For the session-disjoint split, train both encoders with `--split_mode session` and extract the features with
`--split_mode session`. For the fully random orders, `tools/run_fullorder.sh` does every step automatically.

**Corruptions.** All corruptions are applied to the raw test clips and then passed through the frozen encoders:

| Corruption | Severities 1 / 2 / 3 |
|---|---|
| Audio white noise | SNR 10 / 0 / −10 dB |
| Video Gaussian noise | σ = 20 / 40 / 80 |
| Gaussian blur | σ = 2 / 4 / 8 px |
| Occlusion | 25 / 50 / 75 % of the frame |
| Turbidity | blending ratio 0.5 / 0.7 / 0.9 |
| Missing audio / missing video | – |

## Class-incremental benchmarks (AVE-CI, K-S-CI)

`ave_cil.py` uses the official splits, class orders and pre-extracted features released with
[AV-CIL (Pian et al., ICCV 2023)](https://github.com/weiguoPian/AV-CIL_ICCV2023).

```bash
git clone https://github.com/weiguoPian/AV-CIL_ICCV2023 external/AV-CIL_ICCV2023   # class lists / metadata
# AVE features    -> external/ave_features/   (as distributed by AV-CIL)
# K-S features    -> external/ks_features/    (KS_features.zip.a{a,b,c}; or set AVSIL_KS_ROOT)
python ave_cil.py --dataset ave --methods finetune lwf ewc er icarl avcil acil dsal mmal joint joint_ridge hail hail_exact
python ave_cil.py --dataset ks  --methods hail hail_exact acil dsal mmal
python tools/prior_sweep.py ave ks && python tools/plot_prior_sweep.py
```

`tools/run_ks.sh` downloads the K-S features, checks their SHA-256 and runs the K-S-CI experiments. The third-party
`.npy` pickles are loaded only through the restricted unpickler in `tools/safe_npy.py`.

## Citation

```bibtex
@article{cui2026hailffir,
  title   = {Reliability-Aware Audio-Visual Species-Incremental Learning for Fish Feeding Intensity Recognition},
  author  = {Cui, Meng and Yue, Xianghu and Qian, Xinyuan and Zhao, Jinzheng and Liu, Haohe and Liu, Xubo and Li, Daoliang and Wang, Wenwu},
  journal = {IEEE Transactions on Emerging Topics in Computational Intelligence},
  year    = {2026},
  note    = {under review}
}
```

## Acknowledgements

This code builds on [AV-CIL](https://github.com/weiguoPian/AV-CIL_ICCV2023), [ACIL / DS-AL](https://github.com/ZHUANGHP/Analytic-continual-learning),
[PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) and [S3D](https://github.com/kylemin/S3D).
Some code comments are written in Chinese.
