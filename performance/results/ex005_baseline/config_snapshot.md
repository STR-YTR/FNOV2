# ex005_baseline 設定スナップショット

記録作成日: 2026-11-11（日本時間）。実行用設定ではありません。

## 設定の出典

学習設定は当該実験の `outputs/training/ex005/configs/` に保存されたYAML全文です。現在の共通YAMLで置き換えていません。相対パスは元ファイルの位置を基準に解釈します。

評価時のtest.yaml・rollout.yaml、pair生成YAMLの独立した保存記録は確認できないため、現在の設定を過去の実行設定として転載していません。確認できる評価条件は保存済みmetrics.jsonとrollout_metrics.csvから収録します。

## dataset.yaml

出典: [dataset.yaml](../../../outputs/training/ex005/configs/dataset.yaml)

```yaml
dataset:
  data_dir: ../data/processed/pairs
  file_pattern: "pairs_case*_Re*.pt"

  # チャネル順・チャネル数の正本。Modelのin/out channelsはこのリストから導出する(mask,Re,x,y,u,v,p),(u,v,p)
  input_channels:
    - mask
    - Re
    - x
    - y
    - u
    - v
    - p
  output_channels:
    - u
    - v
    - p

  # 大容量データを一括ロードせず、直近に参照したpairファイルだけを保持する。
  cache_file: true

normalization:
  # u/v/pの統計値と、それを算出したtrain splitのmetadata。
  statistics_file: normalization_stats.yaml
  mask:
    type: none
  Re:
    type: scale
    scale: 250.0
  x:
    type: scale
    scale: 1.6
  y:
    type: scale
    scale: 0.41
  fields:
    type: standardize
    statistics_source: train
    valid_region_only: true

# dataset.pyを直接実行した場合のNormalization確認表示。
debug:
  enabled: true
  sample_count: 5
```

## normalization_stats.yaml

出典: [normalization_stats.yaml](../../../outputs/training/ex005/configs/normalization_stats.yaml)

```yaml
# configs/dataloader.yamlのtrain_reだけから算出した再現用metadata。
normalization_statistics:
  statistics_source: train
  valid_region_only: true
  sample_count: 2100
  valid_point_count: 33999800
  train_re:
    - 10
    - 20
    - 30
    - 40
    - 60
    - 70
    - 80
    - 90
    - 110
    - 120
    - 130
    - 140
    - 160
    - 170
    - 180
    - 190
    - 210
    - 220
    - 230
    - 240
    - 250
  fields:
    u:
      mean: 0.01956091086922085
      std: 0.014513064333257285
    v:
      mean: 0.000003338172258003842
      std: 0.006574265401363537
    p:
      mean: 0.00004195688495005484
      std: 0.00017544342254628847
```

## dataloader.yaml

出典: [dataloader.yaml](../../../outputs/training/ex005/configs/dataloader.yaml)

```yaml
# Datasetの場所・チャネル構成はdataset.yamlを正本とする。
dataset_config: dataset.yaml

# Reケース単位で分割する。同一Reを両方へ指定してはならない。
split:
  train_re:
    - 10
    - 20
    - 30
    - 40
    - 60
    - 70
    - 80
    - 90
    - 110
    - 120
    - 130
    - 140
    - 160
    - 170
    - 180
    - 190
    - 210
    - 220
    - 230
    - 240
    - 250
  val_re:
    - 50
    - 100
    - 150
    - 200

loader:
  batch_size: 8
  train_shuffle: true
  val_shuffle: false
  num_workers: 0
  pin_memory: true
  drop_last: false
  persistent_workers: false
  seed: 42
```

## model.yaml

出典: [model.yaml](../../../outputs/training/ex005/configs/model.yaml)

```yaml
model:
  name: FNO
  implementation: neuralop

  # 入出力チャネル数はdataset.yamlのチャネルリストから自動導出する。
  dataset_config: dataset.yaml
  dataloader_config: dataloader.yaml

  n_modes: [24, 24]
  hidden_channels: 128
  lifting_channels: 64
  projection_channels: 64
  n_layers: 4
  activation: gelu

  # Datasetがx/yを明示的に入力するため、FNO内部のgrid embeddingは無効にする。
  positional_embedding: null
  domain_padding: 0.0


  norm: null
  use_channel_mlp: true
  channel_mlp_dropout: 0.0
  channel_mlp_expansion: 0.5
  channel_mlp_skip: soft-gating
  fno_skip: linear
  fno_block_precision: full
  factorization: null
  rank: 1.0
  fixed_rank_modes: false
  weight_implementation: factorized
  separable: false
  preactivation: false

debug:
  print_summary: true
  print_architecture: true
  sanity_check: true
  # CPUでも軽量にforward/backwardを確認するため、取得batchの先頭だけを使用する。
  sample_batch_size: 1

visualization:
  # 有効時は外部ライブラリを使わず、完全なPyTorchモデル構造をテキスト保存する。
  enabled: true
  output_path: ../outputs/model/fno_architecture.txt
```

## train.yaml

出典: [train.yaml](../../../outputs/training/ex005/configs/train.yaml)

```yaml
training:
  model_config: model.yaml
  epochs: 200
  seed: 42
  device: cuda
  deterministic: false
  gradient_clip_norm: null
  resume_checkpoint: null

optimizer:
  type: adam
  learning_rate: 0.0005
  weight_decay: 0.0

scheduler:
  type: cosine_annealing
  min_learning_rate: 0.000001

loss:
  # baselineはstandardized u/v/pに対する全格子点MSE。
  type: mse
  valid_region_only: false

output:
  #####################################
  directory: ../outputs/training/ex005
  #####################################
  checkpoint_interval: 10

wandb:
  enabled: true
  project: fnov2-cylinder-wake
  entity: null
  name: null
  group: baseline
  job_type: train
  tags:
    - fno
    - baseline
    - one-step
  notes: Baseline FNO one-step prediction

  # online: クラウドへ同期、offline: ローカル保存のみ、disabled: W&B無効。
  mode: online
  save_code: true
  log_interval_steps: 10

  # 同じW&B runを再開する場合にrun_idを設定する。
  run_id: null
  resume: allow

  watch:
    enabled: false
    log: all
    log_freq: 100

  log_best_checkpoint: true
  artifact_name: fnov2-fno-baseline

  # smoke testは通常クラウドへ送信しない。
  log_smoke_test: false
```

## interpolation: one-step評価の保存metadata

出典: [metrics.json](interpolation/one_step/metrics.json)

```json
{
  "evaluation": "interpolation",
  "mode": "one_step",
  "checkpoint": "/workspace/FNOV2/outputs/training/ex005/best.pt",
  "checkpoint_epoch": 196,
  "test_re": [
    35.0,
    85.0,
    135.0,
    185.0,
    235.0
  ],
  "test_files": 5,
  "test_samples": 500,
  "input_channels": [
    "mask",
    "Re",
    "x",
    "y",
    "u",
    "v",
    "p"
  ],
  "output_channels": [
    "u",
    "v",
    "p"
  ],
  "interpolation_plot": "/workspace/FNOV2/performance/results/ex005_baseline/interpolation/one_step/interpolation_relative_l2.png"
}
```

## interpolation: rolloutの保存時刻・step

出典: [rollout_metrics.csv](interpolation/rollout/rollout_metrics.csv)。初期時刻は `time - elapsed_time` から算出。

| Re | 記録行数 | step範囲 | 初期時刻 | 予測時刻範囲 |
| ---: | ---: | ---: | ---: | ---: |
| 35.0 | 100 | 1–100 | 340 | 340.1–350 |
| 85.0 | 100 | 1–100 | 340 | 340.1–350 |
| 135.0 | 100 | 1–100 | 340 | 340.1–350 |
| 185.0 | 100 | 1–100 | 340 | 340.1–350 |
| 235.0 | 100 | 1–100 | 340 | 340.1–350 |

## extrapolation: one-step評価の保存metadata

出典: [metrics.json](extrapolation/one_step/metrics.json)

```json
{
  "evaluation": "extrapolation",
  "mode": "one_step",
  "checkpoint": "/workspace/FNOV2/outputs/training/ex005/best.pt",
  "checkpoint_epoch": 196,
  "test_re": [
    260.0,
    270.0,
    280.0,
    290.0,
    300.0
  ],
  "test_files": 5,
  "test_samples": 500,
  "input_channels": [
    "mask",
    "Re",
    "x",
    "y",
    "u",
    "v",
    "p"
  ],
  "output_channels": [
    "u",
    "v",
    "p"
  ],
  "extrapolation_plot": "/workspace/FNOV2/performance/results/ex005_baseline/extrapolation/one_step/extrapolation_relative_l2.png"
}
```

## extrapolation: rolloutの保存時刻・step

出典: [rollout_metrics.csv](extrapolation/rollout/rollout_metrics.csv)。初期時刻は `time - elapsed_time` から算出。

| Re | 記録行数 | step範囲 | 初期時刻 | 予測時刻範囲 |
| ---: | ---: | ---: | ---: | ---: |
| 260.0 | 100 | 1–100 | 340 | 340.1–350 |
| 270.0 | 100 | 1–100 | 340 | 340.1–350 |
| 280.0 | 100 | 1–100 | 340 | 340.1–350 |
| 290.0 | 100 | 1–100 | 340 | 340.1–350 |
| 300.0 | 100 | 1–100 | 340 | 340.1–350 |
