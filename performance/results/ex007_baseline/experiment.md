# ex007_baseline 実験記録

## 1. このファイルの役割

保存済みの学習設定・学習履歴・performance評価結果を集約した実験台帳です。記録作成日: 2026-11-11（日本時間）。学習・評価の再実行は行っていません。

設定全文と出典: [config_snapshot.md](config_snapshot.md)。現在の共通設定は変更され得るため、過去の実行条件の根拠には使用していません。

## 2. 実験識別情報

| 項目 | 値 |
| --- | --- |
| 実験ID | `"ex007_baseline"` |
| 学習保存先 | `"outputs/training/ex007"` |
| 評価checkpoint | `{"interpolation": "/workspace/FNOV2/outputs/training/ex007/best.pt", "extrapolation": "/workspace/FNOV2/outputs/training/ex007/best.pt"}` |
| 評価checkpoint epoch | `{"interpolation": 242, "extrapolation": 242}` |

checkpoint内部のGit commit・neuralopバージョンは本記録では未確認です。

## 3. データ生成・Dataset

当該実験専用のpair生成設定は未保存のため、生成時の補間条件・格子条件はここでは断定しません。Dataset設定全文はスナップショットに収録しています。

| 項目 | 値 |
| --- | --- |
| data_dir | `"../data/processed/pairs"` |
| file_pattern | `"pairs_case*_Re*.pt"` |
| input_channels | `["mask", "Re", "x", "y", "u", "v", "p"]` |
| output_channels | `["u", "v", "p"]` |
| cache_file | `true` |

## 4. Normalization

| 項目 | 値 |
| --- | --- |
| statistics_file | `"normalization_stats.yaml"` |
| mask | `{"type": "none"}` |
| Re | `{"type": "scale", "scale": 250.0}` |
| x | `{"type": "scale", "scale": 1.6}` |
| y | `{"type": "scale", "scale": 0.41}` |
| fields | `{"type": "standardize", "statistics_source": "train", "valid_region_only": true}` |

| 項目 | 値 |
| --- | --- |
| normalization_statistics | `{"statistics_source": "train", "valid_region_only": true, "sample_count": 2100, "valid_point_count": 33999800, "train_re": [10, 20, 30, 40, 60, 70, 80, 90, 110, 120, 130, 140, 160, 170, 180, 190, 210, 220, 230, 240, 250], "fields": {"u": {"mean": 0.01956091086922085, "std": 0.014513064333257285}, "v": {"mean": 3.338172258003842e-06, "std": 0.006574265401363537}, "p": {"mean": 4.195688495005484e-05, "std": 0.00017544342254628847}}}` |

統計metadataのtrain_reとDataLoaderのtrain_reが一致するかを下記に記載します。統計値の再計算は行っていません。

## 5. Train / Validation split・DataLoader

| 項目 | 値 |
| --- | --- |
| dataset_config | `"dataset.yaml"` |
| split | `{"train_re": [10, 20, 30, 40, 60, 70, 80, 90, 110, 120, 130, 140, 160, 170, 180, 190, 210, 220, 230, 240, 250], "val_re": [50, 100, 150, 200]}` |
| loader | `{"batch_size": 8, "train_shuffle": true, "val_shuffle": false, "num_workers": 0, "pin_memory": true, "drop_last": false, "persistent_workers": false, "seed": 42}` |

統計metadataと学習splitのtrain_re: **一致**。

## 6. モデル

| 項目 | 値 |
| --- | --- |
| name | `"FNO"` |
| implementation | `"neuralop"` |
| dataset_config | `"dataset.yaml"` |
| dataloader_config | `"dataloader.yaml"` |
| n_modes | `[24, 24]` |
| hidden_channels | `256` |
| lifting_channels | `64` |
| projection_channels | `64` |
| n_layers | `4` |
| activation | `"gelu"` |
| positional_embedding | `null` |
| domain_padding | `0.0` |
| norm | `null` |
| use_channel_mlp | `true` |
| channel_mlp_dropout | `0.0` |
| channel_mlp_expansion | `0.5` |
| channel_mlp_skip | `"soft-gating"` |
| fno_skip | `"linear"` |
| fno_block_precision | `"full"` |
| factorization | `null` |
| rank | `1.0` |
| fixed_rank_modes | `false` |
| weight_implementation | `"factorized"` |
| separable | `false` |
| preactivation | `false` |

## 7. 学習条件

| 項目 | 値 |
| --- | --- |
| training | `{"model_config": "model.yaml", "epochs": 250, "seed": 42, "device": "cuda", "deterministic": false, "gradient_clip_norm": null, "resume_checkpoint": null}` |
| optimizer | `{"type": "adam", "learning_rate": 0.0005, "weight_decay": 0.0}` |
| scheduler | `{"type": "cosine_annealing", "min_learning_rate": 1e-06}` |
| loss | `{"type": "mse", "valid_region_only": false}` |
| output | `{"directory": "../outputs/training/ex007", "checkpoint_interval": 10}` |
| wandb | `{"enabled": true, "project": "fnov2-cylinder-wake", "entity": null, "name": null, "group": "baseline", "job_type": "train", "tags": ["fno", "baseline", "one-step"], "notes": "Baseline FNO one-step prediction", "mode": "online", "save_code": true, "log_interval_steps": 10, "run_id": null, "resume": "allow", "watch": {"enabled": false, "log": "all", "log_freq": 100}, "log_best_checkpoint": true, "artifact_name": "fnov2-fno-baseline", "log_smoke_test": false}` |

## 8. 学習結果

出典: [history.json](../../../outputs/training/ex007/history.json) / [history.csv](../../../outputs/training/ex007/history.csv)。Bestは保存履歴のvalidation loss最小行です。

| 指標 | Best epoch 242 | Last epoch 250 |
| --- | ---: | ---: |
| train_loss | 1.1164365e-06 | 1.0957296e-06 |
| val_loss | 7.3440643e-07 | 7.3858702e-07 |
| train_relative_l2 | 0.0010629098 | 0.0010530066 |
| val_relative_l2 | 0.001057436 | 0.0010604414 |
| learning_rate | 2.593978e-06 | 1.0196995e-06 |

設定epoch数: 250。履歴: 250行、epoch 1–250。記録行のelapsed_seconds合計: 24757.03秒。これは保存履歴の合計であり、未記録実行の時間を含みません。

- [best.pt](../../../outputs/training/ex007/best.pt)
- [last.pt](../../../outputs/training/ex007/last.pt)
- [history.json](../../../outputs/training/ex007/history.json)
- [history.csv](../../../outputs/training/ex007/history.csv)

## 9. One-step評価

物理スケール・有効領域（physical/valid_region）のall指標を掲載しています。全体値はmetrics.jsonの集計値であり、Re別値の単純平均ではありません。

### interpolation

checkpoint epoch: 242、対象Re: [35.0, 85.0, 135.0, 185.0, 235.0]、サンプル数: 500。

| Re | MSE | Relative L2 |
| ---: | ---: | ---: |
| 35 | 2.44661691e-12 | 0.00047541 |
| 85 | 3.81768630e-11 | 0.00076444 |
| 135 | 2.12544676e-11 | 0.00035541 |
| 185 | 3.66537913e-11 | 0.00033875 |
| 235 | 7.13529704e-11 | 0.00037061 |
| 全体 | 3.39770140e-11 | 0.00039600 |

出典: [metrics.json](interpolation/one_step/metrics.json) / [metrics.csv](interpolation/one_step/metrics.csv)

### extrapolation

checkpoint epoch: 242、対象Re: [260.0, 270.0, 280.0, 290.0, 300.0]、サンプル数: 500。

| Re | MSE | Relative L2 |
| ---: | ---: | ---: |
| 260 | 3.80457555e-10 | 0.00077227 |
| 270 | 1.06205702e-09 | 0.00124181 |
| 280 | 2.72062423e-09 | 0.00191554 |
| 290 | 6.06470966e-09 | 0.00275979 |
| 300 | 1.22746283e-08 | 0.00379344 |
| 全体 | 4.50042254e-09 | 0.00246034 |

出典: [metrics.json](extrapolation/one_step/metrics.json) / [metrics.csv](extrapolation/one_step/metrics.csv)

## 10. Rollout評価

保存済みCSVのRe別・step別overall指標を集計しています。平均は記録stepの算術平均、最終は最大stepの行です。時刻条件はconfig_snapshot.mdを参照してください。評価時のcheckpointはCSVに記録されていないため、one-stepと同一だったとは断定していません。

### interpolation

| Re | step数 | 平均Relative L2 | 最終Relative L2 | 最大Relative L2 | 平均MSE | 最終MSE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 35 | 100 | 0.01924323 | 0.06303401 | 0.06303401 | 6.41395699e-09 | 4.30095380e-08 |
| 85 | 100 | 0.01928187 | 0.06586110 | 0.06586110 | 3.47628453e-08 | 2.83255520e-07 |
| 135 | 100 | 0.00557182 | 0.01021333 | 0.01021333 | 6.14482154e-09 | 1.75379598e-08 |
| 185 | 100 | 0.00408427 | 0.00583314 | 0.00583314 | 5.87199442e-09 | 1.08579614e-08 |
| 235 | 100 | 0.00362206 | 0.00512450 | 0.00512450 | 7.34318523e-09 | 1.36336032e-08 |

出典: [rollout_metrics.csv](interpolation/rollout/rollout_metrics.csv)

### extrapolation

| Re | step数 | 平均Relative L2 | 最終Relative L2 | 最大Relative L2 | 平均MSE | 最終MSE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 260 | 100 | 0.01817490 | 0.11522409 | 0.11522409 | 5.50143245e-07 | 8.47063691e-06 |
| 270 | 100 | 0.02352831 | 0.08016563 | 0.08016563 | 5.72588323e-07 | 4.42388938e-06 |
| 280 | 100 | 0.06006734 | 0.26219582 | 0.26219582 | 5.95877433e-06 | 5.09506585e-05 |
| 290 | 100 | 0.18771418 | 0.51634158 | 0.51902176 | 5.94529352e-05 | 2.12125992e-04 |
| 300 | 100 | 0.09058399 | 0.24644105 | 0.24644105 | 1.00064292e-05 | 5.17921927e-05 |

出典: [rollout_metrics.csv](extrapolation/rollout/rollout_metrics.csv)

## 11. 再確認・再現時の注意

- モデル設定とcheckpointは同じ実験の組み合わせを使用します。
- 評価時のYAML全文が未保存の条件は、現在のYAMLから推定していません。
- 学習履歴のBestと評価metadataのcheckpoint epochが異なる場合は、checkpoint本体と履歴の対応を再確認してください。
- one-step精度とautoregressive rolloutの誤差蓄積は分けて判断し、場の画像も確認してください。

Notebook: [学習](../../../experiment/training/training.ipynb) / [One-step](../../../experiment/evaluation/one_step.ipynb) / [Rollout](../../../experiment/evaluation/rollout.ipynb)
