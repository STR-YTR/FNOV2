# ex004_baseline 実験記録

## 1. このファイルの役割

保存済みの学習設定・学習履歴・performance評価結果を集約した実験台帳です。記録作成日: 2026-11-11（日本時間）。学習・評価の再実行は行っていません。

設定全文と出典: [config_snapshot.md](config_snapshot.md)。現在の共通設定は変更され得るため、過去の実行条件の根拠には使用していません。

## 2. 実験識別情報

| 項目 | 値 |
| --- | --- |
| 実験ID | `"ex004_baseline"` |
| 学習保存先 | `"outputs/training/ex004"` |
| 評価checkpoint | `{"interpolation": "/workspace/FNOV2/outputs/training/ex004/best.pt", "extrapolation": "/workspace/FNOV2/outputs/training/ex004/best.pt"}` |
| 評価checkpoint epoch | `{"interpolation": 196, "extrapolation": 196}` |

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
| n_modes | `[12, 12]` |
| hidden_channels | `64` |
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
| training | `{"model_config": "model.yaml", "epochs": 200, "seed": 42, "device": "cuda", "deterministic": false, "gradient_clip_norm": null, "resume_checkpoint": null}` |
| optimizer | `{"type": "adam", "learning_rate": 0.0005, "weight_decay": 0.0}` |
| scheduler | `{"type": "cosine_annealing", "min_learning_rate": 1e-06}` |
| loss | `{"type": "mse", "valid_region_only": false}` |
| output | `{"directory": "../outputs/training/ex004", "checkpoint_interval": 10}` |
| wandb | `{"enabled": true, "project": "fnov2-cylinder-wake", "entity": null, "name": null, "group": "baseline", "job_type": "train", "tags": ["fno", "baseline", "one-step"], "notes": "Baseline FNO one-step prediction", "mode": "online", "save_code": true, "log_interval_steps": 10, "run_id": null, "resume": "allow", "watch": {"enabled": false, "log": "all", "log_freq": 100}, "log_best_checkpoint": true, "artifact_name": "fnov2-fno-baseline", "log_smoke_test": false}` |

## 8. 学習結果

出典: [history.json](../../../outputs/training/ex004/history.json) / [history.csv](../../../outputs/training/ex004/history.csv)。Bestは保存履歴のvalidation loss最小行です。

| 指標 | Best epoch 196 | Last epoch 200 |
| --- | ---: | ---: |
| train_loss | 3.5287123e-06 | 3.5034353e-06 |
| val_loss | 2.4957947e-06 | 2.5136356e-06 |
| train_relative_l2 | 0.0018896768 | 0.0018828965 |
| val_relative_l2 | 0.0019493521 | 0.001956307 |
| learning_rate | 1.7691252e-06 | 1.0307802e-06 |

設定epoch数: 200。履歴: 200行、epoch 1–200。記録行のelapsed_seconds合計: 11950.99秒。これは保存履歴の合計であり、未記録実行の時間を含みません。

- [best.pt](../../../outputs/training/ex004/best.pt)
- [last.pt](../../../outputs/training/ex004/last.pt)
- [history.json](../../../outputs/training/ex004/history.json)
- [history.csv](../../../outputs/training/ex004/history.csv)

## 9. One-step評価

物理スケール・有効領域（physical/valid_region）のall指標を掲載しています。全体値はmetrics.jsonの集計値であり、Re別値の単純平均ではありません。

### interpolation

checkpoint epoch: 196、対象Re: [35.0, 85.0, 135.0, 185.0, 235.0]、サンプル数: 500。

| Re | MSE | Relative L2 |
| ---: | ---: | ---: |
| 35 | 4.41923701e-10 | 0.00638937 |
| 85 | 6.99181379e-10 | 0.00327144 |
| 135 | 1.07306228e-09 | 0.00252536 |
| 185 | 3.34606521e-09 | 0.00323656 |
| 235 | 1.17375744e-08 | 0.00475339 |
| 全体 | 3.45962639e-09 | 0.00399595 |

出典: [metrics.json](interpolation/one_step/metrics.json) / [metrics.csv](interpolation/one_step/metrics.csv)

### extrapolation

checkpoint epoch: 196、対象Re: [260.0, 270.0, 280.0, 290.0, 300.0]、サンプル数: 500。

| Re | MSE | Relative L2 |
| ---: | ---: | ---: |
| 260 | 2.14857845e-08 | 0.00580355 |
| 270 | 2.83360931e-08 | 0.00641433 |
| 280 | 3.60193070e-08 | 0.00696985 |
| 290 | 4.73489301e-08 | 0.00771128 |
| 300 | 6.34111415e-08 | 0.00862207 |
| 全体 | 3.93199900e-08 | 0.00727237 |

出典: [metrics.json](extrapolation/one_step/metrics.json) / [metrics.csv](extrapolation/one_step/metrics.csv)

## 10. Rollout評価

保存済みCSVのRe別・step別overall指標を集計しています。平均は記録stepの算術平均、最終は最大stepの行です。時刻条件はconfig_snapshot.mdを参照してください。評価時のcheckpointはCSVに記録されていないため、one-stepと同一だったとは断定していません。

### interpolation

| Re | step数 | 平均Relative L2 | 最終Relative L2 | 最大Relative L2 | 平均MSE | 最終MSE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 35 | 100 | 0.11045058 | 0.19661674 | 0.19661674 | 1.61003167e-07 | 4.18461891e-07 |
| 85 | 100 | 0.08184506 | 0.16035256 | 0.16035256 | 5.59882615e-07 | 1.67908321e-06 |
| 135 | 100 | 0.09871920 | 0.24192088 | 0.24192088 | 2.33925867e-06 | 9.83990218e-06 |
| 185 | 100 | 0.10278471 | 0.22083272 | 0.22083272 | 4.56687908e-06 | 1.55621911e-05 |
| 235 | 100 | 0.12428322 | 0.28569305 | 0.28569305 | 1.09146165e-05 | 4.23747855e-05 |

出典: [rollout_metrics.csv](interpolation/rollout/rollout_metrics.csv)

### extrapolation

| Re | step数 | 平均Relative L2 | 最終Relative L2 | 最大Relative L2 | 平均MSE | 最終MSE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 260 | 100 | 0.14793955 | 0.30822718 | 0.30822718 | 1.83794795e-05 | 6.06137749e-05 |
| 270 | 100 | 0.16196348 | 0.35274414 | 0.35274414 | 2.45947221e-05 | 8.56539534e-05 |
| 280 | 100 | 0.17272688 | 0.38651580 | 0.38651580 | 2.95497181e-05 | 1.10721710e-04 |
| 290 | 100 | 0.19433943 | 0.43287076 | 0.43287076 | 4.13727813e-05 | 1.49085764e-04 |
| 300 | 100 | 0.21118764 | 0.46419499 | 0.46419499 | 5.10766255e-05 | 1.83754957e-04 |

出典: [rollout_metrics.csv](extrapolation/rollout/rollout_metrics.csv)

## 11. 再確認・再現時の注意

- モデル設定とcheckpointは同じ実験の組み合わせを使用します。
- 評価時のYAML全文が未保存の条件は、現在のYAMLから推定していません。
- 学習履歴のBestと評価metadataのcheckpoint epochが異なる場合は、checkpoint本体と履歴の対応を再確認してください。
- one-step精度とautoregressive rolloutの誤差蓄積は分けて判断し、場の画像も確認してください。

Notebook: [学習](../../../experiment/training/training.ipynb) / [One-step](../../../experiment/evaluation/one_step.ipynb) / [Rollout](../../../experiment/evaluation/rollout.ipynb)
