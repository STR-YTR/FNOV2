# ex002_baseline 実験記録

## 1. このファイルの役割

保存済みの学習設定・学習履歴・performance評価結果を集約した実験台帳です。記録作成日: 2026-11-11（日本時間）。学習・評価の再実行は行っていません。

設定全文と出典: [config_snapshot.md](config_snapshot.md)。現在の共通設定は変更され得るため、過去の実行条件の根拠には使用していません。

## 2. 実験識別情報

| 項目 | 値 |
| --- | --- |
| 実験ID | `"ex002_baseline"` |
| 学習保存先 | `"outputs/training/ex002"` |
| 評価checkpoint | `{"interpolation": "/workspace/FNOV2/outputs/training/ex002/best.pt", "extrapolation": "/workspace/FNOV2/outputs/training/ex002/best.pt"}` |
| 評価checkpoint epoch | `{"interpolation": 99, "extrapolation": 99}` |

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
| normalization_statistics | `{"statistics_source": "train", "valid_region_only": true, "sample_count": 2100, "valid_point_count": 33999800, "train_re": [80, 90, 110, 120, 130, 140, 160, 170, 180, 190, 210, 220, 230, 240, 250], "fields": {"u": {"mean": 0.01956091086922085, "std": 0.014513064333257285}, "v": {"mean": 3.338172258003842e-06, "std": 0.006574265401363537}, "p": {"mean": 4.195688495005484e-05, "std": 0.00017544342254628847}}}` |

統計metadataのtrain_reとDataLoaderのtrain_reが一致するかを下記に記載します。統計値の再計算は行っていません。

## 5. Train / Validation split・DataLoader

| 項目 | 値 |
| --- | --- |
| dataset_config | `"dataset.yaml"` |
| split | `{"train_re": [80, 90, 110, 120, 130, 140, 160, 170, 180, 190, 210, 220, 230, 240, 250], "val_re": [100, 150, 200]}` |
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
| hidden_channels | `32` |
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
| training | `{"model_config": "model.yaml", "epochs": 100, "seed": 42, "device": "cuda", "deterministic": false, "gradient_clip_norm": null, "resume_checkpoint": null}` |
| optimizer | `{"type": "adam", "learning_rate": 0.0005, "weight_decay": 0.0}` |
| scheduler | `{"type": "cosine_annealing", "min_learning_rate": 1e-06}` |
| loss | `{"type": "mse", "valid_region_only": false}` |
| output | `{"directory": "../outputs/training/ex002", "checkpoint_interval": 10}` |
| wandb | `{"enabled": true, "project": "fnov2-cylinder-wake", "entity": null, "name": null, "group": "baseline", "job_type": "train", "tags": ["fno", "baseline", "one-step"], "notes": "Baseline FNO one-step prediction", "mode": "online", "save_code": true, "log_interval_steps": 10, "run_id": null, "resume": "allow", "watch": {"enabled": false, "log": "all", "log_freq": 100}, "log_best_checkpoint": true, "artifact_name": "fnov2-fno-baseline", "log_smoke_test": false}` |

## 8. 学習結果

出典: [history.json](../../../outputs/training/ex002/history.json) / [history.csv](../../../outputs/training/ex002/history.csv)。Bestは保存履歴のvalidation loss最小行です。

| 指標 | Best epoch 99 | Last epoch 100 |
| --- | ---: | ---: |
| train_loss | 2.3968382e-05 | 2.3927387e-05 |
| val_loss | 1.5643605e-05 | 1.5794488e-05 |
| train_relative_l2 | 0.0043870198 | 0.0043832665 |
| val_relative_l2 | 0.0044536854 | 0.0044751118 |
| learning_rate | 1.4923313e-06 | 1.1231132e-06 |

設定epoch数: 100。履歴: 100行、epoch 1–100。記録行のelapsed_seconds合計: 4050.51秒。これは保存履歴の合計であり、未記録実行の時間を含みません。

- [best.pt](../../../outputs/training/ex002/best.pt)
- [last.pt](../../../outputs/training/ex002/last.pt)
- [history.json](../../../outputs/training/ex002/history.json)
- [history.csv](../../../outputs/training/ex002/history.csv)

## 9. One-step評価

物理スケール・有効領域（physical/valid_region）のall指標を掲載しています。全体値はmetrics.jsonの集計値であり、Re別値の単純平均ではありません。

### interpolation

checkpoint epoch: 99、対象Re: [35.0, 85.0, 135.0, 185.0, 235.0]、サンプル数: 500。

| Re | MSE | Relative L2 |
| ---: | ---: | ---: |
| 35 | 1.61397697e-08 | 0.03861299 |
| 85 | 6.11706572e-10 | 0.00305996 |
| 135 | 7.23236510e-10 | 0.00207324 |
| 185 | 1.34077391e-09 | 0.00204878 |
| 235 | 2.98306284e-09 | 0.00239632 |
| 全体 | 4.35983843e-09 | 0.00448581 |

出典: [metrics.json](interpolation/one_step/metrics.json) / [metrics.csv](interpolation/one_step/metrics.csv)

### extrapolation

checkpoint epoch: 99、対象Re: [260.0, 270.0, 280.0, 290.0, 300.0]、サンプル数: 500。

| Re | MSE | Relative L2 |
| ---: | ---: | ---: |
| 260 | 1.02124931e-08 | 0.00400114 |
| 270 | 1.89596178e-08 | 0.00524682 |
| 280 | 3.43891569e-08 | 0.00681031 |
| 290 | 6.08911821e-08 | 0.00874477 |
| 300 | 1.00516313e-07 | 0.01085543 |
| 全体 | 4.49931920e-08 | 0.00777934 |

出典: [metrics.json](extrapolation/one_step/metrics.json) / [metrics.csv](extrapolation/one_step/metrics.csv)

## 10. Rollout評価

保存済みCSVのRe別・step別overall指標を集計しています。平均は記録stepの算術平均、最終は最大stepの行です。時刻条件はconfig_snapshot.mdを参照してください。評価時のcheckpointはCSVに記録されていないため、one-stepと同一だったとは断定していません。

### interpolation

| Re | step数 | 平均Relative L2 | 最終Relative L2 | 最大Relative L2 | 平均MSE | 最終MSE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 35 | 100 | 7.99852150 | 16.63310327 | 16.63310327 | 1.09218297e-03 | 2.99475592e-03 |
| 85 | 100 | 2.51900808 | 5.90353389 | 5.90353389 | 7.01569605e-04 | 2.27585451e-03 |
| 135 | 100 | 1.22108929 | 2.67855301 | 2.67855301 | 3.59424005e-04 | 1.20627026e-03 |
| 185 | 100 | 1.11179634 | 2.08299443 | 2.08299443 | 5.42774998e-04 | 1.38458667e-03 |
| 235 | 100 | 0.96178202 | 1.66787138 | 1.66787138 | 6.22128027e-04 | 1.44421974e-03 |

出典: [rollout_metrics.csv](interpolation/rollout/rollout_metrics.csv)

### extrapolation

| Re | step数 | 平均Relative L2 | 最終Relative L2 | 最大Relative L2 | 平均MSE | 最終MSE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 260 | 100 | 0.99880884 | 1.74775659 | 1.74775659 | 8.40242189e-04 | 1.94890828e-03 |
| 270 | 100 | 0.82896586 | 1.44239415 | 1.44239415 | 5.91916847e-04 | 1.43217370e-03 |
| 280 | 100 | 0.83440304 | 1.32456853 | 1.32456853 | 6.36739827e-04 | 1.30031071e-03 |
| 290 | 100 | 0.74616611 | 1.20561265 | 1.20561265 | 5.34477818e-04 | 1.15647240e-03 |
| 300 | 100 | 0.77381396 | 1.16342381 | 1.16342381 | 6.15884817e-04 | 1.15428781e-03 |

出典: [rollout_metrics.csv](extrapolation/rollout/rollout_metrics.csv)

## 11. 再確認・再現時の注意

- モデル設定とcheckpointは同じ実験の組み合わせを使用します。
- 評価時のYAML全文が未保存の条件は、現在のYAMLから推定していません。
- 学習履歴のBestと評価metadataのcheckpoint epochが異なる場合は、checkpoint本体と履歴の対応を再確認してください。
- one-step精度とautoregressive rolloutの誤差蓄積は分けて判断し、場の画像も確認してください。

Notebook: [学習](../../../experiment/training/training.ipynb) / [One-step](../../../experiment/evaluation/one_step.ipynb) / [Rollout](../../../experiment/evaluation/rollout.ipynb)
