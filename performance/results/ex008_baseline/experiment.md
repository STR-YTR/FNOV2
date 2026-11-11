# ex008_baseline 実験記録

## 1. このファイルの役割

保存済みの学習設定・学習履歴・performance評価結果を集約した実験台帳です。記録作成日: 2026-11-11（日本時間）。学習・評価の再実行は行っていません。

設定全文と出典: [config_snapshot.md](config_snapshot.md)。現在の共通設定は変更され得るため、過去の実行条件の根拠には使用していません。

## 2. 実験識別情報

| 項目 | 値 |
| --- | --- |
| 実験ID | `"ex008_baseline"` |
| 学習保存先 | `"outputs/training/ex008"` |
| 評価checkpoint | `{"interpolation": "/workspace/FNOV2/outputs/training/ex008/best.pt", "extrapolation": "/workspace/FNOV2/outputs/training/ex008/best.pt"}` |
| 評価checkpoint epoch | `{"interpolation": 256, "extrapolation": 256}` |

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
| loader | `{"batch_size": 4, "train_shuffle": true, "val_shuffle": false, "num_workers": 0, "pin_memory": true, "drop_last": false, "persistent_workers": false, "seed": 42}` |

統計metadataと学習splitのtrain_re: **一致**。

## 6. モデル

| 項目 | 値 |
| --- | --- |
| name | `"FNO"` |
| implementation | `"neuralop"` |
| dataset_config | `"dataset.yaml"` |
| dataloader_config | `"dataloader.yaml"` |
| n_modes | `[24, 24]` |
| hidden_channels | `518` |
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
| training | `{"model_config": "model.yaml", "epochs": 300, "seed": 42, "device": "cuda", "deterministic": false, "gradient_clip_norm": null, "resume_checkpoint": null}` |
| optimizer | `{"type": "adam", "learning_rate": 0.0005, "weight_decay": 0.0}` |
| scheduler | `{"type": "cosine_annealing", "min_learning_rate": 1e-06}` |
| loss | `{"type": "mse", "valid_region_only": false}` |
| output | `{"directory": "../outputs/training/ex008", "checkpoint_interval": 10}` |
| wandb | `{"enabled": true, "project": "fnov2-cylinder-wake", "entity": null, "name": null, "group": "baseline", "job_type": "train", "tags": ["fno", "baseline", "one-step"], "notes": "Baseline FNO one-step prediction", "mode": "online", "save_code": true, "log_interval_steps": 10, "run_id": null, "resume": "allow", "watch": {"enabled": false, "log": "all", "log_freq": 100}, "log_best_checkpoint": true, "artifact_name": "fnov2-fno-baseline", "log_smoke_test": false}` |

## 8. 学習結果

出典: [history.json](../../../outputs/training/ex008/history.json) / [history.csv](../../../outputs/training/ex008/history.csv)。Bestは保存履歴のvalidation loss最小行です。

| 指標 | Best epoch 256 | Last epoch 259 |
| --- | ---: | ---: |
| train_loss | 1.2849668e-06 | 1.2361658e-06 |
| val_loss | 6.5581187e-07 | 7.3801428e-07 |
| train_relative_l2 | 0.0011403164 | 0.001118453 |
| val_relative_l2 | 0.00099925308 | 0.0010600302 |
| learning_rate | 2.8193872e-05 | 2.474565e-05 |

設定epoch数: 300。履歴: 259行、epoch 1–259。記録行のelapsed_seconds合計: 65603.52秒。これは保存履歴の合計であり、未記録実行の時間を含みません。

- [best.pt](../../../outputs/training/ex008/best.pt)
- [last.pt](../../../outputs/training/ex008/last.pt)
- [history.json](../../../outputs/training/ex008/history.json)
- [history.csv](../../../outputs/training/ex008/history.csv)

## 9. One-step評価

物理スケール・有効領域（physical/valid_region）のall指標を掲載しています。全体値はmetrics.jsonの集計値であり、Re別値の単純平均ではありません。

### interpolation

checkpoint epoch: 256、対象Re: [35.0, 85.0, 135.0, 185.0, 235.0]、サンプル数: 500。

| Re | MSE | Relative L2 |
| ---: | ---: | ---: |
| 35 | 1.19189655e-11 | 0.00104931 |
| 85 | 2.78922771e-11 | 0.00065341 |
| 135 | 1.48288524e-11 | 0.00029687 |
| 185 | 2.57464929e-11 | 0.00028391 |
| 235 | 5.06874733e-11 | 0.00031237 |
| 全体 | 2.62149381e-11 | 0.00034784 |

出典: [metrics.json](interpolation/one_step/metrics.json) / [metrics.csv](interpolation/one_step/metrics.csv)

### extrapolation

checkpoint epoch: 256、対象Re: [260.0, 270.0, 280.0, 290.0, 300.0]、サンプル数: 500。

| Re | MSE | Relative L2 |
| ---: | ---: | ---: |
| 260 | 2.67594720e-10 | 0.00064767 |
| 270 | 8.66702405e-10 | 0.00112180 |
| 280 | 2.48679536e-09 | 0.00183137 |
| 290 | 6.11744706e-09 | 0.00277176 |
| 300 | 1.29888566e-08 | 0.00390224 |
| 全体 | 4.54540096e-09 | 0.00247261 |

出典: [metrics.json](extrapolation/one_step/metrics.json) / [metrics.csv](extrapolation/one_step/metrics.csv)

## 10. Rollout評価

保存済みCSVのRe別・step別overall指標を集計しています。平均は記録stepの算術平均、最終は最大stepの行です。時刻条件はconfig_snapshot.mdを参照してください。評価時のcheckpointはCSVに記録されていないため、one-stepと同一だったとは断定していません。

### interpolation

| Re | step数 | 平均Relative L2 | 最終Relative L2 | 最大Relative L2 | 平均MSE | 最終MSE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 35 | 100 | 0.01349438 | 0.02672510 | 0.02672510 | 2.60363934e-09 | 7.73132144e-09 |
| 85 | 100 | 0.01601947 | 0.02868345 | 0.02868345 | 2.13486693e-08 | 5.37258460e-08 |
| 135 | 100 | 0.00705847 | 0.01446001 | 0.01446001 | 1.08564141e-08 | 3.51545126e-08 |
| 185 | 100 | 0.00623887 | 0.01338291 | 0.01338291 | 1.60645923e-08 | 5.71538014e-08 |
| 235 | 100 | 0.00586624 | 0.01290886 | 0.01290886 | 2.28497419e-08 | 8.65135456e-08 |

出典: [rollout_metrics.csv](interpolation/rollout/rollout_metrics.csv)

### extrapolation

| Re | step数 | 平均Relative L2 | 最終Relative L2 | 最大Relative L2 | 平均MSE | 最終MSE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 260 | 100 | 0.01364160 | 0.02845309 | 0.02845309 | 1.54933953e-07 | 5.16521436e-07 |
| 270 | 100 | 0.02630878 | 0.05710305 | 0.05710305 | 6.31970498e-07 | 2.24463858e-06 |
| 280 | 100 | 0.04595885 | 0.09960956 | 0.09960956 | 2.08970321e-06 | 7.35360540e-06 |
| 290 | 100 | 0.07458684 | 0.16232514 | 0.16232514 | 5.89304458e-06 | 2.09648250e-05 |
| 300 | 100 | 0.10885824 | 0.23259214 | 0.23259214 | 1.35283293e-05 | 4.61347608e-05 |

出典: [rollout_metrics.csv](extrapolation/rollout/rollout_metrics.csv)

## 11. 再確認・再現時の注意

- モデル設定とcheckpointは同じ実験の組み合わせを使用します。
- 評価時のYAML全文が未保存の条件は、現在のYAMLから推定していません。
- 学習履歴のBestと評価metadataのcheckpoint epochが異なる場合は、checkpoint本体と履歴の対応を再確認してください。
- one-step精度とautoregressive rolloutの誤差蓄積は分けて判断し、場の画像も確認してください。

Notebook: [学習](../../../experiment/training/training.ipynb) / [One-step](../../../experiment/evaluation/one_step.ipynb) / [Rollout](../../../experiment/evaluation/rollout.ipynb)
