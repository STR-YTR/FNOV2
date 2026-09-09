# ex001_baseline 実験記録

## 1. このファイルの役割

本ファイルは、FNO baselineのデータ生成条件、Dataset、モデル、学習条件、学習結果、およびperformance評価結果を集約した実験台帳です。

- 実際にコードが参照する設定の正本は各YAMLです。
- Notebookは設定確認、事前チェック、実行、結果可視化に使用します。
- 本ファイルは結果を読み返すためのスナップショットです。YAMLを変更した場合は内容も更新してください。

### Notebook

- [学習Notebook](../../../experiment/training/training.ipynb)
- [One-step評価Notebook](../../../experiment/evaluation/one_step.ipynb)
- [Rollout評価Notebook](../../../experiment/evaluation/rollout.ipynb)

## 2. 実験識別情報

| 項目 | 値 |
|---|---|
| 実験ID | `ex001_baseline` |
| 実験名 | Baseline FNO one-step prediction |
| 学習checkpoint | [outputs/training/baseline/best.pt](../../../outputs/training/baseline/best.pt) |
| checkpoint epoch | 96 |
| checkpoint内Git commit | `38c527859883b3e41ab7bed92b7e64d1b87441f5` |
| 本記録作成時のHEAD | `237c90dea9ed08874171aba5eab3c7235b58de1a` |
| Neural Operator | 2.0.0 |
| checkpoint global step | 25,248 |
| 実行日時 | 学習履歴に未記録 |

Notebookと本ファイルは、本記録作成時点では未コミットです。再現時はcheckpoint内Git commitと、checkpointに保存されたtrain configを優先して確認してください。

## 3. データ生成

設定: [configs/make_pairs.yaml](../../../configs/make_pairs.yaml)

### 入出力

| 項目 | 値 |
|---|---|
| 入力 | `data/VTK` |
| pair出力 | `data/processed/pairs` |
| 入力形式 | `case(?P<case>\d+)_(?P<step>\d+)\.vtm` |
| pair形式 | `pairs_case{case}_Re{Re}.pt` |
| 探索 | recursive |
| 速度フィールド | `U` |
| 圧力フィールド | `p` |
| 保存済みpair | 30ファイル |

case番号はディレクトリ内だけのラベルです。物理ケースの識別にはpair metadataのReを使用します。

### 時間条件

| 項目 | 値 |
|---|---:|
| 元データの1 step | 0.01 s |
| 使用時間範囲 | 340.0–350.0 s |
| one-step pair間隔 | 0.1 s |
| tolerance | 0.000001 |

各Reのpairは100サンプルです。学習21ケースで2,100サンプル、validation 4ケースで400サンプルを使用します。

### 格子と補間

| 項目 | 値 |
|---|---|
| 格子 | 2次元規則格子 |
| 解像度 | 256 × 64 |
| bounds | VTMから自動取得 |
| z | VTMのz方向中央 |
| 補間 | linear |

OpenFOAM由来の非構造格子を規則格子へ補間しています。

### 生成対象Re

元のmake-pairs設定にはRe 10から300まで10刻みの30ケースが登録されています。ただし学習・validationで実際に使用するReはDataLoaderのsplitで限定されます。

## 4. Datasetとチャネル

設定: [configs/dataset.yaml](../../../configs/dataset.yaml)

### 入力

```text
[mask, Re, x, y, u, v, p]
```

shapeは1サンプルあたり `[7, 64, 256]` です。

### 教師出力

```text
[u(t+0.1), v(t+0.1), p(t+0.1)]
```

shapeは1サンプルあたり `[3, 64, 256]` です。

pairファイルは物理量の生値を保持し、Dataset読込時に正規化・標準化します。直近のpairファイルだけを保持するfile cacheが有効です。

## 5. Normalization

設定: [configs/normalization_stats.yaml](../../../configs/normalization_stats.yaml)

| チャネル | 変換 |
|---|---|
| mask | 変換なし（0/1） |
| Re | `Re / 250` |
| x | `x / 1.6` |
| y | `y / 0.41` |
| u, v, p | trainデータのmean/stdでstandardization |

u/v/p統計量は、train Reのみの有効領域（`mask == 1`）から算出しています。

| 物理量 | mean | std |
|---|---:|---:|
| u | 0.01956091086922085 | 0.014513064333257285 |
| v | 0.000003338172258003842 | 0.006574265401363537 |
| p | 0.00004195688495005484 | 0.00017544342254628847 |

| 統計metadata | 値 |
|---|---:|
| sample count | 2,100 |
| valid point count | 33,999,800 |
| statistics source | train |
| valid region only | true |

validation、performance interpolation、performance extrapolationでも、このtrain統計量を固定して使用します。

## 6. Train / Validation split

設定: [configs/dataloader.yaml](../../../configs/dataloader.yaml)

### Train Re（21ケース）

```text
10, 20, 30, 40, 60, 70, 80, 90, 110, 120, 130, 140,
160, 170, 180, 190, 210, 220, 230, 240, 250
```

### Validation Re（4ケース）

```text
50, 100, 150, 200
```

trainとvalidationはReケース単位で分離されており、重複はありません。学習用pairディレクトリに存在するRe 260–300は、現在のsplitでは学習にもvalidationにも使用されません。

### DataLoader

| 項目 | 値 |
|---|---:|
| batch size | 8 |
| train shuffle | true |
| validation shuffle | false |
| num workers | 0 |
| pin memory | true |
| drop last | false |
| persistent workers | false |
| seed | 42 |

## 7. モデル

設定: [configs/model.yaml](../../../configs/model.yaml)

| 項目 | 値 |
|---|---|
| model | FNO |
| implementation | neuralop |
| n modes | [12, 12] |
| hidden channels | 32 |
| lifting channels | 64 |
| projection channels | 64 |
| layers | 4 |
| activation | GELU |
| positional embedding | 無効 |
| domain padding | 0.0 |
| norm | null |
| channel MLP | 有効 |
| channel MLP dropout | 0.0 |
| channel MLP expansion | 0.5 |
| channel MLP skip | soft-gating |
| FNO skip | linear |
| block precision | full |
| factorization | null |
| rank | 1.0 |
| fixed rank modes | false |
| weight implementation | factorized |
| separable | false |
| preactivation | false |

入出力チャネル数はDataset設定から自動導出されます。

## 8. 学習条件

設定: [configs/train.yaml](../../../configs/train.yaml)

| 項目 | 値 |
|---|---|
| epochs | 100 |
| seed | 42 |
| device | auto |
| deterministic | false |
| gradient clipping | なし |
| resume checkpoint | なし |
| optimizer | Adam |
| initial learning rate | 0.0005 |
| weight decay | 0.0 |
| scheduler | Cosine Annealing |
| minimum learning rate | 0.000001 |
| loss | standardized u/v/pの全格子MSE |
| valid region only | false |
| checkpoint interval | 10 epochs |
| output | `outputs/training/baseline` |

### W&B

| 項目 | 値 |
|---|---|
| enabled | true |
| mode | online |
| project | fnov2-cylinder-wake |
| group | baseline |
| job type | train |
| tags | fno, baseline, one-step |
| log interval | 10 steps |
| model watch | false |
| best checkpoint artifact | fnov2-fno-baseline |

## 9. 学習結果

履歴: [history.csv](../../../outputs/training/baseline/history.csv)

| 指標 | Best epoch 96 | Last epoch 100 |
|---|---:|---:|
| train MSE loss | 1.501873e-05 | 1.488462e-05 |
| validation MSE loss | 1.022395e-05 | 1.023039e-05 |
| train Relative L2 | 0.00389849 | 0.00388104 |
| validation Relative L2 | **0.00394544** | 0.00394668 |
| learning rate | 4.071759e-06 | 1.123113e-06 |

記録された100 epochの合計時間は6,280.43秒（約104.67分）です。

### 保存物

- [best.pt](../../../outputs/training/baseline/best.pt)
- [last.pt](../../../outputs/training/baseline/last.pt)
- [history.csv](../../../outputs/training/baseline/history.csv)
- [history.json](../../../outputs/training/baseline/history.json)
- epoch 10刻みのcheckpoint

## 10. One-step評価

評価は物理スケール、有効領域（`mask == 1`）のoverall値を掲載しています。内挿・外挿とも5ケース、合計500サンプルです。

### Interpolation

対象Re: 35, 85, 135, 185, 235

| Re | MSE | Relative L2 |
|---:|---:|---:|
| 35 | 7.326012e-11 | 0.00260147 |
| 85 | 4.568401e-10 | 0.00264439 |
| 135 | 6.270116e-10 | 0.00193040 |
| 185 | 1.126898e-09 | 0.00187827 |
| 235 | 2.456334e-09 | 0.00217449 |
| **全体** | **9.480766e-10** | **0.00209183** |

- [metrics.json](interpolation/one_step/metrics.json)
- [metrics.csv](interpolation/one_step/metrics.csv)
- [Relative L2グラフ](interpolation/one_step/interpolation_relative_l2.png)

### Extrapolation

対象Re: 260, 270, 280, 290, 300

| Re | Re−250 | MSE | Relative L2 |
|---:|---:|---:|---:|
| 260 | 10 | 8.912402e-09 | 0.00373779 |
| 270 | 20 | 1.713503e-08 | 0.00498797 |
| 280 | 30 | 3.216406e-08 | 0.00658630 |
| 290 | 40 | 5.697266e-08 | 0.00845872 |
| 300 | 50 | 9.500940e-08 | 0.01055387 |
| **全体** | — | **4.203818e-08** | **0.00751954** |

- [metrics.json](extrapolation/one_step/metrics.json)
- [metrics.csv](extrapolation/one_step/metrics.csv)
- [Relative L2グラフ](extrapolation/one_step/extrapolation_relative_l2.png)

## 11. Rollout評価

共通条件:

- 初期入力: 340.0 sのGT
- 予測範囲: 340.1–350.0 s
- step数: 100
- teacher forcing: なし
- 更新チャネル: u, v, pのみ
- metrics: 各stepの物理スケールMSE・Relative L2
- 正式結果: `performance/results/ex001_baseline/<type>/rollout/`
- 可視化: 代表時刻と最大誤差時刻のGT / Prediction / Absolute Error
- ParaView: ReごとのPVD/VTI
- W&B: Reごとの別Run

### Interpolation rollout

| Re | 平均 Relative L2 | 最終 Relative L2 | 最大 Relative L2 | 平均MSE | 最終MSE |
|---:|---:|---:|---:|---:|---:|
| 35 | 1.307951 | 2.280701 | 2.280701 | 2.074252e-05 | 5.630560e-05 |
| 85 | 0.523250 | 0.891298 | 0.891298 | 2.045154e-05 | 5.187599e-05 |
| 135 | 0.727409 | 2.152648 | 2.152648 | 1.454680e-04 | 7.790945e-04 |
| 185 | 0.739237 | 1.456089 | 1.456089 | 2.437801e-04 | 6.765813e-04 |
| 235 | 0.824233 | 1.738492 | 1.738492 | 4.938642e-04 | 1.569111e-03 |

[Interpolation rollout metrics](interpolation/rollout/rollout_metrics.csv)

### Extrapolation rollout

| Re | 平均 Relative L2 | 最終 Relative L2 | 最大 Relative L2 | 平均MSE | 最終MSE |
|---:|---:|---:|---:|---:|---:|
| 260 | 0.773275 | 1.361901 | 1.361901 | 4.895670e-04 | 1.183370e-03 |
| 270 | 0.732157 | 1.349008 | 1.349008 | 4.794587e-04 | 1.252728e-03 |
| 280 | 0.769376 | 1.431166 | 1.431166 | 5.625860e-04 | 1.518023e-03 |
| 290 | 0.673864 | 1.110931 | 1.110931 | 4.492169e-04 | 9.819599e-04 |
| 300 | 0.729811 | 1.282363 | 1.282363 | 5.710376e-04 | 1.402361e-03 |

[Extrapolation rollout metrics](extrapolation/rollout/rollout_metrics.csv)

Relative L2が1を超えるケースがあるため、長期rolloutはone-step精度とは分けて解釈する必要があります。場の崩壊と渦放出の位相ずれは、保存済みGT / Prediction / Absolute Error画像およびParaViewで確認します。

## 12. Performanceデータ

### Interpolation

設定: [make_pairs_interpolation.yaml](../../configs/make_pairs_interpolation.yaml)

- Re: 35, 85, 135, 185, 235
- データ: `performance/data/interpolation`
- 結果: `performance/results/ex001_baseline/interpolation`

### Extrapolation

設定: [make_pairs_extrapolation.yaml](../../configs/make_pairs_extrapolation.yaml)

- Re: 260, 270, 280, 290, 300
- データ: `performance/data/extrapolation`
- 結果: `performance/results/ex001_baseline/extrapolation`

Performanceデータはtrain/validationには使用しません。現在のtest loader YAMLはextrapolationを指しています。interpolationへ切り替える場合は、`evaluation.type`、`data.directory`、`test_re`、およびrolloutの`reynolds`を対応させます。

## 13. 設定の正本

| 工程 | Config |
|---|---|
| 学習pair生成 | [configs/make_pairs.yaml](../../../configs/make_pairs.yaml) |
| Dataset・Normalization | [configs/dataset.yaml](../../../configs/dataset.yaml) |
| Normalization統計値 | [configs/normalization_stats.yaml](../../../configs/normalization_stats.yaml) |
| Train/validation split・loader | [configs/dataloader.yaml](../../../configs/dataloader.yaml) |
| モデル | [configs/model.yaml](../../../configs/model.yaml) |
| 学習 | [configs/train.yaml](../../../configs/train.yaml) |
| One-step評価 | [performance/configs/test.yaml](../../configs/test.yaml) |
| Test data・評価種別 | [performance/configs/test_dataloader.yaml](../../configs/test_dataloader.yaml) |
| Rollout | [performance/configs/rollout.yaml](../../configs/rollout.yaml) |

## 14. 次回更新時のチェック

- 使用したGit commitを記録する
- YAML変更後にNotebookの事前チェックを先頭から実行する
- train/validation/performanceのReが混在していないことを確認する
- best checkpointのepochとmetricを更新する
- one-stepの内挿・外挿結果を更新する
- rolloutのRe別平均・最終・最大metricを更新する
- PNG、PT、PVDの保存先を確認する

## 15. 次の実験との分離

本ディレクトリ全体が`ex001_baseline`のperformance評価単位です。次の実験では本ディレクトリを上書きせず、`performance/results/exNNN_name/`を新しい保存先にします。

新しいディレクトリはone-stepまたはrollout実行時に自動作成されるため、事前の手動作成は不要です。実行前に次の3点を同じ実験へ揃えます。

- Notebookの`EXPERIMENT_ID`
- `test.yaml`と`rollout.yaml`のcheckpoint
- 両YAMLの`output.root_directory`

実験終了後に、使用YAMLの全文を`config_snapshot.txt`へ保存し、設定と結果の要約を`experiment.md`へ記録します。
