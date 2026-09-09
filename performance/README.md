# Performance評価

このディレクトリは、学習およびvalidationには使用しない最終性能評価用データのための領域です。case番号は各データルート内だけで有効なローカル番号です。例えば、`data/interpolation/VTK/case0`と学習用の`data/VTK/case0`は別のケースとして扱います。評価時の物理的なケース識別には、生成した各pairファイル内のReynolds数metadataを使用します。

## ディレクトリ境界

```text
performance/
├── configs/
├── data/
│   ├── interpolation/{VTK,processed/pairs}
│   └── extrapolation/{VTK,processed/pairs}
├── src/
└── results/
    ├── interpolation/{one_step,rollout}
    └── extrapolation/{one_step,rollout}
```

学習用の`configs/dataset.yaml`を`performance/data`へ向けてはいけません。Performance評価コードは専用configの`data.directory`以下だけを探索します。リポジトリ全体を再帰検索してpairファイルを収集してはいけません。

## Pairファイルの生成

既存の`src/make_pairs.py`を再利用し、前処理を複製しません。VTMデータを対応する`VTK/caseN`へ配置し、専用configの`reynolds_by_case`が実データと一致することを確認してから実行します。

> **重要:** Performance用configは必ず`--config`で指定してください。
> `python src/make_pairs.py`だけを実行すると、デフォルトのtrain/validation用
> `configs/make_pairs.yaml`が使用され、`data/processed/pairs`へ出力されます。
> Performance用pairは自動的には生成されません。

```bash
python src/make_pairs.py --config performance/configs/make_pairs_interpolation.yaml
python src/make_pairs.py --config performance/configs/make_pairs_extrapolation.yaml
```

生成された各`.pt`には`meta.case`と`Re`が保存されます。生成後は対象ディレクトリを検証し、`manifest.json`のcase、Re、pair数、shapeが意図どおりか確認します。

```bash
python src/validate_pairs.py performance/data/interpolation/processed/pairs
python src/validate_pairs.py performance/data/extrapolation/processed/pairs
```

## One-step評価

`test_dataloader.py`は、configで指定されたperformanceディレクトリだけを探索し、各pairファイル内部のReynolds数metadataと`test_re`を照合します。チャネル構成とnormalizationには、学習時のDataset設定およびtrainデータだけから算出した統計量を使用します。

`test.py`は確定済みcheckpointを読み込み、`model.eval()`とinference modeでone-step推論を実行し、`results/<evaluation.type>/one_step/`へ評価結果を保存します。

```bash
python performance/src/test_dataloader.py \
  --config performance/configs/test_dataloader.yaml

python performance/src/test.py \
  --config performance/configs/test.yaml
```

JSONとCSVには、全体およびRe別のMSEとRelative L2を保存します。`u`、`v`、`p`別、標準化空間と物理空間、全格子とmask有効領域をそれぞれ記録します。予測Tensor自体は保存しません。

内挿評価では`interpolation_relative_l2.png`を生成します。横軸はReynolds数、縦軸はmask有効領域における物理スケールのRelative L2誤差（%）です。

外挿評価では`extrapolation_relative_l2.png`を生成します。横軸は学習範囲上限からの距離`Re - 250`、縦軸は同じRelative L2誤差（%）です。どちらもOverall、`u`、`v`、`p`を別々の系列として表示します。

内挿と外挿は、`test_dataloader.yaml`の`evaluation.type`、`data.directory`、`test_re`で切り替えます。共通のloader設定は`configs/dataloader.yaml`から取得し、test時のshuffleは常に無効です。


## Autoregressive rollout評価

`rollout.py`は`t=340.0 s`のGTを初期場として使用し、teacher forcingを行わず、予測した`u/v/p`だけを次入力へ戻して`t=350.0 s`まで100 step進めます。mask、座標、Reは固定し、学習時のnormalizationをそのまま使用します。

```bash
python performance/src/rollout.py --config performance/configs/rollout.yaml
```

正式結果は`results/<evaluation.type>/rollout/`へ保存します。各Reの`rollout_ReXX.pt`には100 stepすべてのprediction、ground truth、time、mask、metrics、metadataを保存します。画像は等間隔5点程度と最大Relative L2時刻だけを`plots/ReXX/`へ保存します。

ParaView用にはReごとに`vti/ReXX/prediction_ReXX.pvd`と100 step分のVTIを出力します。ParaViewではPVDファイルを開くと、各`u/v/p`のPrediction、Ground Truth、Absolute Errorを同じtime stepで切り替えて比較できます。また、予測だけをまとめた`rollout_prediction_ReXX.pt`もReごとに保存します。

W&BはReごとに別Runを作成し、各stepの`time`、`elapsed_time`、MSE、Relative L2をoverallおよび`u/v/p`別に記録します。まず`rollout.yaml`の`reynolds: [35]`でsanity checkし、成功後に内挿全Re、最後にtest loaderを外挿へ切り替えて外挿全Reへ展開します。W&Bを使用しない場合は`wandb.enabled: false`にします。


## Test dataset作成・評価時のチェックリスト

入出力ディレクトリを変更するだけでは不十分です。test datasetの生成または評価前に、以下を確認してください。

- 生のVTMファイルを`interpolation/VTK`または`extrapolation/VTK`へ配置し、学習用`data/VTK`と混在させない。
- 対応するmake-pairs configの`reynolds_by_case`を実データに合わせる。case番号はディレクトリ内だけのラベルであり、物理的な識別には設定されたReynolds数を使う。
- `test_dataloader.yaml`の`test_re`を`reynolds_by_case`と一致させる。
- 前処理条件の違い自体を評価する場合を除き、`step_to_time`、時間範囲、`dt_pair`、grid bounds・解像度、field名、補間方法を学習データ生成時と揃える。
- 学習時のDataset設定とtrain統計量を再利用し、内挿・外挿testデータ自身からmean/stdを算出しない。外挿Reのscale後の値が1を超えても、学習時のRe scaleを変更しない。
- pair生成後に`src/validate_pairs.py`を実行し、`manifest.json`のcase、Re、pair数、Tensor shapeを確認する。
- 最終test結果をcheckpoint選択やハイパーパラメータ調整に使用しない。使用した場合、そのデータはvalidation扱いとなり、別の未使用test setが必要になる。
