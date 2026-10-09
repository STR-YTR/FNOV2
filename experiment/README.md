# 実験管理

Notebookは設定確認・事前チェック・実行・可視化に使用し、実験ごとの正式な評価結果と記録は `performance/results/<EXPERIMENT_ID>/` に保存します。

## Notebook

- [学習](training/training.ipynb)
- [One-step評価](evaluation/one_step.ipynb): セクション7でReとサンプル番号を指定し、u/v/pの正解・予測・絶対誤差をmatplotlibで比較。全件評価は不要で、選択サンプルのみtest.yamlのdevice設定で推論。PNG保存は任意。
- [Rollout評価](evaluation/rollout.ipynb)

各Notebookの先頭にある `EXPERIMENT_ID` は、実行対象と保存先の確認に使用します。実行用設定の正本はルートのYAMLです。

## 実験一覧

| 実験ID | 概要 | 学習出力 | 評価結果・実験記録 |
|---|---|---|---|
| `ex001_baseline` | Baseline FNO | `outputs/training/baseline` | [performance/results/ex001_baseline](../performance/results/ex001_baseline/experiment.md) |

## 1実験の保存単位

```text
performance/results/exNNN_name/
├── experiment.md
├── config_snapshot.md
├── interpolation/
│   ├── one_step/
│   └── rollout/
└── extrapolation/
    ├── one_step/
    └── rollout/
```

- `experiment.md`: 学習設定、評価条件、結果、所見
- `config_snapshot.md`: 実験時点のYAML全文。記録専用で実行には使用しない
- `interpolation/`、`extrapolation/`: metrics、PNG、PT、VTI、PVDなどの正式結果

学習checkpointと履歴は容量が大きいため、`outputs/training/<実験名>/`に保存し、experiment.mdから参照します。

## 次の実験を始める手順

例: `ex002_lr1e4`

1. 実験IDを決める。この例では`ex002_lr1e4`とする。結果ディレクトリの事前作成は不要。
2. ルートの実行用YAMLで学習条件を変更する。
3. `configs/train.yaml`の出力先を `outputs/training/ex002_lr1e4` に変更する。
4. 学習Notebookの `EXPERIMENT_ID` を `ex002_lr1e4` に変更し、設定と事前チェックを確認する。
5. smoke test後に本学習を実行する。
6. `performance/configs/test.yaml`と`rollout.yaml`のcheckpointを `outputs/training/ex002_lr1e4/best.pt` に変更する。
7. 両configの`output.root_directory`を `../results/ex002_lr1e4` に変更する。
8. 評価Notebookの `EXPERIMENT_ID`を `ex002_lr1e4` に変更してone-step、rolloutを実行する。
9. 評価コードが自動作成した`performance/results/ex002_lr1e4/`へexperiment.mdとconfig_snapshot.mdを保存する。
10. この実験一覧へ結果を追加する。

## 運用ルール

- 実験IDは `exNNN_説明` とする。
- YAMLは現在実行する設定だけを保持する。
- 過去の設定は各実験の`config_snapshot.md`へ保存する。
- 学習出力とperformance結果は実験ごとに別ディレクトリへ保存する。
- 過去実験のcheckpoint、metrics、PNG、PT、VTIを上書きしない。
- Notebookの`EXPERIMENT_ID`、YAMLの出力先、checkpointが同じ実験を指すことを事前チェックで確認する。
