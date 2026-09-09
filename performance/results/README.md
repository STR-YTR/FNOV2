# Performance評価結果

評価結果は、学習済みモデルに対応する実験ID単位で保存します。

```text
performance/results/
└── exNNN_name/
    ├── experiment.md
    ├── config_snapshot.md
    ├── interpolation/
    │   ├── one_step/
    │   └── rollout/
    └── extrapolation/
        ├── one_step/
        └── rollout/
```

## 1実験に含めるもの

- `experiment.md`: データ、normalization、モデル、学習条件、評価結果、所見
- `config_snapshot.md`: 実行時点のYAML全文をまとめた記録。実行には使用しない
- `one_step/`: metrics CSV/JSONとRe別誤差PNG
- `rollout/`: step別metrics、全時刻PT、代表画像、ParaView用VTI/PVD

大量に生成されるmetrics、配列、画像、fieldファイルはGit管理対象外です。説明文書と設定snapshotだけを必要に応じてGitで管理します。

## 新しい実験

`test.py`または`rollout.py`が`output.root_directory`以下を自動作成するため、実験ディレクトリを事前に手作業で作る必要はありません。

次の実験では、必ず新しい実験IDを指定します。

```yaml
output:
  root_directory: ../results/ex002_example
```

Notebookの`EXPERIMENT_ID`、評価configのcheckpoint、`output.root_directory`が同じ実験を指すことを確認してください。
