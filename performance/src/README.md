# Performance評価コード

- `test_dataloader.py`は、学習時のDataset設定とnormalization条件を使用してperformance用pairを安全に読み込みます。
- `test.py`は、学習済みcheckpointを読み込み、one-step推論と評価指標の出力を行います。
- `rollout.py`は同じtest loaderを再利用し、全test Reのautoregressive評価をone-step評価から分離して実行します。

このディレクトリのコードは`performance/configs`の設定を使用し、pair内部のReynolds数metadataを検証しなければなりません。プロジェクト直下の学習用`data/`を探索してはいけません。
