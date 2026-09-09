# FNOV2: FNO for Unsteady Cylinder Wake Prediction

Fourier Neural Operatorを用いて、円柱周りの二次元非定常流れを予測する研究プロジェクトです。

## Overview

OpenFOAMで生成した円柱周り流れ場を教師データとして用い、現在時刻の流れ場から次時刻の流れ場を予測します。

本プロジェクトでは、以下を主な評価対象とします。

- 異なるレイノルズ数に対する内挿・外挿性能
- 長時間ロールアウトの安定性
- 異なる格子解像度へのzero-shot推論
- 座標情報および圧力情報の追加効果
- Fourier mode数が予測性能に与える影響

## Problem Setting

### Input

入力は7チャネルで構成します。

1. `mask`
2. `Re`
3. `x`
4. `y`
5. `u(t)`
6. `v(t)`
7. `p(t)`

### Output

出力は3チャネルです。

1. `u(t + dt)`
2. `v(t + dt)`
3. `p(t + dt)`

ロールアウト時は、`mask`, `Re`, `x`, `y`を固定し、`u`, `v`, `p`のみを更新します。

## Directory Structure

```text
configs/                学習・Dataset・Modelの実行用YAML
src/                    pair生成、Dataset、Model、学習処理
data/                   train / validation用のVTKとpair
outputs/training/       学習済みcheckpointと学習履歴
performance/configs/    現在実行するperformance評価用YAML
performance/data/       学習に使用しない内挿・外挿評価データ
performance/src/        one-step・rollout評価処理
performance/results/    実験ID別の評価結果・実験記録
experiment/             設定確認・実行・可視化Notebook
docs/                   研究・設計資料