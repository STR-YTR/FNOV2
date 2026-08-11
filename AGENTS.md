# AGENTS.md

## プロジェクト概要

本リポジトリは、流体解析と機械学習を組み合わせた研究を、
効率的かつ再現可能に実施するための研究コード基盤である。

現在の主な研究対象は、Fourier Neural Operator（FNO）を用いた
2次元円柱周り非定常流れの予測である。

本プロジェクトはFNO専用コードとして閉じるのではなく、
将来的に以下の要素を設定ファイルから変更・比較できる
拡張可能な研究基盤とすることを目指す。

- データ前処理
- 格子表現
- 機械学習モデル
- 学習条件
- 損失関数
- 評価方法

ただし、汎用化そのものを目的としない。
研究結果を得ることを最優先とし、
現在必要のない過度な抽象化や機能追加は行わない。


## 現在の研究目的

現在は、ある時刻の流れ場から次時刻の流れ場を予測する
one-step predictionを基本問題とする。

現在予定している入力は以下の7チャネルである。

- mask
- Reynolds number（Re）
- x座標
- y座標
- x方向速度 u
- y方向速度 v
- 圧力 p

概念的には、

X(t) = [mask, Re, x, y, u(t), v(t), p(t)]

とする。

出力は、

Y(t) = [u(t+dt), v(t+dt), p(t+dt)]

の3チャネルとする。

長時間予測では、予測した

[u, v, p]

を次ステップの入力として使用する
autoregressive rolloutを行う。


## CFDデータ

教師データはOpenFOAMによる数値流体解析から生成する。

現在想定しているデータ生成フローは、

OpenFOAM
→ foamToVTK
→ VTK/VTM時系列データ
→ 規則格子への変換
→ one-step pair作成
→ PyTorch Dataset
→ モデル学習
→ rollout
→ 評価

である。

現在確認しているOpenFOAM由来のVTKデータには、
少なくとも以下の物理量が含まれている。

- U
- p
- nut

現在のFNOでは、主にUから取得するu, vと、
圧力pを学習対象とする。

OpenFOAMの元メッシュは非構造格子であることを前提とし、
規則格子であると仮定して処理してはならない。


## 格子処理

現在のFNOでは規則格子データを使用する。

OpenFOAMの非構造格子データを、
指定した規則格子へ補間して使用する。

target gridの物理領域（bounds）は、
原則としてVTK/VTMデータから自動取得する。

格子解像度および補間方法はconfigから変更可能にする。

例：

```yaml
grid:
  bounds: auto
  nx: 256
  ny: 64

interpolation:
  method: linear
```

将来的には、

```yaml
grid:
  bounds: [0.0, 1.6, 0.0, 0.41]
```

のような手動指定にも対応できる構造が望ましい。

補間方法についても、
将来的にlinear、nearestなどを比較できる構造とする。

ただし、現段階では補間方法の網羅的比較を優先しない。
まず1つの妥当な方法でFNOの研究を進め、
前処理依存性を検証する必要が生じた場合に比較を行う。


## configによる実験管理

可能な限り、実験条件をPythonコード内に直接記述せず、
configファイルから変更できるようにする。

configで管理する候補は以下である。

### Data

- 使用するRe
- train / validation / testの分割
- 時間範囲
- pairの時間間隔
- 格子解像度
- bounds
- 補間方法
- 使用チャネル
- normalization

### Model

- model name
- width
- depth
- Fourier modes
- input channels
- output channels

### Training

- batch size
- learning rate
- epochs
- optimizer
- scheduler
- loss function
- random seed

### Inference / Evaluation

- rollout steps
- evaluation metrics
- 保存形式
- 可視化条件


## ソフトウェア設計方針

以下の処理は可能な限り論理的に分離する。

- データ読み込み
- 前処理
- Dataset構築
- モデル定義
- 学習
- 推論
- rollout
- 評価
- 可視化
- config管理

将来的には、各要素を独立して変更できる構造を目指す。

例えば、

regular grid
+
FNO

だけでなく、将来的に

unstructured grid
+
GNO / GINOなど

を扱う可能性がある。

ただし、規則格子と非構造格子を無理に同一のtensor shapeへ
変換して共通化してはならない。

データ形式ごとにreader、adapter、preprocessorなどを
差し替えられる構造とし、
trainingやexperiment managementなど、
共通化可能な部分のみを再利用する。


## 現在の開発優先順位

現在の最優先事項は、
FNOによる研究結果を正しく得ることである。

基本的な開発順序は以下とする。

1. FNOのbaseline pipelineを正しく動作させる
2. 重要な実験条件をconfig化する
3. 結果の妥当性を検証する
4. 必要に応じて共通部分をrefactoringする
5. 必要になった段階で別モデルや別データ形式へ拡張する

最初から万能な機械学習フレームワークを作ろうとしない。

研究の進行に必要になった機能から順番に実装する。


## make_pairs.py の現在の役割

現在のデータ前処理では、
OpenFOAMから生成されたVTK/VTM時系列データを
FNO用のone-step pairへ変換する。

予定している処理は以下である。

1. VTM seriesから時系列情報を取得する
2. 各時刻のVTK/VTMデータを読み込む
3. Uおよびpを取得する
4. 物理領域のboundsを取得する
5. target regular gridを生成する
6. CFD場を規則格子へ補間する
7. maskを生成する
8. x, y座標場を生成する
9. Re情報を追加する
10. 連続する時刻からone-step pairを生成する
11. tensorおよびmetadataを保存する


## pairデータ形式

現在予定している保存形式は以下である。

```python
{
    "x": Tensor,      # [N, 7, H, W]
    "y": Tensor,      # [N, 3, H, W]
    "times": Tensor,
    "Re": float,

    "meta": {
        "dt_pair": ...,
        "nx": ...,
        "ny": ...,
        "bounds": ...,

        "input_channels": [
            "mask",
            "Re",
            "x",
            "y",
            "u",
            "v",
            "p"
        ],

        "output_channels": [
            "u",
            "v",
            "p"
        ],

        "interpolation": ...
    }
}
```

時間間隔、bounds、格子解像度などを、
理由なくコード内へ固定値として埋め込まない。

データから取得できるものはデータから取得し、
実験条件として指定すべきものはconfigから取得する。


## 再現性

可能な限り、各実験を以下の情報から再現できるようにする。

- source code
- config
- dataset metadata
- random seed
- checkpoint
- evaluation result

学習時には、
使用したconfigを結果フォルダへ保存することが望ましい。

将来的にはGit commit hashなども保存できる構造を検討する。


## Git管理

原則としてGitで管理するもの：

- Pythonコード
- config
- README
- AGENTS.md
- docs
- 小さなmetadata
- 実験条件

原則としてGitで管理しないもの：

- OpenFOAMの大量の計算結果
- VTK / VTM
- pair dataset
- checkpoint
- rolloutデータ
- 大量の画像
- log
- その他の大容量生成物

これらは.gitignoreで除外する。


## コーディング方針

研究コードとして読みやすさを優先する。

- 過度に複雑な実装を避ける
- 関数名・変数名は役割が分かる名前にする
- 必要に応じてtype hintを使用する
- tensor shapeが分かりにくい箇所にはコメントを書く
- 物理量と単位を可能な限り明示する
- 前処理を複数箇所に重複実装しない
- normalizationを暗黙的に行わない
- dataset生成条件をmetadataへ残す
- 関係のないコードを理由なく変更しない
- channel構成を変更した場合はconfigとdocumentationも更新する


## AI / Codexへの指示

コードを変更する前に、
既存コードとconfigの構造を確認すること。

大きな設計変更を独断で行わないこと。

特に以下を変更する場合は、
研究結果そのものに影響する可能性があるため注意する。

- データ前処理
- 補間方法
- normalization
- mask
- grid resolution
- input / output channels
- train / validation / test split
- loss
- evaluation metric

研究上の意味が変わる可能性がある変更については、
単なるrefactoringとして扱わないこと。

既存の動作を変更する場合は、
何を変更したか分かるようにすること。


## 重要な研究上の原則

データ前処理も実験条件の一部である。

例えば、

- interpolation method
- grid resolution
- normalization
- masking
- channel selection
- dataset split

の変更は、
単なる実装上の変更ではなく、
研究結果へ影響する可能性がある。

したがって、これらの条件は
明示的かつ再現可能な形で管理する。