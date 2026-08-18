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
## Datasetについて

Datasetは、生成済みのpair `.pt` ファイルを読み込み、
FNOの学習に使用する1サンプル単位のTensorを返す役割を持つ。

Dataset自体は新しいデータファイルを生成・保存せず、
学習時に`.pt`から必要なデータを読み出してTensorとして扱う。

現在の基本入力は以下の7チャネルとする。

```text
[mask, Re, x, y, u, v, p]
```

教師データは以下の3チャネルとする。

```text
[u(t+dt), v(t+dt), p(t+dt)]
```

したがって、1サンプルの基本shapeは、

```text
input  : [7, H, W]
target : [3, H, W]
```

とする。

Dataset実装の最初の目標は、

```python
x, y = dataset[i]
```

によって、正しいチャネル・shape・値を持つTensorを取得できることである。

Reごとのtrain / validation / test分割、batch化、shuffleなどは
Datasetに直接埋め込まず、後段の設定およびDataLoaderで管理する。

また、将来的なablation studyに対応できるよう、
入力チャネルを固定的にハードコードしすぎない設計とする。

例えば、

```text
Baseline:
[mask, Re, x, y, u, v, p]

No-Re:
[mask, x, y, u, v, p]
```

のような比較を、同じ`.pt`データからチャネルを選択することで実行可能にする。

Reの有無などの実験条件を変更するためだけに、
`.pt`データを再生成・複製する設計は避ける。

## DataLoaderについて

DataLoaderは、Datasetから取得したサンプルを
train / validationに分割し、FNOの学習に使用するbatchを生成する役割を持つ。

Datasetでは1サンプル単位のTensorを定義し、
DataLoaderでは主に以下を担当する。

- Reynolds number単位でのtrain / validation分割
- batch化
- shuffle
- num_workers
- pin_memory
- drop_last

testデータについては現段階ではDataLoaderに含めず、
学習およびハイパーパラメータ決定後に独立したtest処理で評価する。


### Reによるtrain / validation分割

今回のbaselineでは、Re=10〜250の25ケースを使用する。

validationはReケース単位で分離し、
同一Reのサンプルをtrainとvalidationの両方に含めない。

基本設定：

```text
Train:
10, 20, 30, 40,
60, 70, 80, 90,
110, 120, 130, 140,
160, 170, 180, 190,
210, 220, 230, 240, 250

Validation:
50, 100, 150, 200
```

ケース数：

```text
Train      : 21 cases
Validation : 4 cases
```

各Reケースが100 pairsの場合：

```text
Train      : 2100 samples
Validation : 400 samples
```

おおよそ、

```text
Train : Validation = 84 : 16
```

となる。

Re分割はPythonコードへハードコードせず、
`configs/dataloader.yaml`を正本として管理する。


### dataloader.yaml

初期設定の例：

```yaml
split:
  train_re:
    - 10
    - 20
    - 30
    - 40
    - 60
    - 70
    - 80
    - 90
    - 110
    - 120
    - 130
    - 140
    - 160
    - 170
    - 180
    - 190
    - 210
    - 220
    - 230
    - 240
    - 250

  val_re:
    - 50
    - 100
    - 150
    - 200

loader:
  batch_size: 16
  train_shuffle: true
  val_shuffle: false
  num_workers: 0
  pin_memory: true
  drop_last: false
```

batch size等は今後のGPUメモリ使用量や学習速度を確認して変更可能とする。

### 学習対象ReおよびTrain / Validation分割

初期のbaseline実験では、学習・validationに使用するReynolds numberの範囲を以下とする。

```text
Re = 10〜250
```

10刻みで生成した25ケースを対象とする。

train / validationの分割はサンプル単位のランダム分割ではなく、
**Reケース単位で分割する**。

初期分割は以下とする。

```text
Train:
Re = 10, 20, 30, 40,
     60, 70, 80, 90,
     110, 120, 130, 140,
     160, 170, 180, 190,
     210, 220, 230, 240, 250

Validation:
Re = 50, 100, 150, 200
```

ケース数ベースでは、

```text
Train      : 21 cases（84%）
Validation :  4 cases（16%）
```

となる。

各Reケースが100 pairsの場合、サンプル数は、

```text
Train      : 2100 samples
Validation :  400 samples
```

となる。

このRe分割は`dataloader.yaml`を正本として管理し、
`dataloader.py`内へ具体的なRe値をハードコードしない。

また、同一Reのデータがtrainとvalidationの両方へ含まれることを禁止する。
実装時には、

```text
train_re ∩ val_re = ∅
```

であることを必ず検証する。

Re=260〜300の既存データは、現時点ではtrain / validationには使用しない。
最終的なtest条件については、モデルおよび学習条件をtrain / validationで確定した後に別途定義する。

最終testでは、学習に使用していないReを用いた
内挿・外挿性能の評価を予定している。

### Datasetとの連携

DataLoader用にデータを再生成しない。

既存の`dataset.py`を利用し、
Datasetが読み込んだサンプルについてReを識別して
train / validationへ振り分ける。

Dataset側では、各サンプルがどのReケースに属するかを
DataLoaderから安全に取得できる構造にする。

ただし、Datasetの入力チャネルとしてReを使用するかどうかと、
サンプルをtrain / validationへ分割するためのRe metadataは区別すること。

例えば、

```yaml
input_channels:
  - mask
  - x
  - y
  - u
  - v
  - p
```

としてFNO入力からReチャネルを除外した場合でも、
DataLoaderはmetadata等から各サンプルのReを識別できなければならない。

つまり、

```text
ReをFNOへ入力するか
```

と、

```text
どのReケースのデータか
```

は別の情報として扱う。


### Batch shape

Baselineの7入力チャネルでは、
Datasetの1サンプルは、

```text
x : [7, H, W]
y : [3, H, W]
```

である。

batch_size=16の場合、DataLoader出力は、

```text
x : [16, 7, H, W]
y : [16, 3, H, W]
```

となることを確認する。

入力チャネルを変更した場合は、
`dataset.yaml`の設定に自動追従すること。

例えばReチャネルを除外した場合：

```text
x : [16, 6, H, W]
y : [16, 3, H, W]
```

となる。

チャネル数をDataLoader側へハードコードしない。


### Shuffle

基本設定：

```text
train      : shuffle = true
validation : shuffle = false
```

とする。

validationでは再現性および評価結果の追跡を容易にするため、
原則としてshuffleしない。


### DataLoader実装後のsanity check

実装後は、学習コードへ進む前に最低限以下を自動または明示的に確認すること。

1. 読み込まれた`.pt`ファイル数
2. Dataset全体のサンプル数
3. trainに含まれるRe一覧
4. validationに含まれるRe一覧
5. train / validation間にReの重複がないこと
6. 指定したReが存在せず、splitから欠落していないこと
7. 意図していないReがtrain / validationに混入していないこと
8. train sample数
9. validation sample数
10. train batchのx/y shape
11. validation batchのx/y shape
12. input channel設定変更時にbatch shapeが自動追従すること
13. NaN / Infが含まれていないこと


baselineでは少なくとも、

```text
Train samples      = 2100
Validation samples = 400
```

であることを確認する。


### データリーク防止

今回の研究ではReynolds numberに対する汎化性能を評価するため、
train / validation間のReの重複を禁止する。

以下のような状態を検出した場合は、
警告だけで続行せずエラーとして停止すること。

```text
train_re ∩ val_re != ∅
```

また、splitとして指定したReがDatasetに存在しない場合も、
設定ミスの可能性があるため原則としてエラーとする。


### Testデータについて

最終testは学習およびハイパーパラメータ調整には使用しない。

モデルおよび学習条件をtrain / validationによって確定した後、
独立したtestデータを用いて評価する。

予定している評価例：

```text
Interpolation:
Re = 85, 135, 185, 235 など

Extrapolation:
Re = 260, 275, 300 など
```

test結果を参照してモデル構造やハイパーパラメータを調整した場合、
そのtestデータは純粋な最終評価用データとして扱わない。


### 実装上の原則

DataLoader実装時は、既存の`dataset.py`および
`dataset.yaml`の設計を確認してから実装すること。

特に以下を避ける。

- input channel数の再定義
- DatasetとDataLoaderでのRe情報の重複管理
- train / validation ReのPythonコードへのハードコード
- Datasetと異なるchannel順の独自定義
- train / validation間のデータリーク
- testデータを学習中のモデル選択に使用すること

既存の設定から取得・導出可能な値は再定義せず、
Single Source of Truthの原則に従うこと。

## YAML設定管理の注意事項

本プロジェクトでは、Dataset、DataLoader、Model、Trainingなどの
実験条件をYAMLで管理する。

設定値の重複による変更漏れや不整合を防ぐため、
同じ意味を持つ設定値を複数のYAMLやPythonコードに重複して定義しない。

各設定値には原則として1つの正本
（Single Source of Truth）を持たせる。

上流工程ですでに決定されている情報を下流工程で使用する場合は、
同じ値を再定義せず、上流の設定から参照または自動導出する。

例えば、Dataset設定で、

```yaml
input_channels:
  - mask
  - Re
  - x
  - y
  - u
  - v
  - p

output_channels:
  - u
  - v
  - p
```

と定義されている場合、Model側で、

```yaml
in_channels: 7
out_channels: 3
```

のように同じ情報を手動で再定義することは避ける。

代わりにPython側で、

```python
in_channels = len(input_channels)
out_channels = len(output_channels)
```

のように自動導出する。

これにより、例えばDataset設定からReを除外した場合、

```text
7 channels
    ↓
6 channels
```

という変更が、

```text
Dataset
  ↓
Tensor shape
  ↓
Modelのin_channels
  ↓
Training
  ↓
Inference / Evaluation
```

まで自動的に反映される設計を目指す。

基本原則は、

```text
人間が決める実験条件
        ↓
      YAML
        ↓
そこから一意に決まる値
        ↓
Python側で自動導出
```

とする。

YAMLファイルが複数に分かれている場合も、
同じ設定値をそれぞれに記述するのではなく、
その値を管理する上流側のYAMLを正本とする。

YAML同士を複雑に直接参照させるのではなく、
実行するPythonコード側で必要なYAMLを読み込み、
設定値を統合してDataset、DataLoader、Model、Trainingなどへ渡す。

新しい設定値を追加する際は、以下を確認すること。

1. この設定値の正本はどこか
2. 既存の設定値から自動導出できないか
3. 同じ意味の値を別のYAMLやPythonコードに重複定義していないか
4. 上流設定を変更したときに下流処理が自動的に追従するか
5. チャネル数、Re条件、解像度などをハードコードしていないか

実験条件を1つ変更するために複数のPythonファイルやYAMLを
手作業で同時修正する必要がある設計は、可能な限り避けること。
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