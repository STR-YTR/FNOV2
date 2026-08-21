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
## DatasetにおけるNormalization / Standardization

Datasetは、生成済みのpair `.pt` に保存された物理量をそのまま返すだけでなく、
必要に応じて学習用のスケーリングおよび標準化を適用する役割を持つ。

元の`.pt`データは物理量の生値を保持し、
正規化・標準化のためだけに別の`.pt`データを再生成しない。

基本的な処理フローは以下とする。

```text
pair .pt（生の物理量）
        ↓
Dataset
        ↓
channel selection
        ↓
normalization / standardization
        ↓
Tensor
        ↓
DataLoader
        ↓
Model
```

### Baselineの変換方針

現在のbaselineでは、以下の変換を採用候補とする。

```text
mask
→ 変換しない
→ 0 / 1 のまま使用

Re
→ 学習範囲上限によるscaling
→ Re_scaled = Re / 250

x
→ 0〜1へscaling
→ x_scaled = x / 1.6

y
→ 0〜1へscaling
→ y_scaled = y / 0.41

u
→ trainデータのmean/stdによるstandardization

v
→ trainデータのmean/stdによるstandardization

p
→ trainデータのmean/stdによるstandardization
```

u, v, pの標準化は、

```text
z = (x - mean_train) / std_train
```

を用いる。

meanおよびstdは必ずtrainデータから算出し、
validationおよびtestデータの統計量を使用してはならない。


### 統計量の算出範囲

u, v, pのmean/stdは、
train splitに含まれるReケースのみから算出する。

validationおよびtestのReは統計量計算に含めない。

また、u, v, pについては、
物理的に有効な流体領域のみを統計量算出対象とする。

現在のpair生成では、

```text
mask == 1
```

を有効領域とし、

```text
mask == 0
```

ではu, v, pが0として保存されている。

この人工的な0値がmean/stdへ影響しないよう、
u, v, pの統計量は原則として`mask == 1`の要素のみから計算する。


### Input / Outputの統計量

one-step predictionでは、

```text
input:
u(t), v(t), p(t)

output:
u(t+dt), v(t+dt), p(t+dt)
```

を扱う。

現在のtrainデータではinputとoutputの統計量がほぼ一致しているため、
baselineでは同一のtrain統計量をinputおよびtargetのu, v, pへ使用する設計を基本候補とする。

これにより、autoregressive rolloutにおいても、

```text
予測されたstandardized u/v/p
        ↓
次stepの入力u/v/p
```

を一貫したスケールで扱えるようにする。

ただし、inputとoutputで異なる統計量を使用する実装へ変更する場合は、
rollout時の変換方法も含めて明示的に設計すること。


### Inverse Transform

評価・可視化・VTK出力などで物理量が必要な場合は、
標準化された予測値を元の物理スケールへ戻せるようにする。

u, v, pについては、

```text
x = z * std_train + mean_train
```

によってinverse transformできる構造とする。

Datasetまたは共通のnormalization utilityに、
forward transformとinverse transformの両方を定義することが望ましい。

学習時・推論時・rollout時・評価時で、
異なる変換式を重複実装しないこと。


### YAMLによる管理

Normalization / standardizationの条件はPythonコードへ直接ハードコードせず、
`dataset.yaml`を正本として管理する。

例：

```yaml
normalization:

  mask:
    type: none

  Re:
    type: scale
    scale: 250.0

  x:
    type: scale
    scale: 1.6

  y:
    type: scale
    scale: 0.41

  fields:
    type: standardize
    statistics_source: train
    valid_region_only: true
```

具体的なmean/stdの値を手作業で複数箇所へ記述しない。

統計量を計算・保存する場合は、
どのtrain splitから計算された値か分かるmetadataを保持する。


### Single Source of Truth

Normalizationに関する設定値も、
既存のSingle Source of Truthの原則に従う。

例えば、

```text
Re scale = 250
```

をDataset、Model、Train、Inferenceで個別に定義してはならない。

`dataset.yaml`または共通のnormalization設定を正本とし、
学習・推論・評価は同じ設定を参照すること。


### Dataset実装後のSanity Check

Normalization対応後は、学習へ進む前に最低限以下を確認する。

1. maskが0/1のまま保持されていること
2. Reが想定したscaleで変換されていること
3. xおよびyが想定した範囲に変換されていること
4. u, v, pの標準化後meanが概ね0になること
5. u, v, pの標準化後stdが概ね1になること
6. validationデータにtrain統計量が使用されていること
7. validation自身の統計量を使用していないこと
8. mask == 0の領域が統計量計算へ混入していないこと
9. NaN / Infが生成されていないこと
10. inverse transformによって元の値を十分な精度で復元できること
11. Datasetのinput channel選択を変更してもNormalization処理が破綻しないこと

特に、

```text
original
→ normalize / standardize
→ inverse transform
```

を行った結果が元データと一致することを確認する。


### Codexへの注意

Normalization処理を追加する際、
学習コードやModel側へ同じ変換処理を重複実装しないこと。

train / validation / testで異なる統計量を独立に計算してはならない。

統計量の算出にはtrainデータのみを使用し、
validationおよびtest情報のリークを防止すること。

Normalizationの変更は研究条件の変更であるため、
単なるrefactoringとして扱わず、YAMLおよび実験記録へ明示すること。

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

## Model

FNOモデルには、公式 `neuralop` ライブラリの `FNO` 実装を使用する。

FNO内部のSpectral Convolution、Fourier Transform、FNO Block等を独自に再実装せず、
公式実装を基準とする。

`src/model.py` は主に以下を担当する。

```text
model.yaml
    ↓
モデル設定の読み込み

dataset.yaml
    ↓
input / output channel情報の取得

    ↓

neuralop.models.FNO
    ↓
FNOモデル生成
```

### Input / Output channels

`in_channels` および `out_channels` は、
`model.yaml`へ重複して記述しない。

Datasetで定義された、

```text
input_channels
output_channels
```

から自動的に決定する。

例えば、

```text
input_channels
= [mask, Re, x, y, u, v, p]

→ in_channels = 7

output_channels
= [u, v, p]

→ out_channels = 3
```

となる。

入力チャネル構成を変更した場合、
Model側も自動的に追従する構造とする。


### Model configuration

FNO固有のハイパーパラメータは、
原則として `configs/model.yaml` を正本として管理する。

対象には少なくとも以下を含む。

```text
n_modes
hidden_channels
lifting_channels
projection_channels
n_layers
activation
domain_padding
positional_embedding
```

実際に使用する項目は、使用中の `neuralop` バージョンの
公式FNO APIに存在する引数を確認した上で定義すること。

Pythonコード内へ実験条件となる値を不必要にハードコードしない。


### Explicit configuration

`neuralop.models.FNO` のデフォルト設定を無条件に使用しない。

今回のモデル構成・Dataset構成に影響する設定については、
公式実装のデフォルト値と挙動を確認し、
採用する値を明示的に設定する。

特に以下は確認対象とする。

```text
positional_embedding
n_modes
hidden_channels
lifting_channels
projection_channels
n_layers
domain_padding
activation
```

公式ライブラリのバージョン変更によって
意図せずモデル構造が変化しないよう注意する。


### Positional embedding

現在のDatasetでは入力チャネルとして、

```text
x
y
```

の座標情報を明示的に与えている。

したがって、
FNO内部でgrid positional embeddingを追加すると
位置情報が重複する可能性がある。

Dataset側でx,yを入力する構成では、
FNO側の自動positional embeddingを使用しない構成を基本とする。

将来Dataset側からx,yを除去して
FNO側のpositional embeddingを使用する場合は、
DatasetとModelの両方の設定を整合させること。


### Forward

FNOのforward処理そのものは
`neuralop.models.FNO` の公式実装を使用する。

FNOV2側でSpectralConvやFNOBlocksのforward処理を
重複実装しない。

基本的な入出力は、

```text
input
[B, C_in, H, W]

    ↓
neuralop FNO

output
[B, C_out, H, W]
```

とする。


### Model sanity check

Model実装後、学習へ進む前に最低限以下を確認する。

1. Dataset / DataLoaderから得られるTensorをそのまま入力できること
2. input shapeが `[B, C_in, H, W]` であること
3. output shapeが `[B, C_out, H, W]` であること
4. `C_in` がDatasetのinput_channels数と一致すること
5. `C_out` がDatasetのoutput_channels数と一致すること
6. forward passが正常に完了すること
7. outputにNaN / Infが存在しないこと
8. backward passを1回実行できること
9. Datasetのinput_channelsを変更した場合にModelが追従すること
10. 使用しているFNO設定値を実行時に確認できること


### Reproducibility

使用している `neuralop` のバージョンを記録すること。

モデルの再現に必要な設定は、
コード内の暗黙的なデフォルト値だけに依存させず、
可能な範囲でYAMLおよび実験記録から確認できる状態にする。


### Single Source of Truth

既に他の設定ファイルで定義されている値を
`model.yaml`へ重複して定義しない。

特に、

```text
input_channels
output_channels
normalization条件
空間解像度
train / validation split
```

などはModel固有の設定ではないため、
Model側で独立して管理しない。

Modelは必要な情報を上流の設定から参照し、
FNO固有の設定のみを管理する。

### Model summary / visualization

`model.py` を単体実行した際に、
生成されたモデルの基本情報を確認できるようにする。

最低限、以下を表示する。

```text
model name
implementation
neuralop version
in_channels
out_channels
n_modes
hidden_channels
lifting_channels
projection_channels
n_layers
positional_embedding
domain_padding
total parameters
trainable parameters
```

表示例：

```text
=== Model Summary ===
model                  : FNO
implementation         : neuralop
neuralop version       : ...
input channels         : 7
output channels        : 3
n_modes                : [12, 12]
hidden_channels        : 32
lifting_channels       : ...
projection_channels    : ...
n_layers               : 4
positional_embedding   : disabled
domain_padding         : ...
total parameters       : ...
trainable parameters   : ...
```

モデル名および各設定値は実際に生成されたモデルおよびYAML設定と整合していること。
表示専用の値を別途ハードコードしない。


### Model architecture visualization

モデル構造を確認できる機能を用意する。

少なくとも、PyTorchのモデル構造をテキスト形式で確認できるようにする。

```text
FNO
├── lifting
├── FNO blocks
│   ├── block 0
│   ├── block 1
│   ├── ...
│   └── block N
└── projection
```

可能であれば、入力Tensorを用いたモデルsummaryも表示し、

```text
Input:
[B, C_in, H, W]

Output:
[B, C_out, H, W]
```

および各主要層の入出力shapeを確認できるようにする。

また、モデル構造を画像またはグラフとして保存できる可視化機能を持たせることが望ましい。

可視化機能は学習処理そのものに必須とせず、
デバッグ・モデル確認用の機能として分離する。

可視化のON/OFFおよび保存先はYAMLから変更可能な構造とする。

例：

```yaml
debug:
  print_summary: true

visualization:
  enabled: true
  output_dir: outputs/model
```

可視化ライブラリを追加する場合は、
そのライブラリが利用できないことによって
通常のmodel生成や学習が失敗しない構造とする。


### Model sanity check output

`model.py` の単体実行時には、

```text
model生成
    ↓
model summary
    ↓
architecture確認
    ↓
dummyまたはDataLoader由来のinputでforward
    ↓
input/output shape確認
    ↓
finite check
    ↓
backward check
```

の順に確認できるようにする。

最終的に少なくとも以下のような結果が確認できること。

```text
model: FNO
input shape : (8, 7, 64, 256)
output shape: (8, 3, 64, 256)
finite check: passed
backward check: passed
```

## Training

`src/train.py` は、既に実装されている Dataset / DataLoader / Model を利用して、
FNOの学習、validation、評価指標の計算、checkpoint保存、学習状況の記録を行う。

Training側でDataset、DataLoader、Modelの処理を再実装しない。

基本的な処理フローは以下とする。

```text
Dataset
   ↓
DataLoader
   ↓
FNO Model
   ↓
Forward
   ↓
Loss
   ↓
Backward
   ↓
Optimizer step
   ↓
Validation
   ↓
Logging
   ↓
Checkpoint
```


### Baseline Training

初期baselineでは以下の条件を基本とする。

```text
epochs        : 30
loss          : MSE
optimizer     : Adam
learning rate : 5e-4
weight decay  : 0.0
scheduler     : none
```

初回の30 epochは最終的な学習条件を確定するための本学習ではなく、
学習パイプライン全体が正常に機能し、
train / validationの損失が適切に推移することを確認するためのbaseline runとする。

必要に応じて本格的な30 epoch学習の前に、
1〜2 epochの短時間runで動作確認を行えるようにする。

初期baselineでは、

```text
scheduler
early stopping
gradient clipping
mixed precision
```

などを必須としない。

これらはbaselineの学習曲線を確認した後、
必要性を判断して追加・比較する。


## train.yaml

Training固有の設定は `configs/train.yaml` を正本として管理する。

例：

```yaml
training:
  epochs: 30

loss:
  name: mse

optimizer:
  name: adam
  learning_rate: 0.0005
  weight_decay: 0.0

scheduler:
  name: none

checkpoint:
  save_best: true
  save_last: true
  monitor: val_loss
  mode: min

logging:
  wandb:
    enabled: true
    project: FNOV2
    entity: null
    run_name: baseline
    mode: online
    log_every_n_epochs: 1

runtime:
  device: auto
  seed: 42
```

各値は将来的にYAMLから変更可能な構造とし、
実験条件をPythonコードへ固定的にハードコードしない。


## Single Source of Truth

既に他の設定ファイルで定義されている値を
`train.yaml`へ重複して記載しない。

DataLoader固有の、

```text
batch_size
train_shuffle
val_shuffle
num_workers
pin_memory
train / validation split
```

などはDataLoader側の設定を参照する。

Model固有の、

```text
n_modes
hidden_channels
lifting_channels
projection_channels
n_layers
activation
positional_embedding
domain_padding
```

などはModel側の設定を参照する。

Dataset固有の、

```text
input_channels
output_channels
normalization
standardization
```

などもDataset側の設定を参照する。

Trainingは上流設定を利用し、
学習固有の条件のみを管理する。


## Loss

baselineの学習損失にはMean Squared Error（MSE）を使用する。

```text
loss = MSE(prediction, target)
```

現在のDatasetでは、
u, v, pがtrainデータの統計量によって標準化されているため、
MSEは標準化後のpredictionとtargetに対して計算する。

学習用Lossと性能評価指標は区別する。

```text
Optimization:
MSE

Evaluation:
MSE
Relative L2
```

を基本とする。


## Optimizer

baselineではAdam optimizerを使用する。

初期learning rateは、

```text
5e-4
```

とする。

初期baselineではschedulerを使用せず、
30 epochを一定learning rateで学習する。

```text
optimizer : Adam
lr        : 5e-4
scheduler : none
```

learning rateやoptimizerはYAMLから変更可能とし、
将来的な比較実験に対応できる構造とする。


## Training Loop

各epochでは以下の順序で処理する。

```text
model.train()

for batch in train_loader:

    input / target取得

        ↓

    optimizer.zero_grad()

        ↓

    prediction = model(input)

        ↓

    loss計算

        ↓

    backward

        ↓

    optimizer.step()

        ↓

    metric集計

        ↓

epoch終了

        ↓

validation

        ↓

logging

        ↓

checkpoint判定
```

Validationデータをoptimizerの更新に使用してはならない。


## Validation

各epoch終了後にvalidationを実行する。

validationでは、

```text
model.eval()
```

を使用し、
勾配計算を無効化する。

最低限、

```text
Validation MSE
Validation Relative L2
```

を計算する。

Validation結果は、

```text
学習状況の確認
過学習の確認
モデル性能の確認
best checkpointの選択
```

に使用する。


## Metrics

各epochについて最低限以下を計算・記録する。

```text
Train:
- MSE
- Relative L2

Validation:
- MSE
- Relative L2
```

本研究は回帰問題であるため、
classificationで使用されるaccuracyを主要指標とはしない。

Validation性能の主要指標には、

```text
Validation MSE
Validation Relative L2
```

を使用する。

将来的には必要に応じて、

```text
u Relative L2
v Relative L2
p Relative L2
```

など変数別の評価指標を追加可能な構造とする。


## Console Output

学習が正常に進んでいることをターミナルから確認できるよう、
各epoch終了時に学習状況をprintする。

最低限以下を表示する。

```text
Epoch
Train Loss
Validation Loss
Train Relative L2
Validation Relative L2
Learning Rate
Best Validation Loss
Epoch Time
```

表示例：

```text
Epoch [001/030]
  train_loss      : 1.234567e-02
  val_loss        : 1.456789e-02
  train_rel_l2    : 8.123456e-02
  val_rel_l2      : 9.012345e-02
  learning_rate   : 5.000000e-04
  best_val_loss   : 1.456789e-02
  epoch_time      : 12.4 s
```

これにより、

```text
Epoch 1
Epoch 2
Epoch 3
...
```

と学習が実際に進行していることを確認可能にする。

batchごとに大量のprintを行わず、
基本的にはepoch単位で表示する。


## Weights & Biases

学習状況のリアルタイム監視および実験管理には、
Weights & Biases（W&B）を使用する。

W&Bはlogging機能として扱い、
W&Bが無効でも通常の学習が実行可能な構造とする。


### W&B Logging

各epoch終了時に最低限以下をW&Bへ記録する。

```text
epoch

train/loss
val/loss

train/relative_l2
val/relative_l2

learning_rate
epoch_time
```

これによりW&B上で、

```text
Epoch vs Train MSE
Epoch vs Validation MSE

Epoch vs Train Relative L2
Epoch vs Validation Relative L2

Epoch vs Learning Rate
```

をリアルタイムで確認可能にする。

TrainとValidationの同じ指標について、
学習曲線を比較しやすい形で記録する。


### W&B Configuration

1回の学習を1つのW&B Runとして管理する。

各Runには、
実際に使用した実験条件をW&B configとして保存する。

最低限以下を記録する。

```text
Dataset
- input_channels
- output_channels
- normalization settings

DataLoader
- train Re cases
- validation Re cases
- batch_size

Model
- model name
- neuralop version
- n_modes
- hidden_channels
- lifting_channels
- projection_channels
- n_layers
- activation
- positional_embedding
- parameter count

Training
- epochs
- loss
- optimizer
- learning_rate
- weight_decay
- scheduler
- seed
```

W&B用としてこれらを手作業で再入力しない。

既存のDataset / DataLoader / Model / Train設定から
自動的にW&B configへ渡す。


### W&B Run Name

各実験を識別できるよう、
Run Nameを設定可能にする。

例：

```text
baseline
baseline_lr1e-3
no_re
dropout_01
modes_16
```

Run Nameは識別用とし、
実際の実験条件の正本にはしない。

実際の設定値はYAMLおよびW&B configに保存する。


### W&B Failure Handling

W&Bは学習そのものへの必須依存としない。

以下の場合でも学習およびcheckpoint保存が継続できる構造とする。

```text
W&B disabled
ネットワーク切断
W&B logging error
offline環境
```

必要に応じて、

```yaml
logging:
  wandb:
    enabled: false
```

または、

```yaml
logging:
  wandb:
    mode: offline
```

を使用可能にする。


## Checkpoint

学習中にcheckpointを保存する。

最低限、

```text
best checkpoint
last checkpoint
```

を保存可能にする。


### Best Checkpoint

baselineでは、

```text
Validation MSE Lossが最小
```

となったepochをbest modelとする。

各epoch終了後、

```text
current val_loss < best val_loss
```

の場合にbest checkpointを更新する。

更新時には、

```text
[checkpoint] new best model saved
```

などのメッセージをターミナルへ表示する。


### Checkpoint Contents

checkpointには最低限以下を保存する。

```text
model state
optimizer state
epoch
train loss
validation loss
best validation loss
```

加えて、
モデルおよび学習条件を後から再現できる情報を保持する。

checkpoint保存はW&Bから独立して実行する。

W&Bへの通信失敗によって、
学習済みcheckpointが失われてはならない。


## Training History / Curves

W&Bによるリアルタイム監視を基本とするが、
学習履歴はローカルにも保持する。

最低限、

```text
epoch
train_loss
val_loss
train_relative_l2
val_relative_l2
learning_rate
epoch_time
```

を後から参照可能にする。

必要に応じて学習終了後、

```text
loss_curve.png
relative_l2_curve.png
```

などの静的な学習曲線を生成可能な構造とする。

可視化処理の失敗によって、
checkpointや学習結果が失われないようにする。


## Output Directory

各実験の出力を混在させない。

例えば、

```text
outputs/
└── baseline/
    ├── checkpoints/
    │   ├── best.pt
    │   └── last.pt
    │
    ├── history/
    │   └── history.*
    │
    └── plots/
        ├── loss_curve.png
        └── relative_l2_curve.png
```

のように実験単位で管理する。

既存runの結果を誤って上書きしない構造とする。


## Training Start Summary

学習開始時に、
今回実行する主要条件をターミナルへ表示する。

最低限以下を表示する。

```text
device
model name
parameter count

train samples
validation samples
batch size

epochs
loss
optimizer
learning rate
weight decay
scheduler

W&B project
W&B run name
checkpoint directory
```

表示例：

```text
=== Training Configuration ===

device          : cuda

model           : FNO
parameters      : 357,603

train samples   : 2100
val samples     : 400
batch size      : 8

epochs          : 30
loss            : MSE
optimizer       : Adam
learning rate   : 5.0e-4
weight decay    : 0.0
scheduler       : none

wandb project   : FNOV2
wandb run       : baseline

checkpoint dir  : outputs/baseline/checkpoints
```

これにより、
学習開始前に設定ミスを目視確認できるようにする。


## Training Sanity Check

本格的な30 epoch学習の前に、
1〜2 epochの短時間runで動作確認可能にする。

最低限以下を確認する。

1. DataLoaderからbatchを取得できる
2. input / target shapeが正しい
3. Model forwardが成功する
4. MSEが有限値である
5. backwardが成功する
6. optimizer.step()が成功する
7. model parameterが実際に更新される
8. validationが正常に実行される
9. Relative L2が計算できる
10. NaN / Infが発生しない
11. epoch console outputが正常に表示される
12. W&B Runが開始される
13. W&Bへtrain / val metricsが記録される
14. W&B上で学習曲線が更新される
15. best checkpointが保存される
16. last checkpointが保存される
17. 保存したcheckpointを再度loadできる


## Reproducibility

Trainingでは再現性を可能な限り確保する。

seedは設定ファイルから取得し、
使用する乱数生成器へ適切に反映する。

使用した、

```text
Dataset config
DataLoader config
Model config
Train config
neuralop version
PyTorch version
```

などを後から確認できる状態にする。


## Training Implementation Principles

`train.py` は学習処理を担当し、
既存モジュールの責務を重複して持たない。

以下をTraining側で再実装しない。

```text
Datasetの読み込み処理
Normalization / Standardization
train / validation split
DataLoaderのbatch生成
FNO内部構造
Model forwardの独自再実装
```

Trainingは、

```text
Dataset
DataLoader
Model
```

の既存実装を利用し、

```text
Optimization
Validation
Metrics
Logging
Checkpoint
```

を担当する。

実験条件は可能な限りYAMLから変更可能とし、
実験ごとにPythonコードを書き換える設計を避ける。

## Performance Test DataLoader

performance評価では、train / validation用のDataLoaderとは処理を分離し、
評価専用のtest loaderを使用する。

ただし、Dataset処理やDataLoader共通設定を重複実装しない。

基本構成は以下とする。

```text
dataset.py
   ↓
共通のDataset処理
   ↓
test_dataloader.py
   ↓
test_loader
   ↓
performance/src/test.py
```

### 責務

`test_dataloader.py` は主に以下を担当する。

```text
performance用データディレクトリの指定
評価対象Reの選択
batch化
test_loader生成
```

以下はtest_dataloader側で再実装しない。

```text
pair .ptの生成処理
Datasetのchannel選択
Normalization / Standardization
train統計量の算出
FNOモデル処理
評価metric計算
```


## Train / Validationとの分離

train / validation用データとperformance評価用データは、
ディレクトリレベルで明確に分離する。

例：

```text
FNOV2/
├── data/
│   # train / validation専用
│
└── performance/
    └── data/
        ├── interpolation/
        └── extrapolation/
```

train / validation側のDataLoaderは、
`data/` 配下のみを探索する。

performance用test loaderは、
`performance/data/` 配下のみを探索する。

performanceデータをtrainまたはvalidationへ混入させてはならない。


## Interpolation / Extrapolation

interpolation用とextrapolation用に
別々のDataLoader実装を作成しない。

共通の`test_dataloader.py`を使用し、
評価対象ディレクトリおよびReynolds numberを
performance用configから切り替える。

概念的には、

```text
test_dataloader.py
       │
       ├── interpolation data
       │
       └── extrapolation data
```

とする。

コードは共通化し、
データ・設定・評価結果のみを分離する。


## test.yaml

performance評価固有の条件は、
performance側のconfigで管理する。

例：

```yaml
evaluation:
  type: interpolation

data:
  directory: ../data/interpolation

test_re:
  - 35
  - 85
  - 135
  - 185
  - 235
```

外挿評価では例えば、

```yaml
evaluation:
  type: extrapolation

data:
  directory: ../data/extrapolation

test_re:
  - 260
  - 275
  - 300
```

のように変更可能とする。

Pythonコード内へ評価対象Reをハードコードしない。


## DataLoader共通設定

test loaderのbatch生成に関する共通設定は、
既存の`configs/dataloader.yaml`を参照する。

例えば以下の設定は、
既存DataLoader設定を正本とする。

```text
batch_size
num_workers
pin_memory
drop_last
persistent_workers
seed
```

test用として同じ値を別のYAMLへ重複記載しない。

概念的には、

```text
configs/dataloader.yaml
        ↓
共通loader設定
        ↓
test_dataloader.py
        ↑
performance test config
        ↓
data directory / test Re
```

という依存関係とする。

ただしtestでは原則として、

```text
shuffle = false
```

とする。


## Datasetの再利用

test用に新しいDatasetクラスを複製しない。

既存の`dataset.py`を使用して、
train / validation時と同じchannel処理、
Normalization / Standardizationを適用する。

特にtestデータに対して、
test自身のmean / stdを新たに算出してはならない。

u / v / pのStandardizationには、
学習時に確定したtrain統計量を使用する。


## Extrapolation時のScaling

外挿評価でも、
学習時に使用したNormalization / Scaling条件を変更しない。

例えばReynolds numberについて、

```text
Re_scaled = Re / 250
```

を学習時に使用した場合、
外挿Reでも同じ式を使用する。

例：

```text
Re = 250 -> 1.00
Re = 260 -> 1.04
Re = 275 -> 1.10
Re = 300 -> 1.20
```

test条件に合わせてscale値を変更してはならない。

同様に、

```text
x / y scaling
u / v / p mean
u / v / p std
```

についても学習時の変換を固定して使用する。


## Case IDの扱い

performance用データでは、
train / validation用データとcase番号が重複する可能性がある。

例：

```text
train:
case0_Re10

performance:
case0_Re85
```

このため、`case_idx`をグローバルに一意な識別子として扱わない。

test loaderでは、

```text
検索対象directory
Re metadata
```

を利用して評価対象を識別する。

case番号だけを使って
train / validationデータとperformanceデータを横断検索しない。


## Test Loader Sanity Check

test loader実装後は、
推論処理へ進む前に最低限以下を確認する。

1. 指定したperformance directoryのみを探索している
2. 指定したReのみが読み込まれている
3. train / validation用データが混入していない
4. test用Re一覧が想定通りである
5. test sample数が想定通りである
6. batch shapeが正しい
7. input channel数がDataset設定と一致する
8. output channel数がDataset設定と一致する
9. NaN / Infが存在しない
10. testデータにtrain統計量が使用されている
11. test自身からNormalization統計量を算出していない
12. interpolation / extrapolationの切替がconfigのみで可能である

表示例：

```text
=== Test DataLoader Summary ===
evaluation     : interpolation
data directory : performance/data/interpolation

test Re:
[35, 85, 135, 185, 235]

test files     : 5
test samples   : 500

batch x shape  : (8, 7, 64, 256)
batch y shape  : (8, 3, 64, 256)

shuffle        : false
finite check   : passed
```

外挿評価でも同一のtest loaderを使用し、
config変更のみで切り替えられることを確認する。


## Implementation Principle

performance評価では、

```text
コードは可能な限り共通化
データは明確に分離
設定はYAMLで切替
結果は評価条件ごとに分離
```

を基本原則とする。

特に、
interpolation / extrapolationの違いだけを理由として
ほぼ同一のDataset・DataLoader・testコードを複製しないこと。

## Performance One-Step Test

`performance/src/test.py` は、学習済みFNOモデルに対して未学習Reの1-step予測性能を評価するためのスクリプトとする。

interpolation / extrapolationで別々のtestスクリプトを作成せず、共通の`test.py`を使用する。

評価対象の種類、データディレクトリ、Reynolds numberはperformance用YAMLから切り替える。


### 基本処理フロー

```text
test config
   ↓
test_dataloader.py
   ↓
test_loader
   ↓
best checkpoint load
   ↓
model.eval()
   ↓
1-step inference
   ↓
inverse transform
   ↓
MSE / Relative L2
   ↓
prediction / ground truth保存
   ↓
代表時刻のみ可視化
```

test処理ではモデルの重みを更新しない。

必ず推論モードを使用し、勾配計算を無効化する。


### Test Metrics

1-step性能の主要評価指標は以下とする。

- MSE
- Relative L2

評価値はtest対象の全サンプルを使用して計算する。

例えば、1つのReケースについて100 pairs存在する場合、100サンプルすべてをmetric計算対象とする。

可能であれば以下を保存する。

- overall MSE
- overall Relative L2
- u MSE
- v MSE
- p MSE
- u Relative L2
- v Relative L2
- p Relative L2

評価値については、標準化空間で計算したものなのか、inverse transform後の物理量で計算したものなのかを明確にすること。

研究結果として使用するmetricについては、どのスケールで算出した値か必ず記録する。


### Test Result Storage

各Reの1-step予測結果は、後から再推論せず可視化・追加解析できるよう保存する。

想定構成：

```text
performance/
└── results/
    └── interpolation/
        └── one_step/
            ├── metrics.csv
            ├── metrics.json
            ├── 1step_Re35.pt
            ├── 1step_Re85.pt
            ├── 1step_Re135.pt
            ├── 1step_Re185.pt
            └── 1step_Re235.pt
```

extrapolationの場合も同様に、

```text
performance/results/extrapolation/one_step/
```

へ保存する。

`1step_ReXX.pt`には最低限以下を保存する。

- prediction
- ground truth
- Re
- time / step information
- mask
- 必要なmetadata

predictionとground truthは、後からu / v / pの可視化や誤差場計算が可能な形式で保存する。

保存するpredictionが標準化後なのかinverse transform後なのかをmetadataで明確にする。

可能であれば、可視化や物理的評価に直接利用できるよう、inverse transform後の物理量を保存することを優先する。


### Visualization Policy

metric計算にはtest対象の全サンプルを使用する。

一方、100サンプルすべてを静止画として可視化する必要はない。

可視化は代表時刻のみを抽出する。

1つのReについて100サンプル存在する場合、基本的には時間範囲から等間隔に5点程度を選択する。

代表点は概念的に以下とする。

- 開始付近
- 25%地点
- 50%地点
- 75%地点
- 終了付近

100サンプルの場合の例：

```text
sample 0
sample 25
sample 50
sample 75
sample 99
```

ただし、固定indexをPythonコードへ直接ハードコードするのではなく、実際のサンプル数に応じて等間隔に自動選択できる実装を優先する。


### Maximum Error Visualization

等間隔の代表時刻に加えて、1-step Relative L2が最大となるサンプルを自動的に検出し、可視化対象へ追加する。

これにより、

```text
時間的に均等な代表場
+
最も予測誤差が大きかった場
```

の両方を確認可能にする。

最大誤差サンプルがすでに等間隔代表点に含まれている場合は、重複して画像生成しない。


### Visualization Contents

代表時刻について最低限以下を比較可能にする。

- Ground Truth
- Prediction
- Absolute Error

対象物理量は以下とする。

- u
- v
- p

各物理量について、Ground Truth / Prediction / Absolute Errorの空間分布を確認できるようにする。

可視化処理はtestのmetric計算・prediction保存と可能な限り分離する。

画像生成処理に失敗した場合でも、metric計算結果や`.pt`ファイルが失われない構造にする。


### Visualization Output

例えばRe=85の場合、以下のようにReごとに可視化結果を追跡できる構造とする。

```text
performance/
└── results/
    └── interpolation/
        └── one_step/
            ├── 1step_Re85.pt
            └── plots/
                └── Re85/
                    ├── sample000_u.png
                    ├── sample000_v.png
                    ├── sample000_p.png
                    ├── sample025_u.png
                    ├── sample025_v.png
                    ├── sample025_p.png
                    ├── ...
                    └── max_error_*.png
```

ファイル名には少なくとも以下が分かる情報を含める。

- Re
- sample indexまたはtime
- physical variable

後からどの条件・どの時刻・どの物理量の結果か判別できる命名にする。


### Interpolation / Extrapolation

同じ`test.py`を用いて、interpolation / extrapolationをYAMLから切り替える。

例：

```yaml
evaluation:
  type: interpolation

data:
  directory: ../data/interpolation

test_re:
  - 35
  - 85
  - 135
  - 185
  - 235
```

外挿では例えば、

```yaml
evaluation:
  type: extrapolation

data:
  directory: ../data/extrapolation

test_re:
  - 260
  - 275
  - 300
```

のように指定する。

コードを複製して`interpolation_test.py`と`extrapolation_test.py`を別々に作成しない。

コードは共通化し、データ・設定・結果のみを分離する。


### Normalization / Standardization

test時にも学習時と完全に同じ変換条件を使用する。

testデータ自身からmean / stdを計算してはならない。

基本方針：

```text
mask
→ 学習時と同じ処理

Re
→ 学習時と同じscale

x / y
→ 学習時と同じscale

u / v / p
→ train統計量でstandardization
```

外挿時もscaleを変更しない。

例えば学習時に、

```text
Re_scaled = Re / 250
```

を使用した場合、

```text
Re260 -> 1.04
Re275 -> 1.10
Re300 -> 1.20
```

としてそのまま入力する。

外挿データに合わせてReのscale値を変更してはならない。

u / v / pについても、performanceデータ自身から新しいmean / stdを算出せず、学習時に確定したtrain統計量を使用する。


### Test Sanity Check

test実装後は最低限以下を確認する。

1. 指定したcheckpointが正常にloadされる
2. modelがeval modeになっている
3. gradient計算が無効である
4. 指定したtest Reのみが使用される
5. train / validationデータが混入していない
6. train統計量によるNormalization / Standardizationが使用されている
7. input / output shapeが正しい
8. predictionにNaN / Infが存在しない
9. MSEを全test sampleで計算できる
10. Relative L2を全test sampleで計算できる
11. Reごとの結果が独立して保存される
12. `1step_ReXX.pt`が正常に再読み込みできる
13. inverse transformが正常に機能する
14. 等間隔代表点が正しく選択される
15. 最大誤差sampleが正しく検出される
16. 可視化処理の失敗がmetric / prediction保存へ影響しない


### Console Output

test実行時には、最低限以下をターミナルへ表示する。

例：

```text
=== One-Step Performance Test ===
evaluation : interpolation
checkpoint : .../best.pt

Re=85
samples       : 100

MSE
overall       : ...
u             : ...
v             : ...
p             : ...

Relative L2
overall       : ...
u             : ...
v             : ...
p             : ...

saved:
1step_Re85.pt

visualization samples:
[0, 25, 50, 75, 99]

maximum error sample:
index       : ...
time        : ...
Relative L2 : ...
```

複数Reを評価した場合は、最後にReごとの結果を一覧表示できるようにする。


### Implementation Principle

one-step performance評価では、

```text
全testサンプル
→ quantitative evaluation
→ MSE / Relative L2

代表5時刻程度
→ qualitative visualization

最大Relative L2時刻
→ worst-case visualization
```

と役割を分ける。

100枚すべてを静止画として生成・確認する設計にはしない。

ただしprediction / ground truth自体は全サンプル保存し、後から任意の時刻を再可視化できる状態を維持する。

また、performance評価は最終評価用であり、このtest結果をtrain / validation処理へフィードバックしてデータを混在させないこと。
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