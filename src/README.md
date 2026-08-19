# pairデータ作成・確認手順

このフォルダのスクリプトは、OpenFOAM由来のVTM時系列からFNO学習用の
one-step pairを作成し、生成した`.pt`の内容を検査するために使用します。

## 作業の流れ

```text
VTM時系列
  ↓ make_pairs.py
学習用pairs_case*_Re*.pt
  ↓ validate_pairs.py
shape・channel・時刻・Re・NaN/Inf・maskを確認
  ↓ dataset.py
チャネル選択・Normalization
  ↓ dataloader.py
Reケース単位のtrain / validation分割・batch化
```

以下のコマンド例は、プロジェクトルート
`/workspace/FNOV2`にいる状態を基準にしています。

```bash
cd /workspace/FNOV2
```

## 1. 入力データと設定を確認する

VTMファイルを次の場所に置きます。

```text
data/VTK/case0/case0_34000.vtm
data/VTK/case0/case0_34010.vtm
...
```

作成条件は`configs/make_pairs.yaml`で設定します。主な項目は次のとおりです。

- `data.input_dir`: VTMの入力フォルダ
- `data.output_dir`: `.pt`の出力フォルダ
- `data.reynolds_by_case`: case番号とReynolds数の対応
- `time.dt_pair`: 入力時刻と教師時刻の間隔
- `grid.nx`, `grid.ny`: 規則格子の解像度

## 2. pairデータを作成する

```bash
python src/make_pairs.py --config configs/make_pairs.yaml
```

正常に終了すると、既定では次のようなファイルが作成されます。

```text
data/processed/pairs/pairs_case0_Re10.pt
data/processed/pairs/manifest.json
```

`.pt`の主な内容は次のとおりです。

- `x`: `[N, 7, H, W]`、チャネルは`mask, Re, x, y, u(t), v(t), p(t)`
- `y`: `[N, 3, H, W]`、チャネルは`u(t+dt), v(t+dt), p(t+dt)`
- `times`: 各pairの入力時刻と教師時刻
- `steps`: 各pairの入力stepと教師step
- `Re`: Reynolds数
- `meta`: 格子、時刻間隔、チャネルなどの作成条件

## 3. 作成した`.pt`を検査する

プロジェクトルートから実行する場合：

```bash
python src/validate_pairs.py data/processed/pairs/pairs_case0_Re10.pt
```

どのディレクトリからでも実行できる絶対パスの例：

```bash
python /workspace/FNOV2/src/validate_pairs.py \
  /workspace/FNOV2/data/processed/pairs/pairs_case0_Re10.pt
```

`src`フォルダにいる場合：

```bash
cd /workspace/FNOV2/src
python validate_pairs.py ../data/processed/pairs/pairs_case0_Re10.pt
```

先頭から目視表示するpair数は`--rows`で変更できます。

```bash
python src/validate_pairs.py data/processed/pairs/pairs_case0_Re10.pt --rows 10
```

値の行を表示せず、検査結果だけ確認する場合：

```bash
python src/validate_pairs.py data/processed/pairs/pairs_case0_Re10.pt --rows 0
```

## 4. 検査結果を確認する

正常な場合は、各検査が`[OK]`になり、最後に`[PASS]`が表示されます。

```text
[OK] shape: x=(100, 7, 64, 256), y=(100, 3, 64, 256), pairs=100
[OK] channels: input=['mask', 'Re', 'x', 'y', 'u', 'v', 'p'], output=['u', 'v', 'p']
[OK] time pairing: dt=0.1, time=[340.0, 340.1], steps=[34000, 34010]
[OK] Re: Re=10, channel range=10..10
[OK] NaN/Inf: finite values: ...
[OK] mask: binary, valid fraction=...

--- sample pairs (first 5) ---
idx  steps          times             cell(y,x)  mask Re ...
  0  34000->34010     340->340.1      ...
  1  34010->34020   340.1->340.2      ...

[PASS] pairs sanity check
```

検査内容は次のとおりです。

- `shape`: `x`, `y`, `times`, `steps`の次元とpair数が一致しているか
- `channels`: 入出力チャネルの名前と順番が正しいか
- `time pairing`: `t → t+dt`、step、`step_to_time`が整合しているか
- `Re`: 保存されたReと入力のReチャネルが一致しているか
- `NaN/Inf`: テンソル内にNaNまたはInfがないか
- `mask`: maskが0/1で、mask外の`u, v, p`が0になっているか

どれかに問題がある場合は`[FAIL]`を表示し、終了コード`1`で終了します。

## よくあるエラー

### `the following arguments are required: pt`

検査する`.pt`を指定していません。

```bash
python src/validate_pairs.py data/processed/pairs/pairs_case0_Re10.pt
```

### `file does not exist`

指定したパスにファイルがありません。相対パスは、スクリプトの場所ではなく
現在いるディレクトリを基準に解釈されます。迷う場合は絶対パスを使用してください。

### `unrecognized arguments`

フォルダ名とファイル名の間に空白が入り、別々の引数として解釈されています。

```text
# 誤り
.../processed/pairs/ pairs_case0_Re10.pt

# 正しい
.../processed/pairs/pairs_case0_Re10.pt
```

Linuxではファイル名の大文字・小文字も区別されます。
`Pairs_case0_Re10.pt`と`pairs_case0_Re10.pt`は別の名前です。

## フォルダ内の`.pt`を一括検査する

`data/processed/pairs`直下のすべての`.pt`をファイル名順に検査できます。

```bash
cd /workspace/FNOV2
python src/validate_pairs.py data/processed/pairs
```

`src`フォルダにいる場合は次のコマンドです。

```bash
python validate_pairs.py ../data/processed/pairs
```

各ファイルで先頭3 pairだけ目視確認する例：

```bash
python validate_pairs.py ../data/processed/pairs --rows 3
```

各ファイルの詳細ログの末尾と、全体ログの末尾は次の形式になります。

```text
[PASS] pairs sanity check | ../data/processed/pairs/pairs_case0_Re10.pt
[PASS] pairs sanity check | ../data/processed/pairs/pairs_case1_Re20.pt
...
[PASS] 6/6 .pt files | /workspace/FNOV2/data/processed/pairs
```

不合格のファイルがあっても残りを最後まで検査します。最後の集計は
`[FAIL] 5/6 passed, 1 failed | ...`の形式になり、終了コードは`1`です。

## 5. Datasetを確認する

`dataset.py`は生成済みのpair `.pt`を読み込み、1サンプルずつ
`[C, H, W]`形式で返します。設定は`configs/dataset.yaml`で管理します。

```bash
python src/dataset.py --config configs/dataset.yaml
```

Pythonコードから使用する例：

```python
from src.dataset import PairDataset

dataset = PairDataset.from_config("configs/dataset.yaml")
x, y = dataset[0]
```

既定のshapeは次のとおりです。

```text
x: [7, 64, 256]
y: [3, 64, 256]
```

`input_channels`および`output_channels`は`dataset.yaml`を正本とします。
例えば`Re`を入力から除外すると、Datasetが名前からチャネル位置を解決し、
返される入力チャネル数も自動的に変わります。チャネル選択のためにpair `.pt`を
再生成する必要はありません。

## 6. Train / Validation DataLoaderを確認する

Reケース単位の分割条件とDataLoader設定は`configs/dataloader.yaml`で管理します。

```bash
python src/dataloader.py --config configs/dataloader.yaml
```

baselineでは次の分割を使用します。

- train: 21ケース、Re 10〜250のうちvalidationを除いたケース
- validation: Re 50、100、150、200の4ケース
- Re 260〜300: train / validationには含めない

実行すると、train / validationの割合、caseとReの一覧、サンプル数、
最初のbatch shape、Re重複の有無が表示されます。

```text
pair files: train=21, val=4, total=25
train cases (84.0%): case0 (Re=10), ...
val cases   (16.0%): case4 (Re=50), ...
samples: train=2100 (84.0%), val=400 (16.0%), total=2500
train batch: x=(8, 7, 64, 256), y=(8, 3, 64, 256)
val batch:   x=(8, 7, 64, 256), y=(8, 3, 64, 256)
```

Pythonコードから使用する例：

```python
from src.dataloader import create_dataloaders

train_loader, val_loader = create_dataloaders("configs/dataloader.yaml")
```

trainとvalidationに同一Reが指定された場合、存在しないReが指定された場合、
またはNormalization統計の`train_re`と現在のsplitが一致しない場合は、
データリークや設定不整合を防ぐためエラーで停止します。

## 7. Trainデータのチャネル統計を確認する

`check_channel_stats.py`は`dataloader.yaml`の`split.train_re`だけを使用します。
validationおよびtestは統計量計算に含めません。

```bash
python src/check_channel_stats.py --config configs/dataloader.yaml
```

入力・出力の各チャネルについて、次の値を表示します。

- `min`, `max`, `mean`, `std`
- `p01`, `p50`, `p99`
- 全格子点の統計
- `mask == 1`だけの統計
- mask内外の点数と割合
- mask外に保存された各チャネルの非ゼロ数と最大絶対値

全サンプルを巨大なTensorへ結合せず、1サンプルずつ二周して集計します。
mean/stdは逐次計算し、percentileは既定で4096-binヒストグラムから近似します。
精度と処理時間のバランスは次のように変更できます。

```bash
python src/check_channel_stats.py \
  --config configs/dataloader.yaml \
  --histogram-bins 8192
```

現在のpairでは、maskはsourceとtargetの両方で有効な点を1とします。
mask外の入力・出力`u/v/p`は0ですが、`Re/x/y`はmask外でも保持されています。

## 8. Normalization / Standardization

Normalization条件は`configs/dataset.yaml`、trainから算出した統計値と
算出元metadataは`configs/normalization_stats.yaml`を正本とします。
元のpair `.pt`は物理量の生値のまま保持されます。

baselineの変換は次のとおりです。

- `mask`: 変換しない
- `Re`: 250で除算
- `x`: 1.6で除算
- `y`: 0.41で除算
- `u/v/p`: trainの`mask == 1`領域から算出したmean/stdで標準化

`u/v/p`には入力とtargetで同じtrain統計を使用します。validationにもtrain統計を
適用し、validation自身の統計は使用しません。標準化後もmask外の`u/v/p`は0です。

物理スケールへ戻す場合はDatasetの共通inverse transformを使用します。

```python
from src.dataset import PairDataset

dataset = PairDataset.from_config("configs/dataset.yaml")
x, y = dataset[0]

mask_index = dataset.input_channels.index("mask")
mask = x[mask_index].bool()

x_physical = dataset.inverse_transform_input(x, mask)
y_physical = dataset.inverse_transform_output(y, mask)
```

`inverse_transform_input`と`inverse_transform_output`は、単一サンプル
`[C, H, W]`とbatch `[B, C, H, W]`の両方に対応します。batchの場合はmaskも
`[B, H, W]`で渡します。

生データを明示的に確認する必要がある場合は、Normalizationを無効にして読み込めます。

```python
raw_dataset = PairDataset.from_config(
    "configs/dataset.yaml",
    apply_normalization=False,
)
```
