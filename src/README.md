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
