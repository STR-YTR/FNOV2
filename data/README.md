# データフォルダ構成

```text
data/
├── openfoam/         OpenFOAMのケースデータ（必要な場合のみ保管）
├── vtk/              foamToVTKで変換したVTK/VTM時系列データ
└── processed/pairs/  FNO学習用のPyTorch pairデータとmanifest.json
```

## 各フォルダの役割

### `openfoam/`

OpenFOAMの計算ケース本体を保管するフォルダです。
`constant/`、`system/`、各時刻ディレクトリなどを含むケースを置きます。

`make_pairs.py`はこのフォルダを直接読み込みません。
OpenFOAMケースをプロジェクト内に保管する必要がない場合は、空のままで問題ありません。

### `vtk/`

`foamToVTK`でOpenFOAMの計算結果から変換した、VTK/VTM時系列データを置きます。
`src/make_pairs.py`が入力として読み込むフォルダです。

既定のファイル名は、次の形式を想定しています。

```text
case<ケース番号>_<ステップ番号>.vtm
```

例：

```text
case0_34000.vtm
case0_34010.vtm
case1_34000.vtm
```

入力フォルダやファイル名の規則は、`configs/make_pairs.yaml`で変更できます。

### `processed/pairs/`

`src/make_pairs.py`が生成したFNO学習用データを保存するフォルダです。
ケースごとの`.pt`ファイルと、生成結果をまとめた`manifest.json`が保存されます。

## Gitでの管理について

OpenFOAMの計算結果、VTK/VTM、生成済みdatasetは容量が大きくなるため、
このリポジトリではGitの管理対象外です。
これらのファイルは`.gitignore`で除外されているため、Gitへcommitしないでください。
