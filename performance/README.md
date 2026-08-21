# Performance evaluation

This directory is reserved for final evaluation data that must never be used for training or validation. Case numbers are local to each data root: `data/interpolation/VTK/case0` and the training `data/VTK/case0` are unrelated. The Reynolds-number metadata in each generated pair file is the physical case identifier used for evaluation.

## Directory boundaries

```text
performance/
├── configs/
├── data/
│   ├── interpolation/{VTK,processed/pairs}
│   └── extrapolation/{VTK,processed/pairs}
├── src/
└── results/
    ├── interpolation/{one_step,rollout}
    └── extrapolation/{one_step,rollout}
```

Never point the training `configs/dataset.yaml` at `performance/data`. Performance evaluation code must only discover files below the `data_dir` in its own config. Do not use a repository-wide recursive search for pair files.

## Generate pair files

The existing `src/make_pairs.py` is reused; no preprocessing implementation is duplicated. Put VTM data in the matching `VTK/caseN` directory, verify the local `reynolds_by_case` mapping, then run:

```bash
python src/make_pairs.py --config performance/configs/make_pairs_interpolation.yaml
python src/make_pairs.py --config performance/configs/make_pairs_extrapolation.yaml
```

Each output `.pt` stores both `meta.case` and `Re`. Validate the generated directory and confirm its manifest contains the intended Re values:

```bash
python src/validate_pairs.py performance/data/interpolation/processed/pairs
python src/validate_pairs.py performance/data/extrapolation/processed/pairs
```

## One-step evaluation

`test_dataloader.py` resolves only the configured performance directory, checks every pair file's internal Reynolds-number metadata against `test_re`, and applies the training Dataset configuration and train-only normalization statistics.

`test.py` then loads the fixed checkpoint, runs one-step inference with `model.eval()` and inference mode, and writes metrics under `results/<evaluation.type>/one_step/`.

```bash
python performance/src/test_dataloader.py \
  --config performance/configs/test_dataloader.yaml

python performance/src/test.py \
  --config performance/configs/test.yaml
```

The saved JSON and CSV contain overall and per-Re MSE/Relative L2, split by `u`, `v`, and `p`. Both standardized and physical scales, and both the full grid and mask-valid region, are reported. Prediction tensors are not saved.

For extrapolation only, `extrapolation_relative_l2.png` is also generated. Its horizontal axis is the distance from the training boundary (`Re - 250`), and its vertical axis is the physical-scale Relative L2 error on the mask-valid region in percent. Overall, `u`, `v`, and `p` are plotted separately. Interpolation evaluation does not generate this plot.

Switch interpolation/extrapolation through `evaluation.type`, `data.directory`, and `test_re` in `test_dataloader.yaml`. Common loader settings come from `configs/dataloader.yaml`, and test shuffle is always disabled.

## Test dataset checklist

Changing only the input/output directory is not sufficient. Before generating or evaluating a test dataset, check all of the following:

- Place raw VTM files under the intended `interpolation/VTK` or `extrapolation/VTK` root. Never combine them with the training `data/VTK` tree.
- Update `reynolds_by_case` in the corresponding make-pairs config. A case number is only a directory-local label; its configured Reynolds number is the physical identity of the case.
- Keep `test_re` in `test_dataloader.yaml` consistent with `reynolds_by_case`.
- Unless the evaluation intentionally studies preprocessing changes, keep `step_to_time`, time range, `dt_pair`, grid bounds/resolution, field names, and interpolation method identical to the training-data generation conditions.
- Reuse the training Dataset configuration and train-only normalization statistics. Never calculate mean/std from interpolation or extrapolation test data. In particular, keep the trained Re scaling unchanged even when an extrapolation Re produces a scaled value greater than one.
- After pair generation, run `src/validate_pairs.py` on the output directory and inspect `manifest.json` for the expected case, Re, pair count, and tensor shape.
- Do not use final test results to select checkpoints or tune model/training hyperparameters. Data used for those decisions becomes validation data and a separate untouched final test set is then required.
