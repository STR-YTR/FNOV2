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

## Evaluation contract

The evaluation configs refer to the training `dataset.yaml`. Its channel selection and train-only normalization statistics remain the single source of truth; performance data must not produce its own normalization statistics.

Evaluation code added under `performance/src` must:

1. resolve only the configured performance `data_dir`;
2. verify every `.pt` file's internal `Re` metadata against `expected_re`;
3. treat `meta.case` as display/provenance metadata, never as a global ID;
4. load the fixed trained checkpoint without optimization or model selection;
5. save one-step and rollout outputs to their distinct configured directories.

The test DataLoader can be checked before implementing inference:

```bash
python performance/src/test_dataloader.py \
  --config performance/configs/test_dataloader.yaml
```

Switch interpolation/extrapolation by changing only `evaluation.type`,
`data.directory`, and `test_re` in that config. Common loader settings come from
`configs/dataloader.yaml`, and test shuffle is always disabled.

No one-step or rollout evaluator is added yet because the repository currently
has no inference implementation to reuse. The configs and test loader establish
the path and metadata contract without changing the working training path.

## Test dataset checklist

Changing only the input/output directory is not sufficient. Before generating or evaluating a test dataset, check all of the following:

- Place raw VTM files under the intended `interpolation/VTK` or `extrapolation/VTK` root. Never combine them with the training `data/VTK` tree.
- Update `reynolds_by_case` in the corresponding make-pairs config. A case number is only a directory-local label; its configured Reynolds number is the physical identity of the case.
- Keep `expected_re` in the evaluation config consistent with `reynolds_by_case`.
- Unless the evaluation intentionally studies preprocessing changes, keep `step_to_time`, time range, `dt_pair`, grid bounds/resolution, field names, and interpolation method identical to the training-data generation conditions.
- Reuse the training Dataset configuration and train-only normalization statistics. Never calculate mean/std from interpolation or extrapolation test data. In particular, keep the trained Re scaling unchanged even when an extrapolation Re produces a scaled value greater than one.
- After pair generation, run `src/validate_pairs.py` on the output directory and inspect `manifest.json` for the expected case, Re, pair count, and tensor shape.
- Do not use final test results to select checkpoints or tune model/training hyperparameters. Data used for those decisions becomes validation data and a separate untouched final test set is then required.
