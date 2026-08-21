# Performance evaluation source

- `test_dataloader.py` safely loads performance pairs with the training Dataset and normalization settings.
- `test.py` loads the trained checkpoint and performs one-step inference and metric reporting.
- A future `rollout.py` should reuse the same loader and keep autoregressive evaluation separate from one-step evaluation.

All code here must use configs from `performance/configs`, validate internal Reynolds-number metadata, and never search the project-level training `data/` directory.
