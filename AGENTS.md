# Repository Guidelines

## Project Structure and Module Organization
Core code lives in `cs336_basics/` with training and generation entry points. Tests are in `tests/`, and fixtures/snapshots are in `tests/fixtures` and `tests/_snapshots`. Data and artifacts are stored outside the code.
- `cs336_basics/myoperator.py` implements model ops, optimizers, and schedulers.
- `cs336_basics/train.py` runs training using `local_training_config.json` by default.
- `cs336_basics/generate.py` runs sampling using `generation_config.json`.
- `data/` holds downloaded text and token id `.npy` files.
- `*_tokenizer/` holds BPE vocab/merges; `checkpoints/` stores model weights.
- `run.slurm` and `slurm_commands.sh` contain cluster helpers; `make_submission.sh` packages a submission zip.

## Build, Test, and Development Commands
Use `uv` for environment management; it resolves and runs dependencies on demand.
- `uv run pytest` runs the full test suite.
- `uv run pytest tests/test_model.py` runs a focused test file.
- `uv run cs336_basics/train.py` trains with `local_training_config.json`.
- `uv run cs336_basics/generate.py` generates text from the configured checkpoint.
- `bash make_submission.sh` runs tests and creates `cs336-spring2025-assignment-1-submission.zip`.
- See `README.md` for data download commands into `data/`.

## Coding Style and Naming Conventions
Python 3.13, 4-space indentation, and a 120-character line length (Ruff config in `pyproject.toml`). Prefer type hints where practical (many functions already use typing/jaxtyping). Use `snake_case` for functions/variables, `CamelCase` for classes, and `UPPER_SNAKE` for constants. No auto-formatter is enforced, so keep diffs tidy and consistent with existing files.

## Testing Guidelines
Tests use pytest and follow the `tests/test_*.py` naming pattern. Implementations are wired through `tests/adapters.py`, so update that file when swapping implementations. Keep new tests deterministic and avoid large data downloads inside tests; use `tests/fixtures` for small artifacts.

## Commit and Pull Request Guidelines
Commit history uses short, descriptive, one-line subjects (often in Chinese) without scope prefixes. Keep messages concise and focused on what changed. For PRs, include a brief summary, list tests run (e.g. `uv run pytest`), and call out any changes to configs, datasets, or checkpoints. Avoid committing large artifacts (model weights, datasets, wandb logs) unless explicitly required.

## Configuration and Secrets
Training and generation behavior is controlled by `local_training_config.json`, `training_config.json`, and `generation_config.json`. When enabling Weights & Biases logging, configure credentials via environment variables or local setup and do not commit keys or tokens.
