# Repository Guidelines

## Project Structure & Module Organization
PND/ houses the Flower segmentation system. Put the VOC-style dataset in `Panax notoginseng disease dataset/VOC2007/...` so `fl_pnd/dataset.py` can locate JPEGImages, SegmentationClass, and ImageSets. Runtime logic stays under `fl-pnd/fl_pnd/` (client/server apps, dataset helpers, ladder data classes, serde tools, model code). `run.py` runs the Ray-managed Ladder strategy, `test_ray_gpu.py` is the CUDA smoke test, `ladder-fl/` tracks decentralized prototypes, `doc/` stores research notes, and `logs/` holds long experiments.

## Build, Test, and Development Commands
Activate a virtualenv (`python -m venv venv && source venv/bin/activate`) and install with `pip install -e fl-pnd/`. After curating the dataset path, `python run.py` initializes Ray, partitions ten clients, and executes the configured federated rounds. Use `cd fl-pnd && flwr run .` when Flower’s CLI should manage orchestration, and re-run `python test_ray_gpu.py` whenever CUDA, drivers, or Ray versions change.

## Coding Style & Naming Conventions
Code targets Python 3.9+, PEP 8, four-space indents, snake_case functions, CapWords classes. Prefer explicit type hints and dataclasses for new ledger or serde structures, reuse the docstring tone in `task.py`, and keep module-level constants uppercase (`DEVICE`, `NUM_CLIENTS`). Favor f-strings for logging and park tunables in `pyproject.toml` or `run.py` rather than scattering literals.

## Testing Guidelines
`python run.py` already reports foreground pixel accuracy and mIoU on the shared validation loader; treat that console output (or the saved `logs/` entry) as the acceptance signal before pushing. Extend `test_ray_gpu.py` or add `tests/test_*.py` scripts collected by `pytest` to cover dataset transforms, serialization, and Ladder strategy math with lightweight synthetic samples.

## Commit & Pull Request Guidelines
History follows Conventional Commits (`feat:`, `fix:`, `chore:`). Use `type(scope): summary` where scope maps to the main module (`feat(client): reuse cached weights`). Every PR should link an issue or task, list the commands you ran (`python run.py`, `flwr run .`, `pytest`), and include the tail of the latest training log or Ray screenshot so reviewers can verify behaviour quickly.

## Security & Configuration Tips
Never commit the dataset or secrets; ensure `.gitignore` continues to exclude `Panax notoginseng disease dataset/` and bulky log files. Keep Ray GPU settings consistent between `tool.flwr.client_resources` and the custom `ray-init-args`, set `CUDA_VISIBLE_DEVICES` explicitly on shared hosts, and store federation endpoints in environment variables rather than source.
