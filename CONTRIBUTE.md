# Contributing to Emotion Recognition - Real-time

Thank you for considering contributing to this project. Below are guidelines to keep the codebase consistent and make reviews smooth.

## How to contribute

- **Report bugs** – Open an issue describing the problem, your environment (OS, Python version), and steps to reproduce.
- **Suggest features** – Open an issue with a clear description and use case.
- **Code changes** – Open an issue first to discuss, then submit a pull request (or push to a branch and share the link).

## Development setup

```bash
git clone <repo-url>
cd emotion_rt
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Ensure you have data in `data/train/` (one subfolder per emotion class) if you need to run training or evaluation.

## Code style

- **Python**: Follow PEP 8. Use a formatter (e.g. Black) if you like; the project does not enforce a specific line length.
- **Docstrings**: Use English. Document modules, public functions, and classes (purpose, main args, return value). Existing code uses this style.
- **Imports**: Group standard library, then third-party, then local (`src.*`). Prefer one import per line for readability when the list is long.

## Project structure

- **`src/data/`** – Dataset paths, sample indexing, train/val split, `FerDataset`.
- **`src/models/`** – Model definitions (CNN, ResNet); input is 48×48 grayscale, output is 7-class logits.
- **`src/utils/`** – Model loading, class names, single-image prediction.
- **Root scripts** – `train.py`, `eval.py`, `realtime_cam.py`; keep CLI and high-level logic here, detailed logic in `src/`.

When adding features (e.g. a new model or data source), try to keep this separation and reuse existing helpers (`load_model`, `get_class_names`, `FerDataset`, etc.).

## Pull request checklist

- Code runs (training/eval/cam as relevant) and does not break existing behavior.
- New or modified functions/classes have English docstrings.
- No unnecessary dependencies added; if you add one, document it in `requirements.txt` and in the PR.

## Questions

If something is unclear, open an issue and we can sort it out.
