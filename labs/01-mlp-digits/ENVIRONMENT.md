# Verified environment

This file records the environment used for the committed Lab 01 results. It is an environment snapshot, not a claim that a GPU is required.

## Verified run

| Item | Value |
| --- | --- |
| Operating system | Windows 11 |
| Python | 3.13.3 |
| PyTorch | 2.6.0+cu124 |
| CUDA build reported by PyTorch | 12.4 |
| scikit-learn | 1.7.2 |
| NumPy | 2.2.5 |
| Matplotlib | 3.10.1 |
| Execution device used by this script | CPU |

The local machine had CUDA available, but `src/run_experiments.py` creates CPU tensors and does not require CUDA to reproduce its outputs.

## Installing the verified direct dependencies

For the closest match to the verified environment, install the exact direct dependencies below. The regular `requirements.txt` deliberately uses compatible version ranges for a more portable setup.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements-verified.txt
python src/run_experiments.py --output results/reproduced
```

## Reproduction notes

- Results depend on the fixed dataset split and random seeds encoded in the script.
- Minor timing differences are expected across machines; compare accuracy and learning curves rather than wall-clock time alone.
- If an exact historical environment is needed later, create a platform-specific lock file rather than committing a virtual environment directory.
