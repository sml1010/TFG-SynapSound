![cabeceraSalud](https://github.com/user-attachments/assets/d272dd39-1d24-477c-b57e-08d1b0b581b5)

Alumna: Sara Meda López 
2025-2026

Instituto de Bioingeniería UMH
- Eduardo Fernández Jover
- Antonio Lozano
- David García García

# neurofeedback 
New topic

# neural-sonification

Minimal analysis repo for Blackrock recordings (`.nev`, `.ns2`, `.ns5`) using `brpylib`.

## Install

```powershell
uv venv
uv sync
```

## Configure

Edit `config/config.yaml`:
- `paths.data_root`
- `paths.recording_dir`
- `defaults.recording_name`

Set `defaults.window_s` to `null` for full recordings, or a small value like `2.0` for quick checks.

## Confidential data

- Put local recording files inside `data/`.
- Do not commit anything from `data/`.
- `results/` is for generated outputs and is not committed.

## Run

Debug check:
```powershell
uv run python src/debug_neural_data.py --window-s 2.0 --read-waveforms
```

Interactive walkthrough (`# %%` cells):
```powershell
uv run python -m ipykernel install --user --name neural-sonification --display-name "Python (neural-sonification)"
```
Open and run `src/neural_data_walkthrough.py` in VS Code/Jupyter.

The walkthrough script writes plots/videos automatically under `results/` after you run it.

More details: `NEURAL_DATA.md`.

