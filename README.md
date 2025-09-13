# Inter-shot Motion Correction of Segmented 3D-GRASE ASL Perfusion Imaging with Self-Navigation and CAIPI

This repository contains a complete reference implementation of the proposed method in paper "Inter-shot Motion Correction of Segmented 3D-GRASE ASL Perfusion Imaging with Self-Navigation and CAIPI". It reconstructs per‑shot self‑navigators, estimates rigid motion, and integrates those estimates in a motion‑compensated SENSE forward model for final ASL image reconstruction.

## Installation
Recommended: use Conda for Python ≥3.11.7.

1) Create Python environment

```bash
conda env create -f requirements.yaml
conda activate mrtk
```

2) External toolboxes
- FSL (for `mcflirt`): install FSL and ensure `mcflirt` is on `PATH`.
  - Website: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/
  - Typical env vars: set `FSLDIR` and add `$FSLDIR/bin` to `PATH`.
- BART (for coil compression): install BART and point Python to its bindings.
  - Website: https://mrirecon.github.io/bart/
  - Set `BART_PATH` to the BART Python module folder (usually `<bart-root>/python`).

3) Environment via .env (dotenv)
- Create a file named `.env` at the repo root (already git‑ignored) and add:

```
# BART
BART_PATH=/absolute/path/to/bart/python

# FSL
FSLDIR=/absolute/path/to/fsl
FSLOUTPUTTYPE=NIFTI_GZ
PATH=$FSLDIR/bin:$PATH
```

## Data Preparation
Raw data are Siemens Twix (`.dat`) plus optional scanner NIfTI for header/spacing. You can run directly from paths via CLI (see below).

## Quick Start
Run the end‑to‑end pipeline directly with paths (argparse‑based `main.py`):

```bash
python main.py \
  --twix-path /path/to/data.dat \
  --output-root ./output/my_run \
  --scanner-nii /path/to/scanner.nii.gz \
  --caipi-style caipi1x4 \
  --config config/config.yaml
```

## Tests
Run unit tests:

```bash
pip install pytest
pytest -q
```

Notes
- Tests patch heavy dependencies where possible (SigPy calibration is mocked; BART/FSL not required).


<!-- ## Citation
Please cite the accompanying paper when using this code. Add full citation details here (authors, title, venue, year, DOI/URL).

```
@article{YOURKEY,
  title   = {Self‑navigated Inter‑shot Motion Correction for Segmented 3D‑GRASE ASL with CAIPI},
  author  = {…},
  journal = {…},
  year    = {…},
  doi     = {…}
}
``` -->

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements
- FSL (mcflirt), BART, SigPy, mapVBVD, Hydra/OmegaConf.
