# ASL Inter‑shot Motion Correction for Segmented 3D‑GRASE (CAIPI)

This repository contains a complete reference implementation of a retrospective, self‑navigated inter‑shot motion correction method for segmented 3D‑GRASE ASL imaging with CAIPI sampling. It reconstructs per‑shot self‑navigators, estimates rigid motion, and integrates those estimates in a motion‑compensated SENSE forward model for final ASL image reconstruction.

## Installation
Recommended: use Conda for Python ≥3.10.

1) Create python environment:

```bash
conda env create -f requirements.yaml
conda activate mrtk
```

2) External toolboxes:
- FSL (for `mcflirt`): install FSL and ensure `mcflirt` is on `PATH` (fslpy will call the binary). Typical envs set `FSLDIR` and add `$FSLDIR/bin` to `PATH`.
- BART (for coil compression): build/install BART and set the Python binding path: `export BART_PATH=/path/to/bart/python`.

Notes
- macOS/Linux are supported. On Windows, WSL is recommended for FSL/BART.
- GPU is not required. The POGM‑LLR implementation is NumPy‑based; SigPy can use GPUs if configured separately.

## Data Preparation
Raw data are expected as Siemens Twix (`.dat`) plus optional scanner NIfTI for header/spacing. Edit the dataset stubs to point to your files:
- `datasets/exp_20250523.py`: set `RAW_DATA_ROOT`, `TWIX_DIR`, `NIFTI_DIR`, and the per‑scan `recon_settings`.

Alternatively, bypass `datasets/` and directly override Hydra config values via CLI (see below).

## Quick Start
Minimal end‑to‑end example using the provided dataset description:

```bash
# Ensure dataset paths in datasets/exp_20250523.py are valid
python main_proj2.py
```

This runs:
- Preprocessing (`mrtk.prep`): load Twix, EPI phase correction, optional coil compression, save ASL HDF5.
- Image reconstruction (`img_recon`): default `cg_sense` to NIfTI (+ ASL perfusion subtraction products).

Outputs land under `output/<exp_name>/<scan_name>/...`, for example:
- `${output_root}/prep_twix/asl.h5` – preprocessed ASL k‑space and metadata.
- `${output_root}/img_recon_cg_sense/img.nii.gz` – reconstructed magnitude ASL time series.
- `${output_root}/img_recon_pogm_llr/...` – if you switch `recon_method` to `pogm_llr`.

### Motion‑Correction Pipelines
Switch to motion pipelines in `main_proj2.py` (uncomment as needed) or call from your own script:

- `pipeline_nav_moco_proj1(cfg)`:
  1) Preprocess Twix → ASL HDF5
  2) Self‑navigator recon (per‑shot)
  3) Motion estimation via FSL `mcflirt`
  4) Motion‑compensated SENSE (`mc_sense`) → `${output_root}/.../moco_mc_sense/img_moco.nii.gz`

- `pipeline_joint_moco_proj1(cfg)`:
  1) Preprocess + baseline recon
  2) Prepare inter‑shot joint estimation inputs for external solver (Matlab alignedSENSE‑style)
  3) Combine inter‑shot + inter‑volume motions
  4) Final mcSENSE reconstruction

Tip: use Hydra overrides to change parameters without editing YAML, for example:

```bash
python main_proj2.py base_info.n_rep=6 base_info.samp_type=GRASE3DCAIPI \
  base_info.Ry=2 base_info.Rz=8 base_info.Dz=4 img_recon.recon_method=cg_sense
```

## Configuration Highlights (`config/config_proj2_nav.yaml`)
- `base_info.*`: matrix size, FOV, CAIPI factors (`Ry`, `Rz`, `Dz`), number of segments/reps, tag‑first, subtraction mode.
- `prep_twix.*`: Twix path, output dir, EPI phase correction mode, coil compression, temporary files.
- `img_recon.*`: choose `rss_ifft`, `cg_sense`, `cg_sense_scipy`, `wavelet`, or `pogm_llr`; method‑specific parameters.
- `nav_recon.*`: per‑shot recon for self‑navigation.
- `motion_estimation.*`: mcFLIRT settings and output paths.
- `motion_correction.*`: mcSENSE options (interpolation method, iterations, tolerance).
- `joint_motion_correction.*`: alignedSENSE‑style inter‑shot estimation + inter‑volume registration + combined mcSENSE.

## Reusing the Library
You can use the modules directly without Hydra. Example (mcSENSE on your arrays):

```python
from mrtk.recon.mr3d import ASL
from mrtk.math.rigid_transform import RigidTransformList

asl = ASL.load("/path/to/asl.h5")
rigid = RigidTransformList.load_from_npy("/path/to/motion_par.npy")
res = asl.recon_img(recon_method='mc_sense', rigid_transforms=rigid,
                    mc_interpolation_method='sinc', mc_maxiter=500, mc_atol=1e-6)
img = res['recon']  # (Nx, Ny, Nz, Nt=1) motion‑corrected image
```

## Tests
Run unit tests where available:

```bash
pip install pytest
pytest -q
```

## Troubleshooting
- mapVBVD cannot read Twix: confirm the `.dat` file is valid and not truncated; try `mapvbvd` in a Python REPL.
- BART not found: set `export BART_PATH=/path/to/bart/python` so that `import bart` works.
- FSL `mcflirt` not found: ensure FSL is installed and `$FSLDIR/bin` is on `PATH`.
- Large memory/time: mcSENSE and POGM‑LLR are compute‑intensive; consider reducing matrix size, reps, or iterations.

## Citation
Please cite the accompanying paper when using this code. Add full citation details here (authors, title, venue, year, DOI/URL).

```
@article{YOURKEY,
  title   = {Self‑navigated Inter‑shot Motion Correction for Segmented 3D‑GRASE ASL with CAIPI},
  author  = {…},
  journal = {…},
  year    = {…},
  doi     = {…}
}
```

## License
Add your chosen open‑source license (e.g., MIT, BSD‑3‑Clause, Apache‑2.0). Include a `LICENSE` file at the repository root.

## Acknowledgements
- FSL (mcflirt), BART, SigPy, mapVBVD, Hydra/OmegaConf.

---

## What I Still Need From You
- Paper details for the Citation block (authors, venue, year, DOI/URL).
- License choice and a `LICENSE` file.
- Example dataset or a public sample link, or anonymized Twix+NIfTI for quick testing.
- Verified dependency versions and OS that you used (Python version; FSL version; BART commit; SigPy, fslpy, mapvbvd versions).
- Default config presets for multiple experiments (you can add more files under `datasets/` or `config/`).
- Whether to keep `pipeline_nav_recon_dl` (expects `nav.nii.gz`/`ref.nii.gz` placed manually) or remove/simplify it for open‑source users.
- Any trained models if the DL navigator path should be public; otherwise hide or document how to generate them.
- Project name you want to use on GitHub and short tagline.

