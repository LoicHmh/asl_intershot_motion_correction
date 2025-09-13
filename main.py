import argparse
import os
import sys
import traceback

from omegaconf import OmegaConf
from mrtk.pipelines.pipelines import pipeline_nav_moco


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ASL 3D-GRASE reconstruction and inter-shot motion correction."
    )
    parser.add_argument("--twix-path", required=True, help="Path to Siemens twix .dat file")
    parser.add_argument("--output-root", required=True, help="Output root directory for this run")
    parser.add_argument("--scanner-nii", required=True, help="Path to reference scanner NIfTI (.nii/.nii.gz)")
    parser.add_argument(
        "--caipi-style",
        default="caipi1x4",
        help=(
            "CAIPI shift style. Accepts 'caipi1x4', 'caipi2x2'"
        ),
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to the base OmegaConf YAML config",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    try:
        cfg = OmegaConf.load(args.config)
    except Exception as e:
        print(f"Failed to load config from {args.config}: {e}")
        sys.exit(1)

    # Populate required base_info from CLI
    cfg.base_info.twix_path = args.twix_path
    cfg.base_info.output_root = args.output_root
    cfg.base_info.scanner_nii = args.scanner_nii
    cfg.base_info.samp_type = "GRASE3DCAIPI"

    if args.caipi_style == "caipi1x4":
        cfg.base_info.turbo_factor = 8
        cfg.base_info.epi_factor = 64
        cfg.base_info.Rz = 4
        cfg.base_info.Ry = 1
        cfg.base_info.Dz = 2

    elif args.caipi_style == "caipi2x2":
        cfg.base_info.turbo_factor = 16
        cfg.base_info.epi_factor = 32
        cfg.base_info.Rz = 2
        cfg.base_info.Ry = 2
        cfg.base_info.Dz = 1
    else:
        print(f"Unrecognized CAIPI style {args.caipi_style}. Supported: 'caipi1x4', 'caipi2x2'")
        sys.exit(1)

    try:
        pipeline_nav_moco(cfg=cfg, always_run=True)
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        sys.exit(2)


if __name__ == "__main__":
    main()
