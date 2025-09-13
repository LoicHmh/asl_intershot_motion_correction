import os
from mrtk.utils.timer import clock
from omegaconf import DictConfig
from .steps import *


def pipeline_img_recon(cfg: DictConfig, always_run=True) -> None:
    with clock('pipeline_img_recon'):
        run_preprocess_asl_tgse_from_twix_data(cfg, always_run=False)
        run_img_recon(cfg.img_recon, always_run=always_run)


def pipeline_nav_recon(cfg: DictConfig, always_run=True) -> None:
    with clock('pipeline_nav_recon'):
        run_preprocess_asl_tgse_from_twix_data(cfg, always_run=False)
        run_img_recon(cfg.img_recon, always_run=False)
        run_nav_recon(cfg.nav_recon, always_run=always_run)


def pipeline_nav_moco(cfg: DictConfig, always_run=True) -> None:
    with clock('pipeline_nav_moco'):
        run_preprocess_asl_tgse_from_twix_data(cfg, always_run=False)
        run_img_recon(cfg.img_recon, always_run=False)
        run_nav_recon(cfg.nav_recon, always_run=False)
        run_motion_estimation_mcFLIRT(cfg.motion_estimation, always_run=always_run)
        run_motion_compensated_SENSE(cfg.motion_correction, always_run=always_run)


def delete_dir(dir_path):
    if os.path.exists(dir_path):
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f'Error: {e}')
        os.removedirs(dir_path)

def pipeline_archieve(cfg: DictConfig) -> str:

    ## TODO
    return 'Successful'
