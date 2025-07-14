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


def pipeline_nav_moco_proj1(cfg: DictConfig, always_run=True) -> None:
    with clock('pipeline_nav_moco_proj1'):
        run_preprocess_asl_tgse_from_twix_data(cfg, always_run=False)     
        run_img_recon(cfg.img_recon, always_run=False)
        # run_nav_recon(cfg.nav_recon, always_run=False)
        run_nav_recon_dl(cfg.nav_recon, always_run=always_run)
        run_motion_estimation_mcFLIRT_dl(cfg.motion_estimation, always_run=always_run)
        # run_motion_estimation_mcFLIRT(cfg.motion_estimation, always_run=always_run)
        run_motion_compensated_SENSE(cfg.motion_correction, always_run=always_run)


def pipeline_sim_motion_proj1(cfg: DictConfig, always_run=True) -> None:
    with clock('pipeline_sim_motion_proj1'):
        run_preprocess_asl_tgse_from_twix_data(cfg, always_run=False)     
        cfg.base_info.n_rep = 2
        asl_mot_path = run_generate_simulated_motion_data(cfg.img_recon, always_run=always_run)
        cfg.base_info.output_root = Path(cfg.base_info.output_root) / 'sim_motion'
        cfg.prep_twix.asl_path = asl_mot_path

        run_nav_recon(cfg.nav_recon, always_run=False)
        run_motion_estimation_mcFLIRT(cfg.motion_estimation, always_run=False)
        run_motion_compensated_SENSE(cfg.motion_correction, always_run=False)

        cfg_joint = cfg.joint_motion_correction
        cfg_moco = cfg_joint.motion_correction_inter_shot

        run_prep_mcSENSE_solve_T_x(cfg=cfg_moco, always_run=False)

        if not (Path(cfg_moco.temp_dir) / 'output_rep1.mat').exists():
            print('Use matlab to solve T and x')
            return os.path.abspath(cfg_moco.output_dir)
            
        run_combine_motion_estimation(cfg=cfg_joint, always_run=True)
        run_motion_compensated_SENSE(cfg_joint.motion_correction_combined, always_run=True)



def pipeline_joint_moco_proj1(cfg: DictConfig, always_run=True) -> None:
    with clock('pipeline_joint_moco_proj1'):
        run_preprocess_asl_tgse_from_twix_data(cfg, always_run=False)
        run_img_recon(cfg.img_recon, always_run=False)

        cfg_joint = cfg.joint_motion_correction
        cfg_moco = cfg_joint.motion_correction_inter_shot

        run_prep_mcSENSE_solve_T_x(cfg=cfg_joint, always_run=True)

        if not (Path(cfg_moco.temp_dir) / 'output_rep1.mat').exists():
            print('Use matlab to solve T and x')
            return os.path.abspath(cfg_moco.output_dir)
            
        run_combine_motion_estimation(cfg=cfg_joint, always_run=always_run)
        run_motion_compensated_SENSE(cfg_joint.motion_correction_combined, always_run=always_run)


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
