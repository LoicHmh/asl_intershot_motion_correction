import os
import traceback
import hydra
from omegaconf import DictConfig
from mrtk.pipelines.pipelines import pipeline_joint_moco_proj1, pipeline_img_recon
from datasets.exp_20250523 import exp

@hydra.main(version_base=None, config_path="./config", config_name="config_proj2_nav")
def main(cfg: DictConfig):
    OUTPUT_ROOT = os.path.join('output', exp.exp_name)

    alias_list = [
        'Flex4I_100ms_WE121_high_res_mov',
    ]

    msg = []

    for alias in alias_list:
        try:
            rawdata = exp(alias)
            print(rawdata)

            cfg.base_info.twix_path = rawdata.twix_path
            cfg.base_info.output_root = os.path.join(OUTPUT_ROOT, rawdata.name)
            cfg.base_info.scanner_nii = rawdata.online_recon_nii_path
            cfg.base_info.n_rep = rawdata.recon_settings.get('n_rep')
            cfg.base_info.matrix_size = rawdata.recon_settings.get('matrix_size')
            cfg.base_info.fov = rawdata.recon_settings.get('fov')
            cfg.base_info.turbo_factor = rawdata.recon_settings.get('turbo_factor')
            cfg.base_info.epi_factor = rawdata.recon_settings.get('epi_factor')
            cfg.base_info.n_segments = rawdata.recon_settings.get('n_segments')

            cfg.base_info.samp_type = 'GRASE3DCAIPI'
            cfg.base_info.R_factor = rawdata.recon_settings['CAIPI']['Ry'] * rawdata.recon_settings['CAIPI']['Rz']
            cfg.base_info.Rz = rawdata.recon_settings['CAIPI']['Rz']
            cfg.base_info.Ry = rawdata.recon_settings['CAIPI']['Ry']
            cfg.base_info.Dz = rawdata.recon_settings['CAIPI']['z-shift']
            cfg.base_info.flag_echo_shift = rawdata.recon_settings.get('Inner T/C loop') 

            pipeline_img_recon(cfg=cfg, always_run=True)
            # msg.append(pipeline_joint_moco_proj1(cfg=cfg, always_run=True))
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            
    print(msg)

if __name__ == '__main__':

    main()