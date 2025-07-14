import os
import traceback
import hydra
from omegaconf import DictConfig
from mrtk.pipelines.pipelines import pipeline_img_recon, pipeline_nav_moco_proj1, pipeline_joint_moco_proj1, pipeline_sim_motion_proj1
# from datasets.exp_20250328 import exp
# from datasets.exp_20250409 import exp
# from datasets.exp_20250423 import exp
from datasets.exp_20250523 import exp
# from datasets.exp_20241024 import exp

@hydra.main(version_base=None, config_path="../config", config_name="config_proj2_nav")
# @hydra.main(version_base=None, config_path="../config", config_name="config_proj2_joint")
def main(cfg: DictConfig):
    # OUTPUT_ROOT = os.path.join('output', exp.exp_name + '_test_for_proj2')
    OUTPUT_ROOT = os.path.join('output', exp.exp_name + '_halfres')

    alias_list = [
        # 'Flex4I_0ms',
        # 'Flex4I_100ms',
        # 'FlexDI_100ms',
        # # 'M0',
        # # 'M0_WE',
        # 'Flex4I_100ms_we121',
        # 'Flex4I_100ms_we11',
        # 'Flex4I_100ms_we11_fast',
        # 'Flex4I_100ms_we121_90',
        # 'Flex4I_100ms_we1331',
        # 'Flex4I_100ms_we121_180_ovs_off',
        # 'Flex4I_100ms_we121_180_ovs_0',
        # 'Flex4I_100ms_we121_CAIPI441',
        # 'Flex4I_100ms_None',
        # 'Flex4I_100ms_FatSat',

        # 'Flex4I_100ms_None',
        # 'Flex4I_100ms_FatSat',
        # 'Flex4I_100ms_WE121',
        # 'Flex4I_100ms_WE11',
        # 'Flex4I_100ms_WE1331',
        # 'Flex4I_0ms_None',
        # 'Flex4I_0ms_FatSat',
        # 'Flex4I_0ms_WE121',
        # 'Flex4I_0ms_WE11',
        # 'Flex4I_0ms_WE1331',
        # 'M0_WE121_high_res',
        # 'Flex4I_0ms_WE121_high_res',
        # 'Flex4I_100ms_WE121_high_res',
        # 'Flex4I_0ms_WE121_high_res_mov',
        'Flex4I_100ms_WE121_high_res_mov',
        # 'TRAJ-5TC',
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

            # pipeline_img_recon(cfg=cfg)
            # msg.append(pipeline_recon_per_shot(cfg=cfg))
            # pipeline_nav_moco_proj1(cfg=cfg)
            # msg.append(pipeline_sim_motion_proj1(cfg=cfg))
            msg.append(pipeline_joint_moco_proj1(cfg=cfg, always_run=True))
            # msg.append(exp_proj2_fullysampled_recon_method(cfg=cfg))
            # msg.append(exp_proj2_self_navigator_recon_method(cfg=cfg))
            # msg.append(exp_proj2_joint_moco(cfg=cfg))
            # pipeline_archieve(cfg)
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            
    
    print(msg)

if __name__ == '__main__':

    main()