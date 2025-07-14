from omegaconf import DictConfig
from .steps import *

def exp_img_recon_pogm_llr_params(cfg: DictConfig) -> None:
    run_preprocess_asl_tgse_from_twix_data(cfg, always_run=False)     

    output_dir_ori = cfg.img_recon.output_dir
    for pogm_llr_lambda in [
        # 1e-3,
        # 5e-4,
        # 1e-4,
        # 12e-5,
        # 1e-4,
        # 8e-5,
        5e-5,
    ]:
        for pogm_llr_patch_size in [
            [7, 7, 7],
            # [11, 11, 11],
            # [15, 15, 15],
            # [19, 19, 19],
            # [23, 23, 23],
        ]:
            for pogm_llr_sub_before_recon, pogm_llr_separate_tc in [
                # (True, True),
                (False, False),
                # (False, True),
            ]:
                cfg.img_recon.pogm_llr_lambda = pogm_llr_lambda
                cfg.img_recon.pogm_llr_patch_size = pogm_llr_patch_size
                cfg.img_recon.pogm_llr_sub_before_recon = pogm_llr_sub_before_recon
                cfg.img_recon.pogm_llr_separate_tc = pogm_llr_separate_tc
                cfg.img_recon.output_dir = output_dir_ori + f'_lambda_{pogm_llr_lambda:.0e}_patchsize_{pogm_llr_patch_size[0]}'

                print(f'pogm_llr_lambda: {cfg.img_recon.pogm_llr_lambda}, pogm_llr_patch_size: {cfg.img_recon.pogm_llr_patch_size}, save_intermediate: {cfg.img_recon.pogm_llr_save_intermediate}')
                run_img_recon(cfg.img_recon, always_run=True)


def exp_img_recon_methods(cfg: DictConfig) -> None:
    run_preprocess_asl_tgse_from_twix_data(cfg, always_run=False)     

    for recon_method in [
        'cg_sense',
        'rss_ifft',
        'wavelet',
        'pogm_llr'
    ]:
        cfg.img_recon.recon_method = recon_method
        run_img_recon(cfg.img_recon, always_run=True)