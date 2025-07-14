from .datacfg import Experiment, RawData
# from datacfg import Experiment, RawData

RAW_DATA_ROOT = '/Users/minhao/Projects/mr-recon/rawdata/'
TWIX_DIR = 'Raw_data_2025-05-23'
NIFTI_DIR = 'F3T_2024_025_038'

N_REP = 6 # Number of repetitions (pairs * 2)

exp = Experiment(
    subject='MMMM', 
    magnetic_field='3T', 
    date='20250523', 
    data_root=RAW_DATA_ROOT,
    exp_name='invivo_test_20250523',
    description=f"High resolution (2mm, matrix size: 64x64x32) VEPCASL with CAIPI Ry=8, Rz=2, Dz=4, 3 pairs. Different BGS (Flex4I 100ms, Flex4I 0ms, FlexDI 100ms). TA:10min40s. T/C inner loop: True. No intended motion. We used WE121(water excitation) this time instead of fat saturation. We also acquired MO images with WE121 and with Fat Sat"
)

exp.add_raw_data(
    RawData(
        alias='Flex4I_100ms_WE121_high_res_mov',
        protocol='mh_jw_tgse_VEPCASL', 
        twix_path=f'{TWIX_DIR}/meas_MID00074_FID61408_mh_jw_tgse_VEPCASL_we_4I_100ms_mov.dat', 
        online_recon_nii_path=f'{NIFTI_DIR}/images_015_mh_jw_tgse_VEPCASL_we_4I_100ms_mov.nii', 
        data_dir=exp.data_dir,
        recon_settings={
            'matrix_size': [114, 114, 64],
            'fov': [228, 228, 128],
            'turbo_factor': 8,
            'epi_factor': 57,
            'n_segments': 16,
            'n_rep': N_REP,
            'CAIPI': {
                'R': 16,
                'Rz': 8,
                'Ry': 2,
                'z-shift':4,
            },
            'Inner T/C loop': True,
        }
    )
)



if __name__ == '__main__':

    exp.check_integrity()
    exp.check_names()
    print(exp)