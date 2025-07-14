import numpy as np
import os
import scipy
import sigpy.mri
from mrtk.recon.mr3d import MR3D
from mrtk.recon.samp_mask import SampMask
from fsl.data.image import Image
from pathlib import Path
from mrtk.prep.epi_phase_correction import epi_phase_correction_3d
from mrtk.prep.coil_compression import coil_compression
from mrtk.prep.inter_shot_phase_shift_correction import inter_shot_phase_shift_correction
from abc import ABC, abstractmethod

from typing import Literal, Optional, Tuple

class EPI3D(ABC):  
    def __init__(self, 
                 twix_path: str | Path,
                 output_dir: str | Path,
                 flag_epi_phase_correction: bool = True,
                 mode_epi_phase_correction: Literal['avg-volume', 'single-volume'] = 'avg-volume',
                 flag_coil_compression: bool = True,
                 n_virtural_coil: int = 16,
                 flag_external_mtx_cc: bool = False,
                 external_mtx_cc_path: Optional[str | Path] = None,
                 flag_espirit: bool = True,
                 espirit_mode: Literal['avg', 'first', 'external'] = 'avg',
                 espirit_crop: int = 0,
                 espirit_calib_width: int = 16,
                 espirit_thresh: float = 0.02,
                 espirit_max_iter: int = 100,
                 espirit_kernel_width: int = 6,
                 external_espirit_path: Optional[str | Path] = None,
                 flag_save_espirit_coil_map: bool = False,
                 flag_clean_temp_files: bool = True,
                 flag_correct_inter_shot_phase_shift: bool = True,
                 flag_save_temp_files: bool = True,
                 partial_fourier: Tuple[int] = (1, 1),
                 **kwargs) -> None:
        
        self.flag_epi_phase_correction = flag_epi_phase_correction
        self.mode_epi_phase_correction = mode_epi_phase_correction
        self.flag_coil_compression = flag_coil_compression
        self.n_virtural_coil = n_virtural_coil
        self.flag_external_mtx_cc = flag_external_mtx_cc
        self.external_mtx_cc_path = external_mtx_cc_path
        self.flag_espirit = flag_espirit
        self.espirit_mode = espirit_mode
        self.espirit_crop = espirit_crop
        self.espirit_calib_width = espirit_calib_width
        self.espirit_thresh = espirit_thresh
        self.espirit_max_iter = espirit_max_iter
        self.espirit_kernel_width = espirit_kernel_width
        self.external_espirit_path = external_espirit_path
        self.flag_save_espirit_coil_map = flag_save_espirit_coil_map
        self.flag_clean_temp_files = flag_clean_temp_files
        self.flag_correct_inter_shot_phase_shift = flag_correct_inter_shot_phase_shift
        self.flag_save_temp_files = flag_save_temp_files
        self.partial_fourier = partial_fourier
        
        self.twix_path = Path(twix_path)
        self.out_dir = Path(output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.ksp_ori = None
        self.par_order = None
        self.ksp_pc = None
        self.ksp_cc = None
        self.sensitivity_maps = None
        
        # Initialize

        self.ksp_ori, self.par_order = self.load_twix()
        self.ksp_ori = self.partial_fourier_padding(method='zero_padding')

        self.ksp_pc = self.epi_phase_correction()

        if self.flag_correct_inter_shot_phase_shift:
            self.ksp_pc = inter_shot_phase_shift_correction(self.ksp_pc, samp_mask=self.samp_mask, t=1)

        self.ksp_cc = self.coil_compression()
        
        # if self.flag_espirit:
        #     self.sensitivity_maps = self.generate_espirit_sensitivity_maps()

    @abstractmethod
    def load_twix(self):
        pass

    @abstractmethod
    def load_twix_pc(self):
        pass

    def partial_fourier_padding(self, method='zero_padding'):

        if any(pf < 1 for pf in self.partial_fourier):
            shape_ori = np.asarray(self.ksp_ori.shape[2: 4])
            shape_pad = (shape_ori / np.asarray(self.partial_fourier)).astype(int)

            if method == 'zero_padding':
                ksp_ori = np.pad(self.ksp_ori, 
                            ((0, 0), (0, 0), 
                                (0, shape_pad[0] - shape_ori[0]), 
                                (0, shape_pad[1] - shape_ori[1]), 
                                (0, 0), (0, 0)), 
                            'constant')
            else:
                raise NotImplementedError
        
            return ksp_ori
        else:
            return self.ksp_ori

    def epi_phase_correction(self, point_to_point=True):
        if self.flag_epi_phase_correction:
            ckp_path = self.out_dir / 'ksp_pc.mat'
            if ckp_path.exists():
                print(f'Loading ksp_pc from {ckp_path}')
                checkpoint = scipy.io.loadmat(ckp_path)
                ksp_pc = checkpoint['ksp_pc']
            else:
                pc = self.load_twix_pc()
                ksp_pc = epi_phase_correction_3d(ksp=self.ksp_ori, pc=pc, point_to_point=point_to_point, mode=self.mode_epi_phase_correction)
                if self.flag_save_temp_files:
                    scipy.io.savemat(ckp_path, {'ksp_pc': ksp_pc})
                del pc
        else:
            print(f'No phase correction')
            ksp_pc = np.sum(self.ksp_ori, axis=-1)
        return ksp_pc

    def coil_compression(self):
        if self.flag_coil_compression:
            ckp_path = self.out_dir / 'ksp_cc.mat'
            if ckp_path.exists():
                print(f'Loading ksp_cc from {ckp_path}')
                checkpoint = scipy.io.loadmat(ckp_path)
                ksp_cc = checkpoint['ksp_cc']
            else:
                print(f'Calculating ksp_cc using BART...')

                if self.flag_external_mtx_cc:
                    external_mtx_cc = scipy.io.loadmat(self.external_mtx_cc_path)['mtx_cc']
                else:
                    external_mtx_cc = None

                ksp_cc, mtx_cc = coil_compression(self.ksp_pc, ncc=self.n_virtural_coil, external_mtx_cc=external_mtx_cc)
                if self.flag_save_temp_files:
                    scipy.io.savemat(ckp_path, {
                        'mtx_cc': mtx_cc,
                        'ksp_cc': ksp_cc,
                    })
        else:
            print(f'No coil compression')
            ksp_cc = self.ksp_pc.transpose((1, 0, 2, 3, 4))
        return ksp_cc

    # def generate_espirit_sensitivity_maps(self):
    #     ckp_path = self.out_dir / 'espirit.mat'

    #     if ckp_path.exists():
    #         print(f'Loading espirit sens map from {ckp_path}')
    #         checkpoint = scipy.io.loadmat(ckp_path)
    #         espirit_sensitivity_maps = checkpoint['espirit']
    #     else:
    #         if self.espirit_mode == 'external':
    #             checkpoint = scipy.io.loadmat(self.external_espirit_path)
    #             espirit_sensitivity_maps = checkpoint['espirit']
    #         else:
    #             if self.espirit_mode == 'avg':
    #                 ksp_espirit = np.mean(self.ksp_cc, axis=-1)
    #             elif self.espirit_mode == 'first':
    #                 ksp_espirit = self.ksp_cc[..., 0]
    #             else:
    #                 raise ValueError(f'Invalid mode_espirit: {self.mode_espirit}')

    #             espirit_sensitivity_maps = sigpy.mri.app.EspiritCalib(
    #                 ksp_espirit,
    #                 calib_width=self.espirit_calib_width,
    #                 thresh=self.espirit_thresh,
    #                 kernel_width=self.espirit_kernel_width,
    #                 crop=self.espirit_crop,
    #                 max_iter=self.espirit_max_iter,
    #                 show_pbar=True).run()

    #             if self.flag_save_temp_files:
    #                 scipy.io.savemat(ckp_path, {'espirit': espirit_sensitivity_maps})
    #     return espirit_sensitivity_maps

    def remove_temp_files(self):

        try:
            print('Deleting...')
            output_sub_dir = os.path.join(self.out_dir, 'preprocessing')
            temp_files = [
                'ksp_ori.mat',
                'ksp_pc.mat',
                'ksp_cc.mat',
            ]
            
            for file in temp_files:
                file_path = os.path.join(output_sub_dir, file)
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            if os.path.exists(output_sub_dir):
                os.removedirs(output_sub_dir)
                
        except Exception as e:
            print(f'Error when deleting temp files: {str(e)}') 