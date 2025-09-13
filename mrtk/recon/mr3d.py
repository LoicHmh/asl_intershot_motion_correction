import numpy as np
import numpy.typing as npt

import sigpy.mri
from fsl.data.image import Image

from ..utils.io import save_data, load_data, save_nii, save_asl

from pathlib import Path
from typing import Literal, Optional
from .mrop import SenseOP, MotionCompensatedSenseOP
import sigpy.mri

from multiprocessing import Pool


def _recon_img_wrapper(args):
    return _recon_img(**args)

def _recon_img(ksp: npt.NDArray[np.complex128], 
                sampling_mask: np.ndarray,
                sensitivity_maps: npt.NDArray[np.complex128],
                recon_method: Literal['rss_ifft', 'cg_sense', 'cg_sense_scipy', 'wavelet', 'ksp_check'], 
                **kwargs) -> dict[str, np.ndarray]:
    binary_sampling_mask = sampling_mask > 0
    if recon_method == 'rss_ifft':
        print(f'Reconstructing image using RSS IFFT...')
        sense_op = SenseOP(sampling_mask=binary_sampling_mask,
                            sensitivity_maps=sensitivity_maps,
                            ksp_size=ksp.shape)
        img = sense_op.run_rss_ifft(ksp)
        res = {'recon': img}

    elif recon_method == 'cg_sense':
        lambda_l2 = kwargs.get('lambda_l2', 0.01)
        print(f'Reconstructing image using CG-SENSE with lambda_2 {lambda_l2}...')
        img = np.zeros(ksp.shape[1:5], dtype=np.complex128)
        Nt = ksp.shape[4]
        for t in range(Nt):
            img[..., t] = sigpy.mri.app.SenseRecon(
                y=ksp[..., t] * binary_sampling_mask[..., t],
                mps=sensitivity_maps[..., 0],
                lamda=lambda_l2,
                show_pbar=False).run()
        res = {'recon': img}
        
    elif recon_method == 'cg_sense_scipy':
        lambda_l2 = kwargs.get('lambda_l2', 0.01)
        maxiter = kwargs.get('max_iter', 100)
        atol = kwargs.get('atol', 1e-6)
        print(f'Reconstructing image using CG-SENSE with lambda_2 {lambda_l2}, max_iter {maxiter} and atol {atol}...')
        sense_op = SenseOP(sampling_mask=binary_sampling_mask,
                            sensitivity_maps=sensitivity_maps,
                            ksp_size=ksp.shape)
        img = sense_op.run_cg_sense_l2(ksp, lambda_l2, maxiter, atol)
        res = {'recon': img}

    elif recon_method == 'ksp_check':
        print(f'Checking k-space data...')
        img = ksp * binary_sampling_mask
        res = {'recon': img}

    elif recon_method == 'wavelet':
        lambda_l1_wavelet = kwargs.get('lambda_l1_wavelet', 1e-6)
        print(f'Reconstructing image using L1-Wavelet with lambda_l1_wavelet {lambda_l1_wavelet}...')
        img = np.zeros(ksp.shape[1:5], dtype=np.complex128)
        Nt = ksp.shape[4]
        for t in range(Nt):
            img[..., t] = sigpy.mri.app.L1WaveletRecon(
                y=ksp[..., t] * binary_sampling_mask[..., t],
                mps=sensitivity_maps[..., 0],
                lamda=lambda_l1_wavelet,
                show_pbar=False).run()
        res = {'recon': img}
    
    elif recon_method == 'mc_sense':
        rigid_transforms = kwargs.get('rigid_transforms', None)
        mc_interpolation_method: str = kwargs.get('mc_interpolation_method', 'sinc')
        mc_maxiter: int = kwargs.get('mc_maxiter', 100)
        mc_atol: float = kwargs.get('mc_atol', 1e-3)

        n_rep = kwargs.get('n_rep', ksp.shape[-1])
        n_segments = kwargs.get('n_segments', sampling_mask.max())
        img = np.zeros(ksp.shape[1:], dtype=ksp.dtype)

        multiprocessing = kwargs.get('multiprocessing', 0)
        binary_sampling_mask = np.zeros((*sampling_mask.shape[:4], n_segments), dtype=np.uint8)
        for i_shot in range(n_segments):
            binary_sampling_mask[sampling_mask[..., 0] == (i_shot + 1), i_shot] = 1

        def gen_args():
            for i_rep in range(n_rep):
                yield {
                    'sampling_mask': binary_sampling_mask,
                    'sensitivity_maps': sensitivity_maps,
                    'ksp_size': [*ksp.shape[:4], n_segments],
                    'rigid_transforms': rigid_transforms[i_rep * n_segments: (i_rep + 1) * n_segments],
                    'interpolation_method': mc_interpolation_method,
                    'ksp': ksp[..., i_rep],
                    'mc_maxiter': mc_maxiter,
                    'mc_atol': mc_atol,
                }
        if multiprocessing == 0:
            i_rep = 0
            for args in gen_args():
                img[..., i_rep: i_rep + 1] = mc_sense_wrapper(args)
                i_rep += 1
        else:
            pool_results = []
            with Pool(processes=multiprocessing) as pool:
                for pool_res in pool.imap(mc_sense_wrapper, gen_args(), chunksize=1):
                    pool_results.append(pool_res)
            for i_rep, pool_res in enumerate(pool_results):
                img[..., i_rep: i_rep + 1] = pool_res
        res = {'recon': img}
    else:
        raise NotImplementedError(f"Unsupported reconstruction method: {recon_method}, only accept 'rss_ifft', 'cg_sense', 'cg_sense_scipy' and 'wavelet'")

    return res


def mc_sense_wrapper(args):
    mc_sense_op = MotionCompensatedSenseOP(
        sampling_mask=args['sampling_mask'],
        sensitivity_maps=args['sensitivity_maps'],
        ksp_size=args['ksp_size'],
        rigid_transforms=args['rigid_transforms'],
        interpolation_method=args['interpolation_method'],
    )
    return mc_sense_op.solve_x(args['ksp'], maxiter=args['mc_maxiter'], atol=args['mc_atol'])

class MR3D():
    """MR3D class for Multi-coil single-shot 3D Cartesian MRI reconstruction.
    
    Attributes:
        ksp (NDArray[complex128]): k-space data with shape (N_coil, N_readout, N_phase_encoding, N_partition)
        sampling_mask (NDArray): Sampling mask with shape (1, 1, N_phase_encoding, N_partition)
        sensitivity_maps (NDArray): Coil sensitivity maps with shape (N_coil, N_readout, N_phase_encoding, N_partition)
        scanner_nii (str): Path to scanner NIFTI file
    """
    def __init__(self, 
                 ksp: npt.NDArray[np.complex128],
                 sampling_mask: npt.NDArray | None = None,
                 sensitivity_maps_method: Literal['espirit', 'adaptive', 'external', 'one'] = 'external',
                 sensitivity_maps: npt.NDArray | None = None,
                 scanner_nii: Path | None = None,
                 flip: Optional[tuple[int]] = None,
                 roll: tuple[int, int, int] = (0, -1, -1),
                 **kwargs,
                 ) -> None:
        """Initialize MR3D instance.
        
        Args:
            ksp: k-space data
            sampling_mask: Sampling mask, defaults to None (all ones) but must be binary mask
            sensitivity_maps: Coil sensitivity maps, defaults to None
            scanner_nii: Path to scanner NIFTI file, defaults to None
        """
        # Validate input dimensions
        if ksp.ndim != 5:
            raise ValueError("k-space data must be 5-dimensional")
            
        self.ksp = ksp
        self.Nc, self.Nx, self.Ny, self.Nz, self.Nt = self.ksp.shape

        self.ksp_shape = (self.Nc, self.Nx, self.Ny, self.Nz, self.Nt)
        self.img_shape = (self.Nx, self.Ny, self.Nz, self.Nt)
        self.sensitivity_maps_shape = (self.Nc, self.Nx, self.Ny, self.Nz, 1)
        self.sampling_mask_shape = (1, self.Nx, self.Ny, self.Nz, self.Nt)

        self.flip = flip
        self.roll = roll        

        # Initialize sampling mask
        if sampling_mask is None:
            self.sampling_mask = np.ones(self.sampling_mask_shape, dtype=np.uint8)
        else:
            assert sampling_mask.ndim == 5, "Sampling mask must be 5-dimensional, with shape (1, 1, Ny, Nz, Nt)"
            if sampling_mask.shape != self.sampling_mask_shape:
                if sampling_mask.shape == (1, 1, self.Ny, self.Nz, 1) or sampling_mask.shape == (1, 1, self.Ny, self.Nz, self.Nt):
                    self.sampling_mask = np.ones(self.sampling_mask_shape, dtype=np.uint8) * sampling_mask.astype(np.uint8)
                else:
                    raise ValueError(f"Sampling mask dimensions do not match k-space data, ksp shape: {self.ksp.shape}, sampling mask shape: {sampling_mask.shape}")
            else:
                self.sampling_mask = sampling_mask.astype(np.uint8)

        self.shot_idx_list = np.unique(self.sampling_mask)
        if 0 in self.shot_idx_list:
            self.shot_idx_list = np.delete(self.shot_idx_list, np.where(self.shot_idx_list == 0))
        self.Ns = len(self.shot_idx_list)


        # Validate sensitivity maps
        self.sensitivity_maps_method = sensitivity_maps_method
        
        if sensitivity_maps is not None:
            if sensitivity_maps.shape != self.sensitivity_maps_shape:
                raise ValueError("Sensitivity maps dimensions mismatch with k-space data")
            self.sensitivity_maps = sensitivity_maps

        else:
            if sensitivity_maps_method == 'external':
                raise ValueError("Sensitivity maps are required for external method")
            elif sensitivity_maps_method == 'adaptive':
                ## TODO: add other options for sensitivity maps esitmation
                raise NotImplementedError("Adaptive sensitivity maps method is not implemented yet")
            elif sensitivity_maps_method == 'espirit':
                self.sensitivity_maps = self.generate_espirit_sensitivity_maps(**kwargs)
            elif sensitivity_maps_method == 'one':
                self.sensitivity_maps = np.ones(self.sensitivity_maps_shape, dtype=np.complex128)
            else:
                raise NotImplementedError(f"Unsupported sensitivity maps method: {sensitivity_maps_method}, only accept 'espirit', 'adaptive', 'external' and 'one'")
        
        
        # Validate scanner NIfTI file
        self.scanner_nii = scanner_nii
        if self.scanner_nii is not None:
            self.scanner_nii = Path(self.scanner_nii)
            if self.scanner_nii.exists():
                scanner_img = Image(self.scanner_nii)
                self.header = scanner_img.header
            else:
                print(f"Warning: Cannot find nii file: {self.scanner_nii}")
                self.header = None

        self.img_center = (self.Nx // 2, self.Ny // 2, self.Nz // 2)


    def __repr__(self) -> str:
        """String representation of MR3D instance."""
        return (f"MR3D(ksp_shape={self.ksp_shape}, "
                f"sampling_mask_shape={self.sampling_mask_shape}, "
                f"Ns={self.Ns}, shot_idx_list={self.shot_idx_list}, "
                f"sensitivity_maps_method={self.sensitivity_maps_method}, "
                f"sensitivity_maps_shape={self.sensitivity_maps_shape}, "
                f"scanner_nii={self.scanner_nii})")
            

    def save(self, save_path: Path | str) -> None:
        """Save MR3D instance data to file.

        Args:
            save_path: Path to save the data
        """
        save_data(save_path, data_dict={
            'ksp': self.ksp,
            'sampling_mask': self.sampling_mask,
            'sensitivity_maps_method': self.sensitivity_maps_method,
            'sensitivity_maps': self.sensitivity_maps,
            'scanner_nii': self.scanner_nii,
            'flip': None if self.flip is None else list(self.flip),
            'roll': None if self.roll is None else list(self.roll),
        })
    

    @staticmethod
    def load(load_path: str) -> 'MR3D':
        """Load MR3D instance from file.

        Args:
            load_path: Path to data file

        Returns:
            Loaded MR3D instance
        """
        data_dict = load_data(load_path)
        mr3d = MR3D(
            ksp=data_dict['ksp'],
            sampling_mask=data_dict['sampling_mask'],
            sensitivity_maps_method=data_dict['sensitivity_maps_method'],
            sensitivity_maps=data_dict['sensitivity_maps'],
            scanner_nii=data_dict['scanner_nii'],
            flip=data_dict['flip'],
            roll=data_dict['roll'],
        )
        return mr3d


    def get_ksp(self) -> npt.NDArray[np.complex128]:
        """获取k空间数据的副本。

        Returns:
            k空间数据数组
        """
        return np.copy(self.ksp)
    

    def save_nii(self, img: npt.NDArray, save_path: str | Path, roll_and_flip: bool = True) -> None:
        """将图像保存为NIFTI格式。

        Args:
            img3d: 要保存的3D图像数组
            save_path: 保存文件的路径
        """
        # os.makedirs(os.path.dirname(save_path), exist_ok='True')
        # Image(np.abs(img3d), header=self.header).save(save_path)
        if roll_and_flip:
            save_nii(img=img, save_path=save_path, header=self.header, flip=self.flip, roll=self.roll)
        else:
            save_nii(img=img, save_path=save_path, header=self.header)


    def _recon_img_per_shot(self, 
                            ksp: npt.NDArray[np.complex128],
                            sampling_mask: npt.NDArray,
                            sensitivity_maps: npt.NDArray[np.complex128],
                            shot_idx_list_to_recon: list[int] | None = None, **kwargs) -> dict[str, npt.NDArray]:
        """Perform image reconstruction for each shot.

        Args:
            shot_idx_list_to_recon: List of shot indices to reconstruct, defaults to None (all shots)

        Returns:
            Reconstructed 3D image array
        """
        shot_idx_list_to_recon = kwargs.get('shot_idx_list_to_recon', shot_idx_list_to_recon)
        multiprocessing = kwargs.get('multiprocessing', 0)

        if shot_idx_list_to_recon is None:
            shot_idx_list_to_recon = np.unique(sampling_mask)

        if 0 in shot_idx_list_to_recon:
            shot_idx_list_to_recon = np.delete(shot_idx_list_to_recon, np.where(shot_idx_list_to_recon == 0))

        if len(shot_idx_list_to_recon) == 0:
            raise ValueError("No shot indices to reconstruct")
        
        Ns = len(shot_idx_list_to_recon)
        
        Nc, Nx, Ny, Nz, Nt = ksp.shape
        img3d_per_shot = np.zeros((Nx, Ny, Nz, Nt * Ns), dtype=np.complex128)
        if multiprocessing == 0:
            for t in range(Nt):
                for i_shot, shot_idx in enumerate(shot_idx_list_to_recon):
                    img3d_per_shot[..., t * Ns + i_shot: t * Ns + i_shot + 1] = _recon_img(ksp=ksp[..., t: t + 1], 
                                                                                                sampling_mask=sampling_mask[..., t: t + 1] == shot_idx,
                                                                                                sensitivity_maps=sensitivity_maps,
                                                                                                **kwargs)['recon']
        else:
            args_list = []
            for t in range(Nt):
                for i_shot, shot_idx in enumerate(shot_idx_list_to_recon):
                    args_list.append({
                        'ksp': ksp[..., t: t + 1],
                        'sampling_mask': sampling_mask[..., t: t + 1] == shot_idx, 
                        'sensitivity_maps': sensitivity_maps, 
                        **kwargs
                        })

            with Pool(processes=multiprocessing) as pool:
                res = pool.map(_recon_img_wrapper, args_list)
            
            for idx, res_item in enumerate(res):
                img3d_per_shot[..., idx: idx + 1] = res_item['recon']

        return {'recon': img3d_per_shot}
    

    def recon_img(self, **kwargs) -> npt.NDArray:
        return self.recon_img_external(**kwargs)


    def recon_img_external(self, 
                           external_sampling_mask: np.ndarray | None = None, 
                           external_sensitivity_maps: npt.NDArray | None = None,
                           external_ksp: npt.NDArray | None = None,
                           flag_recon_per_shot: bool = False,
                           **kwargs) -> dict[str, npt.NDArray]:

        if external_ksp is not None:
            assert external_ksp.shape == self.ksp.shape, f"external_ksp shape {external_ksp.shape} does not match ksp shape {self.ksp.shape}"
            ksp = external_ksp
        else:
            ksp = self.ksp

        if external_sensitivity_maps is not None:
            assert external_sensitivity_maps.shape == self.sensitivity_maps.shape, f"external_sensitivity_maps shape {external_sensitivity_maps.shape} does not match ksp shape {self.ksp.shape}"
            sensitivity_maps = external_sensitivity_maps
        else:
            sensitivity_maps = self.sensitivity_maps

        if external_sampling_mask is not None:
            assert self.sampling_mask.shape == self.sampling_mask_shape, f"external_sampling_mask shape {external_sampling_mask.shape} does not match ksp shape {self.ksp.shape}"
            sampling_mask = external_sampling_mask
        else:
            sampling_mask = self.sampling_mask    

        if flag_recon_per_shot:
            return self._recon_img_per_shot(ksp=ksp,
                                            sampling_mask=sampling_mask,
                                            sensitivity_maps=sensitivity_maps,
                                            **kwargs)
        else:
            return _recon_img(ksp=ksp,
                                sampling_mask=sampling_mask,
                                sensitivity_maps=sensitivity_maps,
                                **kwargs)


    def generate_espirit_sensitivity_maps(self,
                                          espirit_calib_width: int = 16,
                                          espirit_thresh: float = 0.02,
                                          espirit_kernel_width: int = 6,
                                          espirit_crop: float = 0.95,
                                          espirit_max_iter: int = 100,
                                          espirit_mode: Literal['first', 'avg'] = 'avg',
                                          **kwargs) -> npt.NDArray[np.complex128]:
        """Generate sensitivity maps using ESPIRiT method.

        Args:
            method: Algorithm implementation ('sigpy' or 'python')
            calib_width: Size of calibration region
            thresh: Threshold for eigenvalue selection
            kernel_width: Size of k-space kernel
            crop: Crop ratio for sensitivity maps
            max_iter: Maximum number of iterations
            kwargs: Additional arguments

        Returns:
            Generated sensitivity maps

        Raises:
            NotImplementedError: When specified method is not supported
        """
        print('Calculating ESPIRiT sensitivity maps...')

        if espirit_mode == 'first':
            ksp = self.ksp[..., 0]
        elif espirit_mode == 'avg':
            ksp = np.mean(self.ksp, axis=-1)
        else:
            raise NotImplementedError(f"Unsupported espirit_mode: {espirit_mode}, only accept 'first' and 'avg'")
        
        espirit_sensitivity_maps = sigpy.mri.app.EspiritCalib(ksp,
                                                              calib_width=espirit_calib_width,
                                                              thresh=espirit_thresh,
                                                              kernel_width=espirit_kernel_width,
                                                              crop=espirit_crop,
                                                              max_iter=espirit_max_iter,
                                                              show_pbar=False).run()
            
        if espirit_sensitivity_maps.ndim == 4:
            espirit_sensitivity_maps = np.expand_dims(espirit_sensitivity_maps, axis=-1)
        return espirit_sensitivity_maps
    

class ASL(MR3D):
    def __init__(self,
                 flag_tag_first: bool = True,
                 perfusion_subtraction: Literal['magnitude', 'both', 'complex', 'none'] = 'both',
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.flag_tag_first = flag_tag_first
        self.perfusion_subtraction = perfusion_subtraction

    def __repr__(self) -> str:
        """String representation of MR3D instance."""
        return (f"ASL(ksp_shape={self.ksp_shape}, "
                f"sampling_mask_shape={self.sampling_mask_shape}, "
                f"Ns={self.Ns}, shot_idx_list={self.shot_idx_list}, "
                f"sensitivity_maps_method={self.sensitivity_maps_method}, "
                f"sensitivity_maps_shape={self.sensitivity_maps_shape}, "
                f"scanner_nii={self.scanner_nii}), "
                f"flag_tag_first={self.flag_tag_first}, "
                f"perfusion_subtraction={self.perfusion_subtraction})")

    
    def save_asl(self, img: np.ndarray, save_path: str | Path, roll_and_flip=True) -> None:
        if roll_and_flip:
            save_asl(img4d=img, 
                    save_path=save_path, 
                    header=self.header, 
                    perfusion_subtraction=self.perfusion_subtraction, 
                    flag_tag_first=self.flag_tag_first,
                    roll=self.roll,
                    flip=self.flip)
        else:
            save_asl(img4d=img, 
                    save_path=save_path, 
                    header=self.header, 
                    perfusion_subtraction=self.perfusion_subtraction, 
                    flag_tag_first=self.flag_tag_first)
        
    @staticmethod
    def load(load_path: str) -> 'ASL':
        """Load MR3D instance from file.

        Args:
            load_path: Path to data file

        Returns:
            Loaded MR3D instance
        """
        data_dict = load_data(load_path)
        asl = ASL(
            ksp=data_dict['ksp'],
            sampling_mask=data_dict['sampling_mask'],
            sensitivity_maps_method=data_dict['sensitivity_maps_method'],
            sensitivity_maps=data_dict['sensitivity_maps'],
            scanner_nii=data_dict['scanner_nii'],
            flag_tag_first=bool(data_dict['flag_tag_first']),
            perfusion_subtraction=data_dict['perfusion_subtraction'],
            flip=data_dict['flip'],
            roll=data_dict['roll'],
        )
        return asl
    

    def save(self, save_path: Path | str) -> None:
        """Save MR3D instance data to file.

        Args:
            save_path: Path to save the data
        """
        save_data(save_path, data_dict={
            'ksp': self.ksp,
            'sampling_mask': self.sampling_mask,
            'sensitivity_maps_method': self.sensitivity_maps_method,
            'sensitivity_maps': self.sensitivity_maps,
            'scanner_nii': self.scanner_nii,
            'flag_tag_first': self.flag_tag_first,
            'perfusion_subtraction': self.perfusion_subtraction,
            'flip': list(self.flip),
            'roll': list(self.roll),
        })
    

    def half_res(self) -> 'ASL':

        ksp_half = self.ksp[:, self.Nx // 4: self.Nx // 4 + self.Nx // 2, 
                            self.Ny // 4: self.Ny // 4 + self.Ny // 2,
                            self.Nz // 4: self.Nz // 4 + self.Nz // 2, :]
        
        sampling_mask_half = self.sampling_mask[:, self.Nx // 4: self.Nx // 4 + self.Nx // 2, 
                            self.Ny // 4: self.Ny // 4 + self.Ny // 2,
                            self.Nz // 4: self.Nz // 4 + self.Nz // 2, :]
        

        asl_half = ASL(ksp=ksp_half,
                       sampling_mask=sampling_mask_half,
                       sensitivity_maps_method=self.sensitivity_maps_method,
                       sensitivity_maps=None,
                       scanner_nii=self.scanner_nii,
                       flag_tag_first=self.flag_tag_first,
                       perfusion_subtraction=self.perfusion_subtraction,
                       flip=self.flip,
                       roll=self.roll,
                       espirit_crop=0)
    

        return asl_half