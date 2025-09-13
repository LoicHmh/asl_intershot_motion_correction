from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
from ..math.fft_utils import fftnd, ifftnd
import scipy.sparse
from scipy.sparse.linalg import LinearOperator
from ..math.rigid_transform import RigidTransform, build_trilinear_interpolation_matrix, SincRigidTransformOP
from typing import List, Literal, Optional


class MROP(ABC):
    """
    Abstract base class for MRI reconstruction operators.

    This class defines the interface for forward (fwd) and adjoint (adj) operations.
    """

    @abstractmethod
    def fwd(self, x: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
        """
        Perform the forward operation (E * x).

        Args:
            x: Input array.

        Returns:
            Result of the forward operation.
        """
        pass

    @abstractmethod
    def adj(self, y: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
        """
        Perform the adjoint operation (E' * y).

        Args:
            y: Input array.

        Returns:
            Result of the adjoint operation.
        """
        pass

    def mtimes(self, x: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
        """
        Perform the forward operation (alias for fwd).

        Args:
            x: Input array.

        Returns:
            Result of the forward operation.
        """
        return self.fwd(x)

    def mtimes2(self, x: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
        """
        Perform the operation E' * (E * x).

        Args:
            x: Input array.

        Returns:
            Result of the operation.
        """
        return self.adj(self.fwd(x))
    
    def max_step(self, n: int = 10) -> float:
        """
        Use the power method to estimate the maximum eigenvalue of E'E.

        Args:
            n: Maximum number of iterations for the power method.

        Returns:
            The reciprocal of the maximum eigenvalue of E'E.
        """
        y = np.random.randn(*self.flattened_ksp_size).astype(np.complex128)
        norm_y = 0
        for _ in range(n):
            y = self.fwd(self.adj(y))
            new_norm_y = np.linalg.norm(y)
            if np.abs(new_norm_y - norm_y) / (norm_y if norm_y != 0 else 1) < 1e-4:
                break
            norm_y = new_norm_y
        return 1.0 / norm_y
    

class SenseOP(MROP):
    """
    Sense operator for MRI reconstruction.

    Args:
        sampling_mask: Sampling mask.
        sensitivity_maps: Sensitivity maps.
    """
    def __init__(self, sampling_mask: np.ndarray, sensitivity_maps: np.ndarray, ksp_size: tuple[int, ...]) -> None:
        """
        Initialize the Sense operator.
        Args:
            sampling_mask: Sampling mask (Nc=1, Nx=1, Ny, Nz, Nt).
            sensitivity_maps: Sensitivity maps (Nc, Nx, Ny, Nz, Nt).
            ksp_size: Size of the k-space data (Nc, Nx, Ny, Nz, Nt). 
        """

        if len(ksp_size) == 4:
            # Add a dummy dimension for time
            ksp_size = (*ksp_size, 1)

        assert len(ksp_size) == 5, \
            f"ksp_size {ksp_size} should be of length 5, but got {len(ksp_size)}, it should be (Nc, Nx, Ny, Nz, Nt)"
        ksp_size = np.asarray(ksp_size, dtype=np.int64)
        self.Nc, self.Nx, self.Ny, self.Nz, self.Nt = ksp_size
        self.Nd = self.Nx * self.Ny * self.Nz
        self.flattened_ksp_size = (self.Nc, self.Nd * self.Nt)
        self.ksp_size = ksp_size

        if sampling_mask.ndim == 4:
            sampling_mask = np.expand_dims(sampling_mask, axis=-1)

        assert sampling_mask.shape in [(self.Nc, self.Nx, self.Ny, self.Nz, self.Nt), 
                                       (1, self.Nx, self.Ny, self.Nz, self.Nt),
                                       (1, 1, self.Ny, self.Nz, self.Nt)], \
            f"sampling_mask shape {sampling_mask.shape} does not match ksp_size {ksp_size}"
        
        # sampling_mask should be binary
        sampling_mask = np.where(sampling_mask > 0, 1, 0).astype(np.bool_)
        sampling_mask = np.ones(ksp_size, dtype=np.bool_) * sampling_mask

        self.sampling_mask = sampling_mask.reshape(self.flattened_ksp_size)
        
        if sensitivity_maps.ndim == 4:
            sensitivity_maps = np.expand_dims(sensitivity_maps, axis=-1)
        
        assert sensitivity_maps.shape in [(self.Nc, self.Nx, self.Ny, self.Nz, self.Nt),
                                          (self.Nc, self.Nx, self.Ny, self.Nz, 1)], \
            f"sensitivity_maps shape {sensitivity_maps.shape} does not match ksp_size {ksp_size}"
        
        sensitivity_maps = np.ones(ksp_size, dtype=np.complex128) * sensitivity_maps
        self.sensitivity_maps = sensitivity_maps.reshape(self.flattened_ksp_size)

    def A(self, ksp: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
        """
        Apply the binary sampling mask.

        Args:
            ksp: Input k-space data. (Nc, Nd x Nt)

        Returns:
            masked k-space data. (Nc, Nd x Nt)
        """
        assert ksp.shape == self.flattened_ksp_size, \
            f"ksp shape {ksp.shape} does not match ksp_size {self.flattened_ksp_size}"
        
        return ksp * self.sampling_mask
    
    def S(self, single_coil_img: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
        """
        Apply the forward operation with sensitivity maps.

        Args:
            single_coil_img: Input single-coil image. (1, Nd x Nt)

        Returns:
            multi_coil_img: Multi-coil image. (Nc, Nd x Nt)
        """

        assert single_coil_img.shape == (1, self.Nd * self.Nt), \
            f"single_coil_img shape {single_coil_img.shape} does not match (1, Nd x Nt) {(1, self.Nd * self.Nt)}"
        
        return single_coil_img * self.sensitivity_maps
    
    def S_adj(self, multi_coil_img: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
        """
        Apply the adjoint operation with sensitivity maps.

        Args:
            multi_coil_img: Input multi-coil image. (Nc, Nd x Nt)

        Returns:
            single_coil_img: Single-coil image. (1, Nd x Nt)
        """

        assert multi_coil_img.shape == self.flattened_ksp_size, \
            f"multi_coil_img shape {multi_coil_img.shape} does not match (Nc, Nd x Nt) {self.flattened_ksp_size}"
        
        return np.sum(np.conj(self.sensitivity_maps) * multi_coil_img, axis=0, keepdims=True)
    
    def F(self, 
          multi_coil_img: npt.NDArray[np.complex128], 
          shift: bool = True, 
          axes: tuple[int, ...] = (1, 2, 3)) -> npt.NDArray[np.complex128]:
        """
        Apply the Fourier transform.

        Args:
            img: Input image. (Nc, Nd x Nt)
            shift: Whether to perform FFT shift 
            axes: Axes along which to perform FFT

        Returns:
            k-space data. (Nc, Nd x Nt)
        """
        assert multi_coil_img.shape == self.flattened_ksp_size, \
            f"multi_coil_img shape {multi_coil_img.shape} does not match (Nc, Nd x Nt) {self.flattened_ksp_size}"
        
        multi_coil_img = multi_coil_img.reshape(self.ksp_size)
        ksp = fftnd(multi_coil_img, axes=axes, shift=shift)
        return ksp.reshape(self.flattened_ksp_size)

    def F_adj(self, 
              ksp: npt.NDArray[np.complex128], 
              shift: bool = True, 
              axes: tuple[int, ...] = (1, 2, 3)) -> npt.NDArray[np.complex128]:
        """
        Apply inverse Fourier transform.

        Args:
            ksp: Input k-space data (Nc, Nd x Nt)
            shift: Whether to perform IFFT shift
            axes: Axes along which to perform IFFT

        Returns:
            Image domain data (Nc, Nd x Nt)
        """

        assert ksp.shape == self.flattened_ksp_size, \
            f"ksp shape {ksp.shape} does not match flattened_ksp_size {self.flattened_ksp_size}"

        ksp = ksp.reshape(self.ksp_size)
        multi_coil_img = ifftnd(ksp, axes=axes, shift=shift)

        return multi_coil_img.reshape(self.flattened_ksp_size)
    
    def fwd(self, single_coil_img: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
        """
        Apply SENSE forward model.

        Args:
            img: Input image (1, Nd x Nt)

        Returns:
            k-space data (Nc, Nd x Nt)
        """
        if single_coil_img.ndim == 1:
            single_coil_img = np.expand_dims(single_coil_img, axis=0)

        assert single_coil_img.shape == (1, self.Nd * self.Nt), \
            f"img shape {single_coil_img.shape} does not match (1, Nd x Nt) {(1, self.Nd * self.Nt)}"

        ksp = self.A(self.F(self.S(single_coil_img)))
        return ksp

    def adj(self, ksp: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
        """
        Apply SENSE adjoint model.

        Args:
            ksp: Input k-space data (Nc, Nd x Nt)

        Returns:
            Image domain data (1, Nd x Nt)
        """

        assert ksp.shape == self.flattened_ksp_size, \
            f"ksp shape {ksp.shape} does not match flattened_ksp_size {self.flattened_ksp_size}"

        img = self.S_adj(self.F_adj(self.A(ksp)))
        return img

    
    def run_cg_sense_l2(self, 
                        ksp: npt.NDArray[np.complex128], 
                        lambda_l2: float, 
                        maxiter: int = 100, 
                        atol: float = 1e-6, 
                        Nt_out: int = -1, 
                        x0: Optional[npt.NDArray[np.complex128]] = None) -> npt.NDArray[np.complex128]:
        """
        Perform SENSE reconstruction using L2 norm.

        Args:
            ksp: Input k-space data (Nc, Nx, Ny, Nz, Nt)
            niter: Number of iterations 

        Returns:
            Reconstructed image (Nc, Nx, Ny, Nz, Nt)
        """
        if Nt_out > 0:
            Nt = Nt_out
        else:
            Nt = self.Nt

        if Nt > 1:
            print(f"Warning: You are using cg sense l2 jointly with all multiple time frames Nt = {self.Nt}")

        if ksp.ndim == 4:
            ksp = np.expand_dims(ksp, axis=-1)
        if ksp.shape[-1] != self.ksp_size[-1]:
            ksp = np.repeat(ksp, self.Nt, axis=-1)

        assert np.all(np.array(ksp.shape) == self.ksp_size), \
            f"ksp shape {ksp.shape} does not match ksp_size {self.ksp_size}"

        ksp = ksp.reshape(self.flattened_ksp_size)

        def mtimes2_l2(img: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
            return self.mtimes2(img) + lambda_l2 * img
        
        A = LinearOperator(
            shape=(self.Nd * Nt, self.Nd * Nt),
            matvec=mtimes2_l2,
            dtype=np.complex128)
        
        b = self.adj(ksp).reshape(self.Nd * Nt)

        if x0 is None:
            x0 = np.zeros(self.Nd * Nt, dtype=np.complex128)
        else:
            x0 = x0.reshape(self.Nd * Nt)


        img, exit_code = scipy.sparse.linalg.cg(A=A, b=b, x0=x0, maxiter=maxiter, atol=atol)
        
        if exit_code == 0:
            print("Successfully converged.")

        img = img.reshape((self.Nx, self.Ny, self.Nz, Nt))
        return img
    
    def run_rss_ifft(self, ksp: npt.NDArray[np.complex128], shift: bool = True) -> npt.NDArray[np.complex128]:
        """
        Perform root-sum-of-squares (RSS) reconstruction using IFFT.

        Args:
            ksp: Input k-space data
            shift: Whether to perform FFT shift

        Returns:
            Reconstructed image
        """
        if ksp.ndim == 4:
            ksp = np.expand_dims(ksp, axis=-1)
        assert np.all(ksp.shape == self.ksp_size), \
            f"ksp shape {ksp.shape} does not match flattened_ksp_size {self.ksp_size}"
        ksp = ksp.reshape(self.flattened_ksp_size)
        ksp = self.A(ksp)
        img = self.F_adj(ksp, shift=shift)
        img = np.sqrt(np.sum(np.abs(img) ** 2, axis=0))
        img = img.reshape((self.Nx, self.Ny, self.Nz, self.Nt))
        return img
    

class MotionCompensatedSenseOP(SenseOP):
    """
    Motion Compensated SENSE operator for MRI reconstruction.
    In this class, we only accept one volume of motion-corrupted image with multiple shots.
    This implies that only one motion-free image will be reconstructed for the multiple shots.
    Nt = number of shots.

    Args:
        sampling_mask: Sampling mask.
        sensitivity_maps: Sensitivity maps.

        
    """
    def __init__(self, 
                 sampling_mask: np.ndarray, 
                 sensitivity_maps: np.ndarray, 
                 rigid_transforms: List[RigidTransform], 
                 ksp_size: tuple[int, ...],
                 interpolation_method: Literal['trilinear', 'sinc'] = 'sinc',
                 inverse_transforms: bool = False,
                 ) -> None:
        super().__init__(sampling_mask, sensitivity_maps, ksp_size)
        if inverse_transforms:
            rigid_transforms = [rigid_transform.inverse() for rigid_transform in rigid_transforms]
        self.rigid_transforms = rigid_transforms
        assert len(rigid_transforms) == self.Nt, 'Number of rigid transforms should match Nt'
        self.interpolation_method = interpolation_method

        if interpolation_method == 'trilinear':
            self.trilinear_interpolation_matrices = []
            for rigid_transform in rigid_transforms:
                trilinear_interpolation_matrix = build_trilinear_interpolation_matrix((self.Nx, self.Ny, self.Nz), rigid_transform.mtx)
                self.trilinear_interpolation_matrices.append(trilinear_interpolation_matrix)
        elif interpolation_method == 'sinc':
            self.sinc_rigid_transforms = [SincRigidTransformOP(rigid_transform=rigid_transform, img_size=(self.Nx, self.Ny, self.Nz)) for rigid_transform in rigid_transforms]


    def check(self):
        if self.interpolation_method == 'sinc':
            return all([sinc_rigid_transform.check() for sinc_rigid_transform in self.sinc_rigid_transforms])
        elif self.interpolation_method == 'trilinear':
            raise NotImplementedError("Trilinear interpolation check not implemented.")

    def fwd(self, single_coil_img: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
        """
        Apply SENSE forward model.

        Args:
            img: Input image

        Returns:
            k-space data
        """
        if single_coil_img.ndim == 1:
            single_coil_img = np.expand_dims(single_coil_img, axis=0)

        assert single_coil_img.shape == (1, self.Nd * 1), \
            f"img shape {single_coil_img.shape} does not match (1, Nd x Nt) {(1, self.Nd * 1)}"

        ksp = self.A(self.F(self.S(self.T(single_coil_img))))
        return ksp


    def adj(self, ksp: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
        """
        Apply SENSE adjoint model.

        Args:
            ksp: Input k-space data

        Returns:
            Image domain data
        """

        assert ksp.shape == self.flattened_ksp_size, \
            f"ksp shape {ksp.shape} does not match flattened_ksp_size {self.flattened_ksp_size}"

        img = self.T_adj(self.S_adj(self.F_adj(self.A(ksp))))
        return img


    def T(self, img_free: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
        """
        Apply the rigid transform to the image.

        Args:
            img: Motion-free image.

        Returns:
            Transformed image.
        """

        if self.interpolation_method == 'sinc':
            img_free = img_free.reshape((self.Nx, self.Ny, self.Nz))
            img_move = np.zeros((self.Nx, self.Ny, self.Nz, self.Nt), dtype=np.complex128)
            for t in range(self.Nt):
                img_move[..., t] = self.sinc_rigid_transforms[t].fwd(img_free)

            return img_move.reshape((1, self.Nd * self.Nt))

        elif self.interpolation_method == 'trilinear':
            img_move = np.zeros((self.Nd, self.Nt), dtype=np.complex128)
            for t in range(self.Nt):
                
                img_move[..., t] = self.trilinear_interpolation_matrices[t] @ img_free.flatten()

            return img_move.reshape((1, self.Nd * self.Nt))
        else:
            raise ValueError(f"Interpolation method {self.interpolation_method} not supported. Use 'sinc' or 'trilinear'.")
        

    def T_adj(self, img_move: npt.NDArray[np.complex128], reduction: Literal['sum', 'none'] = 'sum') -> npt.NDArray[np.complex128]:
        """
        Apply the adjoint of the rigid transform to the image.

        Args:
            img: Motion-corrupted image.

        Returns:
            Transformed image.
        """
        if self.interpolation_method == 'sinc':
            img_move = img_move.reshape((self.Nx, self.Ny, self.Nz, self.Nt))
            img_free = np.zeros((self.Nx, self.Ny, self.Nz, self.Nt), dtype=np.complex128)
            for t in range(self.Nt):
                img_free[..., t] = self.sinc_rigid_transforms[t].adj(img_move[..., t])
            if reduction == 'sum':
                img_free = np.sum(img_free, axis=-1)
                return img_free.reshape((1, self.Nd * 1))
            elif reduction == 'none':
                return img_free.reshape((1, self.Nd * self.Nt))

        elif self.interpolation_method == 'trilinear':
            img_move = img_move.reshape((self.Nd, self.Nt))
            img_free = np.zeros((self.Nd, self.Nt), dtype=np.complex128)
            for t in range(self.Nt):
                img_free[..., t] = self.trilinear_interpolation_matrices[t].T @ img_move[..., t].flatten()

            img_free = np.sum(img_free, axis=-1)
            return img_free.reshape((1, self.Nd * 1))
        else:
            raise ValueError(f"Interpolation method {self.interpolation_method} not supported. Use 'sinc' or 'trilinear'.")


    def update_x_cg(self, ksp: npt.NDArray[np.complex128], maxiter: int = 100, atol: float = 1e-6, x0: Optional[npt.NDArray[np.complex128]] = None) -> npt.NDArray[np.complex128]:
        return super().run_cg_sense_l2(ksp, lambda_l2=0, maxiter=maxiter, atol=atol, Nt_out=1, x0=x0)
    

    def update_T_newton(self, ksp: npt.NDArray[np.complex128], maxiter: int = 100, atol: float = 1e-6) -> npt.NDArray[np.complex128]:
        pass

    def solve_x(self, y: npt.NDArray[np.complex128], maxiter: int = 100, atol: float = 1e-6) -> npt.NDArray[np.complex128]:
        """
        Solve for the motion-compensated image using CG SENSE.

        Args:
            ksp: Input k-space data (Nc, Nx, Ny, Nz, Nt)
            maxiter: Number of iterations
            atol: Tolerance for convergence

        Returns:
            Reconstructed image (Nc, Nx, Ny, Nz)
        """

        # only for testing purpose

        # if y.ndim == 4:
        #     import nibabel as nib
        #     affine = np.eye(4)
        #     y_ = np.repeat(y, self.Nt, axis=-1)

        #     img = self.T_adj(self.S_adj(self.F_adj(self.A(y_.reshape(self.flattened_ksp_size)))), reduction='none').reshape(64, 64, 32, 4)
        #     # img = self.S_adj(self.F_adj(self.A(y_.reshape(self.flattened_ksp_size)))).reshape(64, 64, 32, 4)
        #     # img = self.S_adj(self.F_adj(y_.reshape(self.flattened_ksp_size))).reshape(64, 64, 32, 4)

        #     return img
        
        x = np.zeros((self.Nx, self.Ny, self.Nz, 1), dtype=np.complex128)

        yX = ifftnd(y, axes=(1, 2, 3))
        max_norm = np.max(np.abs(yX))
        y = y / max_norm

        error_min = 1e6
        x_with_min_error = x.copy()

        for n in range(maxiter):

            x_prev = x.copy()
            x = self.update_x_cg(ksp=y, x0=x_prev, maxiter=5, atol=atol)

            error = np.max(np.abs(x - x_prev) ** 2)
            print(f'Iteration {n+1:04d}, error = {error:.4e}')
            if error < error_min:
                x_with_min_error = x.copy()
                error_min = error
            
            if error < atol:
                print(f'Convergence reached at iteration {n+1:04d}')
                break

            elif n == maxiter - 1:
                print('Maximum iterations reached without convergence')



        x_with_min_error = x_with_min_error * max_norm

        return x_with_min_error
