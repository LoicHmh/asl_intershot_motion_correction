import numpy as np
from enum import IntEnum
from ..math.rigid_transform import RigidTransform, sinc_rigid_transform
import sigpy as sp
from mrtk.math.fft_utils import ifftnd, fftnd


class VERBOSE(IntEnum):
    NONE = 0
    INFO = 1
    DEBUG = 2


class AlignedSENSE:
    def __init__(self,
                 tolerance=1e-12,
                 max_iter=1000,
                 n_cg=5,
                 cg_tol=1e-10,
                 n_newton=1,
                 w_init=1,
                 flag_solve_T=False,
                 verbose=VERBOSE.DEBUG):
        
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.n_cg = n_cg
        self.cg_tol = cg_tol
        self.n_newton = n_newton
        self.w_init = w_init
        self.flag_solve_T = flag_solve_T
        self.verbose = verbose

    
    def solve_x(self, x, y, T, S, A):
        """
        Reconstructs an image using CG SENSE with multishot alignment.
        x: (Ny, Nz, Nx)
        y: (Nc, Nx, Ny, Nz)
        A: (Nshot, 1, 1, Ny, Nz)
        S: (Nc, Nx, Ny, Nz)
        T: list of rigid_transform, length is Nshot
        
        """
        n_shot = A.shape[0]
        rigid_transforms = T
        
        def A_op(x_):
            xEnd = np.zeros_like(x_, dtype=np.complex128)
    
            for s in range(n_shot):
                xS = sinc_rigid_transform(x_, rigid_transforms[s])
                xS = xS[np.newaxis, ...] * S
                xS = fftnd(xS, axes=(1, 2, 3))
                xS = xS * A[s]
                xS = ifftnd(xS, axes=(1, 2, 3))
                xS = np.sum(xS * np.conj(S), axis=0)
                xEnd += sinc_rigid_transform(xS, rigid_transforms[s].inverse())
            
            return xEnd

        b = np.zeros_like(x, dtype=np.complex128)

        for s in range(n_shot):
            yS = y * A[s]
            yS = ifftnd(yS, axes=(1, 2, 3))
            yS = np.sum(yS * np.conj(S), axis=0)
            b += sinc_rigid_transform(yS, rigid_transforms[s].inverse())
        
        cg = sp.alg.ConjugateGradient(A_op, b, x, max_iter=self.n_cg, tol=self.cg_tol)
        while not cg.done():
            cg.update()
        
        return cg.x


    def solve_T(self, x, yIn, M, T, S, A, kGrid, kkGrid, rGrid, rkGrid, w, flagw):
        pass


    def solve(self, args=None, y=None, A=None, S=None, T=None, flag_return_T=False):
        '''
        y: (Ny, Nz, Nx, Nc)  -> (Nc, Nx, Ny, Nz)
        A: (Ny, Nz, 1, 1, Nshot) -> (1, 1, Ny, Nz, Nshot)
        S: (Ny, Nz, Nx, Nc) -> (Nc, Nx, Ny, Nz)
        T: (1, 1, 1, 1, n_shot, 6) -> (n_shot, 
        '''
        if args is not None:
            y = args.get('y', y)
            A = args.get('A', A)
            S = args.get('S', S)
            T = args.get('T', T)
            flag_return_T = args.get('flag_return_T', flag_return_T)

        Nc, Nx, Ny, Nz = y.shape
        x = np.zeros((Nx, Ny, Nz), dtype=np.complex128)

        yX = ifftnd(y, axes=(0, 1, 2))
        maximNormalize = np.max(np.abs(yX))
        yIn = y / maximNormalize

        error_min = 1e6
        x_with_min_error = x.copy()

        for n in range(self.max_iter):
            x_prev = x.copy()
            x = self.solve_x(x, yIn, T, S, A)

            # if self.flag_solve_T:
            #     # To check convergence
            #     flagwprev = flagw.copy()
            #     T, x, w, flagw = self.solve_T(x, yIn, M, T, S, A, kGrid, kkGrid, rGrid, rkGrid, w, flagw)

            #     # To avoid numeric instabilities
            #     w = np.where(w < 1e-4, 2 * w, w)
            #     w = np.where(w > 1e16, w / 2, w)
            
            error = np.max(np.real((x - x_prev) * np.conj(x - x_prev)))
            if error < error_min:
                x_with_min_error = x.copy()
                error_min = error

            if self.verbose >= VERBOSE.DEBUG:
                print(f'Iteration {n+1:04d} - Error {error:.2e}')
            
            # if error < self.tolerance and np.sum(flagwprev != 1) > 0:
            if error < self.tolerance:
                if self.verbose >= VERBOSE.INFO:
                    print(f'Convergence reached at iteration {n+1:04d}')
                break
            elif n == self.max_iter - 1 and self.verbose >= VERBOSE.INFO:
                print('Maximum iterations reached without convergence')

            elif error / error_min > 10:
                print('Error increased, early stopping')
                break
        
        x_with_min_error = x_with_min_error * maximNormalize
        
        if flag_return_T:
            return x_with_min_error, T
        else:
            return x_with_min_error
