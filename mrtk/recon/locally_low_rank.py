import numpy as np
import tqdm
from einops import rearrange

def pogm_LLR(E, dd, lam, patch_size, im_size, niter, save_intermediate=False):
    """
    Locally-low-rank constrained reconstruction using POGM.

    Parameters:
        E: Object with methods `mtimes2` and `max_step`
        dd: Input data
        lam: Regularization parameter
        patch_size: Size of the patches
        im_size: Size of the image (Nx, Ny, Nz, Nt)
        niter: Number of iterations

    Returns:
        x: Reconstructed image
    """
    # Initialize
    x = np.zeros(im_size, dtype=dd.dtype)
    y = np.zeros(im_size, dtype=dd.dtype)
    z = np.zeros(im_size, dtype=dd.dtype)
    y0 = np.zeros(im_size, dtype=dd.dtype)

    dd = dd.reshape(im_size)

    p = patch_size
    L = 1 / E.max_step(10)

    a = 1  # theta in algorithm
    b = 1  # gamma in algorithm

    if save_intermediate:
        extra = dict(
            x_list=[],
            iter_list=[]
        )
    else:
        extra = None


    # Main loop
    for iter in tqdm.tqdm(range(niter), total=niter, desc='POGM-LLR'):
        # y-update
        y0 = y.copy()
        y = x - (1 / L) * (E.mtimes2(rearrange(x, 'Nx Ny Nz Nt -> 1 (Nx Ny Nz Nt)')).reshape(im_size) - dd)

        # a-update (theta)
        a0 = a
        if iter < niter:
            a = (1 + np.sqrt(4 * a**2 + 1)) / 2
        else:
            a = (1 + np.sqrt(8 * a**2 + 1)) / 2

        # z-update
        z = y + ((a0 - 1) / a) * (y - y0) + (a0 / a) * (y - x) + ((a0 - 1) / (L * b * a)) * (z - x)

        # b-update (gamma)
        b0 = b
        b = (2 * a0 + a - 1) / (L * a)  

        # x-update
        starts = [np.random.permutation(p[i])[0] - (p[i] - 1) / 2 for i in range(3)]
        axes = [np.arange(starts[i], im_size[i], p[i]) for i in range(3)]
        ii, jj, kk = np.meshgrid(axes[0], axes[1], axes[2], indexing='ij')

        for i in range(ii.size):
            i0, j0, k0 = int(ii.flat[i]), int(jj.flat[i]), int(kk.flat[i])
            q = get_patch(z, i0, j0, k0, p)
            u, s, vh = np.linalg.svd(q.reshape(-1, im_size[3]), full_matrices=False)
            s = shrink(s, lam * b)
            q = (u @ np.diag(s) @ vh).reshape(q.shape)
            x = put_patch(x, q, i0, j0, k0, p)

        if save_intermediate and (iter + 1) % 10 == 0:
            extra['x_list'].append(x.copy())
            extra['iter_list'].append(iter + 1)


    return x, extra


def get_patch(X, i, j, k, p):
    sx, sy, sz, st = X.shape
    return X[
        max(i - (p[0] - 1) // 2, 0):min(i + (p[0] - 1) // 2 + 1, sx),
        max(j - (p[1] - 1) // 2, 0):min(j + (p[1] - 1) // 2 + 1, sy),
        max(k - (p[2] - 1) // 2, 0):min(k + (p[2] - 1) // 2 + 1, sz),
        :
    ]


def put_patch(X, q, i, j, k, p):
    sx, sy, sz, st = X.shape
    X[
        max(i - (p[0] - 1) // 2, 0):min(i + (p[0] - 1) // 2 + 1, sx),
        max(j - (p[1] - 1) // 2, 0):min(j + (p[1] - 1) // 2 + 1, sy),
        max(k - (p[2] - 1) // 2, 0):min(k + (p[2] - 1) // 2 + 1, sz),
        :
    ] = q
    return X


def shrink(x, thresh):
    return np.maximum(x - thresh, 0)


'''
TODO: extend this function to torch with GPU support
However, 
1) Currently, complex numbers are not supported in torch with mps;
2) torch.unfold does not support 3D data (5D tensor);
so we need to wait for the torch to support this in the future.

Currently, what we can do now is to use multiprocessing to replace the for loop in the code.

'''