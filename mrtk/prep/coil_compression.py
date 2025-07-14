import os
import sys
import numpy as np
from dotenv import load_dotenv
from einops import rearrange


def find_bart_path():
    load_dotenv()
    bart_path = os.getenv('BART_PATH')
    if bart_path and os.path.exists(bart_path):
        return bart_path
    else:
        print('Warning: BART_PATH not set or does not exist. Please set the BART_PATH environment variable.')
        return None


def coil_compression(ksp: np.ndarray, ncc: int = 16, external_mtx_cc: np.ndarray = None) -> tuple:
    """
    ksp: k-space data need to be compressed, (n_col, n_cha, n_lin, n_sli, n_rep)
    ncc: number of coils after compression

    """

    bart_path = find_bart_path()
    sys.path.append(bart_path)
    import bart

    if ksp.ndim == 4:
        ksp = np.expand_dims(ksp, axis=-1)
    elif ksp.ndim != 5:
        raise ValueError(f'ksp must be 4D or 5D, but got {ksp.ndim}D')
    
    n_col, n_cha, n_lin, n_sli, n_rep = ksp.shape

    if n_cha < ncc:
        raise ValueError(f'Number of coils {n_cha} must be greater than or equal to {ncc}')

    ksp_bart = rearrange(ksp, 'n_col n_cha n_lin n_sli n_rep -> n_col n_lin n_sli n_cha n_rep')

    if external_mtx_cc is None:
        mtx_cc = bart.bart(1, 'cc -S -M', ksp_bart[..., 0])
    else:
        mtx_cc = external_mtx_cc
    
    ksp_cc = np.zeros((n_col, n_lin, n_sli, ncc, n_rep), dtype=ksp.dtype)
    for i_rep in range(n_rep):
        ksp_cc[..., i_rep] = bart.bart(1, f'ccapply -p {ncc} -S', ksp_bart[..., i_rep], mtx_cc)

    ksp_cc = rearrange(ksp_cc, 'n_col n_lin n_sli n_cha n_rep -> n_cha n_col n_lin n_sli n_rep')
    return ksp_cc, mtx_cc
