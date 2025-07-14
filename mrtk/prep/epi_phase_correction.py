import numpy as np
from mrtk.math.fft_utils import ifftnd, fftnd
from mrtk.utils.twix_utils import load_twix_pc, load_twix
from typing import Literal


def phase_correction(im_array, pc_array, point_to_point=False):
#  The im_array has the format: [Col Cha Lin Seg], note that odd and even
#  lines are stored in separate segments, the rest of the lines are zeros.
#  Odd and even lines can be combined by adding over segments
#  The navigators have the format: [Col Cha Ave Seg],
#  there are two "averages" because we acquire the one of the directions
#  twice (three line navigator)
    
#  "Segment 1" contains the data in the odd lines (zeros in even lines) =
#  forward lines 
#  "Segment 2" contains the data in the even lines (zeros in the odd lines)
#  = reversed lines 
#  Two averages (as there are three phase correction lines) 
    
    n_col, n_cha, n_lin, n_seg = im_array.shape
    
    # ifft col dim.
    pc_fft = ifftnd(pc_array, [0])

    # find the interval without background
    pc_fft_abs = np.abs(pc_fft)
    valid_interval = np.int16(np.sum(pc_fft_abs[..., 0, 0] / np.max(pc_fft_abs[..., 0, 0], axis=0, keepdims=True) > 0.6, axis=0) / 2)
    

    # (1.2) find which line is reflected (taking the mean is relevant for the
    # one where there are two lines...) 
    pc_fft_fwd = pc_fft[:, :, 0, 0] # take the odd line 
    pc_fft_bwd = pc_fft[:, :, :, 1].mean(axis = 2) # take the mean of the even lines 

    # (1.3) find phase difference (complex division, then taking angle) 
    phase_diff_orig = np.angle(pc_fft_fwd * np.conj(pc_fft_bwd))
    magnitude_diff = np.abs(pc_fft_fwd / pc_fft_bwd)
    
    if point_to_point:
        phase_diff = phase_diff_orig

        # (3) Apply the phase correction to the data
        im_fft = ifftnd(im_array, [0])
        im_fft[:, :, :, 0] = im_fft[:, :, :, 0] * np.exp(-1j * phase_diff.reshape((n_col, n_cha, 1)) / 2)
        im_fft[:, :, :, 1] = im_fft[:, :, :, 1] * np.exp(1j * phase_diff.reshape((n_col, n_cha, 1)) / 2)
        im_array = fftnd(im_fft, [0])

    else:
        # TODO: debug with this branch
        # (1.4) fit phase difference to straight line (polynomial of order 1)
        x = np.arange(n_col) - n_col//2
        angle_fit = []
        for i_cha in range(n_cha):
            v = valid_interval[i_cha]
            m = n_col // 2 + 1
            angle_p = np.polyfit(x[m - v: m + v], phase_diff_orig[m - v: m + v, i_cha], deg=1)
            angle_fit.append(angle_p)
        angle_fit = np.array(angle_fit)

        # (2.1) In this case take the average over all coils (excluding outlier coils)
        angle_mean = np.mean(angle_fit, axis=0)
        angle_std = np.std(angle_fit, axis=0)
        i_not_outlier = np.all(np.abs(angle_fit - angle_mean) < 2 * angle_std, axis=1)

        # print(f"{(np.logical_not(i_not_outlier)).sum()} of outlier coils not used. ")

        # Take the mean accross all coils, which are not outliers: 
        angle_mean = np.mean(angle_fit[i_not_outlier], axis=0)
        phase_diff = np.poly1d(angle_mean)(x)

        # np.save('phase_diff_orig_', phase_diff, allow_pickle=True)
    
        # (3) Apply the phase correction to the data
        im_fft = ifftnd(im_array, [0])
        im_fft[:, :, :, 0] = im_fft[:, :, :, 0] * np.exp(-1j * phase_diff.reshape((n_col, 1, 1)) / 2)
        im_fft[:, :, :, 1] = im_fft[:, :, :, 1] * np.exp(1j * phase_diff.reshape((n_col, 1, 1)) / 2)
        im_array = fftnd(im_fft, [0])



    im_array = im_array.sum(axis=3)
    return im_array


def epi_phase_correction_3d(ksp, pc, point_to_point: bool = False, logger = None, mode: Literal['single-volume', 'avg-volume'] = 'single-volume'):
    if logger is not None:
        logger.debug('EPI phase correction...')

    # Invert Fourier Transform on Axis Z for 3D GRASE Image
    ksp_ifft_z = ifftnd(ksp, [3])

    if mode == 'avg-volume':
        pc = pc.mean(axis=3)

    # calculate phase-correction
    n_col, n_cha, n_lin, n_sli, n_rep, n_seg = ksp_ifft_z.shape
    ksp_pc = np.zeros((n_col, n_cha, n_lin, n_sli, n_rep), dtype=ksp_ifft_z.dtype)
    for i_rep in range(n_rep):
        for i_sli in range(n_sli):
            if mode == 'single-volume':
                ksp_pc[:, :, :, i_sli, i_rep] = phase_correction(ksp_ifft_z[:, :, :, i_sli, i_rep, :], pc[:, :, :, i_rep, :], point_to_point)
            elif mode == 'avg-volume':
                ksp_pc[:, :, :, i_sli, i_rep] = phase_correction(ksp_ifft_z[:, :, :, i_sli, i_rep, :], pc[:, :, :, :], point_to_point)
    ksp_pc = fftnd(ksp_pc, [3])

    return ksp_pc