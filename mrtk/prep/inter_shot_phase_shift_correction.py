import numpy as np
import matplotlib.pyplot as plt
from mrtk.recon.samp_mask import SampMask
from mrtk.math.fft_utils import fftnd, ifftnd


"""
In ASL imaging, the control and tag images are acquired separately, and due to field inhomogeneities 
or system drift, there may be a small phase shift between them. If we directly subtract the magnitude 
or even the complex values without correction, this phase difference can lead to incorrect perfusion 
signal estimation.

To correct for this, we estimate the global phase shift between the tag and control images. 
We then rotate the control image in the complex plane by multiplying it with exp(-1j * phase_shift), 
effectively aligning its phase with the tag image. This ensures that the subtraction is performed 
correctly, preserving the true perfusion signal.
"""

def estimate_phase_shift(reference, measurement):
    """
    Estimate the global phase shift between complex reference and measurement images.
    Usually we take the first shot as reference and the following shots as measurements.
    """
    magnitude_reference = np.abs(reference)
    magnitude_measurement = np.abs(measurement)
    threshold = 0.1 * magnitude_reference.max()
    mask = (magnitude_reference > threshold) & (magnitude_measurement > threshold)

    phase_reference = np.angle(reference)
    phase_measurement = np.angle(measurement)
    phase_diff = phase_reference[mask] - phase_measurement[mask]
    global_phase_shift = np.angle(np.exp(1j * phase_diff).mean()) # Mean of phase differences, circular mean
    return global_phase_shift

def apply_phase_correction(measurement, global_phase_shift):
    """
    Apply phase correction to measurement image before subtraction.
    """
    measurement_corrected = measurement * np.exp(1j * global_phase_shift)
    return measurement_corrected


# def inter_shot_phase_shift_correction(ksp_pc: np.ndarray, samp_mask: SampMask) -> np.ndarray:
#     print("Performing inter-shot phase shift correction...")

#     ksp_pc_res = np.zeros_like(ksp_pc, dtype=ksp_pc.dtype)

#     # Plot the phase shift estimation
#     fig, axs = plt.subplots(nrows=ksp_pc.shape[-1], ncols=samp_mask.n_shot)

#     reference_img = ifftnd(ksp_pc[:, :, :, :, 0] * samp_mask.get_binary_mask(i_shot=1), axes=(0, 2, 3))
#     for i_rep in range(ksp_pc.shape[-1]):
#         for i_shot in range(1, samp_mask.n_shot + 1):
#             if i_shot == 1 and i_rep == 0:
#                 ksp_pc_res[:, :, :, :, 0] = ksp_pc[:, :, :, :, 0] * samp_mask.get_binary_mask(i_shot=1)
#                 continue

#             measurement_img = ifftnd(ksp_pc[:, :, :, :, i_rep] * samp_mask.get_binary_mask(i_shot=i_shot), axes=(0, 2, 3))
#             global_phase_shift = estimate_phase_shift(reference_img, measurement_img)
#             corrected_img = apply_phase_correction(measurement_img, global_phase_shift)
#             ksp_pc_res[:, :, :, :, i_rep] = fftnd(corrected_img, axes=(0, 2, 3)) * samp_mask.get_binary_mask(i_shot=i_shot)

#             # Plot the phase shift estimation
#             magnitude_reference = np.abs(reference_img)
#             magnitude_measurement = np.abs(measurement_img)
#             threshold = 0.1 * magnitude_reference.max()
#             mask = (magnitude_reference > threshold) & (magnitude_measurement > threshold)
#             phase_diff = np.angle(measurement_img)*mask - np.angle(corrected_img)*mask
#             im = axs[i_rep][i_shot - 1].imshow(phase_diff[:, 0, :, 32], cmap='jet')
#             if global_phase_shift < 0:
#                 global_phase_shift = global_phase_shift + np.pi 
#             if global_phase_shift > np.pi:
#                 global_phase_shift = global_phase_shift - np.pi
#             axs[i_rep][i_shot - 1].set_title(f"{global_phase_shift:.2f}")
#             axs[i_rep][i_shot - 1].axis('off')
#             fig.colorbar(im, ax=axs[i_rep, i_shot - 1])


#     plt.show()  
#     return ksp_pc_res



# def inter_shot_phase_shift_correction(ksp_pc: np.ndarray, samp_mask: SampMask) -> np.ndarray:
#     print("Performing inter-shot phase shift correction...")

#     ksp_pc_res = np.zeros_like(ksp_pc, dtype=ksp_pc.dtype)

#     # Plot the phase shift estimation
#     fig, axs = plt.subplots(nrows=2, ncols=ksp_pc.shape[-1])

#     reference_img = ifftnd(ksp_pc[:, :, :, :, 0], axes=(0, 2, 3))
#     for i_rep in range(ksp_pc.shape[-1]):
#         if i_rep == 0:
#             ksp_pc_res[:, :, :, :, 0] = ksp_pc[:, :, :, :, 0]
#             continue

#         measurement_img = ifftnd(ksp_pc[:, :, :, :, i_rep], axes=(0, 2, 3))
#         global_phase_shift = estimate_phase_shift(reference_img, measurement_img)
#         corrected_img = apply_phase_correction(measurement_img, global_phase_shift)
#         ksp_pc_res[:, :, :, :, i_rep] = fftnd(corrected_img, axes=(0, 2, 3))

#         # Plot the phase shift estimation
#         magnitude_reference = np.abs(reference_img)
#         magnitude_measurement = np.abs(measurement_img)
#         threshold = 0.1 * magnitude_reference.max()
#         mask = (magnitude_reference > threshold) & (magnitude_measurement > threshold)
#         phase_diff = np.angle(measurement_img)*mask - np.angle(corrected_img)*mask
#         im = axs[0][i_rep].imshow(phase_diff[:, 0, :, 32], cmap='jet')
#         im = axs[1][i_rep].imshow(mask[:, 0, :, 32], cmap='jet')
#         if global_phase_shift < 0:
#             global_phase_shift = global_phase_shift + np.pi 
#         if global_phase_shift > np.pi:
#             global_phase_shift = global_phase_shift - np.pi
#         axs[0][i_rep].set_title(f"{global_phase_shift:.2f}")
#         axs[0][i_rep].axis('off')
#         fig.colorbar(im, ax=axs[0, i_rep])


#     plt.show()  
#     return ksp_pc_res


def inter_shot_phase_shift_correction(ksp_pc: np.ndarray, samp_mask: SampMask, t=0, flag_plot=False) -> np.ndarray:
    print("Performing inter-shot phase shift correction...")

    ksp_pc_res = np.zeros_like(ksp_pc, dtype=ksp_pc.dtype)
    ksp_pc_res[:, :, :, :, 0] = ksp_pc[:, :, :, :, 0]
    # Plot the phase shift estimation
    if flag_plot:
        fig, axs = plt.subplots(nrows=8, ncols=samp_mask.n_shot)

    for i_rep in range(1, ksp_pc.shape[-1]):
        for i_shot in range(1, samp_mask.n_shot + 1):
            reference_img = ifftnd(ksp_pc[:, :, :, :, 0] * samp_mask.get_binary_mask(i_shot=i_shot), axes=(0, 2, 3))
            measurement_img = ifftnd(ksp_pc[:, :, :, :, i_rep] * samp_mask.get_binary_mask(i_shot=i_shot), axes=(0, 2, 3))

            global_phase_shift = estimate_phase_shift(reference_img, measurement_img)
            corrected_img = apply_phase_correction(measurement_img, global_phase_shift)
            ksp_pc_res[:, :, :, :, i_rep] += fftnd(corrected_img, axes=(0, 2, 3)) * samp_mask.get_binary_mask(i_shot=i_shot)

            if flag_plot:
                # Plot the phase shift estimation
                magnitude_reference = np.abs(reference_img)
                magnitude_measurement = np.abs(measurement_img)
                threshold = 0.1 * magnitude_reference.max()
                mask = (magnitude_reference > threshold) & (magnitude_measurement > threshold)
                phase_reference = np.angle(reference_img)
                phase_measurement = np.angle(measurement_img)
                phase_diff = phase_reference * mask - phase_measurement * mask

                im = axs[0][i_shot - 1].imshow(magnitude_reference[:, 0, :, 32], cmap='grey')
                im = axs[1][i_shot - 1].imshow(phase_reference[:, 0, :, 32], cmap='jet')
                im = axs[2][i_shot - 1].imshow(magnitude_measurement[:, 0, :, 32], cmap='grey')
                im = axs[3][i_shot - 1].imshow(phase_measurement[:, 0, :, 32], cmap='jet')
                im = axs[4][i_shot - 1].imshow(mask[:, 0, :, 32], cmap='grey')
                im = axs[5][i_shot - 1].imshow(phase_diff[:, 0, :, 32], cmap='jet')
                im = axs[6][i_shot - 1].imshow(np.abs(corrected_img)[:, 0, :, 32], cmap='jet')
                im = axs[7][i_shot - 1].imshow(np.angle(corrected_img)[:, 0, :, 32], cmap='jet')

                if global_phase_shift < 0:
                    global_phase_shift = global_phase_shift + np.pi 
                if global_phase_shift > np.pi:
                    global_phase_shift = global_phase_shift - np.pi

                axs[5][i_shot - 1].set_title(f"{global_phase_shift:.2f}")

                for i in range(8):
                    axs[i][i_shot - 1].axis('off')
                    fig.colorbar(im, ax=axs[i, i_shot - 1])

    if flag_plot:
        plt.savefig(f"inter_shot_phase_shift_correction_{t}.png", dpi=600, bbox_inches="tight", transparent=False)
    return ksp_pc_res