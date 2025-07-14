import numpy as np
import sigpy.mri
import os
from fsl.data.image import Image
from multiprocessing import Pool



def simulate_wrapper(kwargs):
    return simulate(ksp=kwargs['ksp'], sampling_mask=kwargs['sampling_mask'], sensitivity_maps=kwargs['sensitivity_maps'], Psi=kwargs['Psi'])


def simulate(ksp, sampling_mask, sensitivity_maps, Psi):
    Nc, Nx, Ny, Nz = ksp.shape
    # 4. Sythesize Gaussian noise
    # 5. Correlate by matrix square root of noise covariance
    noise = np.random.multivariate_normal(np.zeros(Nc), Psi, (Nx, Ny, Nz)).transpose((3, 0, 1, 2)).astype(np.complex128) * 1e-5 #3e-06

    # 6. Add noise to k-space
    noise_ksp = noise + ksp

    # 7. Reconstruct the noisy image
    noisy_acc_image = sigpy.mri.app.SenseRecon(noise_ksp * sampling_mask, sensitivity_maps, lamda=0, show_pbar=True).run()
    noise_full_image = sigpy.mri.app.SenseRecon(noise_ksp, sensitivity_maps, lamda=0, show_pbar=True).run()

    return noisy_acc_image, noise_full_image



def monte_carlo_gfactor(ksp, sensitivity_maps, sampling_mask, Psi, R_factor, num_simulations=50, multiprocessing=10, save_dir='./temp'):
    """
    Monte Carlo method to calculate the g-factor for a parallel MRI system.
    """

    debug = True

    # Image dimensions
    Nc, Nx, Ny, Nz = ksp.shape

    if Psi is None:
        Psi = np.eye(Nc)
    

    noisy_acc_image_replicas = []
    noisy_full_image_replicas = []


    if multiprocessing == 0:

        for i in range(num_simulations):

            noisy_acc_image, noise_full_image = simulate(ksp, sampling_mask, sensitivity_maps, Psi)
            # 8. Add image to stack of image replicas
            noisy_acc_image_replicas.append(noisy_acc_image)
            noisy_full_image_replicas.append(noise_full_image)

    else:
        pool = Pool(processes=multiprocessing)
        args_list = [
            {
                'ksp': ksp,
                'sampling_mask': sampling_mask,
                'sensitivity_maps': sensitivity_maps,
                'Psi': Psi,
            }
            for i in range(num_simulations)
        ]

        results = [pool.apply_async(simulate_wrapper, (args,)) for args in args_list]
        pool.close()
        pool.join()
        for result in results:
            noisy_acc_image, noise_full_image = result.get()
            noisy_acc_image_replicas.append(noisy_acc_image)
            noisy_full_image_replicas.append(noise_full_image)


    # 9. Calculate image noise: Find noise SD of real part through complex image stack per pixel; form noise SD map
    noisy_acc_image_replicas = np.asarray(noisy_acc_image_replicas)     # Nrep, Nx, Ny, Nz
    noisy_acc_image_SD_map = np.std(np.abs(noisy_acc_image_replicas), axis=0)     # Nx, Ny, Nz

    noisy_full_image_replicas = np.asarray(noisy_full_image_replicas)     # Nrep, Nx, Ny, Nz
    noisy_full_image_SD_map = np.std(np.abs(noisy_full_image_replicas), axis=0)     # Nx, Ny, Nz

    g_factor = noisy_acc_image_SD_map / noisy_full_image_SD_map / np.sqrt(R_factor)

    if debug:
        os.makedirs(save_dir, exist_ok=True)
        Image(np.abs(noisy_acc_image_replicas.transpose((1, 2, 3, 0)))).save(os.path.join(save_dir, f'noisy_acc_image_replicas.nii.gz'))
        Image(np.abs(noisy_full_image_replicas.transpose((1, 2, 3, 0)))).save(os.path.join(save_dir, f'noisy_full_image_replicas.nii.gz'))
        Image(np.abs(noisy_acc_image_SD_map)).save(os.path.join(save_dir, f'noisy_acc_image_SD_map.nii.gz'))
        Image(np.abs(noisy_full_image_SD_map)).save(os.path.join(save_dir, f'noisy_full_image_SD_map.nii.gz'))
        Image(g_factor).save(os.path.join(save_dir, f'g_factor.nii.gz'))

    return g_factor


