import numpy as np


def gaussian_weight(n_dims, sigma, size, spacing):
    """
    Generates a normalized d-dimensional Gaussian weight with support for non-isotropic spacing.

    Parameters:
    - n_dims (int): Number of dimensions.
    - sigma (float or list of float): Standard deviation(s) of the Gaussian.
    - size (list of int): Number of points in each dimension.
    - spacing (list of float): Physical spacing between points in each dimension.

    Returns:
    - np.ndarray: Normalized d-dimensional Gaussian weight.
    """
    if len(size) != n_dims or len(spacing) != n_dims:
        raise ValueError("'size' and 'spacing' must have the same length as 'n_dims'.")

    # Ensure sigma is a list if a single value is provided
    if isinstance(sigma, (int, float)):
        sigma = [sigma] * n_dims
    elif len(sigma) != n_dims:
        raise ValueError("'sigma' must be a single value or a list of length 'n_dims'.")

    # Create a coordinate grid with non-isotropic spacing
    coords = [np.linspace(-3 * s, 3 * s, sz) * sp
            for s, sz, sp in zip(sigma, size, spacing)]
    grid = np.meshgrid(*coords, indexing='ij')

    # Compute the Gaussian function
    gauss = np.exp(-sum((g**2) / (2 * s**2) for g, s in zip(grid, sigma)))

    # Normalize by integrating over the actual spacing
    gauss /= np.sum(gauss * np.prod(spacing))

    return gauss


if __name__ == '__main__':
    gaussian = gaussian_weight(n_dims=2, sigma=[1.0, 2.0], size=[50, 60], spacing=[0.5, 1.0])
    print(gaussian.shape)  # (50, 60)
    print(np.sum(gaussian))  # Should be very close to 1