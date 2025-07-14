import pytest
import numpy as np
from mrtk.recon.mr3d import MR3D

@pytest.fixture
def mock_mr3d():
    """
    Fixture to create a mock MR3D instance for testing.
    """
    Nc, Nx, Ny, Nz, Nt = 4, 16, 16, 16, 3  # Example dimensions
    ksp = np.random.randn(Nc, Nx, Ny, Nz, Nt) + 1j * np.random.randn(Nc, Nx, Ny, Nz, Nt)  # Random k-space data 
    # sampling_mask = np.ones((1, 1, Ny, Nz, 1), dtype=np.uint8)  # Fully sampled mask
    Ns = 5
    sampling_mask = np.random.randint(0, Ns + 1, (1, 1, Ny, Nz, 1), dtype=np.uint8)  # Random sampling mask
    sensitivity_maps = np.ones((Nc, Nx, Ny, Nz, 1), dtype=np.complex128)  # Uniform sensitivity maps
    return MR3D(ksp=ksp, sampling_mask=sampling_mask, sensitivity_maps=sensitivity_maps, flip=None, roll=[0, 1, 2])

def test_mr3d_initialization(mock_mr3d):
    """
    Test the initialization of the MR3D class.
    """
    mr3d = mock_mr3d
    assert mr3d.ksp.shape == (4, 16, 16, 16, 3), "k-space shape mismatch"
    assert mr3d.sampling_mask.shape == (1, 16, 16, 16, 3), "Sampling mask shape mismatch"
    assert mr3d.sensitivity_maps.shape == (4, 16, 16, 16, 1), "Sensitivity maps shape mismatch"

def test_generate_espirit_sensitivity_maps(mock_mr3d):
    """
    Test the generate_espirit_sensitivity_maps method.
    """
    mr3d = mock_mr3d
    espirit_maps = mr3d.generate_espirit_sensitivity_maps()
    assert espirit_maps.shape == mr3d.sensitivity_maps.shape, "ESPIRiT sensitivity maps shape mismatch"

def test_recon_img(mock_mr3d):
    """
    Test the recon_img method.
    """
    mr3d = mock_mr3d
    img, extra = mr3d.recon_img(recon_method='rss_ifft')
    assert img.shape == (16, 16, 16, 3), "Reconstructed image shape mismatch"

    img, extra = mr3d.recon_img(recon_method='cg_sense')
    assert img.shape == (16, 16, 16, 3), "Reconstructed image shape mismatch"

    img, extra = mr3d.recon_img(recon_method='wavelet')
    assert img.shape == (16, 16, 16, 3), "Reconstructed image shape mismatch"\

    ksp, extra = mr3d.recon_img(recon_method='ksp_check')
    assert ksp.shape == (4, 16, 16, 16, 3), "Reconstructed image shape mismatch"

    img, extra = mr3d.recon_img(recon_method='cg_sense_scipy')
    assert img.shape == (16, 16, 16, 3), "Reconstructed image shape mismatch"

def test_recon_img_external(mock_mr3d):
    """
    Test the recon_img_external method with external inputs.
    """
    mr3d = mock_mr3d
    external_ksp = np.random.randn(4, 16, 16, 16, 3) + 1j * np.random.randn(4, 16, 16, 16, 3)
    img, extra = mr3d.recon_img_external(
        external_ksp=external_ksp,
        recon_method='rss_ifft'
    )
    assert img.shape == (16, 16, 16, 3), "Reconstructed image shape mismatch with external inputs"

    external_sampling_mask = np.ones((1, 1, 16, 16, 3), dtype=np.uint8)
    img, extra = mr3d.recon_img_external(
        external_sampling_mask=external_sampling_mask,
        recon_method='rss_ifft'
    )
    assert img.shape == (16, 16, 16, 3), "Reconstructed image shape mismatch with external inputs"

    mr3d = mock_mr3d
    external_sensitivity_maps = np.ones((4, 16, 16, 16, 1), dtype=np.complex128)
    img, extra = mr3d.recon_img_external(
        external_sensitivity_maps=external_sensitivity_maps,
        recon_method='rss_ifft'
    )
    assert img.shape == (16, 16, 16, 3), "Reconstructed image shape mismatch with external inputs"
    

def test_recon_img_per_shot(mock_mr3d):
    """
    Test the recon_img method.
    """
    mr3d = mock_mr3d
    img, extra = mr3d.recon_img(recon_method='rss_ifft', flag_recon_per_shot=True)
    assert img.shape == (16, 16, 16, 15), "Reconstructed image shape mismatch"

    img, extra = mr3d.recon_img(recon_method='cg_sense', flag_recon_per_shot=True)
    assert img.shape == (16, 16, 16, 15), "Reconstructed image shape mismatch"

    img, extra = mr3d.recon_img(recon_method='cg_sense_scipy', flag_recon_per_shot=True)
    assert img.shape == (16, 16, 16, 15), "Reconstructed image shape mismatch"

    img, extra = mr3d.recon_img(recon_method='wavelet', flag_recon_per_shot=True)
    assert img.shape == (16, 16, 16, 15), "Reconstructed image shape mismatch"

    # img = mr3d.recon_img(method='ksp_check', flag_recon_per_shot=True)
    # assert img.shape == (16, 16, 16, 15), "Reconstructed image shape mismatch"

def test_invalid_sampling_mask():
    """
    Test MR3D initialization with an invalid sampling mask.
    """
    Nc, Nx, Ny, Nz, Nt = 4, 16, 16, 16, 3
    ksp = np.random.randn(Nc, Nx, Ny, Nz, Nt) + 1j * np.random.randn(Nc, Nx, Ny, Nz, Nt)
    invalid_sampling_mask = np.ones((1, 1, 8, 8, 1), dtype=np.uint8)  # Invalid shape
    with pytest.raises(ValueError, match="Sampling mask dimensions do not match k-space data"):
        MR3D(ksp=ksp, sampling_mask=invalid_sampling_mask)

def test_invalid_sensitivity_maps():
    """
    Test MR3D initialization with invalid sensitivity maps.
    """
    Nc, Nx, Ny, Nz, Nt = 4, 16, 16, 16, 3
    ksp = np.random.randn(Nc, Nx, Ny, Nz, Nt) + 1j * np.random.randn(Nc, Nx, Ny, Nz, Nt)
    invalid_sensitivity_maps = np.ones((2, Nx, Ny, Nz, Nt), dtype=np.complex128)  # Invalid shape
    with pytest.raises(ValueError, match="Sensitivity maps dimensions mismatch with k-space data"):
        MR3D(ksp=ksp, sensitivity_maps=invalid_sensitivity_maps)

def test_save_and_load(mock_mr3d, tmp_path):
    """
    Test saving and loading an MR3D instance.
    """
    mr3d = mock_mr3d
    save_path = tmp_path / "mr3d_data.h5"
    mr3d.save(save_path)  
    loaded_mr3d = MR3D.load(save_path)
    assert np.allclose(mr3d.ksp, loaded_mr3d.ksp), "Loaded k-space data mismatch"
    assert np.allclose(mr3d.sampling_mask, loaded_mr3d.sampling_mask), "Loaded sampling mask mismatch"
    assert np.allclose(mr3d.sensitivity_maps, loaded_mr3d.sensitivity_maps), "Loaded sensitivity maps mismatch"
    if mr3d.flip is not None:
        assert np.allclose(mr3d.flip, loaded_mr3d.flip), "Loaded flip mismatch"
    else:
        assert loaded_mr3d.flip is None, "Loaded flip mismatch"

    if mr3d.roll is not None:
        assert np.allclose(mr3d.roll, loaded_mr3d.roll), "Loaded roll mismatch"
    else:
        assert loaded_mr3d.roll is None, "Loaded roll mismatch"


def test_recon_img_pogm_llr(mock_mr3d):
    """
    Test the recon_img method.
    """
    mr3d = mock_mr3d
    img, extra = mr3d.recon_img(recon_method='pogm_llr')
    assert img.shape == (16, 16, 16, 3), "Reconstructed image shape mismatch"