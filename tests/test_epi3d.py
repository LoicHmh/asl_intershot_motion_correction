import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch
from omegaconf import OmegaConf
from mrtk.prep.epi3d import EPI3D

@pytest.fixture
def mock_cfg():
    """
    Fixture to load a mock configuration from a YAML file for EPI3D.
    """
    yaml_path = Path(__file__).parent / "config_mock.yaml"  # Path to the YAML file
    cfg = OmegaConf.load(yaml_path)
    return cfg

@pytest.fixture
def mock_epi3d(mock_cfg, tmp_path):
    """
    Fixture to create a mock EPI3D instance for testing.
    """
    mock_cfg.output_dir = str(tmp_path)  # Update output directory to a temporary path
    return EPI3D(mock_cfg)

def test_epi3d_initialization(mock_epi3d):
    """
    Test the initialization of the EPI3D class.
    """
    epi3d = mock_epi3d
    assert epi3d.twix_path == Path(mock_epi3d.twix_path), "Twix path mismatch"
    assert epi3d.out_dir.exists(), "Output directory was not created"
    assert epi3d.flag_epi_phase_correction is True, "Flag for EPI phase correction mismatch"

def test_partial_fourier_padding(mock_epi3d):
    """
    Test the partial_fourier_padding method.
    """
    epi3d = mock_epi3d
    epi3d.ksp_ori = np.random.randn(4, 16, 16, 16, 1, 1)  # Mock k-space data
    epi3d.partial_fourier = [0.8, 1]  # Partial Fourier factors
    padded_ksp = epi3d.partial_fourier_padding(method="zero_padding")
    assert padded_ksp.shape[2] > epi3d.ksp_ori.shape[2], "Partial Fourier padding failed"

def test_epi_phase_correction(mock_epi3d):
    """
    Test the epi_phase_correction method.
    """
    epi3d = mock_epi3d
    epi3d.ksp_ori = np.random.randn(4, 16, 16, 16, 1, 1)  # Mock k-space data
    with patch("mrtk.prep.epi3d.epi_phase_correction_3d", return_value=np.random.randn(4, 16, 16, 16, 1, 1)) as mock_correction:
        ksp_pc = epi3d.epi_phase_correction()
        mock_correction.assert_called_once()
        assert ksp_pc.shape == epi3d.ksp_ori.shape, "EPI phase correction failed"

def test_coil_compression(mock_epi3d):
    """
    Test the coil_compression method.
    """
    epi3d = mock_epi3d
    epi3d.ksp_pc = np.random.randn(4, 16, 16, 16, 1, 1)  # Mock phase-corrected k-space data
    with patch("mrtk.prep.epi3d.coil_compression", return_value=(np.random.randn(4, 16, 16, 16, 1, 1), None)) as mock_compression:
        ksp_cc = epi3d.coil_compression()
        mock_compression.assert_called_once()
        assert ksp_cc.shape == epi3d.ksp_pc.shape, "Coil compression failed"

def test_generate_espirit_sensitivity_maps(mock_epi3d):
    """
    Test the generate_espirit_sensitivity_maps method.
    """
    epi3d = mock_epi3d
    epi3d.ksp_cc = np.random.randn(4, 16, 16, 16, 1)  # Mock coil-compressed k-space data
    with patch("sigpy.mri.app.EspiritCalib.run", return_value=np.random.randn(4, 16, 16, 16)) as mock_espirit:
        sensitivity_maps = epi3d.generate_espirit_sensitivity_maps()
        mock_espirit.assert_called_once()
        assert sensitivity_maps.shape == (4, 16, 16, 16), "ESPIRiT sensitivity map generation failed"

def test_remove_temp_files(mock_epi3d, tmp_path):
    """
    Test the remove_temp_files method.
    """
    epi3d = mock_epi3d
    temp_files = ["ksp_ori.mat", "ksp_pc.mat", "ksp_cc.mat"]
    output_sub_dir = tmp_path / "preprocessing"
    output_sub_dir.mkdir(parents=True, exist_ok=True)
    for file in temp_files:
        (output_sub_dir / file).touch()  # Create temp files

    epi3d.out_dir = tmp_path
    epi3d.remove_temp_files()

    for file in temp_files:
        assert not (output_sub_dir / file).exists(), f"Temp file {file} was not deleted"