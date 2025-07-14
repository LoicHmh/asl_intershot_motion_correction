import pytest
import numpy as np
from pathlib import Path
from mrtk.utils.io import save_data, load_data, save_nii, add_suffix, save_asl

@pytest.fixture
def tmp_h5_file(tmp_path):
    """
    Fixture to create a temporary .h5 file for testing.
    """
    return tmp_path / "test_data.h5"

@pytest.fixture
def tmp_nii_file(tmp_path):
    """
    Fixture to create a temporary .nii file for testing.
    """
    return tmp_path / "test_image.nii"

@pytest.fixture
def sample_data():
    """
    Fixture to provide sample data for testing.
    """
    return {
        "array": np.random.randn(10, 10),
        "string": "test_string",
        "none_value": None,
        "list": [1, 2, 3],
        "bool": True,
        "path": Path("test_path"),
        "complex_array": np.random.randn(10, 10) + 1j * np.random.randn(10, 10),
    }

def test_save_and_load_data(tmp_h5_file, sample_data):
    """
    Test the save_data and load_data functions.
    """
    save_data(tmp_h5_file, sample_data)
    loaded_data = load_data(tmp_h5_file)

    # Check that the loaded data matches the original data
    assert np.allclose(loaded_data["array"], sample_data["array"]), "Array data mismatch"
    assert loaded_data["string"] == sample_data["string"], "String data mismatch"
    assert loaded_data["none_value"] is None, "None value mismatch"
    for l, s in zip(loaded_data["list"], sample_data["list"]):
        assert l == s, "List data mismatch"
    assert loaded_data["bool"] == sample_data["bool"], "Boolean data mismatch"
    assert loaded_data["path"] == str(sample_data["path"]), "Path data mismatch"
    assert np.allclose(loaded_data["complex_array"].real, sample_data["complex_array"].real), "Complex array real part mismatch"
    assert np.allclose(loaded_data["complex_array"].imag, sample_data["complex_array"].imag), "Complex array imaginary part mismatch"


def test_save_data_invalid_type(tmp_h5_file):
    """
    Test save_data with an unsupported data type.
    """
    invalid_data = {"unsupported": 123.456}  # Float is not supported
    with pytest.raises(TypeError, match="Unsupported data type"):
        save_data(tmp_h5_file, invalid_data)

def test_save_nii(tmp_nii_file):
    """
    Test the save_nii function.
    """
    img = np.random.randn(10, 10, 10)
    save_nii(img, tmp_nii_file)

    # Check that the file was created
    assert tmp_nii_file.exists(), "NIfTI file was not created"

def test_save_nii_complex(tmp_nii_file):
    """
    Test the save_nii function with complex data.
    """
    img = np.random.randn(10, 10, 10) + 1j * np.random.randn(10, 10, 10)
    save_nii(img, tmp_nii_file)

    # Check that the file was created
    assert tmp_nii_file.exists(), "NIfTI file was not created"

def test_add_suffix():
    """
    Test the add_suffix function.
    """
    path = Path("test_file.nii")
    new_path = add_suffix(path, "_suffix")
    assert new_path.name == "test_file_suffix.nii", "Suffix was not added correctly"

    path = Path("test_file.nii.gz")
    new_path = add_suffix(path, "_suffix")
    assert new_path.name == "test_file_suffix.nii.gz", "Suffix was not added correctly"

def test_save_asl(tmp_nii_file):
    """
    Test the save_asl function.
    """
    img4d = np.random.randn(10, 10, 10, 4) + 1j * np.random.randn(10, 10, 10, 4)
    save_asl(img4d, tmp_nii_file, perfusion_subtraction="both")

    # Check that the main file and additional files were created
    assert tmp_nii_file.exists(), "Main ASL file was not created"
    assert tmp_nii_file.with_name("test_image_perf_magn_sub.nii").exists(), "Magnitude subtraction file was not created"
    assert tmp_nii_file.with_name("test_image_perf_cplx_sub.nii").exists(), "Complex subtraction file was not created"