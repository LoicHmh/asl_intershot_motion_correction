import pytest
import numpy as np
from mrtk.math.rigid_transform import RigidTransform

@pytest.fixture
def mock_rigid_transform():
    """
    Fixture to create a mock RigidTransform instance.
    """
    par_degree = [30, 45, 60, 10, 20, 30]  # Rotation (degrees) and translation
    return RigidTransform(par_degree=par_degree)

def test_rigid_transform_initialization_with_parameters():
    """
    Test initialization of RigidTransform with rotation and translation parameters.
    """
    par_degree = [30, 45, 60, 10, 20, 30]  # Rotation (degrees) and translation
    rt = RigidTransform(par_degree=par_degree)

    # Check that the transformation matrix is created
    assert rt.mtx.shape == (4, 4), "Transformation matrix shape is incorrect"

def test_rigid_transform_initialization_with_matrix():
    """
    Test initialization of RigidTransform with a transformation matrix.
    """
    mtx = np.eye(4)  # Identity matrix
    rt = RigidTransform(mtx=mtx)

    # Check that the transformation matrix is correctly set
    assert np.allclose(rt.mtx, mtx), "Transformation matrix is not correctly set"

def test_get_par_degree(mock_rigid_transform):
    """
    Test the get_par method to retrieve parameters in degrees.
    """
    rt = mock_rigid_transform
    par_degree = rt.get_par(type='degree')

    # Check that the returned parameters are in degrees
    assert len(par_degree) == 6, "Parameter length is incorrect"
    assert np.isclose(par_degree[0], 30, atol=1e-2), "Rotation parameter (rx) is incorrect"
    assert np.isclose(par_degree[3], 10, atol=1e-2), "Translation parameter (tx) is incorrect"

def test_get_par_radian(mock_rigid_transform):
    """
    Test the get_par method to retrieve parameters in radians.
    """
    rt = mock_rigid_transform
    par_radian = rt.get_par(type='radian')

    # Check that the returned parameters are in radians
    assert len(par_radian) == 6, "Parameter length is incorrect"
    assert np.isclose(par_radian[0], np.radians(30), atol=1e-2), "Rotation parameter (rx) is incorrect"
    assert np.isclose(par_radian[3], 10, atol=1e-2), "Translation parameter (tx) is incorrect"

def test_change_rotation_center(mock_rigid_transform):
    """
    Test changing the rotation center of the RigidTransform.
    """
    rt = mock_rigid_transform
    new_center = [5, 5, 5]
    rt_new = rt.change_rotation_center(new_center)

    # Check that the new transformation matrix is different
    assert not np.allclose(rt.mtx, rt_new.mtx), "Rotation center change did not affect the transformation matrix"

def test_inverse_transform(mock_rigid_transform):
    """
    Test the inverse method of RigidTransform.
    """
    rt = mock_rigid_transform
    rt_inv = rt.inverse()

    # Check that the inverse transformation matrix is correct
    identity = np.dot(rt.mtx, rt_inv.mtx)
    assert np.allclose(identity, np.eye(4), atol=1e-5), "Inverse transformation is incorrect"

def test_degree_to_radian_conversion(mock_rigid_transform):
    """
    Test the degree2radian method.
    """
    rt = mock_rigid_transform
    par_degree = [30, 45, 60, 10, 20, 30]
    par_radian = rt.degree2radian(par_degree)

    # Check that the conversion is correct
    assert np.isclose(par_radian[0], np.radians(30), atol=1e-5), "Degree to radian conversion is incorrect"

def test_radian_to_degree_conversion(mock_rigid_transform):
    """
    Test the radian2degree method.
    """
    rt = mock_rigid_transform
    par_radian = [np.radians(30), np.radians(45), np.radians(60), 10, 20, 30]
    par_degree = rt.radian2degree(par_radian)

    # Check that the conversion is correct
    assert np.isclose(par_degree[0], 30, atol=1e-5), "Radian to degree conversion is incorrect"

def test_invalid_initialization():
    """
    Test that initializing RigidTransform without parameters raises an error.
    """
    with pytest.raises(AssertionError, match=""):
        RigidTransform()

def test_change_rotation_center_identity():
    """
    Test changing the rotation center with an identity matrix.
    """
    rt = RigidTransform(mtx=np.eye(4))
    new_center = [5, 5, 5]
    rt_new = rt.change_rotation_center(new_center)

    # Check that the new transformation matrix is still valid
    assert rt_new.mtx.shape == (4, 4), "Transformation matrix shape is incorrect after changing rotation center"