import pytest
import numpy as np
from mrtk.math.rigid_transform import RigidTransform
from mrtk.recon.mrop import SenseOP, MotionCompensatedSenseOP

@pytest.fixture
def sense_op_fixture():
    """
    Fixture to create a SenseOP instance for testing.
    """
    Nc, Nx, Ny, Nz, Nt = 4, 8, 8, 8, 2  # Example dimensions
    ksp_size = (Nc, Nx, Ny, Nz, Nt)
    sampling_mask = np.ones((1, 1, Ny, Nz, Nt), dtype=np.bool_)  # Fully sampled mask
    sensitivity_maps = np.random.randn(Nc, Nx, Ny, Nz) + 1j * np.random.randn(Nc, Nx, Ny, Nz)  # Random sensitivity maps
    return SenseOP(sampling_mask=sampling_mask, sensitivity_maps=sensitivity_maps, ksp_size=ksp_size)

def test_fwd(sense_op_fixture):
    """
    Test the forward operation (fwd) of SenseOP.
    """
    sense_op = sense_op_fixture
    img = np.random.randn(1, sense_op.Nd * sense_op.Nt) + 1j * np.random.randn(1, sense_op.Nd * sense_op.Nt)
    ksp = sense_op.fwd(img)
    assert ksp.shape == sense_op.flattened_ksp_size, "Forward operation output shape mismatch"

def test_adj(sense_op_fixture):
    """
    Test the adjoint operation (adj) of SenseOP.
    """
    sense_op = sense_op_fixture
    ksp = np.random.randn(*sense_op.flattened_ksp_size) + 1j * np.random.randn(*sense_op.flattened_ksp_size)
    img = sense_op.adj(ksp)
    assert img.shape == (1, sense_op.Nd * sense_op.Nt), "Adjoint operation output shape mismatch"

def test_mtimes2(sense_op_fixture):
    """
    Test the E' * (E * x) operation of SenseOP.
    """
    sense_op = sense_op_fixture
    img = np.random.randn(1, sense_op.Nd * sense_op.Nt) + 1j * np.random.randn(1, sense_op.Nd * sense_op.Nt)
    result = sense_op.mtimes2(img)
    assert result.shape == img.shape, "mtimes2 output shape mismatch"

def test_max_step(sense_op_fixture):
    """
    Test the max_step method of SenseOP.
    """
    sense_op = sense_op_fixture
    step = sense_op.max_step(n=10)
    assert step > 0, "max_step should return a positive value"

def test_run_rss_ifft(sense_op_fixture):
    """
    Test the RSS IFFT reconstruction method.
    """
    sense_op = sense_op_fixture
    ksp = np.random.randn(*sense_op.ksp_size) + 1j * np.random.randn(*sense_op.ksp_size)
    img = sense_op.run_rss_ifft(ksp)
    assert img.shape == (sense_op.Nx, sense_op.Ny, sense_op.Nz, sense_op.Nt), "RSS IFFT output shape mismatch"

def test_run_cg_sense_l2(sense_op_fixture):
    """
    Test the CG-SENSE reconstruction method with L2 regularization.
    """
    sense_op = sense_op_fixture
    ksp = np.random.randn(*sense_op.ksp_size) + 1j * np.random.randn(*sense_op.ksp_size)
    lambda_l2 = 0.01
    img = sense_op.run_cg_sense_l2(ksp, lambda_l2=lambda_l2, maxiter=10, atol=1e-6)
    assert img.shape == (sense_op.Nx, sense_op.Ny, sense_op.Nz, sense_op.Nt), "CG-SENSE output shape mismatch"


@pytest.fixture
def mc_sense_op_fixture():
    """
    Fixture to create a MotionCompensatedSenseOP instance for testing.
    """
    Nc, Nx, Ny, Nz, Nt = 4, 8, 8, 8, 2  # Example dimensions
    ksp_size = (Nc, Nx, Ny, Nz, Nt)
    sampling_mask = np.ones((1, 1, Ny, Nz, Nt), dtype=np.bool_)  # Fully sampled mask
    sensitivity_maps = np.random.randn(Nc, Nx, Ny, Nz) + 1j * np.random.randn(Nc, Nx, Ny, Nz)  # Random sensitivity maps
    rigid_transforms = [
        RigidTransform(par=[1, 2, 3, 4, 5, 6], is_radian=False),
        RigidTransform(par=[-2, -3, -4, -5, -6, -7], is_radian=False),
    ]
    return MotionCompensatedSenseOP(sampling_mask=sampling_mask, sensitivity_maps=sensitivity_maps, rigid_transforms=rigid_transforms, ksp_size=ksp_size)

def test_mcsense_fwd(mc_sense_op_fixture):
    """
    Test the forward operation (fwd) of MotionCompensatedSenseOP.
    """
    mcsense_op = mc_sense_op_fixture
    img = np.random.randn(1, mcsense_op.Nd * 1) + 1j * np.random.randn(1, mcsense_op.Nd * 1)
    ksp = mcsense_op.fwd(img)
    assert ksp.shape == mcsense_op.flattened_ksp_size, "Forward operation output shape mismatch"

def test_mcsense_adj(mc_sense_op_fixture):
    """
    Test the adjoint operation (adj) of MotionCompensatedSenseOP.
    """
    mcsense_op = mc_sense_op_fixture
    ksp = np.random.randn(mcsense_op.Nc, mcsense_op.Nd * mcsense_op.Nt) + 1j * np.random.randn(mcsense_op.Nc, mcsense_op.Nd * mcsense_op.Nt)
    img = mcsense_op.adj(ksp)
    assert img.shape == (1, mcsense_op.Nd), "Adjoint operation output shape mismatch"

def test_mcsense_run_update_x_cg(mc_sense_op_fixture):
    """
    Test the rigid transform solve_x_cg operation of MotionCompensatedSenseOP.
    """
    mcsense_op = mc_sense_op_fixture
    ksp = np.random.randn(*mcsense_op.ksp_size) + 1j * np.random.randn(*mcsense_op.ksp_size)
    img = mcsense_op.update_x_cg(ksp, maxiter=10, atol=1e-6)
    assert img.shape == (mcsense_op.Nx, mcsense_op.Ny, mcsense_op.Nz, 1), "run_cg_sense_l2 operation output shape mismatch"
