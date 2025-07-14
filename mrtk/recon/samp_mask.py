import numpy as np
import numpy.typing as npt
from typing import Optional, Literal
import matplotlib.pyplot as plt

NDArrayBool = npt.NDArray[np.bool_]

class SampMask:
    """A class to handle k-space sampling masks for MRI reconstruction.
    
    This class manages sampling patterns for MR image reconstruction, supporting both
    single-shot and multi-shot acquisitions. The mask indicates which k-space points
    are sampled in each shot.
    
    Attributes:
        mask (NDArray[np.int16]): The sampling mask with shape (1, N_readout, N_phase_encoding, N_partition).
                                 Values indicate shot numbers (0 = not sampled, 1+ = shot number).
        n_shot (int): The total number of shots in the acquisition.
        bin_mask (NDArray[bool]): Binary mask indicating all sampled points (mask > 0).
        bin_masks (Optional[list[NDArray[bool]]]): List of binary masks for each shot.
    """
    
    def __init__(self, mask: npt.NDArray[np.int16]):
        """Initialize the sampling mask.
        
        Args:
            mask (NDArray[np.int16]): Input sampling mask. Can be 3D or 4D.
                                     If 3D, it will be expanded to 4D with leading singleton dimension.
                                     Values should be non-negative integers where:
                                     0 = not sampled
                                     1...n = shot number for sampled points
        """
        self.mask, self.n_shot = self.__init_mask(mask)
        _, Nx, Ny, Nz = self.mask.shape
        self.bin_masks = None
        self.bin_mask = self.mask > 0

    def __init_mask(self, mask: npt.NDArray[np.int16]) -> tuple[npt.NDArray[np.int16], int]:
        """Initialize and validate the sampling mask.
        
        Args:
            mask (NDArray[np.int16]): Input sampling mask.
            
        Returns:
            tuple: (processed_mask, number_of_shots)
            
        Raises:
            AssertionError: If mask dimensions or values are invalid.
        """
        assert mask.ndim == 3 or mask.ndim == 4, 'mask must be 3D or 4D'
        assert np.all(mask >= 0), 'mask must be non-negative'

        if mask.ndim == 3:
            mask = np.expand_dims(mask, axis=0)
        
        return mask, np.max(mask)

    def __init_bin_mask(self) -> list[NDArrayBool]:
        """Initialize binary masks for each shot.
        
        Returns:
            list[NDArray[bool]]: List of binary masks, one per shot (including shot 0).
        """
        bin_masks = []
        for i_shot in range(0, self.n_shot + 1):
            bin_masks.append(self.mask == i_shot)
        return bin_masks

    def get_binary_mask(self, i_shot: Optional[int] = None) -> NDArrayBool:
        """Get binary mask for all sampled points or for a specific shot.
        
        Args:
            i_shot (Optional[int]): Shot number to get mask for.
                                   If None, returns mask for all sampled points.
                                   If int, returns mask for specific shot (0 = not sampled).
        
        Returns:
            NDArray[bool]: Binary mask indicating sampled points.
        """
        if i_shot is None:
            return self.bin_mask
        else:
            if self.bin_masks is None:
                self.bin_masks = self.__init_bin_mask()
            return self.bin_masks[i_shot]
        
    def visualize(self, view: Literal['yz', 'xy', 'xz'] = 'yz', shot: Optional[int] = None):
        """Visualize the sampling mask in specified view.
        
        Args:
            view (str): Viewing plane: 'yz' (default), 'xy', or 'xz'
            shot (Optional[int]): Specific shot to visualize. If None, shows all shots
                                with different colors.
        
        Returns:
            tuple: (figure, axis) matplotlib objects
        """
        if shot is not None:
            assert shot <= self.n_shot and shot >= 0, 'shot number out of range'

        mask_to_plot = self.mask[0]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if view == 'yz':
            data = mask_to_plot[mask_to_plot.shape[0]//2, :, :]
            xlabel, ylabel = 'Phase Encoding', 'Partition'
        elif view == 'xy':
            data = mask_to_plot[:, :, mask_to_plot.shape[2]//2]
            xlabel, ylabel = 'Readout', 'Phase Encoding'
        elif view == 'xz':
            data = mask_to_plot[:, mask_to_plot.shape[1]//2, :]
            xlabel, ylabel = 'Readout', 'Partition'
        else:
            raise ValueError("view must be 'yz', 'xy', or 'xz'")
        
        # Create denser grid for ticks but only show a subset
        ax.set_xticks(np.arange(-0.5, data.shape[0], 1), minor=True)  # Dense ticks
        ax.set_yticks(np.arange(-0.5, data.shape[1], 1), minor=True)  # Dense ticks
        ax.grid(True, which='minor', color='gray', linestyle='-', linewidth=0.2, alpha=0.2)

        ax.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.set_axisbelow(True)  # Place grid behind data
        if shot is not None:
            if shot > 0:
                im = ax.imshow(data.T == shot, cmap='binary')
                title = f'Shot {shot}' 
            else:
                im = ax.imshow(data.T == 0, cmap='binary')
                title = f'Unsampled' 
        else:
            im = ax.imshow(data.T, cmap='viridis')
            title = 'All Shots'
            plt.colorbar(im, ax=ax, label='Shot Number', shrink=0.5)
        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        return fig, ax

class UnderSampMask(SampMask):
    """A class to generate undersampled k-space masks with specified acceleration rates.
    
    This class creates sampling patterns with regular undersampling in phase encoding (y)
    and partition encoding (z) directions. Each sampled point is assigned a shot number
    based on its position.
    """
    
    def __init__(self, shape: tuple[int, int, int], Ry: float, Rz: float):
        """Initialize and generate an undersampled mask.
        
        Args:
            shape (tuple[int, int, int]): Shape of the mask (N_readout, N_phase_encoding, N_partition)
            Ry (float): Acceleration rate in phase encoding (y) direction
            Rz (float): Acceleration rate in partition encoding (z) direction
        
        Raises:
            ValueError: If acceleration rates are less than 1 or greater than dimension size
        """
        Nx, Ny, Nz = shape
        
        # Validate acceleration rates
        if Ry < 1 or Ry > Ny or Rz < 1 or Rz > Nz:
            raise ValueError("Acceleration rates must be between 1 and dimension size")
            
        # Calculate sampling steps
        step_y = int(np.ceil(Ry))
        step_z = int(np.ceil(Rz))
        
        # Generate mask with regular undersampling
        mask = np.zeros((Nx, Ny, Nz), dtype=np.int16)
        
        # Assign interleaved pattern based on position
        for y in range(Ny):
            for z in range(Nz):
                shot = 1 + (y % step_y) + (z % step_z) * step_y
                mask[:, y, z] = shot
                    
        super().__init__(mask)
        
class CAIPISampMask(SampMask):
    """A class to generate CAIPIRINHA sampling masks.
    
    This class creates k-space sampling patterns following the CAIPIRINHA technique,
    which introduces controlled aliasing by shifting the sampling pattern between
    different partition encoding positions. This helps improve parallel imaging
    reconstruction quality.
    
    The sampling pattern is defined by:
    - Ry: Acceleration factor in phase encoding direction
    - Rz: Acceleration factor in partition encoding direction
    - Dz: CAIPI shift pattern in phase encoding direction
    """
    
    def __init__(self, shape: tuple[int, int, int], Ry: float, Rz: float, Dz: int):
        """Initialize and generate a CAIPIRINHA sampling mask.
        
        Args:
            shape (tuple[int, int, int]): Shape of the mask (N_readout, N_phase_encoding, N_partition)
            Ry (float): Acceleration rate in phase encoding (y) direction
            Rz (float): Acceleration rate in partition encoding (z) direction
            Dz (int): CAIPI shift pattern in partition encoding (z) direction
        
        Raises:
            ValueError: If acceleration rates are invalid or Dz is incompatible with Ry
        """
        Nx, Ny, Nz = shape
        
        # Validate acceleration rates
        if Ry < 1 or Ry > Ny or Rz < 1 or Rz > Nz:
            raise ValueError("Acceleration rates must be between 1 and dimension size")
        
        # Validate CAIPI shift
        if Dz >= Rz:
            raise ValueError("CAIPI shift (Dz) must be less than Rz")
            
        R = Ry * Rz
        mask = np.zeros((Nx, Ny, Nz), dtype=np.int16)

        for y in range(Ny):
            for z in range(Nz):
                mask[:, y, z] = 1 + y % Ry + (z - y // Ry * Dz) % Rz * Ry

            
        super().__init__(mask)
        