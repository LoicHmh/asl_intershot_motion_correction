import numpy as np
import os
from fsl.transform.affine import rotMatToAxisAngles, axisAnglesToRotMat
from typing import Optional, List, Literal
from pathlib import Path
import scipy.sparse
import matplotlib.pyplot as plt



class RigidTransform:
    def __init__(self, 
                 mtx: Optional[np.ndarray] = None,
                 par: Optional[List[float]] = None, 
                 is_radian: bool = False,
                 rotation_center: List[float] = (0., 0., 0.), 
                 dtype = np.float64) -> None:
        '''
        RigidTransform class for 3D rigid transformation.
        Args:
            mtx: 4x4 transformation matrix
            par: 6 parameters rx, ry, rz, tx, ty, tz for rotation and translation, rotation in degree
            is_radian: False if in degree
            rotation_center: only when initialized by using par_degree or par_radian, center of rotation
        '''

        self.dtype = dtype

        assert mtx is not None or par is not None, "Either mtx or par must be provided"

        if mtx is None and par is not None:
            rotation_center = np.asarray(rotation_center, dtype=self.dtype)

            assert len(par) == 6, "par should be a list of 6 elements, rx, ry, rz, tx, ty, tz "
            par = np.asarray(par, dtype=self.dtype)
            if not is_radian:
                par = self.degree2radian(par)

            rx, ry, rz, tx, ty, tz = par
            r_mtx = np.eye(4, dtype=self.dtype)
            r_mtx[:3, :3] = np.asarray(axisAnglesToRotMat(xrot=rx, yrot=ry, zrot=rz), dtype=self.dtype)
            t_mtx = np.eye(4, dtype=self.dtype)
            t_mtx[:3, 3] = (np.eye(3) - r_mtx[:3, :3]) @ rotation_center.reshape(3) + np.asarray([tx, ty, tz], dtype=self.dtype)
            self.mtx = np.dot(t_mtx, r_mtx)
        
        else:
            self.mtx = mtx.astype(self.dtype)
            

    def get_par(self,
                type: Literal['degree', 'radian'],
                rotation_center: List[float] = (0., 0., 0.)) -> np.ndarray:

        R_mat = self.mtx[:3, :3]
        t_vec = self.mtx[:3, 3]

        tx, ty, tz = t_vec - (np.eye(3) - R_mat) @ rotation_center
        rx, ry, rz = np.asarray(rotMatToAxisAngles(self.mtx[:3, :3]), dtype=self.dtype)
        par = np.asarray([rx, ry, rz, tx, ty, tz], dtype=self.dtype)
        if type == 'degree':
            return self.radian2degree(par)
        elif type == 'radian':
            return par


    def degree2radian(self, par: List[float] | np.ndarray) -> np.ndarray:
        par[:3] = np.deg2rad(par[:3])
        return par
    

    def radian2degree(self, par: List[float] | np.ndarray) -> np.ndarray:
        par[:3] = np.rad2deg(par[:3])
        return par   


    def inverse(self) -> 'RigidTransform':
        return RigidTransform(mtx=np.linalg.inv(self.mtx))
    

    def rescale(self, scale: List[float]) -> 'RigidTransform':
        scale = np.asarray(scale, dtype=self.dtype)
        S = np.eye(4)
        S[:3, :3] = np.diag(scale)
        S_inv = np.eye(4)
        S_inv[:3, :3] = np.diag(1 / scale)
        return RigidTransform(mtx=np.dot(S_inv, np.dot(self.mtx, S)))
    

    def apply(self, image: np.ndarray, method: Literal['sinc'] = 'sinc', type: Literal['fwd', 'adj', 'inv'] = 'fwd'):
        assert image.ndim == 3
        if method == 'sinc':
            if type == 'fwd':
                sinc_rt = SincRigidTransformOP(rigid_transform=self, img_size=image.shape)
                return sinc_rt.fwd(image)
            elif type == 'adj':
                sinc_rt = SincRigidTransformOP(rigid_transform=self, img_size=image.shape)
                return sinc_rt.adj(image)
            elif type == 'inv':
                sinc_rt = SincRigidTransformOP(rigid_transform=self.inverse(), img_size=image.shape)
                return sinc_rt.fwd(image)

    

class RigidTransformList:
    def __init__(self, 
                 rigid_transforms: List[RigidTransform]) -> None:
        self.rigid_transforms = rigid_transforms

    def __getitem__(self, idx):
        return self.rigid_transforms[idx]

    def __len__(self):
        return len(self.rigid_transforms)
    
    def __iter__(self):
        return iter(self.rigid_transforms)
    
    @staticmethod
    def load_from_flirt(file_path: str | Path, spacing: List[float] = [1., 1., 1.]) -> None:
        file_path = Path(file_path)
        assert file_path.exists(), f"file {file_path} does not exist"
        assert file_path.is_dir(), f"file {file_path} is not a directory"
        rigid_transforms = []
        for mat_file in sorted(os.listdir(file_path)):
            mtx = np.loadtxt(os.path.join(file_path, mat_file))
            rigid_transform = RigidTransform(mtx=mtx).rescale(scale=spacing)
            rigid_transforms.append(rigid_transform)
        return RigidTransformList(rigid_transforms=rigid_transforms)
    
    @staticmethod
    def load_from_npy(file_path: str | Path, spacing: List[float] = [1., 1., 1.]) -> None:
        file_path = Path(file_path)
        assert file_path.exists(), f"file {file_path} does not exist"
        assert file_path.suffix == '.npy', f"file {file_path} is not a npy file"
        rigid_transforms = []
        for mtx in np.load(file_path):
            rigid_transform = RigidTransform(mtx=mtx).rescale(scale=spacing)
            rigid_transforms.append(rigid_transform)
        return RigidTransformList(rigid_transforms=rigid_transforms)

    def save(self, file_path: str | Path) -> None:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path = file_path.with_suffix('.npy')
        mtx_array = [rigid_transform.mtx for rigid_transform in self.rigid_transforms]
        mtx_array = np.asarray(mtx_array).astype(np.float64)
        np.save(file_path, mtx_array)


    def plot(self, 
            title: str = '', 
            save_path: Optional[str | Path] = None, 
            lims: Optional[List[float]] = None,  # rx, ry, rz, tx, ty, tz
            flag_col=False,
            x_ticks=None,
            others: Optional[List['RigidTransformList']] = None,
            title_others: Optional[List[str]] = None,
            style_others: Optional[List[str]] = None,
            ) -> None:
        if flag_col:
            fig, axs = plt.subplots(6, 1, figsize=(6, 10), sharex=True)
        else:
            fig, axs = plt.subplots(3, 2, figsize=(12, 6), sharex=True)
        legend_fontsize = "10"
        styles = ['r-', 'g-', 'b-', 'y-', 'm-', 'c-', 'k-']
        
        # Collect lines and labels for a single combined legend
        lines = []
        labels = []

        motion_params = np.asarray([rigid_transform.get_par('degree') for rigid_transform in self.rigid_transforms])
        
        xs = list(range(len(motion_params)))
        style = styles[0]

        ax_titles = ['Rx (degree)', 'Ry (degree)', 'Rz (degree)', 'Tx (pixel)', 'Ty (pixel)', 'Tz (pixel)']

        for i in range(6):
            if flag_col:
                ax = axs[i]
            else:
                ax = axs[i % 3, i // 3]
            if x_ticks is not None:
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_ticks)
            if i < 5:
                ax.set_xlabel('')
            ax.set_ylabel(ax_titles[i])
            if lims is not None and lims[i]:
                ax.set_ylim(lims[i])
            line, = ax.plot(xs, motion_params[:, i], style, label=f'{title}')
            if i == 0:
                lines.append(line)
                labels.append(title)


        if others is not None:
            for j, other in enumerate(others):
                if title_others is not None:
                    title = title_others[j]
                else:
                    title = f'Other {j+1}'
                
                if style_others is not None:
                    style = style_others[j]
                else:
                    style = styles[(j + 1) % len(styles)]

                motion_params = np.asarray([rigid_transform.get_par('degree') for rigid_transform in other.rigid_transforms])

                for i in range(6):
                    if flag_col:
                        ax = axs[i]
                    else:
                        ax = axs[i % 3, i // 3]
                    if x_ticks is not None:
                        ax.set_xticks(x_ticks)
                        ax.set_xticklabels(x_ticks)
                    if i < 5:
                        ax.set_xlabel('')
                    ax.set_ylabel(ax_titles[i])
                    if lims is not None and lims[i]:
                        ax.set_ylim(lims[i])
                    line, = ax.plot(xs, motion_params[:, i], style, label=f'{title}')
                    if i == 0:
                        lines.append(line)
                        labels.append(title)


        # Add a single legend to the right
        fig.legend(lines, labels, loc='center left', bbox_to_anchor=(0.92, 0.5), fontsize=legend_fontsize)
        fig.tight_layout(rect=[0, 0, 0.88, 1])  # Leave space on the right for the legend

        if save_path is None:
            plt.show()
        else:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        


def rigid_transform_ksp(ksp, sampling_mask, rigid_transform):
    Nc, Nx, Ny, Nz = ksp.shape
    ksp_size = np.asarray([Nx, Ny, Nz], dtype=np.float64)

    coord = np.indices(ksp_size.astype(np.int16)).astype(np.float64) / ksp_size[:, np.newaxis, np.newaxis, np.newaxis].astype(np.float64) - 0.5

    if sampling_mask is None:
        ksp_value = ksp.reshape(Nc, -1)
        coord = coord.reshape((3, -1))
    else:
        keep_idx = np.where(sampling_mask)
        ksp_value = ksp[:, :, keep_idx[0], keep_idx[1]].reshape(Nc, -1)
        coord = coord[:, :, keep_idx[0], keep_idx[1]].reshape((3, -1))

    T = rigid_transform.mtx.astype(np.float64)
    T_im_center_to_rotation_center = np.eye(4, dtype=np.float64)
    T_im_center_to_rotation_center[:3, 3] = 1 * ksp_size // 2
    T_rotation_center_to_im_center = np.eye(4, dtype=np.float64)
    T_rotation_center_to_im_center[:3, 3] = -1 * ksp_size // 2
    T = np.dot(T_rotation_center_to_im_center, np.dot(T, T_im_center_to_rotation_center))

    r_mtx = T[:3, :3]
    coord_rot = np.dot(r_mtx, coord)

    t_mtx = T[:3, 3] * np.asarray([-1, -1, -1])
    # t_mtx = T[:3, 3]
    phase_ramp = np.exp(1j * 2 * np.pi * np.dot(np.dot(t_mtx, r_mtx), coord))
    ksp_value = ksp_value * phase_ramp

    # out_idx = np.where(np.abs(coord_rot).max(axis=0) > 0.5)
    # ksp_value[:, out_idx] = 0
    # coord_rot = np.clip(coord_rot, a_min=-0.5, a_max=0.5)

    in_idx = np.where(np.abs(coord_rot).max(axis=0) <= 0.5)[0]
    ksp_value = ksp_value[:, in_idx]
    coord_rot = coord_rot[:, in_idx]

    return coord_rot, ksp_value, ksp_size.astype(np.int16)


def get_ksp_coord(ksp, sampling_mask, offset=[0, 0, 0]):
    Nc, Nx, Ny, Nz = ksp.shape
    ksp_size = np.asarray([Nx, Ny, Nz])
    coord = np.indices(ksp_size.astype(np.int16)) - (ksp_size / 2).reshape((3, 1, 1, 1)) + 0.5 * np.asarray(offset).reshape((3, 1, 1, 1))

    print(coord.max(axis=(1,2,3)), coord.min(axis=(1,2,3)))

    keep_idx = np.where(sampling_mask)
    ksp_value = ksp[:, :, keep_idx[0], keep_idx[1]]
    coord = coord[:, :, keep_idx[0], keep_idx[1]].transpose((1, 2, 0))
    

    return coord, ksp_value


class SincRigidTransformOP:
    def __init__(self, rigid_transform: RigidTransform, img_size: List[int]) -> None:
        self.img_size = img_size
        self.rigid_transform = rigid_transform

        theta = self.rigid_transform.get_par(type='radian')[:3]
        img_center = np.asarray(img_size) // 2
        t = self.rigid_transform.get_par(type='radian', rotation_center=img_center)[3:]

        # forward factors
        self.t_fwd = -1j * t
        self.shear_factor_1_fwd = 1j * np.tan(theta / 2)
        self.shear_factor_2_fwd = -1j * np.sin(theta)
        self.shear_factor_3_fwd = self.shear_factor_1_fwd

        # backward factors
        self.t_inv = 1j * t
        self.shear_factor_1_inv = -1j * np.tan(theta / 2)
        self.shear_factor_2_inv = 1j * np.sin(theta)
        self.shear_factor_3_inv = self.shear_factor_1_inv
    
    @staticmethod
    def __gen_grid(img_size, return_k_grid, centered):
        if centered:
            grid = [np.arange(-np.floor(img_size[d] / 2), np.ceil(img_size[d] / 2)) for d in range(3)]
        else:
            grid = [np.arange(0, img_size[d]) for d in range(3)]
        grid[0] = grid[0].reshape((-1, 1, 1))
        grid[1] = grid[1].reshape((1, -1, 1))
        grid[2] = grid[2].reshape((1, 1, -1))
        if return_k_grid:
            for d in range(3):
                grid[d] = 2 * np.pi * grid[d] / img_size[d]
        return grid

    def __fftnd(self, image, axis):
        return np.fft.fftshift(np.fft.fftn(image, axes=axis, norm='ortho'), axes=axis)
    
    def __ifftnd(self, kspace, axis):
        return np.fft.ifftn(np.fft.ifftshift(kspace, axes=axis), axes=axis, norm='ortho')

    def __shear_in_k_space(self, image, axis, phase_ramp):
        k_space = self.__fftnd(image, axis)
        k_space_sheared = k_space * phase_ramp
        sheared_image = self.__ifftnd(k_space_sheared, axis)
        return sheared_image
    

    def ___compute_required_padding(self, r_d, k_d, t_d):
        theta = self.img_size[t_d]
        H = self.img_size[k_d]
        W = self.img_size[r_d]

        pad_H = int(np.ceil(2 * H * np.abs(np.tan(theta / 2)) / 2))
        pad_W = int(np.ceil(W * abs(np.sin(theta)) / 2))

        pads = [(0, 0), (0, 0), (0, 0)]
        pads[k_d] = (pad_H, pad_H)
        pads[r_d] = (pad_W, pad_W)

        slices = [slice(0, self.img_size[d]) for d in range(3)]
        slices[k_d] = slice(pad_H, self.img_size[k_d] + pad_H)
        slices[r_d] = slice(pad_W, self.img_size[r_d] + pad_W)
        return pads, slices
    
    def fwd(self, image: np.ndarray) -> np.ndarray:

        # Rotation
        for t_d in [0, 1, 2]:
            k_d = (t_d + 1) % 3
            r_d = (t_d + 2) % 3

            pads, slices = self.___compute_required_padding(r_d, k_d, t_d)
            image = np.pad(image, pads, mode='constant', constant_values=0)
            k_grid = self.__gen_grid(img_size=image.shape, return_k_grid=True, centered=True)
            r_grid = self.__gen_grid(img_size=image.shape, return_k_grid=False, centered=True)

            phase_ramp = np.exp(r_grid[r_d] * k_grid[k_d] * self.shear_factor_1_fwd[t_d])
            image = self.__shear_in_k_space(image, axis=(k_d,), phase_ramp=phase_ramp)

            phase_ramp = np.exp(r_grid[k_d] * k_grid[r_d] * self.shear_factor_2_fwd[t_d])
            image = self.__shear_in_k_space(image, axis=(r_d,), phase_ramp=phase_ramp)

            phase_ramp = np.exp(r_grid[r_d] * k_grid[k_d] * self.shear_factor_3_fwd[t_d])
            image = self.__shear_in_k_space(image, axis=(k_d,), phase_ramp=phase_ramp)

            image = image[*slices]

        # Translation
        pads = [(int(np.ceil(np.abs(self.t_fwd[d]))), int(np.ceil(np.abs(self.t_fwd[d])))) for d in range(3)]
        slices = [slice(pads[d][0], self.img_size[d] + pads[d][0]) for d in range(3)]
        image = np.pad(image, pads, mode='constant', constant_values=0)
        k_grid = self.__gen_grid(img_size=image.shape, return_k_grid=True, centered=True)
        k_space = self.__fftnd(image, axis=(0, 1, 2))
        k_space = k_space * np.exp(k_grid[0] * self.t_fwd[0] + k_grid[1] * self.t_fwd[1] + k_grid[2] * self.t_fwd[2])
        image = self.__ifftnd(k_space, axis=(0, 1, 2))
        image = image[*slices]

        return image
    
    def adj(self, image: np.ndarray) -> np.ndarray:

        # Translation
        pads = [(int(np.ceil(np.abs(self.t_fwd[d]))), int(np.ceil(np.abs(self.t_fwd[d])))) for d in range(3)]
        slices = [slice(pads[d][0], self.img_size[d] + pads[d][0]) for d in range(3)]
        image = np.pad(image, pads, mode='constant', constant_values=0)
        k_grid = self.__gen_grid(img_size=image.shape, return_k_grid=True, centered=True)
        k_space = self.__fftnd(image, axis=(0, 1, 2))
        k_space = k_space * np.exp(k_grid[0] * self.t_inv[0] + k_grid[1] * self.t_inv[1] + k_grid[2] * self.t_inv[2])
        image = self.__ifftnd(k_space, axis=(0, 1, 2))
        image = image[*slices]

        # Rotation
        for t_d in [2, 1, 0]:
            k_d = (t_d + 1) % 3
            r_d = (t_d + 2) % 3

            pads, slices = self.___compute_required_padding(r_d, k_d, t_d)
            image = np.pad(image, pads, mode='constant', constant_values=0)
            k_grid = self.__gen_grid(img_size=image.shape, return_k_grid=True, centered=True)
            r_grid = self.__gen_grid(img_size=image.shape, return_k_grid=False, centered=True)

            phase_ramp = np.exp(r_grid[r_d] * k_grid[k_d] * self.shear_factor_1_inv[t_d])
            image = self.__shear_in_k_space(image, axis=(k_d,), phase_ramp=phase_ramp)

            phase_ramp = np.exp(r_grid[k_d] * k_grid[r_d] * self.shear_factor_2_inv[t_d])
            image = self.__shear_in_k_space(image, axis=(r_d,), phase_ramp=phase_ramp)

            phase_ramp = np.exp(r_grid[r_d] * k_grid[k_d] * self.shear_factor_3_inv[t_d])
            image = self.__shear_in_k_space(image, axis=(k_d,), phase_ramp=phase_ramp)

            image = image[*slices]


        return image
    

    def check(self) -> bool:
        '''
        check <Mx, y> == <x, M^H y>
        '''
        x = np.random.randn(*self.img_size) + 1j * np.random.randn(*self.img_size)
        y = np.random.randn(*self.img_size) + 1j * np.random.randn(*self.img_size)

        lhs = np.vdot(self.fwd(x), y)     # <Mx, y>
        rhs = np.vdot(x, self.adj(y))     # <x, M^H y>
        diff = np.abs(lhs - rhs)
        return diff < 1e-8


def build_trilinear_interpolation_matrix(shape, T):
    """
    构建从目标网格向原始图像进行刚体插值的稀疏矩阵 A。
    
    输入:
        shape: (D, H, W) 图像尺寸
        T: (4, 4) numpy.ndarray，刚体变换矩阵（目标 → 原图）

    输出:
        A: scipy.sparse.coo_matrix, 稀疏插值矩阵，大小为 (N, N)
    """
    D, H, W = shape
    X, Y, Z = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')
    coords = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)  # (N, 3)

    ones = np.ones((coords.shape[0], 1))
    homo_coords = np.concatenate([coords, ones], axis=1)  # (N, 4)
    warped_coords = homo_coords @ T.T  # (N, 4)
    xw, yw, zw = warped_coords[:, 0], warped_coords[:, 1], warped_coords[:, 2]

    x0 = np.floor(xw).astype(int)
    y0 = np.floor(yw).astype(int)
    z0 = np.floor(zw).astype(int)

    dx = xw - x0
    dy = yw - y0
    dz = zw - z0

    valid = (
        (x0 >= 0) & (x0 < W - 1) &
        (y0 >= 0) & (y0 < H - 1) &
        (z0 >= 0) & (z0 < D - 1)
    )

    rows, cols, vals = [], [], []

    for dx_i in [0, 1]:
        for dy_i in [0, 1]:
            for dz_i in [0, 1]:
                w = (
                    (1 - dx if dx_i == 0 else dx) *
                    (1 - dy if dy_i == 0 else dy) *
                    (1 - dz if dz_i == 0 else dz)
                )

                x_idx = x0 + dx_i
                y_idx = y0 + dy_i
                z_idx = z0 + dz_i

                flat_input_idx = (z_idx * H + y_idx) * W + x_idx
                flat_target_idx = np.arange(xw.size)

                mask = valid & (flat_input_idx >= 0) & (flat_input_idx < D*H*W)

                rows.append(flat_target_idx[mask])
                cols.append(flat_input_idx[mask])
                vals.append(w[mask])

    row = np.concatenate(rows)
    col = np.concatenate(cols)
    val = np.concatenate(vals)

    A = scipy.sparse.coo_matrix((val, (row, col)), shape=(D*H*W, D*H*W))
    return A
