import skimage
import numpy as np 
import pandas as pd
from scipy.signal import find_peaks

from mrtk.recon.samp_mask import SampMask

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from typing import List, Optional, Literal
from enum import IntEnum


def get_sampling_mask(samp_type: Literal['GRASE3DCAIPI', 'GRASE3D'],
                      **kwargs,
                      ):
    if samp_type == 'GRASE3DCAIPI':
        sampling_mask = GRASE3DCAIPI(**kwargs).get_sampling_mask()
    elif samp_type == 'GRASE3D':
        sampling_mask = GRASE3D(**kwargs).get_sampling_mask()
    else:
        raise NotImplementedError(f"Sampling mask type {samp_type} not implemented")
    return sampling_mask


class VERBOSE(IntEnum):
    NONE = 0
    INFO = 1
    DEBUG = 2


def order_strategy(num, strategy):
    if strategy == 'LINEAR_UP':
        return np.arange(num)
    elif strategy == 'LINEAR_DOWN':
        return np.arange(num)[::-1]
    elif strategy == 'CENTRIC_DOWN':
        a = np.arange(num // 2) + num // 2
        b = np.arange(num // 2)[::-1]
        c = np.array([a, b]).T.flatten()
        return c
    elif strategy == 'CENTRIC_UP':
        b = np.arange(num // 2) + num // 2
        a = np.arange(num // 2)[::-1]
        c = np.array([a, b]).T.flatten()
        return c
    else:
        raise NotImplementedError


def analyse_psf(psf):

    # Find peaks in 2d image
    peaks = skimage.feature.peak_local_max(psf, min_distance=1, threshold_rel=0.01)
    peaks_height = psf[peaks[:, 0], peaks[:, 1]]
    center_peak = peaks[np.argmax(peaks_height)]

    # Find peaks along z and y axis
    peaks_y, properties_y = find_peaks(psf[center_peak[0], :], width=0, height=0.01 * np.max(psf))
    peaks_z, properties_z = find_peaks(psf[:, center_peak[1]], width=0, height=0.01 * np.max(psf))

    assert len(peaks_y) >= 1 and len(peaks_z) >= 1

    # Effective resolution
    main_lobe_y_index = np.argmax(properties_y['peak_heights'])
    main_lobe_y_range = [properties_y['left_ips'][main_lobe_y_index], properties_y['right_ips'][main_lobe_y_index]]
    effective_resolution_y = properties_y['widths'][main_lobe_y_index]

    main_lobe_z_index = np.argmax(properties_z['peak_heights'])
    main_lobe_z_range = [properties_z['left_ips'][main_lobe_z_index], properties_z['right_ips'][main_lobe_z_index]]
    effective_resolution_z = properties_z['widths'][main_lobe_z_index]

    # Specificity
    main_lobe_sum = np.sum(psf[int(np.floor(main_lobe_z_range[0])): int(np.ceil(main_lobe_z_range[1])), int(np.floor(main_lobe_y_range[0])): int(np.ceil(main_lobe_y_range[1]))])
    specificity = main_lobe_sum / np.sum(psf)

    # Largest side lobe height over main lobe height
    if len(peaks_height) > 1:
        highest_side_lobe = sorted(peaks_height)[-2]
    else:
        highest_side_lobe = 0
    second_lobe_ratio = highest_side_lobe / peaks_height[np.argmax(peaks_height)]
    return effective_resolution_z, effective_resolution_y, specificity, second_lobe_ratio, center_peak


class GRASE3D:
    def __init__(self,
                 matrix_size: List[int] = [64, 64, 32],
                 fov: List[float] = [230, 230, 115],
                 turbo_factor : int = 8,
                 epi_factor : int = 64,
                 n_segments : int = 4,
                 verbose: VERBOSE = VERBOSE.INFO,

                 TE: float = 34, 
                 T2: float = 110, 
                 T2_star: float = 66, 
                 echo_spacing: float = 0.5, 
                 off_resonance_feild_map: float = 100,
                 **kwargs,
                 ) -> None:
        
        self.TE = TE
        self.T2 = T2
        self.T2_star = T2_star
        self.echo_spacing = echo_spacing
        self.off_resonance_feild_map = off_resonance_feild_map
        
        self.verbose = verbose
        self.matrix_size = matrix_size
        self.Nx, self.Ny, self.Nz = self.matrix_size
        self.fov = fov
        self.turbo_factor = turbo_factor
        self.epi_factor = epi_factor
        self.n_segments = n_segments
        self.resolution = [self.fov[i] / self.matrix_size[i] for i in range(3)]

        if self.verbose >= VERBOSE.DEBUG:
            print(f"{self.Ny}, {self.Nz}, {self.turbo_factor = }, {self.epi_factor = }, {self.n_segments = }")
        assert self.Ny * self.Nz == self.turbo_factor * self.epi_factor * self.n_segments, f"Check the Ny * Nz == turbo_factor * epi_factor * n_segments"
        self.readout_name = '3DGRASE Centre-out'
        self.readout_table = dict()
        self.init_readout_table()

        self.flag_calc_readout_table_done = False
        self.flag_calc_psf_done = False


    def init_readout_table(self):
        self.readout_table.setdefault('i_line_in_measurement', np.zeros((self.Nz, self.Ny), dtype=np.int16))
        self.readout_table.setdefault('i_segment', np.zeros((self.Nz, self.Ny), dtype=np.int16))
        self.readout_table.setdefault('i_line_in_segment', np.zeros((self.Nz, self.Ny), dtype=np.int16))
        self.readout_table.setdefault('i_subsegment', np.zeros((self.Nz, self.Ny), dtype=np.int16))
        self.readout_table.setdefault('i_line_in_subsegment', np.zeros((self.Nz, self.Ny), dtype=np.int16))
        self.readout_table.setdefault('signal', np.zeros((self.Nz, self.Ny), dtype=np.float32))
        self.readout_table.setdefault('phase_accu', np.zeros((self.Nz, self.Ny), dtype=np.float32))
        self.flag_calc_readout_table_done = False
        self.flag_calc_psf_done = False

    def get_segment_map(self, start_from_1=True, transpose=True, dtype=np.int16):
        if not self.flag_calc_readout_table_done:
            self.calc_readout_table()

        res = self.readout_table['i_segment'].astype(dtype)
        if start_from_1:
            res += 1
        if transpose:
            res = res.transpose([1, 0])
        return res
    

    def get_samp_mask(self):
        if not self.flag_calc_readout_table_done:
            self.calc_readout_table()

        mask = self.get_segment_map(start_from_1=True, transpose=True, dtype=np.int16)
        return SampMask(mask[np.newaxis, np.newaxis, :, :])
    
    def get_sampling_mask(self):
        if not self.flag_calc_readout_table_done:
            self.calc_readout_table()

        mask = self.get_segment_map(start_from_1=True, transpose=True, dtype=np.int16)
        return mask[np.newaxis, np.newaxis, :, :, np.newaxis]
        

    def vis_readout_table(self, key: str = 'signal', cmap='viridis'):
        if not self.flag_calc_readout_table_done:
            self.calc_readout_table()
        plt.figure(figsize=(10, 10))
        plt.imshow(self.readout_table[key], cmap=cmap)
        plt.colorbar()
        plt.title(key)
        plt.show()
    

    def signal_in_segment(self, i_line_in_segment: int,
                               echo_shift: float = 0.0,
                               ):
        T2_prime = 1 / (1 / self.T2_star - 1 / self.T2)

        t = ((i_line_in_segment + echo_shift) / self.epi_factor) * self.TE
        t2_decay = np.exp(-t / self.T2)

        k = np.floor(t / self.TE)
        t2_prime_decay = np.exp(-(0.5 * self.TE - np.abs((t - k * self.TE) - 0.5 * self.TE)) / T2_prime)

        signal = t2_decay * t2_prime_decay
        return t, t2_decay, signal
    

    def phase_accu_in_subsegment(self, i_line_in_subsegment: int, 
                              echo_shift: float = 0.0):
        
        return 2 * np.pi * (echo_shift + i_line_in_subsegment + 1 - self.epi_factor / 2) * self.echo_spacing / 1000 * self.off_resonance_feild_map


    def calc_readout_table(self):
        if self.verbose >= VERBOSE.DEBUG:
            print(f"DEBUG: Using 3D GRASE JW_CENTRIC readout order strategy")

        n_lines_per_segment = self.Ny * self.Nz // self.n_segments
        
        assert self.n_segments % 2 == 0 or self.n_segments==1, f"Check the n_segments should be even number"

        for i_segment in range(self.n_segments):
            partitions_in_segment = order_strategy(self.Nz, 'CENTRIC_UP')[i_segment::self.n_segments]
            if self.verbose >= VERBOSE.DEBUG:
                print(f"DEBUG: Partition order in segment {i_segment}: {partitions_in_segment}")
            
            i_line_in_segment = 0

            for z in partitions_in_segment:
                for y in range(self.Ny):
                    i_line_in_measurement = i_segment * n_lines_per_segment + i_line_in_segment
                    i_subsegment = i_line_in_segment // self.epi_factor
                    i_line_in_subsegment = i_line_in_segment % self.epi_factor

                    self.readout_table['i_line_in_measurement'][z, y] = i_line_in_measurement
                    self.readout_table['i_segment'][z, y] = i_segment
                    self.readout_table['i_line_in_segment'][z, y] = i_line_in_segment
                    self.readout_table['i_subsegment'][z, y] = i_subsegment
                    self.readout_table['i_line_in_subsegment'][z, y] = i_line_in_subsegment
                    self.readout_table['signal'][z, y] = self.signal_in_segment(i_line_in_segment)[2]
                    self.readout_table['phase_accu'][z, y] = self.phase_accu_in_subsegment(i_line_in_subsegment)
                    i_line_in_segment += 1
        self.flag_calc_readout_table_done = True


    def calc_psf(self, type='off_resonance'):
        if not self.flag_calc_readout_table_done:
            self.calc_readout_table()

        assert type in ['on_resonance', 'off_resonance'], f"Check the type should be on_resonance or off_resonance"

        psf_on_resonance = np.fft.fftshift(np.fft.ifftn(self.readout_table['signal'], axes=[0, 1]), axes=[0, 1])

        if type == 'on_resonance':
            psf = np.abs(psf_on_resonance)
        else:
            psf_off_resonance = np.zeros((self.Nz, self.Ny), dtype=np.complex128)

            for z in range(self.Nz):
                for y in range(self.Ny):
                    psf_off_resonance[z, y] += np.fft.fftn(psf_on_resonance * np.exp(1j * self.readout_table['phase_accu'][z, y]), axes=[0, 1])[z, y]

            psf = np.abs(np.fft.ifftn(psf_off_resonance, axes=[0, 1]))
        
        self.flag_calc_psf_done = True
        return psf / np.max(psf)
    

    def display_psf(self, method_name='', figsize=(12, 8)):
        if not self.flag_calc_readout_table_done:
            self.calc_readout_table()
        # assert self.flag_calc_psf_done, 'Run calc_psf() before display_psf()'
        res_dict = dict()
        
        fig = plt.figure(figsize=figsize)
        # fig.suptitle(method_name)

        ax = fig.add_subplot(2, 2, 1)
        ax.set_title(f'Simulated Signal Intensity in K-space\n Considering T2 Decay')
        im = ax.imshow(self.readout_table['signal'], cmap='gray', vmax=1, vmin=0)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')

        ax = fig.add_subplot(2, 2, 2)
        ax.set_title(f'Phase Evolution in the Presence of\n Off-resonance Effects')
        im = ax.imshow(self.readout_table['phase_accu'], cmap='gray')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')

        z = np.linspace(0, self.Nz, self.Nz)
        y = np.linspace(0, self.Ny, self.Ny)
        y, z = np.meshgrid(y, z)
        
        psf_on = self.calc_psf(type='on_resonance')
        ax = fig.add_subplot(2, 2, 3, projection='3d')
        surf = ax.plot_surface(y, z, psf_on, cmap='plasma', alpha=0.7, vmin=0, vmax=1)
        effective_resolution_z, effective_resolution_y, specificity, second_lobe_ratio, center_peak = analyse_psf(psf_on)
        res_dict.setdefault('Method', []).append(method_name)
        res_dict.setdefault('Type', []).append('On-resonance')
        res_dict.setdefault('Effective Resolution z', []).append(effective_resolution_z)
        res_dict.setdefault('Effective Resolution y', []).append(effective_resolution_y)
        res_dict.setdefault('Specificity', []).append(specificity)
        res_dict.setdefault('Second Lobe Ratio', []).append(second_lobe_ratio)
        res_dict.setdefault('Center Peak', []).append(center_peak)
        res_dict.setdefault('Center Shift', []).append(0)
        center_peak_on_resonance = center_peak

        ax.contour(y, z, psf_on, levels=list(range(self.Ny)), zdir='x', offset=0, vmin=0, vmax=1)#, cmap='coolwarm')
        ax.contour(y, z, psf_on, levels=list(range(self.Nz)), zdir='y', offset=self.Nz, vmin=0, vmax=1)#, cmap='coolwarm')
        ax.set(xlim=(0, self.Ny), ylim=(0, self.Nz), zlim=(0, 1), xlabel='y', ylabel='z', zlabel='PSF')      
        # ax.set_title(f'Point Spread Function (On-resonance)\n {effective_resolution_z = :0.2f}\n{effective_resolution_y = :0.2f}\n{specificity = :0.4f}\n{second_lobe_ratio = :0.4f}\n')
        ax.set_title(f'Point Spread Function\n (On-resonance)')

        psf_off = self.calc_psf(type='off_resonance')
        ax = fig.add_subplot(2, 2, 4, projection='3d')
        surf = ax.plot_surface(y, z, psf_off, cmap='plasma', alpha=0.7, vmin=0, vmax=1)
        effective_resolution_z, effective_resolution_y, specificity, second_lobe_ratio, center_peak = analyse_psf(psf_off)
        res_dict.setdefault('Method', []).append(method_name)
        res_dict.setdefault('Type', []).append('Off-resonance')
        res_dict.setdefault('Effective Resolution z', []).append(effective_resolution_z)
        res_dict.setdefault('Effective Resolution y', []).append(effective_resolution_y)
        res_dict.setdefault('Specificity', []).append(specificity)
        res_dict.setdefault('Second Lobe Ratio', []).append(second_lobe_ratio)
        res_dict.setdefault('Center Peak', []).append(center_peak)
        res_dict.setdefault('Center Shift', []).append(np.linalg.norm(center_peak_on_resonance.astype(np.float32) - center_peak.astype(np.float32)))

        ax.contour(y, z, psf_off, zdir='x', levels=list(range(self.Ny)), offset=0, vmin=0, vmax=1)#, cmap='coolwarm')
        ax.contour(y, z, psf_off, zdir='y', levels=list(range(self.Nz)), offset=self.Nz, vmin=0, vmax=1)#, cmap='coolwarm')
        ax.set(xlim=(0, self.Ny), ylim=(0, self.Nz), zlim=(0, 1), xlabel='y', ylabel='z', zlabel='PSF')      
        # ax.set_title(f'Point Spread Function (Off-resonance)\n {effective_resolution_z = :0.2f}\n{effective_resolution_y = :0.2f}\n{specificity = :0.4f}\n{second_lobe_ratio = :0.4f}\n')
        ax.set_title(f'Point Spread Function\n (Off-resonance)')
        plt.show()

        res_df = pd.DataFrame(res_dict)

        for k in [
            'Effective Resolution z',
            'Effective Resolution y',
            'Specificity',
            'Second Lobe Ratio',
            'Center Shift',
        ]:
            res_df[k] = res_df[k].map('{:.4f}'.format)

        res_df.set_index(['Method', 'Type'], inplace=True)
        res_df = res_df.T.style.set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
        res_df.set_properties(**{'text-align': 'center'})

        return res_dict, res_df


    def display_readout_table(self, segments_to_plot=[], show_arrows=False, show_time=True, display_segments_separately=True, y_range=None, z_range=None, show_colorbar=True, figsize_scale=10):

        plt.rcParams.update({'font.size': 10 + 0.2 * figsize_scale}) 
        if not self.flag_calc_readout_table_done:
            self.calc_readout_table()

        if y_range is None:
            y_range = (0, self.Ny)
        else:
            y_range = (max(0, y_range[0]), min(self.Ny, y_range[1]))
        if z_range is None:
            z_range = (0, self.Nz)
        else:
            z_range = (max(0, z_range[0]), min(self.Nz, z_range[1]))

        n_segments_to_plot = len(segments_to_plot)
        if n_segments_to_plot == 0:
            segments_to_plot = list(range(self.n_segments))
            n_segments_to_plot = self.n_segments

        if display_segments_separately:

            w = y_range[1] - y_range[0]
            h = z_range[1] - z_range[0]

            w = figsize_scale * w / h

            fig, axes = plt.subplots(1, n_segments_to_plot, figsize=(w, figsize_scale))
            if n_segments_to_plot == 1:
                axes = [axes]
        else:
            fig, axes = plt.subplots(1, 1, figsize=(15, 10))
            axes = [axes]

        colors = ['red', 'green', 'blue', 'orange', 'purple', 'gold', 'grey', 'cyan']
    
        for s, ax in zip(segments_to_plot, axes):
            ax.set_aspect('equal')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            ax.set_xticks(range(y_range[0] - 1, y_range[1] + 1))
            ax.set_yticks(range(z_range[0] - 1, z_range[1] + 1))
            ax.set_xticklabels(['y' if i == y_range[0] - 1 else '' if i == y_range[1] else f'{i}' for i in range(y_range[0] - 1, y_range[1] + 1)])
            # ax.set_yticklabels(['z   reorder' if i == -1 else '' if i == self.sz else f'{i:2d}         {np.argwhere(np.asarray(self.partition_order) == i)[0][0]:2d}' for i in range(z_range[0], z_range[1])])
            ax.set_yticklabels(['z' if i == z_range[0] - 1 else '' if i == z_range[1] else f'{i}' for i in range(z_range[0] - 1, z_range[1] + 1)])
            ax.set_xlim(y_range[0] - 1, y_range[1])
            ax.set_ylim(z_range[0] - 1, z_range[1])
            ax.tick_params(axis='both', which='both', length=0)
            ax.invert_yaxis()
            if display_segments_separately:
                ax.set_title(f"{self.readout_name}\nShot #{s + 1}")
            else:
                ax.set_title(f"Segment {segments_to_plot}")
        

        if display_segments_separately:
            i_subsegment_min = np.min(self.readout_table['i_line_in_segment'][z_range[0]:z_range[1], y_range[0]:y_range[1]])
            i_subsegment_max = np.max(self.readout_table['i_line_in_segment'][z_range[0]:z_range[1], y_range[0]:y_range[1]])


            for z in range(z_range[0], z_range[1]):
                for y in range(y_range[0], y_range[1]):
                    for s, ax in zip(segments_to_plot, axes):
                        if self.readout_table['i_segment'][z, y] == s:
                            if show_time:
                                alpha = (1 - ((self.readout_table['i_line_in_segment'][z, y] - i_subsegment_min) / (i_subsegment_max - i_subsegment_min))) * 0.8 + 0.2
                            else:
                                alpha = 0.8
                            color = colors[self.readout_table['i_segment'][z, y] % len(colors)]
                            color_curr = color
                            ax.add_patch(plt.Circle((y, z), 0.3, facecolor=color, alpha=alpha, 
                                                    # hatch='/', 
                                                    edgecolor='k', linewidth=1, linestyle='-'))                             
                        else:
                            ax.add_patch(plt.Circle((y, z), 0.3, facecolor='w', edgecolor='k'))
        
        if show_arrows:
            arrow_start = np.zeros((self.n_segments, 2)) + -1
            for s, ax in zip(segments_to_plot, axes):
                for subs in [0, ]:
                    mask_y_z_range = np.zeros((self.Nz, self.Ny), dtype=bool)
                    mask_y_z_range[z_range[0]:z_range[1], y_range[0]:y_range[1]] = True
                    mask_first_subsegment_in_segment = np.logical_and(self.readout_table['i_segment'] == s, self.readout_table['i_subsegment'] == subs)
                    mask_first_subsegment_in_segment = np.logical_and(mask_first_subsegment_in_segment, mask_y_z_range)
                    locations = np.argwhere(mask_first_subsegment_in_segment)
                    i_line_in_subsegment = self.readout_table['i_line_in_subsegment'][np.where(mask_first_subsegment_in_segment)]

                    dtype = [('location', list), ('i_line_in_subsegment', int)]
                    values = [((locations[i, 0], locations[i, 1]), i_line_in_subsegment[i]) for i in range(len(locations))]
                    locations_in_subsegment = np.array(values, dtype=dtype)
                    # np.sort(locations_in_subsegment, order='i_line_in_subsegment')

                    i_old = -1
                    for ((z, y), i) in np.sort(locations_in_subsegment, order='i_line_in_subsegment'):
                        assert i >= i_old
                        lw = 3
                        if arrow_start[s, 0] != -1:
                            ax.annotate('', xy=(y, z), xytext=arrow_start[s], arrowprops=dict(arrowstyle='-|>', lw=lw, mutation_scale=20, color='k'))
                        arrow_start[s] = [y, z]
                        i_old = i
                if not display_segments_separately:
                    break

        if show_colorbar:
            # --------------- 添加 colorbar (仅表示透明度) ---------------
            # 创建从红色到透明的 colormap
            alphas = np.linspace(0.2, 1.0, 256)  # 透明度范围
            colors = np.array([[*mcolors.to_rgb(color_curr), a] for a in alphas])  # 固定红色 (R=1, G=0, B=0)，alpha 变化
            alpha_cmap = mcolors.ListedColormap(colors)

            # 归一化透明度范围
            norm = mcolors.Normalize(vmin=0.2, vmax=1.0)  # 透明度 0.2 - 1.0 映射

            # 创建 ScalarMappable 并添加 colorbar
            sm = plt.cm.ScalarMappable(cmap=alpha_cmap, norm=norm)
            sm.set_array([])  # 让 colorbar 仅显示透明度渐变
            cbar = fig.colorbar(sm, ax=ax, shrink=0.5)
            cbar.ax.yaxis.set_label_position("left")  # 标签左侧
            cbar.set_label("Acquisition Time")  # 设置 colorbar 标签
            cbar.set_ticks([])  # 在 colorbar 的两端
            cbar.ax.text(0.1, 1.1, "Early", ha='center', va='center', transform=cbar.ax.transAxes)  # 在顶部
            cbar.ax.text(0.1, -0.1, "Late", ha='center', va='center', transform=cbar.ax.transAxes)  # 在底部

        plt.tight_layout()
        plt.show()


class GRASE3DCAIPI(GRASE3D):
    def __init__(self, 
                 R_factor: int = 4,
                 Rz: int = 2,
                 Ry: int = 2,
                 Dz: int = 1,
                 flag_echo_shift: bool = True,
                 **kwargs
                 ) -> None:
        
        super().__init__(**kwargs)

        self.R_factor = kwargs.get('R_factor', R_factor)
        self.Rz = kwargs.get('Rz', Rz)
        self.Ry = kwargs.get('Ry', Ry)
        self.readout_name = f'3DGRASE CAIPI Sampling (Rz={self.Rz}, Ry={self.Ry}, Dz={Dz})'
        assert self.R_factor == self.Rz * self.Ry, f"Check the R_factor({self.R_factor}) should be equal to Rz * Ry ({self.Rz} * {self.Ry})"

        self.Dz = kwargs.get('Dz', Dz)
        assert self.Dz < self.Rz, f"Check the Dz should be less than Rz"

        assert self.R_factor == self.n_segments, f"Check the R_factor should be equal to n_segments"
        assert self.turbo_factor * self.Rz == self.Nz, f"Check the turbo_factor * Rz should be equal to Nz"

        self.flag_echo_shift = kwargs.get('flag_echo_shift', flag_echo_shift)

    def calc_readout_table(self):
        if self.verbose >= VERBOSE.DEBUG:
            print(f"DEBUG: Using 3D GRASE CAIPI readout order strategy")
            
        subsegment_order = order_strategy(self.turbo_factor, 'CENTRIC_UP')
        partitions_in_segment = np.array([list(range(i_subsegment * self.Rz, (i_subsegment + 1) * self.Rz)) for i_subsegment in subsegment_order])
        # print(partitions_in_segment)

        for y in range(self.Ny):
            for z in range(self.Nz):
                i_segment = (z - y // self.Ry * self.Dz) % self.Rz + (y % self.Ry) * self.Rz
                
                i_subsegment = np.where(partitions_in_segment == z)[0][0]
                i_line_in_subsegment = y // self.Ry
                i_line_in_segment = i_subsegment * self.epi_factor + i_line_in_subsegment
                i_line_in_measurement = i_segment * self.epi_factor * self.turbo_factor + i_line_in_segment

                if self.flag_echo_shift:
                    echo_shift = i_segment // self.Rz / self.Ry
                else:
                    echo_shift = 0.0

                self.readout_table['i_line_in_measurement'][z, y] = i_line_in_measurement
                self.readout_table['i_segment'][z, y] = i_segment
                self.readout_table['i_line_in_segment'][z, y] = i_line_in_segment
                self.readout_table['i_subsegment'][z, y] = i_subsegment
                self.readout_table['i_line_in_subsegment'][z, y] = i_line_in_subsegment
                self.readout_table['signal'][z, y] = self.signal_in_segment(i_line_in_segment, echo_shift=echo_shift)[2]
                self.readout_table['phase_accu'][z, y] = self.phase_accu_in_subsegment(i_line_in_subsegment, echo_shift=echo_shift)
        self.flag_calc_readout_table_done = True





if __name__ == '__main__':

    # myROInfo = GRASE3D(
    #     turbo_factor=32,
    #     epi_factor=64,
    #     n_segments=1,
    #     verbose=VERBOSE.DEBUG)
    
    # myROInfo = GRASE3D(
    #     turbo_factor=8,
    #     epi_factor=64,
    #     n_segments=4,
    #     verbose=VERBOSE.DEBUG)
    

    # myROInfo.vis_readout_table(key='i_segment')
    # myROInfo.vis_readout_table(key='i_subsegment')
    # myROInfo.vis_readout_table(key='i_line_in_subsegment')
    # myROInfo.vis_readout_table(key='i_line_in_segment')
    # myROInfo.vis_readout_table(key='i_line_in_measurement')
    # myROInfo.vis_readout_table(key='signal')
    # myROInfo.vis_readout_table(key='phase_accu')


    # myROInfo = GRASE3DCAIPI(
    #     turbo_factor=16,
    #     epi_factor=32,
    #     n_segments=4,
    #     verbose=VERBOSE.DEBUG,
    #     )

    myROInfo = GRASE3DCAIPI(
        turbo_factor=8,
        epi_factor=64,
        n_segments=4,
        R_factor=4,
        Rz=4,
        Ry=1,
        Dz=2,
        flag_echo_shift=True,
        verbose=VERBOSE.DEBUG,
        )
    # myROInfo.calc_readout_table()
    # myROInfo.vis_readout_table(key='i_segment')
    # myROInfo.vis_readout_table(key='i_subsegment')
    # myROInfo.vis_readout_table(key='i_line_in_subsegment')
    # myROInfo.vis_readout_table(key='i_line_in_segment')
    # myROInfo.vis_readout_table(key='i_line_in_measurement')
    # myROInfo.vis_readout_table(key='signal')
    # myROInfo.vis_readout_table(key='phase_accu')
    # myROInfo.calc_psf()
    # myROInfo.display_psf()
    myROInfo.display_readout_table()