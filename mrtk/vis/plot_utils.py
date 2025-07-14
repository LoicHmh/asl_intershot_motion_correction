import numpy as np
from typing import Literal
from fsl.data.image import Image
# from .plot import gen_info, plot_mosaic
from typing import List
import os
import matplotlib
import matplotlib.pyplot as plt


def gen_info(img, title='', row_title='', col_title='', cmap='gray', vmin=None, vmax=None, invert_yaxis=True, colorbar=False, scale=1, text=None):
    return {
        'img': img,
        'title': title,
        'row_title': row_title,
        'col_title': col_title,
        'cmap': cmap,
        'vmin': vmin,
        'vmax': vmax,
        'invert_yaxis': invert_yaxis,
        'colorbar': colorbar,
        'scale': scale,
        'text': text,
    }


def plot_mosaic(infos, colorbar_end=True, title_only_top_left=True, save_path=None, dpi=300, n_row=None, n_col=None):
    if n_row is None:
        n_row = len(infos)
    if n_col is None:
        n_col = len(infos[0])

    h, w = infos[0][0]['img'].shape
    im_shape = infos[0][0]['img'].shape
    max_hw = max(h, w)
    h, w = 2.3 * h / max_hw, 2.3 * w / max_hw

    fig, axs = plt.subplots(n_row, n_col, figsize=(w * n_col, h * n_row))
    # fig, axs = plt.subplots(n_row, n_col, figsize=(w * n_col, h * n_row), constrained_layout=True)
    # fig.set_constrained_layout_pads(w_pad=0./72., h_pad=0./72.,
    #         hspace=0./72., wspace=0./72.)

    matplotlib.rcParams['axes.labelsize'] = 'large'
    matplotlib.rcParams['axes.labelpad'] = 6.0
    matplotlib.rcParams['figure.titlesize'] = 16
    

    for i_row in range(n_row):
        for i_col in range(n_col):
            if n_row == 1 and n_col == 1:
                ax = axs
            elif n_row == 1:
                ax = axs[i_col]
            elif n_col == 1:
                ax = axs[i_row]
            else:
                ax = axs[i_row][i_col]

            if i_row >= len(infos) or i_col >= len(infos[i_row]):
                info = {'img': None, 'cmap': infos[0][0]['cmap'], 'vmin': infos[0][0]['vmin'], 'vmax': infos[0][0]['vmax']}
            else:
                info = infos[i_row][i_col]

            if info['img'] is not None:
                plot = ax.imshow(info['img'], cmap=info['cmap'], vmin=info['vmin'], vmax=info['vmax'])
            else:
                plot = ax.imshow(np.zeros(im_shape) * info['vmax'], cmap=info['cmap'], vmin=info['vmin'], vmax=info['vmax'])

            if title_only_top_left and i_row == 0:
                ax.set_title(info['col_title'])

            if title_only_top_left and i_col == 0:
                ax.set_ylabel(info['row_title'], va='center', labelpad=20)

            if info['invert_yaxis']:
                ax.invert_yaxis()

            if (colorbar_end and i_col == n_col - 1) or info['colorbar']:
                # # option 1: 
                # cbar = plt.colorbar(plot, ax=ax, shrink=0.8, format='%.0e')

                # option 2:
                cbar = plt.colorbar(plot, ax=ax, shrink=0.8)
                cbar.formatter.set_scientific(False)

                # # option 3:
                # cbar = plt.colorbar(plot, ax=ax, shrink=0.8)
                # cbar.formatter.set_powerlimits((0, 1000))
                # # to get 10^3 instead of 1e3
                # cbar.formatter.set_useMathText(True)

            if info['text']:
                text = f"{info['text']}"
                ax.text(0.95, 0.05, text, transform=ax.transAxes, 
                        fontsize=12, color='white', ha='right', va='bottom', 
                        bbox=dict(facecolor='black', alpha=0.5))
                
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)  # 移除边框

    # plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    # plt.tight_layout()

    if save_path is not None:
        # plt.savefig(save_path, dpi=dpi)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
    else:
        plt.show()


class SingleImage:
    def __init__(self, 
                 img: np.ndarray | None = None, 
                 img_path: str | None = None, 
                 config_dict: dict | None = None, 
                 intensity_scale: float = 1.0,
                 invert_z: bool = False,
                 ) -> None:

        if img is None:
            assert img_path is not None, 'Either img or img_path should be given'
            img = self._load_img(img_path)
        
        self.img_ori = img
        self.config_dict = config_dict
        self.intensity_scale = intensity_scale
        self.img = self.intensity_scale * self.img_ori
        if invert_z:
            self.img = self.img[:, :, ::-1]
            roll = [0, 1, 1]
            self.img = np.roll(self.img, shift=roll, axis=(0, 1, 2))


    def temporal_slicer(self, img: np.ndarray, temporal_slice_idx: int | slice, reduction: Literal['mean', 'none'] = 'mean'):
        if img.ndim == 4:
            img = img[:, :, :, temporal_slice_idx]

            if reduction == 'mean':
                img = np.mean(img, axis=-1)
        return img


    def spatial_slicer(self, img: np.ndarray, slice_idx: int | float, slice_dim: int):
        assert img.ndim == 3, f'Input image should be 3D, {img.ndim}D image is given'
        assert slice_dim < 3, 'Invalid dimension'

        if type(slice_idx) == float and 0 <= slice_idx <= 1:
            # print(f'slice_idx = {slice_idx} ')
            slice_idx = int(np.round(slice_idx * img.shape[slice_dim]))
            # print(f'slice_idx = {slice_idx} shape={img.shape[slice_dim]}')

        if slice_dim == 0:
            return img[slice_idx, :, :]
        elif slice_dim == 1:
            return img[:, slice_idx, :]
        elif slice_dim == 2:
            return img[:, :, slice_idx]
        

    def _load_img(self, img_path: str) -> np.ndarray:
        if os.path.exists(img_path):
            return Image(img_path).data
        else:
            return 0


    def get_img3d(self, temporal_slice_idx: int | slice):
        return self.temporal_slicer(self.img, temporal_slice_idx)


    def get_img2d(self, temporal_slice_idx: int | slice, slice_idx: int, slice_dim: int, flag_transpose=True):
        img3d = self.get_img3d(temporal_slice_idx)
        img2d = self.spatial_slicer(img3d, slice_idx=slice_idx, slice_dim=slice_dim)
        if flag_transpose:
            img2d = img2d.T
        return img2d


    def diff(self, another_img_manager: 'SingleImage', intensity_scale: float = 1.0):
        diff_img = self.img - another_img_manager.img
        return SingleImage(diff_img, intensity_scale=intensity_scale)


    def get_vrange(self, temporal_slice_idx: int | slice, percentile: int = 99):
        img3d = self.get_img3d(temporal_slice_idx)
        img_vmax, img_vmin = np.percentile(img3d, q=percentile), np.percentile(self.img, q=100 - percentile)
        return img_vmin, img_vmax


class TagControlImage(SingleImage):
    def __init__(self, 
                 img: np.ndarray | None = None, 
                 img_path: str | None = None, 
                 config_dict: dict | None = None, 
                 intensity_scale: float = 1.0,
                 flag_tag_first: bool = True,
                 img_type: Literal['Tag', 'Control', 'Perfusion'] = 'Tag',
                 ) -> None:
        super().__init__(img, img_path, config_dict, intensity_scale)

        if img_type in ['Tag', 'Control']:
            start_idx = flag_tag_first ^ (img_type == 'Tag')
            self.img = self.temporal_slicer(self.img, slice(start_idx, None, 2), reduction='none')
        elif img_type == 'Perfusion':

            img_tag = self.temporal_slicer(self.img, slice(0, None, 2), reduction='none')
            img_ctl = self.temporal_slicer(self.img, slice(1, None, 2), reduction='none')

            if not flag_tag_first:
                img_tag, img_ctl = img_ctl, img_tag
                            
            self.img = img_ctl - img_tag

        self.img_type = img_type
        self.flag_tag_first = flag_tag_first

    
    def diff(self, another_img_manager: 'TagControlImage', intensity_scale: float = 1.0):
        diff_img = self.img_ori - another_img_manager.img_ori

        if self.img_type != another_img_manager.img_type:
            print(f'Warning: img_type of two images are different, {self.img_type} and {another_img_manager.img_type}')
        if self.flag_tag_first != another_img_manager.flag_tag_first:
            print(f'Warning: flag_tag_first of two images are different, {self.flag_tag_first} and {another_img_manager.flag_tag_first}')

        return TagControlImage(diff_img, intensity_scale=intensity_scale, flag_tag_first=self.flag_tag_first, img_type=self.img_type)
    

class MosaicImage():
    def __init__(self, n_row: int, n_col: int) -> None:
        self.n_row = n_row
        self.n_col = n_col

        self.row_titles = ['' for _ in range(n_row)]
        self.col_titles = ['' for _ in range(n_col)]

        self.img2ds = [[None for _ in range(n_col)] for _ in range(n_row)]
        self.texts = [[None for _ in range(n_col)] for _ in range(n_row)]
        self.vranges = np.zeros((n_row, n_col, 2))
        self.mosaic_infos = [[None for _ in range(self.n_col)] for _ in range(self.n_row)]


    def set_row_titles(self, row_titles: List[str]) -> None:
        assert len(row_titles) == self.n_row, f'Length of row_titles should be {self.n_row}, {len(row_titles)} is given'
        self.row_titles = row_titles


    def set_col_titles(self, col_titles: List[str]) -> None:
        assert len(col_titles) == self.n_col, f'Length of col_titles should be {self.n_col}, {len(col_titles)} is given'
        self.col_titles = col_titles
    

    def add_img2d(self, img2d: np.ndarray, i_row: int, i_col: int, text: str | None = None, vmin: float | None = None, vmax: float | None = None, col_title: str = '', row_title: str = '') -> None:
        self.img2ds[i_row][i_col] = img2d
        self.texts[i_row][i_col] = text
        if vmin is None:
            vmin = img2d.min()
        if vmax is None:
            vmax = img2d.max()
        self.vranges[i_row, i_col] = [vmin, vmax]

        if self.col_titles[i_col] != '' and self.col_titles[i_col] != col_title:
            print(f'Warning: col_title of {i_col}th column is changed from {self.col_titles[i_col]} to {col_title}')
        self.col_titles[i_col] = col_title
        if self.row_titles[i_row] != '' and self.row_titles[i_row] != row_title:
            print(f'Warning: row_title of {i_row}th row is changed from {self.row_titles[i_row]} to {row_title}')
        self.row_titles[i_row] = row_title


    def add_img_manager(self, img_manager: SingleImage | TagControlImage, i_row: int, i_col: int, text: str | None = None, col_title: str = '', row_title: str = '', temporal_slice_idx: int | slice = slice(0, None, 1), slice_idx: int | slice = 0, slice_dim: int = 0, vmin: float | None = None, vmax: float | None = None) -> None:
        self.add_img2d(
            img2d = img_manager.get_img2d(temporal_slice_idx=temporal_slice_idx, slice_idx=slice_idx, slice_dim=slice_dim),
            i_row=i_row, 
            i_col=i_col,
            text=text,
            vmin=img_manager.get_vrange(temporal_slice_idx)[0] if vmin is None else vmin,
            vmax=img_manager.get_vrange(temporal_slice_idx)[1] if vmax is None else vmax,
            col_title=col_title,
            row_title=row_title
        )


    def update_infos(self) -> None:
        for i_row in range(self.n_row):
            for i_col in range(self.n_col):
                self.mosaic_infos[i_row][i_col] = gen_info(
                    img=self.img2ds[i_row][i_col],
                    row_title=self.row_titles[i_row],
                    col_title=self.col_titles[i_col],
                    vmin=self.vranges[i_row, :, 0].min(),
                    vmax=self.vranges[i_row, :, 1].max(),
                    invert_yaxis=True,
                    colorbar=False,
                    scale=1,
                    text=self.texts[i_row][i_col],
                )


    def plot(self, colorbar_end=False) -> None:
        self.update_infos()
        plot_mosaic(self.mosaic_infos, colorbar_end=colorbar_end, title_only_top_left=True, dpi=300, n_row=self.n_row, n_col=self.n_col)

    def plot_with_diff(self, diff_pairs, colorbar_end=False) -> None:
        self.update_infos()
        for i_row in range(self.n_row):
            for new_col, diff_pair in enumerate(diff_pairs):
                if len(diff_pair) == 2:
                    a_col, b_col = diff_pair
                    intensity_scale = 1
                    col_title=f"diff\n{self.col_titles[a_col]}\nvs{self.col_titles[b_col]}"
                elif len(diff_pair) == 3:
                    a_col, b_col, intensity_scale = diff_pair
                    col_title=f"diff\n{self.col_titles[a_col]}\nvs{self.col_titles[b_col]}"
                elif len(diff_pair) == 4:
                    a_col, b_col, intensity_scale, col_title = diff_pair
                else:
                    raise ValueError(f'Invalid diff_pairs length {len(diff_pair)}')
                self.mosaic_infos[i_row].append(gen_info(
                    img=np.abs(self.img2ds[i_row][a_col] - self.img2ds[i_row][b_col]) * intensity_scale,
                    row_title=self.row_titles[i_row],
                    col_title=col_title,
                    vmin=self.vranges[i_row, :, 0].min(),
                    vmax=self.vranges[i_row, :, 1].max(),
                    invert_yaxis=True,
                    colorbar=False,
                    scale=1,
                    text=f"x{intensity_scale}",
                    )
                )

        plot_mosaic(self.mosaic_infos, colorbar_end=colorbar_end, title_only_top_left=True, dpi=300, n_row=self.n_row, n_col=self.n_col + len(diff_pairs))


    def save_individual(self, fig_dir: str) -> None:
        self.update_infos()
        os.makedirs(fig_dir, exist_ok=True)

        for i_row in range(self.n_row):
            for i_col in range(self.n_col):
                if self.img2ds[i_row][i_col] is not None:
                    fig_name = f'{self.row_titles[i_row]}_{self.col_titles[i_col]}.png'.replace(' ', '_').replace('\n', '_').replace('/', '')
                    plot_mosaic([[self.mosaic_infos[i_row][i_col]]], colorbar_end=False, title_only_top_left=False, dpi=300, n_row=1, n_col=1, save_path=os.path.join(fig_dir, fig_name))