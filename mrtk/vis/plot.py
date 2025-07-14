import numpy as np
from typing import Optional, Literal
from fsl.data.image import Image
import matplotlib.pyplot as plt
import matplotlib
import os

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mrtk.math.fft_utils import roll 

from PIL import Image as PILImage
from pathlib import Path

np2dcm = lambda x: np.abs(roll(x, r=[0, -1, -1])[:, :, ::-1, :]).astype(float)



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


def plot_mosaic(infos, suptitle='', colorbar_end=True, title_only_top_left=True, save_path=None, dpi=300, n_row=None, n_col=None):
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
                plot = ax.imshow(np.ones(im_shape) * info['vmax'], cmap=info['cmap'], vmin=info['vmin'], vmax=info['vmax'])

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

    # plt.suptitle(suptitle)
    # plt.tight_layout()

    if save_path is not None:
        # plt.savefig(save_path, dpi=dpi)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_asl(img_dict: dict[str, str | Image], 
             suptitle: str = '', 
             img_vmax: Optional[int] = None, 
             img_vmin: Optional[int] = None,
             perf_vmax: Optional[int] = None,
             get_slice: Optional[callable] = None,
             special_case_handler: Optional[callable] = None,
             ):

    infos_per = []
    infos_tag = []
    infos_ctl = []
    for (title, img) in img_dict.items():
        if type(img) == str:
            img = Image(img).data
        else:
            img = img.data

        if special_case_handler:
            img = special_case_handler(title, img)

        img_tag = np.abs(img[..., 0::2])
        img_ctl = np.abs(img[..., 1::2])
        img_per = img_ctl - img_tag

        if get_slice:
            img_tag = get_slice(img_tag)
            img_ctl = get_slice(img_ctl)
            img_per = get_slice(img_per)

        infos_per.append(
                gen_info(img_per, 
                        title + ' Perfusion', 
                        row_title='Perfusion',
                        col_title=title,
                        vmin=img_vmin, 
                        vmax=perf_vmax))
        
        infos_tag.append(
                gen_info(img_tag, 
                        title + ' Tag',
                        row_title='Tag',
                        col_title=title,
                        vmin=img_vmin, 
                        vmax=img_vmax))

        infos_ctl.append(
                gen_info(img_ctl, 
                        title + ' Control', 
                        row_title='Control',
                        col_title=title,
                        vmin=img_vmin, 
                        vmax=img_vmax))

    plot_mosaic([infos_tag, infos_ctl, infos_per], suptitle=suptitle)


def plot_moco(
                method_dict: dict[str, dict[str, str | Image]],
                suptitle: str = '', 
                img_vmax: Optional[int] = None, 
                img_vmin: Optional[int] = None,
                # diff_vmax: Optional[int] = None,
                get_slice_temporal: Optional[callable] = None,
                get_slice_spatial: Optional[callable] = None,
                special_case_handler: Optional[callable] = None,
                img_type: str = Literal['tag', 'ctl', 'per'],
                apply_mask: Optional[callable] = None,
                get_corr: Optional[callable] = None,
             ):

    infos = dict()
    key_list = []
    for (second_title, img_dict) in method_dict.items():
        for (title, img) in img_dict.items():
            if title not in key_list:
                key_list.append(title)
            if type(img) == str:
                if os.path.exists(img):
                    img = Image(img).data
                else:
                    img = None
                    infos.setdefault(second_title, []).append(
                        gen_info(img,
                                f'{title}\n{second_title}',
                                row_title=second_title,
                                col_title=title,
                                vmin=img_vmin,
                                vmax=img_vmax))
                    continue
            else:
                img = img.data

            if special_case_handler:
                img = special_case_handler(second_title, img)

            if img_type == 'tag':
                img = np.abs(img[..., 0::2])
            elif img_type == 'ctl':
                img = np.abs(img[..., 1::2])
            elif img_type in ['per', 'tSNR', 'corr']:
                img = np.abs(img[..., 1::2]) - np.abs(img[..., 0::2])
            elif img_type == 'tag_diff':
                img = np.abs(img[..., 0::2]) - np.abs(img[..., 0:1])
            elif img_type == 'ctl_diff':
                img = np.abs(img[..., 1::2]) - np.abs(img[..., 1:2])

            if img_type == 'corr' and get_corr is not None:
                # 这里就得用和ref一样的mask，并且把ref也得传进来进行计算，或许还是放弃在这个plot里做这些计算。
                corr, ssim_value, psnr_value = get_corr(title, second_title, img, get_slice_temporal)

            if get_slice_temporal:
                img = get_slice_temporal(img)

            if img_type == 'tSNR':
                t_snr_3d = np.mean(img, axis=-1) / (np.std(img, axis=-1) + 1e-13)
                if apply_mask:
                    t_snr_3d, mask = apply_mask(title, second_title, t_snr_3d)
                    t_snr_3d = np.sum(t_snr_3d) / np.sum(mask)
                else:
                    t_snr_3d = np.mean(t_snr_3d)

            if img_type == 'tSNR':
                img = np.mean(img, axis=-1) / (np.std(img, axis=-1) + 1e-13)
            else:
                img = img.mean(axis=-1)

            if apply_mask:
                img, _ = apply_mask(title, second_title, img)
            
            if get_slice_spatial:
                img = get_slice_spatial(img)

            if img_type == 'tSNR':
                text = f't-SNR: {t_snr_3d:.2f}'
            elif img_type == 'corr':
                text = f'corr: {corr:.2f}\nssim: {ssim_value:.2f}\npsnr: {psnr_value:.2f}'
            else:
                text = None
        
            infos.setdefault(second_title, []).append(
                gen_info(img.T,
                         f'{title}\n{second_title}',
                         row_title=second_title,
                         col_title=title,
                         vmin=img_vmin,
                         vmax=img_vmax,
                         text=text,
                         ))
            
    # for i, title in enumerate(key_list):
    #     diff = infos['w/ motion'][i]['img'] - infos['moco'][i]['img']
    #     infos.setdefault('diff', []).append(
    #             gen_info(diff,
    #                      f'{title}\ndiff',
    #                      row_title='diff',
    #                      col_title=title,
    #                      vmin=-1 * diff_vmax if diff_vmax else None,
    #                      vmax=diff_vmax))

            
    plot_mosaic([infos[k] for k in method_dict.keys()], suptitle=suptitle)


def plot_motion_par(motion_lists, title_list, ry_lim=None, ty_lim=None):
    plt.figure(figsize=(6, 3))

    if len(motion_lists) == 2:
        mse = np.asarray([motion.get_par('degree') for motion in motion_lists[0]]) - np.asarray([motion.get_par('degree') for motion in motion_lists[1]])
        mse = np.mean(mse ** 2, axis=0)

    for i_par, (motion_list, title) in enumerate(zip(motion_lists, title_list)):

        motion_params = np.asarray([motion.get_par('degree') for motion in motion_list])
        
        xs = list(range(len(motion_params)))
        

        plt.subplot(2, 1, 1)
        plt.xlabel('')
        # plt.ylabel('Rotation (degree)')
        if ry_lim:
            plt.ylim(ry_lim)

        style = '--' if title == 'GT' else '-' 
        if len(motion_lists) <= 4:
            styles = ['--', '-.', ':', '-']
            style == styles[i_par]

        plt.plot(xs, motion_params[:, 0], f'r{style}', label=f'Rx {title}')
        plt.plot(xs, motion_params[:, 1], f'g{style}', label=f'Ry {title}')
        plt.plot(xs, motion_params[:, 2], f'b{style}', label=f'Rz {title}')
        plt.legend(loc="upper left", fontsize="8")
        if len(motion_lists) == 2:
            plt.title(f'MSE rx: {mse[0]:0.4f}, ry: {mse[1]:0.4f}, rz: {mse[2]:0.4f}')

        plt.subplot(2, 1, 2)
        plt.xlabel('')
        # plt.ylabel('Translation (pixel)')
        if ty_lim:
            plt.ylim(ty_lim)

        plt.plot(xs, motion_params[:, 3], f'r{style}', label=f'Tx {title}')
        plt.plot(xs, motion_params[:, 4], f'g{style}', label=f'Ty {title}')
        plt.plot(xs, motion_params[:, 5], f'b{style}', label=f'Tz {title}')
        if len(motion_lists) == 2:
            plt.title(f'MSE tx: {mse[3]:0.4f}, ty: {mse[4]:0.4f}, tz: {mse[5]:0.4f}')
        plt.legend(loc="upper left", fontsize="8")
    plt.tight_layout()
    plt.show()


def plot_single_shot_recons(img_dict: dict[str, str | Image], 
             suptitle: str = '', 
             img_vmax: Optional[int] = None, 
             img_vmin: Optional[int] = None,
            #  perf_vmax: Optional[int] = None,
             get_slice: Optional[callable] = None,
             special_case_handler: Optional[callable] = None,
             ):

    infos_shots = []

    for i_shot in range(4):
        infos = []
        for (title, img) in img_dict.items():
            if type(img) == str:
                img = Image(img).data
            else:
                img = img.data

            img = np2dcm(img)

            if special_case_handler:
                img = special_case_handler(title, img)

            img = np.abs(img[..., i_shot: i_shot + 1])
            
            if get_slice:
                img = get_slice(img)

            infos.append(
                    gen_info(img, 
                            title + '', 
                            row_title=f'Shot No.{i_shot}',
                            col_title=title,
                            vmin=img_vmin, 
                            vmax=img_vmax))
        infos_shots.append(infos)
            

    plot_mosaic(infos_shots, suptitle=suptitle)



def plot_gif(img_path_list, gif_path, duration=100, loop=0):
    imgs = []
    for img_path in img_path_list:
        imgs.append(PILImage.open(img_path))
    imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=duration, loop=loop)

    # # Display the generated GIF
    # from IPython.display import Image as IPyImage
    # IPyImage(gif_filename)




# def plot_motion_params(motion_lists, title_list, ry_lim=None, ty_lim=None, x_ticks=None, save_path=None):
#     plt.figure(figsize=(12, 6))
#     legend_fontsize = "10"

#     if len(motion_lists) == 2:
#         mse = np.asarray([motion.get_par('degree') for motion in motion_lists[0]]) - np.asarray([motion.get_par('degree') for motion in motion_lists[1]])
#         mse = np.mean(mse ** 2, axis=0)

#     styles = ['r-', 'g-', 'b-', 'y-', 'm-', 'c-', 'k-']

#     for i_par, (motion_list, title) in enumerate(zip(motion_lists, title_list)):

#         motion_params = np.asarray([motion.get_par('degree') for motion in motion_list])
        
#         xs = list(range(len(motion_params)))
#         style = styles[i_par]

#         plt.subplot(3, 2, 1)
#         if x_ticks is not None:
#             plt.xticks(x_ticks, labels=x_ticks)
#         plt.xlabel('')
#         # plt.ylabel('Rotation on x-axis (degree)')
#         plt.ylabel('Rx (degree)')
#         if ry_lim:
#             plt.ylim(ry_lim)

#         plt.plot(xs, motion_params[:, 0], style, label=f'{title}')
#         plt.legend(loc="upper left", fontsize=legend_fontsize)

#         plt.subplot(3, 2, 3)
#         if x_ticks is not None:
#             plt.xticks(x_ticks, labels=x_ticks)
#         plt.xlabel('')
#         # plt.ylabel('Rotation on y-axis (degree)')
#         plt.ylabel('Ry (degree)')
#         if ry_lim:
#             plt.ylim(ry_lim)

#         plt.plot(xs, motion_params[:, 1], style, label=f'{title}')
#         plt.legend(loc="upper left", fontsize=legend_fontsize)

#         plt.subplot(3, 2, 5)
#         if x_ticks is not None:
#             plt.xticks(x_ticks, labels=x_ticks)
#         plt.xlabel('')
#         # plt.ylabel('Rotation on z-axis (degree)')
#         plt.ylabel('Rz (degree)')
#         if ry_lim:
#             plt.ylim(ry_lim)

#         plt.plot(xs, motion_params[:, 2], style, label=f'{title}')
#         plt.legend(loc="upper left", fontsize=legend_fontsize)

#         plt.subplot(3, 2, 2)
#         if x_ticks is not None:
#             plt.xticks(x_ticks, labels=x_ticks)
#         plt.xlabel('')
#         # plt.ylabel('Translation on x-axis (pixel)')
#         plt.ylabel('Tx (pixel)')
#         if ty_lim:
#             plt.ylim(ty_lim)

#         plt.plot(xs, motion_params[:, 3], style, label=f'{title}')
#         plt.legend(loc="upper left", fontsize=legend_fontsize)

#         plt.subplot(3, 2, 4)
#         if x_ticks is not None:
#             plt.xticks(x_ticks, labels=x_ticks)
#         plt.xlabel('')
#         # plt.ylabel('Translation on y-axis (pixel)')
#         plt.ylabel('Ty (pixel)')
#         if ty_lim:
#             plt.ylim(ty_lim)

#         plt.plot(xs, motion_params[:, 4], style, label=f'{title}')
#         plt.legend(loc="upper left", fontsize=legend_fontsize)

#         plt.subplot(3, 2, 6)
        
#         if x_ticks is not None:
#             plt.xticks(x_ticks, labels=x_ticks)
#         plt.xlabel('')
#         # plt.ylabel('Translation on z-axis (pixel)')
#         plt.ylabel('Tz (pixel)')
#         if ty_lim:
#             plt.ylim(ty_lim)

#         plt.plot(xs, motion_params[:, 5], style, label=f'{title}')
#         plt.legend(loc="upper left", fontsize=legend_fontsize)

#     plt.tight_layout()

#     if save_path is None:
#         plt.show()
#     else:
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         plt.savefig(save_path)
#         plt.close()



def plot_motion_params(motion_lists, title_list, ry_lim=None, ty_lim=None, x_ticks=None, save_path: Optional[str | Path] = None):
    fig, axs = plt.subplots(3, 2, figsize=(12, 6), sharex=True)
    legend_fontsize = "10"
    styles = ['r-', 'g-', 'b-', 'y-', 'm-', 'c-', 'k-']
    
    # Collect lines and labels for a single combined legend
    lines = []
    labels = []

    if len(motion_lists) == 2:
        mse = np.asarray([motion.get_par('degree') for motion in motion_lists[0]]) - \
              np.asarray([motion.get_par('degree') for motion in motion_lists[1]])
        mse = np.mean(mse ** 2, axis=0)

    for i_par, (motion_list, title) in enumerate(zip(motion_lists, title_list)):
        motion_params = np.asarray([motion.get_par('degree') for motion in motion_list])
        xs = list(range(len(motion_params)))
        style = styles[i_par]

        ax_titles = ['Rx (degree)', 'Ry (degree)', 'Rz (degree)', 'Tx (pixel)', 'Ty (pixel)', 'Tz (pixel)']
        lims = [ry_lim, ry_lim, ry_lim, ty_lim, ty_lim, ty_lim]

        for i in range(6):
            ax = axs[i // 2, i % 2]
            if x_ticks is not None:
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_ticks)
            if i < 5:
                ax.set_xlabel('')
            ax.set_ylabel(ax_titles[i])
            if lims[i]:
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