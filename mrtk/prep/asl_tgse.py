import numpy as np
import scipy
import mapvbvd
from mrtk.prep.epi3d import EPI3D
from typing import Literal

class ASL_GRASE3D(EPI3D):
    """
    ASL TGSE (Turbo Gradient Spin Echo) 数据处理类
    继承自 EPI3D 基类
    """
    
    def __init__(self,
                 protocol: Literal['fme', 'tgse'],
                 **kwargs):
        """
        初始化 ASL_TGSE 类
        
        参数:

            **kwargs: 传递给父类的其他参数
        """
        self.protocol = protocol
        super().__init__(**kwargs)

    def load_twix(self, 
                 flagRemoveOS=True,
                 flagAverageReps=False,
                 flagIgnoreSeg=False,
                 flagSqueeze=True):
        """
        从 TWIX 文件加载数据
        
        返回:
            tuple: (ksp, par_order) k空间数据和并行成像顺序
        """

        ckp_path = self.out_dir / 'ksp_ori.mat'

        if ckp_path.exists():
            print(f'Loading ksp from {ckp_path}')
            checkpoint = scipy.io.loadmat(ckp_path)
            ksp = checkpoint['ksp']
            par_order = checkpoint['par_order']
        else:
            print(f'Loading twix data from {self.twix_path}...')
            twixObj = mapvbvd.mapVBVD(self.twix_path, quiet=True)

            if 'image' in twixObj:
                im_twix = twixObj['image']
            else:
                im_twix = twixObj[1].image
                
            # 设置 TWIX 对象的标志
            im_twix.flagRemoveOS = flagRemoveOS 
            try:
                im_twix.flagRampSampRegrid = self.flagRampSampRegrid
            except:
                print('no flagRampSampRegrid')
            im_twix.flagAverageReps = flagAverageReps
            im_twix.flagIgnoreSeg = flagIgnoreSeg

            # 获取并行成像顺序
            par_order = []
            for i in im_twix.Par:
                if i not in par_order:
                    par_order.append(int(i))
            par_order = np.asarray(par_order)

            # 重塑数据
            ksp = im_twix[:].reshape(im_twix.dataSize)
            
            # 打印数据维度信息
            print(f"Dimension:\t|" + "\t|  ".join(im_twix.dataDims))
            full_size = [f'{d:3d}' for d in im_twix.fullSize]
            print(f"Full size:\t|" + "\t|  ".join(full_size))

            if im_twix.flagRemoveOS:
                data_size = [f'{d:3d}' for d in im_twix.dataSize]
                print(f"Data size:\t|" + "\t|  ".join(data_size))

            if flagSqueeze:
                ksp = ksp.squeeze()
                squeeze_size = [f'{d:3d}' if d>1 else '   ' for d in im_twix.dataSize]
                print(f"Squeeze size:\t|" + "\t|  ".join(squeeze_size))

            if self.flag_save_temp_files:
                scipy.io.savemat(ckp_path, {
                    'ksp': ksp,
                    'par_order': par_order
                })
            
            # 处理 FME 协议的特殊情况
            if self.protocol == 'fme':
                ksp = np.pad(ksp, ((0, 0), (0, 0), (0, 0), (8, 0), (0, 0), (0, 0), (0, 0)))
                n_col, n_cha, n_lin, n_sli, n_rep, n_set, n_seg = ksp.shape
                ksp_ = np.zeros((n_col, n_cha, n_lin, n_sli, n_rep * n_set, n_seg), 
                              dtype=ksp.dtype)
                for i_rep in range(n_rep):
                    for i_set in range(n_set):
                        ksp_[..., i_rep * n_set + i_set, :] = ksp[..., i_rep, i_set, :]
                ksp = ksp_
                par_order = np.asarray(par_order) + 8

        return ksp, par_order

    def load_twix_pc(self,
                    flagRemoveOS=True,
                    flagAverageReps=False,
                    flagIgnoreSeg=False):
        """
        加载相位校正数据
        
        返回:
            ndarray: 相位校正数据
        """
        print(f'Loading twix data (phasecor) from {self.twix_path}')
        twixObj = mapvbvd.mapVBVD(self.twix_path, quiet=True)
        
        if 'phasecor' in twixObj:
            pc_twix = twixObj['phasecor']
        else:
            pc_twix = twixObj[1].phasecor
            
        # 设置相位校正数据的标志
        pc_twix.flagRemoveOS = flagRemoveOS
        try:
            pc_twix.flagRampSampRegrid = self.flagRampSampRegrid
        except:
            print('no flagRampSampRegrid')
        pc_twix.flagSkipToFirstLine = True
        pc_twix.flagAverageReps = flagAverageReps
        pc_twix.flagIgnoreSeg = flagIgnoreSeg

        # 获取相位校正数据
        pc = pc_twix[:].reshape(pc_twix.dataSize)

        # 打印数据维度信息
        print(f"Dimension:\t|" + "\t|  ".join(pc_twix.dataDims))
        full_size = [f'{d:3d}' for d in pc_twix.fullSize]
        print(f"Full size:\t|" + "\t|  ".join(full_size))
        
        if pc_twix.flagRemoveOS:
            data_size = [f'{d:3d}' for d in pc_twix.dataSize]
            print(f"Data size:\t|" + "\t|  ".join(data_size))
            
        squeeze_size = [f'{d:3d}' if d>1 else '   ' for d in pc_twix.dataSize]
        print(f"Squeeze size:\t|" + "\t|  ".join(squeeze_size))
        pc = pc.squeeze()

        # 处理 FME 协议的特殊情况
        if self.protocol == 'fme':
            n_col, n_cha, n_ave, n_rep, n_set, n_seg = pc.shape
            pc_ = np.zeros((n_col, n_cha, n_ave, n_rep * n_set, n_seg), 
                          dtype=pc.dtype)
            for i_rep in range(n_rep):
                for i_set in range(n_set):
                    pc_[..., i_rep * n_set + i_set, :] = pc[..., i_rep, i_set, :]
            pc = pc_

        return pc
