from mrtk.utils.timer import clock_func
from mrtk.prep.asl_tgse import ASL_GRASE3D
import numpy as np
from mrtk.seq.seq_design import get_sampling_mask
from omegaconf import DictConfig
from pathlib import Path
from mrtk.recon.mr3d import MR3D, ASL
import shutil
from mrtk.utils.io import add_suffix
from fsl.wrappers.flirt import mcflirt
# from mrtk.math.rigid_transform import load_rigid_transforms, save_rigid_transforms
from mrtk.vis.plot import plot_motion_params
from mrtk.math.rigid_transform import RigidTransform, RigidTransformList
from fsl.data.image import Image
from mrtk.recon.mrop import MotionCompensatedSenseOP, SenseOP
import scipy.io


@clock_func('run_preprocess_asl_tgse_from_twix_data')
def run_preprocess_asl_tgse_from_twix_data(cfg: DictConfig, always_run: bool = False) -> None:
    asl_path = Path(cfg.prep_twix.asl_path)
    if not always_run and asl_path.exists():
        print("prep_twix already done.")
        return

    asl_tgse = ASL_GRASE3D(**cfg.prep_twix)
    sampling_mask = get_sampling_mask(**cfg.base_info)
    asl = ASL(ksp=asl_tgse.ksp_cc, 
                sampling_mask=sampling_mask,
                **cfg.base_info)
    asl.save(save_path=asl_path)
    if cfg.prep_twix.flag_clean_temp_files:
        asl_tgse.remove_temp_files()


@clock_func('run_img_recon')
def run_img_recon(cfg: DictConfig, always_run=False):
    save_path = Path(cfg.output_path)

    if not always_run and save_path.exists():
        print("run_img_recon already done.")
        return 
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    asl = ASL.load(cfg.input_path)
    res = asl.recon_img(**cfg)
    img = res.get('recon', None)
    extra = res.get('extra', None)

    if cfg.recon_method == 'ksp_check':
        asl.save_nii(img=np.log(np.abs(img)), save_path=add_suffix(save_path, '_log_abs'))
        asl.save_nii(img=np.angle(img), save_path=add_suffix(save_path, '_phase'))

    elif cfg.recon_method == 'pogm_llr':
        if cfg.pogm_llr_sub_before_recon:
            asl.save_nii(img=np.abs(img), save_path=add_suffix(save_path, '_cplx_sub_before_recon'))
        else:
            if cfg.pogm_llr_separate_tc:
                asl.save_asl(img=img, save_path=add_suffix(save_path, '_stc_sub_after_recon'))
            else:
                asl.save_asl(img=img, save_path=add_suffix(save_path, '_nstc_sub_after_recon'))
    else:
        asl.save_asl(img=img, save_path=save_path, roll_and_flip=False)

    if cfg.recon_method == 'pogm_llr' and cfg.pogm_llr_save_intermediate:
        assert extra is not None, "extra should not be None when save_intermediate is True"
        assert 'x_list' in extra, "x_list should be in extra"
        assert 'iter_list' in extra, "iter_list should be in extra"
        for iter, x in zip(extra['iter_list'], extra['x_list']):
            iter_save_path = add_suffix(save_path, f'_iter_{iter}')
            if cfg.pogm_llr_sub_before_recon:
                asl.save_nii(img=np.abs(x), save_path=add_suffix(iter_save_path, '_cplx_sub_before_recon'))
            else:
                if cfg.pogm_llr_separate_tc:
                    asl.save_asl(img=img, save_path=add_suffix(iter_save_path, '_stc_sub_after_recon'))
                else:
                    asl.save_asl(img=img, save_path=add_suffix(iter_save_path, '_nstc_sub_after_recon'))


@clock_func('run_nav_recon')
def run_nav_recon(cfg: DictConfig, always_run=False):
    save_path = Path(cfg.output_path)

    if not always_run and save_path.exists():
        print("run_img_recon already done.")
        return 
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    asl = ASL.load(cfg.input_path)
    res = asl.recon_img(flag_recon_per_shot=True, **cfg)
    img = res.get('recon', None)
    if img is None:
        raise ValueError("Reconstruction failed, 'recon' key not found in the result.")
    else:
        asl.save_asl(img=img, save_path=save_path, roll_and_flip=False)


@clock_func('run_nav_recon')
def run_nav_recon_dl(cfg: DictConfig, always_run=False):
    save_path = Path(cfg.output_path)

    if not always_run and save_path.exists():
        print("run_img_recon already done.")
        return 
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    asl = ASL.load(cfg.input_path)
    img = Image(save_path.parent / 'nav.nii.gz').data
    asl.save_asl(img=img, save_path=save_path, roll_and_flip=False)

    if (save_path.parent / 'ref.nii.gz').exists():
        ref_img = Image(save_path.parent / 'ref.nii.gz').data
        asl.save_asl(img=ref_img, save_path=add_suffix(save_path, '_ref'), roll_and_flip=False)


@clock_func('run_motion_estimation_mcFLIRT')
def run_motion_estimation_mcFLIRT(cfg: DictConfig, always_run=False):
    save_path = Path(cfg.output_path)
    if not always_run and save_path.exists():
        print("run_motion_estimation_mcFLIRT already done.")
    else:
        input_path = Path(cfg.input_path)
        mcflirt_mat_file = add_suffix(input_path, '_moco_mcflirt').with_suffix('').with_suffix('').with_suffix('.mat')
        if mcflirt_mat_file.exists():
            shutil.rmtree(mcflirt_mat_file)

        mcflirt(infile=input_path, mats=True, refvol=0, out=mcflirt_mat_file.with_suffix(''))
        
        spacing = np.asarray(cfg.spacing)
        rigid_transform_flirt = RigidTransformList.load_from_flirt(mcflirt_mat_file, spacing)
        rigid_transform_flirt = RigidTransformList([rigid_transform.inverse() for rigid_transform in rigid_transform_flirt])

        rigid_transform_flirt.save(save_path)
        rigid_transform_flirt.plot(save_path=save_path.with_suffix('.png'))


@clock_func('run_motion_estimation_mcFLIRT_dl')
def run_motion_estimation_mcFLIRT_dl(cfg: DictConfig, always_run=False):
    save_path = Path(cfg.output_path)
    if not always_run and save_path.exists():
        print("run_motion_estimation_mcFLIRT_dl already done.")
    else:
        input_path = Path(cfg.input_path)

        flag_use_ref = True

        input_image = Image(input_path)
        data = input_image.data
        header = input_image.header

        n_rep = 6
        n_shot = 16

        if flag_use_ref:
            ref_path = input_path.parent / 'img_ref.nii.gz'
            ref_data = Image(ref_path).data
            ref_data = ref_data[..., :n_shot]
            data = np.concatenate((ref_data, data), axis=-1)

        temp_dir = Path(cfg.output_dir) / 'temp'
        temp_dir.mkdir(parents=True, exist_ok=True)
        rigid_transform_flirt_rep = []
        for i_shot in range(n_shot):
            data_shot = data[..., i_shot::n_shot]
            temp_path = temp_dir / f'input_shot{i_shot}'

            Image(image=data_shot, header=header).save(temp_path)

            mcflirt_mat_file = add_suffix(temp_path, '_moco_mcflirt').with_suffix('').with_suffix('').with_suffix('.mat')
            if mcflirt_mat_file.exists():
                shutil.rmtree(mcflirt_mat_file)
            
            mcflirt(infile=temp_path, mats=True, refvol=0, out=mcflirt_mat_file.with_suffix(''))
        
            spacing = np.asarray(cfg.spacing)
            rigid_transform_flirt_shot = RigidTransformList.load_from_flirt(mcflirt_mat_file, spacing)
            rigid_transform_flirt_shot = RigidTransformList([rigid_transform.inverse() for rigid_transform in rigid_transform_flirt_shot])
            rigid_transform_flirt_rep.append(rigid_transform_flirt_shot)
        
        rigid_transform_flirt = []

        for i_rep in range(n_rep):
            if flag_use_ref:
                i_rep = i_rep + 1  # skip the reference shot
            for i_shot in range(n_shot):
                rigid_transform_flirt.append(rigid_transform_flirt_rep[i_shot][i_rep])

        rigid_transform_flirt = RigidTransformList(rigid_transforms=rigid_transform_flirt)

        rigid_transform_flirt.save(save_path)
        rigid_transform_flirt.plot(save_path=save_path.with_suffix('.png'))
        
        

@clock_func('run_motion_compensated_SENSE')
def run_motion_compensated_SENSE(cfg: DictConfig, always_run=False):
    save_path = Path(cfg.output_path)
    if not always_run and save_path.exists():
        print("run_motion_compensated_SENSE already done.")
    else:
        asl_mot: ASL = ASL.load(cfg.asl_path)
        rigid_transform_flirt = RigidTransformList.load_from_npy(file_path=cfg.motion_params)
        res = asl_mot.recon_img(rigid_transforms=rigid_transform_flirt, **cfg)
        img = res.get('recon', None)
        if img is None:
            raise ValueError("Reconstruction failed, 'recon' key not found in the result.")
        else:
            asl_mot.save_asl(img=img, save_path=save_path)




@clock_func('run_generate_simulated_motion_data')
def run_generate_simulated_motion_data(cfg: DictConfig, always_run=False):
    save_path = Path(cfg.output_path).parent.parent / 'sim_motion' / 'asl_mot.h5'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if not always_run and save_path.exists():
        print("run_motion_compensated_SENSE already done.")
    else:
        asl_ori: ASL = ASL.load(cfg.input_path)

        rigid_transform_gt = RigidTransformList([
            RigidTransform(par=[0, 0, 0, 0, 0, 0], is_radian=False, rotation_center=[0, 0, 0]),
            RigidTransform(par=[1, 0, 0, 0, 0, 0], is_radian=False, rotation_center=[0, 0, 0]),
            RigidTransform(par=[0, 2, 0, 0, 0, 0], is_radian=False, rotation_center=[0, 0, 0]),
            RigidTransform(par=[0, 0, 3, 0, 0, 0], is_radian=False, rotation_center=[0, 0, 0]),
            RigidTransform(par=[0, 0, 0, 1, 0, 0], is_radian=False, rotation_center=[0, 0, 0]),
            RigidTransform(par=[0, 0, 0, 0, 2, 0], is_radian=False, rotation_center=[0, 0, 0]),
            RigidTransform(par=[0, 0, 0, 0, 0, 3], is_radian=False, rotation_center=[0, 0, 0]),
            RigidTransform(par=[-1, -2, -3, -1, -2, -3], is_radian=False, rotation_center=[0, 0, 0]),
        ])

        rigid_transform_gt.plot(save_path=save_path.parent / 'par_gt.png')

        n_rep = cfg.n_rep
        ksp_mot = np.zeros_like(asl_ori.ksp)[..., :n_rep]
        for i_rep in range(n_rep):
            asl_new = ASL(ksp=asl_ori.ksp[..., i_rep: i_rep + 1],
                sampling_mask=asl_ori.sampling_mask[..., i_rep: i_rep + 1],
                sensitivity_maps=asl_ori.sensitivity_maps,
                scanner_nii=asl_ori.scanner_nii,
                flip=asl_ori.flip,
                roll=asl_ori.roll,
                )
            n_segments = asl_new.Ns
            binary_sampling_mask = np.zeros((*asl_new.sampling_mask.shape[:4], n_segments), dtype=int)

            for i_shot in range(n_segments):
                binary_sampling_mask[asl_new.sampling_mask[..., 0] == (i_shot + 1), i_shot] = 1
            
            sense_op = SenseOP(
                sampling_mask=asl_new.sampling_mask[..., 0],
                sensitivity_maps=asl_new.sensitivity_maps,
                ksp_size=asl_new.ksp.shape[:4],
            )

            mc_sense_op = MotionCompensatedSenseOP(
                sampling_mask=binary_sampling_mask,
                sensitivity_maps=asl_new.sensitivity_maps,
                ksp_size=[*asl_new.ksp.shape[:4], n_segments],
                rigid_transforms=rigid_transform_gt[i_rep * n_segments:(i_rep + 1) * n_segments],
                interpolation_method='sinc',
            )

            ksp_mot_ = mc_sense_op.fwd(sense_op.adj(asl_new.ksp.reshape(asl_new.Nc, -1))).reshape(*asl_new.ksp.shape[:4], n_segments)
            ksp_mot[..., i_rep] = np.sum(ksp_mot_, axis=-1)

        asl_mot = ASL(ksp=ksp_mot,
                    sampling_mask=asl_ori.sampling_mask[..., :n_rep],
                    sensitivity_maps=asl_new.sensitivity_maps,
                    scanner_nii=asl_new.scanner_nii,
                    flip=asl_new.flip,
                    roll=asl_new.roll)  
        
        asl_mot.save(save_path=save_path)
        return save_path

        # res = asl_mot.recon_img(flag_recon_per_shot=True, multiprocessing=4, **cfg)
        # img = res.get('recon', None)
        # asl_mot.save_asl(img=img, save_path=save_path.parent / 'img_mot.nii.gz', roll_and_flip=False)

        # mcflirt_mat_file = add_suffix(save_path.parent / 'img_mot.nii.gz', '_moco_mcflirt').with_suffix('').with_suffix('').with_suffix('.mat')
        # if mcflirt_mat_file.exists():
        #     shutil.rmtree(mcflirt_mat_file)

        # mcflirt(infile=save_path.parent / 'img_mot.nii.gz', mats=True, refvol=0, out=mcflirt_mat_file.with_suffix(''))
        
        # spacing = np.asarray([3.6, 3.6, 3.6])
        # # spacing = np.asarray([1, 1, 1])

        # rigid_transform_flirt = RigidTransformList.load_from_flirt(mcflirt_mat_file, spacing)
        # # rigid_transform_flirt = RigidTransformList([rigid_transform.inverse() for rigid_transform in rigid_transform_flirt])
        # rigid_transform_flirt.save(save_path.parent / 'par_mcflirt.npy')
        # rigid_transform_flirt.plot(save_path=save_path.parent / 'par_mcflirt.png')

        # rigid_transform_flirt = RigidTransformList.load_from_npy(file_path=save_path.parent / 'par_mcflirt.npy')

        # res = asl_mot.recon_img(rigid_transforms=rigid_transform_flirt, recon_method='mc_sense', mc_interpolation_method='sinc', mc_maxiter=100, mc_atol=1e-1)
        # img = res.get('recon', None)
        # if img is None:
        #     raise ValueError("Reconstruction failed, 'recon' key not found in the result.")
        # else:
        #     asl_mot.save_nii(img=img, save_path=save_path.parent / 'img_moco_mcsense.nii.gz', roll_and_flip=True)



@clock_func('run_prep_mcSENSE_solve_T_x')
def run_prep_mcSENSE_solve_T_x(cfg: DictConfig, always_run=False) -> None:
    flag_half_res = cfg.flag_half_res
    cfg = cfg.motion_correction_inter_shot
    temp_dir = Path(cfg.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(cfg.output_path)
    if not always_run and (output_path.exists() or (temp_dir / 'input_rep1.mat').exists()):
        print("run_prep_mcSENSE_solve_T_x already done.")
    else:
        if flag_half_res:
            asl_half_path = add_suffix(Path(cfg.asl_path), '_half_res')
            if asl_half_path.exists():
                asl = ASL.load(asl_half_path)
            else:
                asl = ASL.load(cfg.asl_path).half_res()
                asl.save(save_path=asl_half_path)

            res = asl.recon_img(recon_method='cg_sense')
            img = res.get('recon', None)
            asl.save_asl(img=img, save_path=temp_dir / 'recon_half_res', roll_and_flip=True)
        else:
            asl = ASL.load(cfg.asl_path)
        
        
        binary_sampling_mask = []
        for i_shot in range(asl.Ns):
            binary_sampling_mask.append(asl.sampling_mask[..., 0] == (i_shot + 1))
        binary_sampling_mask = np.stack(binary_sampling_mask, axis=4)

        for i_rep in range(asl.Nt):
            ckp_path = temp_dir / f'input_rep{i_rep+1}.mat'

            res = {
                'y': asl.ksp[..., i_rep].transpose((2, 3, 1, 0)),                  # Nc, Nx, Ny, Nz ->  Ny, Nz, Nx, Nc
                'S': asl.sensitivity_maps[..., 0].transpose((2, 3, 1, 0)),         # Nc, Nx, Ny, Nz -> Ny, Nz, Nx, Nc
                'A': binary_sampling_mask.transpose((2, 3, 1, 0, 4)),              # Ny, Nz, 1, 1, Nshot
                'NS': asl.Ns,
                'n_rep': asl.Nt,
            }
            scipy.io.savemat(ckp_path, res)


def convert_TGT_to_rigid_transform(TGT, img_center):
    ty, tz, tx, rx, rz, ry = TGT  # (ty, tz, tx, rx, rz, ry) radian pixel
    rigid_transform = RigidTransform(par=(rx, ry, rz, tx, ty, tz), is_radian=True, rotation_center=img_center)
    return rigid_transform


@clock_func('run_combine_motion_estimation')
def run_combine_motion_estimation(cfg: DictConfig, always_run=False) -> None:
    asl: ASL = ASL.load(cfg.motion_correction_inter_shot.asl_path)
    
    # Step 1: Load intra-volume motion estimated by mcSENSE
    save_path = Path(cfg.motion_correction_inter_shot.output_path)
    if not always_run and save_path.exists():
        print("Inter shot motion estimation has already done.")
    else:
        img_moco_inter_shot = []
        rigid_transforms = []
        
        for i_rep in range(asl.Nt):
            res_path = Path(cfg.motion_correction_inter_shot.temp_dir) / f'output_rep{i_rep+1}.mat'
            assert res_path.exists(), f"Expected file {res_path} does not exist. Please run the mcSENSE solve T and x step first."
            res = scipy.io.loadmat(res_path)
            img_moco_inter_shot.append(res['xEst'][0, 0])
            img_center = np.asarray(asl.img_center, dtype=float)
            for i_shot in range(asl.Ns):
                ty, tz, tx, rx, rz, ry = res['TEst'][0, 0][0, 0, 0, 0][i_shot]  # (ty, tz, tx, rx, rz, ry) radian pixel
                rigid_transform = RigidTransform(par=(rx, ry, rz, tx, ty, tz), is_radian=True, rotation_center=img_center)
                rigid_transforms.append(rigid_transform)      

        img_moco_inter_shot= np.stack(img_moco_inter_shot, axis=-1).transpose((2, 0, 1, 3))
        asl.save_asl(img_moco_inter_shot, save_path=save_path, roll_and_flip=False)

        rigid_transform_alignedSENSE = RigidTransformList(rigid_transforms=rigid_transforms)
        rigid_transform_alignedSENSE.save(cfg.motion_correction_inter_shot.motion_params)
        rigid_transform_alignedSENSE.plot(save_path=Path(cfg.motion_correction_inter_shot.output_dir) / 'par_alignedSENSE.png')

    # Step 2: Inter-volume motion correction by mcFLIRT
    save_path = Path(cfg.motion_estimation_inter_volume.motion_params)
    if not always_run and save_path.exists():
        print("Inter volume motion estimation has already done.")
    else:
#         motion_flirt = motion_estimation_mcflirt(cfg.motion_estimation_inter_volume.input_path, registration_reference_img_path=cfg.motion_estimation_inter_volume.registration_reference_img_path)
#         spacing = np.asarray(mr4d_mot)
#         motion_flirt_iso = [despacing(RigidTransform(mtx=m.mtx), spacing) for m in motion_flirt]

#         save_rigid_transforms(cfg.motion_estimation_inter_volume.motion_params, motion_flirt_iso)
#         plot_motion_params(motion_lists=[motion_flirt_iso], title_list=['inter_volume'], save_path=cfg.motion_estimation_inter_volume.motion_params.replace('.npy', '.png'))
# #    
        input_path = Path(cfg.motion_estimation_inter_volume.input_path)
        mcflirt_mat_file = add_suffix(input_path, '_moco_mcflirt').with_suffix('').with_suffix('').with_suffix('.mat')
        if mcflirt_mat_file.exists():
            shutil.rmtree(mcflirt_mat_file)

        mcflirt(infile=input_path, mats=True, refvol=0, out=mcflirt_mat_file.with_suffix(''))
        
        rigid_transform_flirt = RigidTransformList.load_from_flirt(mcflirt_mat_file, asl.header.get_zooms()[:3])
        # rigid_transform_flirt = RigidTransformList([rigid_transform.inverse() for rigid_transform in rigid_transform_flirt])

        rigid_transform_flirt.save(save_path)
        rigid_transform_flirt.plot(save_path=save_path.with_suffix('.png'))

    
    # Step 3: Combine Inter-shot and Inter-volume motions
    save_path = Path(cfg.motion_correction_combined.motion_params)
    if not always_run and save_path.exists():
        print("Motion estimation combining has already done.")
    else:
        rigid_transforms = []
        Nt = asl.Nt
        Ns = asl.Ns
        for i_rep in range(Nt):
            for i_shot in range(Ns):
                rt_inter_volume = rigid_transform_flirt[i_rep]
                rt_inter_shot = rigid_transform_alignedSENSE[i_rep * Ns + i_shot]
                # rt_combined = RigidTransform(mtx=np.dot(rt_inter_volume.mtx, rt_inter_shot.mtx)).inverse()
                rt_combined = RigidTransform(mtx=np.dot(rt_inter_volume.inverse().mtx, rt_inter_shot.mtx))
                rigid_transforms.append(rt_combined)
                # rigid_transforms.append(rt_inter_shot)
                # rigid_transforms.append(rt_inter_volume)

        rigid_transform_combined = RigidTransformList(rigid_transforms=rigid_transforms)
        rigid_transform_combined.save(save_path)
        rigid_transform_combined.plot(save_path=save_path.with_suffix('.png'))
