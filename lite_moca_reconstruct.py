import torch
import os.path as osp
import logging
from omegaconf import OmegaConf

from lib_prior.prior_loading import Saved2D
from lib_moca.moca import moca_solve
from lib_moca.camera import MonocularCameras

from mosca_evaluate import test_tum_cam, test_sintel_cam

from data_utils.iphone_helpers import load_iphone_gt_poses, load_nvidia_gt_poses
from data_utils.nvidia_helpers import load_nvidia_gt_pose, get_nvidia_dummy_test

from recon_utils import (
    seed_everything,
    setup_recon_ws,
    auto_get_depth_dir_tap_mode,
    SEED,
)


def load_gt_cam(ws, fit_cfg, nvidia_mode=False):
    mode = getattr(fit_cfg, "mode", "iphone")
    logging.info(f"Loading gt camera poses in mode {mode}")
    if mode == "iphone":
        if nvidia_mode:
            return load_nvidia_gt_poses(ws, t_subsample=getattr(fit_cfg, "t_subsample", 1))
        else:
            return load_iphone_gt_poses(ws, t_subsample=getattr(fit_cfg, "t_subsample", 1))
    elif mode == "nvidia":
        (gt_training_cam_T_wi, gt_training_fov, gt_training_cxcy_ratio) = (
            load_nvidia_gt_pose(osp.join(ws, "poses_bounds.npy"))
        )
        # gt_training_cam_T_wi[:, :3, 3] = gt_training_cam_T_wi[:, :3, 3] * 0.01
        # logging.warning(f"Manually rescale the translation by 0.1")
        (
            gt_testing_cam_T_wi_list,
            gt_testing_tids_list,
            gt_testing_fns_list,
            gt_testing_fov_list,
            gt_testing_cxcy_ratio_list,
        ) = get_nvidia_dummy_test(gt_training_cam_T_wi, gt_training_fov)
        return (
            gt_training_cam_T_wi,
            gt_testing_cam_T_wi_list,
            gt_testing_tids_list,
            gt_testing_fns_list,
            gt_training_fov,
            gt_testing_fov_list,
            gt_training_cxcy_ratio,
            gt_testing_cxcy_ratio_list,
        )
    else:
        raise RuntimeError(f"Unknown mode: {mode}")
    return


def static_reconstruct(ws, log_path, fit_cfg, nvidia_mode=False):
    seed_everything(SEED) # 12345
    DEPTH_DIR, TAP_MODE = auto_get_depth_dir_tap_mode(ws, fit_cfg) # iphone: sensor_depth, bootstapir; nvidia: metric3d_depth, cotracker
    DEPTH_BOUNDARY_TH = getattr(fit_cfg, "depth_boundary_th", 1.0) # iphone: 1.0; nvidia: 0.3 NOTE laplacian_filter_depth
    INIT_GT_CAMERA_FLAG = getattr(fit_cfg, "init_gt_camera", False) # True: clomap; False: colmap-free
    DEP_MEDIAN = getattr(fit_cfg, "dep_median", 1.0) # iphone: -1; nvidia: -1

    EPI_TH = getattr(fit_cfg, "ba_epi_th", getattr(fit_cfg, "epi_th", 1e-3)) # iphone: 3e-5; nvidia: 0.001; nvidia_marble:1e-5
    logging.info(f"Static BA with EPI_TH={EPI_TH}")
    print(f"Static BA with EPI_TH={EPI_TH}")
    device = torch.device("cuda:0")

    s2d: Saved2D = (
        Saved2D(ws, nvidia_mode)
        .load_epi()
        .load_dep(DEPTH_DIR, DEPTH_BOUNDARY_TH, nvidia_mode)
        .normalize_depth(median_depth=DEP_MEDIAN) # NOTE depth scale; iphone: no scale, 
        .recompute_dep_mask(depth_boundary_th=DEPTH_BOUNDARY_TH) # recompute depth mask because normalize depth
        .load_track(
            f"*uniform*{TAP_MODE}", # NOTE uniform
            min_valid_cnt=getattr(fit_cfg, "tap_loading_min_valid_cnt", 4), # cnt: count
        )
        .load_vos() # video object segmentation
    )

    if INIT_GT_CAMERA_FLAG: # use gt camera / use  gt focal
        # if start form gt camera, load gt camera here
        logging.info(f"Initializing from GT camera")
        (
            gt_training_cam_T_wi,
            gt_testing_cam_T_wi_list,
            gt_testing_tids_list,
            gt_testing_fns_list,
            gt_training_fov,
            gt_testing_fov_list,
            gt_training_cxcy_ratio,
            gt_testing_cxcy_ratio_list,
        ) = load_gt_cam(ws, fit_cfg, nvidia_mode) # NOTE load gt camera information
        gt_fovdeg = float(gt_training_fov) # gt camera fov
        cxcy_ratio = gt_training_cxcy_ratio[0]  # gt camera center ratio
        if getattr(fit_cfg, "init_gt_camera_focal_only", False): # only init camera focal
            logging.info(f"Only init focal length")
            cams = MonocularCameras(
                n_time_steps=s2d.T,
                default_H=s2d.H,
                default_W=s2d.W,
                fxfycxcy=[gt_fovdeg, gt_fovdeg] + cxcy_ratio,
                delta_flag=True, # for init_camera_pose
                init_camera_pose=torch.eye(4)
                .to(gt_training_cam_T_wi)[None]
                .expand(len(gt_training_cam_T_wi) - 1, -1, -1),
                iso_focal=getattr(fit_cfg, "iso_focal", False),
            )
        else: # use camera pose & focal
            cams = MonocularCameras( # NOTE only train camera
                n_time_steps=s2d.T, # apple: 475
                default_H=s2d.H,
                default_W=s2d.W,
                fxfycxcy=[gt_fovdeg, gt_fovdeg] + cxcy_ratio, # [fovdeg, fovdeg, cx_ratio, cy_ratio]
                delta_flag=False, # for init_camera_pose
                init_camera_pose=gt_training_cam_T_wi, # T,4,4; NOTE c2w
                iso_focal=getattr(fit_cfg, "iso_focal", False), # False
            )
    else:
        cams = None

    logging.info("*" * 20 + "MoCa BA" + "*" * 20)
    cams, s2d, _ = moca_solve( # monocular camera, optimal gt camera
        ws=log_path,
        s2d=s2d,
        device=device,
        epi_th=EPI_TH,
        ba_total_steps=getattr(fit_cfg, "ba_total_steps", 2000),
        ba_switch_to_ind_step=getattr(fit_cfg, "ba_switch_to_ind_step", 500),
        ba_depth_correction_after_step=getattr(fit_cfg, "ba_depth_correction_after_step", 500),
        ba_max_frames_per_step=32,
        static_id_mode="raft" if s2d.has_epi else "track",
        # * robust setting
        robust_depth_decay_th=getattr(fit_cfg, "robust_depth_decay_th", 2.0),
        robust_depth_decay_sigma=getattr(fit_cfg, "robust_depth_decay_sigma", 1.0),
        robust_std_decay_th=getattr(fit_cfg, "robust_std_decay_th", 0.2),
        robust_std_decay_sigma=getattr(fit_cfg, "robust_std_decay_sigma", 0.2),
        #
        gt_cam=cams,
        iso_focal=getattr(fit_cfg, "iso_focal", False),
        rescale_gt_cam_transl=getattr(fit_cfg, "rescale_gt_cam_transl", False),
        ba_lr_cam_f=getattr(fit_cfg, "ba_lr_cam_f", 0.0003),
        ba_lr_dep_c=getattr(fit_cfg, "ba_lr_dep_c", 0.001),
        ba_lr_dep_s=getattr(fit_cfg, "ba_lr_dep_s", 0.001),
        ba_lr_cam_q=getattr(fit_cfg, "ba_lr_cam_q", 0.0003),
        ba_lr_cam_t=getattr(fit_cfg, "ba_lr_cam_t", 0.0003),
        #
        ba_lambda_flow=getattr(fit_cfg, "ba_lambda_flow", 1.0),
        ba_lambda_depth=getattr(fit_cfg, "ba_lambda_depth", 0.1),
        ba_lambda_small_correction=getattr(fit_cfg, "ba_lambda_small_correction", 0.03),
        ba_lambda_cam_smooth_trans=getattr(fit_cfg, "ba_lambda_cam_smooth_trans", 0.0),
        ba_lambda_cam_smooth_rot=getattr(fit_cfg, "ba_lambda_cam_smooth_rot", 0.0),
        #
        depth_filter_th=getattr(fit_cfg, "ba_depth_remove_th", -1.0),
        init_cam_with_optimal_fov_results=getattr(fit_cfg, "init_cam_with_optimal_fov_results", True),
        # fov
        fov_search_fallback=getattr(fit_cfg, "ba_fov_search_fallback", 53.0),
        fov_search_N=getattr(fit_cfg, "ba_fov_search_N", 100),
        fov_search_start=getattr(fit_cfg, "ba_fov_search_start", 30.0),
        fov_search_end=getattr(fit_cfg, "ba_fov_search_end", 90.0),
        viz_valid_ba_points=getattr(fit_cfg, "ba_viz_valid_points", False),
    )  # ! S2D is changed becuase the depth is re-scaled

    datamode = getattr(fit_cfg, "mode", "iphone")
    if datamode == "sintel":
        test_func = test_sintel_cam
    elif datamode == "tum":
        test_func = test_tum_cam
    else:
        test_func = None
    if test_func is not None:
        test_func(
            cam_pth_fn=osp.join(log_path, "bundle", "bundle_cams.pth"),
            ws=ws,
            save_path=osp.join(log_path, "cam_metrics_ba.txt"),
        )

    return s2d


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("MoCa Reconstruction Camera Only")
    parser.add_argument("--ws", type=str, help="Source folder", required=True)
    parser.add_argument("--cfg", type=str, help="profile yaml file path", required=True)
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.load(args.cfg)
    cli_cfg = OmegaConf.from_dotlist([arg.lstrip("--") for arg in unknown])
    cfg = OmegaConf.merge(cfg, cli_cfg)

    logdir = setup_recon_ws(args.ws, fit_cfg=cfg)

    static_reconstruct(args.ws, logdir, cfg)
