# from origial reconstruction.py import all rendering related stuff
import sys, os, os.path as osp
import time
import torch
import logging
from tqdm import tqdm
from omegaconf import OmegaConf
from misc import configure_logging, get_timestamp
import numpy as np
import imageio
import matplotlib.pyplot as plt
from matplotlib import cm
import kornia
import colorsys
import cv2
import torch.nn.functional as F

sys.path.append(osp.dirname(osp.abspath(__file__)))

from pytorch3d.ops import knn_points
from dynamic_gs import DynSCFGaussian
from static_gs import StaticGaussian
from mosca import MoSca
from camera import MonocularCameras
from photo_recon_utils import fetch_leaves_in_world_frame, estimate_normal_map
from lib_render.render_helper import GS_BACKEND, render, RGB2SH
from lib_prior.prior_loading import Saved2D
from gs_utils.gs_optim_helper import update_learning_rate
from gs_utils.loss_helper import (
    compute_rgb_loss,
    compute_dep_loss,
    compute_normal_loss,
    compute_normal_reg_loss,
    compute_dep_reg_loss,
)
from dynamic_solver import get_dynamic_curves

from photo_recon_utils import (
    fetch_leaves_in_world_frame,
    GSControlCFG,
    OptimCFG,
    apply_gs_control,
    error_grow_dyn_model,
    identify_traj_id,
    get_colorplate,
    error_grow_node_model,
    get_move_around_cam_T_cw,
    calculate_relative_angles,
    estimate_focus_depth_from_dyn_gs,
)
from photo_recon_viz_utils import (
    viz_hist,
    viz_dyn_hist,
    viz2d_total_video,
    viz3d_total_video,
    viz_curve,
)


def get_recon_cfg(cfg_fn=None):
    if cfg_fn is None:
        logging.info("No cfg_fn provided, use dummy cfg")
        # get dummy
        cfg = OmegaConf.create()
        cfg.s_ctrl = OmegaConf.create()
        cfg.d_ctrl = OmegaConf.create()
        return cfg
    cfg = OmegaConf.load(cfg_fn)
    for key in ["photometric", "geometric"]:
        if not hasattr(cfg, key):
            setattr(cfg, key, OmegaConf.create())
    for key in ["d_ctrl", "s_ctrl"]:
        if not hasattr(cfg.photometric, key):
            setattr(cfg.photometric, key, None)
    OmegaConf.set_readonly(cfg, True)
    logging.info(f"Load cfg from {cfg_fn}: {cfg}")
    return cfg


class DynReconstructionSolver:
    def __init__(
        self,
        working_dir,
        device=torch.device("cuda:0"),
        # cfg
        radius_init_factor=4.0,  # ! 1.0
        opacity_init_factor=0.95,
    ):
        self.src = working_dir 
        self.device = device
        timestamp = get_timestamp()
        configure_logging(
            osp.join(self.src, f"dynamic_reconstruction_{timestamp}.log"),
            debug=False,
        )

        self.log_dir = self.src
        self.viz_dir = osp.join(self.src, f"mosca_photo_viz_{timestamp}") # mosca_photo_viz_20250925_210838
        os.makedirs(self.viz_dir, exist_ok=True)

        self.radius_init_factor = radius_init_factor # iphone: 4; nvidia_rodynrf: 1
        self.opacity_init_factor = opacity_init_factor # iphone: 0.95; nvidia_rodynrf: 0.95
        return

    @torch.no_grad()
    def identify_fg_mask_by_nearest_curve(
        self, s2d: Saved2D, cams: MonocularCameras, viz_fname=None
    ):
        # get global anchor
        curve_xyz, curve_mask, _, _ = get_dynamic_curves(
            s2d, cams, return_all_curves=True
        ) # curve_xyz: [T_all,N_all,3] curve_mask: [T_all,N_all]
        assert curve_xyz.shape[1] == len(s2d.dynamic_track_mask)

        # optimize static curve by computing meaning, only consider the valid case
        static_curve_mean = (
            curve_xyz[:, ~s2d.dynamic_track_mask]
            * curve_mask[:, ~s2d.dynamic_track_mask, None]
        ).sum(0, keepdim=True) / curve_mask[:, ~s2d.dynamic_track_mask, None].sum(
            0, keepdim=True
        ).expand(
            len(curve_xyz), -1, -1
        ) # T_all,N_sta,3
        curve_xyz[:, ~s2d.dynamic_track_mask] = static_curve_mean
        np.savetxt(
            osp.join(self.viz_dir, "fg_id_non_dyn_curve_meaned.xyz"),
            static_curve_mean.reshape(-1, 3).cpu().numpy(),
            fmt="%.4f",
        ) 

        with torch.no_grad():
            fg_mask_list = []
            for query_tid in tqdm(range(s2d.T)):
                query_dep = s2d.dep[query_tid].clone()  # H,W
                query_xyz_cam = cams.backproject(
                    cams.get_homo_coordinate_map(), query_dep
                ) # H,W,3
                query_xyz_world = cams.trans_pts_to_world(
                    query_tid, query_xyz_cam
                ) # H,W,3

                # find the nearest distance and acc sk weight
                # use all the curve at this position to id the fg and bg
                _, knn_id, _ = knn_points(
                    query_xyz_world.reshape(1, -1, 3), curve_xyz[query_tid][None], K=1
                ) # 1,H*W,1
                knn_id = knn_id[0, :, 0] # H*W
                fg_mask = s2d.dynamic_track_mask[knn_id].reshape(s2d.H, s2d.W) # H,W
                fg_mask_list.append(fg_mask.cpu())
            fg_mask_list = torch.stack(fg_mask_list, 0) # T,H,W
        if viz_fname is not None: # 'gs_model_construct_fg_mask.gif'
            viz_rgb = s2d.rgb.clone().cpu() # T,H,W,3
            viz_fg_mask_list = fg_mask_list * s2d.dep_mask.to(fg_mask_list)
            viz_rgb = viz_rgb * viz_fg_mask_list.float()[..., None] + viz_rgb * 0.1 * (
                1 - viz_fg_mask_list.float()[..., None]
            ) # only fg
            imageio.mimsave(osp.join(self.viz_dir, viz_fname), ((viz_rgb.numpy())*255).astype(np.uint8)) 

        fg_mask_list = fg_mask_list.to(cams.rel_focal.device).bool() # T,H,W
        s2d.register_2d_identification(
            static_2d_mask=~fg_mask_list, dynamic_2d_mask=fg_mask_list
        ) # NOTE Save2d
        return fg_mask_list

    @torch.no_grad()
    def compute_normals_for_s2d(
        self,
        s2d,
        cams,
        patch_size=7,
        nn_dist_th=0.03,
        nn_min_cnt=4,
        viz_fn=None,
        viz_subsample=4,
    ):
        # compute normal maps for s2d
        logging.info(f"Computing normal maps from depth maps using local SVD...")
        # the computed normals are always pointing backward, on -z direction
        ray_direction = cams.backproject(
            s2d.homo_map.clone(), torch.ones_like(s2d.homo_map[..., 0])
        )
        ray_direction = F.normalize(ray_direction, dim=-1)
        normal_map_list = []
        new_mask_list = []
        for t in tqdm(range(s2d.T)):
            dep = s2d.dep[t].clone()
            dep_mask = s2d.dep_mask[t].clone()
            normal_map = torch.zeros(*dep.shape, 3).to(dep)
            xyz = cams.backproject(s2d.homo_map[dep_mask].clone(), dep[dep_mask])
            vtx_map = torch.zeros_like(normal_map).float()
            vtx_map[dep_mask] = xyz
            normal_map, mask = estimate_normal_map(
                vtx_map, dep_mask, patch_size, nn_dist_th, nn_min_cnt
            )

            normal = normal_map[mask]
            inner = (normal * ray_direction[mask]).sum(-1)
            correct_orient = inner < 0
            sign = torch.ones_like(normal[..., :1])
            sign[~correct_orient] = -1.0
            normal = normal.clone() * sign
            normal_map[mask] = normal
            new_mask = dep_mask * mask
            normal_map_list.append(normal_map)
            new_mask_list.append(new_mask)
        ret_nrm = torch.stack(normal_map_list, 0)
        ret_mask = torch.stack(new_mask_list, 0)

        if viz_fn is not None:
            viz_fn = osp.join(self.viz_dir, viz_fn)
            logging.info(f"Viz normal maps to {viz_fn}")
            viz_frames = (
                ((-ret_nrm + 1) / 2.0 * 255).detach().cpu().numpy().astype(np.uint8)
            )
            if len(viz_frames) > 50:
                _step = max(1, len(viz_frames) // 50)
            else:
                _step = 1
            # skip to boost viz
            viz_frames = viz_frames[:, ::viz_subsample, ::viz_subsample, :]
            imageio.mimsave(viz_fn, viz_frames[::_step])

        s2d.register_buffer("nrm", ret_nrm.detach().clone())
        s2d.dep_mask = ret_mask.detach().clone()
        return

    @torch.no_grad()
    def get_static_model(
        self,
        s2d: Saved2D,
        cams,
        n_init=30000, # iphone_formal: 30000, nvidia_rodynrf_warmup: 100000
        radius_max=0.05, # iphone_formal: 0.03, nvidia_rodynrf_warmup: 1.0
        max_sph_order=0, # 3
        image_stride=1, # 1
        viz_fn=None,
        mask_type="static_depth", # depth
    ):
        device = self.device

        if mask_type == "static_depth": # NOTE photometric_warmup
            gather_mask = s2d.sta_mask * s2d.dep_mask # depth mask & static mask
        elif mask_type == "depth": # NOTE photometric_reconstruct
            gather_mask = s2d.dep_mask # only depth mask
        else:
            raise ValueError(f"Unknown mask_type={mask_type}")

        mu_init, q_init, s_init, o_init, rgb_init, id_init, _ = ( # _: ref_time
            fetch_leaves_in_world_frame(
                cams=cams,
                n_attach=n_init, # 30000
                input_mask_list=gather_mask, # T,H,W
                input_dep_list=s2d.dep, # T,H,W
                input_rgb_list=s2d.rgb, # T,H,W,3
                save_xyz_fn=viz_fn, # None
                subsample=image_stride, # 1
            )
        )
        s_model: StaticGaussian = StaticGaussian(
            init_mean=mu_init.clone().to(device),
            init_q=q_init,
            init_s=s_init * self.radius_init_factor, # 4
            init_o=o_init * self.opacity_init_factor, # 0.95
            init_rgb=rgb_init,
            init_id=id_init, # None
            max_scale=radius_max, # 0.03
            min_scale=0.0,
            max_sph_order=max_sph_order, # 3
        )
        s_model.to(device)
        return s_model

    @torch.no_grad()
    def get_dynamic_model(
        self,
        s2d: Saved2D,
        cams: MonocularCameras,
        scf: MoSca,
        n_init=10000, # 30000
        image_stride=1, # 1
        radius_max=0.05, # 0.03
        max_sph_order=0, # 3
        leaf_local_flag=True, # True
        topo_th_ratio=None,
        dyn_o_flag=True,
        additional_mask=None,
        nn_fusion=-1, # -1
        max_node_num=100000, # 100000
    ):
        device = self.device
        collect_t_list = torch.arange(0, s2d.T, 1) # all_T
        logging.info(f"Collect GS at t={collect_t_list}")
        input_mask_list = s2d.dyn_mask * s2d.dep_mask
        if additional_mask is not None: # iphone: None; nvidia: not None
            assert additional_mask.shape == s2d.dep_mask.shape
            input_mask_list = input_mask_list * additional_mask
        mu_init, q_init, s_init, o_init, rgb_init, id_init, time_init = (
            fetch_leaves_in_world_frame(
                cams=cams,
                n_attach=n_init, # 30000
                input_mask_list=input_mask_list, # T,H,W
                input_dep_list=s2d.dep, # T,H,W
                input_rgb_list=s2d.rgb, # T,H,W,3
                subsample=image_stride, # 1
            )
        )
        # * Reset SCF topo th!
        if topo_th_ratio is not None: # iphone: None; NVIDIA: 
            old_th_ratio = scf.topo_th_ratio # iphone: 5.0
            scf.topo_th_ratio = torch.ones_like(scf.topo_th_ratio) * topo_th_ratio
            logging.info(
                f"Reset SCF topo th ratio from {old_th_ratio} to {scf.topo_th_ratio}"
            )

        # * Init the scf
        d_model: DynSCFGaussian = DynSCFGaussian(
            scf=scf,
            max_scale=radius_max, # 0.03
            min_scale=0.0,
            max_sph_order=max_sph_order, # 3
            device=device,
            leaf_local_flag=leaf_local_flag, # True
            dyn_o_flag=dyn_o_flag, # True
            nn_fusion=nn_fusion, # -1
            max_node_num=max_node_num, # 100000
        ) # empty dynamic gaussian
        d_model.to(device)

        # * Init the leaves
        optimizer = torch.optim.Adam(d_model.get_optimizable_list())
        unique_tid = time_init.unique()
        logging.info("Attach to Dynamic Scaffold ...")
        for tid in tqdm(unique_tid): # [0 1 2 3 ...]
            t_mask = time_init == tid
            d_model.append_new_gs(
                optimizer,
                tid=tid,
                mu_w=mu_init[t_mask], # gs position in tid in world system
                quat_w=q_init[t_mask], # gs quaternion in tid in world system
                scales=s_init[t_mask] * self.radius_init_factor, # 4.0
                opacity=o_init[t_mask] * self.opacity_init_factor, # 0.95
                rgb=rgb_init[t_mask],
            )
        # d_model.scf.update_topology()
        d_model.summary()
        return d_model

    def photometric_fit(
        self,
        s2d: Saved2D,
        cams: MonocularCameras,
        s_model: StaticGaussian,
        d_model: DynSCFGaussian = None,
        optim_cam_after_steps=0,
        total_steps=8000,
        topo_update_feq=50,
        skinning_corr_start_steps=1e10,
        s_gs_ctrl_cfg: GSControlCFG = GSControlCFG(
            densify_steps=400,
            reset_steps=2000,
            prune_steps=400,
            densify_max_grad=0.00025,
            densify_percent_dense=0.01,
            prune_opacity_th=0.012,
            reset_opacity=0.01,
        ),
        d_gs_ctrl_cfg: GSControlCFG = GSControlCFG(
            densify_steps=400,
            reset_steps=2000,
            prune_steps=400,
            densify_max_grad=0.00015,
            densify_percent_dense=0.01,
            prune_opacity_th=0.012,
            reset_opacity=0.01,
        ),
        s_gs_ctrl_start_ratio=0.2,
        s_gs_ctrl_end_ratio=0.9,
        d_gs_ctrl_start_ratio=0.2,
        d_gs_ctrl_end_ratio=0.9,
        # optim
        optimizer_cfg: OptimCFG = OptimCFG(
            lr_cam_f=0.0,
            lr_cam_q=0.0001,
            lr_cam_t=0.0001,
            lr_p=0.0003,
            lr_q=0.002,
            lr_s=0.01,
            lr_o=0.1,
            lr_sph=0.005,
            # dyn
            lr_np=0.001,
            lr_nq=0.01,
            lr_w=0.3,
        ),
        # cfg loss
        lambda_rgb=1.0,
        lambda_dep=1.0,
        lambda_mask=0.5,
        dep_st_invariant=True,
        lambda_normal=1.0,
        lambda_depth_normal=0.05,  # from GOF
        lambda_distortion=100.0,  # from GOF
        lambda_arap_coord=3.0,
        lambda_arap_len=0.0,
        lambda_vel_xyz_reg=0.0,
        lambda_vel_rot_reg=0.0,
        lambda_acc_xyz_reg=0.5,
        lambda_acc_rot_reg=0.5,
        lambda_small_w_reg=0.0,
        #
        lambda_track=0.0,
        track_flow_chance=0.0,
        track_flow_interval_candidates=[1],
        track_loss_clamp=100.0,
        track_loss_protect_steps=100,
        track_loss_interval=3,  # 1/3 steps are used for track loss and does not count the grad
        track_loss_start_step=-1,
        track_loss_end_step=100000,
        ######
        reg_radius=None,
        geo_reg_start_steps=0,
        viz_interval=1000,
        viz_cheap_interval=1000,
        viz_skip_t=5,
        viz_move_angle_deg=30.0,
        phase_name="photometric",
        use_decay=True,
        decay_start=2000,
        temporal_diff_shift=[1, 3, 6],
        temporal_diff_weight=[0.6, 0.3, 0.1],
        # * error grow
        dyn_error_grow_steps=[3350, 3850], # [4750, 5750]
        dyn_error_grow_th=0.5, # 0.2
        dyn_error_grow_num_frames=4,
        dyn_error_grow_subsample=1,
        # ! warning, the corr loss is in pixel int unit!!
        dyn_node_densify_steps=[],
        dyn_node_densify_grad_th=0.2,
        dyn_node_densify_record_start_steps=2000,
        dyn_node_densify_max_gs_per_new_node=100000,
        # * scf pruning
        dyn_scf_prune_steps=[],
        dyn_scf_prune_sk_th=0.02,
        prune_control_point_steps=[3250, 3750], # [3250, 3750]
        # other
        random_bg=False,
        default_bg_color=[1.0, 1.0, 1.0],
        # * DynGS cleaning
        photo_s2d_trans_steps=[],
        # camera smoothing loss
        lambda_cam_smooth=0.001, # 0.01
        use_cam_smooth_loss=True,
        # zero123 sds loss
        lambda_zero123_sds=0.001, # 0.01
        use_zero123_sds_loss=True,
        zero123_sds_start_step=5000, # 5000 NOTE For stable training
        zero123_sds_interval=1,
        zero123_sds_render_reso_min=128,
        zero123_sds_render_reso_mid=192,
        zero123_sds_render_reso_max=256,
    ):
        logging.info(f"Finetune with GS-BACKEND={GS_BACKEND.lower()}")
        # total_steps = 100 # 8000 NOTE for debug

        torch.cuda.empty_cache()
        n_frame = 1 # batch size

        d_flag = d_model is not None # True

        corr_flag = lambda_track > 0.0 and d_flag # True; NOTE track loss, lambda_track: 0.1
        if corr_flag: # correspondence flag
            logging.info(
                f"Enabel Flow/Track backing with supervision interval={track_loss_interval}"
            ) # track_loss_interval: 3

        '''
        1. set optimizer and scheduler 2. training loop
        2.1 update optimizer and scheduler 2.2 render 2.3 compute loss 2.4 backward and step
        '''

        # * 1. set optimizer and scheduler
        optimizer_static = torch.optim.Adam(
            s_model.get_optimizable_list(**optimizer_cfg.get_static_lr_dict)
        ) # NOTE optimizer_static
        if d_flag:
            optimizer_dynamic = torch.optim.Adam(
                d_model.get_optimizable_list(**optimizer_cfg.get_dynamic_lr_dict)
            ) # NOTE optimizer_dynamic
            if reg_radius is None: # None
                reg_radius = int(np.array(temporal_diff_shift).max()) * 2 # 12 NOTE arap loss detal_time range
            logging.info(f"Set reg_radius={reg_radius} for dynamic model")
            sup_mask_type = "all" # supervision mask type
        else:
            sup_mask_type = "static" # supervision mask type NOTE only supervision static region
        cam_param_list = cams.get_optimizable_list(**optimizer_cfg.get_cam_lr_dict)[:2] # NOTE exclude focal, only R and t
        if len(cam_param_list) > 0:
            optimizer_cam = torch.optim.Adam(
                cams.get_optimizable_list(**optimizer_cfg.get_cam_lr_dict)[:2]
            ) # NOTE optimizer_cam, iphone: R,t
        else:
            optimizer_cam = None
        if use_decay: # iphone: True
            gs_scheduling_func_dict, cam_scheduling_func_dict = (
                optimizer_cfg.get_scheduler(total_steps=total_steps - decay_start) # 8000 - 2000
            )
        else:
            gs_scheduling_func_dict, cam_scheduling_func_dict = {}, {}

        loss_rgb_list, loss_dep_list, loss_nrm_list = [], [], []
        loss_mask_list = []
        loss_dep_nrm_reg_list, loss_distortion_reg_list = [], []

        loss_arap_coord_list, loss_arap_len_list = [], []
        loss_vel_xyz_reg_list, loss_vel_rot_reg_list = [], []
        loss_acc_xyz_reg_list, loss_acc_rot_reg_list = [], []
        s_n_count_list, d_n_count_list = [], []
        d_m_count_list = []
        loss_sds_list = []
        loss_small_w_list = []
        loss_track_list = []

        s_gs_ctrl_start = int(total_steps * s_gs_ctrl_start_ratio) # 80
        d_gs_ctrl_start = int(total_steps * d_gs_ctrl_start_ratio) # 80
        s_gs_ctrl_end = int(total_steps * s_gs_ctrl_end_ratio) # 7200
        d_gs_ctrl_end = int(total_steps * d_gs_ctrl_end_ratio) # 7200
        assert s_gs_ctrl_start >= 0
        assert d_gs_ctrl_start >= 0

        latest_track_event = 0 # for stable track loss compute after gaussian densify and prune
        base_u, base_v = np.meshgrid(np.arange(s2d.W), np.arange(s2d.H)) # base_u range: 0~W, base_v range: 0~H
        base_uv = np.stack([base_u, base_v], -1)
        base_uv = torch.tensor(base_uv, device=s2d.rgb.device).long() # H,W,2

        # prepare a color-plate for seamntic rendering
        if d_flag: # True
            # ! for now the group rendering only works for dynamic joitn mode
            n_group_static = len(s_model.group_id.unique()) # 1
            n_group_dynamic = len(d_model.scf.unique_grouping) # 1
            color_plate = get_colorplate(n_group_static + n_group_dynamic) # [n_group_static+n_group_dynamic, 3]
            # random permute
            color_permute = torch.randperm(len(color_plate))
            color_plate = color_plate[color_permute]
            s_model.get_cate_color(
                color_plate=color_plate[:n_group_static].to(s_model.device)
            ) # NOTE instance color
            d_model.get_cate_color(
                color_plate=color_plate[n_group_static:].to(d_model.device)
            ) # NOTE instnace color

        for step in tqdm(range(total_steps)): # 8000
            # NOTE control w_correction_flag and fixed_topology_flag
            if d_flag and step == skinning_corr_start_steps: # 6000
                logging.info(
                    f"at {step} stop all the topology update and add skinning weight correction"
                )
                d_model.set_surface_deform() # open w_correction_flag, open fixed_topology_flag
            corr_exe_flag = ( # NOTE execute track loss flag
                corr_flag # iphone: True
                and step > latest_track_event + track_loss_protect_steps # 0+100, NOTE latest_track_event
                and step % track_loss_interval == 0 # 3
                and step >= track_loss_start_step # -1
                and step < track_loss_end_step # 100000
            )
            
            # * 2.1 update optimizer and scheduler 
            optimizer_static.zero_grad(set_to_none=True)
            if optimizer_cam is not None: # iphone: R,t
                optimizer_cam.zero_grad(set_to_none=True)
            if d_flag:
                optimizer_dynamic.zero_grad(set_to_none=True)
                if step % topo_update_feq == 0: # 50
                    d_model.scf.update_topology() # no mask! MARK: updata topology

            if step > decay_start: # 2000
                for k, v in gs_scheduling_func_dict.items():
                    update_learning_rate(v(step), k, optimizer_static)
                    if d_flag:
                        update_learning_rate(v(step), k, optimizer_dynamic)
                if optimizer_cam is not None:
                    for k, v in cam_scheduling_func_dict.items():
                        update_learning_rate(v(step), k, optimizer_cam)

            view_ind_list = np.random.choice(cams.T, n_frame, replace=False).tolist() # NOTE random sample
            if corr_exe_flag: # NOTE execute track loss, correspondence execute flag
                # select another ind different than the view_ind_list
                corr_dst_ind_list, corr_flow_flag_list = [], []
                for view_ind in view_ind_list:
                    flow_flag = np.random.rand() < track_flow_chance
                    corr_flow_flag_list.append(flow_flag)
                    if flow_flag:
                        corr_dst_ind_candidates = []
                        for flow_interval in track_flow_interval_candidates: # [1 3]
                            if view_ind + flow_interval < cams.T:
                                corr_dst_ind_candidates.append(view_ind + flow_interval)
                            if view_ind - flow_interval >= 0:
                                corr_dst_ind_candidates.append(view_ind - flow_interval)
                        corr_dst_ind = np.random.choice(corr_dst_ind_candidates)
                        corr_dst_ind_list.append(corr_dst_ind)
                    else:
                        corr_dst_ind = view_ind
                        while corr_dst_ind == view_ind: # bad code
                            corr_dst_ind = np.random.choice(cams.T)
                        corr_dst_ind_list.append(corr_dst_ind)
                corr_dst_ind_list = np.array(corr_dst_ind_list)
            else:
                corr_dst_ind_list = view_ind_list
                corr_flow_flag_list = [False] * n_frame

            render_dict_list, corr_render_dict_list = [], []
            loss_rgb, loss_dep, loss_nrm = 0.0, 0.0, 0.0
            loss_dep_nrm_reg, loss_distortion_reg = 0.0, 0.0
            loss_mask = 0.0
            loss_track = 0.0
            sds_view_ind = None
            sds_dyn_gs = None
            for _inner_loop_i, view_ind in enumerate(view_ind_list): # batch size loop
                dst_ind = corr_dst_ind_list[_inner_loop_i]
                flow_flag = corr_flow_flag_list[_inner_loop_i]

                # * 2.2 deform and render 
                gs5 = [list(s_model())] # [[xyz, rot, s, o, sh]]
                dyn_gs = list(d_model(view_ind)) if d_flag else None
                if _inner_loop_i == 0:
                    sds_view_ind = view_ind
                    sds_dyn_gs = dyn_gs

                add_buffer = None # what this?
                if corr_exe_flag: # NOTE execute track loss
                    # NOTE detach bg pts
                    dst_xyz = torch.cat([gs5[0][0].detach(), d_model(dst_ind)[0]], 0)
                    dst_xyz_cam = cams.trans_pts_to_cam(dst_ind, dst_xyz) # MARK: density
                    if GS_BACKEND in ["native_add3"]: # native_add3
                        add_buffer = dst_xyz_cam

                if dyn_gs is not None:
                    gs5.append(dyn_gs)
                if random_bg: # True
                    bg_color = np.random.rand(3).tolist()
                else:
                    bg_color = default_bg_color  # [1.0, 1.0, 1.0]
                if GS_BACKEND in ["natie_add3"]: # 'native_add3'
                    # the render internally has another protection, because if not set, the grad has bug
                    bg_color += [0.0, 0.0, 0.0]

                render_dict = render(
                    gs5,
                    s2d.H,
                    s2d.W,
                    cams.K(s2d.H, s2d.W),
                    cams.T_cw(view_ind), # NOTE w2c
                    bg_color=bg_color, # iphone: random
                    add_buffer=add_buffer, # None
                ) # rgb:[3,H,W] buf:[3,H,W] dep:[1,H,W] alpha:[1,H,W] viewspace_points:[N,3] visibility_filter:[N] raddi:[N]
                render_dict_list.append(render_dict)

                # * 2.3 compute losses
                rgb_sup_mask = s2d.get_mask_by_key(sup_mask_type)[view_ind] # 'all'
                _l_rgb, _, _, _ = compute_rgb_loss(
                    s2d.rgb[view_ind].detach().clone(), render_dict, rgb_sup_mask
                )
                dep_sup_mask = rgb_sup_mask * s2d.dep_mask[view_ind] # NOTE process supervision mask
                _l_dep, _, _, _ = compute_dep_loss(
                    s2d.dep[view_ind].detach().clone(),
                    render_dict,
                    dep_sup_mask,
                    st_invariant=dep_st_invariant, # True, NOTE normal depth flag
                )
                loss_rgb = loss_rgb + _l_rgb
                loss_dep = loss_dep + _l_dep

                if corr_exe_flag:
                    # NOTE Track Loss
                    if GS_BACKEND in ["native_add3"]: # iphone, nvidia_marbles
                        corr_render_dict = render_dict
                        rendered_xyz_map = render_dict["buf"].permute(1, 2, 0)  # H,W,3
                    else:
                        corr_render_dict = render(
                            # # ! use detached bg gs
                            # [[it.detach() for it in gs5[0]], gs5[1]],
                            # ! debug, align wiht .bck old version
                            gs5,
                            s2d.H,
                            s2d.W,
                            cams.K(s2d.H, s2d.W),
                            cams.T_cw(view_ind),
                            bg_color=[0.0, 0.0, 0.0],
                            colors_precomp=dst_xyz_cam,
                        )
                        rendered_xyz_map = corr_render_dict["rgb"].permute(
                            1, 2, 0
                        )  # H,W,3
                    corr_render_dict_list.append(corr_render_dict)
                    # get the flow
                    with torch.no_grad():
                        if flow_flag: # True or False NOTE random
                            flow_ind = s2d.flow_ij_to_listind_dict[(view_ind, dst_ind)]
                            flow = s2d.flow[flow_ind].detach().clone() # H,W,2
                            flow_mask = s2d.flow_mask[flow_ind].detach().clone().bool() # H,W
                            track_src = base_uv.clone().detach()[flow_mask] # N_valid,2
                            flow = flow[flow_mask] # N_valid,2
                            track_dst = track_src.float() + flow # N_valid,2
                        else:
                            # contruct target by track
                            track_valid = (
                                s2d.track_mask[view_ind] & s2d.track_mask[dst_ind]
                            )
                            track_src = s2d.track[view_ind][track_valid][..., :2] # N_valid,2
                            track_dst = s2d.track[dst_ind][track_valid][..., :2] # N_valid,2
                        src_fetch_index = (
                            track_src[:, 1].long() * s2d.W + track_src[:, 0].long()
                        ) # N_valid
                    if len(track_src) == 0:
                        _loss_track = torch.zeros_like(_l_rgb)
                    else:
                        warped_xyz_cam = rendered_xyz_map.reshape(-1, 3)[
                            src_fetch_index
                        ] # N_valid,3
                        # filter the pred, only add loss to points that are infront of the camera
                        track_loss_mask = warped_xyz_cam[:, 2] > 1e-4 # N_valid
                        if track_loss_mask.sum() == 0:
                            _loss_track = torch.zeros_like(_l_rgb)
                        else:
                            pred_track_dst = cams.project(warped_xyz_cam) # N_valid,2 NOTE value range [-1, 1]
                            L = min(s2d.W, s2d.H)
                            pred_track_dst[:, :1] = (
                                (pred_track_dst[:, :1] + s2d.W / L) / 2.0 * L
                            ) # NOTE value range [0, W]
                            pred_track_dst[:, 1:] = (
                                (pred_track_dst[:, 1:] + s2d.H / L) / 2.0 * L
                            ) # NOTE value range [0, H]
                            _loss_track = (pred_track_dst - track_dst).norm(dim=-1)[
                                track_loss_mask
                            ]
                            _loss_track = torch.clamp(
                                _loss_track, 0.0, track_loss_clamp
                            ) # NOTE track_loss_clamp 100.0
                            _loss_track = _loss_track.mean()
                else:
                    _loss_track = torch.zeros_like(_l_rgb)
                loss_track = loss_track + _loss_track

                # GOF normal and regularization
                if GS_BACKEND == "gof": # 'native_add3'
                    _l_nrm, _, _, _ = compute_normal_loss(
                        s2d.nrm[view_ind].detach().clone(), render_dict, dep_sup_mask
                    )
                    loss_nrm = loss_nrm + _l_nrm # NOTE loss_nrm
                    if step > geo_reg_start_steps:
                        _l_reg_nrm, _, _, _ = compute_normal_reg_loss(
                            s2d, cams, render_dict
                        )
                        _l_reg_distortion, _ = compute_dep_reg_loss(
                            s2d.rgb[view_ind].detach().clone(), render_dict
                        )
                    else:
                        _l_reg_nrm = torch.zeros_like(_l_rgb)
                        _l_reg_distortion = torch.zeros_like(_l_rgb)
                    loss_dep_nrm_reg = loss_dep_nrm_reg + _l_reg_nrm 
                    loss_distortion_reg = loss_distortion_reg + _l_reg_distortion 
                else:
                    loss_nrm = torch.zeros_like(loss_rgb) 
                    loss_dep_nrm_reg = torch.zeros_like(loss_rgb) 
                    loss_distortion_reg = torch.zeros_like(loss_rgb) 

                # instance mask loss
                if d_flag and lambda_mask > 0.0: # iphone: lambda_mask: 0.0
                    # * do the mask loss, including the background
                    s_cate_sph, s_gid2color = s_model.get_cate_color(
                        perm=torch.randperm(len(s_model.group_id.unique()))
                    )
                    d_cate_sph, d_gid2color = d_model.get_cate_color(
                        perm=torch.randperm(len(d_model.scf.unique_grouping))
                    )
                    with torch.no_grad():
                        inst_map = s2d.inst[view_ind]
                        gt_mask = torch.zeros_like(s2d.rgb[0]) # H,W
                        for gid, color in d_gid2color.items():
                            gt_mask[inst_map == gid] = color[None]
                        for gid, color in s_gid2color.items():
                            gt_mask[inst_map == gid] = color[None]
                    gs5[1][-1] = d_cate_sph
                    gs5[0][-1] = s_cate_sph
                    render_dict = render(
                        gs5,
                        s2d.H,
                        s2d.W,
                        cams.K(s2d.H, s2d.W),
                        cams.T_cw(view_ind),
                        bg_color=[0.0, 0.0, 0.0],
                    )
                    pred_mask = render_dict["rgb"].permute(1, 2, 0)
                    l_mask = torch.nn.functional.mse_loss(pred_mask, gt_mask)
                    loss_mask = loss_mask + l_mask # NOTE loss_mask
                else:
                    loss_mask = torch.zeros_like(loss_rgb) # NOTE loss_mask
            # arap-loss, acc-loss, vel-loss
            if d_flag: # True
                _l = max(0, view_ind_list[0] - reg_radius) # left
                _r = min(cams.T, view_ind_list[0] + 1 + reg_radius) # right
                reg_tids = torch.arange(_l, _r, device=s_model.device)
            if (lambda_arap_coord > 0.0 or lambda_arap_len > 0.0) and d_flag: # iphone: 10.0, 10.0
                loss_arap_coord, loss_arap_len = d_model.scf.compute_arap_loss(
                    reg_tids,
                    temporal_diff_shift=temporal_diff_shift, # [1 3 6]
                    temporal_diff_weight=temporal_diff_weight, # [0.6 0.3 0.1]
                )
                assert torch.isnan(loss_arap_coord).sum() == 0
                assert torch.isnan(loss_arap_len).sum() == 0
            else:
                loss_arap_coord = torch.zeros_like(loss_rgb)
                loss_arap_len = torch.zeros_like(loss_rgb)

            if (
                lambda_vel_xyz_reg > 0.0    # iphone: 5.0
                or lambda_vel_rot_reg > 0.0 # iphone: 5.0
                or lambda_acc_xyz_reg > 0.0 # iphone: 5.0
                or lambda_acc_rot_reg > 0.0 # iphone: 5.0
            ) and d_flag:
                (
                    loss_vel_xyz_reg,
                    loss_vel_rot_reg,
                    loss_acc_xyz_reg,
                    loss_acc_rot_reg,
                ) = d_model.scf.compute_vel_acc_loss(reg_tids)
                assert torch.isnan(loss_vel_xyz_reg).sum() == 0
                assert torch.isnan(loss_vel_rot_reg).sum() == 0
                assert torch.isnan(loss_acc_xyz_reg).sum() == 0
                assert torch.isnan(loss_acc_rot_reg).sum() == 0
            else:
                loss_vel_xyz_reg = loss_vel_rot_reg = loss_acc_xyz_reg = (
                    loss_acc_rot_reg
                ) = torch.zeros_like(loss_rgb)

            if d_flag: # skinning weight reg loss
                loss_small_w = abs(d_model._skinning_weight).mean() # d_model._skinning_weight: N_dyn,16
            else:
                loss_small_w = torch.zeros_like(loss_rgb)
            
            if d_flag and use_cam_smooth_loss: # True
                loss_cam_smooth = cams.camera_smoothness_loss() # NOTE camera smooth loss
            else:
                loss_cam_smooth = torch.zeros_like(loss_rgb)
            step_ratio = 1.0 * (step + 1) / total_steps
            run_zero123_sds = (
                d_flag
                and use_zero123_sds_loss
                and step > zero123_sds_start_step
                and step % zero123_sds_interval == 0
            )

            if run_zero123_sds:
                # sds_start = time.time()
                if step_ratio < 0.6:
                    sds_hw = zero123_sds_render_reso_min
                elif step_ratio < 0.9:
                    sds_hw = zero123_sds_render_reso_mid
                else:
                    sds_hw = zero123_sds_render_reso_max

                with torch.no_grad():
                    focus_depth = estimate_focus_depth_from_dyn_gs(
                        sds_dyn_gs, cams, sds_view_ind
                    )
                    pseudo_pose, focus_point = get_move_around_cam_T_cw(
                        cams,
                        sds_view_ind,
                        focus_depth,
                        move_around_angle_deg=np.pi / 36,
                    )
                    sds_K = cams.K(sds_hw, sds_hw).detach()

                    pseudo_pose_inv = torch.linalg.inv(pseudo_pose)
                    train_view_pose = cams.T_wc(sds_view_ind).detach()
                    pseudo_pose_center = pseudo_pose_inv[:3, 3]
                    train_view_pose_center = train_view_pose[:3, 3]
                    focus_point_world = (
                        train_view_pose[:3, :3] @ focus_point + train_view_pose[:3, 3]
                    )
                    ver, hor = calculate_relative_angles(
                        focus_point_world,
                        train_view_pose_center,
                        pseudo_pose_center,
                    )
                    radii = 0.0

                sds_render_dict = render(
                    [sds_dyn_gs],
                    sds_hw,
                    sds_hw,
                    sds_K,
                    pseudo_pose,
                    bg_color=[0.0, 0.0, 0.0],
                )
                sds_images = sds_render_dict["rgb"].unsqueeze(0)

                loss_zero123_sds = s2d.guidance_zero123.train_step(
                    sds_images,
                    np.array([ver]),
                    np.array([hor]),
                    np.array([radii]),
                    sds_view_ind,
                    step_ratio,
                )
                
                # print(f"sds time{time.time()-sds_start:.2f}s") # 0.5 second
            else:
                loss_zero123_sds = torch.zeros_like(loss_rgb)
            

            loss = (
                loss_rgb * lambda_rgb
                + loss_dep * lambda_dep
                + loss_mask * lambda_mask
                + loss_nrm * lambda_normal
                + loss_dep_nrm_reg * lambda_depth_normal
                + loss_distortion_reg * lambda_distortion
                + loss_arap_coord * lambda_arap_coord
                + loss_arap_len * lambda_arap_len
                + loss_vel_xyz_reg * lambda_vel_xyz_reg
                + loss_vel_rot_reg * lambda_vel_rot_reg
                + loss_acc_xyz_reg * lambda_acc_xyz_reg
                + loss_acc_rot_reg * lambda_acc_rot_reg
                + loss_small_w * lambda_small_w_reg # skinning weight reg loss
                + loss_track * lambda_track
                + loss_cam_smooth * lambda_cam_smooth
                + loss_zero123_sds * lambda_zero123_sds # zero123 sds loss
            )

            # back_t = time.time()
            loss.backward()

            optimizer_static.step() # static gaussian
            if d_flag:
                optimizer_dynamic.step() # dynamic gaussian
            if step >= optim_cam_after_steps and optimizer_cam is not None:
                optimizer_cam.step() # camera
            # print(f"backward in {time.time()-back_t:.2f}s") # 40 second

            # * 2.4 GS control
            # d_model to s_model transfer [1] copy the d_gs5
            dynamic_to_static_transfer_flag = step in photo_s2d_trans_steps and d_flag # [3000 4000 5000]
            if dynamic_to_static_transfer_flag:
                with torch.no_grad():
                    # before the gs control to append full opacity GS
                    random_select_t = np.random.choice(cams.T)
                    trans_d_gs5 = d_model(random_select_t)
                    logging.info(f"Transfer dynamic to static at step={step}")

            if (
                (
                    step in d_gs_ctrl_cfg.reset_steps # [0 1000 2000 3000 ...]
                    and step >= d_gs_ctrl_start # 80
                    and step < d_gs_ctrl_end # 7200
                )
                or (
                    step in s_gs_ctrl_cfg.reset_steps # [0 1000 2000 3000 ...]
                    and step >= s_gs_ctrl_start # 80
                    and step < s_gs_ctrl_end # 7200
                )
                or dynamic_to_static_transfer_flag # [3000 4000 5000]
            ):
                if corr_flag: # iphone: True
                    logging.info(f"Reset event happened, protect tracking loss")
                    latest_track_event = step # NOTE update latest_track_event

            if (
                s_gs_ctrl_cfg is not None
                and step >= s_gs_ctrl_start
                and step < s_gs_ctrl_end
            ):
                apply_gs_control(
                    render_list=render_dict_list,
                    model=s_model,
                    gs_control_cfg=s_gs_ctrl_cfg,
                    step=step,
                    optimizer_gs=optimizer_static,
                    first_N=s_model.N, # static gs number
                    record_flag=(not corr_exe_flag)
                    or (GS_BACKEND not in ["native_add3"]),
                ) # static gaussian
            if (
                d_gs_ctrl_cfg is not None
                and step >= d_gs_ctrl_start
                and step < d_gs_ctrl_end
                and d_flag
            ):
                apply_gs_control(
                    render_list=render_dict_list,
                    model=d_model,
                    gs_control_cfg=d_gs_ctrl_cfg,
                    step=step,
                    optimizer_gs=optimizer_dynamic,
                    last_N=d_model.N, # dynamic gs number
                    record_flag=(not corr_exe_flag)
                    or (GS_BACKEND not in ["native_add3"]),
                ) # dynamic gaussian

                if corr_exe_flag and step > dyn_node_densify_record_start_steps: # 2000
                    # record the geo gradient
                    for corr_render_dict in corr_render_dict_list:
                        d_model.record_corr_grad(
                            # ! normalize the gradient by loss weight.
                            corr_render_dict["viewspace_points"].grad[-d_model.N :] # only dynamic gaussian
                            / lambda_track,
                            corr_render_dict["visibility_filter"][-d_model.N :], # only dynamic gaussian
                        )

            # d_model to s_model transfer [2] append to static model
            if dynamic_to_static_transfer_flag: # [3000 4000 5000]
                s_model.append_gs(optimizer_static, *trans_d_gs5, new_group_id=None)

            if d_flag and step in dyn_node_densify_steps: # [4250 5250]
                d_model.gradient_based_node_densification(
                    optimizer_dynamic,
                    gradient_th=dyn_node_densify_grad_th, # 0.03
                    max_gs_per_new_node=dyn_node_densify_max_gs_per_new_node, # 100000
                )

            # NOTE densify nodes gaussians
            if d_flag and step in dyn_error_grow_steps: # [4750 5750]
                error_grow_node_model(
                    s2d,
                    cams,
                    s_model,
                    d_model,
                    optimizer_dynamic,
                    step,
                    dyn_error_grow_th,
                    viz_dir=self.viz_dir,
                )
            if d_flag and step in dyn_scf_prune_steps: # [4500 5500 6500]
                d_model.prune_nodes(
                    optimizer_dynamic,
                    prune_sk_th=dyn_scf_prune_sk_th, # 0.02
                    viz_fn=osp.join(self.viz_dir, f"scf_node_prune_at_step={step}"), # scf_node_prune_at_step=4500prune_sk_hist.jpg
                )
            # NOTE prune control points
            if d_flag and step in prune_control_point_steps: # [3250 6250]
                prune_t = d_model.scf.se3_bspline.prune_control_points()  
                d_model.scf.prune_control_points(prune_t)

            # NOTE COMPUTE SE(3) B-SPLINE
            if d_flag:
                # start_t = time.time()
                d_model.scf.compute_se3_bspline()
                # print(f"compute se3 in {time.time()-start_t:.2f}s") # 4 second


            loss_rgb_list.append(loss_rgb.item())
            loss_dep_list.append(loss_dep.item())
            loss_nrm_list.append(loss_nrm.item())
            loss_mask_list.append(loss_mask.item())

            loss_dep_nrm_reg_list.append(loss_dep_nrm_reg.item())
            loss_distortion_reg_list.append(loss_distortion_reg.item())

            loss_arap_coord_list.append(loss_arap_coord.item())
            loss_arap_len_list.append(loss_arap_len.item())
            loss_vel_xyz_reg_list.append(loss_vel_xyz_reg.item())
            loss_vel_rot_reg_list.append(loss_vel_rot_reg.item())
            loss_acc_xyz_reg_list.append(loss_acc_xyz_reg.item())
            loss_acc_rot_reg_list.append(loss_acc_rot_reg.item())
            s_n_count_list.append(s_model.N)
            d_n_count_list.append(d_model.N if d_flag else 0)
            d_m_count_list.append(d_model.M if d_flag else 0)

            loss_small_w_list.append(loss_small_w.item())
            loss_track_list.append(loss_track.item())

            # * 2.5 viz NOTE NO CHANGE
            viz_flag = viz_interval > 0 and (step % viz_interval == 0) # viz_interval: -1
            if viz_flag: # iphone: False, nvidia: Fasle

                if d_flag:
                    viz_hist(d_model, self.viz_dir, f"{phase_name}_step={step}_dynamic")
                    viz_dyn_hist(
                        d_model.scf,
                        self.viz_dir,
                        f"{phase_name}_step={step}_dynamic",
                    )
                    viz_path = osp.join(
                        self.viz_dir, f"{phase_name}_step={step}_3dviz.gif"
                    )
                    viz3d_total_video(
                        cams,
                        d_model,
                        0,
                        cams.T - 1,
                        save_path=viz_path,
                        res=480,  # 240
                        s_model=s_model,
                    )

                    # * viz grouping
                    if lambda_mask > 0.0:
                        d_model.return_cate_colors_flag = True
                        viz_path = osp.join(
                            self.viz_dir, f"{phase_name}_step={step}_3dviz_group.gif"
                        )
                        viz3d_total_video(
                            cams,
                            d_model,
                            0,
                            cams.T - 1,
                            save_path=viz_path,
                            res=480,  # 240
                            s_model=s_model,
                        )
                        viz2d_total_video(
                            viz_vid_fn=osp.join(
                                self.viz_dir,
                                f"{phase_name}_step={step}_2dviz_group.gif",
                            ),
                            s2d=s2d,
                            start_from=0,
                            end_at=cams.T - 1,
                            skip_t=viz_skip_t,
                            cams=cams,
                            s_model=s_model,
                            d_model=d_model,
                            subsample=1,
                            mask_type=sup_mask_type,
                            move_around_angle_deg=viz_move_angle_deg,
                        )
                        d_model.return_cate_colors_flag = False

                viz_hist(s_model, self.viz_dir, f"{phase_name}_step={step}_static")
                viz2d_total_video(
                    viz_vid_fn=osp.join(
                        self.viz_dir, f"{phase_name}_step={step}_2dviz.gif"
                    ),
                    s2d=s2d,
                    start_from=0,
                    end_at=cams.T - 1,
                    skip_t=viz_skip_t,
                    cams=cams,
                    s_model=s_model,
                    d_model=d_model,
                    subsample=1,
                    mask_type=sup_mask_type,
                    move_around_angle_deg=viz_move_angle_deg,
                )
            # NOTE viz cheap
            if viz_cheap_interval > 0 and (
                step % viz_cheap_interval == 0 or step == total_steps - 1
            ): # viz_cheap_interval: 1001
                # viz the accumulated grad
                with torch.no_grad():
                    photo_grad = [
                        s_model.xyz_gradient_accum
                        / torch.clamp(s_model.xyz_gradient_denom, min=1e-6)
                    ] # only static gaussian
                    corr_grad = [torch.zeros_like(photo_grad[0])] # only static gaussian
                    if d_flag: # add dynamic gaussian
                        photo_grad.append(
                            d_model.xyz_gradient_accum
                            / torch.clamp(d_model.xyz_gradient_denom, min=1e-6)
                        )
                        corr_grad.append(
                            d_model.corr_gradient_accum
                            / torch.clamp(d_model.corr_gradient_denom, min=1e-6)
                        )

                    photo_grad = torch.cat(photo_grad, 0)
                    viz_grad_color = (
                        torch.clamp(photo_grad, 0.0, d_gs_ctrl_cfg.densify_max_grad)
                        / d_gs_ctrl_cfg.densify_max_grad
                    )
                    viz_grad_color = viz_grad_color.detach().cpu().numpy()
                    viz_grad_color = cm.viridis(viz_grad_color)[:, :3]
                    viz_render_dict = render(
                        [s_model(), d_model(view_ind)],
                        s2d.H,
                        s2d.W,
                        cams.K(s2d.H, s2d.W),
                        cams.T_cw(view_ind),
                        bg_color=[0.0, 0.0, 0.0],
                        colors_precomp=torch.from_numpy(viz_grad_color).to(photo_grad),
                    )
                    viz_grad = (
                        viz_render_dict["rgb"].permute(1, 2, 0).detach().cpu().numpy()
                    )
                    imageio.imsave(
                        osp.join(
                            self.viz_dir, f"{phase_name}_photo_grad_step={step}.jpg"
                        ),
                        (viz_grad*255).astype(np.uint8),
                    ) 

                    corr_grad = torch.cat(corr_grad, 0)
                    viz_grad_color = (
                        torch.clamp(corr_grad, 0.0, dyn_node_densify_grad_th)
                        / dyn_node_densify_grad_th
                    )
                    viz_grad_color = viz_grad_color.detach().cpu().numpy()
                    viz_grad_color = cm.viridis(viz_grad_color)[:, :3]
                    viz_render_dict = render(
                        [s_model(), d_model(view_ind)],
                        s2d.H,
                        s2d.W,
                        cams.K(s2d.H, s2d.W),
                        cams.T_cw(view_ind),
                        bg_color=[0.0, 0.0, 0.0],
                        colors_precomp=torch.from_numpy(viz_grad_color).to(corr_grad),
                    )
                    viz_grad = (
                        viz_render_dict["rgb"].permute(1, 2, 0).detach().cpu().numpy()
                    )
                    imageio.imsave(
                        osp.join(
                            self.viz_dir, f"{phase_name}_corr_grad_step={step}.jpg"
                        ),
                        (viz_grad*255).astype(np.uint8),
                    ) 

                fig = plt.figure(figsize=(30, 8))
                for plt_i, plt_pack in enumerate(
                    [
                        ("loss_rgb", loss_rgb_list),
                        ("loss_dep", loss_dep_list),
                        ("loss_nrm", loss_nrm_list),
                        ("loss_mask", loss_mask_list),
                        ("loss_sds", loss_sds_list),
                        ("loss_dep_nrm_reg", loss_dep_nrm_reg_list),
                        ("loss_distortion_reg", loss_distortion_reg_list),
                        ("loss_arap_coord", loss_arap_coord_list),
                        ("loss_arap_len", loss_arap_len_list),
                        ("loss_vel_xyz_reg", loss_vel_xyz_reg_list),
                        ("loss_vel_rot_reg", loss_vel_rot_reg_list),
                        ("loss_acc_xyz_reg", loss_acc_xyz_reg_list),
                        ("loss_acc_rot_reg", loss_acc_rot_reg_list),
                        ("loss_small_w", loss_small_w_list),
                        ("loss_track", loss_track_list),
                        ("S-N", s_n_count_list),
                        ("D-N", d_n_count_list),
                        ("D-M", d_m_count_list),
                    ]
                ):
                    plt.subplot(2, 10, plt_i + 1)
                    value_end = 0 if len(plt_pack[1]) == 0 else plt_pack[1][-1]
                    plt.plot(plt_pack[1]), plt.title(
                        plt_pack[0] + f" End={value_end:.4f}"
                    )
                plt.savefig(
                    osp.join(self.viz_dir, f"{phase_name}_optim_loss_step={step}.jpg")
                ) 
                plt.close()

        # * 3. save models
        # after training, save static, camera and dynamic model; MARK: save ckpt
        s_save_fn = osp.join(
            self.log_dir, f"{phase_name}_s_model_{GS_BACKEND.lower()}.pth"
        ) 
        torch.save(s_model.state_dict(), s_save_fn) 
        torch.save(cams.state_dict(), osp.join(self.log_dir, f"{phase_name}_cam.pth")) 

        if d_model is not None:
            d_save_fn = osp.join(
                self.log_dir, f"{phase_name}_d_model_{GS_BACKEND.lower()}.pth"
            ) 
            torch.save(d_model.state_dict(), d_save_fn) 

        # * 4. viz
        fig = plt.figure(figsize=(30, 8))
        for plt_i, plt_pack in enumerate(
            [
                ("loss_rgb", loss_rgb_list),
                ("loss_dep", loss_dep_list),
                ("loss_nrm", loss_nrm_list),
                ("loss_mask", loss_mask_list),
                ("loss_dep_nrm_reg", loss_dep_nrm_reg_list),
                ("loss_distortion_reg", loss_distortion_reg_list),
                ("loss_arap_coord", loss_arap_coord_list),
                ("loss_arap_len", loss_arap_len_list),
                ("loss_vel_xyz_reg", loss_vel_xyz_reg_list),
                ("loss_vel_rot_reg", loss_vel_rot_reg_list),
                ("loss_acc_xyz_reg", loss_acc_xyz_reg_list),
                ("loss_acc_rot_reg", loss_acc_rot_reg_list),
                ("loss_small_w", loss_small_w_list),
                ("loss_track", loss_track_list),
                ("S-N", s_n_count_list),
                ("D-N", d_n_count_list),
                ("D-M", d_m_count_list),
            ]
        ):
            plt.subplot(2, 10, plt_i + 1)
            plt.plot(plt_pack[1]), plt.title(
                plt_pack[0] + f" End={plt_pack[1][-1]:.6f}"
            )
        plt.savefig(osp.join(self.log_dir, f"{phase_name}_optim_loss.jpg")) 
        plt.close()
        viz2d_total_video(
            viz_vid_fn=osp.join(self.log_dir, f"{phase_name}_2dviz.gif"), 
            s2d=s2d,
            start_from=0,
            end_at=cams.T - 1,
            skip_t=viz_skip_t, # 1
            cams=cams,
            s_model=s_model,
            d_model=d_model,
            move_around_angle_deg=viz_move_angle_deg, # 10.0
            print_text=False,
        )
        viz_path = osp.join(self.log_dir, f"{phase_name}_3Dviz.gif")
        if d_flag:
            viz3d_total_video(
                cams,
                d_model,
                0, # start_t
                cams.T - 1, # end_t
                save_path=viz_path, # photometric_3Dviz.gif
                res=480,
                s_model=s_model,
            )
            if lambda_mask > 0.0: # iphone: 0.0 nvidia: 0.0
                # * viz grouping
                d_model.return_cate_colors_flag = True
                s_model.return_cate_colors_flag = True
                viz_path = osp.join(self.log_dir, f"{phase_name}_3Dviz_group.gif") 
                viz3d_total_video(
                    cams,
                    d_model,
                    0,
                    cams.T - 1,
                    save_path=viz_path,
                    res=480,
                    s_model=s_model,
                )
                viz2d_total_video(
                    viz_vid_fn=osp.join(self.log_dir, f"{phase_name}_2dviz_group.gif"), 
                    s2d=s2d,
                    start_from=0,
                    end_at=cams.T - 1,
                    skip_t=viz_skip_t,
                    cams=cams,
                    s_model=s_model,
                    d_model=d_model,
                    move_around_angle_deg=viz_move_angle_deg,
                    print_text=False,
                )
                d_model.return_cate_colors_flag = False
                s_model.return_cate_colors_flag = False
        torch.cuda.empty_cache()
        return

    @torch.no_grad()
    def render_all(self, cams: MonocularCameras, s_model=None, d_model=None):
        ret = []
        assert s_model is not None or d_model is not None, "No model to render"
        s_gs5 = s_model()
        for t in tqdm(range(cams.T)):
            gs5 = [s_gs5]
            if d_model is not None:
                gs5.append(d_model(t))
            render_dict = render(
                gs5,
                cams.default_H,
                cams.default_W,
                cams.K(),
                cams.T_cw(t),
                bg_color=[1.0, 1.0, 1.0],
            )
            ret.append(render_dict)
        rgb = torch.stack([r["rgb"] for r in ret], 0)
        dep = torch.stack([r["dep"] for r in ret], 0)
        alp = torch.stack([r["alpha"] for r in ret], 0)
        return rgb, dep, alp
