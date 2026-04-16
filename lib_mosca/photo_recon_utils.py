# useful function for reconstruction solver

# Single File
from matplotlib import pyplot as plt
import torch, numpy as np
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_quaternion,
    quaternion_to_matrix,
    quaternion_to_axis_angle,
)
import os, sys, os.path as osp
import torch.nn.functional as F
from tqdm import tqdm
from pytorch3d.ops import knn_points
from torch import nn
import kornia
import logging
import imageio
import colorsys


sys.path.append(osp.dirname(osp.abspath(__file__)))

from camera import MonocularCameras

from dynamic_gs import DynSCFGaussian
from static_gs import StaticGaussian
from gs_utils.gs_optim_helper import update_learning_rate, get_expon_lr_func

import logging
import sys, os, os.path as osp


def apply_gs_control(
    render_list,
    model,
    gs_control_cfg,
    step,
    optimizer_gs,
    first_N=None,
    last_N=None,
    record_flag=True,
):
    for render_dict in render_list:
        if first_N is not None:
            assert last_N is None
            grad = render_dict["viewspace_points"].grad[:first_N]
            radii = render_dict["radii"][:first_N]
            visib = render_dict["visibility_filter"][:first_N]
        elif last_N is not None:
            assert first_N is None
            grad = render_dict["viewspace_points"].grad[-last_N:]
            radii = render_dict["radii"][-last_N:]
            visib = render_dict["visibility_filter"][-last_N:]
        else:
            grad = render_dict["viewspace_points"].grad
            radii = render_dict["radii"]
            visib = render_dict["visibility_filter"]
        if record_flag: # True
            model.record_xyz_grad_radii(grad, radii, visib)
    if (
        step in gs_control_cfg.densify_steps # [0 300 600 900 ...]
        or step in gs_control_cfg.prune_steps # [0 300 600 900 ...]
        or step in gs_control_cfg.reset_steps # [0 1000 2000 3000 ...]
    ):
        logging.info(f"GS Control at {step}")
    if step in gs_control_cfg.densify_steps:
        N_old = model.N
        model.densify(
            optimizer=optimizer_gs,
            max_grad=gs_control_cfg.densify_max_grad,
            percent_dense=gs_control_cfg.densify_percent_dense,
            extent=0.5,
            verbose=True,
        )
        logging.info(f"Densify: {N_old}->{model.N}")
    if step in gs_control_cfg.prune_steps:
        N_old = model.N
        model.prune_points(
            optimizer_gs,
            min_opacity=gs_control_cfg.prune_opacity_th,
            max_screen_size=1e10,  # disabled
        )
        logging.info(f"Prune: {N_old}->{model.N}")
    if step in gs_control_cfg.reset_steps:
        model.reset_opacity(optimizer_gs, gs_control_cfg.reset_opacity)
    return


class GSControlCFG:
    def __init__(
        self,
        densify_steps=300,
        reset_steps=900,
        prune_steps=300,
        densify_max_grad=0.0002,
        densify_percent_dense=0.01,
        prune_opacity_th=0.012,
        reset_opacity=0.01,
    ):
        if isinstance(densify_steps, int):
            densify_steps = [densify_steps * i for i in range(100000)]
        if isinstance(reset_steps, int):
            reset_steps = [reset_steps * i for i in range(100000)]
        if isinstance(prune_steps, int):
            prune_steps = [prune_steps * i for i in range(100000)]
        self.densify_steps = densify_steps
        self.reset_steps = reset_steps
        self.prune_steps = prune_steps
        self.densify_max_grad = densify_max_grad
        self.densify_percent_dense = densify_percent_dense
        self.prune_opacity_th = prune_opacity_th
        self.reset_opacity = reset_opacity
        self.summary()
        return

    def summary(self):
        logging.info("GSControlCFG: Summary")
        logging.info(
            f"GSControlCFG: densify_steps={self.densify_steps[:min(5, len(self.densify_steps))]}..."
        )
        logging.info(
            f"GSControlCFG: reset_steps={self.reset_steps[:min(5, len(self.densify_steps))]}..."
        )
        logging.info(
            f"GSControlCFG: prune_steps={self.prune_steps[:min(5, len(self.densify_steps))]}..."
        )
        logging.info(f"GSControlCFG: densify_max_grad={self.densify_max_grad}")
        logging.info(
            f"GSControlCFG: densify_percent_dense={self.densify_percent_dense}"
        )
        logging.info(f"GSControlCFG: prune_opacity_th={self.prune_opacity_th}")
        logging.info(f"GSControlCFG: reset_opacity={self.reset_opacity}")
        return


class OptimCFG:
    def __init__(
        self,
        # GS
        lr_p=0.00016,
        lr_q=0.001,
        lr_s=0.005,
        lr_o=0.05,
        lr_sph=0.0025,
        lr_sph_rest_factor=20.0,
        lr_p_final=None,
        lr_cam_q=0.0001,
        lr_cam_t=0.0001,
        lr_cam_f=0.00,
        lr_cam_q_final=None,
        lr_cam_t_final=None,
        lr_cam_f_final=None,
        # # dyn
        lr_np=0.00016,
        lr_nq=0.001,
        lr_nsig=0.00001,
        lr_w=0.0,  # ! use 0.0
        lr_dyn=0.01,
        lr_np_final=None,
        lr_nq_final=None,
        lr_w_final=None,
    ) -> None:
        # gs
        self.lr_p = lr_p
        self.lr_q = lr_q
        self.lr_s = lr_s
        self.lr_o = lr_o
        self.lr_sph = lr_sph
        self.lr_sph_rest = lr_sph / lr_sph_rest_factor
        # cam
        self.lr_cam_q = lr_cam_q
        self.lr_cam_t = lr_cam_t
        self.lr_cam_f = lr_cam_f
        # # dyn
        self.lr_np = lr_np
        self.lr_nq = lr_nq
        self.lr_w = lr_w
        self.lr_dyn = lr_dyn
        self.lr_nsig = lr_nsig

        # gs scheduler
        self.lr_p_final = lr_p_final if lr_p_final is not None else lr_p / 100.0
        self.lr_cam_q_final = (
            lr_cam_q_final if lr_cam_q_final is not None else lr_cam_q / 10.0
        )
        self.lr_cam_t_final = (
            lr_cam_t_final if lr_cam_t_final is not None else lr_cam_t / 10.0
        )
        self.lr_cam_f_final = (
            lr_cam_f_final if lr_cam_f_final is not None else lr_cam_f / 10.0
        )
        self.lr_np_final = lr_np_final if lr_np_final is not None else lr_np / 100.0
        self.lr_nq_final = lr_nq_final if lr_nq_final is not None else lr_nq / 10.0
        if lr_w is not None:
            self.lr_w_final = lr_w_final if lr_w_final is not None else lr_w / 10.0
        else:
            self.lr_w_final = 0.0
        return

    def summary(self):
        logging.info("OptimCFG: Summary")
        logging.info(f"OptimCFG: lr_p={self.lr_p}")
        logging.info(f"OptimCFG: lr_q={self.lr_q}")
        logging.info(f"OptimCFG: lr_s={self.lr_s}")
        logging.info(f"OptimCFG: lr_o={self.lr_o}")
        logging.info(f"OptimCFG: lr_sph={self.lr_sph}")
        logging.info(f"OptimCFG: lr_sph_rest={self.lr_sph_rest}")
        logging.info(f"OptimCFG: lr_cam_q={self.lr_cam_q}")
        logging.info(f"OptimCFG: lr_cam_t={self.lr_cam_t}")
        logging.info(f"OptimCFG: lr_cam_f={self.lr_cam_f}")
        logging.info(f"OptimCFG: lr_p_final={self.lr_p_final}")
        logging.info(f"OptimCFG: lr_cam_q_final={self.lr_cam_q_final}")
        logging.info(f"OptimCFG: lr_cam_t_final={self.lr_cam_t_final}")
        logging.info(f"OptimCFG: lr_cam_f_final={self.lr_cam_f_final}")
        logging.info(f"OptimCFG: lr_np={self.lr_np}")
        logging.info(f"OptimCFG: lr_nq={self.lr_nq}")
        logging.info(f"OptimCFG: lr_w={self.lr_w}")
        logging.info(f"OptimCFG: lr_dyn={self.lr_dyn}")
        logging.info(f"OptimCFG: lr_nsig={self.lr_nsig}")
        logging.info(f"OptimCFG: lr_np_final={self.lr_np_final}")
        logging.info(f"OptimCFG: lr_nq_final={self.lr_nq_final}")
        logging.info(f"OptimCFG: lr_w_final={self.lr_w_final}")
        return

    @property
    def get_static_lr_dict(self):
        return {
            "lr_p": self.lr_p,
            "lr_q": self.lr_q,
            "lr_s": self.lr_s,
            "lr_o": self.lr_o,
            "lr_sph": self.lr_sph,
            "lr_sph_rest": self.lr_sph_rest,
        }

    @property
    def get_dynamic_lr_dict(self):
        return {
            "lr_p": self.lr_p,
            "lr_q": self.lr_q,
            "lr_s": self.lr_s,
            "lr_o": self.lr_o,
            "lr_sph": self.lr_sph,
            "lr_sph_rest": self.lr_sph_rest,
            "lr_np": self.lr_np, # mosca points
            "lr_nq": self.lr_nq, # mosca quaternions
            "lr_w": self.lr_w, # skin weight bias
            "lr_dyn": self.lr_dyn, # dynamic degree
            "lr_nsig": self.lr_nsig, # mosca sigma
        }

    @property
    def get_dynamic_node_lr_dict(self):
        return {
            "lr_p": 0.0,
            "lr_q": 0.0,
            "lr_s": 0.0,
            "lr_o": 0.0,
            "lr_sph": 0.0,
            "lr_sph_rest": 0.0,
            "lr_np": self.lr_np,
            "lr_nq": self.lr_nq,
            "lr_w": 0.0,
            "lr_dyn": 0.0,
            "lr_nsig": self.lr_nsig,
        }

    @property
    def get_cam_lr_dict(self):
        return {
            "lr_q": self.lr_cam_q, # camera quaternion
            "lr_t": self.lr_cam_t, # camera translation
            "lr_f": self.lr_cam_f, # camera focal
        }

    def get_scheduler(self, total_steps):
        # todo: decide whether to decay skinning weights
        gs_scheduling_dict = {
            "xyz": get_expon_lr_func(
                lr_init=self.lr_p,
                lr_final=self.lr_p_final,
                lr_delay_mult=0.01,  # 0.02
                max_steps=total_steps,
            ),
            "node_xyz": get_expon_lr_func(
                lr_init=self.lr_np,
                lr_final=self.lr_np_final,
                lr_delay_mult=0.01,  # 0.02
                max_steps=total_steps,
            ),
            "node_rotation": get_expon_lr_func(
                lr_init=self.lr_nq,
                lr_final=self.lr_nq_final,
                lr_delay_mult=0.01,  # 0.02
                max_steps=total_steps,
            ),
        }
        cam_scheduling_dict = {
            "R": get_expon_lr_func(
                lr_init=self.lr_cam_q,
                lr_final=self.lr_cam_q_final,
                lr_delay_mult=0.01,  # 0.02
                max_steps=total_steps,
            ),
            "t": get_expon_lr_func(
                lr_init=self.lr_cam_t,
                lr_final=self.lr_cam_t_final,
                lr_delay_mult=0.01,  # 0.02
                max_steps=total_steps,
            ),
            "f": get_expon_lr_func(
                lr_init=self.lr_cam_f,
                lr_final=self.lr_cam_f_final,
                lr_delay_mult=0.01,  # 0.02
                max_steps=total_steps,
            ),
        }
        return gs_scheduling_dict, cam_scheduling_dict


@torch.no_grad()
def fetch_leaves_in_world_frame(
    cams: MonocularCameras,
    n_attach: int,  # if negative use all
    #
    input_mask_list,
    input_dep_list,
    input_rgb_list,
    input_normal_list=None,  # ! in new version, do not use this
    input_inst_list=None,  # ! in new version, do not use this
    #
    save_xyz_fn=None,
    start_t=0,
    end_t=-1,
    t_list=None,
    subsample=1,
    squeeze_normal_ratio=1.0, # 1.0
):
    device = cams.rel_focal.device

    if end_t == -1: # -1
        end_t = cams.T
    if t_list is None: # None
        t_list = list(range(start_t, end_t))
    if subsample > 1: # subsample == 1, NOTE 2D spacial subsample
        logging.info(f"2D Subsample {subsample} for fetching ...")

    mu_list, quat_list, scale_list, rgb_list, time_index_list = [], [], [], [], []
    inst_list = []  # collect the leaf id as well; semantic segmentation / instance segmentation

    for t in tqdm(t_list):
        mask2d = input_mask_list[t].bool()
        H, W = mask2d.shape
        if subsample > 1:
            mask2d[::subsample, ::subsample] = False

        dep_map = input_dep_list[t].clone()
        cam_pcl = cams.backproject(
            cams.get_homo_coordinate_map(H, W)[mask2d].clone(), dep_map[mask2d]
        ) # N,3
        cam_R_wc, cam_t_wc = cams.Rt_wc(t) # c2w
        mu = cams.trans_pts_to_world(t, cam_pcl) # N,3
        rgb_map = input_rgb_list[t].clone() # H,W,3
        rgb = rgb_map[mask2d] # N,3
        K = cams.K(H, W)
        radius = cam_pcl[:, -1] / (0.5 * K[0, 0] + 0.5 * K[1, 1]) * float(subsample) # N, NOTE 2D spacial subsample
        scale = torch.stack([radius / squeeze_normal_ratio, radius, radius], dim=-1) # N,3
        time_index = torch.ones_like(mu[:, 0]).long() * t # NOTE ref_time

        if input_normal_list is not None: # NOTE normal list; None
            nrm_map = input_normal_list[t].clone()
            cam_nrm = nrm_map[mask2d]
            nrm = F.normalize(torch.einsum("ij,nj->ni", cam_R_wc, cam_nrm), dim=-1)
            rx = nrm.clone()
            ry = F.normalize(torch.cross(rx, mu, dim=-1), dim=-1)
            rz = F.normalize(torch.cross(rx, ry, dim=-1), dim=-1)
            rot = torch.stack([rx, ry, rz], dim=-1)
        else:
            rot = torch.eye(3)[None].expand(len(radius), -1, -1)
        quat = matrix_to_quaternion(rot)

        mu_list.append(mu.cpu())
        quat_list.append(quat.cpu())
        scale_list.append(scale.cpu())
        rgb_list.append(rgb.cpu())
        time_index_list.append(time_index.cpu()) # NOTE ref_time

        if input_inst_list is not None: # NOTE instance list; None
            inst_map = inst_list[t].clone()
            inst = inst_map[mask2d]
            inst_list.append(inst.cpu())

    mu_all = torch.cat(mu_list, 0) # N,3
    quat_all = torch.cat(quat_list, 0) # N,4
    scale_all = torch.cat(scale_list, 0) # N,3
    rgb_all = torch.cat(rgb_list, 0) # N,3

    logging.info(f"Fetching {n_attach/1000.0:.3f}K out of {len(mu_all)/1e6:.3}M pts")
    if n_attach > len(mu_all) or n_attach <= 0: # NOTE downsample
        choice = torch.arange(len(mu_all)) # use all
    else:
        choice = torch.randperm(len(mu_all))[:n_attach] # downsample

    # make gs5 param (mu, fr, s, o, sph) no rescaling
    mu_init = mu_all[choice].clone()
    q_init = quat_all[choice].clone()
    s_init = scale_all[choice].clone()
    o_init = torch.ones(len(choice), 1).to(mu_init)
    rgb_init = rgb_all[choice].clone()
    time_init = torch.cat(time_index_list, 0)[choice]
    if len(inst_list) > 0: # NOTE instance list
        inst_all = torch.cat(inst_list, 0)
        inst_init = inst_all[choice].clone().to(device)
    else:
        inst_init = None
    if save_xyz_fn is not None: # None
        np.savetxt(
            save_xyz_fn,
            torch.cat([mu_init, rgb_init * 255], 1).detach().cpu().numpy(),
            fmt="%.6f",
        )
    torch.cuda.empty_cache()
    return (
        mu_init.to(device),
        q_init.to(device),
        s_init.to(device),
        o_init.to(device),
        rgb_init.to(device),
        inst_init, # None
        time_init.to(device), # NOTE ref_time
    )


def estimate_normal_map(
    vtx_map,
    mask,
    normal_map_patchsize=7,
    normal_map_nn_dist_th=0.03,
    normal_map_min_nn=6,
):
    # * this func is borrowed from my pytorch4D repo in 2022 May.
    # the normal neighborhood is estimated
    # the normal computation refer to pytorch3d 0.6.1, but updated with linalg operations in newer pytroch version

    # note, here the path has zero padding on the border, but due to the computation, sum the zero zero covariance matrix will make no difference, safe!
    H, W = mask.shape
    v_map_patch = F.unfold(
        vtx_map.permute(2, 0, 1).unsqueeze(0),
        normal_map_patchsize,
        dilation=1,
        padding=(normal_map_patchsize - 1) // 2,
        stride=1,
    ).reshape(3, normal_map_patchsize**2, H, W)

    mask_patch = F.unfold(
        mask.unsqueeze(0).unsqueeze(0).float(),
        normal_map_patchsize,
        dilation=1,
        padding=(normal_map_patchsize - 1) // 2,
        stride=1,
    ).reshape(1, normal_map_patchsize**2, H, W)

    # Also need to consider the neighbors distance for occlusion boundary
    nn_dist = (vtx_map.permute(2, 0, 1).unsqueeze(1) - v_map_patch).norm(dim=0)
    valid_nn_mask = (nn_dist < normal_map_nn_dist_th)[None, ...]
    v_map_patch[~valid_nn_mask.expand(3, -1, -1, -1)] = 0.0
    mask_patch = mask_patch * valid_nn_mask

    # only meaningful when there are at least 3 valid pixels in the neighborhood, the pixels with less nn need to be exclude when computing the final output normal map, but the mask_patch shouldn't be updated because they still can be used to compute normals for other pixels
    neighbor_counts = mask_patch.sum(dim=1).squeeze(0)  # H,W
    valid_mask = torch.logical_and(mask, neighbor_counts >= normal_map_min_nn)

    v_map_patch = v_map_patch.permute(2, 3, 1, 0)  # H,W,Patch,3
    mask_patch = mask_patch.permute(2, 3, 1, 0)  # H,W,Patch,1
    vtx_patch = v_map_patch[valid_mask]  # M,Patch,3
    neighbor_counts = neighbor_counts[valid_mask]
    mask_patch = mask_patch[valid_mask]  # M,Patch,1

    # compute the curvature normal for the neighbor hood
    # fix several bug here: 1.) the centroid mean bug 2.) the local coordinate should be mask to zero for cov mat
    assert neighbor_counts.min() > 0
    centroid = vtx_patch.sum(dim=1, keepdim=True) / (neighbor_counts[:, None, None])
    vtx_patch = vtx_patch - centroid
    vtx_patch = vtx_patch * mask_patch
    # vtx_patch = vtx_patch.double()
    W = torch.matmul(vtx_patch.unsqueeze(-1), vtx_patch.unsqueeze(-2))
    # todo: here can use distance/confidence to weight the contribution
    W = W.sum(dim=1)  # M,3,3

    curvature, local_frame = torch.linalg.eigh(W)
    normal = local_frame[..., 0]

    normal_map = torch.zeros_like(vtx_map)
    normal_map[valid_mask] = normal

    return normal_map, valid_mask


########################################################################
# ! node grow
########################################################################

from lib_render.render_helper import render


@torch.no_grad()
def error_grow_dyn_model(
    s2d,
    cams: MonocularCameras,
    s_model,
    d_model,
    optimizer_dynamic,
    step,
    dyn_error_grow_th,
    dyn_error_grow_num_frames,
    dyn_error_grow_subsample,
    viz_dir,
    open_k_size=3,
    opacity_init_factor=0.98,
):
    # * identify the error mask
    device = s2d.rgb.device # cuda

    # NOTE COMPUTE SE(3) B-SPLINE
    d_model.scf.compute_se3_bspline()
    
    error_list = identify_rendering_error(cams, s_model, d_model, s2d) # T,H,W NOTE CPU
    T = len(error_list)
    imageio.mimsave(
        osp.join(viz_dir, f"error_raw_{step}.mp4"), error_list.cpu().numpy()
    )
    # * refine the error mask
    grow_fg_masks = (error_list > dyn_error_grow_th).to(device)
    open_kernel = torch.ones(open_k_size, open_k_size)
    # handle large time by chunk the time
    cur = 0
    chunk = 50
    grow_fg_masks_morph = []
    while cur < T:
        _grow_fg_masks = kornia.morphology.opening(
            grow_fg_masks[cur : cur + chunk, None].float(),
            kernel=open_kernel.to(grow_fg_masks.device),
        ).squeeze(1)
        grow_fg_masks_morph.append(_grow_fg_masks.bool())
        cur = cur + chunk
    grow_fg_masks = torch.cat(grow_fg_masks_morph, 0)
    grow_fg_masks = grow_fg_masks * s2d.dep_mask * s2d.dyn_mask # T,H,W NOTE final error mask
    # viz
    imageio.mimsave(
        osp.join(viz_dir, f"error_{step}.mp4"),
        grow_fg_masks.detach().cpu().float().numpy(),
    )

    # * select error grow time
    if dyn_error_grow_num_frames < T: # dyn_error_grow_num_frames: 4
        # sample some frames to grow
        grow_cnt = grow_fg_masks.reshape(T, -1).sum(-1)
        grow_cnt = grow_cnt.detach().cpu().numpy()
        grow_tids = np.argsort(grow_cnt)[-dyn_error_grow_num_frames:][::-1]
        # plot the grow_cnt with bars
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(grow_cnt)), grow_cnt)
        plt.savefig(osp.join(viz_dir, f"error_{step}_grow_cnt.jpg"))
        plt.close()
    else:
        grow_tids = [i for i in range(T)]
    logging.info(f"Grow Error at {step} on {grow_tids}")

    for grow_tid in tqdm(grow_tids): # T_sub
        grow_mask = grow_fg_masks[grow_tid] # H,W
        if grow_mask.sum() == 0:
            continue
        # * prepare point parameter
        # ! The append points must be in front of the model depth!!!
        grow_mu_cam, grow_mask = align_to_model_depth( # 2D -> 3D
            s2d,
            working_mask=grow_mask.bool(),
            cams=cams,
            tid=grow_tid,
            s_model=s_model,
            d_model=d_model,
            dep_align_knn=-1,  # 400, #32,
            sub_sample=dyn_error_grow_subsample, # 1
        ) # [N,3], [H,W]
        if len(grow_mu_cam) == 0:
            continue
        # convert to mu_w and get s_inti
        R_wc, t_wc = cams.Rt_wc(grow_tid)
        grow_mu_w = torch.einsum("ij,aj->ai", R_wc, grow_mu_cam) + t_wc[None] # N,3
        K = cams.K(s2d.H, s2d.W)
        grow_s = (
            grow_mu_cam[:, -1]
            / (0.5 * K[0, 0] + 0.5 * K[1, 1])
            * float(dyn_error_grow_subsample)
        )

        # ! special function that sample nodes from candidates and select attached nodes
        quat_w = torch.zeros(len(grow_mu_w), 4).to(grow_mu_w)
        quat_w[:, 0] = 1.0

        # * add node and dynamic gaussian
        d_model.append_new_node_and_gs(
            optimizer_dynamic,
            tid=grow_tid,
            mu_w=grow_mu_w,
            quat_w=quat_w,
            scales=grow_s[:, None].expand(-1, 3),
            opacity=torch.ones_like(grow_s)[:, None] * opacity_init_factor,
            rgb=s2d.rgb[grow_tid][grow_mask],
        )
    return

@torch.no_grad()
def error_grow_node_model(
    s2d,
    cams: MonocularCameras,
    s_model,
    d_model,
    optimizer_dynamic,
    step,
    dyn_error_grow_th, # 0.5
    viz_dir,
    open_k_size=3,
    max_gs_per_new_node=200,
):
    # * identify the error mask
    device = s2d.rgb.device # cuda
    # NOTE COMPUTE SE(3) B-SPLINE
    d_model.scf.compute_se3_bspline()
    error_list = identify_rendering_error(cams, s_model, d_model, s2d) # T,H,W NOTE CPU
    T = len(error_list)
    imageio.mimsave(
        osp.join(viz_dir, f"error_raw_{step}.mp4"), error_list.cpu().numpy()
    )
    # * refine the error mask
    grow_fg_masks = (error_list > dyn_error_grow_th).to(device)
    open_kernel = torch.ones(open_k_size, open_k_size)
    # handle large time by chunk the time
    cur = 0
    chunk = 50
    grow_fg_masks_morph = []
    while cur < T:
        _grow_fg_masks = kornia.morphology.opening(
            grow_fg_masks[cur : cur + chunk, None].float(),
            kernel=open_kernel.to(grow_fg_masks.device),
        ).squeeze(1)
        grow_fg_masks_morph.append(_grow_fg_masks.bool())
        cur = cur + chunk
    grow_fg_masks = torch.cat(grow_fg_masks_morph, 0)
    grow_fg_masks = grow_fg_masks * s2d.dep_mask * s2d.dyn_mask # T,H,W NOTE final error mask
    # viz
    imageio.mimsave(
        osp.join(viz_dir, f"error_{step}.mp4"),
        grow_fg_masks.detach().cpu().float().numpy(),
    )

    # NOTE compute 2d node tracklets
    node_tracklets = []
    for view_ind in tqdm(range(T)):
        node_mu_w = d_model.scf._node_xyz[d_model.get_tlist_ind(view_ind)] # M,3 NOTE world
        R_cw, t_cw = cams.Rt_cw(view_ind) # w2c
        node_mu = node_mu_w @ R_cw.T + t_cw[None] # M,3 NOTE camera
        node_mu_uv = cams.project(node_mu, 0) # M,2 NOTE uv
        node_mu_xy = cams.uv_to_pixel(node_mu_uv, s2d.H, s2d.W) # M,2 NOTE xy
        node_tracklets.append(node_mu_xy)
    node_tracklets = torch.stack(node_tracklets, dim=0) # T,M,2
    
    t = torch.arange(T, device=node_tracklets.device)[:, None].expand(T, node_tracklets.shape[1]) # T,M
    x = node_tracklets[..., 0].long() # T,M
    y = node_tracklets[..., 1].long() # T,M
    x = x.clamp(0, s2d.W-1)
    y = y.clamp(0, s2d.H-1)

    hit = grow_fg_masks[t, y, x] # T,M
    node_grow_num = hit.sum(dim=0) # M
    node_grow_mask = node_grow_num > T // 2 # M
    node_grow_ind = torch.where(node_grow_mask)[0] # M_sub

    if len(node_grow_ind) == 0:
        logging.info(f"No node to grow at {step}")
        return
    
    to_grow_node_xyz = d_model.scf._node_xyz[:,node_grow_ind] # T,M_sub,3
    to_grow_node_quat = d_model.scf._node_rotation[:,node_grow_ind] # T,M_sub,4
    new_group_id = d_model.scf._node_grouping[node_grow_ind] # M_sub

    # * add nodes and dynamic gaussian
    
    # add nodes
    old_M = d_model.scf.M

    # add node noise (only position)
    # Give each duplicated node one small, time-consistent xyz offset so the
    # new node separates slightly from the original one without trajectory jitter.
    xyz_noise_scale = 0.01
    xyz_noise_std = xyz_noise_scale * float(d_model.scf.spatial_unit)
    xyz_noise = (
        torch.randn(
            1,
            to_grow_node_xyz.shape[1],
            3,
            device=to_grow_node_xyz.device,
            dtype=to_grow_node_xyz.dtype,
        )
        * xyz_noise_std
    )
    to_grow_node_xyz = to_grow_node_xyz + xyz_noise

    d_model.scf.append_nodes_traj(
        optimizer_dynamic, to_grow_node_xyz, to_grow_node_quat, new_group_id
    )
    d_model.scf.incremental_topology()  # ! manually must set this
    
    # add dynamic gaussian
    original_attach_ind = node_grow_ind # new_M NOTE attach index is ref node index

    new_attach_ind_list, new_gs_ind_list = [], []
    for _i in range(to_grow_node_xyz.shape[1]): # new_M
        _attach_ind = old_M + _i
        # the same carrying leaves duplicate them
        neighbors_mask = d_model.attach_ind == original_attach_ind[_i] # N_dyn
        if not neighbors_mask.any():
            continue
        # ! bound the number of leaves here
        # !  WARNING, THIS MODIFICATION IS AFTER MANY BASE VERSION, BE CAREFUL
        if neighbors_mask.long().sum() > float(max_gs_per_new_node): # random downsample neighbors_mask
            # random sample max_gs_per_new_node and mark the flag
            neighbors_ind = torch.arange(d_model.N, device=d_model.device)[neighbors_mask]
            neighbors_ind = neighbors_ind[
                torch.randperm(neighbors_ind.shape[0])[:max_gs_per_new_node]
            ]
            neighbors_mask = torch.zeros_like(neighbors_mask)
            neighbors_mask[neighbors_ind] = True
        #
        gs_ind = torch.arange(d_model.N, device=d_model.device)[neighbors_mask] # N_sub
        new_attach_ind_list.append(torch.ones_like(gs_ind) * _attach_ind) # N_sub
        new_gs_ind_list.append(gs_ind)
    if len(new_attach_ind_list) == 0:
        logging.info(f"No new leaves to append")
        return
    new_attach_ind = torch.cat(new_attach_ind_list, dim=0) # N_all
    new_gs_ind = torch.cat(new_gs_ind_list, dim=0) # N_all

    logging.info(
        f"Append {to_grow_node_xyz.shape[1]} new nodes and dup {new_gs_ind.shape[0]} leaves"
    )

    assert new_gs_ind.max() < d_model.N, f"{new_gs_ind.max()}, {d_model.N}"

    # add gaussian noise (position scale rotation)

    base_xyz = d_model._xyz[new_gs_ind].detach().clone()          # NOTE local leaf xyz
    base_r = d_model._rotation[new_gs_ind].detach().clone()       # NOTE local leaf quat
    base_r = base_r / base_r.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    base_s = d_model.get_s[new_gs_ind].detach().clone()
    
    # position noise:
    # add a small perturbation in the Gaussian's own principal-axis frame.
    # magnitude is tied to current scale, with a small floor from node jitter.
    xyz_std = torch.maximum(
        0.10 * base_s,
        torch.full_like(base_s, 0.25 * xyz_noise_std),
    )
    xyz_noise_local = torch.randn_like(base_xyz) * xyz_std
    xyz_noise = torch.einsum(
        "nij,nj->ni", quaternion_to_matrix(base_r), xyz_noise_local
    )
    new_xyz = base_xyz + xyz_noise

    # scale noise:
    # perturb in real scale space, then map back to logit space for stability.
    scale_jitter = torch.exp(
        torch.clamp(0.05 * torch.randn_like(base_s), min=-0.12, max=0.12)
    )
    scale_min = (d_model.min_scale + 1e-6).to(base_s)
    scale_max = (d_model.max_scale - 1e-6).to(base_s)
    new_s = torch.clamp(base_s * scale_jitter, min=scale_min, max=scale_max)
    new_s_logit = d_model.s_inv_act(new_s)

    # rotation noise:
    # small axis-angle perturbation with hard clipping, then renormalize quaternion.
    rot_std = np.deg2rad(2.0)
    rot_max = np.deg2rad(5.0)
    rand_axis = torch.randn_like(base_xyz)
    rand_axis = rand_axis / rand_axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    rand_angle = (
        torch.randn(
            base_xyz.shape[0], 1, device=base_xyz.device, dtype=base_xyz.dtype
        )
        * rot_std
    ).clamp(min=-rot_max, max=rot_max)
    rot_noise = axis_angle_to_matrix(rand_axis * rand_angle)
    new_r = matrix_to_quaternion(
        torch.einsum("nij,njk->nik", rot_noise, quaternion_to_matrix(base_r))
    )
    new_r = new_r / new_r.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    d_model._densification_postprocess(
        optimizer_dynamic,
        new_xyz=d_model._xyz[new_gs_ind].detach().clone(),
        new_r=d_model._rotation[new_gs_ind].detach().clone(),
        new_s_logit=d_model._scaling[new_gs_ind].detach().clone(),
        new_o_logit=d_model._opacity[new_gs_ind].detach().clone(),
        new_sph_dc=d_model._features_dc[new_gs_ind].detach().clone(),
        new_sph_rest=d_model._features_rest[new_gs_ind].detach().clone(),
        new_skinning_w=d_model._skinning_weight[new_gs_ind].detach().clone(),
        new_dyn_logit=d_model.o_inv_act(
            torch.ones_like(d_model._xyz[new_gs_ind][:, :1]) * 0.99
        ),
    )
    
    d_model.attach_ind = torch.cat(
        [d_model.attach_ind, new_attach_ind.to(d_model.attach_ind)], dim=0
    )
    assert (
        d_model.attach_ind.max() < d_model.scf.M
    ), f"{d_model.attach_ind.max()}, {d_model.scf.M}"
    d_model.ref_time = torch.cat(
        [d_model.ref_time, d_model.ref_time.clone()[new_gs_ind]], dim=0
    )
    assert d_model.ref_time.max() < d_model.T, f"{d_model.ref_time.max()}, {d_model.T}"

    d_model.clean_corr_control_record() # corr_gradient_accum and corr_gradient_denom

    return

@torch.no_grad()
def identify_rendering_error(
    cams: MonocularCameras,
    s_model: StaticGaussian,
    d_model: DynSCFGaussian,
    s2d,
):
    # render all frames and compare photo error
    logging.info("Compute rendering errors ...")
    error_list = []
    for t in tqdm(range(d_model.T)):  # ! warning, d_model.T may smaller than cams.T
        gs5 = [s_model(), d_model(t)]
        render_dict = render(gs5, s2d.H, s2d.W, cams.K(s2d.H, s2d.W), cams.T_cw(t))
        rgb_pred = render_dict["rgb"].permute(1, 2, 0)
        rgb_gt = s2d.rgb[t].clone()
        error = (rgb_pred - rgb_gt).abs().max(dim=-1).values
        error_list.append(error.detach().cpu())
        # torch.cuda.empty_cache() # clear GPU cache
    error_list = torch.stack(error_list, 0)
    return error_list


def get_subsample_mask_like(buffer, sub_sample):
    # buffer is H,W,...
    assert buffer.ndim >= 2
    ret = torch.zeros_like(buffer).bool()
    ret[::sub_sample, ::sub_sample, ...] = True
    return ret


@torch.no_grad()
def __align_pixel_depth_scale_backproject__(
    homo_map, # H,W,2 NOTE Normalized coordinates
    src_align_mask,
    src, # source depth
    src_mask, # source depth mask
    dst, # destination depth
    dst_mask, # destination depth mask
    cams: MonocularCameras,
    knn=8,
    infrontflag=True,
):
    # src_align_mask: which pixel is going to be aligned
    # src_mask: the valid pixel in the src
    # dst_mask: the valid pixel in the dst
    # use 3D nearest knn nn to find the best scaling, local rigid warp

    # the base pts build correspondence between current frame and
    base_mask = src_mask * dst_mask * (~src_align_mask)
    query_mask = src_align_mask

    ratio = dst / (src + 1e-6)
    base_pts_ratio = ratio[base_mask]

    query_pts = cams.backproject(homo_map[query_mask], src[query_mask]) # 2D -> 3D

    # backproject src depth to 3D
    if knn > 0:
        base_pts = cams.backproject(homo_map[base_mask], src[base_mask])
        _, ind, _ = knn_points(query_pts[None], base_pts[None], K=knn)
        ind = ind[0]
        ratio = base_pts_ratio[ind]
    else:
        ratio = base_pts_ratio.mean()[None, None].expand(len(query_pts), -1)
    ret = ratio.mean(-1, keepdim=True) * query_pts

    if infrontflag: # True
        # ! make sure the ret-z is always smaller than the dst z
        ret_z = ret[:, -1]
        dst_z = dst[query_mask]
        assert (dst_z > -1e-6).all()
        new_z = torch.min(ret_z, dst_z - 1e-4)
        logging.info(f"Make sure the aligned points is in front of the dst depth")
        ratio = new_z / torch.clamp(ret_z, min=1e-6)
        # assert (ratio <= 1 + 1e-6).all()
    ret = ret * ratio[:, None]
    return ret


def align_to_model_depth(
    s2d,
    working_mask, # grow_mask
    cams,
    tid,
    s_model,
    d_model=None,
    dep_align_knn=9,
    sub_sample=1,
):
    gs5 = [s_model()]
    if d_model:
        gs5.append(d_model(tid))
    render_dict = render(gs5, s2d.H, s2d.W, cams.K(s2d.H, s2d.W), cams.T_cw(tid))
    model_alpha = render_dict["alpha"].squeeze(0)
    model_dep = render_dict["dep"].squeeze(0)
    # align prior depth to current depth
    sub_mask = get_subsample_mask_like(working_mask, sub_sample) # H,W NOTE downsample
    ret_mask = working_mask * sub_mask # final mask
    new_mu_cam = __align_pixel_depth_scale_backproject__(
        homo_map=s2d.homo_map,
        src_align_mask=ret_mask,
        src=s2d.dep[tid].clone(),
        src_mask=s2d.dep_mask[tid] * sub_mask,
        dst=model_dep,
        dst_mask=(model_alpha > 0.5)
        & (
            ~working_mask
        ),  # ! warning, here manually use the original non-subsampled mask, because the dilated place is not reliable!
        cams=cams,
        knn=dep_align_knn,
    ) # N,3
    return new_mu_cam, ret_mask


########################################################################
# ! end node grow
########################################################################


def __query_image_buffer__(uv, buffer):
    # buffer: H, W, C; uv: N, 2
    if buffer.ndim == 2:
        buffer = buffer[..., None]
    H, W = buffer.shape[:2]
    uv = torch.round(uv).long()
    uv[:, 0] = torch.clamp(uv[:, 0], 0, W - 1)
    uv[:, 1] = torch.clamp(uv[:, 1], 0, H - 1)
    uv = torch.round(uv).long()
    ind = uv[:, 1] * W + uv[:, 0]
    return buffer.reshape(-1, buffer.shape[-1])[ind]


def identify_traj_id(uv_trajs, visibs, idmap_list):
    # ! this is only for point odeyssey
    assert (
        len(uv_trajs) == len(visibs) == len(idmap_list)
    ), f"{len(uv_trajs)} vs {len(visibs)} vs {len(idmap_list)}"
    T, N = uv_trajs.shape[:2]
    collected_id = torch.ones_like(visibs, dtype=idmap_list.dtype) * -1
    for t in range(T):
        uv = uv_trajs[t]
        visib = visibs[t]
        idmap = idmap_list[t]
        traj_id = __query_image_buffer__(uv[visib], torch.as_tensor(idmap)).squeeze(-1)
        buff = collected_id[t]
        buff[visib] = traj_id
        collected_id[t] = buff

    valid_mask = collected_id >= 0
    ret = []
    for i in range(N):
        traj_id = collected_id[:, i]
        mask = valid_mask[:, i]
        if not mask.any():
            ret.append(-1)
            continue
        ids = traj_id[mask]
        # majority voting
        id_count = torch.bincount(ids)
        id_count[0] = 0
        max_id = torch.argmax(id_count)
        ret.append(max_id.item())
    ret = torch.as_tensor(ret).to(uv_trajs.device)
    return ret


def get_colorplate(n):
    hue = np.linspace(0, 1, n + 1)[:-1]
    color_plate = torch.Tensor([colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in hue])
    return color_plate

@torch.no_grad()
def estimate_focus_depth_from_dyn_gs(
    dyn_gs,
    cams: MonocularCameras,
    view_ind,
    min_depth=1e-2,
):
    if dyn_gs is None or dyn_gs[0].numel() == 0:
        return 1.0

    dyn_xyz_cam = cams.trans_pts_to_cam(view_ind, dyn_gs[0].detach())
    dyn_opacity = dyn_gs[3].detach().squeeze(-1)

    valid = torch.isfinite(dyn_xyz_cam[:, 2]) & (dyn_xyz_cam[:, 2] > min_depth)
    if valid.sum() == 0:
        return 1.0

    z = dyn_xyz_cam[:, 2][valid]
    w = dyn_opacity[valid].clamp_min(1e-6)
    return (z * w).sum().div(w.sum()).item()


@torch.no_grad()
def get_move_around_cam_T_cw(  # return w2c
    cams: MonocularCameras,
    view_ind,
    focus_depth,
    move_around_angle_deg=np.pi / 36,
    rand_angle=None,
):
    device = cams.rel_focal.device
    dtype = cams.rel_focal.dtype

    focus_depth = max(float(focus_depth), 1e-2)
    focus_point = torch.tensor([0.0, 0.0, focus_depth], device=device, dtype=dtype)

    move_around_radius = np.tan(move_around_angle_deg) * focus_depth
    if rand_angle is None:
        rand_angle = np.random.rand()
    x = move_around_radius * np.cos(2 * np.pi * rand_angle)
    y = move_around_radius * np.sin(2 * np.pi * rand_angle)
    T_c_new = torch.eye(4, device=device, dtype=dtype)
    T_c_new[0, -1] = x
    T_c_new[1, -1] = y

    up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
    _z_dir = F.normalize(focus_point - T_c_new[:3, -1], dim=0)
    _x_dir = F.normalize(torch.cross(up, _z_dir, dim=0), dim=0)
    _y_dir = F.normalize(torch.cross(_z_dir, _x_dir, dim=0), dim=0)
    T_c_new[:3, 0] = _x_dir
    T_c_new[:3, 1] = _y_dir
    T_c_new[:3, 2] = _z_dir
    T_w_new = cams.T_wc(view_ind).detach() @ T_c_new
    T_new_w = torch.linalg.inv(T_w_new).detach()

    return T_new_w, focus_point.detach()


def calculate_relative_angles(origin, center1, center2):

    origin = np.array(origin.cpu(), dtype=np.float64)
    c1 = np.array(center1.cpu(), dtype=np.float64)
    c2 = np.array(center2.cpu(), dtype=np.float64)

    # generate z_ axis (from origin to center1)
    z_axis = c1 - origin
    dist_o_c1 = np.linalg.norm(z_axis)
    if dist_o_c1 < 1e-8: # TODO
        return 0.0, 0.0
        # raise ValueError("Origin and Center1 are the same point, cannot define Z-axis.")
    z_axis /= dist_o_c1

    # Build the local X/Y axes with a stable world-up reference
    # Use [0, 0, 1] as the default up direction.
    up = np.array([0, 0, 1])
    # If Z is almost parallel to the default up axis, switch the reference axis.
    if abs(np.dot(z_axis, up)) > 0.99:
        up = np.array([0, 1, 0])
    
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    v_12 = c2 - c1
    
    # Project v_12 onto the local coordinate frame rooted at c1.
    x_prime = np.dot(v_12, x_axis)
    y_prime = np.dot(v_12, y_axis)
    z_prime = np.dot(v_12, z_axis)

    # Compute elevation and azimuth.
    # Elevation is the angle relative to the local XY plane in [-90, 90] degrees.
    dist_v12 = np.linalg.norm(v_12)
    if dist_v12 < 1e-8:
        return 0.0, 0.0 # c1 and c2 overlap
    
    elevation = np.arcsin(z_prime / dist_v12)
    azimuth = np.arctan2(y_prime, x_prime)

    return elevation, azimuth