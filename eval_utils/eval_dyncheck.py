# ! warning this is from iclr24 paper, should calibrate with the original neurips dycheck metrics

import os, sys, os.path as osp
import logging, imageio
from tqdm import tqdm
import numpy as np
import torch
import lpips, time
import pandas as pd
import cv2 as cv
from skimage.metrics import structural_similarity

from typing import Literal
import torch.nn.functional as F
from torchmetrics.functional.image.lpips import _NoTrainLpips
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.metric import Metric
from torchmetrics.utilities import dim_zero_cat
from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE


def compute_psnr(
    preds: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor | None = None,
) -> float:
    """
    Args:
        preds (torch.Tensor): (..., 3) predicted images in [0, 1].
        targets (torch.Tensor): (..., 3) target images in [0, 1].
        masks (torch.Tensor | None): (...,) optional binary masks where the
            1-regions will be taken into account.

    Returns:
        psnr (float): Peak signal-to-noise ratio.
    """
    if masks is None:
        masks = torch.ones_like(preds[..., 0])
    return (
        -10.0
        * torch.log(
            F.mse_loss(
                preds * masks[..., None],
                targets * masks[..., None],
                reduction="sum",
            )
            / masks.sum().clamp(min=1.0)
            / 3.0
        )
        / np.log(10.0)
    ).item()


def compute_pose_errors(
    preds: torch.Tensor, targets: torch.Tensor
) -> tuple[float, float, float]:
    """
    Args:
        preds: (N, 4, 4) predicted camera poses.
        targets: (N, 4, 4) target camera poses.

    Returns:
        ate (float): Absolute trajectory error.
        rpe_t (float): Relative pose error in translation.
        rpe_r (float): Relative pose error in rotation (degree).
    """
    # Compute ATE.
    ate = torch.linalg.norm(preds[:, :3, -1] - targets[:, :3, -1], dim=-1).mean().item()
    # Compute RPE_t and RPE_r.
    # It's important to use numpy here for the accuracy of RPE_r.
    # torch has numerical issues for acos when the value is close to 1.0, i.e.
    # RPE_r is supposed to be very small, and will result in artificially large
    # error.
    preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    pred_rels = np.linalg.inv(preds[:-1]) @ preds[1:]
    pred_rels = np.linalg.inv(preds[:-1]) @ preds[1:]
    target_rels = np.linalg.inv(targets[:-1]) @ targets[1:]
    error_rels = np.linalg.inv(target_rels) @ pred_rels
    traces = error_rels[:, :3, :3].trace(axis1=-2, axis2=-1)
    rpe_t = np.linalg.norm(error_rels[:, :3, -1], axis=-1).mean().item()
    rpe_r = (
        np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0)).mean().item()
        / np.pi
        * 180.0
    )
    return ate, rpe_t, rpe_r


class mPSNR(PeakSignalNoiseRatio):
    sum_squared_error: list[torch.Tensor]
    total: list[torch.Tensor]

    def __init__(self, **kwargs) -> None:
        super().__init__(
            data_range=1.0,
            base=10.0,
            dim=None,
            reduction="elementwise_mean",
            **kwargs,
        )
        self.add_state("sum_squared_error", default=[], dist_reduce_fx="cat")
        self.add_state("total", default=[], dist_reduce_fx="cat")

    def __len__(self) -> int:
        return len(self.total)

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor | None = None,
    ):
        """Update state with predictions and targets.

        Args:
            preds (torch.Tensor): (..., 3) float32 predicted images.
            targets (torch.Tensor): (..., 3) float32 target images.
            masks (torch.Tensor | None): (...,) optional binary masks where the
                1-regions will be taken into account.
        """
        if masks is None:
            masks = torch.ones_like(preds[..., 0])
        self.sum_squared_error.append(
            torch.sum(torch.pow((preds - targets) * masks[..., None], 2))
        )
        self.total.append(masks.sum().to(torch.int64) * 3)

    def compute(self) -> torch.Tensor:
        """Compute peak signal-to-noise ratio over state."""
        sum_squared_error = dim_zero_cat(self.sum_squared_error)
        total = dim_zero_cat(self.total)
        return -10.0 * torch.log(sum_squared_error / total).mean() / np.log(10.0)


class mSSIM(StructuralSimilarityIndexMeasure):
    similarity: list

    def __init__(self, **kwargs) -> None:
        super().__init__(
            reduction=None,
            data_range=1.0,
            return_full_image=False,
            **kwargs,
        )
        assert isinstance(self.sigma, float)

    def __len__(self) -> int:
        return sum([s.shape[0] for s in self.similarity])

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor | None = None,
    ):
        """Update state with predictions and targets.

        Args:
            preds (torch.Tensor): (B, H, W, 3) float32 predicted images.
            targets (torch.Tensor): (B, H, W, 3) float32 target images.
            masks (torch.Tensor | None): (B, H, W) optional binary masks where
                the 1-regions will be taken into account.
        """
        if masks is None:
            masks = torch.ones_like(preds[..., 0])

        # Construct a 1D Gaussian blur filter.
        assert isinstance(self.kernel_size, int)
        hw = self.kernel_size // 2
        shift = (2 * hw - self.kernel_size + 1) / 2
        assert isinstance(self.sigma, float)
        f_i = (
            (torch.arange(self.kernel_size, device=preds.device) - hw + shift)
            / self.sigma
        ) ** 2
        filt = torch.exp(-0.5 * f_i)
        filt /= torch.sum(filt)

        # Blur in x and y (faster than the 2D convolution).
        def convolve2d(z, m, f):
            # z: (B, H, W, C), m: (B, H, W), f: (Hf, Wf).
            z = z.permute(0, 3, 1, 2)
            m = m[:, None]
            f = f[None, None].expand(z.shape[1], -1, -1, -1)
            z_ = torch.nn.functional.conv2d(
                z * m, f, padding="valid", groups=z.shape[1]
            )
            m_ = torch.nn.functional.conv2d(m, torch.ones_like(f[:1]), padding="valid")
            return torch.where(
                m_ != 0, z_ * torch.ones_like(f).sum() / (m_ * z.shape[1]), 0
            ).permute(0, 2, 3, 1), (m_ != 0)[:, 0].to(z.dtype)

        filt_fn1 = lambda z, m: convolve2d(z, m, filt[:, None])
        filt_fn2 = lambda z, m: convolve2d(z, m, filt[None, :])
        filt_fn = lambda z, m: filt_fn1(*filt_fn2(z, m))

        mu0 = filt_fn(preds, masks)[0]
        mu1 = filt_fn(targets, masks)[0]
        mu00 = mu0 * mu0
        mu11 = mu1 * mu1
        mu01 = mu0 * mu1
        sigma00 = filt_fn(preds**2, masks)[0] - mu00
        sigma11 = filt_fn(targets**2, masks)[0] - mu11
        sigma01 = filt_fn(preds * targets, masks)[0] - mu01

        # Clip the variances and covariances to valid values.
        # Variance must be non-negative:
        sigma00 = sigma00.clamp(min=0.0)
        sigma11 = sigma11.clamp(min=0.0)
        sigma01 = torch.sign(sigma01) * torch.minimum(
            torch.sqrt(sigma00 * sigma11), torch.abs(sigma01)
        )

        assert isinstance(self.data_range, float)
        c1 = (self.k1 * self.data_range) ** 2
        c2 = (self.k2 * self.data_range) ** 2
        numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
        denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
        ssim_map = numer / denom

        self.similarity.append(ssim_map.mean(dim=(1, 2, 3)))

    def compute(self) -> torch.Tensor:
        """Compute final SSIM metric."""
        return torch.cat(self.similarity).mean()


class mLPIPS(Metric):
    sum_scores: list[torch.Tensor]
    total: list[torch.Tensor]

    def __init__(
        self,
        net_type: Literal["vgg", "alex", "squeeze"] = "alex",
        **kwargs,
    ):
        super().__init__(**kwargs)

        if not _TORCHVISION_AVAILABLE:
            raise ModuleNotFoundError(
                "LPIPS metric requires that torchvision is installed."
                " Either install as `pip install torchmetrics[image]` or `pip install torchvision`."
            )

        valid_net_type = ("vgg", "alex", "squeeze")
        if net_type not in valid_net_type:
            raise ValueError(
                f"Argument `net_type` must be one of {valid_net_type}, but got {net_type}."
            )
        self.net = _NoTrainLpips(net=net_type, spatial=True)

        self.add_state("sum_scores", [], dist_reduce_fx="cat")
        self.add_state("total", [], dist_reduce_fx="cat")

    def __len__(self) -> int:
        return len(self.total)

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor | None = None,
    ):
        """Update internal states with lpips scores.

        Args:
            preds (torch.Tensor): (B, H, W, 3) float32 predicted images.
            targets (torch.Tensor): (B, H, W, 3) float32 target images.
            masks (torch.Tensor | None): (B, H, W) optional float32 binary
                masks where the 1-regions will be taken into account.
        """
        if masks is None:
            masks = torch.ones_like(preds[..., 0])
        scores = self.net(
            (preds * masks[..., None]).permute(0, 3, 1, 2),
            (targets * masks[..., None]).permute(0, 3, 1, 2),
            normalize=True,
        )
        self.sum_scores.append((scores * masks[:, None]).sum())
        self.total.append(masks.sum().to(torch.int64))

    def compute(self) -> torch.Tensor:
        """Compute final perceptual similarity metric."""
        return (
            torch.tensor(self.sum_scores, device=self.device)
            / torch.tensor(self.total, device=self.device)
        ).mean()

def generate_background_masks(mask, val_names, threshold=0.9, kernel_size=(3, 3), erode_iterations=5):
    prefixes = list(set(int(name.split("_")[0]) for name in val_names)) # 1, 2 camera id
    
    # Initialize masks for each unique prefix
    aggregated_masks = {prefix: np.zeros_like(mask[0], dtype=np.float64) for prefix in prefixes}

    # Aggregate masks based on the unique prefixs
    for i, name in enumerate(val_names):
        prefix = int(name.split("_")[0])
        aggregated_masks[prefix] += mask[i]
    
    processed_masks = {}
    kernel = np.ones(kernel_size, dtype=np.uint8)

    for prefix, agg_mask in aggregated_masks.items():
        binary_mask = (agg_mask > agg_mask.max() * threshold).astype(np.float64)
        eroded_mask = cv.erode(binary_mask, kernel, iterations=erode_iterations) # shrink / small
        processed_masks[prefix] = eroded_mask
    
    # Update original masks using the processed masks
    for i, name in enumerate(val_names):
        prefix = int(name.split("_")[0])
        mask[i] = np.clip(mask[i] * processed_masks[prefix], 0., 255.0)
    
    return mask

sys.path.append(osp.dirname(osp.abspath(__file__)))

def im2tensor(img):
    return (torch.Tensor(img.transpose(2, 0, 1) / 127.5 - 1.0)[None, ...]).cuda()

def evaluate_nv(gt, pred, gt_mask): # MARK: NVS
    device = "cuda"
    psnr_metric = mPSNR().to(device)
    ssim_metric = mSSIM().to(device)
    lpips_metric = mLPIPS().to(device)

    val_imgs = torch.from_numpy(gt[..., :3]).to(device) # H, W, 3
    val_covisibles = torch.from_numpy(gt_mask).to(device) # H, W, 1
    pred_val_imgs = torch.from_numpy(pred).to(device) # H, W, 3


    val_img = val_imgs / 255.0 # H, W, 3
    pred_val_img = pred_val_imgs / 255.0 # H, W, 3
    val_covisible = val_covisibles.squeeze(-1).float() # H, W
    psnr_metric.update(val_img, pred_val_img, val_covisible)
    ssim_metric.update(val_img[None], pred_val_img[None], val_covisible[None])
    lpips_metric.update(val_img[None], pred_val_img[None], val_covisible[None])

    mpsnr = psnr_metric.compute().item()
    mssim = ssim_metric.compute().item()
    mlpips = lpips_metric.compute().item()
    return mpsnr, mssim, mlpips


def eval_dycheck(
    save_dir,
    gt_rgb_dir,
    gt_mask_dir,
    pred_dir,
    strict_eval_all_gt_flag=False,
    eval_non_masked=False,
    save_prefix="",
    viz_interval=50,
    eval_background_covisible_mask=False,
):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_t = time.time()
    lpips_fn = lpips.LPIPS(net="alex").to(device)
    logging.info(f"lpips_fn init time: {time.time() - start_t:.2f}s")

    gt_rgb_fns = sorted(os.listdir(gt_rgb_dir))
    gt_rgb_fns = [f for f in gt_rgb_fns if not os.path.basename(f).startswith("0_")]
    gt_mask_fns = sorted(os.listdir(gt_mask_dir))
    gt_mask_fns = [f for f in gt_rgb_fns if not os.path.basename(f).startswith("0_")]
    pred_fns = sorted(os.listdir(pred_dir))


    if pred_dir.endswith("/"):
        pred_dir = pred_dir[:-1]
    eval_name = osp.basename(pred_dir) # tto_test
    save_viz_dir = osp.join(save_dir, f"{eval_name}_viz") # tto_test_viz
    os.makedirs(save_viz_dir, exist_ok=True)

    if strict_eval_all_gt_flag: # True
        assert (
            len(gt_rgb_fns) == len(gt_mask_fns) == len(pred_fns)
        ), "Number of files must match"
    else:
        pred_ids = [f[:-4] for f in pred_fns]
        if len(gt_rgb_fns) != len(pred_fns):
            logging.warning(
                f"Only eval predicted images {len(pred_ids)} < all gt {len(gt_rgb_fns)}"
            )
            assert len(gt_rgb_fns) == len(gt_mask_fns)
            filtered_gt_rgb_fns, filtered_gt_mask_fns = [], []
            for i in range(len(gt_rgb_fns)):
                if gt_rgb_fns[i][:-4] in pred_ids:
                    filtered_gt_rgb_fns.append(gt_rgb_fns[i])
                    filtered_gt_mask_fns.append(gt_mask_fns[i])
            gt_rgb_fns = filtered_gt_rgb_fns
            gt_mask_fns = filtered_gt_mask_fns
        assert (
            len(gt_rgb_fns) == len(gt_mask_fns) == len(pred_fns)
        ), "Number of files must match"

    # prepare background covisible mask 
    if eval_background_covisible_mask: # False
        # from himor
        # https://github.com/hhhddddddd/motiongs/blob/main/eval_utils/eval_dyncheck.py # L358
        gt_masks = torch.tensor(np.array([imageio.imread(osp.join(gt_mask_dir, f)) for f in gt_mask_fns])) # N, H, W
        frame_names = gt_mask_fns
        gt_mask_background = generate_background_masks(gt_masks.numpy(), frame_names) / 255.0 # background covisible_masks

    psnr_list, ssim_list, lpips_list = [], [], []
    mpsnr_all, mssim_all, mlpips_all = [], [], []
    for i in tqdm(range(len(gt_rgb_fns))):
        # gt = imageio.imread(osp.join(gt_rgb_dir, gt_rgb_fns[i])).astype(float) / 255.0 # H,W,3
        gt = cv.imread(osp.join(gt_rgb_dir, gt_rgb_fns[i])) # H,W,3; 
        gt_mask = (imageio.imread(osp.join(gt_mask_dir, gt_mask_fns[i])) > 0).astype(
            float
        )[..., None] # H,W,1 NOTE dynamic mask
        # pred = imageio.imread(osp.join(pred_dir, pred_fns[i])).astype(float) / 255.0 # H,W,3
        pred = cv.imread(osp.join(pred_dir, pred_fns[i])) # H,W,3; 
        
        # iphone data has 4 channels images
        gt = gt[..., :3]
        pred = pred[..., :3]

        import jax

        device_cpu = jax.devices("cpu")[0]
        with jax.default_device(device_cpu):
            from dycheck_metrics import compute_psnr, compute_ssim, compute_lpips

            if eval_non_masked: # False
                tmp_psnr = cv.PSNR(gt, pred) # NOTE no covisible mask
                tmp_ssim = structural_similarity(gt, pred, channel_axis=-1, data_range=255)
                tmp_lpips = lpips_fn.forward(im2tensor(gt), im2tensor(pred)).item()
            
            else:

                tmp_psnr = cv.PSNR(gt, pred)
                tmp_ssim = structural_similarity(gt, pred, channel_axis=-1, data_range=255)
                tmp_lpips = lpips_fn.forward(im2tensor(gt), im2tensor(pred)).item()

        if eval_background_covisible_mask:
            mpsnr, mssim, mlpips = evaluate_nv(gt, pred, gt_mask_background[i])
        else:
            mpsnr, mssim, mlpips = evaluate_nv(gt, pred, gt_mask)
        mpsnr_all.append(mpsnr)
        mssim_all.append(mssim)
        mlpips_all.append(mlpips)

        psnr_list.append(tmp_psnr)
        ssim_list.append(tmp_ssim)
        lpips_list.append(tmp_lpips)

        if i % viz_interval == 0: # 50
            # viz
            gt = imageio.imread(osp.join(gt_rgb_dir, gt_rgb_fns[i])).astype(float) / 255.0 # H,W,3
            pred = imageio.imread(osp.join(pred_dir, pred_fns[i])).astype(float) / 255.0 # H,W,3
            m_error = abs(pred - gt).max(axis=-1) * gt_mask.squeeze(-1)
            m_error = cv.applyColorMap(
                (m_error * 255).astype(np.uint8), cv.COLORMAP_JET
            )[..., [2, 1, 0]]
            error = abs(pred - gt).max(axis=-1)
            error = cv.applyColorMap((error * 255).astype(np.uint8), cv.COLORMAP_JET)[
                ..., [2, 1, 0]
            ]
            viz_img = np.concatenate(
                [gt * 255, pred * 255, error, m_error], axis=1
            ).astype(np.uint8)
            imageio.imwrite(osp.join(save_viz_dir, f"{gt_rgb_fns[i]}"), viz_img)

    ave_psnr = np.mean(psnr_list)
    ave_ssim = np.mean(ssim_list)
    ave_lpips = np.mean(lpips_list)

    ave_mpsnr = np.mean(mpsnr_all)
    ave_mssim = np.mean(mssim_all)
    ave_mlpips = np.mean(mlpips_all)

    logging.info(
        f"ave_psnr: {ave_psnr:.2f}, ave_ssim: {ave_ssim:.4f}, ave_lpips: {ave_lpips:.4f}"
    )
    logging.info(
        f"ave_mpsnr: {ave_mpsnr:.2f}, ave_mssim: {ave_mssim:.4f}, ave_mlpips: {ave_mlpips:.4f}"
    )

    # * save and viz
    # save excel with pandas, each row is a frame
    df = pd.DataFrame(
        {
            "fn": ["AVE"],
            "psnr": [ave_psnr],
            "ssim": [ave_ssim],
            "lpips": [ave_lpips],
            "mpsnr": [ave_mpsnr],
            "mssim": [ave_mssim],
            "mlpips": [ave_mlpips],
        }
    )
    for i in range(len(gt_rgb_fns)):
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "fn": [gt_rgb_fns[i]],
                        "psnr": [psnr_list[i]],
                        "ssim": [ssim_list[i]],
                        "lpips": [lpips_list[i]],
                        "mpsnr": [mpsnr_all[i]],
                        "mssim": [mssim_all[i]],
                        "mlpips": [mlpips_all[i]],
                    }
                ),
            ],
            ignore_index=True,
        )
    save_prefix = save_prefix + ("background_" if eval_background_covisible_mask else "") 
    df.to_excel(osp.join(save_dir, f"{save_prefix}dycheck_metrics.xlsx"), index=False)

    viz_fns = sorted(
        [f for f in os.listdir(save_viz_dir)]
    )
    frames = [imageio.imread(osp.join(save_viz_dir, f)) for f in viz_fns]
    imageio.mimsave(save_viz_dir + ".gif", frames)
    return


if __name__ == "__main__":
    # gt_rgb_dir = "../../data/iphone/spin/test_images/"
    # gt_mask_dir = "../../data/iphone/spin/test_covisible/"
    # pred_rgb_dir = "../../data/iphone/spin/log/20240401_104110/test/"
    # save_dir = "../../data/iphone/spin/log/20240401_104110/"

    # gt_rgb_dir = "../../data/iphone/spin/test_images/"
    # gt_mask_dir = "../../data/iphone/spin/test_covisible/"

    # pred_rgb_dir = ""
    # save_dir = "../../data/iphone/spin/log/20240401_104110/"
    scenes = ["block", "paper-windmill", "space-out", "spin", "teddy", "wheel"]
    project = {}
    project["block"] = "iphone_fit_native_add3_20250426_002431"
    project["paper-windmill"] = "iphone_fit_native_add3_20250426_015123"
    project["space-out"] = "iphone_fit_native_add3_20250426_023903"
    project["spin"] = "iphone_fit_native_add3_20250426_051525"
    project["teddy"] = "iphone_fit_native_add3_20250426_075618"
    project["wheel"] = "iphone_fit_native_add3_20250426_104515"

    for scene in scenes:

        gt_rgb_dir = os.path.join("./data/iphone", scene, "test_images")
        gt_mask_dir = os.path.join("./data/iphone", scene, "test_covisible")

        pred_rgb_dir = os.path.join("./data/iphone", scene, "logs", project[scene], "tto_test")
        save_dir = os.path.join("./data/iphone", scene, "logs", project[scene])

        # eval_dycheck(
        #     save_dir, gt_rgb_dir, gt_mask_dir, pred_rgb_dir, strict_eval_all_gt_flag=False
        # )

        eval_dycheck(
            save_dir, gt_rgb_dir, gt_mask_dir, pred_rgb_dir, strict_eval_all_gt_flag=True
        )
        print(scene, 'is ok!')
