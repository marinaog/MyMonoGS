import json
import os


import cv2
import evo
import numpy as np
import torch
from evo.core import metrics, trajectory
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import plot
from evo.tools.plot import PlotMode
from evo.tools.settings import SETTINGS
from matplotlib import pyplot as plt
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm
import wandb
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.utils.loss_utils import ssim
from gaussian_splatting.utils.system_utils import mkdir_p
from utils.logging_utils import Log
from pathlib import Path
from PIL import Image


def evaluate_evo(poses_gt, poses_est, plot_dir, label, monocular=False):
    ## Plot
    traj_ref = PosePath3D(poses_se3=poses_gt)
    traj_est = PosePath3D(poses_se3=poses_est)
    traj_est_aligned = trajectory.align_trajectory(
        traj_est, traj_ref, correct_scale=monocular
    )

    ## RMSE
    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_ref, traj_est_aligned)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ape_stats = ape_metric.get_all_statistics()
    Log("RMSE ATE [m]", ape_stat, tag="Eval")


    if label == "final":
        print("label should be final but it is ", label)
        with open(
            os.path.join(plot_dir, "stats_{}.json".format(str(label))),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(ape_stats, f, indent=4)

        plot_mode = evo.tools.plot.PlotMode.xy
        fig = plt.figure()
        ax = evo.tools.plot.prepare_axis(fig, plot_mode)
        ax.set_title(f"ATE RMSE: {ape_stat}")
        evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")
        evo.tools.plot.traj_colormap(
            ax,
            traj_est_aligned,
            ape_metric.error,
            plot_mode,
            min_map=ape_stats["min"],
            max_map=ape_stats["max"],
        )
        ax.legend()
        plt.savefig(os.path.join(plot_dir, "evo_2dplot_{}.png".format(str(label))), dpi=90)

    return ape_stat


def eval_ate(frames, kf_ids, save_dir, iterations, final=False, monocular=False):
    trj_data = dict()
    latest_frame_idx = kf_ids[-1] + 2 if final else kf_ids[-1] + 1
    trj_id, trj_est, trj_gt = [], [], []
    trj_est_np, trj_gt_np = [], []

    def gen_pose_matrix(R, T):
        pose = np.eye(4)
        pose[0:3, 0:3] = R.cpu().numpy()
        pose[0:3, 3] = T.cpu().numpy()
        return pose

    for kf_id in kf_ids:
        kf = frames[kf_id]
        pose_est = np.linalg.inv(gen_pose_matrix(kf.R, kf.T))
        pose_gt = np.linalg.inv(gen_pose_matrix(kf.R_gt, kf.T_gt))

        trj_id.append(frames[kf_id].uid)
        trj_est.append(pose_est.tolist())
        trj_gt.append(pose_gt.tolist())

        trj_est_np.append(pose_est)
        trj_gt_np.append(pose_gt)

    trj_data["trj_id"] = trj_id
    trj_data["trj_est"] = trj_est
    trj_data["trj_gt"] = trj_gt

    plot_dir = os.path.join(save_dir, "plot")
    mkdir_p(plot_dir)

    label_evo = "final" if final else "{:04}".format(iterations)
    with open(
        os.path.join(plot_dir, f"trj_{label_evo}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(trj_data, f, indent=4)

    ate = evaluate_evo(
        poses_gt=trj_gt_np,
        poses_est=trj_est_np,
        plot_dir=plot_dir,
        label=label_evo,
        monocular=monocular,
    )
    wandb.log({"frame_idx": latest_frame_idx, "ate": ate})
    return ate

def raw2normal(img, is_torch=False):
    if is_torch:
        bright_factor = 0.98 / torch.quantile(img, 0.99)
        img = torch.clamp(img * bright_factor, 0.0, 1.0)
        
        gamma = 2.4
        slope = 12.92
        threshold = 0.04045 / slope

        out = torch.zeros_like(img)
        low = img <= threshold
        high = img > threshold

        out[low] = img[low] * slope
        out[high] = 1.055 * torch.pow(img[high], 1.0 / gamma) - 0.055

    else:
        bright_factor = 0.98 / np.percentile(img, 99)
        img = np.clip(img * bright_factor, 0, 1)
        
        gamma = 2.4
        slope = 12.92
        threshold = (0.04045 / slope)
        low = img <= threshold 
        high = img > threshold
        out = np.zeros_like(img)
        out[low] = img[low] * slope
        out[high] = 1.055 * (img[high] ** (1/gamma)) - 0.055
    return out

def eval_rendering(
    frames,
    gaussians,
    dataset,
    save_dir,
    pipe,
    background,
    kf_indices,
    iteration="final",
    raw=False,
):
    interval = 1
    save_dir = Path(save_dir)

    (save_dir / "renders").mkdir(parents=True, exist_ok=True)
    (save_dir / "gt").mkdir(parents=True, exist_ok=True)
    print('Saving it ', save_dir)
    img_pred, img_gt, saved_frame_idx = [], [], []
    end_idx = len(frames) - 1 if iteration == "final" or "before_opt" else iteration
    psnr_array, ssim_array, lpips_array = [], [], []
    depth_l1 = []
    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to("cuda")
    s = 0
    print(f'Saving every {int(len(kf_indices)/5)} frames')
    for idx in tqdm(kf_indices):
        saved_frame_idx.append(idx)
        frame = frames[idx]
        gt_image, _, _ = dataset[idx]

        render_pkg = render(frame, gaussians, pipe, background)
        depth = render_pkg["depth"]
        gt_depth = torch.from_numpy(frame.depth).to(depth.device)
        dl1 = torch.abs(depth - gt_depth).mean()
        depth_l1.append(dl1.detach().cpu().float())

        rendering = render_pkg["render"]
        image = torch.clamp(rendering, 0.0, 1.0)

        if raw:
            gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 65535).astype(np.uint16)
            pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 65535).astype(
                np.uint16
            )
            gt = raw2normal(gt)
            gt = (gt * 255).astype(np.uint8)
            pred = raw2normal(pred)
            pred = (pred * 255).astype(np.uint8)
        else: 
            gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
            pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
                np.uint8
            )
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        if s == int(len(kf_indices)/5):
            f"Saving img {idx}"
            s=0
            cv2.imwrite(str(save_dir / "renders" / f'{idx:05d}.png'), pred)
            cv2.imwrite(str(save_dir / "gt" / f'{idx:05d}.png'), gt)
        s+=1
        img_pred.append(pred)
        img_gt.append(gt)

        if raw:
            gt_image = raw2normal(gt_image, is_torch = True)
            image = raw2normal(image, is_torch = True)
        mask = gt_image > 0

        psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
        ssim_score = ssim((image).unsqueeze(0), (gt_image).unsqueeze(0))
        lpips_score = cal_lpips((image).unsqueeze(0), (gt_image).unsqueeze(0))

        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())

    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))
    output["depth_l1"] = float(np.mean(depth_l1))

    # print("depth_l1",output["depth_l1"])
    Log(
        f'depth_l1: {output["depth_l1"]}, mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}',
        tag="Eval",
    )


    psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    mkdir_p(psnr_save_dir)

    json.dump(
        output,
        open(os.path.join(psnr_save_dir, "final_result.json"), "w", encoding="utf-8"),
        indent=4,
    )
    return output


def eval_rendering_both(
    frames,
    gaussians,
    dataset,
    save_dir,
    pipe,
    background,
    kf_indices,
    iteration="final",
    raw=False,
    kf_ind_to_save = None
):
    interval = 1
    save_dir = Path(save_dir)

    (save_dir / "renders").mkdir(parents=True, exist_ok=True)
    (save_dir / "gt").mkdir(parents=True, exist_ok=True)
    print('Saving it ', save_dir)
    img_pred, img_gt, saved_frame_idx = [], [], []
    end_idx = len(frames) - 1 if iteration == "final" or "before_opt" else iteration
    psnr_array, ssim_array, lpips_array = [], [], []
    depth_l1 = []
    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to("cuda")
    # s = 0
    # print(f'Saving every {int(len(kf_indices)/5)} frames')
    for idx in tqdm(kf_indices):
        saved_frame_idx.append(idx)
        frame = frames[idx]
        gt_image, _, _ = dataset[idx]

        render_pkg = render(frame, gaussians, pipe, background)
        depth = render_pkg["depth"]
        gt_depth = torch.from_numpy(frame.depth).to(depth.device)
        dl1 = torch.abs(depth - gt_depth).mean()
        depth_l1.append(dl1.detach().cpu().float())

        rendering = render_pkg["render"]
        image = torch.clamp(rendering, 0.0, 1.0)

        if raw:
            gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 65535).astype(np.uint16)
            pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 65535).astype(
                np.uint16
            )
            gt = raw2normal(gt)
            gt = (gt * 255).astype(np.uint8)
            pred = raw2normal(pred)
            pred = (pred * 255).astype(np.uint8)
        else: 
            gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
            pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
                np.uint8
            )
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        if idx in kf_ind_to_save:
            f"Saving img {idx}"
            cv2.imwrite(str(save_dir / "renders" / f'{idx:05d}.png'), pred)
            cv2.imwrite(str(save_dir / "gt" / f'{idx:05d}.png'), gt)
        img_pred.append(pred)
        img_gt.append(gt)

        if raw:
            gt_image = raw2normal(gt_image, is_torch = True)
            image = raw2normal(image, is_torch = True)
        mask = gt_image > 0

        psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
        ssim_score = ssim((image).unsqueeze(0), (gt_image).unsqueeze(0))
        lpips_score = cal_lpips((image).unsqueeze(0), (gt_image).unsqueeze(0))

        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())

    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))
    output["depth_l1"] = float(np.mean(depth_l1))

    # print("depth_l1",output["depth_l1"])
    Log(
        f'depth_l1: {output["depth_l1"]}, mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}',
        tag="Eval",
    )


    psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    mkdir_p(psnr_save_dir)

    json.dump(
        output,
        open(os.path.join(psnr_save_dir, "final_result.json"), "w", encoding="utf-8"),
        indent=4,
    )
    return output

def save_gaussians(gaussians, name, iteration, final=False):
    if name is None:
        return
    if final:
        point_cloud_path = os.path.join(name, "point_cloud/final")
    else:
        point_cloud_path = os.path.join(
            name, "point_cloud/iteration_{}".format(str(iteration))
        )
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))