import torch
import os
import json
import numpy as np
import cv2
from tqdm import tqdm
from munch import munchify
from pathlib import Path

# --- Project Imports ---
import utils.color_mlp_arch
from utils.config_utils import load_config
from utils.dataset import load_dataset
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.gaussian_renderer import render
from utils.camera_utils import Camera
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2

# --- Correct import for your specific raw2normal function ---
from utils.eval_utils_posteval import raw2normal

def main():
    paths = [
        "results/datasets_rawslam/candles_original/candles_raw_loss",
        "results/datasets_rawslam/candles_original/candles_raw_loss_2",
        "results/datasets_rawslam/candles_original/candles_raw_loss_3",
        "results/datasets_rawslam/candles_original/candles_srgb",
        "results/datasets_rawslam/candles_original/candles_srgb_2",
        "results/datasets_rawslam/candles_original/candles_srgb_3"
    ]

    scene_base = "candles"
    output_base = Path("comparison_results")
    num_frames_to_save = 7

    # 1. Identify common frames
    common_ids = None
    for p in paths:
        traj_path = os.path.join(p, "plot/trj_final.json")
        if not os.path.exists(traj_path): continue
        with open(traj_path, 'r') as f:
            data = json.load(f)
            ids = set(data['trj_id'])
            common_ids = ids if common_ids is None else common_ids.intersection(ids)

    if not common_ids: return
    kf_list = sorted(list(common_ids))
    indices = np.unique(np.linspace(0, len(kf_list) - 1, num_frames_to_save, dtype=int))
    kf_to_save = [kf_list[i] for i in indices]

    for path in paths:
        is_raw = "raw" in path.lower()
        cfg_name = os.path.basename(path)
        save_dir = output_base / cfg_name
        save_dir.mkdir(parents=True, exist_ok=True)

        result_config_path = os.path.join(path, "config.yml")
        config = load_config(result_config_path if os.path.exists(result_config_path) else f"configs/rgbd/rawslam/{scene_base}.yaml")

        actual_use_mlp = config.get("pipeline_params", {}).get("use_mlp", False) and "mlp_opt_params" in config

        gaussians = GaussianModel(sh_degree=config["model_params"]["sh_degree"], config=config, raw=is_raw, use_mlp=actual_use_mlp)
        gaussians.load_ply(os.path.join(path, "point_cloud/final/point_cloud.ply"))

        if actual_use_mlp:
            gaussians.load_mlp_weights(path)
            gaussians.color_mlp = gaussians.color_mlp.to("cuda:0")

        dataset = load_dataset(munchify(config["model_params"]), munchify(config["model_params"]).source_path, config=config)
        background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda:0")
        proj_matrix = getProjectionMatrix2(znear=0.01, zfar=100.0, fx=dataset.fx, fy=dataset.fy,
                                          cx=dataset.cx, cy=dataset.cy, W=dataset.width, H=dataset.height).transpose(0, 1).cuda()

        with open(os.path.join(path, "plot/trj_final.json"), 'r') as f:
            traj_data = json.load(f)

        print(f"Rendering: {cfg_name}")
        for fid in tqdm(kf_to_save):
            idx = traj_data['trj_id'].index(fid)
            pose = np.linalg.inv(np.array(traj_data['trj_est'][idx]))
            cam = Camera.init_from_dataset(dataset, fid, proj_matrix, postproc=True, pose=pose)
            cam.update_RT(cam.R_gt, cam.T_gt)

            if actual_use_mlp:
                num_pts = gaussians.get_xyz.shape[0]
                b_i = gaussians._features_dc.detach().reshape(num_pts, 3).to("cuda:0")
                f_i = gaussians._features_rest.detach().reshape(num_pts, -1).to("cuda:0")
                xyz = gaussians.get_xyz.detach().to("cuda:0")
                c_center = cam.camera_center
                c_center = torch.tensor(c_center, device="cuda:0", dtype=torch.float32) if not isinstance(c_center, torch.Tensor) else c_center.to("cuda:0").float()

                dir_pp = (xyz - c_center)
                v = (dir_pp / (dir_pp.norm(dim=1, keepdim=True) + 1e-6)).reshape(num_pts, 3).to("cuda:0")

                if hasattr(gaussians.color_mlp, 'final_bias'):
                    gaussians.color_mlp.final_bias = gaussians.color_mlp.final_bias.to("cuda:0")

                with torch.no_grad():
                    f_i = (b_i, f_i)
                    colors_to_pass = gaussians.color_mlp(f_i, v)

                out = render(cam, gaussians, munchify(config["pipeline_params"]), background, colors_precomp=colors_to_pass)
            else:
                out = render(cam, gaussians, munchify(config["pipeline_params"]), background)

            rendering = out["render"]
            image = torch.clamp(rendering, 0.0, 1.0)

            # --- REPLICATING YOUR TRAINING eval_rendering LOGIC EXACTLY ---
            if is_raw:
                pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 65535).astype(np.uint16)
                final_img = (raw2normal(pred.astype(np.uint16)) * 255).astype(np.uint8)
            else:
                final_img = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)

            final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
            # Save via OpenCV
            cv2.imwrite(str(save_dir / f"{fid:05d}.png"), final_img)

    # 3. Grid Generation
    grid_dir = output_base / "grids"
    grid_dir.mkdir(exist_ok=True)
    for fid in kf_to_save:
        imgs = []
        for p in paths:
            f_n = os.path.basename(p)
            img_p = output_base / f_n / f"{fid:05d}.png"
            if img_p.exists():
                tmp = cv2.imread(str(img_p))
                cv2.putText(tmp, f_n, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                imgs.append(tmp)
        if len(imgs) == 6:
            cv2.imwrite(str(grid_dir / f"grid_{fid:05d}.png"), np.vstack([np.hstack(imgs[:3]), np.hstack(imgs[3:])]))

if __name__ == "__main__":
    main()