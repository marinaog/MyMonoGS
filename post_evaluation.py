""""
This file was created by me to perform evaluation after SLAM, using the saved point cloud and estimated poses.
It uses both, raw and sRGB estimated poses, in order to be able to compare both resulting renders from the same camera poses.
"""""
import torch
from munch import munchify

from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.eval_utils_posteval import eval_rendering, eval_rendering_both
from gaussian_splatting.scene.gaussian_model import GaussianModel
from utils.camera_utils import Camera
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2
import os
import json
import numpy as np
from tqdm import tqdm
import argparse

import utils.color_mlp_arch

def _fix_mlp_feature_layout_if_needed(gaussians):
    """Post-eval compatibility fix for legacy saved MLP feature layouts."""
    if not getattr(gaussians, "use_mlp", False):
        return

    # Expected for MLP: _features_dc -> (N, 1, 3), _features_rest -> (N, 1, feat_len)
    f_dc = gaussians._features_dc
    if f_dc.ndim == 3 and f_dc.shape[1] != 1 and f_dc.shape[2] == 1:
        gaussians._features_dc = torch.nn.Parameter(
            f_dc.transpose(1, 2).contiguous().requires_grad_(True)
        )

    f_rest = gaussians._features_rest
    if f_rest.ndim == 3 and f_rest.shape[1] != 1 and f_rest.shape[2] == 1:
        gaussians._features_rest = torch.nn.Parameter(
            f_rest.transpose(1, 2).contiguous().requires_grad_(True)
        )

def _get_model_root_from_ply(ply_path):
    # .../<run_dir>/point_cloud/final/point_cloud.ply -> .../<run_dir>
    return os.path.dirname(os.path.dirname(os.path.dirname(ply_path)))

def main(args):
    # Access arguments from args.<name>
    scene = args.scene
    runnum_raw = "_2"
    runnum_rgb = "_3"
    data_type = args.data_type

    pathresults = f"results/datasets_rawslam/"+scene

    # Paths RAW
    ply_path_raw = pathresults + f"_raw_l1_mlp{runnum_raw}/point_cloud/final/point_cloud.ply"
    config_path_raw = f"configs/rgbd/rawslam/{scene}/{scene}_raw_l1_mlp.yaml"
    est_pose_path_raw = pathresults + f"_raw_l1_mlp{runnum_raw}/plot/trj_final.json"
    save_dir_raw = f"results_posteval/BOTHrawslam_{scene}_raw_l1_mlp{runnum_raw}"

    # Paths sRGB
    ply_path_srgb = pathresults + f"_srgb_l1{runnum_rgb}/point_cloud/final/point_cloud.ply"
    config_path_srgb = f"configs/rgbd/rawslam/{scene}/{scene}.yaml"
    est_pose_path_srgb = pathresults + f"_srgb_l1{runnum_rgb}/plot/trj_final.json"
    save_dir_srgb = f"results_posteval/BOTHrawslam{scene}_srgb_l1{runnum_rgb}"

    # Load config and dataset RAW
    config_raw = load_config(config_path_raw)
    config_raw["Dataset"]["type"] = data_type
    pipeline_params_raw = munchify(config_raw["pipeline_params"])
    model_params_raw = munchify(config_raw["model_params"])
    dataset_raw = load_dataset(model_params_raw, model_params_raw.source_path, config=config_raw)
    background_raw = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    print('len(dataset_raw)',len(dataset_raw))

    # Load config and dataset sRGB
    config_srgb = load_config(config_path_srgb)
    config_srgb["Dataset"]["type"] = data_type
    pipeline_params_srgb = munchify(config_srgb["pipeline_params"])
    model_params_srgb = munchify(config_srgb["model_params"])
    dataset_srgb = load_dataset(model_params_srgb, model_params_srgb.source_path, config=config_srgb)
    background_srgb = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    print('len(dataset_srgb)',len(dataset_srgb))

    # Load pointcloud and gaussians RAW
    opt_params_raw = munchify(config_raw["opt_params"])
    use_mlp_raw = (
        config_raw.get("pipeline_params", {}).get("use_mlp", False)
        and "mlp_opt_params" in config_raw
    )
    gaussians_raw = GaussianModel(
        model_params_raw.sh_degree,
        config=config_raw,
        raw=config_raw["Dataset"].get("raw", False),
        use_mlp=use_mlp_raw,
    )
    gaussians_raw.load_ply(ply_path_raw)
    if use_mlp_raw:
        _fix_mlp_feature_layout_if_needed(gaussians_raw)
        gaussians_raw.load_mlp_weights(_get_model_root_from_ply(ply_path_raw))
    projection_matrix_raw = getProjectionMatrix2(znear=0.01, zfar=100.0, fx=dataset_raw.fx, fy=dataset_raw.fy,
                                            cx=dataset_raw.cx, cy=dataset_raw.cy, W=dataset_raw.width, H=dataset_raw.height).transpose(0, 1)
    projection_matrix_raw = projection_matrix_raw.to(device="cuda")

    # Load pointcloud and gaussians sRGB
    opt_params_srgb = munchify(config_srgb["opt_params"])
    use_mlp_srgb = (
        config_srgb.get("pipeline_params", {}).get("use_mlp", False)
        and "mlp_opt_params" in config_srgb
    )
    gaussians_srgb = GaussianModel(
        model_params_srgb.sh_degree,
        config=config_srgb,
        raw=config_srgb["Dataset"].get("raw", False),
        use_mlp=use_mlp_srgb,
    )
    gaussians_srgb.load_ply(ply_path_srgb)
    if use_mlp_srgb:
        _fix_mlp_feature_layout_if_needed(gaussians_srgb)
        gaussians_srgb.load_mlp_weights(_get_model_root_from_ply(ply_path_srgb))
    projection_matrix_srgb = getProjectionMatrix2(znear=0.01, zfar=100.0, fx=dataset_srgb.fx, fy=dataset_srgb.fy,
                                            cx=dataset_srgb.cx, cy=dataset_srgb.cy, W=dataset_srgb.width, H=dataset_srgb.height).transpose(0, 1)
    projection_matrix_srgb = projection_matrix_srgb.to(device="cuda")

    # Pick keyframes to save that are common to RAW and sRGB
    with open(est_pose_path_raw, 'r') as f:
        data_raw = json.load(f)
    kf_indices_raw = data_raw['trj_id']
    trj_est_raw = data_raw['trj_est']

    with open(est_pose_path_srgb, 'r') as f:
        data_srgb = json.load(f)
    kf_indices_srgb = data_srgb['trj_id']
    trj_est_srgb = data_srgb['trj_est']

    common_kf_indices = list(set(kf_indices_raw).intersection(kf_indices_srgb))
    common_kf_indices.sort()

    if len(common_kf_indices) == 0:
        raise ValueError("No common keyframes found between raw and srgb poses.")
    elif len(common_kf_indices) < 7:
        kf_to_save = common_kf_indices
    else:
        indices = np.linspace(0, len(common_kf_indices) - 1, 7, dtype=int)
        kf_to_save = [common_kf_indices[i] for i in indices]
        print('Number of common keyframes:', len(common_kf_indices))
        print('Keyframes to save:', kf_to_save)

    # Load estimated poses RAW
    print("Loading raw estimated poses...")
    cameras_raw = {}
    for i in tqdm(kf_indices_raw):
        index_of_frame = kf_indices_raw.index(i)
        est_pose = np.linalg.inv(np.array(trj_est_raw[index_of_frame]))
        viewpoint = Camera.init_from_dataset(dataset_raw, i, projection_matrix_raw, postproc = True, pose = est_pose)
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)
        viewpoint.compute_grad_mask(config_raw)
        cameras_raw[i] = viewpoint

    # Load estimated poses sRGB
    print("Loading sRGB estimated poses...")
    cameras_srgb = {}
    for i in tqdm(kf_indices_srgb):
        index_of_frame = kf_indices_srgb.index(i)
        est_pose = np.linalg.inv(np.array(trj_est_srgb[index_of_frame]))
        viewpoint = Camera.init_from_dataset(dataset_srgb, i, projection_matrix_srgb, postproc = True, pose = est_pose)
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)
        viewpoint.compute_grad_mask(config_srgb)
        cameras_srgb[i] = viewpoint

    # Evaluate RAW
    print("Evaluating raw poses...")
    rendering_result_raw = eval_rendering_both(
                    cameras_raw,
                    gaussians_raw,
                    dataset_raw,
                    save_dir_raw,
                    pipeline_params_raw,
                    background_raw,
                    kf_indices= kf_indices_raw,
                    iteration="final",
                    raw=True,
                    kf_ind_to_save = kf_to_save)

    # Evaluate sRGB
    print("Evaluating sRGB poses...")
    rendering_result_srgb = eval_rendering_both(
                    cameras_srgb,
                    gaussians_srgb,
                    dataset_srgb,
                    save_dir_srgb,
                    pipeline_params_srgb,
                    background_srgb,
                    kf_indices= kf_indices_srgb,
                    iteration="final",
                    raw=False,
                    kf_ind_to_save = kf_to_save)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation with specified settings.")
    parser.add_argument("--scene", type=str, default="candles",
                        help="Scene name (e.g., candles, boxes, cabin, etc.)")
    parser.add_argument("--data_type", type=str, default="rawslam",
                        help="Type of dataset (e.g., rawslam, something_else)")

    args = parser.parse_args()
    main(args)
