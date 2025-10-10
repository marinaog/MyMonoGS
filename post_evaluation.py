""""
This file was created by me to perform evaluation after SLAM, using the saved point cloud and estimated poses.
It uses both, raw and sRGB estimated poses, in order to be able to compare both resulting renders from the same camera poses.
"""""


import torch
from munch import munchify

from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.eval_utils_posteval import eval_rendering
from gaussian_splatting.scene.gaussian_model import GaussianModel
from utils.camera_utils import Camera
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2
import os
import json
import numpy as np
from tqdm import tqdm

both = True
raw = True
scene = "candles"
runnum = "short"

pathresults = f"results/datasets_rawslam/"+scene+runnum
# pathraw = f"results/datasets_rawslam/"+scene+"short_raw"
# pathsrgb = f"results/datasets_rawslam/"+scene+"short_srgb"
if raw:
    scene_name = scene + "_raw"
    ply_path = pathresults + f"_raw/point_cloud/final/point_cloud.ply"
    config_path = f"configs/rgbd/rawslam/{scene_name}.yaml"
    est_pose_path = pathresults + f"_raw/plot/trj_final.json"
else:
    scene_name = scene+"_srgb"
    ply_path = pathresults + f"_srgb/point_cloud/final/point_cloud.ply"
    config_path = f"configs/rgbd/rawslam/{scene}.yaml"
    est_pose_path = pathresults + f"_srgb/plot/trj_final.json"

# est_pose_path = path + f"/plot/trj_final.json"
data_type = "rawslam"
save_dir = f"test{data_type}_{scene_name}_{runnum}"


config = load_config(config_path)
config["Dataset"]["type"] = data_type
pipeline_params = munchify(config["pipeline_params"])
model_params = munchify(config["model_params"])
dataset = load_dataset(model_params, model_params.source_path, config=config)
background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

print('len(dataset)',len(dataset))


opt_params = munchify(config["opt_params"])
gaussians = GaussianModel(model_params.sh_degree, config=config)
# gaussians.training_setup(opt_params)

gaussians.load_ply(ply_path)
projection_matrix = getProjectionMatrix2(znear=0.01, zfar=100.0, fx=dataset.fx, fy=dataset.fy,
                                         cx=dataset.cx, cy=dataset.cy, W=dataset.width, H=dataset.height).transpose(0, 1)
projection_matrix = projection_matrix.to(device="cuda")

cameras = {}

# kf_indices = list(range(0, len(dataset), 5))
kf_indices = []
with open(est_pose_path, 'r') as f:
    data = json.load(f)
kf_indices = data['trj_id']
trj_est = data['trj_est']

for i in tqdm(kf_indices):
    index_of_frame = kf_indices.index(i)
    est_pose = np.linalg.inv(np.array(trj_est[index_of_frame]))
    viewpoint = Camera.init_from_dataset(dataset, i, projection_matrix, postproc = True, pose = est_pose)
    viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)
    viewpoint.compute_grad_mask(config)
    cameras[i] = viewpoint

rendering_result = eval_rendering(
                cameras,
                gaussians,
                dataset,
                save_dir,
                pipeline_params,
                background,
                kf_indices= kf_indices,
                iteration="final",
                raw=raw)