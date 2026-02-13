#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math

import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.sh_utils import eval_sh
from gaussian_splatting.utils.graphics_utils import fov2focal

from hist_rasterization import (
    GaussianRasterizationSettings as GaussianRasterizationSettings32,
    GaussianRasterizer as GaussianRasterizer32
)

def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    mask=None,
    colors_precomp=None
):
    """
    Render the scene.
    Background tensor (bg_color) must be on GPU!
    """

    if pc.get_xyz.shape[0] == 0:
        return None

    # Create zero tensor for gradients of the 2D means
    screenspace_points = torch.zeros_like(
        pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
    ) + 0
    
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        projmatrix_raw=viewpoint_camera.projection_matrix,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling.repeat(1, 3) if pc.get_scaling.shape[-1] == 1 else pc.get_scaling
        rotations = pc.get_rotation

    # --- UPDATED COLOR LOGIC ---
    shs = None
    # If colors are NOT provided by the script, use the original logic
    if colors_precomp is None:
        if not pc.use_mlp:
            if pipe.convert_SHs_python:
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
                dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = pc.get_features
        else:
            if hasattr(pc, 'get_mlp_color'):
                colors_precomp = pc.get_mlp_color(viewpoint_camera)
            elif hasattr(pc, 'mlp_colors') and pc.mlp_colors is not None:
                colors_precomp = pc.mlp_colors
    else:
        shs = None

    # Rasterization call
    if mask is not None:
        rendered_image, radii, depth, opacity, n_touched, near_idx, far_idx = rasterizer(
            means3D=means3D[mask],
            means2D=means2D[mask],
            shs=shs[mask] if shs is not None else None, # Added safety check
            colors_precomp=colors_precomp[mask] if colors_precomp is not None else None,
            opacities=opacity[mask],
            scales=scales[mask] if scales is not None else None,
            rotations=rotations[mask] if rotations is not None else None,
            cov3D_precomp=cov3D_precomp[mask] if cov3D_precomp is not None else None,
            theta=viewpoint_camera.cam_rot_delta,
            rho=viewpoint_camera.cam_trans_delta,
        )
    else:
        rendered_image, radii, depth, opacity, n_touched, near_idx, far_idx = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
            theta=viewpoint_camera.cam_rot_delta,
            rho=viewpoint_camera.cam_trans_delta,
        )

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": depth,
        "opacity": opacity,
        "n_touched": n_touched,
        "near_idx": near_idx, 
        "far_idx": far_idx
    }

def render_depth_with_filter(gs_model, viewpoint_camera, mask, background, shape=None):
    xyz = gs_model.get_xyz[mask, :]
    opacity = gs_model.get_opacity[mask, :]
    scales = gs_model.get_scaling[mask, :]
    rotations = gs_model.get_rotation[mask, :]

    screenspace_points = torch.zeros_like(xyz, requires_grad=True)
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    H, W = shape if shape is not None else (int(viewpoint_camera.image_height), int(viewpoint_camera.image_width))

    raster_settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=background, # TODO: replace bg_color as depth_max
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        projmatrix_raw=viewpoint_camera.projection_matrix,
        sh_degree=gs_model.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    colors_precomp = torch.zeros((xyz.shape[0], 3), device=xyz.device)

    _, radii, depth, final_opacity, _, _, _ = rasterizer(
        means3D=xyz, means2D=screenspace_points, shs=None, colors_precomp=colors_precomp,
        opacities=opacity, scales=scales, rotations=rotations, cov3D_precomp=None
    )

    return {
        "render": depth, "final_opacity": final_opacity,
        "viewspace_points": screenspace_points, "visibility_filter": radii > 0, "radii": radii
    }

# def render_near_far(gs_model, viewpoint_camera, near_far_indexes, shape=None):
#     with torch.no_grad():
#         near_masks = torch.zeros_like(gs_model._opacity).squeeze().bool()
#         far_masks = torch.zeros_like(gs_model._opacity).squeeze().bool()
        
#         near_idx = near_far_indexes[0]
#         far_idx = near_far_indexes[1]
        
#         near_masks[near_idx[near_idx != -1].long()] = True
#         far_masks[far_idx[far_idx != -1].long()] = True

#     far_pack = render_depth_with_filter(gs_model, viewpoint_camera, far_masks, shape=shape)
#     near_pack = render_depth_with_filter(gs_model, viewpoint_camera, near_masks, shape=shape)

#     return {
#         'near': near_pack['render'], 'near_final_opacity': near_pack['final_opacity'],
#         'far': far_pack['render'], 'far_final_opacity': far_pack['final_opacity']
#     }

def render_depth_raywise(gs_model, viewpoint_camera, shape=None, return_filter=False, return_all=False):
    H, W = shape if shape is not None else (viewpoint_camera.image_height, viewpoint_camera.image_width)
    
    w2c = viewpoint_camera.world_view_transform.transpose(0, 1)
    point_at_cam_view = (w2c[:3, :3] @ gs_model.get_xyz.transpose(0, 1) + w2c[:3, -1:]).transpose(0, 1)
    point_z = point_at_cam_view[:, -1:]

    focal_x = fov2focal(viewpoint_camera.FoVx, W)
    focal_y = fov2focal(viewpoint_camera.FoVy, H)
    c2i = torch.tensor([[focal_x, 0, W / 2], [0, focal_y, H / 2], [0, 0, 1]], device=point_z.device)
    
    point_at_im_view = (c2i @ point_at_cam_view.transpose(0, 1)).transpose(0, 1)
    point_at_im_view = (point_at_im_view / point_at_im_view[:, -1:]).round().long()
    
    x, y = point_at_im_view[:, 0], point_at_im_view[:, 1]
    
    pixel_filter = None
    if (not return_all) or return_filter:
        pixel_filter = (x >= 0) & (x < W) & (y >= 0) & (y < H)

    if return_all:
        res = [x, y, point_z.squeeze()]
        if return_filter:
            res.append(pixel_filter) # Añadimos el 4º valor si se pide
        return res
    
    res = [x[pixel_filter], y[pixel_filter], point_z.squeeze()[pixel_filter]]
    if return_filter:
        res.append(pixel_filter)
    return res

def render_hist(gs_model, viewpoint_camera, num_bins=128, shape=None, near_far=None, scaling_modifier=1.0):
    NUM_CHANNELS = 32
    assert num_bins % NUM_CHANNELS == 0, 'num_bins must be divisible by 32!'

    w2c = viewpoint_camera.world_view_transform.transpose(0, 1)
    point_at_cam_view = (w2c[:3, :3] @ gs_model.get_xyz.transpose(0, 1) + w2c[:3, -1:]).transpose(0, 1)
    point_z = point_at_cam_view[:, -1]

    screenspace_points = torch.zeros_like(gs_model.get_xyz, requires_grad=True)
    try: screenspace_points.retain_grad()
    except: pass

    H, W = shape if shape is not None else (int(viewpoint_camera.image_height), int(viewpoint_camera.image_width))
    raster_settings = GaussianRasterizationSettings32(
        image_height=H, image_width=W, tanfovx=math.tan(viewpoint_camera.FoVx * 0.5),
        tanfovy=math.tan(viewpoint_camera.FoVy * 0.5), bg=torch.zeros(NUM_CHANNELS, device=point_z.device),
        scale_modifier=scaling_modifier, viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform, sh_degree=gs_model.active_sh_degree,
        campos=viewpoint_camera.camera_center, prefiltered=False
    )

    rasterizer = GaussianRasterizer32(raster_settings=raster_settings)

    with torch.no_grad():
        near = max(point_z.min().item(), 0) if near_far is None else near_far[0]
        far = point_z.max().item() if near_far is None else near_far[1]
        z_blocksize = (far - near) / (num_bins - 1)
        point_z_block = torch.floor((point_z - near) / z_blocksize).long().clip(0, num_bins - 1)
        colors_precomp = torch.nn.functional.one_hot(point_z_block, num_classes=num_bins).float()
        if near_far is None:
            colors_precomp[(point_z < near) | (point_z > far), :] = 0

    common_args = {
        "means3D": gs_model.get_xyz, "means2D": screenspace_points, "shs": None,
        "opacities": gs_model.get_opacity, "scales": gs_model.get_scaling,
        "rotations": gs_model.get_rotation, "cov3D_precomp": None
    }

    if num_bins == NUM_CHANNELS:
        rendered_image, radii, _ = rasterizer(colors_precomp=colors_precomp, **common_args)
        return {"render": rendered_image, "near": near, "far": far, "radii": radii, "visibility_filter": radii > 0}
    else:
        rendered_hist = []
        for i in range(num_bins // NUM_CHANNELS):
            curr_colors = colors_precomp[:, i*NUM_CHANNELS : (i+1)*NUM_CHANNELS]
            res = rasterizer(colors_precomp=curr_colors, **common_args)
            rendered_image = res[0]
            radii = res[1]
            rendered_hist.append(rendered_image)
        return {
            "render": torch.cat(rendered_hist),
            "near": near, 
            "far": far,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii
        }