import torch


def image_gradient(image):
    # Compute image gradient using Scharr Filter
    c = image.shape[0]
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
    )
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0], img_grad_h[0]


def image_gradient_mask(image, eps=0.01):
    # Compute image gradient mask
    c = image.shape[0]
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(
        p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = torch.nn.functional.conv2d(
        p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
    )

    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)


def depth_reg(depth, gt_image, huber_eps=0.1, mask=None):
    mask_v, mask_h = image_gradient_mask(depth)
    gray_grad_v, gray_grad_h = image_gradient(gt_image.mean(dim=0, keepdim=True))
    depth_grad_v, depth_grad_h = image_gradient(depth)
    gray_grad_v, gray_grad_h = gray_grad_v[mask_v], gray_grad_h[mask_h]
    depth_grad_v, depth_grad_h = depth_grad_v[mask_v], depth_grad_h[mask_h]

    w_h = torch.exp(-10 * gray_grad_h**2)
    w_v = torch.exp(-10 * gray_grad_v**2)
    err = (w_h * torch.abs(depth_grad_h)).mean() + (
        w_v * torch.abs(depth_grad_v)
    ).mean()
    return err


def get_loss_tracking(config, image, depth, opacity, viewpoint, initialization=False):
    image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if config["Training"]["monocular"]:
        return get_loss_tracking_rgb(config, image_ab, depth, opacity, viewpoint)
    return get_loss_tracking_rgbd(config, image_ab, depth, opacity, viewpoint)


def get_loss_tracking_rgb(config, image, depth, opacity, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask
    if config.get("Training") and config["Training"].get("loss") == "rawnerf":
        eps = 1e-2 
        rgb_render_clip = torch.clamp(image, max=1.0)
        resid_sq = (rgb_render_clip - gt_image) ** 2
        
        # Scaling by the gradient of the log curve: 1 / (x + eps)
        # We detach the denominator so it acts as a fixed weight per pixel
        scaling_grad = 1.0 / (rgb_render_clip.detach() + eps)
        
        # Apply mask and opacity weighting
        # We include opacity because in tracking, we only want to trust well-reconstructed regions
        loss_rgb = (resid_sq * (scaling_grad ** 2)) * rgb_pixel_mask * opacity
    else:
        # Original L1 loss
        loss_rgb = opacity * torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    
    return loss_rgb.sum() / (rgb_pixel_mask.sum() + 1e-6)


def get_loss_tracking_rgbd(
    config, image, depth, opacity, viewpoint, initialization=False
):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    opacity_mask = (opacity > 0.95).view(*depth.shape)

    l1_rgb = get_loss_tracking_rgb(config, image, depth, opacity, viewpoint)
    depth_mask = depth_pixel_mask * opacity_mask
    l1_depth = torch.abs(depth * depth_mask - gt_depth * depth_mask)
    return alpha * l1_rgb + (1 - alpha) * l1_depth.mean()


def get_loss_mapping(config, image, depth, viewpoint, opacity, initialization=False):
    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if config["Training"]["monocular"]:
        return get_loss_mapping_rgb(config, image_ab, depth, viewpoint)
    return get_loss_mapping_rgbd(config, image_ab, depth, viewpoint)


def get_loss_mapping_rgb(config, image, depth, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    
    if config["Training"]["loss"] == "rawnerf":
        rgb_render_clip = torch.clamp(image, max=1.0)
        resid_sq_clip = (rgb_render_clip - gt_image) ** 2
        resid_sq_clip_masked = resid_sq_clip * rgb_pixel_mask
        # Scale by gradient of log tonemapping curve.
        scaling_grad = 1.0 / (rgb_render_clip.detach() + 1e-2)
        # Reweighted L2 loss.
        loss_rgb = (resid_sq_clip_masked * scaling_grad**2)     

    else: # default and original l1 loss 
        loss_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)

    return loss_rgb.sum() / (rgb_pixel_mask.sum() + 1e-6) # So that the mean is only among the valid pixels


def get_loss_mapping_rgbd(config, image, depth, viewpoint, initialization=False):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    gt_image = viewpoint.original_image.cuda()

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)

    if config["Training"].get("loss") and config["Training"]["loss"] == "rawnerf":
        rgb_render_clip = torch.clamp(image, max=1.0)
        resid_sq_clip = (rgb_render_clip - gt_image) ** 2
        resid_sq_clip_masked = resid_sq_clip * rgb_pixel_mask
        # Scale by gradient of log tonemapping curve.
        scaling_grad = 1.0 / (rgb_render_clip.detach() + 1e-2)
        # Reweighted L2 loss.
        loss_rgb = (resid_sq_clip_masked * scaling_grad**2)     

    else: # default and original l1 loss 
        loss_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    
    l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)

    return alpha * loss_rgb.sum() / (rgb_pixel_mask.sum() + 1e-6) + (1 - alpha) * l1_depth.mean()


def get_median_depth(depth, opacity=None, mask=None, return_std=False):
    depth = depth.detach().clone()
    opacity = opacity.detach()
    valid = depth > 0
    if opacity is not None:
        valid = torch.logical_and(valid, opacity > 0.95)
    if mask is not None:
        valid = torch.logical_and(valid, mask)
    valid_depth = depth[valid]
    if return_std:
        return valid_depth.median(), valid_depth.std(), valid
    return valid_depth.median()

def get_reg_loss(self, gaussians, viewpoint, render_pkg, config):
    # NearFarReg
    loss_nearfar = get_loss_nearfar(self, gaussians, viewpoint, render_pkg)

    # DistortionReg
    loss_dist = get_loss_dist(gaussians, viewpoint, render_pkg, config)  

    return 0.1 * loss_dist + 0.01 * loss_nearfar

def get_loss_nearfar(self, gaussians, viewpoint, render_pkg):
    H, W = viewpoint.image_height, viewpoint.image_width
    near_far_indexes = render_pkg['near_idx'].detach(), render_pkg['far_idx'].detach()

    near_far_pack = render_near_far(self, gaussians, viewpoint, near_far_indexes, shape=(H, W))
    near   = near_far_pack['near']
    far    = near_far_pack['far']
    near_T = near_far_pack['near_final_opacity']
    far_T  = near_far_pack['far_final_opacity']

    if near.ndim == 3: near = near.squeeze(0)
    if far.ndim == 3: far = far.squeeze(0)
    if near_T.ndim == 3: near_T = near_T.squeeze(0)
    if far_T.ndim == 3: far_T = far_T.squeeze(0)

    with torch.no_grad():
        max_near = near.max().item()
        max_far = far.max().item()
        min_near_T = near_T.min().item()
        
        # Si detectamos valores sospechosos, imprimimos
        if max_near > 100 or max_far > 100 or torch.isnan(near).any():
            print(f"\n[DEBUG Near-Far] Val_Max: Near={max_near:.2f}, Far={max_far:.2f}")
            print(f"[DEBUG Near-Far] Opacidad Min: {min_near_T:.6f}")
    near   = near / (near_T + 1e-6)
    far    = far / (far_T + 1e-6)
    mask = ~((torch.isnan(near) | torch.isinf(near)) | (torch.isnan(far) | torch.isinf(far)))
    l_near_far_reg = torch.abs(near[mask] - far[mask]) * near_T[mask] * far_T[mask]
    # l_near_far_reg = weight_reduce_loss(l_near_far_reg, weight, reduction=self.reduction)
    final_loss = l_near_far_reg.mean()
    if final_loss > 10.0 or torch.isnan(final_loss):
        print(f"⚠️ Alerta Loss: {final_loss.item():.4f} | Probable causa de colapso")
    return l_near_far_reg.clamp(max=10.0).mean()

def get_loss_dist(gaussians, viewpoint, render_pkg, config, res_scale=0.5):
    from gaussian_splatting.gaussian_renderer import render_depth_raywise, render_hist
    assert 'near_idx' in render_pkg and 'far_idx' in render_pkg
    H, W = int(viewpoint.image_height * res_scale), int(viewpoint.image_width * res_scale)
    bins = config["Training"].get("bins", 64)

    with torch.no_grad():
        # render_depth_raywise proyecta los centros de los Gaussians a la cámara
        xx, yy, zz, pixel_filter = render_depth_raywise(gaussians, viewpoint, shape=(H, W), return_filter=True, return_all=True)
        visible_zz = zz[pixel_filter]
        # Si no hay puntos visibles, evitamos el crash
        near_val = max(0.2, visible_zz.min().item()) if visible_zz.numel() > 0 else 0.2
        far_val  = min(1000.0, visible_zz.max().item()) if visible_zz.numel() > 0 else 1000.0
        
    out_pkg = render_hist(gaussians, viewpoint, bins, (H, W), (near_val, far_val))
    
    hist = out_pkg['render']
    curr_rays = hist.permute(1, 2, 0).reshape(-1, bins)

    loss_dist = dist_loss(near_val, far_val, curr_rays, inter_weight=1.0, intra_weight=1.0)
    return loss_dist.mean()


def render_near_far(self, gs_model, viewpoint_camera, near_far_indexes, shape=None):
    from gaussian_splatting.gaussian_renderer import render_depth_with_filter
    with torch.no_grad():
        near_indexes = near_far_indexes[0]
        far_indexes = near_far_indexes[1]
        uniq_near_indexes = torch.unique(near_indexes[near_indexes != -1]).long()
        uniq_far_indexes = torch.unique(far_indexes[far_indexes != -1]).long()
        near_masks = torch.zeros_like(gs_model._opacity).squeeze()
        near_masks[uniq_near_indexes] = 1
        far_masks  = torch.zeros_like(gs_model._opacity).squeeze()
        far_masks[uniq_far_indexes] = 1
        near_masks = near_masks.bool()
        far_masks  = far_masks.bool()

    far_pack  = render_depth_with_filter(gs_model, viewpoint_camera, far_masks,  self.background, shape=shape)
    near_pack = render_depth_with_filter(gs_model, viewpoint_camera, near_masks, self.background, shape=shape)

    return {'near': near_pack['render'],
            'near_final_opacity': near_pack['final_opacity'],
            'far': far_pack['render'],
            'far_final_opacity': far_pack['final_opacity']}


def dist_loss(near, far, ws, inter_weight=1.0, intra_weight=1.0, eps=torch.finfo(torch.float32).eps):
    g = lambda x : 1 / x
    bins = ws.size(1)
    t = torch.linspace(near+eps, far, bins+1, device=ws.device)  # same naming as multinerf
    s = (g(t) - g(near+eps)) / (g(far) - g(near+eps))            # convert t to s
    us = (s[1:] + s[:-1]) / 2
    dus = torch.abs(us[:, None] - us[None, :])
    loss_inter = torch.sum(ws * torch.sum(ws[..., None, :] * dus[None, ...], dim=-1), dim=-1)
    ds = s[1:] - s[:-1]
    loss_intra = torch.sum(ws**2 * ds[None, :], dim=-1) / 3
    return loss_inter * inter_weight + loss_intra * intra_weight