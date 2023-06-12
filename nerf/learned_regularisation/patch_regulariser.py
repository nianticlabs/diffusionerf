from dataclasses import dataclass
import cv2
import random
from typing import Optional, Dict, Tuple

import matplotlib.cm
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.functional import grid_sample

from nerf.learned_regularisation.diffusion.denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer, \
    normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from nerf.learned_regularisation.intrinsics import Intrinsics
from nerf.learned_regularisation.patch_pose_generator import PatchPoseGenerator, FrustumRegulariser
from nerf.renderer import NeRFRenderer
from nerf.utils import get_rays


class DepthPreprocessor:
    """
    Preprocesses arrays of depths for feeding into the diffusion model.
    Must be used with the same arguments for both train and test.
    """
    def __init__(self, min_depth: float):
        self.min_depth = min_depth

    def __call__(self, depth):
        """
        :param depth: Array of depths.
        :returns: Inverse depths, clipped so that the minimum depth is self.min_depth
        """
        depth = torch.maximum(depth, torch.full_like(input=depth, fill_value=self.min_depth))

        # Depth in range [self._min_depth, inf] -> inv_depth in range [0, 1/self._min_depth]
        inv_depth = 1. / depth

        # Linearly transform to range [0, 1], just like an rgb channel. The trainer will then transform to [-1, 1].
        inv_depth = inv_depth * self.min_depth

        return inv_depth

    def invert(self, inv_depth):
        inv_depth = inv_depth / self.min_depth

        depth = 1. / inv_depth

        return depth


# Approximately matches the intrinsics for the FOV dataset
LLFF_DEFAULT_PSEUDO_INTRINSICS = Intrinsics(
    fx=700.,
    fy=700.,
    cx=512.,
    cy=384.,
    width=1024,
    height=768,
)


def make_random_patch_intrinsics(patch_size: int, full_image_intrinsics: Intrinsics,
                                 downscale_factor: int = 1) -> Intrinsics:
    """
    Makes intrinsics corresponding to a random patch sampled from the original image.
    This is required when we want to sample a patch from a training image rather than render one.
    :param patch_size: Size of patch in pixels
    :param full_image_intrinsics: Intrinsics of full original image
    :param downscale_factor: Number of original image pixels per patch pixel. If 1, no downscaling occurs
    :return: Intrinsics for patch as described above
    """
    intrinsics_downscaled = Intrinsics(
        fx=full_image_intrinsics.fx // downscale_factor,
        fy=full_image_intrinsics.fy // downscale_factor,
        cx=full_image_intrinsics.cx // downscale_factor,
        cy=full_image_intrinsics.cy // downscale_factor,
        width=full_image_intrinsics.width // downscale_factor,
        height=full_image_intrinsics.height // downscale_factor,
    )

    # Allow our sampled patch to extend past the edges of the img, as long as it overlaps at least marginally with it
    extra_margin = patch_size/2.

    delta_x = intrinsics_downscaled.width - patch_size
    delta_y = intrinsics_downscaled.height - patch_size
    patch_centre_x = random.uniform(intrinsics_downscaled.cx - delta_x - extra_margin,
                                    intrinsics_downscaled.cx + extra_margin)
    patch_centre_y = random.uniform(intrinsics_downscaled.cy - delta_y,
                                    intrinsics_downscaled.cy + extra_margin)

    return Intrinsics(
        fx=intrinsics_downscaled.fx,
        fy=intrinsics_downscaled.fy,
        cx=patch_centre_x,
        cy=patch_centre_y,
        width=patch_size,
        height=patch_size,
    )


def load_patch_diffusion_model(path: Path) -> nn.Module:
    """
    Load the patch denoising diffusion model.
    :param path: Path to a checkpoint for the model.
    """
    image_size = 48
    reg_checkpoint_path = path
    channels = 4
    denoising_model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=channels,
        self_condition=False,
    ).cuda()

    diffusion = GaussianDiffusion(
        denoising_model,
        image_size=image_size,
        timesteps=1000,  # number of steps
        sampling_timesteps=250,
        loss_type='l1',  # L1 or L2
    ).cuda()
    trainer = Trainer(
        diffusion,
        [None],
        train_batch_size=1,
        train_lr=1e-4,
        train_num_steps=700000,  # total training steps
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=False,  # turn on mixed precision
        results_folder=reg_checkpoint_path.parent,
        save_and_sample_every=250,
        num_samples=1,
        tb_path=None,
    )
    trainer.load(str(reg_checkpoint_path))
    trainer.ema.ema_model.eval()
    return trainer.ema.ema_model


class DiffusionTimeHandler:
    def __init__(self, diffusion_model: GaussianDiffusion):
        times = torch.linspace(-1, diffusion_model.num_timesteps - 1,
                               steps=diffusion_model.sampling_timesteps + 1)  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        self._time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

    def get_timesteps(self, time: float) -> Tuple[int, int]:
        """
        :param time: Value of tau, the diffusion time parameter which runs from one to zero during denoising.
        :return: Tuple of (current, next) integer timestep for the diffusion model.
        """
        time = 1. - time
        assert 0. <= time <= 1.
        time_idx = int(round(time * len(self._time_pairs)))
        time_idx = min(time_idx, len(self._time_pairs) - 1)
        time_idx = max(time_idx, 0)

        time, time_next = self._time_pairs[time_idx]
        return time, time_next


@dataclass
class PatchOutputs:
    # Dataclass to store all outputs of a call to the PatchRegulariser.
    images: Dict[str, torch.Tensor]
    loss: torch.Tensor
    rgb_patch: torch.Tensor
    depth_patch: torch.Tensor
    disp_patch: torch.Tensor
    render_outputs: Dict


DISPARITY_CMAP = matplotlib.cm.get_cmap('plasma')
DEPTH_CMAP = matplotlib.cm.get_cmap('plasma')


class PatchRegulariser:
    """
    Main class for using the denoising diffusion patch model to regularise NeRFs.
    """
    def __init__(self, pose_generator: PatchPoseGenerator, patch_diffusion_model: GaussianDiffusion,
                 full_image_intrinsics: Intrinsics, device,
                 planar_depths: bool, frustum_regulariser: Optional[FrustumRegulariser], image_sample_prob: float = 0.,
                 uniform_in_depth_space: bool = False,
                 sample_downscale_factor: int = 4):
        """
        :param pose_generator: PatchPoseGenerator which will be used to provide camera poses from which to render
            patches.
        :param patch_diffusion_model: Denoising diffusion model to use as the score function for regularisation.
        :param full_image_intrinsics: Intrinsics for the training images.
        :param device: Torch device to do calculations on.
        :param planar_depths: Should be true if the diffusion model was trained using depths projected along the z-axis
            rather than Cartesian distances.
        :param frustum_regulariser: Frustum regulariser instance, or None of the frustum loss is not desired.
        :param image_sample_prob: Fraction of the time to sample a patch from a training image directly, rather than
            rendering.
        :param uniform_in_depth_space: If True, losses will be normalised w.r.t. the depth as described in the paper.
        :param sample_downscale_factor: Downscale factor to apply before sampling. This will allow the patch to
            correspond to a wider FOV.
        """
        self._pose_generator = pose_generator
        self._device = device
        self._diffusion_model = patch_diffusion_model.to(self._device)
        self._planar_depths = planar_depths
        self._image_sample_prob = image_sample_prob
        self.frustum_regulariser = frustum_regulariser
        self._uniform_in_depth_space = uniform_in_depth_space
        self._sample_downscale_factor = sample_downscale_factor
        self._depth_preprocessor = DepthPreprocessor(min_depth=0.2)
        self._patch_size = 48
        self._full_image_intrinsics = full_image_intrinsics
        self._time_handler = DiffusionTimeHandler(diffusion_model=self._diffusion_model)

        print('Num channels in diffusion model:', self._diffusion_model.channels)

    def get_diffusion_loss_with_rendered_patch(self, model: NeRFRenderer, time) -> PatchOutputs:
        depth_patch, rgb_patch, render_outputs = self._render_random_patch(model)
        return self.get_loss_for_patch(depth_patch=depth_patch, rgb_patch=rgb_patch, time=time,
                                       render_outputs=render_outputs)

    def get_diffusion_loss_with_sampled_patch(self, model: NeRFRenderer, time, image, image_intrinsics,
                                              pose) -> PatchOutputs:
        # As described in the paper, we sometimes sample a patch from the image rather than rendering it using the NeRF.
        # This function does that (though we still have to render the depth channel using the NeRF).
        while True:
            try:
                depth_patch, rgb_patch, render_outputs = self._sample_patch(
                    image=image, image_intrinsics=image_intrinsics, pose=pose, model=model
                )
                return self.get_loss_for_patch(depth_patch=depth_patch, rgb_patch=rgb_patch, time=time,
                                               update_depth_only=True, render_outputs=render_outputs)
            except AssertionError as e:
                print('Exception:', str(e))

    def get_loss_for_patch(self, depth_patch, rgb_patch, time, render_outputs,
                           update_depth_only: bool = False) -> PatchOutputs:
        disparity_patch = self._depth_preprocessor(depth_patch)

        time, time_next = self._time_handler.get_timesteps(time)

        # Possibly regularise RGB too, if the model has that
        if self._diffusion_model.channels == 4:
            patch = torch.cat([rgb_patch, disparity_patch], dim=-1)
        elif self._diffusion_model.channels == 1:
            patch = disparity_patch
        else:
            raise ValueError('Diffusion model must have 1 channel (D) or 4 channels (RGBD)')

        # Go from [B, H, W, C] to [B, C, H, W]
        patch = patch.moveaxis(-1, 1)

        patch = normalize_to_neg_one_to_one(patch)

        model_predictions = self._diffusion_model.model_predictions(
            x=patch,
            t=torch.Tensor([time], ).to(torch.int64).to(self._device),
            clip_x_start=True
        )
        sigma_lambda = (1. - self._diffusion_model.alphas_cumprod[time]).sqrt()
        assert sigma_lambda > 0.
        grad_log_prior_prob = -model_predictions.pred_noise.detach() * (1. / sigma_lambda)

        # Multipliers so that the input weight parameters for depth and rgb can be kept close to unity for convenience
        depth_weight = 2e-6
        rgb_weight = 1e-9 if not update_depth_only else 0.

        # NOTE: below we compute a loss L = -(constant * patch * grad log P).
        # This is done so that dL/d(patch) = -constant * grad log P, so that we are injecting grad log P into
        #   the gradients while we fit our nerfs.

        # First calculate the loss for the depth channel
        # If requested, we also normalise by multiplying by the inverse depth (i.e. dividing by the depth),
        #   as described in the supplemental.
        if self._uniform_in_depth_space:
            depth_patch_detached = depth_patch.moveaxis(-1, 1).detach()
            normalisation_const = 1.
            multiplier = depth_patch_detached
            assert multiplier.isfinite().all()
            diffusion_pseudo_loss = -torch.sum(depth_weight * multiplier * grad_log_prior_prob[:, -1, :, :] * patch[:, -1, :, :])
        else:
            diffusion_pseudo_loss = -torch.sum(depth_weight * grad_log_prior_prob[:, -1, :, :] * patch[:, -1, :, :])

        # 4-channel models are assumed to be RGBD:
        if self._diffusion_model.channels == 4:
            normalisation_const = 3000
            multiplier = normalisation_const / torch.linalg.norm(grad_log_prior_prob[:, :-1, :, :].detach())
            diffusion_pseudo_loss += -torch.sum(multiplier * rgb_weight * grad_log_prior_prob[:, :-1, :, :] * patch[:, :-1, :, :])

        assert grad_log_prior_prob.isfinite().all()

        pred_noise_bhwc = unnormalize_to_zero_to_one(torch.moveaxis(model_predictions.pred_noise, 1, -1))
        pred_x0_bhwc = unnormalize_to_zero_to_one(torch.moveaxis(model_predictions.pred_x_start, 1, -1))

        patch_outputs = PatchOutputs(
            images={
                'rendered_rgb': rgb_patch,
                'rendered_depth': depth_patch,
                'rendered_disp': disparity_patch,
                'pred_disp_noise': pred_noise_bhwc[..., -1],
                'pred_disp_x0': pred_x0_bhwc[..., -1],
                'pred_depth_x0': self._depth_preprocessor.invert(pred_x0_bhwc[..., -1]),
            },
            loss=diffusion_pseudo_loss,
            depth_patch=depth_patch,
            disp_patch=disparity_patch,
            rgb_patch=rgb_patch,
            render_outputs=render_outputs,
        )
        # Also compute 'what would the patch look like if we took a step in the
        #   direction that the diffusion model wants to go?'
        # The scale factor which multiplies the step below is basically arbitrary since this is just
        #   for visualisation purposes.
        step_scale_factor = 5e-4
        patch_outputs.images['disp_plus_step'] = patch_outputs.images['rendered_disp'] - \
            patch_outputs.images['pred_disp_noise'].unsqueeze(-1) * step_scale_factor / sigma_lambda

        if self._diffusion_model.channels == 4:
            patch_outputs.images['pred_rgb_noise'] = pred_noise_bhwc[..., :-1]
            patch_outputs.images['pred_rgb_x0'] = pred_x0_bhwc[..., :-1]

        return patch_outputs

    def _render_random_patch(self, model: NeRFRenderer):
        pose = self._pose_generator.generate_random().to(self._device)
        print('Pose', pose)
        intrinsics = self._get_random_patch_intrinsics()
        print('Patch intrinsics', intrinsics)
        pred_depth, pred_rgb, _, render_outputs = self._render_patch_with_intrinsics(intrinsics=intrinsics,
                                                                                     pose=pose, model=model)
        return pred_depth, pred_rgb, render_outputs

    def _render_patch_with_intrinsics(self, intrinsics, pose, model):
        pseudo_intrinsics = (intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy)
        patch_rays = get_rays(poses=pose.unsqueeze(0), intrinsics=pseudo_intrinsics,
                              H=self._patch_size, W=self._patch_size, N=-1)
        outputs = model.render(patch_rays['rays_o'], patch_rays['rays_d'])

        if self._planar_depths:
            depth = outputs['depth'] * patch_rays['rays_d_cam_z']
        else:
            depth = outputs['depth']

        B = 1
        pred_depth = depth.reshape(B, intrinsics.height, intrinsics.width, 1)
        pred_rgb = outputs['image'].reshape(B, intrinsics.height, intrinsics.width, 3)

        return pred_depth, pred_rgb, patch_rays, outputs

    def _sample_patch(self, image, image_intrinsics, pose, model):
        patch_intrinsics = self._get_random_patch_intrinsics()
        rendered_depth, rendered_rgb, patch_rays, render_outputs = self._render_patch_with_intrinsics(
            intrinsics=patch_intrinsics, pose=pose, model=model
        )
        print('pose', pose)
        with torch.no_grad():
            gt_rgb = sample_patch_from_img(rays_d=patch_rays['rays_d_cam'], img=image,
                                           img_intrinsics=image_intrinsics, patch_size=self._patch_size)

        return rendered_depth, gt_rgb, render_outputs

    def _get_random_patch_intrinsics(self) -> Intrinsics:
        return make_random_patch_intrinsics(
            patch_size=self._patch_size,
            full_image_intrinsics=self._full_image_intrinsics,
            downscale_factor=self._sample_downscale_factor,
        )

    def dump_debug_visualisations(self, output_folder: Path, output_prefix: str, patch_outputs: PatchOutputs) -> None:
        disp_keys = ('pred_disp_x0', 'rendered_disp', 'disp_plus_step')
        depth_keys = ('pred_depth_x0', 'rendered_depth')
        for key_set in (disp_keys, depth_keys):
            keys_present = [k for k in patch_outputs.images if k in key_set]
            imgs = normalise_together([patch_outputs.images[k] for k in keys_present])
            for k, img_normed in zip(keys_present, imgs):
                patch_outputs.images[k] = img_normed

        for k, img in patch_outputs.images.items():
            print('key', k)
            print('img shape', img.shape)
            img = img[0]
            img = img.squeeze(dim=-1)
            img = img.detach().cpu().numpy()

            if 'disp' in k:
                img = DISPARITY_CMAP(img)
            elif 'depth' in k:
                img = DEPTH_CMAP(img)

            image_path = output_folder / f'{output_prefix}-{k}.png'
            cv2.imwrite(str(image_path), cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def normalise_together(imgs):
    max_val = max(img.max() for img in imgs)
    return [img/max_val for img in imgs]


def sample_patch_from_img(rays_d, img, img_intrinsics, patch_size: int):
    """
    Sample a patch from an image.
    :param rays_d: rays_d as returned by the get_rays() function
    :param img: Image to sample the patch from from
    :param img_intrinsics: Intrinsics for the image as a four-element sequence (fx, fy, cx, cy) in pixel coords.
    :param patch_size: Side length of the patch to make, in units of pixels
    """
    fx, fy, cx, cy = img_intrinsics
    h, w, c = img.shape

    b, num_rays, n_dim = rays_d.shape
    assert n_dim == 3
    assert b == 1

    # Go from ray directions in camera frame to pixel coordinates
    pixel_i = fx * rays_d[..., 0] / rays_d[..., 2] + cx + 0.5
    pixel_j = fy * rays_d[..., 1] / rays_d[..., 2] + cy + 0.5

    assert (pixel_i >= 0.).all()
    assert (pixel_j >= 0.).all()

    assert (pixel_i <= w).all()
    assert (pixel_j <= h).all()

    # Normalise - grid_sample wants query locations in [-1, 1].
    pixel_i = pixel_i / w
    pixel_j = pixel_j / h
    pixel_i = 2. * pixel_i - 1.
    pixel_j = 2. * pixel_j - 1.

    pixel_coords = torch.cat([pixel_i.unsqueeze(-1).unsqueeze(-1), pixel_j.unsqueeze(-1).unsqueeze(-1)], dim=-1)
    img_reshaped = img.moveaxis(-1, 0).unsqueeze(0)
    sampled = grid_sample(input=img_reshaped, grid=pixel_coords, align_corners=True)

    sampled = sampled.reshape(1, 3, patch_size, patch_size)
    sampled = sampled.moveaxis(1, -1)

    return sampled
