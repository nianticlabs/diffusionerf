import math
import trimesh
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import raymarching
from .rendering_helpers import xyzs_from_z_vals, get_foreground_z_vals, perturb_z_vals, unflatten_density_outputs, \
    upsample_z_values, get_alpha_compositing_weights, flatten_density_outputs
from .sphere_geometry import SphericalSceneBounds, get_z_val_bounds
from .helpers import custom_meshgrid


def plot_pointcloud(pc, color=None):
    # pc: [N, 3]
    # color: [N, 3/4]
    print('[visualize points]', pc.shape, pc.dtype, pc.min(0), pc.max(0))
    pc = trimesh.PointCloud(pc, color)
    # axis
    axes = trimesh.creation.axis(axis_length=4)
    # sphere
    sphere = trimesh.creation.icosphere(radius=1)
    trimesh.Scene([pc, axes, sphere]).show()


class NeRFRenderer(nn.Module):
    def __init__(self,
                 bound=1,
                 cuda_ray=False,
                 density_scale=1, # scale up deltas (or sigmas), to make the density grid more sharp. larger value than 1 usually improves performance.
                 min_near=0.05,
                 density_thresh=0.01,
                 bg_radius=-1,
                 ):
        super().__init__()

        self.bound = bound
        self.cascade = 1 + math.ceil(math.log2(bound))
        self.grid_size = 128
        self.density_scale = density_scale
        self.min_near = min_near
        self.density_thresh = density_thresh
        self.bg_radius = bg_radius # radius of the background sphere.

        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        # NOTE: aabb (can be rectangular) is only used to generate points, we still rely on bound (always cubic) to calculate density grid and hashing.
        aabb_train = torch.FloatTensor([-bound, -bound, -bound, bound, bound, bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_infer)

        # extra state for cuda raymarching
        self.cuda_ray = cuda_ray
        if cuda_ray:
            # density grid
            density_grid = torch.zeros([self.cascade, self.grid_size ** 3]) # [CAS, H * H * H]
            density_bitfield = torch.zeros(self.cascade * self.grid_size ** 3 // 8, dtype=torch.uint8) # [CAS * H * H * H // 8]
            self.register_buffer('density_grid', density_grid)
            self.register_buffer('density_bitfield', density_bitfield)
            self.mean_density = 0
            self.iter_density = 0
            # step counter
            step_counter = torch.zeros(16, 2, dtype=torch.int32) # 16 is hardcoded for averaging...
            self.register_buffer('step_counter', step_counter)
            self.mean_count = 0
            self.local_step = 0

    def forward(self, x, d):
        raise NotImplementedError()

    # separated density and color query (can accelerate non-cuda-ray mode.)
    def density(self, x):
        raise NotImplementedError()

    def color(self, x, d, mask=None, **kwargs):
        raise NotImplementedError()

    def reset_extra_state(self):
        if not self.cuda_ray:
            return
        # density grid
        self.density_grid.zero_()
        self.mean_density = 0
        self.iter_density = 0
        # step counter
        self.step_counter.zero_()
        self.mean_count = 0
        self.local_step = 0

    def run(self, rays_o, rays_d, num_steps=128, upsample_steps=128, bg_color=None, perturb=False, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # bg_color: [3] in range [0, 1]
        # return: image: [B, N, 3], depth: [B, N]
        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device

        # choose aabb
        aabb = self.aabb_train if self.training else self.aabb_infer
        aabb = aabb.to(device)

        # aabb is a 6-element array (min_x, min_y, min_z, max_x, max_y, max_z).
        # This will form the boundary of our sphere.
        # Construct scene bounds
        bounds_min = aabb[:3]
        bounds_max = aabb[3:]
        sphere_radius = 0.5 * (bounds_max - bounds_min)[0]
        sphere_centre = 0.5 * (bounds_max + bounds_min)

        # The SphericalSceneBounds class allows for a separate background radius, but
        # here we do not support rendering the background, so we set it to None always.
        scene_bounds = SphericalSceneBounds(centre=sphere_centre,
                                            foreground_radius=sphere_radius,
                                            background_radius=None)

        z_max_fg, z_max_bg = get_z_val_bounds(rays_o, rays_d, scene_bounds)
        z_min_fg = torch.full_like(z_max_fg, fill_value=self.min_near)
        assert z_max_bg is None  # Foreground-background separation not supported in this code release

        z_vals = get_foreground_z_vals(z_min_fg, z_max_fg, num_steps)

        # perturb z_vals
        if perturb:
            z_vals = perturb_z_vals(z_vals)

        # impose scene bounds in case perturbation has pushed us outside of them
        z_vals = torch.minimum(z_vals, z_max_fg.unsqueeze(-1))
        z_vals = torch.maximum(z_vals, z_min_fg.unsqueeze(-1))

        # generate xyzs
        xyzs = xyzs_from_z_vals(rays_o, rays_d, z_vals)

        # query density
        density_outputs = self.density(xyzs.reshape(-1, 3))
        density_outputs = unflatten_density_outputs(density_outputs, N, num_steps)

        # Upsampling
        if upsample_steps > 0:
            new_z_vals = upsample_z_values(z_vals, density_outputs['sigma'],
                                           upsample_steps=upsample_steps, deterministic=not self.training)
            new_xyzs = xyzs_from_z_vals(rays_o, rays_d, new_z_vals)

            # only forward new points to save computation
            assert new_xyzs.isfinite().all()
            new_density_outputs = self.density(new_xyzs.reshape(-1, 3))
            new_density_outputs = unflatten_density_outputs(new_density_outputs, N, upsample_steps)

            # re-order
            z_vals = torch.cat([z_vals, new_z_vals], dim=1) # [N, T+t]
            z_vals, z_index = torch.sort(z_vals, dim=1)

            xyzs = torch.cat([xyzs, new_xyzs], dim=1) # [N, T+t, 3]
            xyzs = torch.gather(xyzs, dim=1, index=z_index.unsqueeze(-1).expand_as(xyzs))

            for k in density_outputs:
                tmp_output = torch.cat([density_outputs[k], new_density_outputs[k]], dim=1)
                density_outputs[k] = torch.gather(tmp_output, dim=1, index=z_index.unsqueeze(-1).expand_as(tmp_output))

        density_outputs_full = density_outputs

        deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T+t-1]
        assert (deltas >= 0.).all()

        augmented_deltas = torch.cat((deltas, deltas[:, -1].unsqueeze(-1)), dim=-1)
        weights = get_alpha_compositing_weights(augmented_deltas, self.density_scale * density_outputs_full['sigma'])

        dirs = rays_d.view(-1, 1, 3).expand_as(xyzs)
        density_outputs = flatten_density_outputs(density_outputs)

        mask = weights > 1e-4 # hard coded
        rgbs = self.color(xyzs.reshape(-1, 3), dirs.reshape(-1, 3), mask=mask.reshape(-1), **density_outputs)
        rgbs = rgbs.view(N, -1, 3) # [N, T+t, 3]

        # calculate weight_sum (mask)
        weights_sum = weights.sum(dim=-1) # [N]

        # Normalise depths in [0, 1]
        clamped_z_vals = z_vals.clamp(0, scene_bounds.foreground_radius)

        clamped_z_vals /= scene_bounds.foreground_radius

        depth = torch.sum(weights * z_vals, dim=-1)
        depth_normed = torch.sum(weights * clamped_z_vals, dim=-1)

        # calculate color
        image = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2) # [N, 3], in [0, 1]

        # Set white background color
        bg_color = 206. / 255.
        bg_color = torch.Tensor([bg_color, bg_color, bg_color]).to(device)
        image = image + (1. - weights_sum).unsqueeze(-1) * bg_color

        image = image.view(*prefix, 3)
        depth = depth.view(*prefix)

        sample_dist = 1. / num_steps

        z_vals_shifted = torch.cat([z_vals[..., 1:], sample_dist * torch.ones_like(z_vals[..., :1])], dim=-1)
        mid_zs = 0.5 * z_vals + 0.5 * z_vals_shifted  # [N, T]
        loss_dist_per_ray = (torch.abs(mid_zs.unsqueeze(1) - mid_zs.unsqueeze(2)) *
                                (weights.unsqueeze(1) * weights.unsqueeze(2))).sum(dim=[1, 2]) \
                    + 1/3 * ((z_vals_shifted - z_vals_shifted) * (weights ** 2)).sum(dim=1)  # [N]

        # Impose loss dist per ray relative to distance, so as not to be too harsh on very distant scene regions
        # Introduce a scaling factor to compensate for this
        loss_dist_per_ray = loss_dist_per_ray / (torch.sum(weights * z_vals, dim=-1) + 1e-6)

        # In extremis this term can drive densities to infinity as it keeps squeezing the density towards a delta
        # function (under which the dist loss is zero). So we prevent this by not applying this loss if the spread is
        # low enough.
        loss_dist_cutoff = 1e-3
        loss_dist_per_ray[loss_dist_per_ray < loss_dist_cutoff] = 0.

        loss_dist = loss_dist_per_ray.sum()

        results = {
            'depth': depth,
            'depth_normed': depth_normed,
            'image': image,
            'loss_dist': loss_dist.view(1,),
            'loss_dist_per_ray': loss_dist_per_ray.view(*prefix),
            'weights_sum': weights_sum,
            'rgbs': rgbs.view(*prefix, -1, 3),
            'xyzs': xyzs.view(*prefix, -1, 3),
            'densities': density_outputs['sigma'].view(*prefix, -1),
            'weights': weights.view(*prefix, -1),
        }

        return results

    @torch.no_grad()
    def update_extra_state(self, decay=0.95, S=128):
        # call before each epoch to update extra states.

        if not self.cuda_ray:
            return

        ### update density grid

        tmp_grid = - torch.ones_like(self.density_grid)

        # full update.
        if self.iter_density < 16:
        #if True:
            X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_grid.device).split(S)
            Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_grid.device).split(S)
            Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_grid.device).split(S)

            for xs in X:
                for ys in Y:
                    for zs in Z:

                        # construct points
                        xx, yy, zz = custom_meshgrid(xs, ys, zs)
                        coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                        indices = raymarching.morton3D(coords).long() # [N]
                        xyzs = 2 * coords.float() / (self.grid_size - 1) - 1 # [N, 3] in [-1, 1]

                        # cascading
                        for cas in range(self.cascade):
                            bound = min(2 ** cas, self.bound)
                            half_grid_size = bound / self.grid_size
                            # scale to current cascade's resolution
                            cas_xyzs = xyzs * (bound - half_grid_size)
                            # add noise in [-hgs, hgs]
                            cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                            # query density
                            sigmas = self.density(cas_xyzs)['sigma'].reshape(-1).detach()
                            sigmas *= self.density_scale
                            # assign
                            tmp_grid[cas, indices] = sigmas

        # partial update (half the computation)
        # TODO: why no need of maxpool ?
        else:
            N = self.grid_size ** 3 // 4 # H * H * H / 4
            for cas in range(self.cascade):
                # random sample some positions
                coords = torch.randint(0, self.grid_size, (N, 3), device=self.density_grid.device) # [N, 3], in [0, 128)
                indices = raymarching.morton3D(coords).long() # [N]
                # random sample occupied positions
                occ_indices = torch.nonzero(self.density_grid[cas] > 0).squeeze(-1) # [Nz]
                rand_mask = torch.randint(0, occ_indices.shape[0], [N], dtype=torch.long, device=self.density_grid.device)
                occ_indices = occ_indices[rand_mask] # [Nz] --> [N], allow for duplication
                occ_coords = raymarching.morton3D_invert(occ_indices) # [N, 3]
                # concat
                indices = torch.cat([indices, occ_indices], dim=0)
                coords = torch.cat([coords, occ_coords], dim=0)
                # same below
                xyzs = 2 * coords.float() / (self.grid_size - 1) - 1 # [N, 3] in [-1, 1]
                bound = min(2 ** cas, self.bound)
                half_grid_size = bound / self.grid_size
                # scale to current cascade's resolution
                cas_xyzs = xyzs * (bound - half_grid_size)
                # add noise in [-hgs, hgs]
                cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                # query density
                sigmas = self.density(cas_xyzs)['sigma'].reshape(-1).detach()
                sigmas *= self.density_scale
                # assign
                tmp_grid[cas, indices] = sigmas

        ## max-pool on tmp_grid for less aggressive culling [No significant improvement...]
        # invalid_mask = tmp_grid < 0
        # tmp_grid = F.max_pool3d(tmp_grid.view(self.cascade, 1, self.grid_size, self.grid_size, self.grid_size), kernel_size=3, stride=1, padding=1).view(self.cascade, -1)
        # tmp_grid[invalid_mask] = -1

        # ema update
        valid_mask = (self.density_grid >= 0) & (tmp_grid >= 0)
        self.density_grid[valid_mask] = torch.maximum(self.density_grid[valid_mask] * decay, tmp_grid[valid_mask])
        self.mean_density = torch.mean(self.density_grid.clamp(min=0)).item() # -1 non-training regions are viewed as 0 density.
        self.iter_density += 1

        # convert to bitfield
        density_thresh = min(self.mean_density, self.density_thresh)
        self.density_bitfield = raymarching.packbits(self.density_grid, density_thresh, self.density_bitfield)

        ### update step counter
        total_step = min(16, self.local_step)
        if total_step > 0:
            self.mean_count = int(self.step_counter[:total_step, 0].sum().item() / total_step)
        self.local_step = 0

        #print(f'[density grid] min={self.density_grid.min().item():.4f}, max={self.density_grid.max().item():.4f}, mean={self.mean_density:.4f}, occ_rate={(self.density_grid > 0.01).sum() / (128**3 * self.cascade):.3f} | [step counter] mean={self.mean_count}')

    def render(self, rays_o, rays_d, staged=False, max_ray_batch=2048, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: pred_rgb: [B, N, 3]

        if self.cuda_ray:
            _run = self.run_cuda
        else:
            _run = self.run

        B, N = rays_o.shape[:2]
        device = rays_o.device
        results = {}

        # never stage when cuda_ray
        if staged and not self.cuda_ray:
            depth = torch.empty((B, N), device=device)
            image = torch.empty((B, N, 3), device=device)

            for b in range(B):
                head = 0
                while head < N:
                    tail = min(head + max_ray_batch, N)
                    with torch.no_grad():
                        results_ = _run(rays_o[b:b+1, head:tail], rays_d[b:b+1, head:tail], **kwargs)
                    depth[b:b+1, head:tail] = results_['depth']
                    image[b:b+1, head:tail] = results_['image']
                    for k, v in results_.items():
                        if k not in results:
                            # Infer correct shape
                            prefix = v.shape[2:]
                            results[k] = torch.empty((B, N, *prefix), device=device)
                        results[k][b:b+1, head:tail] = v

                    head += max_ray_batch

            results['depth'] = depth
            results['image'] = image

        else:
            results = _run(rays_o, rays_d, **kwargs)

        return results
