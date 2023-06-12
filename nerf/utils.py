import os
from typing import Optional

import matplotlib.cm
from pathlib import Path
import glob
import tqdm
import random
import tensorboardX

import numpy as np

import time

import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

import trimesh
import mcubes
from rich.console import Console
from torch_ema import ExponentialMovingAverage

from nerf.helpers import custom_meshgrid
from nerf.learned_regularisation.patch_pose_generator import FrustumRegulariser
from nerf.metrics import calculate_all_metrics, write_metrics_to_disk


@torch.jit.script
def linear_to_srgb(x):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x ** 0.41666 - 0.055)


@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, error_map=None):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = custom_meshgrid(torch.linspace(0, W - 1, W, device=device), torch.linspace(0, H - 1, H, device=device))
    i = i.t().reshape([1, H*W]).expand([B, H*W]) + 0.5
    j = j.t().reshape([1, H*W]).expand([B, H*W]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H*W)

        if error_map is None:
            inds = torch.randint(0, H*W, size=[N], device=device) # may duplicate
            inds = inds.expand([B, N])
        else:

            # weighted sample on a low-reso grid
            inds_coarse = torch.multinomial(error_map.to(device), N, replacement=False) # [B, N], but in [0, 128*128)

            # map to the original resolution with random perturb.
            inds_x, inds_y = inds_coarse // 128, inds_coarse % 128 # `//` will throw a warning in torch 1.10... anyway.
            sx, sy = H / 128, W / 128
            inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
            inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
            inds = inds_x * W + inds_y

            results['inds_coarse'] = inds_coarse # need this when updating error_map

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

    else:
        inds = torch.arange(H*W, device=device).expand([B, H*W])

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2) # (B, N, 3)

    rays_o = poses[..., :3, 3] # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d
    results['rays_d_cam'] = directions
    results['rays_d_cam_z'] = directions[..., -1] # Useful because it's the conversion factor to go from spherical to planar depths
    results['inds'] = inds

    return results


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


def torch_vis_2d(x, renormalize=False):
    # x: [3, H, W] or [1, H, W] or [H, W]
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    
    if isinstance(x, torch.Tensor):
        if len(x.shape) == 3:
            x = x.permute(1,2,0).squeeze()
        x = x.detach().cpu().numpy()
        
    print(f'[torch_vis_2d] {x.shape}, {x.dtype}, {x.min()} ~ {x.max()}')
    
    x = x.astype(np.float32)
    
    # renormalize
    if renormalize:
        x = (x - x.min(axis=0, keepdims=True)) / (x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True) + 1e-8)

    plt.imshow(x)
    plt.show()


def extract_fields(bound_min, bound_max, resolution, query_func, S=128):

    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(S)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(S)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(S)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in tqdm.tqdm(enumerate(X)):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [S, 3]
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy() # [S, 1] --> [x, y, z]
                    u[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    #print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)

    #print(u.shape, u.max(), u.min(), np.percentile(u, 50))
    
    vertices, triangles = mcubes.marching_cubes(u, threshold)

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


class PSNRMeter:
    def __init__(self):
        self.V = 0
        self.N = 0

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [B, N, 3] or [B, H, W, 3], range[0, 1]
          
        # simplified since max_pixel_value is 1 here.
        psnr = -10 * np.log10(np.mean((preds - truths) ** 2))
        
        self.V += psnr
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "PSNR"), self.measure(), global_step)

    def report(self):
        return f'PSNR = {self.measure():.6f}'


class Trainer(object):
    def __init__(self, 
                 name, # name of this experiment
                 opt, # extra conf
                 model, # network 
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 max_keep_ckpt=2, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metric
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 patch_regulariser=None,
                 frustum_regulariser: Optional[FrustumRegulariser] = None,
                 ):
        
        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.patch_regulariser = patch_regulariser
        self.frustum_regulariser = frustum_regulariser
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()

        self.spread_loss_strength = opt.spread_loss_strength
        self.seg_loss_strength = opt.seg_loss_strength
        self.weights_sum_loss_strength = opt.weights_sum_loss_strength
        self.net_l2_loss_strength = opt.net_l2_loss_strength
        #self.latent_reg_strength = opt.latent_reg_strength

        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4) # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 1
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
            }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)        
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)
            
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

        # clip loss prepare
        if opt.rand_pose >= 0: # =0 means only using CLIP loss, >0 means a hybrid mode.
            from nerf.clip_utils import CLIPLoss
            self.clip_loss = CLIPLoss(self.device)
            self.clip_loss.prepare_text([self.opt.clip_text]) # only support one text prompt now...


    def __del__(self):
        if self.log_ptr: 
            self.log_ptr.close()


    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute: 
                #print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr: 
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file

    ### ------------------------------	

    def train_step(self, data):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]

        # if there is no gt image, we train with CLIP loss.
        if 'images' not in data:

            B, N = rays_o.shape[:2]
            H, W = data['H'], data['W']

            # currently fix white bg, MUST force all rays!
            outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=None, perturb=True, force_all_rays=True, **vars(self.opt))
            pred_rgb = outputs['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()

            # [debug] uncomment to plot the images used in train_step
            #torch_vis_2d(pred_rgb[0])

            loss = self.clip_loss(pred_rgb)
            
            return pred_rgb, None, loss

        images = data['images'] # [B, N, 3/4]

        B, N, C = images.shape

        if self.opt.color_space == 'linear':
            images[..., :3] = srgb_to_linear(images[..., :3])

        if C == 3 or self.model.bg_radius > 0:
            bg_color = 1
        # train with random background color if not using a bg model and has alpha channel.
        else:
            #bg_color = torch.ones(3, device=self.device) # [3], fixed white background
            #bg_color = torch.rand(3, device=self.device) # [3], frame-wise random.
            bg_color = torch.rand_like(images[..., :3]) # [N, 3], pixel-wise random.

        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images

        outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=bg_color, perturb=True, force_all_rays=False, **vars(self.opt))
    
        pred_rgb = outputs['image']

        loss = self.criterion(pred_rgb, gt_rgb).mean(-1) # [B, N, 3] --> [B, N]

        dynamic_reg_start_step = self.opt.reg_ramp_start_step
        dynamic_reg_max_strength_step = self.opt.reg_ramp_finish_step

        def get_linear_dynamic_reg_modifier():
            # Linear scheme
            if self.global_step > dynamic_reg_max_strength_step:
                dynamic_reg_modifier = 1.
            elif self.global_step > dynamic_reg_start_step:
                dynamic_reg_modifier = (self.global_step - dynamic_reg_start_step) / (
                        dynamic_reg_max_strength_step - dynamic_reg_start_step)
            else:
                dynamic_reg_modifier = 0.
            return dynamic_reg_modifier

        dynamic_reg_modifier = get_linear_dynamic_reg_modifier()

        spread_loss_weight = self.spread_loss_strength * dynamic_reg_modifier

        sparsity_loss_per_ray = outputs.get('sparsity_loss_per_ray', None)
        if sparsity_loss_per_ray is not None:
            loss += self.opt.sparsity_loss_weight * sparsity_loss_per_ray.sum()

        def get_frustum_reg_str():
            # Piecewise schedule
            frustum_reg_weight = self.opt.frustum_reg_initial_weight if self.global_step < 100 \
                else self.opt.frustum_reg_final_weight

            return frustum_reg_weight


        if self.patch_regulariser is not None:
            # t schedule
            initial_diffusion_time = self.opt.initial_diffusion_time
            patch_reg_start_step = self.opt.patch_reg_start_step
            patch_reg_finish_step = self.opt.patch_reg_finish_step
            weight_start = self.opt.patch_weight_start
            weight_finish = self.opt.patch_weight_finish

            lambda_t = (self.global_step - patch_reg_start_step) / (patch_reg_finish_step - patch_reg_start_step)
            lambda_t = np.clip(lambda_t, 0., 1.)
            weight = weight_start + (weight_finish - weight_start) * lambda_t

            if self.global_step > patch_reg_start_step:
                if self.global_step > patch_reg_finish_step:
                    time = 0.
                elif self.global_step > patch_reg_start_step:
                    time = initial_diffusion_time * (1. - lambda_t)
                else:
                    raise RuntimeError('Internal error')
                p_sample_patch = 0.25
                if random.random() >= p_sample_patch:
                    patch_outputs = self.patch_regulariser.get_diffusion_loss_with_rendered_patch(model=self.model,
                                                                                                  time=time)
                else:
                    patch_outputs = self.patch_regulariser.get_diffusion_loss_with_sampled_patch(
                        model=self.model, time=time, image=data['images_full'][0], image_intrinsics=data['intrinsics'],
                        pose=data['pose_c2w'][0]
                    )
                loss += weight * patch_outputs.loss

                # Geometric reg
                if self.opt.apply_geom_reg_to_patches:
                    loss += spread_loss_weight * patch_outputs.render_outputs['loss_dist']

                # Frustum reg
                if self.patch_regulariser.frustum_regulariser is not None:
                    xyzs_flat = patch_outputs.render_outputs['xyzs'].reshape(-1, 3)
                    weights_flat = patch_outputs.render_outputs['weights'].reshape(-1)

                    patch_frustum_reg_weight = get_frustum_reg_str()
                    patch_frustum_loss = patch_frustum_reg_weight * self.patch_regulariser.frustum_regulariser(
                        xyzs=xyzs_flat, weights=weights_flat, frustum_count_thresh=1,
                    )
                    print('Patch frustum loss', patch_frustum_loss)
                    loss += patch_frustum_loss

        # special case for CCNeRF's rank-residual training
        if len(loss.shape) == 3: # [K, B, N]
            loss = loss.mean(0)

        # Print some debug information about the mipnerf-360 distortion loss
        show_dist_loss_debug_info = True
        if show_dist_loss_debug_info:
            loss_dist_per_ray = outputs['loss_dist_per_ray']
            print('Loss dist min, mean, median, max:',
                  loss_dist_per_ray.min(),
                  loss_dist_per_ray.mean(),
                  loss_dist_per_ray.median(),
                  loss_dist_per_ray.max())

        loss += spread_loss_weight * outputs['loss_dist']

        # update error_map
        if self.error_map is not None:
            index = data['index'] # [B]
            inds = data['inds_coarse'] # [B, N]

            # take out, this is an advanced indexing and the copy is unavoidable.
            error_map = self.error_map[index] # [B, H * W]

            # [debug] uncomment to save and visualize error map
            # if self.global_step % 1001 == 0:
            #     tmp = error_map[0].view(128, 128).cpu().numpy()
            #     print(f'[write error map] {tmp.shape} {tmp.min()} ~ {tmp.max()}')
            #     tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
            #     cv2.imwrite(os.path.join(self.workspace, f'{self.global_step}.jpg'), (tmp * 255).astype(np.uint8))

            error = loss.detach().to(error_map.device) # [B, N], already in [0, 1]
            
            # ema update
            ema_error = 0.1 * error_map.gather(1, inds) + 0.9 * error
            error_map.scatter_(1, inds, ema_error)

            # put back
            self.error_map[index] = error_map

        loss = loss.mean()

        if self.frustum_regulariser is not None:
            xyzs_flat = outputs['xyzs'].reshape(-1, 3)
            weights_flat = outputs['weights'].reshape(-1)
            frustum_reg_weight = get_frustum_reg_str()
            loss += frustum_reg_weight * self.frustum_regulariser(
                 xyzs=xyzs_flat, weights=weights_flat
            )

        # Weights sum loss: each ray should terminate within the scene, i.e. the weights should sum
        # to one. If they sum to less than one, then the ray is marching out of the scene without
        # having terminated.
        weights_sum = outputs['weights_sum']
        weights_sum_loss = ((1. - weights_sum)**2).sum()
        loss += self.weights_sum_loss_strength * weights_sum_loss

        # L2 regularisation on the density network. This prevents us from getting infinite densities
        # which otherwise can occasionally happen with the mipnerf distortion loss enabled.
        # May get either a torch tensor or a list of torch tensors, dependong on the model:
        sigma_params = self.model.sigma_params
        if isinstance(sigma_params, torch.Tensor):
            sigma_params = [sigma_params]

        for sigma_param_tensor in sigma_params:
            loss += self.net_l2_loss_strength * (sigma_param_tensor ** 2).sum()

        return pred_rgb, gt_rgb, loss, {}

    def eval_step(self, data):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        images = data['images'] # [B, H, W, 3/4]
        B, H, W, C = images.shape

        if self.opt.color_space == 'linear':
            images[..., :3] = srgb_to_linear(images[..., :3])

        # eval with fixed background color
        bg_color = 1
        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images

        extra_vars = vars(self.opt)
        extra_vars = {k: v for k, v in extra_vars.items()}
        extra_vars['max_ray_batch'] = extra_vars['max_ray_batch'] // 2
        outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=False, **extra_vars)

        pred_rgb = outputs['image'].reshape(B, H, W, 3)

        pred_depth = (outputs['depth'] * data['rays_d_cam_z']).reshape(-1, H, W)

        outputs['loss_dist_per_ray'] = outputs['loss_dist_per_ray'].reshape(B, H, W)
        outputs['weights_sum'] = outputs['weights_sum'].reshape(B, H, W)
        if 'segmentation' in outputs:
            outputs['segmentation'] = outputs['segmentation'].reshape(B, H, W, self.model.num_semantic_channels)

        loss = self.criterion(pred_rgb, gt_rgb).mean()

        return pred_rgb, pred_depth, gt_rgb, loss, outputs

    # moved out bg_color and perturb for more flexible control...
    def test_step(self, data, bg_color=None, perturb=False):  

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        H, W = data['H'], data['W']

        if bg_color is not None:
            bg_color = bg_color.to(self.device)

        extra_vars = vars(self.opt)
        extra_vars = {k: v for k, v in extra_vars.items()}
        extra_vars['max_ray_batch'] = extra_vars['max_ray_batch'] // 2
        outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=perturb, **extra_vars)

        pred_rgb = outputs['image'].reshape(-1, H, W, 3)
        pred_depth = (outputs['depth'] * data['rays_d_cam_z']).reshape(-1, H, W)
        outputs['loss_dist_per_ray'] = outputs['loss_dist_per_ray'].reshape(-1, H, W)
        outputs['weights_sum'] = outputs['weights_sum'].reshape(-1, H, W)
        if 'segmentation' in outputs:
            outputs['segmentation'] = outputs['segmentation'].reshape(-1, H, W, self.model.num_semantic_channels)

        return pred_rgb, pred_depth, outputs


    def save_mesh(self, save_path=None, resolution=256, threshold=10):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'meshes', f'{self.name}_{self.epoch}.ply')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        def query_func(pts):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    sigma = self.model.density(pts.to(self.device))['sigma']
            return sigma

        vertices, triangles = extract_geometry(self.model.aabb_infer[:3], self.model.aabb_infer[3:], resolution=resolution, threshold=threshold, query_func=query_func)

        mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
        mesh.export(save_path)

        self.log(f"==> Finished saving mesh.")

    ### ------------------------------

    def train(self, train_loader, valid_loader, max_epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        # get a ref to error_map
        self.error_map = train_loader._data.error_map
        
        for epoch in range(self.epoch, max_epochs + 1):
            self.epoch = epoch

            self.train_one_epoch(train_loader)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None, write_jsons=False):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name, write_jsons=write_jsons)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, save_path=None, name=None):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)
        
        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()
        with torch.no_grad():

            # update grid
            if self.model.cuda_ray:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()

            for i, data in enumerate(loader):
                
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, outputs = self.test_step(data)
                
                path = os.path.join(save_path, f'{name}_{i:04d}.png')
                path_disp = os.path.join(save_path, f'{name}_{i:04d}_disp.png')
                path_depth = os.path.join(save_path, f'{name}_{i:04d}_depth.png')
                path_loss_dist = os.path.join(save_path, f'{name}_{i:04d}_loss_dist.png')
                path_weights_sum = os.path.join(save_path, f'{name}_{i:04d}_weights_sum.png')

                #self.log(f"[INFO] saving test image to {path}")

                if self.opt.color_space == 'linear':
                    preds = linear_to_srgb(preds)

                pred = preds[0].detach().cpu().numpy()
                pred_depth = preds_depth[0].detach().cpu().numpy()
                loss_dist = outputs['loss_dist_per_ray'][0].detach().cpu().numpy()
                weights_sum = outputs['weights_sum'][0].detach().cpu().numpy()

                cmap = matplotlib.cm.get_cmap('turbo')

                # Write disparity (inverse depth)
                # Clip so that visible range of disparities is finite:
                min_depth = 0.1
                disp = 1. / pred_depth
                max_disp = 1. / min_depth

                disp_cmapped = cmap(disp / max_disp)
                cv2.imwrite(path_disp, cv2.cvtColor((disp_cmapped * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

                # Write depth
                max_depth = 2. * self.model.bound
                depth_cmapped = cmap(pred_depth / max_depth)
                cv2.imwrite(path_depth, cv2.cvtColor((depth_cmapped * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

                loss_dist_cutoff = 0.1
                loss_dist_vis = np.minimum(loss_dist, loss_dist_cutoff) / loss_dist_cutoff

                cv2.imwrite(path, cv2.cvtColor((pred * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                cv2.imwrite(path_weights_sum, (weights_sum * 255).astype(np.uint8))
                cv2.imwrite(path_loss_dist, (loss_dist_vis * 255).astype(np.uint8))

                pbar.update(loader.batch_size)

        self.log(f"==> Finished Test.")
    
    def train_one_epoch(self, loader):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)
        
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:

            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % 16 == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()
                    
            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss, extra_outputs = self.train_step(data)

            self.scaler.scale(loss).backward()
            print('Scaler scale', self.scaler.get_scale())

            self.scaler.step(self.optimizer)

            self.scaler.update()


            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)
                        
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f} global step={self.global_step}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")


    def evaluate_one_epoch(self, loader, name=None, write_jsons=False):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        metrics = []
        with torch.no_grad():
            self.local_step = 0

            # update grid
            if self.model.cuda_ray:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()

            for data in loader:
                self.local_step += 1

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, truths, loss, outputs = self.eval_step(data)

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size
                    
                    preds_list = [torch.zeros_like(preds).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_depth_list = [torch.zeros_like(preds_depth).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)

                    truths_list = [torch.zeros_like(truths).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(truths_list, truths)
                    truths = torch.cat(truths_list, dim=0)
                
                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:

                    if write_jsons:
                        print('Calculating metrics...')
                        metric_results = calculate_all_metrics(
                            rendered=preds.detach().cpu().numpy(),
                            target=truths.detach().cpu().numpy(),
                            scene_name=Path(self.workspace).name,
                            image_name=data['image_filename'],
                        )
                        print('Computed metrics', metric_results)
                        metrics.extend(metric_results)

                    for metric in self.metrics:
                        metric.update(preds, truths)

                    # save image
                    image_name = Path(data['image_filename']).stem
                    save_path = os.path.join(self.workspace, 'validation', f'{name}_{image_name}.png')
                    save_path_disp = os.path.join(self.workspace, 'validation', f'{name}_{image_name}_disp.png')
                    save_path_depth = os.path.join(self.workspace, 'validation', f'{name}_{image_name}_depth.png')
                    path_loss_dist = os.path.join(self.workspace, 'validation', f'{name}_{image_name}_loss_dist.png')
                    path_weights_sum = os.path.join(self.workspace, 'validation', f'{name}_{image_name}_weights_sum.png')
                    save_path_gt = os.path.join(self.workspace, 'validation', f'{name}_{image_name}_gt.png')

                    #self.log(f"==> Saving validation image to {save_path}")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    if self.opt.color_space == 'linear':
                        preds = linear_to_srgb(preds)

                    pred = preds[0].detach().cpu().numpy()
                    truth_rgb = truths[0].detach().cpu().numpy()
                    pred_depth = preds_depth[0].detach().cpu().numpy()
                    loss_dist = outputs['loss_dist_per_ray'][0].detach().cpu().numpy()
                    weights_sum = outputs['weights_sum'][0].detach().cpu().numpy()

                    cmap = matplotlib.cm.get_cmap('turbo')

                    # Write disp
                    min_depth = 0.1
                    disp = 1. / pred_depth
                    max_disp = 1. / min_depth

                    disp_cmapped = cmap(disp / max_disp)
                    cv2.imwrite(save_path_disp, cv2.cvtColor((disp_cmapped * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

                    # Write depth
                    max_depth = 2. * self.model.bound
                    depth_cmapped = cmap(pred_depth / max_depth)
                    cv2.imwrite(save_path_depth, cv2.cvtColor((depth_cmapped * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


                    loss_dist_cutoff = 0.1
                    loss_dist_vis = np.minimum(loss_dist, loss_dist_cutoff) / loss_dist_cutoff

                    cv2.imwrite(save_path, cv2.cvtColor((pred * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                    cv2.imwrite(save_path_gt, cv2.cvtColor((truth_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                    cv2.imwrite(path_weights_sum, (weights_sum * 255).astype(np.uint8))
                    cv2.imwrite(path_loss_dist, (loss_dist_vis * 255).astype(np.uint8))

                    mask_folder = Path(self.workspace) / 'mask-results'
                    mask_folder.mkdir(exist_ok=True)

                    if self.use_tensorboardX:
                        # Have to swap around the axes because img is HWC and tensorboard wants CHW
                        self.writer.add_image('val/img', np.transpose(pred, axes=[2, 0, 1]), global_step=self.global_step)

                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                    pbar.update(loader.batch_size)

                # Free up some memory
                del preds
                del preds_depth
                del truths
                del loss
                del outputs


        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        # If we are using diffusion model patch regularisation, then write out a few sample patches to assist debugging
        if self.patch_regulariser is not None:
            make_test_patches = True
            print('Making debug sample patches...')
            if make_test_patches:
                for i_patch in range(3):
                    outputs = self.patch_regulariser.get_diffusion_loss_with_rendered_patch(model=self.model, time=0.05)
                    print('Disp range', outputs.disp_patch.min(), outputs.disp_patch.max())
                    print('Depth range', outputs.depth_patch.min(), outputs.depth_patch.max())
                    for time in [0.0, 0.1]:
                        outputs_at_time = self.patch_regulariser.get_loss_for_patch(rgb_patch=outputs.rgb_patch,
                                                                                    depth_patch=outputs.depth_patch,
                                                                                    time=time,
                                                                                    render_outputs=outputs.render_outputs)
                        self.patch_regulariser.dump_debug_visualisations(
                            output_folder=Path(self.workspace) / 'validation',
                            output_prefix=f'patch-{name}-rendered-{i_patch}-time-{time}',
                            patch_outputs=outputs_at_time
                        )
                        noise_mag = torch.linalg.norm(outputs_at_time.images["pred_disp_noise"])
                        print(f'Noise mag at t={time} is {noise_mag}')

                for i_patch in range(3):
                    outputs = self.patch_regulariser.get_diffusion_loss_with_sampled_patch(
                        model=self.model, time=0.05, image=data['images_full'][0], image_intrinsics=data['intrinsics'],
                        pose=data['pose_c2w'][0]
                    )
                    print('Disp range', outputs.disp_patch.min(), outputs.disp_patch.max())
                    print('Depth range', outputs.depth_patch.min(), outputs.depth_patch.max())
                    for time in [0.0, 0.1]:
                        outputs_at_time = self.patch_regulariser.get_loss_for_patch(rgb_patch=outputs.rgb_patch,
                                                                                    depth_patch=outputs.depth_patch,
                                                                                    time=time,
                                                                                    render_outputs=outputs.render_outputs)
                        self.patch_regulariser.dump_debug_visualisations(
                            output_folder=Path(self.workspace) / 'validation',
                            output_prefix=f'patch-{name}-sampled-{i_patch}-time-{time}',
                            patch_outputs=outputs_at_time
                        )
                        noise_mag = torch.linalg.norm(outputs_at_time.images["pred_disp_noise"])
                        print(f'Noise mag at t={time} is {noise_mag}')



        if self.local_rank == 0:
            if write_jsons:
                write_metrics_to_disk(metrics=metrics,
                                      path=Path(self.workspace) / 'metrics.json')

            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(result if self.best_mode == 'min' else - result) # if max mode, use -result
            else:
                self.stats["results"].append(average_loss) # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, name=None, full=False, best=False, remove_old=True):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        if self.model.cuda_ray:
            state['mean_count'] = self.model.mean_count
            state['mean_density'] = self.model.mean_density

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()
        
        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{self.ckpt_path}/{name}.pth"

            if remove_old:
                self.stats["checkpoints"].append(file_path)

                if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                    old_ckpt = self.stats["checkpoints"].pop(0)
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)

            torch.save(state, file_path)

        else:    
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results 
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()
                    
                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")
            
    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/{self.name}_ep*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        
        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")   

        if self.ema is not None and 'ema' in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict['ema'])

        if self.model.cuda_ray:
            if 'mean_count' in checkpoint_dict:
                self.model.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']
        
        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")
        
        if self.optimizer and  'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")
        
        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")
        
        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")

