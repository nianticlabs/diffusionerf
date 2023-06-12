import itertools
from pathlib import Path
from nerf.test import run_tests

import torch

from torch.utils.data import DataLoader

from nerf.learned_regularisation.patch_pose_generator import PatchPoseGenerator, FrustumChecker
from nerf.learned_regularisation.patch_regulariser import load_patch_diffusion_model, \
    PatchRegulariser, LLFF_DEFAULT_PSEUDO_INTRINSICS
from nerf.learned_regularisation.intrinsics import Intrinsics
from nerf.parsing import make_parser
from nerf.provider import MultiLoader, NeRFDataset, get_typical_deltas_between_poses
from nerf.network_tcnn import NeRFNetwork
from nerf.utils import *

def run(opt):

    print('main.nerf running with options', opt)

    seed_everything(opt.seed)

    print('Building foreground model')
    model = NeRFNetwork(
        bound=opt.bound,
        density_scale=1,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
    )

    criterion = torch.nn.MSELoss(reduction='none')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if opt.test:
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion,
                          fp16=True, metrics=[PSNRMeter()], use_checkpoint=opt.ckpt)
        run_tests(trainer, opt, device)

    else:
        optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)

        train_set = NeRFDataset(opt, device=device, type='trainval', downscale=opt.downsample_train)
        train_loader = train_set.dataloader()

        # Make it so 1 epoch = multiple actual epochs
        # This is a fudge for performance reasons, since torch-ngp does some costly stuff between each epoch.
        # With few train views, an epoch is otherwise so short that most of the time is spent doing between-epoch stuff
        print('train loader len', len(train_loader))
        print('train loader images', train_set.image_filenames)
        min_steps_per_epoch = 200
        if len(train_loader) < min_steps_per_epoch:
            train_loader = MultiLoader(loader=train_loader, num_repeats=int(round(200 / len(train_loader))))


        # decay to 0.1 * init_lr at last iter step
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

        fx, fy, cx, cy = train_set.intrinsics
        intrinsics = Intrinsics(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=train_set.W,
            height=train_set.H,
        )

        frustum_checker = FrustumChecker(fov_x_rads=train_set.fov_x, fov_y_rads=train_set.fov_y)
        frustum_regulariser = FrustumRegulariser(
            intrinsics=intrinsics,
            reg_strength=1e-5,  # NB in the trainer this gets multiplied by the strength params passed in via the args
            min_near=opt.min_near,
            poses=[torch.Tensor(pose).to(device) for pose in train_set.poses],
        )

        if opt.patch_regulariser_path:
            patch_diffusion_model = load_patch_diffusion_model(Path(opt.patch_regulariser_path))
            delta_pos, delta_ori = get_typical_deltas_between_poses(train_set.poses)
            print('Typical deltas (pos, ori):', delta_pos * 0.5, delta_ori)
            pose_generator = PatchPoseGenerator(poses=train_set.poses,
                                                spatial_perturbation_magnitude=0.2,
                                                angular_perturbation_magnitude_rads=0.2 * np.pi,
                                                no_perturb_prob=0.,
                                                frustum_checker=frustum_checker if opt.frustum_check_patches else None)
            pseudo_intrinsics = LLFF_DEFAULT_PSEUDO_INTRINSICS
            print('Using patch full image pseudo intrinsics', pseudo_intrinsics)
            patch_regulariser = PatchRegulariser(pose_generator=pose_generator,
                                                 patch_diffusion_model=patch_diffusion_model,
                                                 full_image_intrinsics=pseudo_intrinsics,
                                                 device=device,
                                                 planar_depths=True,
                                                 frustum_regulariser=frustum_regulariser if opt.frustum_regularise_patches else None,
                                                 sample_downscale_factor=opt.patch_sample_downscale_factor,
                                                 uniform_in_depth_space=opt.normalise_diffusion_losses)

        else:
            patch_regulariser = None

        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer,
                          criterion=criterion, ema_decay=0.95, fp16=True, lr_scheduler=scheduler,
                          scheduler_update_every_step=True, metrics=[PSNRMeter()], use_checkpoint=opt.ckpt,
                          eval_interval=opt.eval_interval,
                          patch_regulariser=patch_regulariser,
                          frustum_regulariser=frustum_regulariser if opt.use_frustum_regulariser else None)

        valid_set = NeRFDataset(opt, device=device, type='val', downscale=opt.downsample_val)
        if opt.max_val_imgs is not None:
            valid_set.poses = valid_set.poses[:opt.max_val_imgs]  # limit num poses for val

        # Also validate with some test poses
        test_valid_set = NeRFDataset(opt, device=device, type='test', downscale=opt.downsample_val)
        test_valid_set.poses = test_valid_set.poses[:opt.max_val_imgs]

        valid_loader = ConcatLoader([valid_set.dataloader(), test_valid_set.dataloader()])

        max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
        print('Training to max epoch', max_epoch)
        trainer.train(train_loader, valid_loader, max_epoch)

    return trainer


class ConcatLoader:
    def __init__(self, loaders: list[DataLoader]):
        self._loaders = loaders

    def __len__(self):
        return sum([len(loader) for loader in self._loaders])

    def __iter__(self):
        return itertools.chain(*self._loaders)

    @property
    def batch_size(self):
        return self._loaders[0].batch_size


if __name__ == '__main__':
    parser = make_parser()
    opt = parser.parse_args()
    run(opt)
