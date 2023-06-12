import argparse


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)

    ### training options
    parser.add_argument('--iters', type=int, default=40000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
    parser.add_argument('--num_steps', type=int, default=512, help="num steps sampled per ray")
    parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM")
    parser.add_argument('--downsample_val', type=int, default=1, help="Downsample on validation to avoid OOM")
    parser.add_argument('--downsample_train', type=int, default=1, help="Downsample on train")
    parser.add_argument('--test_mode', type=str, nargs='+', default=['test_eval'],
                        help='Test modes to run - can be test_eval, circle_path, or interpolate')
    parser.add_argument('--eval_interval', type=int, default=5)

    parser.add_argument('--max_val_imgs', type=int, default=None,
                        help="Maximum num images to run validation on - useful to avoid slowing down training")
    parser.add_argument('--num_train_poses', type=int, default=None, help="Maximum num images to train on")

    # diffusion regulariser params
    parser.add_argument('--patch_regulariser_path', help="Path to diffusion model to use as 2D patch regulariser")
    parser.add_argument('--patch_sample_downscale_factor', type=int, default=4)
    parser.add_argument('--patch_weight_start', type=float, default=1.)
    parser.add_argument('--patch_weight_finish', type=float, default=1.)
    parser.add_argument('--patch_reg_start_step', type=int, default=500)
    parser.add_argument('--patch_reg_finish_step', type=int, default=2500)
    parser.add_argument('--initial_diffusion_time', type=float, default=0.1)
    parser.add_argument('--normalise_diffusion_losses', action='store_true',
                        help='Normalise diffusion losses by depth as described in the paper')
    parser.add_argument('--apply_geom_reg_to_patches', action='store_true',
                        help='Apply mipnerf regularisation when rendering patches as well as training views')

    ### params for other regularisers
    parser.add_argument('--spread_loss_strength', type=float, default=1e-5)
    parser.add_argument('--seg_loss_strength', type=float, default=1e-6)
    parser.add_argument('--weights_sum_loss_strength', type=float, default=1e-3)
    parser.add_argument('--net_l2_loss_strength', type=float, default=1e-7)

    parser.add_argument('--reg_ramp_start_step', type=int, default=3000,
                        help="Step at which to start ramping up dynamic regularisation terms")
    parser.add_argument('--reg_ramp_finish_step', type=int, default=8000,
                        help="Step at which dynamic regularisation terms should reach their max strength")
    parser.add_argument('--use_frustum_regulariser', action='store_true',
                        help='Apply frustum regularisation as described in the paper')
    parser.add_argument('--frustum_check_patches', action='store_true')
    parser.add_argument('--frustum_regularise_patches', action='store_true')
    parser.add_argument('--frustum_reg_initial_weight', type=float, default=1.,
                        help='Initial strength of frustum regularisation, imposed once step > reg_ramp_start_step')
    parser.add_argument('--frustum_reg_final_weight', type=float, default=1e-2,
                        help='Final weight of frustum regularisation, reached (by '
                        'linear increase from frustum_reg_initial) once step == reg_ramp_finish_step')


    ### dataset options
    parser.add_argument('--mode', type=str, default='colmap', help="dataset mode, supports (colmap, blender)")
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', action='store_true',
                        help="preload all data into GPU, accelerate training but use more GPU memory")
    parser.add_argument('--bound', type=float, default=2,
            help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=0.33,
                        help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--min_near', type=float, default=0.05, help="Minimum near distance for rendering")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")

    ### experimental
    parser.add_argument('--error_map', action='store_true', help="use error map to sample rays")
    parser.add_argument('--clip_text', type=str, default='', help="text input for CLIP guidance")
    parser.add_argument('--rand_pose', type=int, default=-1, help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")
    return parser
