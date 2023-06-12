import copy
import itertools
from pathlib import Path

from torch.utils.data import DataLoader

from main_nerf import run
from nerf.llff import get_llff_scene_paths
from nerf.parsing import make_parser


# These are in 'scene coordinates', i.e. BEFORE multiplying the scale correction factors below.
# They are set so that the relevant parts of a scene fit within the bounds of the NeRF
LLFF_DEFAULT_SCENE_BOUNDS = {
    'horns': 10.,
    'orchids': 9.,
    'trex': 12.,
    'leaves': 1500.,
    'flower': 300.,
    'fern': 15.,
    'fortress': 10.,
    'room': 5.,
}


# We multiply distances in 'scene coordinates' by these to get approximately metric scales.
# (i.e. we multiply the camera centres by these.)
# They are rough values chosen by inspecting the SfM point clouds to estimate an
#   approximate metric scale.
ESTIMATED_SCALE_CORRECTION_FACTORS = {
    'fortress': 0.15,
    'room': 1.5,
    'leaves': 0.007,
    'trex': 1.5,
    'fern': 1.,
    'horns': 1.5,
    'orchids': 0.3,
    'flower': 0.01
}


def make_opt_for_scene(opts, scene_path, num_poses):
    opts_tmp = copy.deepcopy(opts)
    opts_tmp.path = scene_path
    opts_tmp.workspace = Path(opts.workspace) / f'{num_poses}_poses' / scene_path.name
    scale_correction_factor = ESTIMATED_SCALE_CORRECTION_FACTORS[scene_path.name]
    opts_tmp.bound = LLFF_DEFAULT_SCENE_BOUNDS[scene_path.name] * scale_correction_factor
    opts_tmp.scale = scale_correction_factor
    opts_tmp.num_train_poses = num_poses
    print('Using bound', opts_tmp.bound, 'and scale', opts_tmp.scale)

    if opts.normalise_length_scales_to is not None:
        print('Rescaling bound and scale from current values of ', opts_tmp.bound, 'and', opts_tmp.scale)
        scale_factor = opts.normalise_length_scales_to / opts_tmp.bound
        opts_tmp.scale *= scale_factor
        opts_tmp.bound *= scale_factor
    elif opts.normalise_length_scales_to_at_least is not None:
        if opts_tmp.bound < opts.normalise_length_scales_to_at_least:
            print('Rescaling bound and scale from current values of ', opts_tmp.bound, 'and', opts_tmp.scale)
            scale_factor = opts.normalise_length_scales_to_at_least / opts_tmp.bound
            opts_tmp.scale *= scale_factor
            opts_tmp.bound *= scale_factor
    print('Final bound and scale:', opts_tmp.bound, opts_tmp.scale)

    opts_tmp.min_near = 0.2
    return opts_tmp


def make_eval_parser():
    parser = make_parser()
    parser.add_argument('--only_run_on', type=str, nargs='+')
    parser.add_argument('--num_train', type=int, nargs='+', default=[3])
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--normalise_length_scales_to', type=float, default=None)
    parser.add_argument('--normalise_length_scales_to_at_least', type=float, default=None)
    return parser


def main(opts):
    scene_paths = get_llff_scene_paths(Path(opts.path))
    print('Discovered LLFF scenes:')
    print('\n'.join(str(p) for p in scene_paths))

    num_train_poses_list = [n for n in opts.num_train]
    print('Running on num poses:', num_train_poses_list)
    for n_train in num_train_poses_list:
        (Path(opts.workspace) / f'{n_train}_poses').mkdir(parents=True, exist_ok=True)

        for scene_path in scene_paths:
            if opts.only_run_on is not None and scene_path.name not in opts.only_run_on:
                print(f'Skipping scene - not included in specified scenes {opts.only_run_on}')
                continue

            print('Scene', scene_path)
            opts_tmp = make_opt_for_scene(opts, scene_path, n_train)
            if opts_tmp.workspace.exists() and not opts.force:
                print(f'Skipping - {opts_tmp.workspace} already exists')
            else:
                print('Fitting to scene', scene_path)
                run(opts_tmp)

            print('Evaluating on scene', scene_path)
            requested_test_modes = opts.test_mode[:]
            for test_mode in requested_test_modes:
                print('TEST MODE:', test_mode)
                opts_tmp = make_opt_for_scene(opts, scene_path, n_train)
                opts_tmp.test = True
                opts_tmp.test_mode = [test_mode]
                run(opts_tmp)


if __name__ == '__main__':
    opts = make_eval_parser().parse_args()
    main(opts)
