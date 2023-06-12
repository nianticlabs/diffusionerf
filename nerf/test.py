from pathlib import Path
from nerf.provider import NeRFDataset
import torch


def run_tests(trainer, opt, device) -> None:
    test_funcs = {
        'test_eval': test_eval,
        'circle_paths': circle_paths,
        'interpolate': interpolate,
    }

    for requested_test_mode in opt.test_mode:
        test_funcs[requested_test_mode](trainer, opt, device)


def test_eval(trainer, opt, device) -> None:
    # Evaluate accuracy against the test views for this dataset
    torch.cuda.empty_cache()
    test_loader = NeRFDataset(opt, device=device, type='test', downscale=opt.downsample_val,
                              n_test=15).dataloader()
    if opt.mode in ('blender', 'llff', 'dtu'):
        trainer.evaluate(test_loader, write_jsons=True)  # blender has gt, so evaluate it.
    else:
        trainer.test(test_loader)  # colmap doesn't have gt, so just test


def circle_paths(trainer, opt, device) -> None:
    torch.cuda.empty_cache()
    print('before loader:')
    test_loader = NeRFDataset(opt, device=device, type='circle_path', downscale=opt.downsample_val,
                              n_test=15).dataloader()
    print('after loader:')
    trainer.test(test_loader, save_path=Path(trainer.workspace) / 'results' / 'circle_paths')


def interpolate(trainer, opt, device) -> None:
    test_loader = NeRFDataset(opt, device=device, type='interpolate', downscale=opt.downsample_val,
                              n_test=5).dataloader()
    trainer.test(test_loader, save_path=Path(trainer.workspace) / 'results' / 'interpolation')
