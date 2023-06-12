import dataclasses
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import torch
import lpips
import skimage

from nerf.learned_regularisation.diffusion.denoising_diffusion_pytorch import normalize_to_neg_one_to_one

loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores
loss_fn_vgg = lpips.LPIPS(net='vgg')  # closer to "traditional" perceptual loss, when used for optimization



@dataclass
class MetricResult:
    image_name: str
    scene: str
    metric_name: str
    result: float


def lpips(rendered, target):
    # Images needs to be normalised from -1 to 1 for lpips
    rendered = normalize_to_neg_one_to_one(rendered)
    target = normalize_to_neg_one_to_one(target)
    B, H, W, C = target.shape
    assert target.shape == rendered.shape

    rendered = np.moveaxis(rendered, -1, 1)
    target = np.moveaxis(target, -1, 1)

    d = loss_fn_alex(torch.tensor(rendered, dtype=torch.float32), torch.tensor(target, dtype=torch.float32))
    return float(d.squeeze())


def ssim(rendered, target):
    rendered = normalize_to_neg_one_to_one(rendered)[0]
    target = normalize_to_neg_one_to_one(target)[0]
    return float(skimage.metrics.structural_similarity(rendered, target, channel_axis=-1))


def psnr(rendered, target):
    # simplified since max_pixel_value is 1 here.
    return float(-10 * np.log10(np.mean((rendered - target) ** 2)))


METRICS_LOOKUP = {
    'lpips': lpips,
    'ssim': ssim,
    'psnr': psnr
}


def calculate_all_metrics(rendered: np.ndarray, target: np.ndarray, scene_name: str,
                          image_name: str) -> list[MetricResult]:
    print(f'Computing metrics; image shapes are {rendered.shape} and {target.shape}')
    return [
        MetricResult(
            metric_name=metric_name,
            scene=scene_name,
            image_name=image_name,
            result=metric_fn(rendered, target),
        )
        for metric_name, metric_fn in METRICS_LOOKUP.items()
    ]


def write_metrics_to_disk(metrics: list[MetricResult], path: Path) -> None:
    metrics_json = [dataclasses.asdict(metric_result) for metric_result in metrics]
    with open(path, 'w') as fp:
        json.dump(metrics_json, fp, indent=4)
