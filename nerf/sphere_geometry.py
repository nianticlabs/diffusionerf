import torch
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class SphericalSceneBounds:
    # NB this class was introduced for experiments with foreground-background decomposition,
    # but this code release does not include this functionality.
    centre: Tuple[float, float, float]
    foreground_radius: float
    background_radius: Optional[float]

    def __post_init__(self):
        if self.background_radius is not None:
            assert self.foreground_radius < self.background_radius

    @property
    def outer_radius(self):
        return self.background_radius if self.background_radius is not None else self.foreground_radius


def get_z_val_bounds(rays_o, rays_d, scene_bounds: SphericalSceneBounds):
    foreground_bounds = sphere_intersection(rays_o, rays_d, scene_bounds.foreground_radius, positive_only=True)
    foreground_bounds = foreground_bounds[..., 0]
    if scene_bounds.background_radius is not None:
        background_bounds = sphere_intersection(rays_o, rays_d, scene_bounds.background_radius, positive_only=True)
        background_bounds = background_bounds[..., 0]
    else:
        background_bounds = None
    return foreground_bounds, background_bounds


def sphere_intersection(rays_o, rays_d, sphere_radius, positive_only: bool = False):
    # rays_o: [N, 3]
    # rays_d: [N, 3]
    # sphere_radius: [M]. SHOULD BE SORTED ALREADY.
    # returns: [N, 2M] array of distances along n-th ray to intersect with m-th sphere
    # WARNING: returned values will be nan where there is no intersection.

    # formula for sphere intersections is:
    # lambda = -r0.d +/- sqrt(r0.d - r0^2 + r^2) where:
    #   . is vector dot prod,
    #   r0 is the origin of the ray (rays_o)
    #   d is the normalised direction of the ray (rays_d)

    rays_o_dot_d = torch.sum(rays_o * rays_d, dim=1).unsqueeze(1) # [N, 1]
    determinant = rays_o_dot_d ** 2 + \
                  (sphere_radius ** 2).unsqueeze(0) - \
                  torch.sum(rays_o * rays_o, dim=1).unsqueeze(1)
    determinant_root = torch.sqrt(determinant)

    intersection_z_vals_minus = -rays_o_dot_d - determinant_root
    intersection_z_vals_plus = -rays_o_dot_d + determinant_root

    if positive_only:
        return intersection_z_vals_plus

    intersection_z_vals = torch.cat([torch.flip(intersection_z_vals_minus, dims=[1]),
                                     intersection_z_vals_plus], dim=1)

    return intersection_z_vals
