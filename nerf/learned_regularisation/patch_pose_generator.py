import random
from dataclasses import dataclass
from typing import Optional

import torch
import numpy as np
from scipy.spatial.transform import Rotation

from nerf.circle_path import make_4x4_transform
from nerf.learned_regularisation.intrinsics import Intrinsics


@dataclass
class FrustumChecker:
    """
    Simple class to check whether a point lies within the camera frustum.
    Parameterised by the camera FOVs along the horizontal and vertical axes.
    """
    fov_x_rads: float
    fov_y_rads: float

    def is_in_frustum(self, pose_c2w, point_world) -> bool:
        rotation, translation = unpack_4x4_transform(np.linalg.inv(pose_c2w))
        point_cam = rotation @ point_world + translation
        x, y, z = point_cam
        alpha_x = np.arctan(x / (z + 1e-6))
        alpha_y = np.arctan(y / (z + 1e-6))
        # Point must be in front of camera, not behind it
        if z < 0:
            return False
        # Point must be within frustum
        if np.abs(alpha_x) < 0.5 * self.fov_x_rads and np.abs(alpha_y) < 0.5 * self.fov_y_rads:
            return True
        return False


class FrustumRegulariser:
    def __init__(self, poses: list[torch.Tensor], intrinsics: Intrinsics, reg_strength: float,
                 min_near: float = 0.):
        """
        Frustum regulariser as described in the DiffusioNeRF paper. Exists to penalise placement of material
        that is visible only in one frustum.
        :param poses: List of poses for the training views, so that the i-th entry in poses if the pose of i-th training
        view.
        :param intrinsics: Intrinsics for the training views.
        :param reg_strength: Multiplier for the loss function.
        :param min_near: Points will be only considered to lie in a frustum if their depth is at least min_near. This should be the same
        value of min_near which is used in rendering (i.e. whatever opt.min_near is).
        """
        self.transforms_w2c = [torch.linalg.inv(pose) for pose in poses]
        self.intrinsics = intrinsics
        self.reg_strength = reg_strength
        self.min_near = min_near
        assert self.min_near >= 0.
        print('Initialised FrustumRegulariser with min_near', self.min_near)

    def is_in_frustum(self, transform_w2c, points_world) -> float:
        rotation, translation = unpack_4x4_transform(transform_w2c)
        points_cam = points_world @ rotation.T + translation
        x = points_cam[..., 0]
        y = points_cam[..., 1]
        z = points_cam[..., 2]

        pixel_i = self.intrinsics.fx * (x/(z + 1e-12)) + self.intrinsics.cx
        pixel_j = self.intrinsics.fy * (y/(z + 1e-12)) + self.intrinsics.cy

        z_mask = z >= self.min_near
        pixel_i_mask = (0. <= pixel_i) & (pixel_i <= self.intrinsics.width)
        pixel_j_mask = (0. <= pixel_j) & (pixel_j <= self.intrinsics.height)
        return z_mask & pixel_i_mask & pixel_j_mask

    def count_frustums(self, xyzs):
        frustum_counts = torch.zeros(len(xyzs,)).to(xyzs.device)
        for tf in self.transforms_w2c:
            frustum_counts += self.is_in_frustum(tf, xyzs)
        return frustum_counts

    def __call__(self, xyzs: torch.Tensor, weights: torch.Tensor, frustum_count_thresh: int = 2,
                 debug_vis_name: Optional[str] = None) -> torch.Tensor:
        """
        Compute the frustum regularisation loss for some points.
        Will compute a loss proportional to the total amount of alpha-compositing weight which
        is visible from fewer than frustum_count_thresh frustums.

        :param xyzs: Points lying on rays which are being used in rendering.
        :param weights: The weights used for each of those points in alpha-compositing.
        :param frustum_count_thresh:
        :param debug_vis_name: Optional name for writing debug point clouds.
        :return: Frustum regularisation loss.
        """
        with torch.no_grad():
            frustum_counts = self.count_frustums(xyzs)
        print('Frustum count range', frustum_counts.min(), frustum_counts.max())

        penalty_size = frustum_count_thresh - frustum_counts
        penalty_size = torch.clip(penalty_size, min=0.)
        loss = self.reg_strength * weights * penalty_size

        # For debug: write points inside & outside frustum
        if debug_vis_name is not None:
            loss_mask = frustum_counts < frustum_count_thresh
            loss = self.reg_strength * weights[loss_mask].unsqueeze(-1)
            in_frustum = xyzs[~loss_mask].detach().cpu().numpy()
            outside_frustum = xyzs[loss_mask].detach().cpu().numpy()
            np.savetxt(f'in_frustum-{debug_vis_name}.txt', in_frustum)
            np.savetxt(f'outside_frustum-{debug_vis_name}.txt', outside_frustum)
            zero_frustum_xyzs = xyzs[frustum_counts < 0.5].detach().cpu().numpy()
            np.savetxt(f'zero_frustum_{debug_vis_name}.txt', zero_frustum_xyzs)

        return loss.sum()


class PatchPoseGenerator:
    """
    Generates poses at which to render patches, by taking the training poses and perturbing them.
    """
    def __init__(self, poses, spatial_perturbation_magnitude: float, angular_perturbation_magnitude_rads: float,
                 no_perturb_prob: float = 0., frustum_checker: Optional[FrustumChecker] = None):
        """
        Initialise the pose generator with a set of poses and an amount of jitter to apply to them.
        :param poses: List of training poses, s.t. the i-th entry is the pose of the i-th training view.
        :param spatial_perturbation_magnitude: Amount of jitter to apply to camera centres, in units of length.
        :param angular_perturbation_magnitude_rads: Amount of jitter to apply to camera orientations, in radians.
        :param no_perturb_prob: Fraction of the time to not apply any perturbation at all.
        :param frustum_checker: If given, will ensure that the perturbed camera centre lies within at least one
        training frustum.
        """
        self._poses = poses
        self._spatial_mag = spatial_perturbation_magnitude
        self._angular_mag = angular_perturbation_magnitude_rads
        self._no_perturb_prob = no_perturb_prob
        self._frustum_checker = frustum_checker

    def __len__(self):
        return len(self._poses)

    def _perturb_pose(self, pose_to_perturb):
        while True:
            new_pose = perturb_pose(pose_c2w=pose_to_perturb, spatial_mag=self._spatial_mag,
                                    angular_mag=self._angular_mag)
            _, camera_centre = unpack_4x4_transform(new_pose)
            for pose in self._poses:
                if self._frustum_checker is None or self._frustum_checker.is_in_frustum(pose_c2w=pose,
                                                                                        point_world=camera_centre):
                    return new_pose
                else:
                    pass


    def __getitem__(self, idx):
        # Generate a pose by perturbing the idx-th training view.
        pose = self._poses[idx]
        if random.random() > self._no_perturb_prob:
            new_pose = self._perturb_pose(pose)
        else:
            new_pose = pose
        return torch.tensor(new_pose, dtype=torch.float)

    def generate_random(self):
        # Generate a pose by perturbing a random training view.
        idx = random.randint(0, len(self._poses)-1)
        return self[idx]


def unpack_4x4_transform(transform_mat):
    # Unpack transform into rotation & translation parts; obviously not valid if your matrix is more exotic
    # than an affine transform
    return transform_mat[:3, :3], transform_mat[:3, 3]


def perturb_pose(pose_c2w, spatial_mag: float, angular_mag: float):
    camera_orientation, camera_centre = unpack_4x4_transform(pose_c2w)

    # Sample perturbation to camera centre
    cam_centre_perturbation = spatial_mag * (2. * torch.rand(3,) - 1.)

    # Sample perturbation to orientation
    rotation_perturbation = Rotation.random().as_rotvec() * (angular_mag / (2. * torch.pi))
    rotation_perturbation = Rotation.from_rotvec(rotation_perturbation).as_matrix()
    new_rotation = rotation_perturbation @ np.asarray(camera_orientation)

    new_pose = make_4x4_transform(
        rotation=new_rotation,
        translation=camera_centre + cam_centre_perturbation
    )
    return new_pose
