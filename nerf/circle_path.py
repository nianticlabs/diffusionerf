import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp


def make_4x4_transform(rotation, translation):
    transform_4x4 = np.zeros((4, 4))
    transform_4x4[:3, :3] = rotation
    transform_4x4[:3, 3] = translation
    transform_4x4[3, 3] = 1.
    return transform_4x4


def apply_4x4_transform_to_3d_vec(transform_4x4, vec):
    return (transform_4x4[:3, :3] @ np.concatenate(vec, [1.], axis=0))[:3]


def make_circle_path(start_pose, circle_size: float, num_poses: int):
    print('Making circular path')
    start_pose = np.array(start_pose)

    circle_poses = []

    for phi in np.linspace(0., 2. * torch.pi, num_poses):
        camera_centre_delta = circle_size * np.array([np.cos(phi), np.sin(phi), 0.])
        print(f'Phi = {phi}')
        print('Camera centre delta', camera_centre_delta)

        # world_from_cam_prime = world_from_cam_start @ cam_start_from_cam_prime
        cam_start_from_cam_prime = make_4x4_transform(rotation=np.identity(3), translation=camera_centre_delta)
        world_from_cam_prime = start_pose @ cam_start_from_cam_prime
        print('Original pose:')
        print(start_pose)
        print('Output pose:')
        print(world_from_cam_prime)
        circle_poses.append(world_from_cam_prime)

    return circle_poses
