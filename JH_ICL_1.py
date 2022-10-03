import os
import json
import torch
import configargparse

import numpy as np
import depth_transformations as depth_transforms

from tqdm import trange
from features import SpherePosDir
# from util.raygeneration import generate_ray_directions


def load_depth_image(filename, h, w, flip_depth):
    np_file = np.load(filename)
    depth_image = np_file["depth"] if "depth" in np_file.files else np_file[np_file.files[0]]
    depth_image = depth_image.astype(np.float32)
    depth_image = depth_image.reshape(h, w)

    if flip_depth:
        depth_image = np.flip(depth_image, 0)

    return depth_image

def get_min_max_values(depth_image, max_depth_locations, depth_range, device, frame,
                       directions, view_cell_center, view_cell_radius, depth_transform):
    # now the transform is in linear, go back to world and then encoded 0-1
    depth_image = depth_transform.from_world(depth_transforms.LinearTransform.to_world(depth_image, depth_range),
                                             depth_range)

    # set out elements
    depth_image[max_depth_locations] = 1.

    depth_image = torch.from_numpy(depth_image).to(device)

    transform = np.array(frame["transform_matrix"]).astype(np.float32)
    pose = transform[:3, 3:].reshape(-1, 3)
    pose = torch.tensor(pose, device=device, dtype=torch.float32)
    rotation = torch.tensor(transform[:3, :3], device=device, dtype=torch.float32)

    nds = torch.matmul(directions, torch.transpose(rotation, 0, 1))
    distance = SpherePosDir.compute_ray_offset(nds[None], pose, view_cell_center, view_cell_radius)
    mask = depth_image == 1.
    depth_image = depth_transform.to_world(depth_image, depth_range)
    distance = torch.reshape(distance, (depth_image.shape[0], depth_image.shape[1]))
    depth_image = depth_image - distance

    min_v = torch.min(depth_image)
    depth_image[mask] = 0
    max_v = torch.max(depth_image)

    return min_v, max_v
