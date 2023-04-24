import json
import os
import glob
import time

import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from pathlib import Path
from datetime import datetime  
from numpy import random  
import math  

import kaolin as kal
import numpy as np
import matplotlib.pylab as pylab

# from https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def axis_angle_to_quaternion(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    normalized_axis = torch.nn.functional.normalize(axis, p=2.0, dim=0)
    cos = torch.cos(angle/2.0)
    sin = torch.sin(angle/2.0)
    quaternion = torch.stack(
        (
            cos,
            sin * normalized_axis[0],
            sin * normalized_axis[1],
            sin * normalized_axis[2],
        ),
        -1,
    )
    return quaternion

def axis_angle_to_rotation_matrix(axis: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
    quaternion = axis_angle_to_quaternion(axis, radius)
    rotation_matrix = quaternion_to_matrix(quaternion)
    return rotation_matrix

def rotate_v3_along_axis(vector: torch.Tensor, axis: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
    rotation_matrix = axis_angle_to_rotation_matrix(axis, radius)
    return torch.matmul(rotation_matrix, vector)

def test():
    pi = torch.pi
    quaternion = axis_angle_to_quaternion(torch.Tensor((-1, 2, -3)), torch.tensor(pi/4));
    print(f"quaternion = {quaternion}")
    rotation_matrix = quaternion_to_matrix(quaternion)
    print(f"rotation_matrix = {rotation_matrix}")

    # rotate points
    point_x = torch.Tensor([2, 3, 40])
    point_y = torch.Tensor([0, 1, 0])
    point_z = torch.Tensor([0, 0, 1])
    transformed_point_x = torch.matmul(rotation_matrix, point_x)
    transformed_point_y = torch.matmul(rotation_matrix, point_y)
    transformed_point_z = torch.matmul(rotation_matrix, point_z)
    print(f"transformed_point_x = \n{transformed_point_x}\n")
    print(f"transformed_point_y = \n{transformed_point_y}\n")
    print(f"transformed_point_z = \n{transformed_point_z}\n")


if __name__ == "__main__":
    test()