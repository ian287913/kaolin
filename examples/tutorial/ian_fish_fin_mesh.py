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
import ian_utils

import kaolin as kal
import numpy as np
import matplotlib.pylab as pylab

import ian_torch_cubic_spline_interp
import ian_cubic_spline_optimizer
import ian_utils
import ian_renderer
import ian_fish_body_mesh


"""
[FishFinMesh]
generate the mesh of fin on top of a body with trainable parameters:
    silhouette spline,
    start uv,
    end uv,
    start dir,
    end dir

this mesh does NOT contain texture map!
"""
class FishFinMesh:
    def __init__(self, 
                 key_size = 3, y_lr = 5e-1, t_lr = 5e-1, 
                 scheduler_step_size = 20, scheduler_gamma = 0.5,
                 uv_lr = 5e-1, dir_lr = 5e-1,
                 start_uv = [0, 1], end_uv = [1, 1], 
                 start_dir = [90], end_dir = [90], 
                 ):
        
        self.key_size = key_size

        # init sil curve and bottom curve
        self.sil_spline_optimizer = ian_cubic_spline_optimizer.CubicSplineOptimizer(
            key_size,  
            y_lr=y_lr, 
            t_lr=t_lr, 
            scheduler_step_size=scheduler_step_size, 
            scheduler_gamma=scheduler_gamma,
            init_key_ys=0.3)
        
        # initialize leaves
        self.start_uv = torch.tensor(start_uv, dtype=torch.float, device='cuda', requires_grad=True)
        self.end_uv = torch.tensor(end_uv, dtype=torch.float, device='cuda', requires_grad=True)

        self.start_dir = torch.tensor((start_dir), dtype=torch.float, device='cuda', requires_grad=True)
        self.end_dir = torch.tensor((end_dir), dtype=torch.float, device='cuda', requires_grad=True)

        # initialize optimizers and schedulers
        self.start_uv_optim  = torch.optim.Adam(params=[self.start_uv], lr = uv_lr)
        self.end_uv_optim  = torch.optim.Adam(params=[self.end_uv], lr = uv_lr)
        self.start_uv_scheduler = torch.optim.lr_scheduler.StepLR(
            self.start_uv_optim,
            step_size=scheduler_step_size,
            gamma=scheduler_gamma)
        self.end_uv_scheduler = torch.optim.lr_scheduler.StepLR(
            self.end_uv_optim,
            step_size=scheduler_step_size,
            gamma=scheduler_gamma)
        
        self.start_dir_optim  = torch.optim.Adam(params=[self.start_dir], lr = dir_lr)
        self.end_dir_optim  = torch.optim.Adam(params=[self.end_dir], lr = dir_lr)
        self.start_dir_scheduler = torch.optim.lr_scheduler.StepLR(
            self.start_dir_optim,
            step_size=scheduler_step_size,
            gamma=scheduler_gamma)
        self.end_dir_scheduler = torch.optim.lr_scheduler.StepLR(
            self.end_dir_optim,
            step_size=scheduler_step_size,
            gamma=scheduler_gamma)

    # penalize uv that is not inside [(0, 0), (1, 1)]
    def calculate_uv_bound_loss(self):
        exceeded_value = torch.tensor((0), dtype=torch.float, device='cuda')
        if (self.start_uv[0] > 1):
            exceeded_value += torch.square(self.start_uv[0] - 1)
        if (self.start_uv[1] > 1):
            exceeded_value += torch.square(self.start_uv[1] - 1)
        if (self.start_uv[0] < 0):
            exceeded_value += torch.square(self.start_uv[0])
        if (self.start_uv[1] < 0):
            exceeded_value += torch.square(self.start_uv[1])
        
        if (self.end_uv[0] > 1):
            exceeded_value += torch.square(self.end_uv[0] - 1)
        if (self.end_uv[1] > 1):
            exceeded_value += torch.square(self.end_uv[1] - 1)
        if (self.end_uv[0] < 0):
            exceeded_value += torch.square(self.end_uv[0])
        if (self.end_uv[1] < 0):
            exceeded_value += torch.square(self.end_uv[1])

        return exceeded_value

    def zero_grad(self):
        self.sil_spline_optimizer.zero_grad()

        self.start_uv_optim.zero_grad()
        self.end_uv_optim.zero_grad()
        self.start_dir_optim.zero_grad()
        self.end_dir_optim.zero_grad()

    def step(self, step_splines = True):
        if (step_splines == True):
            self.sil_spline_optimizer.step()

        self.start_uv_optim.step()
        self.end_uv_optim.step()
        self.start_dir_optim.step()
        self.end_dir_optim.step()

        self.start_uv_scheduler.step()
        self.end_uv_scheduler.step()
        self.start_dir_scheduler.step()
        self.end_dir_scheduler.step()


    def update_mesh(self, body_mesh: ian_fish_body_mesh.FishBodyMesh, lod_x, lod_y):
        clamped_start_uv = torch.clamp(self.start_uv, 0, 1)
        clamped_end_uv = torch.clamp(self.end_uv, 0, 1)
        root_uvs = ian_utils.torch_linspace(clamped_start_uv, clamped_end_uv, lod_x)
        root_xyzs = body_mesh.get_positions_by_uvs(root_uvs)

        self.set_mesh_by_samples(
            root_xyzs, 
            self.sil_spline_optimizer.calculate_ys_with_lod_x(lod_x),
            lod_y)

    def set_mesh_by_samples(self, root_xyzs: torch.Tensor, ys: torch.Tensor, lod_y):
        assert lod_y > 1, f'lod_y should greater than 1!'
        assert root_xyzs.shape[0] > 1, f'root_xyzs.shape[0] should greater than 1!'
        assert ys.shape[0] > 1, f'ys.shape[0] should greater than 1!'

        # expand top_ys to top_xyzs
        xs = torch.zeros(ys.shape[0], dtype=torch.float, device='cuda', requires_grad=False)
        zs = torch.zeros(ys.shape[0], dtype=torch.float, device='cuda', requires_grad=False)
        grow_xyzs = torch.cat(
            (xs.unsqueeze(1), 
             ys.unsqueeze(1), 
             zs.unsqueeze(1)), 1)
        
        # rotate grow_xyzs
        # TODO...

        # vertices
        self.vertices = torch.zeros((0, 3), dtype=torch.float, device='cuda',
                                requires_grad=True)
        # for each column
        for idx, root in enumerate(root_xyzs):
            new_vertices = ian_utils.torch_linspace(root, root + grow_xyzs[idx], lod_y)
            self.vertices = torch.cat((self.vertices, new_vertices), 0)

        self.vertices = self.vertices.unsqueeze(0)

        # faces
        self.faces = torch.zeros(((lod_y - 1) * (root_xyzs.shape[0] - 1) * 2, 3), dtype=torch.long, device='cuda',
                                requires_grad=False)
        # set the first quad (2 triangles)
        self.faces[0] = torch.Tensor([0, lod_y, 1])
        self.faces[1] = torch.Tensor([1, lod_y, lod_y+1])
        # set all faces
        for tri_idx in range(2, self.faces.shape[0], 2):
            self.faces[tri_idx] = self.faces[tri_idx - 2] + 1
            self.faces[tri_idx + 1] = self.faces[tri_idx - 1] + 1
            if ((tri_idx / 2) % (lod_y - 1) == 0):
                self.faces[tri_idx] = self.faces[tri_idx] + 1
                self.faces[tri_idx + 1] = self.faces[tri_idx + 1] + 1

        # uvs
        self.uvs = torch.zeros((0, 2), dtype=torch.float, device='cuda',
                                requires_grad=False)
        # for each column
        for idx, root in enumerate(root_xyzs):
            column_u = float(idx) / float(root_xyzs.shape[0]-1)
            top_uv = torch.Tensor([column_u, 1.0])
            bottom_uv = torch.Tensor([column_u, 0.0])
            new_uvs = ian_utils.torch_linspace(bottom_uv, top_uv, lod_y).cuda()
            self.uvs = torch.cat((self.uvs, new_uvs), 0)
        self.uvs = self.uvs.cuda().unsqueeze(0)

        # face_uvs_idx
        self.face_uvs_idx = torch.clone(self.faces).detach()
        self.face_uvs_idx.requires_grad = False

        # face_uvs
        self.face_uvs = kal.ops.mesh.index_vertices_by_faces(self.uvs, self.face_uvs_idx).detach()
        self.face_uvs.requires_grad = False
