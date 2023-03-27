import json
import os
import glob
import time

import torch
import torch.nn.functional as F
from torch.nn.functional import normalize
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from pathlib import Path
import ian_utils

import kaolin as kal
import numpy as np
import matplotlib.pylab as pylab
import codecs, json

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
                 start_dir = [0], end_dir = [0], 
                 init_height = 0.2
                 ):
        
        self.key_size = key_size

        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        self.uv_lr = uv_lr
        self.dir_lr = dir_lr

        # init sil curve and bottom curve
        self.sil_spline_optimizer = ian_cubic_spline_optimizer.CubicSplineOptimizer(
            key_size,  
            y_lr=y_lr, 
            t_lr=t_lr, 
            scheduler_step_size=scheduler_step_size, 
            scheduler_gamma=scheduler_gamma,
            init_key_ys=init_height)
        
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
        
        self.parameters = {}
        self.parameters['start_uv'] = self.start_uv
        self.parameters['end_uv'] = self.end_uv
        self.parameters['start_dir'] = self.start_dir
        self.parameters['end_dir'] = self.end_dir
        self.parameters['init_height'] = init_height
        self.parameters['sil_spline'] = self.sil_spline_optimizer.get_json_dict()

        self.optimizer_parameters = {}
        self.optimizer_parameters['uv_lr'] = self.uv_lr
        self.optimizer_parameters['dir_lr'] = self.dir_lr
        self.optimizer_parameters['scheduler_step_size'] = self.scheduler_step_size
        self.optimizer_parameters['scheduler_gamma'] = self.scheduler_gamma

    
    def set_parameters(self, 
                       start_uv, end_uv,
                       start_dir, end_dir,
                       init_height,
                       sil_spline_json_dict):

        # init top curve and bottom curve
        self.sil_spline_optimizer = ian_cubic_spline_optimizer.CubicSplineOptimizer()
        
        self.sil_spline_optimizer.load_from_json_dict(sil_spline_json_dict)
        
        # initialize leaves
        self.start_uv = torch.tensor(start_uv, dtype=torch.float, device='cuda', requires_grad=True)
        self.end_uv = torch.tensor(end_uv, dtype=torch.float, device='cuda', requires_grad=True)

        self.start_dir = torch.tensor((start_dir), dtype=torch.float, device='cuda', requires_grad=True)
        self.end_dir = torch.tensor((end_dir), dtype=torch.float, device='cuda', requires_grad=True)

        self.parameters = {}
        self.parameters['start_uv'] = self.start_uv
        self.parameters['end_uv'] = self.end_uv
        self.parameters['start_dir'] = self.start_dir
        self.parameters['end_dir'] = self.end_dir
        self.parameters['init_height'] = init_height
        self.parameters['sil_spline'] = self.sil_spline_optimizer.get_json_dict()

    def set_optimizer(self, 
                      scheduler_step_size = 20, scheduler_gamma = 0.5,
                      uv_lr = 5e-1, dir_lr = 5e-1):
        # Hyperparameters
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        self.uv_lr = uv_lr
        self.dir_lr = dir_lr
        
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
        
        self.optimizer_parameters = {}
        self.optimizer_parameters['uv_lr'] = self.uv_lr
        self.optimizer_parameters['dir_lr'] = self.dir_lr
        self.optimizer_parameters['scheduler_step_size'] = self.scheduler_step_size
        self.optimizer_parameters['scheduler_gamma'] = self.scheduler_gamma


    def set_uv_lr(self, lr):
        for g in self.start_uv_optim.param_groups:
            g['lr'] = lr
        for g in self.end_uv_optim.param_groups:
            g['lr'] = lr
    def set_dir_lr(self, lr):
        for g in self.start_dir_optim.param_groups:
            g['lr'] = lr
        for g in self.end_dir_optim.param_groups:
            g['lr'] = lr
    def set_y_lr(self, lr):
        self.sil_spline_optimizer.set_y_lr(lr)
    def set_t_lr(self, lr):
        self.sil_spline_optimizer.set_t_lr(lr)
    
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
        
        # calculate root_dirs (by root_xyzs direction).rotate(90 degree)
        # TODO: the "dx" could be smaller to increase precision
        grow_dirs = (root_xyzs[1] - root_xyzs[0]).unsqueeze(0)
        for idx in range(1, root_xyzs.shape[0]):
            root_dir = root_xyzs[idx] - root_xyzs[idx - 1]
            grow_dir_x = -root_dir[1]
            grow_dir_y = root_dir[0]
            grow_dir_z = torch.tensor(0, dtype=grow_dir_x.dtype, device=grow_dir_x.device, requires_grad=False)
            stacked_grow_dir = torch.stack((grow_dir_x, grow_dir_y, grow_dir_z), -1)
            grow_dir = normalize(stacked_grow_dir, p=2.0, dim=0) * ys[idx]
            grow_dirs = torch.cat((grow_dirs, grow_dir.unsqueeze(0)), 0)

        # rotate grow_xyzs
        rotated_grow_xyzs = self.rotate_grow_xyzs(grow_dirs)


        # vertices
        self.vertices = torch.zeros((0, 3), dtype=torch.float, device='cuda',
                                requires_grad=True)
        # for each column
        for idx, root in enumerate(root_xyzs):
            new_vertices = ian_utils.torch_linspace(root, root + rotated_grow_xyzs[idx], lod_y)
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
    
    def rotate_grow_xyzs(self, grow_xyzs: torch.Tensor):
        rotated_grow_xyzs = torch.zeros((0, 3), dtype=grow_xyzs.dtype, device=grow_xyzs.device,
                                requires_grad=True)
        # angles
        rotate_dirs = ian_utils.torch_linspace(self.start_dir, self.end_dir, grow_xyzs.shape[0])

        # for each column
        for idx, grow_xyz in enumerate(grow_xyzs):
            angle = rotate_dirs[idx]
            x = grow_xyz[0] * torch.cos(angle) - grow_xyz[1] * torch.sin(angle)
            y = grow_xyz[0] * torch.sin(angle) + grow_xyz[1] * torch.cos(angle)
            z = grow_xyz[2] * torch.ones(1, dtype=grow_xyzs.dtype, device=grow_xyzs.device)
            rotated_xyz = torch.stack((x, y, z), -1)
            rotated_grow_xyzs = torch.cat((rotated_grow_xyzs, rotated_xyz), 0)

        return rotated_grow_xyzs
    
    def export_to_json(self, path, fin_name):
        export_dict = {}
        export_dict['parameters'] = self.parameters
        export_dict['optimizer_parameters'] = self.optimizer_parameters
        converted_export_dict = ian_utils.convert_tensor_dict(export_dict.copy())
        print(f"converted_export_dict = {converted_export_dict}")

        filepath = os.path.join(path,f'{fin_name}.json')
        with open(filepath, 'w') as fp:
            json.dump(converted_export_dict, fp, indent=4)
        print(f'file exported to {filepath}.')

    def import_from_json(self, path, fin_name):
        json_path = os.path.join(path,f'{fin_name}.json')
        obj_text = codecs.open(json_path, 'r', encoding='utf-8').read()
        json_dict = json.loads(obj_text) #This reads json to list

        self.set_parameters(start_uv=np.array(json_dict['parameters']['start_uv']),
                            end_uv=np.array(json_dict['parameters']['end_uv']),
                            start_dir=np.array(json_dict['parameters']['start_dir']),
                            end_dir=np.array(json_dict['parameters']['end_dir']),
                            init_height=np.array(json_dict['parameters']['init_height']),
                            sil_spline_json_dict=json_dict['parameters']['sil_spline'])

        self.set_optimizer(scheduler_step_size=json_dict['optimizer_parameters']['scheduler_step_size'],
                           scheduler_gamma=json_dict['optimizer_parameters']['scheduler_gamma'],
                           uv_lr=json_dict['optimizer_parameters']['uv_lr'],
                           dir_lr=json_dict['optimizer_parameters']['dir_lr'])
