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
import codecs, json

import ian_torch_cubic_spline_interp
import ian_cubic_spline_optimizer
import ian_utils
import ian_renderer


"""
[FishBodyMesh]
generate the mesh of body with trainable parameters:
    top spline,
    bottom spline,
    origin position,
    length

this mesh does NOT contain texture map!
vertices        (torch.Tensor):     of shape (num_vertices, 3)
faces           (torch.LongTensor): of shape (num_faces, face_size)
uvs             (torch.Tensor):     of shape (num_uvs, 2)
face_uvs_idx    (torch.LongTensor): of shape (num_faces, face_size)
"""
class FishBodyMesh:
    def __init__(self, 
                 key_size = 3, y_lr = 5e-1, t_lr = 5e-1, scheduler_step_size = 20, scheduler_gamma = 0.5,
                 origin_pos_lr = 5e-1, length_xyz_lr = 5e-1
                 ):
        
        # init top curve and bottom curve
        self.top_spline_optimizer = ian_cubic_spline_optimizer.CubicSplineOptimizer(
            key_size,  
            y_lr=y_lr, 
            t_lr=t_lr, 
            scheduler_step_size=scheduler_step_size, 
            scheduler_gamma=scheduler_gamma)
        self.bottom_spline_optimizer = ian_cubic_spline_optimizer.CubicSplineOptimizer(
            key_size,  
            y_lr=y_lr, 
            t_lr=t_lr, 
            scheduler_step_size=scheduler_step_size, 
            scheduler_gamma=scheduler_gamma)
        
        # init mesh
        self.lod_x = None
        self.lod_y = None
        self.vertices = None

        # initialize leaves
        self.origin_xy = torch.tensor((-1, 0), dtype=torch.float, device='cuda', requires_grad=True)
        self.origin_z = torch.tensor((0), dtype=torch.float, device='cuda', requires_grad=False)

        self.length_x = torch.tensor((2), dtype=torch.float, device='cuda', requires_grad=True)
        self.length_y = torch.tensor((0), dtype=torch.float, device='cuda', requires_grad=False)
        self.length_z = torch.tensor((0), dtype=torch.float, device='cuda', requires_grad=False)

        self.parameters = {}
        self.parameters['origin_xy'] = self.origin_xy
        self.parameters['origin_z'] = self.origin_z
        self.parameters['length_x'] = self.length_x
        self.parameters['length_y'] = self.length_y
        self.parameters['length_z'] = self.length_z

        self.parameters['top_spline'] = self.top_spline_optimizer.get_json_dict()
        self.parameters['bottom_spline'] = self.bottom_spline_optimizer.get_json_dict()
        
        self.set_optimizer(scheduler_step_size=scheduler_step_size, scheduler_gamma=scheduler_gamma,
                           origin_pos_lr=origin_pos_lr, length_xyz_lr=length_xyz_lr)
        
    def set_parameters(self, 
                       origin_xy, origin_z,
                       length_x, length_y, length_z,
                       top_spline_json_dict, bottom_spline_json_dict):

        # init top curve and bottom curve
        self.top_spline_optimizer = ian_cubic_spline_optimizer.CubicSplineOptimizer()
        self.bottom_spline_optimizer = ian_cubic_spline_optimizer.CubicSplineOptimizer()
        
        self.top_spline_optimizer.load_from_json_dict(top_spline_json_dict)
        self.bottom_spline_optimizer.load_from_json_dict(bottom_spline_json_dict)
        
        # init mesh
        self.lod_x = None
        self.lod_y = None
        self.vertices = None

        # initialize leaves
        self.origin_xy = torch.tensor(origin_xy, dtype=torch.float, device='cuda', requires_grad=True)
        self.origin_z = torch.tensor(origin_z, dtype=torch.float, device='cuda', requires_grad=False)

        self.length_x = torch.tensor(length_x, dtype=torch.float, device='cuda', requires_grad=True)
        self.length_y = torch.tensor(length_y, dtype=torch.float, device='cuda', requires_grad=False)
        self.length_z = torch.tensor(length_z, dtype=torch.float, device='cuda', requires_grad=False)

        self.parameters = {}
        self.parameters['origin_xy'] = self.origin_xy
        self.parameters['origin_z'] = self.origin_z
        self.parameters['length_x'] = self.length_x
        self.parameters['length_y'] = self.length_y
        self.parameters['length_z'] = self.length_z
        
        self.parameters['top_spline'] = self.top_spline_optimizer.get_json_dict()
        self.parameters['bottom_spline'] = self.bottom_spline_optimizer.get_json_dict()
        
        
    def set_optimizer(self, 
                      scheduler_step_size = 20, scheduler_gamma = 0.5,
                      origin_pos_lr = 5e-1, length_xyz_lr = 5e-1):
        # Hyperparameters
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        self.origin_pos_lr = origin_pos_lr
        self.length_xyz_lr = length_xyz_lr
        
        # initialize optimizers and schedulers
        self.origin_pos_optim  = torch.optim.Adam(params=[self.origin_xy], lr = origin_pos_lr)
        self.length_xyz_optim  = torch.optim.Adam(params=[self.length_x], lr = length_xyz_lr)
        self.origin_pos_scheduler = torch.optim.lr_scheduler.StepLR(
            self.origin_pos_optim,
            step_size=scheduler_step_size,
            gamma=scheduler_gamma)
        self.length_xyz_scheduler = torch.optim.lr_scheduler.StepLR(
            self.length_xyz_optim,
            step_size=scheduler_step_size,
            gamma=scheduler_gamma)
        
        self.optimizer_parameters = {}
        self.optimizer_parameters['origin_pos_lr'] = self.origin_pos_lr
        self.optimizer_parameters['length_xyz_lr'] = self.length_xyz_lr
        self.optimizer_parameters['scheduler_step_size'] = self.scheduler_step_size
        self.optimizer_parameters['scheduler_gamma'] = self.scheduler_gamma


    def zero_grad(self):
        self.top_spline_optimizer.zero_grad()
        self.bottom_spline_optimizer.zero_grad()

        self.origin_pos_optim.zero_grad()
        self.length_xyz_optim.zero_grad()

    def step(self, step_splines = True):
        if (step_splines == True):
            self.top_spline_optimizer.step()
            self.bottom_spline_optimizer.step()

        self.length_xyz_optim.step()
        self.origin_pos_optim.step()
        self.length_xyz_scheduler.step()
        self.origin_pos_scheduler.step()

    def get_positions_by_uvs(self, uvs):
        pos_xyzs = torch.zeros((0, 3), dtype=torch.float, device='cuda',
                                requires_grad=True)
        for uv in uvs:
            new_pos_xyz = self.get_position_by_uv(uv).unsqueeze(0)
            pos_xyzs = torch.cat((pos_xyzs, new_pos_xyz), 0)
        return pos_xyzs

    def get_position_by_uv(self, uv):
        assert self.vertices is not None, f'self.vertices should be set before calling get_positions_by_uvs()!'
        lod_uv = torch.mul(uv, torch.tensor((self.lod_x - 1, self.lod_y - 1), dtype=torch.float, device='cuda', requires_grad=False))

        floor_u = torch.floor(lod_uv[0]).long()
        ceil_u = torch.ceil(lod_uv[0]).long()
        offset_u = lod_uv[0] - floor_u
        floor_v = torch.floor(lod_uv[1]).long()
        ceil_v = torch.ceil(lod_uv[1]).long()
        offset_v = lod_uv[1] - floor_v
        
        bottom_left = self.vertices[0][floor_u * self.lod_y + floor_v] 
        top_left = self.vertices[0][floor_u * self.lod_y + ceil_v] 
        bottom_right = self.vertices[0][ceil_u * self.lod_y + floor_v] 
        top_right = self.vertices[0][ceil_u * self.lod_y + ceil_v] 
        
        left = torch.lerp(bottom_left, top_left, offset_v)
        right = torch.lerp(bottom_right, top_right, offset_v)
        
        return torch.lerp(left, right, offset_u)
        
    def update_mesh(self, lod_x, lod_y):
        self.lod_x = lod_x
        self.lod_y = lod_y
        self.origin_pos = torch.cat((self.origin_xy, self.origin_z.unsqueeze(0)), 0)
        self.length_xyz = torch.cat((self.length_x.unsqueeze(0), self.length_y.unsqueeze(0), self.length_z.unsqueeze(0)), 0)

        self.set_mesh_by_samples(
            ian_renderer.calculate_roots(self.origin_pos, self.length_xyz, lod_x), 
            self.top_spline_optimizer.calculate_ys_with_lod_x(lod_x),
            self.bottom_spline_optimizer.calculate_ys_with_lod_x(lod_x), 
            lod_y)

    def set_mesh_by_samples(self, roots: torch.Tensor, top_ys: torch.Tensor, bottom_ys: torch.Tensor, lod_y):
        assert lod_y > 1, f'lod_y should greater than 1!'
        assert roots.shape[0] > 1, f'roots.shape[0] should greater than 1!'
        assert top_ys.shape[0] > 1, f'top_ys.shape[0] should greater than 1!'
        assert bottom_ys.shape[0] > 1, f'bottom_ys.shape[0] should greater than 1!'
        assert bottom_ys.shape[0] == top_ys.shape[0], f'error: bottom_ys.shape[0] != top_ys.shape[0]'

        # expand top_ys to top_xyzs
        top_xs = torch.zeros(top_ys.shape[0], dtype=torch.float, device='cuda', requires_grad=False)
        top_zs = torch.zeros(top_ys.shape[0], dtype=torch.float, device='cuda', requires_grad=False)
        top_xyzs = torch.cat(
            (top_xs.unsqueeze(1), 
             top_ys.unsqueeze(1), 
             top_zs.unsqueeze(1)), 1)
        # expand bottom_ys to bottom_xyzs (negative)
        bottom_xs = torch.zeros(bottom_ys.shape[0], dtype=torch.float, device='cuda', requires_grad=False)
        bottom_zs = torch.zeros(bottom_ys.shape[0], dtype=torch.float, device='cuda', requires_grad=False)
        bottom_xyzs = torch.cat(
            (bottom_xs.unsqueeze(1), 
             -bottom_ys.unsqueeze(1), 
             bottom_zs.unsqueeze(1)), 1)

        # vertices
        self.vertices = torch.zeros((0, 3), dtype=torch.float, device='cuda',
                                requires_grad=True)
        # for each column
        for idx, root in enumerate(roots):
            top_pos = root + top_xyzs[idx]
            bottom_pos = root + bottom_xyzs[idx]
            new_vertices = ian_utils.torch_linspace(bottom_pos, top_pos, lod_y)
            self.vertices = torch.cat((self.vertices, new_vertices), 0)

        self.vertices = self.vertices.unsqueeze(0)

        # faces
        self.faces = torch.zeros(((lod_y - 1) * (roots.shape[0] - 1) * 2, 3), dtype=torch.long, device='cuda',
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
        for idx, root in enumerate(roots):
            column_u = float(idx) / float(roots.shape[0]-1)
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

    def export_to_json(self, path):
        export_dict = {}
        export_dict['parameters'] = self.parameters
        export_dict['optimizer_parameters'] = self.optimizer_parameters
        converted_export_dict = ian_utils.convert_tensor_dict(export_dict.copy())
        print(f"converted_export_dict = {converted_export_dict}")

        filepath = os.path.join(path,'fish_body.json')
        with open(filepath, 'w') as fp:
            json.dump(converted_export_dict, fp, indent=4)
        print(f'file exported to {filepath}.')

    def import_from_json(self, path):
        json_path = os.path.join(path,'fish_body.json')
        obj_text = codecs.open(json_path, 'r', encoding='utf-8').read()
        json_dict = json.loads(obj_text) #This reads json to list

        self.set_parameters(origin_xy=np.array(json_dict['parameters']['origin_xy']),
                            origin_z=np.array(json_dict['parameters']['origin_z']),
                            length_x=np.array(json_dict['parameters']['length_x']),
                            length_y=np.array(json_dict['parameters']['length_y']),
                            length_z=np.array(json_dict['parameters']['length_z']),
                            top_spline_json_dict=json_dict['parameters']['top_spline'],
                            bottom_spline_json_dict=json_dict['parameters']['bottom_spline'])

        self.set_optimizer(origin_pos_lr=json_dict['optimizer_parameters']['origin_pos_lr'],
                           length_xyz_lr=json_dict['optimizer_parameters']['length_xyz_lr'],
                           scheduler_step_size=json_dict['optimizer_parameters']['scheduler_step_size'],
                           scheduler_gamma=json_dict['optimizer_parameters']['scheduler_gamma'])
