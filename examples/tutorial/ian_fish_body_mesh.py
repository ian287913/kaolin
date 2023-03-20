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
        
        self.key_size = key_size

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
        ##self.origin_pos = torch.cat((self.origin_xy, self.origin_z.unsqueeze(0)), 0)

        self.length_x = torch.tensor((2), dtype=torch.float, device='cuda', requires_grad=True)
        self.length_y = torch.tensor((0), dtype=torch.float, device='cuda', requires_grad=False)
        self.length_z = torch.tensor((0), dtype=torch.float, device='cuda', requires_grad=False)
        ##self.length_xyz = torch.cat((self.length_x.unsqueeze(0), self.length_y.unsqueeze(0), self.length_z.unsqueeze(0)), 0)

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

    def get_positions_by_uv(self, uv):
        assert self.vertices is not None, f'self.vertices should be set before calling get_positions_by_uvs()!'
        lod_uv = torch.mul(uv, torch.tensor(self.lod_x, self.lod_y))

        floor_u = torch.floor(lod_uv[0])
        ceil_u = torch.ceil(lod_uv[0])
        offset_u = lod_uv[0] - floor_u
        floor_v = torch.floor(lod_uv[1])
        ceil_v = torch.ceil(lod_uv[1])
        offset_v = lod_uv[1] - floor_v
        
        bottom_left = self.vertices[floor_u * self.lod_y + floor_v] 
        top_left = self.vertices[floor_u * self.lod_y + ceil_v] 
        bottom_right = self.vertices[ceil_u * self.lod_y + floor_v] 
        top_right = self.vertices[ceil_u * self.lod_y + ceil_v] 
        
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


################################## Utils ##################################

def train_mesh():
    # Hyperparameters
    image_weight = 0
    alpha_weight = 100
    negative_ys_weight = 1 # this will cause explosion!
    y_lr = 5e-2
    t_lr = 5e-2
    scheduler_step_size = 20
    scheduler_gamma = 0.99
    origin_pos_lr = 5e-2
    length_xyz_lr = 5e-2
    render_res = 512
    texture_res = 512
    sigmainv = 14000 # 3000~30000, for soft mask, higher sharper

    spline_start_train_epoch = 0

    # parameters
    rendered_path_single = "./resources/rendered_mojarra/"
    num_epoch = 200
    visualize_epoch_interval = 10
    key_size = 20
    lod_x = 40
    lod_y = 20

    # texture map
    # imported_texture_map = ian_utils.import_rgb(os.path.join(rendered_path_single, f'{0}_texture.png'))
    # texture_map = imported_texture_map.permute(2, 0, 1).unsqueeze(0).cuda()
    # texture_map.requires_grad = False
    texture_map = torch.ones((1, 3, texture_res, texture_res), dtype=torch.float, device='cuda',
                            requires_grad=False)
    

    # get ground truth
    # get data
    train_data, dataloader = ian_utils.create_dataloader_with_single_view(rendered_path_single, 1)
    data = train_data[0]
    data['metadata']['sigmainv'] = sigmainv
    gt_rgb : torch.Tensor = data['rgb'].cuda()
    gt_alpha : torch.Tensor = data['alpha'].cuda()
    gt_body_mask : torch.Tensor = data['body_mask'].cuda()

    render_res = gt_rgb.shape[0]

    # init optimizer
    fish_body_mesh = FishBodyMesh(
        key_size, 
        y_lr=y_lr, 
        t_lr=t_lr, 
        scheduler_step_size=scheduler_step_size, 
        scheduler_gamma=scheduler_gamma,
        origin_pos_lr=origin_pos_lr,
        length_xyz_lr=length_xyz_lr)
    
    # init renderer
    renderer = ian_renderer.Renderer('cuda', 1, (render_res, render_res))
    

    for epoch in range(num_epoch):

        if (epoch % visualize_epoch_interval == 0):
            fish_body_mesh.update_mesh(lod_x, lod_y)
            visualize_results(fish_body_mesh, renderer, texture_map, data, epoch)

        fish_body_mesh.zero_grad()

        fish_body_mesh.update_mesh(lod_x, lod_y)
        rendered_image, mask, soft_mask = renderer.render_image_and_mask_with_camera_params(
            elev = data['metadata']['cam_elev'], 
            azim = data['metadata']['cam_azim'], 
            r = data['metadata']['cam_radius'], 
            look_at_height = data['metadata']['cam_look_at_height'], 
            fovyangle = data['metadata']['cam_fovyangle'],
            mesh = fish_body_mesh,
            sigmainv = data['metadata']['sigmainv'],
            texture_map=texture_map)

        ### Compute Losses ###
        image_loss = torch.mean(torch.abs(rendered_image - gt_rgb))

        alpha_loss = torch.mean(torch.abs(soft_mask - gt_body_mask[:,:,0]))

        top_spline_negative_ys_loss = fish_body_mesh.top_spline_optimizer.calculate_negative_ys_loss(lod_x)
        bottom_spline_negative_ys_loss = fish_body_mesh.bottom_spline_optimizer.calculate_negative_ys_loss(lod_x)
        negative_ys_loss = top_spline_negative_ys_loss + bottom_spline_negative_ys_loss

        loss = (image_loss * image_weight + alpha_loss * alpha_weight + negative_ys_loss * negative_ys_weight)


        # # dump graph
        # if (epoch % visualize_epoch_interval == 0):
        #     from torchviz import make_dot, make_dot_from_trace
        #     g = make_dot(loss, dict(
        #         length_xyz = curve_mesh_optimizer.length_xyz, 
        #         origin_pos = curve_mesh_optimizer.origin_pos, 
        #         key_ys = curve_mesh_optimizer.spline_optimizer.key_ys, 
        #         key_ts = curve_mesh_optimizer.spline_optimizer.key_ts, 
        #         vertices = curve_mesh_optimizer.body_mesh.vertices, 
        #         texture_map = curve_mesh_optimizer.body_mesh.texture_map, 
        #         output = loss))
        #     g.view()

        ### Update the parameters ###
        loss.backward()

        train_spline = True if epoch >= spline_start_train_epoch else False
        fish_body_mesh.step(train_spline)

        print(f"Epoch {epoch} - loss: {float(loss)}")

    fish_body_mesh.update_mesh(lod_x, lod_y)
    visualize_results(fish_body_mesh, renderer, texture_map, data, epoch)


def visualize_results(fish_body_mesh:FishBodyMesh, renderer:ian_renderer.Renderer, texture_map, data, epoch = 0):
    
    print(f"fish_body_mesh.top_spline_optimizer.key_ys = {fish_body_mesh.top_spline_optimizer.key_ys}")
    print(f"fish_body_mesh.top_spline_optimizer.key_ts = {fish_body_mesh.top_spline_optimizer.key_ts}")
    print(f"fish_body_mesh.bottom_spline_optimizer.key_ys = {fish_body_mesh.bottom_spline_optimizer.key_ys}")
    print(f"fish_body_mesh.bottom_spline_optimizer.key_ts = {fish_body_mesh.bottom_spline_optimizer.key_ts}")
    print(f"fish_body_mesh.origin_xy = {fish_body_mesh.origin_xy}")
    print(f"fish_body_mesh.length_x = {fish_body_mesh.length_x}")
    with torch.no_grad():
        rendered_image, mask, soft_mask = renderer.render_image_and_mask_with_camera_params(
            elev = data['metadata']['cam_elev'], 
            azim = data['metadata']['cam_azim'], 
            r = data['metadata']['cam_radius'], 
            look_at_height = data['metadata']['cam_look_at_height'], 
            fovyangle = data['metadata']['cam_fovyangle'],
            mesh = fish_body_mesh,
            sigmainv = data['metadata']['sigmainv'],
            texture_map=texture_map)

    # # print(f"visualize_results: rendered_image.shape = {rendered_image.shape}")
    ##pylab.imshow(soft_mask[0].repeat(3, 1, 1).permute(1,2,0).cpu().detach())
    pylab.imshow(rendered_image[0].cpu().detach())
    pylab.title(f'epoch: {epoch}')
    pylab.waitforbuttonpress(0)
    pylab.cla()
    
    ##pylab.close()

    ##pylab.savefig(f"./optimization record/{epoch}.png")
    ##pylab.close()

if __name__ == "__main__":
    train_mesh()