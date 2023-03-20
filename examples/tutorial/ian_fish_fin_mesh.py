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
[FishBodyMesh]
generate the mesh of body with trainable parameters:
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

        # init top curve and bottom curve
        self.sil_spline_optimizer = ian_cubic_spline_optimizer.CubicSplineOptimizer(
            key_size,  
            y_lr=y_lr, 
            t_lr=t_lr, 
            scheduler_step_size=scheduler_step_size, 
            scheduler_gamma=scheduler_gamma)
        
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
        root_uvs = ian_utils.torch_linspace(self.start_uv, self.end_uv, lod_x)
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


################################## Utils ##################################

def train_mesh():
    # Hyperparameters
    image_weight = 100
    alpha_weight = 0
    negative_ys_weight = 0 # this will cause explosion!
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
    rendered_path_single = "./resources/rendered_goldfish/"
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