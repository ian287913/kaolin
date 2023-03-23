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
import ian_fish_fin_mesh
import ian_fish_body_mesh



def train_mesh():
    # Hyperparameters
    image_weight = 0
    alpha_weight = 100
    negative_ys_weight = 2 # this will cause explosion!
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

    # fin
    # fin_uv_lr = 0
    fin_dir_lr = 5e-2
    fin_uv_lr = 5e-2
    # fin_dir_lr = 5e-2
    fin_start_train_epoch = 100
    fin_uv_bound_weight = 100

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
    if 'dorsal_fin_mask' in data :
        gt_dorsal_fin_mask : torch.Tensor = data['dorsal_fin_mask'].cuda()
    else:
        gt_dorsal_fin_mask : torch.Tensor = None

    render_res = gt_rgb.shape[0]

    # init optimizer
    # body mesh
    fish_body_mesh = ian_fish_body_mesh.FishBodyMesh(
        key_size, 
        y_lr=y_lr, 
        t_lr=t_lr, 
        scheduler_step_size=scheduler_step_size, 
        scheduler_gamma=scheduler_gamma,
        origin_pos_lr=origin_pos_lr,
        length_xyz_lr=length_xyz_lr)
    
    # fin mesh
    fish_fin_mesh = ian_fish_fin_mesh.FishFinMesh(
        key_size, 
        y_lr=y_lr, 
        t_lr=t_lr, 
        scheduler_step_size=scheduler_step_size, 
        scheduler_gamma=scheduler_gamma,
        uv_lr=fin_uv_lr,
        dir_lr=fin_dir_lr,
        start_uv=[0, 1], end_uv=[0.9, 0.9])
    
    # init renderer
    renderer = ian_renderer.Renderer('cuda', 1, (render_res, render_res))
    

    for epoch in range(num_epoch):

        if (epoch % visualize_epoch_interval == 0 and epoch >= fin_start_train_epoch):
            fish_body_mesh.update_mesh(lod_x, lod_y)
            fish_fin_mesh.update_mesh(fish_body_mesh, lod_x, lod_y)
            visualize_results(fish_body_mesh, fish_fin_mesh, renderer, texture_map, data, epoch)


        if (epoch <= fin_start_train_epoch):
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

        else:
            fish_fin_mesh.zero_grad()
            fish_body_mesh.update_mesh(lod_x, lod_y)
            fish_fin_mesh.update_mesh(fish_body_mesh, lod_x, lod_y)
            rendered_image, mask, soft_mask = renderer.render_image_and_mask_with_camera_params(
                elev = data['metadata']['cam_elev'], 
                azim = data['metadata']['cam_azim'], 
                r = data['metadata']['cam_radius'], 
                look_at_height = data['metadata']['cam_look_at_height'], 
                fovyangle = data['metadata']['cam_fovyangle'],
                mesh = fish_fin_mesh,
                sigmainv = data['metadata']['sigmainv'],
                texture_map=texture_map)

            ### Compute Losses ###
            alpha_loss = torch.mean(torch.abs(soft_mask - gt_dorsal_fin_mask[:,:,0]))

            sil_spline_negative_ys_loss = fish_fin_mesh.sil_spline_optimizer.calculate_negative_ys_loss(lod_x)
            negative_ys_loss = sil_spline_negative_ys_loss
            fin_uv_bound_loss = fish_fin_mesh.calculate_uv_bound_loss()

            loss = (alpha_loss * alpha_weight + negative_ys_loss * negative_ys_weight + fin_uv_bound_loss * fin_uv_bound_weight)

            ### Update the parameters ###
            loss.backward()

            train_spline = True if epoch >= spline_start_train_epoch else False
            fish_fin_mesh.step(train_spline)

        print(f"Epoch {epoch} - loss: {float(loss)}")

    fish_body_mesh.update_mesh(lod_x, lod_y)
    fish_fin_mesh.update_mesh(fish_body_mesh, lod_x, lod_y)
    visualize_results(fish_body_mesh, fish_fin_mesh, renderer, texture_map, data, epoch)


def visualize_results(fish_body_mesh:ian_fish_body_mesh.FishBodyMesh, fish_fin_mesh:ian_fish_fin_mesh.FishFinMesh, renderer:ian_renderer.Renderer, texture_map, data, epoch = 0):
    
    print(f"fish_body_mesh.top_spline_optimizer.key_ys = {fish_body_mesh.top_spline_optimizer.key_ys}")
    print(f"fish_body_mesh.top_spline_optimizer.key_ts = {fish_body_mesh.top_spline_optimizer.key_ts}")
    print(f"fish_body_mesh.bottom_spline_optimizer.key_ys = {fish_body_mesh.bottom_spline_optimizer.key_ys}")
    print(f"fish_body_mesh.bottom_spline_optimizer.key_ts = {fish_body_mesh.bottom_spline_optimizer.key_ts}")
    print(f"fish_body_mesh.origin_xy = {fish_body_mesh.origin_xy}")
    print(f"fish_body_mesh.length_x = {fish_body_mesh.length_x}")
    if (fish_fin_mesh is not None):
        print(f"fish_fin_mesh.start_uv = {fish_fin_mesh.start_uv}")
        print(f"fish_fin_mesh.end_uv = {fish_fin_mesh.end_uv}")
        print(f"fish_fin_mesh.start_dir = {fish_fin_mesh.start_dir}")
        print(f"fish_fin_mesh.end_dir = {fish_fin_mesh.end_dir}")


    rendered_image = None
    with torch.no_grad():
        # body
        rendered_body_image, mask, soft_mask = renderer.render_image_and_mask_with_camera_params(
            elev = data['metadata']['cam_elev'], 
            azim = data['metadata']['cam_azim'], 
            r = data['metadata']['cam_radius'], 
            look_at_height = data['metadata']['cam_look_at_height'], 
            fovyangle = data['metadata']['cam_fovyangle'],
            mesh = fish_body_mesh,
            sigmainv = data['metadata']['sigmainv'],
            texture_map=texture_map)
        # set body to blue
        rendered_image = rendered_body_image * torch.tensor((0, 0, 1), dtype=torch.float, device='cuda', requires_grad=False)

        # fin
        if (fish_fin_mesh is not None):
            rendered_fin_image, mask, soft_mask = renderer.render_image_and_mask_with_camera_params(
                elev = data['metadata']['cam_elev'], 
                azim = data['metadata']['cam_azim'], 
                r = data['metadata']['cam_radius'], 
                look_at_height = data['metadata']['cam_look_at_height'], 
                fovyangle = data['metadata']['cam_fovyangle'],
                mesh = fish_fin_mesh,
                sigmainv = data['metadata']['sigmainv'],
                texture_map=texture_map)
            rendered_image += rendered_fin_image * torch.tensor((1, 0, 0), dtype=torch.float, device='cuda', requires_grad=False)
            
    

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