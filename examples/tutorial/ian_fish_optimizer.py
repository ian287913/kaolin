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

import ian_torch_cubic_spline_interp
import ian_cubic_spline_optimizer
import ian_utils
import ian_renderer
import ian_fish_fin_mesh
import ian_fish_body_mesh
import ian_fish_texture


def prepare_body_mesh(hyperparameter:dict, load_path = None):
    fish_body_mesh = ian_fish_body_mesh.FishBodyMesh(
        key_size=               hyperparameter['key_size'], 
        y_lr=                   hyperparameter['y_lr'], 
        t_lr=                   hyperparameter['t_lr'], 
        scheduler_step_size=    hyperparameter['scheduler_step_size'], 
        scheduler_gamma=        hyperparameter['scheduler_gamma'],
        origin_pos_lr=          hyperparameter['origin_pos_lr'],
        length_xyz_lr=          hyperparameter['length_xyz_lr'])
    if (load_path is not None):
        fish_body_mesh.import_from_json(load_path)
    return fish_body_mesh

def prepare_fin_meshes(hyperparameter:dict, load_path = None):
    fish_fin_meshes = {}
    for fin_name in hyperparameter['fin_list']:
        # instantiate a fin
        fish_fin_mesh = ian_fish_fin_mesh.FishFinMesh(
            key_size=               hyperparameter['key_size'], 
            y_lr=                   hyperparameter['y_lr'], 
            t_lr=                   hyperparameter['t_lr'], 
            scheduler_step_size=    hyperparameter['scheduler_step_size'], 
            scheduler_gamma=        hyperparameter['scheduler_gamma'],
            uv_lr=                  hyperparameter['fin_uv_lr'],
            dir_lr=                 hyperparameter['fin_dir_lr'],
            start_uv=[0.5, 0.5], end_uv=[0.5, 0.5])
        # try to load from json
        if (load_path is not None):
            fish_fin_mesh.import_from_json(load_path, fin_name)
        fish_fin_meshes[fin_name] = fish_fin_mesh
    return fish_fin_meshes

def train_fish():
    # Hyperparameters
    image_weight = 1
    alpha_weight = 200
    root_pos_weight = 100
    ###alpha_weight = 4.87 # for IOU
    negative_ys_weight = 0.9 # this will cause explosion!
    y_lr = 5e-2
    t_lr = 5e-2
    scheduler_step_size = 1000
    scheduler_gamma = 0.99
    origin_pos_lr = 5e-4
    length_xyz_lr = 5e-4
    render_res = 512
    texture_res = 256
    sigmainv = 17000 # 3000~30000, for soft mask, higher sharper

    # fin
    fin_dir_lr = 5e-2
    fin_uv_lr = 5e-3
    fin_start_train_epoch = 0
    mask_loss_enable_epoch = 150
    fin_uv_bound_weight = 100

    # parameters
    rendered_path_single = "./resources/rendered_goldfish/"
    str_date_time = datetime.fromtimestamp(datetime.now().timestamp()).strftime("%Y%m%d_%H_%M_%S")
    output_path = './dibr_output/' + str_date_time + '/'
    ian_utils.make_path(Path(output_path))
    ian_utils.make_path(Path(output_path)/'texture')

    visualize_epoch_interval = 10

    key_size = 20
    lod_x = 40
    lod_y = 10

    # texture map
    # imported_texture_map = ian_utils.import_rgb(os.path.join(rendered_path_single, f'{0}_texture.png'))
    # texture_map = imported_texture_map.permute(2, 0, 1).unsqueeze(0).cuda()
    # texture_map.requires_grad = False
    dummy_texture_map = torch.ones((1, 3, texture_res, texture_res), dtype=torch.float, device='cuda',
                            requires_grad=False)
    
    # get ground truth
    train_data, dataloader = ian_utils.create_dataloader_with_single_view(rendered_path_single, 1)
    data = train_data[0]
    data['metadata']['sigmainv'] = sigmainv
    gt_rgb : torch.Tensor = data['rgb'].cuda()

    # update render_res
    render_res = gt_rgb.shape[0]


    # set hyperparameters to data
    hyperparameter = {}
    hyperparameter['num_epoch'] = 500
    hyperparameter['fin_start_train_epoch'] = fin_start_train_epoch
    hyperparameter['mask_loss_enable_epoch'] = mask_loss_enable_epoch
    hyperparameter['key_size'] = key_size
    hyperparameter['lod_x'] = lod_x
    hyperparameter['lod_y'] = lod_y

    hyperparameter['render_res'] = render_res
    hyperparameter['texture_res'] = texture_res
    hyperparameter['texture_lr'] = 5e-2
    hyperparameter['sigmainv'] = sigmainv

    hyperparameter['image_weight'] = image_weight
    hyperparameter['alpha_weight'] = alpha_weight
    hyperparameter['root_pos_weight'] = root_pos_weight
    hyperparameter['negative_ys_weight'] = negative_ys_weight
    hyperparameter['y_lr'] = y_lr
    hyperparameter['t_lr'] = t_lr
    hyperparameter['origin_pos_lr'] = origin_pos_lr
    hyperparameter['length_xyz_lr'] = length_xyz_lr
    hyperparameter['scheduler_step_size'] = scheduler_step_size
    hyperparameter['scheduler_gamma'] = scheduler_gamma

    hyperparameter['fin_dir_lr'] = fin_dir_lr
    hyperparameter['fin_uv_lr'] = fin_uv_lr
    hyperparameter['fin_uv_bound_weight'] = fin_uv_bound_weight

    # lr control
    hyperparameter['body_lr_epoch_list'] =   [0,    100,    150,    300,    400]
    hyperparameter['body_spline_lr_list'] =  [0,    0,      2e-2,   5e-3,   5e-4]
    hyperparameter['body_root_lr_list'] =    [5e-2, 5e-3,   5e-4,   5e-5,   5e-5]

    hyperparameter['fin_lr_epoch_list'] =   [0,     150,    300,    400,    500]
    hyperparameter['fin_t_lr_list'] =       [0,     0,      3e-4,   3e-5,   3e-6]
    hyperparameter['fin_y_lr_list'] =       [0,     0,      5e-2,   3e-2,   3e-2]
    hyperparameter['fin_uv_lr_list'] =      [3e-2,  3e-3,   3e-4,   0,      0]
    hyperparameter['fin_dir_lr_list'] =     [0,     1e-1,   5e-2,   4e-3,   4e-3]
    
    data['hyperparameter'] = hyperparameter
    data['rendered_path'] = rendered_path_single
    data['output_path'] = output_path

    # fins that will be instantiated
    data['hyperparameter']['fin_list'] = ['dorsal_fin', 'caudal_fin', 'anal_fin', 'pelvic_fin', 'pectoral_fin']
    ##data['hyperparameter']['fin_list'] = []

    # init body mesh
    load_body_mesh_from_json = True
    if (load_body_mesh_from_json):
        fish_body_mesh = prepare_body_mesh(hyperparameter, rendered_path_single)
    else:
        fish_body_mesh = prepare_body_mesh(hyperparameter, None)
    
    # init fin mesh
    load_fin_mesh_from_json = True
    if (load_fin_mesh_from_json):
        fish_fin_meshes = prepare_fin_meshes(hyperparameter, rendered_path_single)
    else:
        fish_fin_meshes = prepare_fin_meshes(hyperparameter, None)

    # init texture
    fish_texture = ian_fish_texture.FishTexture(hyperparameter['texture_res'], 
                                 hyperparameter['texture_lr'],
                                 hyperparameter['scheduler_step_size'],
                                 hyperparameter['scheduler_gamma'])

    # init renderer
    renderer = ian_renderer.Renderer('cuda', 1, (render_res, render_res), 'bilinear')
    
    texture_start_train_epoch = 0

    ##################################### TRAINING #####################################
    loss_history = []
    for epoch in range(hyperparameter['num_epoch']):

        
        # print result
        if (epoch % visualize_epoch_interval == 0 and (epoch >= fin_start_train_epoch or True)):
            visualize_results(fish_body_mesh, fish_fin_meshes, renderer, dummy_texture_map, data, epoch, False, fish_texture)

        
        # train texture
        if (epoch >= texture_start_train_epoch):
            loss = train_texture(fish_body_mesh, fish_fin_meshes, renderer, fish_texture, data, epoch)
        # train body
        elif (epoch < fin_start_train_epoch):
            loss = train_body_mesh(fish_body_mesh, renderer, dummy_texture_map, data, epoch)
        # train fins
        else:
            loss = 0
            if(fish_body_mesh.dirty):
                with torch.no_grad():
                    fish_body_mesh.update_mesh(lod_x, lod_y)
            for fin_name in data['hyperparameter']['fin_list']:
                loss += train_fin_mesh(fish_fin_meshes[fin_name], fish_body_mesh, 
                                       renderer, dummy_texture_map, 
                                       data, data[fin_name + '_mask'], data['root_segmentation'][fin_name + '_mask'], 
                                       epoch)
        print(f"Epoch {epoch} - loss: {float(loss)}")
        loss_history.append(loss.detach().cpu().numpy().tolist())


    visualize_results(fish_body_mesh, fish_fin_meshes, renderer, dummy_texture_map, data, epoch + 1)
    visualize_results(fish_body_mesh, fish_fin_meshes, renderer, dummy_texture_map, data, epoch + 1, True)

    # export stuff
    export_fish_body_json(data['output_path'], fish_body_mesh)
    for fin_name in data['hyperparameter']['fin_list']:
        export_fish_fin_json(data['output_path'], fish_fin_meshes[fin_name], fin_name)

    export_hyperparameter_json(data['output_path'], data['hyperparameter'])

    export_loss_history(data['output_path'], loss_history)

# arrange each mesh vu in a NxN grid
def reshape_mesh_uvs(meshes:list):
    mesh_count = len(meshes)
    grid_count = math.ceil(math.sqrt(mesh_count))
    grid_size = 1. / float(grid_count)

    for u in range(grid_count):
        for v in range(grid_count):
            idx = u * grid_count + v
            if (idx >= mesh_count):
                continue
            meshes[idx].reshape_uvs([u * grid_size, v * grid_size, grid_size, grid_size])

def train_texture(fish_body_mesh:ian_fish_body_mesh.FishBodyMesh, 
                  fish_fin_meshes:dict,
                  renderer:ian_renderer.Renderer,
                  fish_texture:ian_fish_texture.FishTexture,
                  data, 
                  epoch):
    lod_x = data['hyperparameter']['lod_x']
    lod_y = data['hyperparameter']['lod_y']
    image_weight = data['hyperparameter']['image_weight']
    
    # override lr
    # TODO...

    # reset texture
    fish_texture.zero_grad()

    # reset mesh
    if(fish_body_mesh.dirty):
        with torch.no_grad():
            fish_body_mesh.update_mesh(lod_x, lod_y)
            # also update fins
            for fin_name, fin_mesh in fish_fin_meshes.items():
                fin_mesh.update_mesh(fish_body_mesh, lod_x, lod_y)
    
    # reshape uvs
    all_meshes = list(fish_fin_meshes.values())
    all_meshes.insert(0, fish_body_mesh)
    reshape_mesh_uvs(all_meshes)

    # jitter the camera to reduce unsampled texture pixels
    random_direction = random.normal(0, 0.1)

    # render mesh
    loss = 0
    for mesh in all_meshes:
        rendered_image, mask, soft_mask = renderer.render_image_and_mask_with_camera_params(
            elev = data['metadata']['cam_elev'] + random_direction, 
            azim = data['metadata']['cam_azim'] + random_direction, 
            r = data['metadata']['cam_radius'], 
            look_at_height = data['metadata']['cam_look_at_height'], 
            fovyangle = data['metadata']['cam_fovyangle'],
            mesh = mesh,
            sigmainv = data['metadata']['sigmainv'],
            texture_map=fish_texture.texture)
    
        ### Image Losses ###
        image_loss = torch.mean(torch.abs(rendered_image - data['rgb'].cuda()))
        loss += image_loss * image_weight

    ##loss = image_loss * image_weight

    ### Update the parameters ###
    loss.backward()

    # step
    fish_texture.step()

    return loss

def train_body_mesh(fish_body_mesh:ian_fish_body_mesh.FishBodyMesh, 
                   renderer:ian_renderer.Renderer, texture_map,
                   data, 
                   epoch):
    
    lod_x = data['hyperparameter']['lod_x']
    lod_y = data['hyperparameter']['lod_y']

    alpha_weight = data['hyperparameter']['alpha_weight']
    negative_ys_weight = data['hyperparameter']['negative_ys_weight']
    root_pos_weight = data['hyperparameter']['root_pos_weight']

    # disable mask weight according to the epoch
    if (epoch < data['hyperparameter']['mask_loss_enable_epoch']):
        alpha_weight = 0

    # override lr
    spline_lr = 0
    root_lr = 0
    for idx, lr_epoch in enumerate(data['hyperparameter']['body_lr_epoch_list']):
        if (epoch >= lr_epoch):
            spline_lr = data['hyperparameter']['body_spline_lr_list'][idx]
            root_lr = data['hyperparameter']['body_root_lr_list'][idx]
    fish_body_mesh.set_t_lr(spline_lr)
    fish_body_mesh.set_y_lr(spline_lr)
    fish_body_mesh.set_root_lr(root_lr)

    # reset mesh
    fish_body_mesh.zero_grad()
    fish_body_mesh.update_mesh(lod_x, lod_y)

    # render mesh
    rendered_image, mask, soft_mask = renderer.render_image_and_mask_with_camera_params(
        elev = data['metadata']['cam_elev'], 
        azim = data['metadata']['cam_azim'], 
        r = data['metadata']['cam_radius'], 
        look_at_height = data['metadata']['cam_look_at_height'], 
        fovyangle = data['metadata']['cam_fovyangle'],
        mesh = fish_body_mesh,
        sigmainv = data['metadata']['sigmainv'],
        texture_map=texture_map)
    
    # calculate projected root position
    projected_start_root_xy, projected_end_root_xy = fish_body_mesh.get_projected_start_and_end_positions(renderer, data['metadata'])
    gt_body_mask_root_xys = data['root_segmentation']['body_mask']
    gt_start_root_xy = torch.tensor(gt_body_mask_root_xys[0], dtype=torch.float, device='cuda', requires_grad=False)
    gt_end_root_xy = torch.tensor(gt_body_mask_root_xys[1], dtype=torch.float, device='cuda', requires_grad=False)
    ### root pos loss ###
    root_pos_loss = torch.abs(projected_start_root_xy - gt_start_root_xy).mean() + torch.abs(projected_end_root_xy - gt_end_root_xy).mean()
    ### Alpha Losses ###
    alpha_loss = torch.mean(torch.abs(soft_mask - data['body_mask'][:,:,0].cuda()))
    ### Negative Ys Losses ###
    top_spline_negative_ys_loss = fish_body_mesh.top_spline_optimizer.calculate_negative_ys_loss(lod_x)
    bottom_spline_negative_ys_loss = fish_body_mesh.bottom_spline_optimizer.calculate_negative_ys_loss(lod_x)
    negative_ys_loss = top_spline_negative_ys_loss + bottom_spline_negative_ys_loss


    loss = (alpha_loss * alpha_weight + 
            negative_ys_loss * negative_ys_weight + 
            root_pos_loss * root_pos_weight)
    
    # # dump graph
    # if (epoch % visualize_epoch_interval == 0):
    #     from torchviz import make_dot, make_dot_from_trace
    #     g = make_dot(root_pos_loss, dict(
    #         origin_xy = fish_body_mesh.origin_xy, 
    #         length_x = fish_body_mesh.length_x, 
    #         length_y = fish_body_mesh.length_y, 
    #         key_ys = fish_body_mesh.top_spline_optimizer.key_ys, 
    #         key_ts = fish_body_mesh.top_spline_optimizer.key_ts, 
    #         vertices = fish_body_mesh.vertices, 
    #         root_pos_loss = root_pos_loss))
    #     g.view()

    ### Update the parameters ###
    loss.backward()

    # step
    train_spline = True
    fish_body_mesh.step(train_spline)

    return loss

def train_fin_mesh(fish_fin_mesh:ian_fish_fin_mesh.FishFinMesh, fish_body_mesh, 
                   renderer, texture_map,
                   data, gt_fin_mask, gt_fin_mask_root_xys, 
                   epoch):
    
    lod_x = data['hyperparameter']['lod_x']
    lod_y = data['hyperparameter']['lod_y']
    alpha_weight = data['hyperparameter']['alpha_weight']
    negative_ys_weight = data['hyperparameter']['negative_ys_weight']
    fin_uv_bound_weight = data['hyperparameter']['fin_uv_bound_weight']
    root_pos_weight = data['hyperparameter']['root_pos_weight']

    # try to expand fin dir
    if (epoch == 300 or epoch == 350):
        for fin_name in data['hyperparameter']['fin_list']:
            fish_fin_mesh.reset_dir(fish_fin_mesh.start_dir + 0.1, fish_fin_mesh.end_dir - 0.1)

    # set lr according to epoch
    t_lr = 0
    y_lr = 0
    uv_lr = 0
    dir_lr = 0
    for idx, lr_epoch in enumerate(data['hyperparameter']['fin_lr_epoch_list']):
        if (epoch >= lr_epoch):
            t_lr = data['hyperparameter']['fin_t_lr_list'][idx]
            y_lr = data['hyperparameter']['fin_y_lr_list'][idx]
            uv_lr = data['hyperparameter']['fin_uv_lr_list'][idx]
            dir_lr = data['hyperparameter']['fin_dir_lr_list'][idx]

    fish_fin_mesh.set_t_lr(t_lr)
    fish_fin_mesh.set_y_lr(y_lr)
    fish_fin_mesh.set_uv_lr(uv_lr)
    fish_fin_mesh.set_dir_lr(dir_lr)

    # disable mask weight according to the epoch
    if (epoch < data['hyperparameter']['mask_loss_enable_epoch']):
        alpha_weight = 0

    fish_fin_mesh.zero_grad()
    fish_fin_mesh.update_mesh(fish_body_mesh, lod_x, lod_y)
    # render the fin
    rendered_image, mask, soft_mask = renderer.render_image_and_mask_with_camera_params(
        elev = data['metadata']['cam_elev'], 
        azim = data['metadata']['cam_azim'], 
        r = data['metadata']['cam_radius'], 
        look_at_height = data['metadata']['cam_look_at_height'], 
        fovyangle = data['metadata']['cam_fovyangle'],
        mesh = fish_fin_mesh,
        sigmainv = data['metadata']['sigmainv'],
        texture_map=texture_map)
    
    # get the projected root positions
    projected_start_root_xy, projected_end_root_xy = fish_fin_mesh.get_projected_start_and_end_positions(fish_body_mesh, renderer, data['metadata'])
    gt_start_root_xy = torch.tensor(gt_fin_mask_root_xys[0], dtype=torch.float, device='cuda', requires_grad=True)
    gt_end_root_xy = torch.tensor(gt_fin_mask_root_xys[1], dtype=torch.float, device='cuda', requires_grad=True)
    ### root pos loss ###
    root_pos_loss = torch.abs(projected_start_root_xy - gt_start_root_xy).mean() + torch.abs(projected_end_root_xy - gt_end_root_xy).mean()

    ### Alpha Losses ###
    alpha_loss = torch.mean(torch.abs(soft_mask - gt_fin_mask[:,:,0].cuda()))
    ##alpha_loss = kal.metrics.render.mask_iou(soft_mask, gt_fin_mask[:,:,0].cuda().unsqueeze(0))
    ### Negative Ys Losses ###
    sil_spline_negative_ys_loss = fish_fin_mesh.sil_spline_optimizer.calculate_negative_ys_loss(lod_x)
    negative_ys_loss = sil_spline_negative_ys_loss
    ### Uv Bound Losses ###
    fin_uv_bound_loss = fish_fin_mesh.calculate_uv_bound_loss()

    loss = (alpha_loss * alpha_weight + 
            negative_ys_loss * negative_ys_weight + 
            fin_uv_bound_loss * fin_uv_bound_weight +
            root_pos_loss * root_pos_weight)

    ### Update the parameters ###
    loss.backward()

    fish_fin_mesh.step()

    return loss

def visualize_results(fish_body_mesh:ian_fish_body_mesh.FishBodyMesh, fish_fin_meshes, renderer:ian_renderer.Renderer, dummy_texture, data, epoch = 0, add_gt = False, fish_texture:ian_fish_texture.FishTexture = None):
    
    # print(f"fish_body_mesh.top_spline_optimizer.key_ys = {fish_body_mesh.top_spline_optimizer.key_ys}")
    # print(f"fish_body_mesh.top_spline_optimizer.key_ts = {fish_body_mesh.top_spline_optimizer.key_ts}")
    # print(f"fish_body_mesh.bottom_spline_optimizer.key_ys = {fish_body_mesh.bottom_spline_optimizer.key_ys}")
    # print(f"fish_body_mesh.bottom_spline_optimizer.key_ts = {fish_body_mesh.bottom_spline_optimizer.key_ts}")
    # print(f"fish_body_mesh.origin_xy = {fish_body_mesh.origin_xy}")
    # print(f"fish_body_mesh.length_x = {fish_body_mesh.length_x}")
    # if (fish_fin_mesh is not None):
    #     print(f"fish_fin_mesh.start_uv = {fish_fin_mesh.start_uv}")
    #     print(f"fish_fin_mesh.end_uv = {fish_fin_mesh.end_uv}")
    #     print(f"fish_fin_mesh.start_dir = {fish_fin_mesh.start_dir}")
    #     print(f"fish_fin_mesh.end_dir = {fish_fin_mesh.end_dir}")

    lod_x = data['hyperparameter']['lod_x']
    lod_y = data['hyperparameter']['lod_y']
    if (fish_texture == None):
        texture_map = dummy_texture
    else:
        texture_map = fish_texture.texture

    rendered_image = None
    with torch.no_grad():
        if (fish_body_mesh.dirty):
            fish_body_mesh.update_mesh(lod_x, lod_y)
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
        if (fish_texture == None):
            rendered_image = rendered_body_image * torch.tensor((0, 0, 1), dtype=torch.float, device='cuda', requires_grad=False)
        else:
            rendered_image = rendered_body_image

        # projected root position
        if (add_gt):
            projected_start_xy, projected_end_xy = fish_body_mesh.get_projected_start_and_end_positions(renderer, data['metadata'])
            pylab.plot(projected_start_xy[0].cpu() * renderer.render_res[0], (1-projected_start_xy[1].cpu()) * renderer.render_res[1], marker='4', color='royalblue')
            pylab.plot(projected_end_xy[0].cpu() * renderer.render_res[0], (1-projected_end_xy[1].cpu()) * renderer.render_res[1], marker='3', color='royalblue')
            gt_body_mask_root_xys = data['root_segmentation']['body_mask']
            pylab.plot(gt_body_mask_root_xys[0][0] * renderer.render_res[0], (1-gt_body_mask_root_xys[0][1]) * renderer.render_res[1], marker='4', color='white')
            pylab.plot(gt_body_mask_root_xys[1][0] * renderer.render_res[0], (1-gt_body_mask_root_xys[1][1]) * renderer.render_res[1], marker='3', color='white')

        # fin
        for fin_name in data['hyperparameter']['fin_list']:
            if (fin_name in fish_fin_meshes):
                fish_fin_mesh:ian_fish_fin_mesh.FishFinMesh = fish_fin_meshes[fin_name]
                # render mesh
                fish_fin_mesh.update_mesh(fish_body_mesh, lod_x, lod_y)
                rendered_fin_image, mask, soft_mask = renderer.render_image_and_mask_with_camera_params(
                    elev = data['metadata']['cam_elev'], 
                    azim = data['metadata']['cam_azim'], 
                    r = data['metadata']['cam_radius'], 
                    look_at_height = data['metadata']['cam_look_at_height'], 
                    fovyangle = data['metadata']['cam_fovyangle'],
                    mesh = fish_fin_mesh,
                    sigmainv = data['metadata']['sigmainv'],
                    texture_map=texture_map)
                
                
                # set fin color and texture
                if (fish_texture == None):
                    draw_color = (1, 0, 0)
                    if (fin_name == 'pelvic_fin'):
                        draw_color = (0, 1, 0)
                    rendered_image += rendered_fin_image * torch.tensor(draw_color, dtype=torch.float, device='cuda', requires_grad=False)
                else:
                    rendered_image += rendered_fin_image

                # projected root position
                if (add_gt):
                    projected_start_xy, projected_end_xy = fish_fin_mesh.get_projected_start_and_end_positions(fish_body_mesh, renderer, data['metadata'])
                    pylab.plot(projected_start_xy[0].cpu() * renderer.render_res[0], (1-projected_start_xy[1].cpu()) * renderer.render_res[1], marker='*', color='salmon')
                    pylab.plot(projected_end_xy[0].cpu() * renderer.render_res[0], (1-projected_end_xy[1].cpu()) * renderer.render_res[1], marker='x', color='salmon')
                    gt_fin_mask_root_xys = data['root_segmentation'][f'{fin_name}_mask']
                    pylab.plot(gt_fin_mask_root_xys[0][0] * renderer.render_res[0], (1-gt_fin_mask_root_xys[0][1]) * renderer.render_res[1], marker='*', color='white')
                    pylab.plot(gt_fin_mask_root_xys[1][0] * renderer.render_res[0], (1-gt_fin_mask_root_xys[1][1]) * renderer.render_res[1], marker='x', color='white')

                # print(f"fish_fin_mesh.start_uv = {fish_fin_mesh.start_uv}")
                # print(f"fish_fin_mesh.end_uv = {fish_fin_mesh.end_uv}")
                # print(f"fish_fin_mesh.start_dir = {fish_fin_mesh.start_dir}")
                # print(f"fish_fin_mesh.end_dir = {fish_fin_mesh.end_dir}")

            else:
                print(f'failed to render fin: {fin_name}. the name does not found in the fish_fin_meshes.')
    
    image_name = str(epoch)

    # gt
    gt_body_draw_color = (0.2, 0.3, 0)
    gt_fin_draw_color = (0.5, 0.5, 0.5)
    if (add_gt):
        image_name += "_with_gt"
        # body
        rendered_image += data['body_mask'].cuda() * torch.tensor(gt_body_draw_color, dtype=torch.float, device='cuda', requires_grad=False)
        # fins
        for fin_name in data['hyperparameter']['fin_list']:
            if (fin_name in fish_fin_meshes):
                gt_fin_mask = data[fin_name + '_mask'].cuda()
                rendered_image += gt_fin_mask * torch.tensor(gt_fin_draw_color, dtype=torch.float, device='cuda', requires_grad=False)

    # # print(f"visualize_results: rendered_image.shape = {rendered_image.shape}")
    #pylab.imshow(soft_mask[0].repeat(3, 1, 1).permute(1,2,0).cpu().detach())
    pylab.imshow(rendered_image[0].cpu().detach())
    pylab.title(f'epoch: {image_name}')

    # save or show image
    save_image = True
    if (save_image):
        fig_path = f"{data['output_path']}{image_name}.png"
        pylab.savefig(fig_path)
        pylab.close()
        print(f'fig saved at {fig_path}')
        if (fish_texture is not None):
            ian_utils.save_image(torch.clamp(texture_map, 0., 1.).permute(0, 2, 3, 1), Path(data['output_path'])/'texture', f'{image_name}_texture')
    else:
        pylab.waitforbuttonpress(0)
        pylab.cla()

    
def export_fish_body_json(path, mesh:ian_fish_body_mesh.FishBodyMesh):
    with torch.no_grad():
        mesh.export_to_json(path)
def export_fish_fin_json(path, mesh:ian_fish_fin_mesh.FishFinMesh, fin_name):
    with torch.no_grad():
        mesh.export_to_json(path, fin_name)
def export_hyperparameter_json(path, hyperparameter):
    export_dict = {}
    export_dict['hyperparameter'] = hyperparameter
    converted_export_dict = ian_utils.convert_tensor_dict(export_dict.copy())
    print(f"converted_export_dict = {converted_export_dict}")

    filepath = os.path.join(path,'hyperparameter.json')
    with open(filepath, 'w') as fp:
        json.dump(converted_export_dict, fp, indent=4)
    print(f'file exported to {filepath}.')

def import_fish_body_json(path, mesh:ian_fish_body_mesh.FishBodyMesh):
    mesh.import_from_json(path)

def export_loss_history(path, loss_history):
    filepath = os.path.join(path,'loss_history.txt')
    with open(filepath, 'w') as fp:
        for loss in loss_history:
            fp.write(str(loss) + "\n")
    print(f'file exported to {filepath}.')

if __name__ == "__main__":
    start_time = time.time()
    train_fish()
    end_time = time.time()
    time_lapsed = end_time - start_time
    print(f'Total time elapsed: {time_lapsed}s')
