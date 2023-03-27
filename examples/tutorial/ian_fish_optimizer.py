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
    alpha_weight = 200
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
    fin_uv_lr = 5e-3
    # fin_dir_lr = 5e-2
    fin_start_train_epoch = 0
    fin_uv_bound_weight = 100

    # parameters
    rendered_path_single = "./resources/rendered_goldfish/"
    str_date_time = datetime.fromtimestamp(datetime.now().timestamp()).strftime("%Y%m%d_%H_%M_%S")
    output_path = './dibr_output/' + str_date_time + '/'
    ian_utils.make_path(Path(output_path))

    num_epoch = 300
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

    # update render_res
    render_res = gt_rgb.shape[0]


    # set hyperparameters to data
    hyperparameter = {}
    hyperparameter['num_epoch'] = num_epoch
    hyperparameter['key_size'] = key_size
    hyperparameter['lod_x'] = lod_x
    hyperparameter['lod_y'] = lod_y

    hyperparameter['render_res'] = render_res
    hyperparameter['texture_res'] = texture_res
    hyperparameter['sigmainv'] = sigmainv

    hyperparameter['image_weight'] = image_weight
    hyperparameter['alpha_weight'] = alpha_weight
    hyperparameter['negative_ys_weight'] = negative_ys_weight
    hyperparameter['y_lr'] = y_lr
    hyperparameter['t_lr'] = t_lr
    hyperparameter['origin_pos_lr'] = origin_pos_lr
    hyperparameter['length_xyz_lr'] = length_xyz_lr

    hyperparameter['fin_dir_lr'] = fin_dir_lr
    hyperparameter['fin_uv_lr'] = fin_uv_lr
    hyperparameter['fin_uv_bound_weight'] = fin_uv_bound_weight

    hyperparameter['fin_lr_epoch_1'] = 0
    hyperparameter['fin_spline_lr_1'] = 0
    hyperparameter['fin_uv_lr_1'] = 5e-3
    # hyperparameter['fin_spline_lr_1'] = 5e-2
    # hyperparameter['fin_uv_lr_1'] = 5e-3

    hyperparameter['fin_lr_epoch_2'] = 150
    hyperparameter['fin_spline_lr_2'] = 5e-2
    hyperparameter['fin_uv_lr_2'] = 5e-5
    # hyperparameter['fin_spline_lr_2'] = 5e-2
    # hyperparameter['fin_uv_lr_2'] = 5e-3
    
    data['hyperparameter'] = hyperparameter

    data['rendered_path'] = rendered_path_single
    
    data['output_path'] = output_path



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
    
    # load body mesh
    import_fish_body_json(rendered_path_single, fish_body_mesh)
    
    # fin mesh
    fish_fin_meshes = {}
    fish_fin_meshes['dorsal_fin'] = ian_fish_fin_mesh.FishFinMesh(
        key_size, 
        y_lr=y_lr, 
        t_lr=t_lr, 
        scheduler_step_size=scheduler_step_size, 
        scheduler_gamma=scheduler_gamma,
        uv_lr=fin_uv_lr,
        dir_lr=fin_dir_lr,
        start_uv=[0, 1], end_uv=[1, 1])
    fish_fin_meshes['caudal_fin'] = ian_fish_fin_mesh.FishFinMesh(
        key_size, 
        y_lr=y_lr, 
        t_lr=t_lr, 
        scheduler_step_size=scheduler_step_size, 
        scheduler_gamma=scheduler_gamma,
        uv_lr=fin_uv_lr,
        dir_lr=fin_dir_lr,
        start_uv=[0, 0], end_uv=[0, 1])
    fish_fin_meshes['anal_fin'] = ian_fish_fin_mesh.FishFinMesh(
        key_size, 
        y_lr=y_lr, 
        t_lr=t_lr, 
        scheduler_step_size=scheduler_step_size, 
        scheduler_gamma=scheduler_gamma,
        uv_lr=fin_uv_lr,
        dir_lr=fin_dir_lr,
        start_uv=[0.3, 0], end_uv=[0.05, 0])
    fish_fin_meshes['pelvic_fin'] = ian_fish_fin_mesh.FishFinMesh(
        key_size, 
        y_lr=y_lr, 
        t_lr=t_lr, 
        scheduler_step_size=scheduler_step_size, 
        scheduler_gamma=scheduler_gamma,
        uv_lr=fin_uv_lr,
        dir_lr=fin_dir_lr,
        start_uv=[0.6, 0.1], end_uv=[0.4, 0.1])
    fish_fin_meshes['pectoral_fin'] = ian_fish_fin_mesh.FishFinMesh(
        key_size, 
        y_lr=y_lr, 
        t_lr=t_lr, 
        scheduler_step_size=scheduler_step_size, 
        scheduler_gamma=scheduler_gamma,
        uv_lr=fin_uv_lr,
        dir_lr=fin_dir_lr,
        start_uv=[0.5, 0.3], end_uv=[0.5, 0.5])
    
    ##fin_list = ['dorsal_fin', 'caudal_fin', 'anal_fin', 'pelvic_fin', 'pectoral_fin']
    fin_list = ['pectoral_fin']
    data['hyperparameter']['fin_list'] = fin_list

    # # load saved fins
    # for fin_name in data['hyperparameter']['fin_list']:
    #     import_fish_fin_json(rendered_path_single, fish_fin_meshes[fin_name], fin_name)

    # init renderer
    renderer = ian_renderer.Renderer('cuda', 1, (render_res, render_res))
    

    for epoch in range(num_epoch):

        if (epoch % visualize_epoch_interval == 0 and (epoch >= fin_start_train_epoch or True)):
            fish_body_mesh.update_mesh(lod_x, lod_y)
            visualize_results(fish_body_mesh, fish_fin_meshes, renderer, texture_map, data, epoch)

        # train body
        if (epoch < fin_start_train_epoch):
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
        # train fins
        else:
            loss = 0
            with torch.no_grad():
                fish_body_mesh.update_mesh(lod_x, lod_y)
            for fin_name in data['hyperparameter']['fin_list']:
                loss += train_fin_mesh(fish_fin_meshes[fin_name], fish_body_mesh, renderer, texture_map, data, data[fin_name + '_mask'], epoch)

        print(f"Epoch {epoch} - loss: {float(loss)}")

    visualize_results(fish_body_mesh, fish_fin_meshes, renderer, texture_map, data, epoch + 1)

    # export stuff
    export_fish_body_json(data['output_path'], fish_body_mesh)
    for fin_name in data['hyperparameter']['fin_list']:
        export_fish_fin_json(data['output_path'], fish_fin_meshes[fin_name], fin_name)

    export_hyperparameter_json(data['output_path'], data['hyperparameter'])



def train_fin_mesh(fish_fin_mesh:ian_fish_fin_mesh.FishFinMesh, fish_body_mesh, 
                   renderer, texture_map,
                   data, gt_fin_mask, epoch):
    lod_x = data['hyperparameter']['lod_x']
    lod_y = data['hyperparameter']['lod_y']
    alpha_weight = data['hyperparameter']['alpha_weight']
    negative_ys_weight = data['hyperparameter']['negative_ys_weight']
    fin_uv_bound_weight = data['hyperparameter']['fin_uv_bound_weight']

    # set lr according to epoch
    spline_lr = 0
    uv_lr = 0
    if (epoch > data['hyperparameter']['fin_lr_epoch_1']):
        spline_lr = data['hyperparameter']['fin_spline_lr_1']
        uv_lr = data['hyperparameter']['fin_uv_lr_1']
    if (epoch > data['hyperparameter']['fin_lr_epoch_2']):
        spline_lr = data['hyperparameter']['fin_spline_lr_2']
        uv_lr = data['hyperparameter']['fin_uv_lr_2']
    fish_fin_mesh.set_t_lr(spline_lr)
    fish_fin_mesh.set_y_lr(spline_lr)
    fish_fin_mesh.set_uv_lr(uv_lr)


    fish_fin_mesh.zero_grad()
    ###fish_body_mesh.update_mesh(lod_x, lod_y)
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
    alpha_loss = torch.mean(torch.abs(soft_mask - gt_fin_mask[:,:,0].cuda()))

    sil_spline_negative_ys_loss = fish_fin_mesh.sil_spline_optimizer.calculate_negative_ys_loss(lod_x)
    negative_ys_loss = sil_spline_negative_ys_loss
    fin_uv_bound_loss = fish_fin_mesh.calculate_uv_bound_loss()

    loss = (alpha_loss * alpha_weight + negative_ys_loss * negative_ys_weight + fin_uv_bound_loss * fin_uv_bound_weight)

    ### Update the parameters ###
    loss.backward()

    fish_fin_mesh.step()

    return loss

def visualize_results(fish_body_mesh:ian_fish_body_mesh.FishBodyMesh, fish_fin_meshes, renderer:ian_renderer.Renderer, texture_map, data, epoch = 0):
    
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

    rendered_image = None
    with torch.no_grad():
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
        rendered_image = rendered_body_image * torch.tensor((0, 0, 1), dtype=torch.float, device='cuda', requires_grad=False)

        # fin
        for fin_name in data['hyperparameter']['fin_list']:
            if (fin_name in fish_fin_meshes):
                fish_fin_mesh:ian_fish_fin_mesh.FishFinMesh = fish_fin_meshes[fin_name]
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
                
                draw_color = (1, 0, 0)
                if (fin_name == 'pelvic_fin'):
                    draw_color = (0, 1, 0)
                rendered_image += rendered_fin_image * torch.tensor(draw_color, dtype=torch.float, device='cuda', requires_grad=False)

                #
                print(f"fish_fin_mesh.start_uv = {fish_fin_mesh.start_uv}")
                print(f"fish_fin_mesh.end_uv = {fish_fin_mesh.end_uv}")
                print(f"fish_fin_mesh.start_dir = {fish_fin_mesh.start_dir}")
                print(f"fish_fin_mesh.end_dir = {fish_fin_mesh.end_dir}")


            else:
                print(f'failed to render fin: {fin_name}. the name does not found in the fish_fin_meshes.')
    
    # # print(f"visualize_results: rendered_image.shape = {rendered_image.shape}")
    #pylab.imshow(soft_mask[0].repeat(3, 1, 1).permute(1,2,0).cpu().detach())
    pylab.imshow(rendered_image[0].cpu().detach())
    pylab.title(f'epoch: {epoch}')

    # save or show image
    save_image = False
    if (save_image):
        fig_path = f"{data['output_path']}{epoch}.png"
        pylab.savefig(fig_path)
        pylab.close()
        print(f'fig saved at {fig_path}')
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

    filepath = os.path.join(path,'pyperparameter.json')
    with open(filepath, 'w') as fp:
        json.dump(converted_export_dict, fp, indent=4)
    print(f'file exported to {filepath}.')

def import_fish_body_json(path, mesh:ian_fish_body_mesh.FishBodyMesh):
    mesh.import_from_json(path)
def import_fish_fin_json(path, mesh:ian_fish_fin_mesh.FishFinMesh, fin_name):
    mesh.import_from_json(path, fin_name)

if __name__ == "__main__":
    start_time = time.time()
    train_mesh()
    end_time = time.time()
    time_lapsed = end_time - start_time
    print(f'Total time elapsed: {time_lapsed}s')
