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
import ian_pixel_filler

# image route (relative)
target_folder = './dibr_output/20230507_20_12_43 (tuna)/'


def load_mesh(folder_path):
    # find obj file
    found_obj_filepath = glob.glob(os.path.join(folder_path,'*.obj'))
    if (len(found_obj_filepath) < 1):
        print(f'No obj file is found in directory {folder_path}')
        return None

    print(f'Found {len(found_obj_filepath)} obj files in directory {folder_path}')

    obj_filepath = found_obj_filepath[0]

    mesh = kal.io.obj.import_mesh(obj_filepath, with_materials=True)
    vertices = mesh.vertices.cuda().unsqueeze(0)
    print(f"vertices.shape = {vertices.shape}")
    faces = mesh.faces.cuda()
    uvs = mesh.uvs.cuda().unsqueeze(0)
    face_uvs_idx = mesh.face_uvs_idx.cuda()
    face_uvs = kal.ops.mesh.index_vertices_by_faces(uvs, face_uvs_idx).detach()
    face_uvs.requires_grad = False

    return (vertices, faces, uvs, face_uvs)

def load_image(path : Path, requires_grad = False):
    loaded_image = ian_utils.import_rgb(path)
    loaded_image = loaded_image.permute(2, 0, 1).unsqueeze(0).cuda()
    loaded_image.requires_grad = requires_grad
    return loaded_image

def load_renderer(texture_res):
    renderer = ian_renderer.Renderer('cuda', 1, (texture_res, texture_res), 'nearest')
    return renderer

# exclued pixels that isn't belong to any triangle
def calculate_inpaint_mask(rendered_mask, triangle_mask):
    for x in range(rendered_mask.shape[1]):
        for y in range(rendered_mask.shape[2]):
            if (triangle_mask[0, x, y, 0] == 0 or rendered_mask[0, x, y, 0] > 0):
                rendered_mask[0, x, y, :] = 1
    return rendered_mask

# image with shape [1, w, h, c]
def show_image(rendered_image):
    pylab.imshow(rendered_image[0].cpu().detach())
    pylab.title(f'rendered_image')
    pylab.waitforbuttonpress(0)
    pylab.cla()

# image with shape [1, w, h, c]
def save_image(rendered_image, folder_path, filename):
    file_path = ian_utils.save_image(torch.clamp(rendered_image, 0., 1.), Path(folder_path), filename)
    print(f'image saved at {file_path}')
    # ian_utils.save_image(torch.clamp(rendered_image, 0., 1.).permute(0, 2, 3, 1), Path(folder_path)/'texture', filename)

def render_front_mesh(data, texture_map, renderer:ian_renderer.Renderer, vertices, faces, uvs, face_uvs, offset = None):
    rendered_image = None
    # body
    rendered_image, mask, soft_mask = renderer.render_image_and_mask_with_camera_params_with_mesh_data(
        elev = data['metadata']['cam_elev'], 
        azim = data['metadata']['cam_azim'] + 90, 
        r = data['metadata']['cam_radius'], 
        look_at_height = data['metadata']['cam_look_at_height'], 
        fovyangle = data['metadata']['cam_fovyangle'] - 30,
        vertices=vertices,
        faces=faces,
        face_uvs=face_uvs,
        sigmainv = 17000,
        texture_map=texture_map,
        offset=offset)
    
    return rendered_image, mask

def mirror_mesh(vertices:torch.Tensor, faces:torch.Tensor, uvs:torch.Tensor, face_uvs:torch.Tensor):
    ##inversed_vertices = vertices.clone().detach() * torch.tensor((1, 1, -1), dtype=vertices.dtype, device=vertices.device, requires_grad=False)
    inversed_vertices = vertices.clone().detach()## * torch.tensor((1, 1, -1), dtype=vertices.dtype, device=vertices.device, requires_grad=False)
    inversed_vertices[:, :, 2] *=  torch.tensor((-1), dtype=vertices.dtype, device=vertices.device, requires_grad=False)
    inversed_faces = faces.clone().detach()
    temp = inversed_faces[:,1].clone().detach()
    inversed_faces[:,1] = inversed_faces[:,2]
    inversed_faces[:,2] = temp

    merged_vertices = torch.cat((vertices.clone().detach(), inversed_vertices), 1)
    merged_faces = torch.cat((faces.clone().detach(), inversed_faces + vertices.shape[1]), 0)
    merged_uvs = torch.cat((uvs, uvs), 1)
    merged_face_uvs = kal.ops.mesh.index_vertices_by_faces(merged_uvs, merged_faces).detach()
    # merged_vertices = torch.cat((vertices.clone().detach(), vertices.clone().detach()), 1)
    # merged_faces = torch.cat((faces, faces), 0)
    # merged_uvs = torch.cat((uvs, uvs), 1)
    # merged_face_uvs = torch.cat((face_uvs, face_uvs), 1)

    print(f'')
    print(f'vertices.shape = {vertices.shape}')
    print(f'faces.shape = {faces.shape}')
    print(f'uvs.shape = {uvs.shape}')
    print(f'face_uvs.shape = {face_uvs.shape}')
    print(f'')
    print(f'merged_vertices.shape = {merged_vertices.shape}')
    print(f'merged_faces.shape = {merged_faces.shape}')
    print(f'merged_uvs.shape = {merged_uvs.shape}')
    print(f'merged_face_uvs.shape = {merged_face_uvs.shape}')
    return (merged_vertices, merged_faces, merged_uvs, merged_face_uvs)


def calculate_valid_texture_pixels(mesh:dict,
                       renderer:ian_renderer.Renderer,
                       data, 
                       texture_res):
    # create dummy texture
    dummy_texture = torch.ones((1, 3, texture_res, texture_res), dtype=torch.float, device='cuda',
                               requires_grad=True)

    # render mesh
    rendered_image, triangle_mask = render_front_mesh(data, dummy_texture, renderer, mesh['vertices'], mesh['faces'], mesh['uvs'], mesh['face_uvs'])
    
    ### Image Losses ###
    loss = torch.mean(torch.abs(rendered_image))
    ### Update the parameters ###
    loss.backward()

    ##torch.set_printoptions(profile="full")
    ##print(f'dummy_texture.grad = {dummy_texture.grad[0]}')
    
    gradient = dummy_texture.grad.clone().cpu()

    valid_pixels = torch.zeros_like(gradient, requires_grad=False)
    for y in range(0, gradient.shape[2]):
        for x in range(0, gradient.shape[3]):
            if (gradient[0, 0, y, x] != 0 or gradient[0, 1, y, x] != 0 or gradient[0, 2, y, x] != 0):
                valid_pixels[0, :, y, x] = 1
    
    return valid_pixels

def merge_front_texture_to_side_texture(front_texture, front_mask, side_texture, side_mask):

    merged_texture = torch.ones_like(front_texture, requires_grad=False)
    merged_mask = torch.zeros_like(front_mask, requires_grad=False)


    for y in range(0, front_texture.shape[2]):
        for x in range(0, front_texture.shape[3]):
            if (side_mask[0, 0, y, x] != 0):
                merged_texture[0, :, y, x] = side_texture[0, :, y, x]
                merged_mask[0, :, y, x] = 1
            elif (front_mask[0, 0, y, x] != 0):
                merged_texture[0, :, y, x] = front_texture[0, :, y, x]
                merged_mask[0, :, y, x] = 1

    return (merged_texture, merged_mask)

def train_texture():
    # load mesh
    (vertices, faces, uvs, face_uvs) = load_mesh(target_folder)
    # load texture and mask
    side_texture = ian_utils.import_rgb(os.path.join(target_folder, 'texture/texture_rgb.png'))
    side_mask = ian_utils.import_rgb(os.path.join(target_folder, 'texture/valid_pixels_rgb.png'))
    # load front_view_rgb
    front_view_rgb = ian_utils.import_rgb(os.path.join(target_folder, 'texture/front_inpainted_rgb.png'))
    # load data
    data = ian_utils.load_rendered_png_and_camera_data(target_folder, 0)
    # load renderer
    renderer = load_renderer(side_texture.shape[0])

    # create mesh mirror
    mirrored_mesh = {}
    (mirrored_mesh['vertices'], mirrored_mesh['faces'], mirrored_mesh['uvs'], mirrored_mesh['face_uvs']) = mirror_mesh(vertices, faces, uvs, face_uvs)

    # load hyperparameter
    hyperparameter = None
    with open(os.path.join(target_folder, f'hyperparameter.json'), 'r') as f:
        hyperparameter = json.load(f)['hyperparameter']

    # create training texture
    training_texture = ian_fish_texture.FishTexture(hyperparameter['texture_res'], 
                                 hyperparameter['texture_lr'],
                                 hyperparameter['scheduler_step_size'],
                                 hyperparameter['scheduler_gamma'])


    # iterations
    texture_train_epoch = 100
    for epoch in range(texture_train_epoch):
        loss = train_texture_iter(mirrored_mesh, renderer, training_texture, front_view_rgb, data, hyperparameter, epoch)
        print(f'epoch {epoch}: loss = {loss}')

    # merge front-view texture into side-view texture
    print(f'    side_texture.shape = {side_texture.shape}')
    print(f'    side_mask.shape = {side_mask.shape}')
    print(f'    front_view_rgb.shape = {front_view_rgb.shape}')
    print(f'    training_texture.texture.shape = {training_texture.texture.shape}')


    front_mask = calculate_valid_texture_pixels(mirrored_mesh, renderer, data, hyperparameter['texture_res'])
    #...
    (merged_texture, merged_mask) = merge_front_texture_to_side_texture(training_texture.texture.cpu(), front_mask.cpu(), side_texture.permute(2, 0, 1).unsqueeze(dim=0).cpu(), side_mask.permute(2, 0, 1).unsqueeze(dim=0).cpu())

    ian_utils.save_image(torch.clamp(training_texture.texture, 0., 1.).permute(0, 2, 3, 1), Path(os.path.join(target_folder, 'texture/')), f'inpainted_texture')
    ian_utils.save_image(torch.clamp(front_mask, 0., 1.).permute(0, 2, 3, 1), Path(os.path.join(target_folder, 'texture/')), f'inpainted_valid_pixels')

    ian_utils.save_image(torch.clamp(merged_texture, 0., 1.).permute(0, 2, 3, 1), Path(os.path.join(target_folder, 'texture/')), f'merged_inpainted_texture')
    ian_utils.save_image(torch.clamp(merged_mask, 0., 1.).permute(0, 2, 3, 1), Path(os.path.join(target_folder, 'texture/')), f'merged_inpainted_valid_pixels')
    
def train_texture_iter(mesh:dict,
                       renderer:ian_renderer.Renderer,
                       training_texture:ian_fish_texture.FishTexture,
                       front_view_rgb,
                       data, 
                       hyperparameter,
                       epoch):
    image_weight = hyperparameter['image_weight']
    
    # jitter the camera to reduce unsampled texture pixels
    jitter_offset_range = 0
    for idx, lr_epoch in enumerate(hyperparameter['texture_jitter_epoch_list']):
        if (epoch >= lr_epoch):
            jitter_offset_range = hyperparameter['texture_jitter_range_list'][idx]
    jitter_offset = torch.tensor((random.normal(0, jitter_offset_range), random.normal(0, jitter_offset_range), random.normal(0, 0)), dtype=torch.float, device='cuda', requires_grad=False)

    # reset texture
    training_texture.zero_grad()

    # render mesh
    rendered_image, triangle_mask = render_front_mesh(data, training_texture.texture, renderer, mesh['vertices'], mesh['faces'], mesh['uvs'], mesh['face_uvs'], offset=jitter_offset)
    
    ### Image Losses ###
    image_loss = torch.mean(torch.abs(rendered_image - front_view_rgb.cuda()))
    loss = image_loss * image_weight

    ### Update the parameters ###
    loss.backward()

    # step
    training_texture.step()

    return loss    

def render_front_image_and_mask():
    # load mesh
    (vertices, faces, uvs, face_uvs) = load_mesh(target_folder)
    # load texture and mask
    texture = load_image(os.path.join(target_folder, 'texture/texture_rgb.png'))
    mask = load_image(os.path.join(target_folder, 'texture/valid_pixels_rgb.png'))
    # load renderer
    renderer = load_renderer(texture.shape[0])
    # load data
    data = ian_utils.load_rendered_png_and_camera_data(target_folder, 0)

    # create mesh mirror
    (vertices2, faces2, uvs2, face_uvs2) = mirror_mesh(vertices, faces, uvs, face_uvs)

    # export rendered texture
    # export rendered mask
    rendered_image, triangle_mask = render_front_mesh(data, texture, renderer, vertices2, faces2, uvs2, face_uvs2)
    rendered_mask, triangle_mask = render_front_mesh(data, mask, renderer, vertices2, faces2, uvs2, face_uvs2)

    inpaint_mask = calculate_inpaint_mask(rendered_mask.detach().cpu(), triangle_mask.detach().cpu())
    save_image(rendered_image, target_folder, 'texture/front_rendered')
    save_image(inpaint_mask, target_folder, 'texture/front_inpaint_mask')
    
    return None

def main():
    global target_folder

    target_folder = Path(input("target dir:"))
    mode = (input("[r]ender or [t]rain?"))

    start_time = time.time()

    if (mode == 'r'):
        render_front_image_and_mask()
    elif (mode == 't'):
        train_texture()
    else:
        print(f'unknown mode: {mode}')

    print(f'Total time elapsed: {time.time() - start_time}s')

if __name__ == "__main__":
    main()
