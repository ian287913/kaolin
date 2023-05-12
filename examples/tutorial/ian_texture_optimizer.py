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
TRAINING_FOLDER = './dibr_output/20230507_20_12_43 (tuna)/'


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

def load_texture_and_mask(folder_path):
    texture = ian_utils.import_rgb(os.path.join(folder_path, 'texture_rgb.png'))
    texture = texture.permute(2, 0, 1).unsqueeze(0).cuda()
    texture.requires_grad = False
    mask = ian_utils.import_rgb(os.path.join(folder_path, 'valid_pixels_rgb.png'))
    mask = mask.permute(2, 0, 1).unsqueeze(0).cuda()
    mask.requires_grad = False
    return (texture, mask)

def load_renderer(texture):
    renderer = ian_renderer.Renderer('cuda', 1, (texture.shape[2], texture.shape[3]), 'nearest')
    return renderer

def calculate_inpaint_mask(rendered_mask, triangle_mask):
    for x in range(rendered_mask.shape[1]):
        for y in range(rendered_mask.shape[2]):
            if (triangle_mask[0, x, y, 0] == 0):
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
    ian_utils.save_image(torch.clamp(rendered_image, 0., 1.), Path(folder_path), filename)
    # ian_utils.save_image(torch.clamp(rendered_image, 0., 1.).permute(0, 2, 3, 1), Path(folder_path)/'texture', filename)

def render_mesh(data, texture_map, renderer:ian_renderer.Renderer, vertices, faces, uvs, face_uvs):
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
        texture_map=texture_map)
    
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

def train_texture():
    # load mesh
    (vertices, faces, uvs, face_uvs) = load_mesh(TRAINING_FOLDER)
    # load texture and mask
    (texture, mask) = load_texture_and_mask(os.path.join(TRAINING_FOLDER, 'texture'))
    # load renderer
    renderer = load_renderer(texture)
    # load gt rgb
    data = ian_utils.load_rendered_png_and_camera_data(TRAINING_FOLDER, 0)

    # create mesh mirror
    (vertices2, faces2, uvs2, face_uvs2) = mirror_mesh(vertices, faces, uvs, face_uvs)
    # train
    # export texture
    # export mask

    # export rendered texture
    # export rendered mask
    # rendered_image, triangle_mask = render_mesh(data, texture, renderer, vertices, faces, uvs, face_uvs)
    # rendered_mask, triangle_mask = render_mesh(data, mask, renderer, vertices, faces, uvs, face_uvs)
    rendered_image, triangle_mask = render_mesh(data, texture, renderer, vertices2, faces2, uvs2, face_uvs2)
    rendered_mask, triangle_mask = render_mesh(data, mask, renderer, vertices2, faces2, uvs2, face_uvs2)

    inpaint_mask = calculate_inpaint_mask(rendered_mask.detach().cpu(), triangle_mask.detach().cpu())
    save_image(rendered_image, TRAINING_FOLDER, 'front_rendered')
    save_image(inpaint_mask, TRAINING_FOLDER, 'front_inpaint_mask')
    
    return None

if __name__ == "__main__":
    start_time = time.time()
    train_texture()
    end_time = time.time()
    time_lapsed = end_time - start_time
    print(f'Total time elapsed: {time_lapsed}s')
