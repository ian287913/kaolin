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


def load_mesh(filepath):
    obj_filepath = filepath

    mesh = kal.io.obj.import_mesh(obj_filepath, with_materials=True)
    vertices = mesh.vertices.cuda().unsqueeze(0)
    print(f"vertices.shape = {vertices.shape}")
    faces = mesh.faces.cuda()
    uvs = mesh.uvs.cuda().unsqueeze(0)
    face_uvs_idx = mesh.face_uvs_idx.cuda()
    face_uvs = kal.ops.mesh.index_vertices_by_faces(uvs, face_uvs_idx).detach()
    face_uvs.requires_grad = False

    return (vertices, faces, uvs, face_uvs)

def mirror_mesh(vertices:torch.Tensor, faces:torch.Tensor, uvs:torch.Tensor, face_uvs:torch.Tensor):
    inversed_vertices = vertices.clone().detach()
    inversed_vertices[:, :, 2] *=  torch.tensor((-1), dtype=vertices.dtype, device=vertices.device, requires_grad=False)
    inversed_faces = faces.clone().detach()
    temp = inversed_faces[:,1].clone().detach()
    inversed_faces[:,1] = inversed_faces[:,2]
    inversed_faces[:,2] = temp

    merged_vertices = torch.cat((vertices.clone().detach(), inversed_vertices), 1)
    merged_faces = torch.cat((faces.clone().detach(), inversed_faces + vertices.shape[1]), 0)
    merged_uvs = torch.cat((uvs, uvs), 1)
    merged_face_uvs = kal.ops.mesh.index_vertices_by_faces(merged_uvs, merged_faces).detach()

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

def main():

    filepath = Path(input("source filename:"))
    output_filepath = Path(input("target filename:"))
    # load mesh
    (vertices, faces, uvs, face_uvs) = load_mesh(filepath)
    # create mesh mirror
    (vertices2, faces2, uvs2, face_uvs2) = mirror_mesh(vertices, faces, uvs, face_uvs)

    mirrored_mesh = {}
    mirrored_mesh['vertices'] = vertices2.squeeze(0)
    mirrored_mesh['faces'] = faces2 + 1
    mirrored_mesh['uvs'] = uvs2.squeeze(0)
    ian_utils.export_mesh(mirrored_mesh, output_filepath)

if __name__ == "__main__":
    main()
