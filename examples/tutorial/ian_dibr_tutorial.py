import json
import os
import glob
import time

from PIL import Image
import torch
import numpy as np
from matplotlib import pyplot as plt

import kaolin as kal

# path to the rendered image (using the data synthesizer)
rendered_path = "../samples/rendered_clock/"
# path to the output logs (readable with the training visualizer in the omniverse app)
logs_path = './logs/'

# We initialize the timelapse that will store USD for the visualization apps
timelapse = kal.visualize.Timelapse(logs_path) 

# Hyperparameters
num_epoch = 50
batch_size = 2
laplacian_weight = 0.03
image_weight = 0.1
mask_weight = 1.
texture_lr = 5e-2
vertice_lr = 5e-4
scheduler_step_size = 20
scheduler_gamma = 0.5

texture_res = 400

# select camera angle for best visualization
test_batch_ids = [2, 5, 10]
test_batch_size = len(test_batch_ids)

num_views = len(glob.glob(os.path.join(rendered_path,'*_rgb.png')))
train_data = []
for i in range(num_views):
    data = kal.io.render.import_synthetic_view(
        rendered_path, i, rgb=True, semantic=True)
    train_data.append(data)

print(num_views)

dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                         shuffle=True, pin_memory=True) 

mesh = kal.io.obj.import_mesh('../samples/sphere.obj', with_materials=True)
# the sphere is usually too small (this is fine-tuned for the clock)
vertices = mesh.vertices.cuda().unsqueeze(0) * 0.75
vertices.requires_grad = True
faces = mesh.faces.cuda()
uvs = mesh.uvs.cuda().unsqueeze(0)
face_uvs_idx = mesh.face_uvs_idx.cuda()


face_uvs = kal.ops.mesh.index_vertices_by_faces(uvs, face_uvs_idx).detach()
face_uvs.requires_grad = False

texture_map = torch.ones((1, 3, texture_res, texture_res), dtype=torch.float, device='cuda',
                         requires_grad=True)

# The topology of the mesh and the uvs are constant
# so we can initialize them on the first iteration only
timelapse.add_mesh_batch(
    iteration=0,
    category='optimized_mesh',
    faces_list=[mesh.faces.cpu()],
    uvs_list=[mesh.uvs.cpu()],
    face_uvs_idx_list=[mesh.face_uvs_idx.cpu()],
)

## Separate vertices center as a learnable parameter
vertices_init = vertices.clone().detach()
vertices_init.requires_grad = False

# This is the center of the optimized mesh, separating it as a learnable parameter helps the optimization. 
vertice_shift = torch.zeros((3,), dtype=torch.float, device='cuda',
                            requires_grad=True)

def recenter_vertices(vertices, vertice_shift):
    """Recenter vertices on vertice_shift for better optimization"""
    vertices_min = vertices.min(dim=1, keepdim=True)[0]
    vertices_max = vertices.max(dim=1, keepdim=True)[0]
    vertices_mid = (vertices_min + vertices_max) / 2
    vertices = vertices - vertices_mid + vertice_shift
    return vertices


nb_faces = faces.shape[0]
nb_vertices = vertices_init.shape[1]
face_size = 3

## Set up auxiliary laplacian matrix for the laplacian loss
vertices_laplacian_matrix = kal.ops.mesh.uniform_laplacian(
    nb_vertices, faces) 

vertices_optim  = torch.optim.Adam(params=[vertices, vertice_shift],
                                   lr=vertice_lr)
texture_optim = torch.optim.Adam(params=[texture_map], lr=texture_lr)
vertices_scheduler = torch.optim.lr_scheduler.StepLR(
    vertices_optim,
    step_size=scheduler_step_size,
    gamma=scheduler_gamma)
texture_scheduler = torch.optim.lr_scheduler.StepLR(
    texture_optim,
    step_size=scheduler_step_size,
    gamma=scheduler_gamma)

for epoch in range(num_epoch):
    for idx, data in enumerate(dataloader):
        vertices_optim.zero_grad()
        texture_optim.zero_grad()
        gt_image = data['rgb'].cuda()
        gt_mask = data['semantic'].cuda()
        cam_transform = data['metadata']['cam_transform'].cuda()
        cam_proj = data['metadata']['cam_proj'].cuda()
        
        ### Prepare mesh data with projection regarding to camera ###
        vertices_batch = recenter_vertices(vertices, vertice_shift)

        face_vertices_camera, face_vertices_image, face_normals = \
            kal.render.mesh.prepare_vertices(
                vertices_batch.repeat(batch_size, 1, 1),
                faces, cam_proj, camera_transform=cam_transform
            )

        ### Perform Rasterization ###
        # Construct attributes that DIB-R rasterizer will interpolate.
        # the first is the UVS associated to each face
        # the second will make a hard segmentation mask
        face_attributes = [
            face_uvs.repeat(batch_size, 1, 1, 1),
            torch.ones((batch_size, nb_faces, 3, 1), device='cuda')
        ]

        # If you have nvdiffrast installed you can change rast_backend to
        # nvdiffrast or nvdiffrast_fwd
        image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
            gt_image.shape[1], gt_image.shape[2], face_vertices_camera[:, :, :, -1],
            face_vertices_image, face_attributes, face_normals[:, :, -1],
            rast_backend='cuda')

        # image_features is a tuple in composed of the interpolated attributes of face_attributes
        texture_coords, mask = image_features
        image = kal.render.mesh.texture_mapping(texture_coords,
                                                texture_map.repeat(batch_size, 1, 1, 1), 
                                                mode='bilinear')
        image = torch.clamp(image * mask, 0., 1.)
        
        ### Compute Losses ###
        image_loss = torch.mean(torch.abs(image - gt_image))
        mask_loss = kal.metrics.render.mask_iou(soft_mask,
                                                gt_mask.squeeze(-1))
        # laplacian loss
        vertices_mov = vertices - vertices_init
        vertices_mov_laplacian = torch.matmul(vertices_laplacian_matrix, vertices_mov)
        laplacian_loss = torch.mean(vertices_mov_laplacian ** 2) * nb_vertices * 3

        loss = (
            image_loss * image_weight +
            mask_loss * mask_weight +
            laplacian_loss * laplacian_weight
        )
        ### Update the mesh ###
        loss.backward()
        vertices_optim.step()
        texture_optim.step()

    vertices_scheduler.step()
    texture_scheduler.step()
    print(f"Epoch {epoch} - loss: {float(loss)}")
    
    ### Write 3D Checkpoints ###
    pbr_material = [
        {'rgb': kal.io.materials.PBRMaterial(diffuse_texture=torch.clamp(texture_map[0], 0., 1.))}
    ]

    vertices_batch = recenter_vertices(vertices, vertice_shift)

    # We are now adding a new state of the mesh to the timelapse
    # we only modify the texture and the vertices position
    timelapse.add_mesh_batch(
        iteration=epoch,
        category='optimized_mesh',
        vertices_list=[vertices_batch[0]],
        materials_list=pbr_material
    ) 



with torch.no_grad():
    # This is similar to a training iteration (without the loss part)
    data_batch = [train_data[idx] for idx in test_batch_ids]
    cam_transform = torch.stack([data['metadata']['cam_transform'] for data in data_batch], dim=0).cuda()
    cam_proj = torch.stack([data['metadata']['cam_proj'] for data in data_batch], dim=0).cuda()

    vertices_batch = recenter_vertices(vertices, vertice_shift)

    face_vertices_camera, face_vertices_image, face_normals = \
        kal.render.mesh.prepare_vertices(
            vertices_batch.repeat(test_batch_size, 1, 1),
            faces, cam_proj, camera_transform=cam_transform
        )
    face_attributes = [
        face_uvs.repeat(test_batch_size, 1, 1, 1),
        torch.ones((test_batch_size, nb_faces, 3, 1), device='cuda'),
    ]

    image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
        256, 256, face_vertices_camera[:, :, :, -1],
        face_vertices_image, face_attributes, face_normals[:, :, -1])

    texture_coords, mask = image_features
    image = kal.render.mesh.texture_mapping(texture_coords,
                                            texture_map.repeat(test_batch_size, 1, 1, 1), 
                                            mode='bilinear')
    image = torch.clamp(image * mask, 0., 1.)
    
    ## Display the rendered images
    f, axarr = plt.subplots(1, test_batch_size, figsize=(7, 22))
    f.subplots_adjust(top=0.99, bottom=0.79, left=0., right=1.4)
    f.suptitle('DIB-R rendering', fontsize=30)
    for i in range(test_batch_size):
        axarr[i].imshow(image[i].cpu().detach())
        
## Display the texture
plt.figure(figsize=(10, 10))
plt.title('2D Texture Map', fontsize=30)
plt.imshow(torch.clamp(texture_map[0], 0., 1.).cpu().detach().permute(1, 2, 0))

