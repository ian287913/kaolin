import json
import os
import glob
import time

import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from pathlib import Path
import ian_utils

import kaolin as kal

# path to the rendered image (using the data synthesizer)
rendered_path = "./samples/rendered_clock/"
rendered_path_single = "./resources/rendered_clock_fake/"
# path to the output logs (readable with the training visualizer in the omniverse app)
logs_path = './logs/'

# We initialize the timelapse that will store USD for the visualization apps
timelapse = kal.visualize.Timelapse(logs_path) 

############################################################################################

# Hyperparameters
num_epoch = 100
batch_size = 2
laplacian_weight = 0.03
image_weight = 0.1
mask_weight = 1.
texture_lr = 5e-2
vertice_lr = 5e-2
# vertice_lr = 5e-4
scheduler_step_size = 20
scheduler_gamma = 0.5

texture_res = 512

##mesh_path = '../samples/sphere.obj'
mesh_path = './resources/goldfish/example_output_goldfish.obj'

# renderer parameters
render_res = 512
render_elev = 90.0
render_azim = 0.0
render_radius = 18.0
render_look_at_height = 0.0
render_fovyangle = 60.0

render_sigmainv = 7000.0


# select camera angle for best visualization
test_batch_ids = [2, 5, 10]
test_batch_size = len(test_batch_ids)

############################################################################################

train_data = None
dataloader = None
def create_dataloader():
    global train_data
    global dataloader
    global rendered_path

    num_views = len(glob.glob(os.path.join(rendered_path,'*_rgb.png')))
    train_data = []
    for i in range(num_views):
        data = kal.io.render.import_synthetic_view(
            rendered_path, i, rgb=True, semantic=True)
        train_data.append(data)
    print(f"type(train_data) = {type(train_data)}")
    print(f"type(train_data[0]) = {type(train_data[0])}")
    print(f"train_data[0] = {train_data[0]}")
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                            shuffle=True, pin_memory=True) 

############################################################################################

mesh = None
vertices = None
faces = None
face_uvs = None
texture_map = None

def prepare_mesh():
    global mesh
    global vertices
    global faces
    global face_uvs
    global texture_map

    mesh = kal.io.obj.import_mesh(mesh_path, with_materials=True)
    # the sphere is usually too small (this is fine-tuned for the clock)
    vertices = mesh.vertices.cuda().unsqueeze(0) * 0.75
    vertices.requires_grad = True
    print(f"vertices.shape = {vertices.shape}")
    faces = mesh.faces.cuda()
    uvs = mesh.uvs.cuda().unsqueeze(0)
    face_uvs_idx = mesh.face_uvs_idx.cuda()
    # # print(f"faces.type = {type(faces)}")
    # # print(f"faces.shape = {faces.shape}")
    # # print(f"uvs.shape = {uvs.shape}")
    # # print(f"face_uvs_idx.shape = {face_uvs_idx.shape}")

    face_uvs = kal.ops.mesh.index_vertices_by_faces(uvs, face_uvs_idx).detach()
    face_uvs.requires_grad = False
    ##print(f"face_uvs.shape = {face_uvs.shape}")

    # ian: device, dtype, req_grad
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

############################################################################################

vertices_init = None
vertice_shift = None
nb_faces = None
face_size = None
vertices_laplacian_matrix = None
nb_vertices = None

def prepare_loss():
    global vertices_init
    global vertice_shift
    global nb_faces
    global face_size
    global vertices_laplacian_matrix
    global nb_vertices

    ## Separate vertices center as a learnable parameter
    vertices_init = vertices.clone().detach()
    vertices_init.requires_grad = False

    # This is the center of the optimized mesh, separating it as a learnable parameter helps the optimization. 
    vertice_shift = torch.zeros((3,), dtype=torch.float, device='cuda',
                                requires_grad=True)
    

    nb_faces = faces.shape[0]
    nb_vertices = vertices_init.shape[1]
    face_size = 3

    ## Set up auxiliary laplacian matrix for the laplacian loss
    vertices_laplacian_matrix = kal.ops.mesh.uniform_laplacian(
        nb_vertices, faces) 


############################################################################################

vertices_optim = None
texture_optim = None
vertices_scheduler = None
texture_scheduler = None

def setup_optimizer():
    global vertices_optim
    global texture_optim
    global vertices_scheduler
    global texture_scheduler
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

############################################################################################

def train_iter_old_data(epoch: int, data):
    return train_iter(epoch, 
                      True,
                      data['rgb'].cuda(),
                      data['semantic'].cuda(),
                      data['metadata']['cam_transform'].cuda(),
                      data['metadata']['cam_proj'].cuda())

def train_iter_new_data(epoch: int, train_vertices: bool, data):

    # TODO: pre-calculate these matrix when loading data
    cam_proj = ian_utils.get_camera_projection(
        data['metadata']['cam_fovyangle']
    ).cuda()
    cam_transform = ian_utils.get_camera_transform_from_view(
        data['metadata']['cam_elev'],
        data['metadata']['cam_azim'],
        data['metadata']['cam_radius'],
        data['metadata']['cam_look_at_height'],
    ).cuda()
    
    return train_iter(epoch,
                      train_vertices,
                      data['rgb'].cuda(),
                      data['alpha'].cuda(),
                      cam_transform,
                      cam_proj)

def train_iter(epoch: int, train_vertices: bool, gt_image, gt_mask, cam_transform, cam_proj):
    # reset gradient
    texture_optim.zero_grad()
    if (train_vertices):
        vertices_optim.zero_grad()
    
    ### Prepare mesh data with projection regarding to camera ###
    vertices_batch = ian_utils.recenter_vertices(vertices, vertice_shift)
    # ian: vertices_batch.size() = vertices.size() = [batch, num_verts, 3]
    # ian: vertice_shift.size() = [3]

    face_vertices_camera, face_vertices_image, face_normals = \
        kal.render.mesh.prepare_vertices(
            vertices_batch.repeat(batch_size, 1, 1),
            faces, cam_proj, camera_transform=cam_transform
        )
    # print(f"face_vertices_camera.size = {face_vertices_camera.size()}")
    # print(f"face_vertices_image.size = {face_vertices_image.size()}")
    # print(f"face_normals.size = {face_normals.size()}")

    ### Perform Rasterization ###
    # Construct attributes that DIB-R rasterizer will interpolate.
    # the first is the UVS associated to each face
    # the second will make a hard segmentation mask
    # ian: torch.ones() makes all pixels inside any triangle get "1" (alpha)
    # ian: which means all the triangles are opaque
    face_attributes = [
        face_uvs.repeat(batch_size, 1, 1, 1),
        torch.ones((batch_size, nb_faces, 3, 1), device='cuda')
    ]

    # If you have nvdiffrast installed you can change rast_backend to
    # nvdiffrast or nvdiffrast_fwd
    # ian: dibr_rasterization(height, width, faces_z, faces_xy, features, faces_normals)
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
    if (train_vertices):
        batched_gt_mask =  gt_mask.repeat(batch_size, 1, 1)

        mask_loss = kal.metrics.render.mask_iou(soft_mask, 
                                                batched_gt_mask)
        # laplacian loss
        vertices_mov = vertices - vertices_init
        vertices_mov_laplacian = torch.matmul(vertices_laplacian_matrix, vertices_mov)
        laplacian_loss = torch.mean(vertices_mov_laplacian ** 2) * nb_vertices * 3

    if (train_vertices):
        loss = (
            mask_loss * mask_weight +
            laplacian_loss * laplacian_weight
        )
        # loss = (
        #     image_loss * image_weight +
        #     mask_loss * mask_weight +
        #     laplacian_loss * laplacian_weight
        # )
    else:
        loss = (image_loss * image_weight)

    ### Update the mesh ###
    loss.backward()
    texture_optim.step()
    if (train_vertices):
        vertices_optim.step()

    return loss

def train_iter_with_single_view(epoch: int, data):
    texture_optim.zero_grad()

    gt_image = data['rgb'].cuda()

    # TODO: pre-calculate these matrix when loading data
    cam_proj = ian_utils.get_camera_projection(
        data['metadata']['cam_fovyangle']
    ).cuda()
    cam_transform = ian_utils.get_camera_transform_from_view(
        data['metadata']['cam_elev'],
        data['metadata']['cam_azim'],
        data['metadata']['cam_radius'],
        data['metadata']['cam_look_at_height'],
    ).cuda()
    
    # render image and mask
    image, mask, soft_mask = render_image_and_mask(cam_proj, cam_transform, gt_image.shape[1], gt_image.shape[2])
    
    ### Compute Losses ###
    image_loss = torch.mean(torch.abs(image - gt_image))

    loss = (
        image_loss * image_weight
    )

    ### Update the mesh ###
    loss.backward()
    texture_optim.step()

    return loss


def render_image_and_mask_with_camera_params(elev, azim, r, look_at_height, fovyangle, sigmainv = 7000):
        cam_transform = ian_utils.get_camera_transform_from_view(elev, azim, r, look_at_height).cuda()
        cam_proj = ian_utils.get_camera_projection(fovyangle).unsqueeze(0).cuda()
         # render image and mask
        image, mask, soft_mask = render_image_and_mask(cam_proj, cam_transform, render_res, render_res, sigmainv)
        return image, mask, soft_mask

def render_image_and_mask(cam_proj, cam_transform, height, width, sigmainv = 7000):
    ### Prepare mesh data with projection regarding to camera ###
    vertices_batch = ian_utils.recenter_vertices(vertices, vertice_shift)
    # ian: vertices_batch.size() = vertices.size() = [batch, num_verts, 3]
    # ian: vertice_shift.size() = [3]

    face_vertices_camera, face_vertices_image, face_normals = \
        kal.render.mesh.prepare_vertices(
            vertices_batch.repeat(batch_size, 1, 1),
            faces, cam_proj, camera_transform=cam_transform
        )

    ### Perform Rasterization ###
    # Construct attributes that DIB-R rasterizer will interpolate.
    # the first is the UVS associated to each face
    # the second will make a hard segmentation mask
    # ian: torch.ones() makes all pixels inside any triangle get "1" (alpha)
    # ian: which means all the triangles are opaque
    face_attributes = [
        face_uvs.repeat(batch_size, 1, 1, 1),
        torch.ones((batch_size, nb_faces, 3, 1), device='cuda')
    ]

    # ian: dibr_rasterization(height, width, faces_z, faces_xy, features, faces_normals)
    image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
        height, width, face_vertices_camera[:, :, :, -1],
        face_vertices_image, face_attributes, face_normals[:, :, -1],
        rast_backend='cuda',
        sigmainv=sigmainv)

    # image_features is a tuple in composed of the interpolated attributes of face_attributes
    texture_coords, mask = image_features
    image = kal.render.mesh.texture_mapping(texture_coords,
                                            texture_map.repeat(batch_size, 1, 1, 1), 
                                            mode='bilinear')
    image = torch.clamp(image * mask, 0., 1.)

    return (image, mask, soft_mask)


def train():
    for epoch in range(num_epoch):
        for idx, data in enumerate(dataloader):
            loss = train_iter_old_data(epoch, data)

        vertices_scheduler.step()
        texture_scheduler.step()
        print(f"Epoch {epoch} - loss: {float(loss)}")
        
        ### Write 3D Checkpoints ###
        pbr_material = [
            {'rgb': kal.io.materials.PBRMaterial(diffuse_texture=torch.clamp(texture_map[0], 0., 1.))}
        ]

        vertices_batch = ian_utils.recenter_vertices(vertices, vertice_shift)

        # We are now adding a new state of the mesh to the timelapse
        # we only modify the texture and the vertices position
        timelapse.add_mesh_batch(
            iteration=epoch,
            category='optimized_mesh',
            vertices_list=[vertices_batch[0]],
            materials_list=pbr_material
        ) 

def train_with_single_view(train_vertices : bool):
    for epoch in range(num_epoch):
        for idx, data in enumerate(dataloader):
            loss = train_iter_new_data(epoch, train_vertices, data)

        # step scheduler
        texture_scheduler.step()
        if (train_vertices):
            vertices_scheduler.step()
        print(f"Epoch {epoch} - loss: {float(loss)}")
        
        ### Write 3D Checkpoints ###
        pbr_material = [
            {'rgb': kal.io.materials.PBRMaterial(diffuse_texture=torch.clamp(texture_map[0], 0., 1.))}
        ]

        vertices_batch = ian_utils.recenter_vertices(vertices, vertice_shift)

        # We are now adding a new state of the mesh to the timelapse
        # we only modify the texture and the vertices position
        timelapse.add_mesh_batch(
            iteration=epoch,
            category='optimized_mesh',
            vertices_list=[vertices_batch[0]],
            materials_list=pbr_material
        ) 

############################################################################################

# setup uis
fig = None
ax = None
elev_slider = None
azim_slider = None
radius_slider = None
sigmainv_slider = None
def init_plt():
    global fig
    global ax
    global elev_slider
    global azim_slider
    global radius_slider
    global sigmainv_slider

    ## Display the rendered images
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.4)
    #f.subplots_adjust(top=0.99, bottom=0.79, left=0., right=1.4)
    fig.suptitle('DIB-R rendering', fontsize=30)

    # set sliders
    ax_elev = plt.axes([0.25, 0.35, 0.65, 0.03])
    elev_slider = Slider(ax_elev, 'elev', 0.0, 360.0, render_elev)
    elev_slider.on_changed(re_render_with_ui_params)

    ax_azim = plt.axes([0.25, 0.3, 0.65, 0.03])
    azim_slider = Slider(ax_azim, 'azim', 0.0, 360, render_azim)
    azim_slider.on_changed(re_render_with_ui_params)

    ax_radius = plt.axes([0.25, 0.25, 0.65, 0.03])
    radius_slider = Slider(ax_radius, 'radius', 0.0, 20, render_radius)
    radius_slider.on_changed(re_render_with_ui_params)
    
    ax_sigmainv = plt.axes([0.25, 0.2, 0.65, 0.03])
    sigmainv_slider = Slider(ax_sigmainv, 'sigmainv', 3000.0, 30000.0, render_sigmainv)
    sigmainv_slider.on_changed(re_render_with_ui_params)

    ax_render = plt.axes([0.25, 0.1, 0.65, 0.06])
    render_button = Button(ax_render, 'render')
    render_button.on_clicked(render_button_clicked)

    ax_saveimg = plt.axes([0.25, 0.02, 0.65, 0.06])
    saveimg_button = Button(ax_saveimg, 'save img')
    saveimg_button.on_clicked(saveimg_button_clicked)

    plt.show()

def render_button_clicked(val):
    re_render_with_ui_params(None)

def saveimg_button_clicked(val):
    # torch.set_printoptions(profile="full")
    # print(f"texture_map: {texture_map}")
    ian_utils.save_image(rendered_image, output_path, 'rendered_image')
    ian_utils.save_image(torch.clamp(texture_map, 0., 1.).permute(0, 2, 3, 1), output_path, 'texture')
    
    print(f"rendered_image and texture_map saved.")

# called when the ui is updated
rendered_image = None
def re_render_with_ui_params(val):
    global fig
    global ax
    global elev_slider
    global azim_slider
    global radius_slider
    global sigmainv_slider
    global rendered_image

    # fetch ui value
    new_elev = elev_slider.val
    new_azim = azim_slider.val
    new_radius = radius_slider.val
    new_sigmainv = sigmainv_slider.val
    new_look_at_height = render_look_at_height
    new_fovyangle = render_fovyangle
    print(f"elev = {new_elev}, new_azim = {new_azim}, new_radius = {new_radius}, new_look_at_height = {new_look_at_height}, new_fovyangle = {new_fovyangle}, new_sigmainv = {new_sigmainv}")

    # render
    with torch.no_grad():
        rendered_image, mask, soft_mask = render_image_and_mask_with_camera_params(
            elev = new_elev, 
            azim = new_azim, 
            r = new_radius, 
            look_at_height = new_look_at_height, 
            fovyangle = new_fovyangle,
            sigmainv = new_sigmainv)
        
    # update ui image
    print(f"soft_mask.size() = {soft_mask.size()}")
    print(f"soft_mask = {soft_mask}")
    ax.imshow(soft_mask[0].repeat(3, 1, 1).permute(1,2,0).cpu().detach())
    ##ax.imshow(rendered_image[0].cpu().detach())
    ##ax.imshow(ian_utils.tensor2numpy(torch.clamp(texture_map, 0., 1.).permute(0,2,3,1))[0])
    fig.canvas.draw_idle()


def visualize_training():
    with torch.no_grad():
        # This is similar to a training iteration (without the loss part)
        data_batch = [train_data[idx] for idx in test_batch_ids]
        cam_transform = torch.stack([data['metadata']['cam_transform'] for data in data_batch], dim=0).cuda()
        cam_proj = torch.stack([data['metadata']['cam_proj'] for data in data_batch], dim=0).cuda()

        vertices_batch = ian_utils.recenter_vertices(vertices, vertice_shift)

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
            
    plt.show()
    
    ## Display the texture
    plt.figure(figsize=(10, 10))
    plt.title('2D Texture Map', fontsize=30)
    plt.imshow(torch.clamp(texture_map[0], 0., 1.).cpu().detach().permute(1, 2, 0))
    plt.show()

############################ UTILITY ############################

# def export_all_rendered_metadata():
#     for idx, data in enumerate(dataloader):
#         train_iter_with_single_view(epoch, data)

############################################################################################
root_path: Path = Path('./dibr_output/')
output_path : Path
def main():
    global train_data
    global dataloader
    global root_path
    global output_path

    import sys
    print(f"sys.path = {sys.path}\n\n")


    root_path, output_path = ian_utils.init_path(root_path)

    ##create_dataloader()
    train_data, dataloader = ian_utils.create_dataloader_with_single_view(rendered_path_single, 1)

    prepare_mesh()
    prepare_loss()
    setup_optimizer()

    ##train()
    train_with_single_view(True)
    
    ## visualize_training()
    init_plt()

if __name__ == '__main__':
    main()

