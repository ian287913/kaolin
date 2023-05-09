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

import ian_torch_cubic_spline_interp
import ian_utils

"""
vertices        (torch.Tensor):     of shape (num_vertices, 3)
faces           (torch.LongTensor): of shape (num_faces, face_size)
uvs             (torch.Tensor):     of shape (num_uvs, 2)
face_uvs_idx    (torch.LongTensor): of shape (num_faces, face_size)
"""
class SplineMesh:
    def __init__(self, dummy):
        print("mesh initialized")
        # # make a dummy mesh data
        # vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
        # faces = [[0, 1, 2], [1, 3, 2]]
        # uvs = [[0, 0], [1, 0], [0, 1], [1, 1]]
        # face_uvs_idx = [[0, 1, 2], [1, 2, 3]]
        # self.set_mesh(vertices, faces, uvs, face_uvs_idx, 512)

    # from list to tensor
    def set_mesh(self, vertices:list, faces:list, uvs:list, face_uvs_idx:list, texture_res):

        # vertices
        self.vertices = torch.FloatTensor([float(el) for sublist in vertices
                                  for el in sublist]).view(-1, 3)
        self.vertices = self.vertices.cuda().unsqueeze(0)
        self.vertices.requires_grad = True

        # uvs
        self.uvs = torch.FloatTensor([float(el) for sublist in uvs
                                 for el in sublist]).view(-1, 2)
        self.uvs = self.uvs.cuda().unsqueeze(0)
        self.uvs.requires_grad = False

        # faces
        ##self.faces = torch.LongTensor(faces) - 1
        self.faces = torch.LongTensor(faces).cuda()

        # face_uvs_idx
        ##self.face_uvs_idx = torch.LongTensor(face_uvs_idx) - 1
        self.face_uvs_idx = torch.LongTensor(face_uvs_idx).cuda()

        # face_uvs
        self.face_uvs = kal.ops.mesh.index_vertices_by_faces(self.uvs, self.face_uvs_idx).detach()
        self.face_uvs.requires_grad = False

        # texture_map
        self.texture_map = torch.ones((1, 3, texture_res, texture_res), dtype=torch.float, device='cuda',
                            requires_grad=True)
        # # print(f"faces.type = {type(faces)}")
        # # print(f"faces.shape = {faces.shape}")

    def set_mesh_by_samples(self, roots: torch.Tensor, ys: torch.Tensor, lod_y, texture_res):
        assert lod_y > 1, f'lod_y should greater than 1!'
        assert roots.shape[0] > 1, f'roots.shape[0] should greater than 1!'
        assert ys.shape[0] > 1, f'ys.shape[0] should greater than 1!'

        unsqueezed_ys = ys.unsqueeze(1)
        direction_x = torch.zeros(ys.shape[0], 
                                  dtype=torch.float, device='cuda', requires_grad=False).unsqueeze(1)
        direction_z = torch.zeros(ys.shape[0], 
                                  dtype=torch.float, device='cuda', requires_grad=False).unsqueeze(1)
        direction_xyz = torch.cat((direction_x, unsqueezed_ys, direction_z), 1)

        # vertices
        self.vertices = torch.zeros((0, 3), dtype=torch.float, device='cuda',
                                requires_grad=True)
        # for each column
        for idx, y in enumerate(ys):
            root_pos = roots[idx]
            # # print(f"y.shape = {y.shape}")
            # # expanded_y = y[None]
            # # print(f"expanded_y.shape = {expanded_y.shape}")
            
            top_pos = roots[idx] + direction_xyz[idx]

            new_vertices = ian_utils.torch_linspace(root_pos, top_pos, lod_y)
            # # print(f"new_vertices.shape = {new_vertices.shape}")
            # # print(f"new_vertices = {new_vertices}")

            # # print(f"before self.vertices.shape = {self.vertices.shape}")
            self.vertices = torch.cat((self.vertices, new_vertices), 0)
            # # print(f"after self.vertices.shape = {self.vertices.shape}\n\n")

        self.vertices = self.vertices.unsqueeze(0)

        # faces
        self.faces = torch.zeros(((lod_y - 1) * (ys.shape[0] - 1) * 2, 3), dtype=torch.long, device='cuda',
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

        # # print(f"self.faces.shape = {self.faces.shape}")
        # # print(f"self.faces = \n{self.faces}")
        

        # uvs
        # TODO: set uvs according to the lod_y and column idx
        self.uvs = torch.zeros((self.vertices.shape[1], 2), dtype=torch.float, device='cuda', requires_grad=False)
        self.uvs = self.uvs.cuda().unsqueeze(0)

        # face_uvs_idx
        self.face_uvs_idx = torch.clone(self.faces).detach()
        self.face_uvs_idx.requires_grad = False

        # face_uvs
        self.face_uvs = kal.ops.mesh.index_vertices_by_faces(self.uvs, self.face_uvs_idx).detach()
        self.face_uvs.requires_grad = False

        # texture_map
        self.texture_map = torch.ones((1, 3, texture_res, texture_res), dtype=torch.float, device='cuda',
                            requires_grad=True)

class Renderer:
    def __init__(self, device, batch_size, render_res=(512, 512), interpolation_mode='bilinear'):
        assert interpolation_mode in ['nearest', 'bilinear', 'bicubic'], f'no interpolation mode {interpolation_mode}'

        self.device = device
        self.interpolation_mode = interpolation_mode
        self.batch_size = batch_size
        self.render_res = render_res
        self.background = torch.ones(render_res).to(device).float()

    # project XYZ world space position to XY camera plane position
    def project_vertices_with_camera_params(self, elev, azim, r, look_at_height, fovyangle, vertices):
        cam_transform = ian_utils.get_camera_transform_from_view(elev, azim, r, look_at_height).cuda()
        cam_proj = ian_utils.get_camera_projection(fovyangle).unsqueeze(0).cuda()

        # pad vertices from XYZ to XYZW
        padded_vertices = torch.nn.functional.pad(
            vertices, (0, 1), mode='constant', value=1.
        )
        vertices_camera = (padded_vertices @ cam_transform)
        # Project the vertices on the camera image plan
        vertices_image = kal.render.camera.perspective_camera(vertices_camera, cam_proj)

        return vertices_image


    def render_image_and_mask_with_camera_params(self, elev, azim, r, look_at_height, fovyangle, mesh, sigmainv = 7000, texture_map = None, offset = None):
        cam_transform = ian_utils.get_camera_transform_from_view(elev, azim, r, look_at_height).cuda()
        cam_proj = ian_utils.get_camera_projection(fovyangle).unsqueeze(0).cuda()
        if (texture_map == None):
            texture_map = mesh.texture_map 
        # render image and mask
        return self.render_image_and_mask(cam_proj, cam_transform, self.render_res[0], self.render_res[1], mesh.vertices, mesh.faces, mesh.get_face_uvs(), sigmainv, texture_map, offset)

    def render_image_and_mask_with_camera_params_with_mesh_data(self, elev, azim, r, look_at_height, fovyangle, vertices, faces, face_uvs, sigmainv = 7000, texture_map = None, offset = None):
        cam_transform = ian_utils.get_camera_transform_from_view(elev, azim, r, look_at_height).cuda()
        cam_proj = ian_utils.get_camera_projection(fovyangle).unsqueeze(0).cuda()
        # render image and mask
        return self.render_image_and_mask(cam_proj, cam_transform, self.render_res[0], self.render_res[1], vertices, faces, face_uvs, sigmainv, texture_map, offset)

    def render_image_and_mask(self, cam_proj, cam_transform, height, width, vertices, faces, face_uvs, sigmainv = 7000, texture_map = None, offset = None):
        if (offset is not None):
            camera_transform = cam_transform + offset
        else:
            camera_transform = cam_transform

        ### Prepare mesh data with projection regarding to camera ###
        face_vertices_camera, face_vertices_image, face_normals = \
            kal.render.mesh.prepare_vertices(
                vertices.repeat(self.batch_size, 1, 1),
                faces,
                cam_proj,
                camera_transform=camera_transform
            )

        ### Perform Rasterization ###
        # Construct attributes that DIB-R rasterizer will interpolate.
        # the first is the UVS associated to each face
        # the second will make a hard segmentation mask
        # ian: torch.ones() makes all pixels inside any triangle get "1" (alpha)
        # ian: which means all the triangles are opaque
        face_attributes = [
            face_uvs.repeat(self.batch_size, 1, 1, 1),
            torch.ones((self.batch_size, faces.shape[0], 3, 1), device='cuda')
        ]

        # print(f"\n\nface_vertices_camera.shape = {face_vertices_camera.shape}")
        # print(f"face_vertices_image.shape = {face_vertices_image.shape}")
        # print(f"face_attributes[0].shape = {face_attributes[0].shape}")
        # print(f"face_attributes[1].shape = {face_attributes[1].shape}")
        # print(f"face_normals.shape = {face_normals.shape}")

        # ian: dibr_rasterization(height, width, faces_z, faces_xy, features, faces_normals)
        image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
            height, width, face_vertices_camera[:, :, :, -1],
            face_vertices_image, face_attributes, face_normals[:, :, -1],
            rast_backend='cuda',
            sigmainv=sigmainv)

        # image_features is a tuple in composed of the interpolated attributes of face_attributes
        texture_coords, mask = image_features
        image = kal.render.mesh.texture_mapping(texture_coords,
                                                texture_map.repeat(self.batch_size, 1, 1, 1), 
                                                mode=self.interpolation_mode)
        image = torch.clamp(image * mask, 0., 1.)

        return (image, mask, soft_mask)
        
####################################################################################################
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
    elev_slider = Slider(ax_elev, 'elev', 0.0, 360.0, 90)
    elev_slider.on_changed(re_render_with_ui_params)

    ax_azim = plt.axes([0.25, 0.3, 0.65, 0.03])
    azim_slider = Slider(ax_azim, 'azim', 0.0, 360, 0)
    azim_slider.on_changed(re_render_with_ui_params)

    ax_radius = plt.axes([0.25, 0.25, 0.65, 0.03])
    radius_slider = Slider(ax_radius, 'radius', 0.0, 20, 4)
    radius_slider.on_changed(re_render_with_ui_params)
    
    ax_sigmainv = plt.axes([0.25, 0.2, 0.65, 0.03])
    sigmainv_slider = Slider(ax_sigmainv, 'sigmainv', 3000.0, 30000.0, 7000)
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
    
    saved_file_path = ian_utils.save_image(rendered_image, output_path, 'rendered_image')
    # ian_utils.save_image(torch.clamp(texture_map, 0., 1.).permute(0, 2, 3, 1), ian_utils.output_path, 'texture')
    print(f"image saved at {saved_file_path}")

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
    new_look_at_height = 0
    new_fovyangle = 60
    print(f"elev = {new_elev}, new_azim = {new_azim}, new_radius = {new_radius}, new_look_at_height = {new_look_at_height}, new_fovyangle = {new_fovyangle}, new_sigmainv = {new_sigmainv}")

    origin_pos = torch.tensor((-1, -1, 0), dtype=torch.float, device='cuda', requires_grad=True)
    length_xyz = torch.tensor((2, 0, 0), dtype=torch.float, device='cuda', requires_grad=True)
    lod = 7

    # roots = calculate_roots(origin_pos, length_xyz, lod)
    # from torchviz import make_dot, make_dot_from_trace
    # g = make_dot(roots, dict(
    #     origin_pos = origin_pos, 
    #     length_xyz = length_xyz, 
    #     roots = roots),
    #     True, True)
    # g.view()

    # gt = calculate_groun_truth(lod)
    # from torchviz import make_dot, make_dot_from_trace
    # g = make_dot(gt, dict(
    #     origin_pos = origin_pos, 
    #     length_xyz = length_xyz, 
    #     gt_key_ys = gt_key_ys,
    #     gt_key_ts = gt_key_ts,
    #     gt = gt),
    #     True, True)
    # g.view()

    newMesh = SplineMesh(123)
    newMesh.set_mesh_by_samples(calculate_roots(origin_pos, length_xyz, lod), calculate_groun_truth(lod), 4, 512)
    print(f"newMesh.vertices.shape = {newMesh.vertices.shape}")
    print(f"newMesh.vertices = \n{newMesh.vertices}")
    
    # # dump graph
    # from torchviz import make_dot, make_dot_from_trace
    # g = make_dot(newMesh.vertices, dict(
    #     origin_pos = origin_pos, 
    #     length_xyz = length_xyz, 
    #     gt_key_ys = gt_key_ys,
    #     gt_key_ts = gt_key_ts,
    #     vertices = newMesh.vertices),
    #     True, True)
    # g.view()

    newRenderer = Renderer('cuda', 1)

    # render
    with torch.no_grad():
        rendered_image, mask, soft_mask = newRenderer.render_image_and_mask_with_camera_params(
            elev = new_elev, 
            azim = new_azim, 
            r = new_radius, 
            look_at_height = new_look_at_height, 
            fovyangle = new_fovyangle,
            mesh = newMesh,
            sigmainv = new_sigmainv)
        
    # update ui image
    ##ax.imshow(soft_mask[0].repeat(3, 1, 1).permute(1,2,0).cpu().detach())
    ax.imshow(rendered_image[0].cpu().detach())
    ##ax.imshow(ian_utils.tensor2numpy(torch.clamp(texture_map, 0., 1.).permute(0,2,3,1))[0])
    fig.canvas.draw_idle()


############################################################################################

gt_key_size = 3
gt_key_xs = torch.as_tensor(np.array([0, 0.5, 1]), dtype=torch.float, device='cuda')
gt_key_ys = torch.as_tensor(np.array([2, 1, 3]), dtype=torch.float, device='cuda')
gt_key_ys.requires_grad = True
gt_key_ts = torch.as_tensor(np.array([10, -2, -1]), dtype=torch.float, device='cuda')
gt_key_ts.requires_grad = True

def calculate_groun_truth(sample_size):
    sample_xs = torch.linspace(0, 1, sample_size, dtype=torch.float, device='cuda')
    
    return ian_torch_cubic_spline_interp.interp_func_with_tangent(gt_key_xs, gt_key_ys, gt_key_ts, sample_xs)

def calculate_roots(origin_pos: torch.Tensor, length_xyz: torch.Tensor, lod):
    start_pos = origin_pos
    end_pos = origin_pos + length_xyz
    ##end_pos = origin_pos + torch.tensor((length_x, 0, 0), dtype=torch.float, device='cuda', requires_grad=False)

    roots = ian_utils.torch_linspace(start_pos, end_pos, lod)

    # # print(f"roots.shape = {roots.shape}")
    # # print(f"roots = {roots}")
    
    return roots

############################################################################################
root_path: Path = Path('./renderer_output/')
output_path : Path
def main():

    
    global root_path
    global output_path

    root_path, output_path = ian_utils.init_path(root_path)

    init_plt()

if __name__ == '__main__':
    main()