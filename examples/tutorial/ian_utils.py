import os
import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
from pathlib import Path
from PIL import Image
import json
import glob

import kaolin as kal

############################ Path ############################

def init_path(root_path: Path):
    root_path = make_path(root_path)
    output_path = make_path(root_path / 'output_images')
    return root_path, output_path

def make_path(path: Path) -> Path:
    path.mkdir(exist_ok=True,parents=True)
    return path

############################ Combine Meshes ############################
def combine_meshes(meshes:list):
    combined_mesh = {}
    combined_mesh['vertices'] = []
    combined_mesh['faces'] = []
    combined_mesh['uvs'] = []
    accumulated_idx = 1
    for mesh in meshes:
        combined_mesh['vertices'].extend(mesh.vertices.squeeze(0).detach().cpu().numpy().tolist())
        combined_mesh['faces'].extend((mesh.faces + accumulated_idx).detach().cpu().numpy().tolist())
        combined_mesh['uvs'].extend(mesh.reshaped_uvs.squeeze(0).detach().cpu().numpy().tolist())
        accumulated_idx += len(mesh.vertices.squeeze(0))
    return combined_mesh

def export_mesh(mesh, path):
    obj_str = ""
    for vertex in mesh['vertices']:
        obj_str += f'v {vertex[0]} {vertex[1]} {vertex[2]}\n'
    for uv in mesh['uvs']:
        obj_str += f'vt {uv[0]} {uv[1]}\n'
    for face in mesh['faces']:
        obj_str += f'f {face[0]}/{face[0]} {face[1]}/{face[1]} {face[2]}/{face[2]}\n'
    
    with open(path, 'w') as f:
        f.write(obj_str)

############################ recenter vertices ############################

def recenter_vertices(vertices, vertice_shift):
    """Recenter vertices on vertice_shift for better optimization"""
    vertices_min = vertices.min(dim=1, keepdim=True)[0]
    vertices_max = vertices.max(dim=1, keepdim=True)[0]
    vertices_mid = (vertices_min + vertices_max) / 2
    vertices = vertices - vertices_mid + vertice_shift
    return vertices

############################ Camera transform & projection ############################

def get_camera_transform_from_view(elev, azim, r=3.0, look_at_height=0.0):
    elev = np.deg2rad(elev)
    azim = np.deg2rad(azim)

    if (not torch.is_tensor(elev)):
        elev = torch.tensor(elev)
    if (not torch.is_tensor(azim)):
        azim = torch.tensor(azim)
    if (not torch.is_tensor(r)):
        r = torch.tensor(r)
    if (not torch.is_tensor(look_at_height)):
        look_at_height = torch.tensor(look_at_height)

    x = r * torch.sin(elev) * torch.sin(azim)
    y = r * torch.cos(elev)
    z = r * torch.sin(elev) * torch.cos(azim)

    pos = torch.tensor([x, y, z]).unsqueeze(0)

    look_at = torch.zeros_like(pos)
    look_at[:, 1] = look_at_height
    up_direction = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0)

    pos = pos.float()
    look_at = look_at.float()
    up_direction = up_direction.float()

    camera_trans = kal.render.camera.generate_transformation_matrix(pos, look_at, up_direction)
    return camera_trans

def get_camera_projection(fovyangle):
    fovyangle = np.deg2rad(fovyangle)

    if (not torch.is_tensor(fovyangle)):
        fovyangle = torch.tensor(fovyangle)

    camera_proj = kal.render.camera.generate_perspective_projection(fovyangle).to('cuda')
    return camera_proj

############################ torch linspace ############################

# from https://github.com/pytorch/pytorch/issues/61292
def torch_linspace(start: torch.Tensor, stop: torch.Tensor, num: int):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
    """
    # create a tensor of 'num' steps from 0 to 1
    steps = torch.arange(num, dtype=torch.float32, device=start.device, requires_grad=False) / (num - 1)
    
    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
    # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but torchscript
    #   "cannot statically infer the expected size of a list in this contex", hence the code below
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)
    
    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps*(stop - start)[None]
    
    return out

############################ Save Image ############################

def tensor2numpy(tensor:torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().numpy()
    tensor = (tensor * 255).astype(np.uint8)
    return tensor

def save_image(tensor_image:torch.Tensor, save_path, save_name):
    numpy_image = tensor2numpy(tensor_image[0])
    file_path = save_path / f"{save_name}_rgb.png"
    Image.fromarray(numpy_image).save(file_path)
    return file_path

############################ Load Image & Metadata ############################

def import_rgb(path: Path):
    if os.path.exists(path):
        return torch.from_numpy(
            np.array(Image.open(path))
        )[:, :, :3].float() / 255.
    else:
        print(f"failed to load rgb: file {path} does not exist.")
        return None

# actually it's getting the 'R' channel
def import_alpha(path: Path):
    if os.path.exists(path):
        return torch.from_numpy(
            np.array(Image.open(path))
        )[:, :, 0].float() / 255.
    else:
        print(f"failed to load alpha: file {path} does not exist.")
        return None

def import_segment_mask(path: Path):
    if os.path.exists(path):
        return torch.from_numpy(
            np.array(Image.open(path))
        )[:, :, :3].float() / 255.
    else:
        print(f"failed to load body_mask: file {path} does not exist.")
        return None

def import_root_segmentation(path: Path, data_idx):
    if not os.path.exists(path):
        return None
    
    with open(path, 'r') as f:
        froot_segs = json.load(f)
        parsed_root_segs = {}
        for key, value in froot_segs.items():
            parsed_key = key
            if (parsed_key.endswith('.png')):
                parsed_key = parsed_key[:-4]
            if (parsed_key.startswith(f'{data_idx}_')):
                parsed_key = parsed_key[len(f'{data_idx}_'):]
            parsed_root_segs[parsed_key] = value
    return parsed_root_segs

def load_rendered_png_and_camera_data(root_dir: Path, data_idx: int = 0):
    # resolution follows 'rgb'
    output = {}
    rgb = import_rgb(os.path.join(root_dir, f'{data_idx}_rgb.png'))
    if (rgb is not None):
        output['rgb'] = rgb
    alpha = import_alpha(os.path.join(root_dir, f'{data_idx}_alpha.png'))
    if (alpha is not None):
        output['alpha'] = alpha
    body_mask = import_segment_mask(os.path.join(root_dir, f'{data_idx}_body_mask.png'))
    if (body_mask is not None):
        output['body_mask'] = body_mask
    dorsal_fin_mask = import_segment_mask(os.path.join(root_dir, f'{data_idx}_dorsal_fin_mask.png'))
    if (dorsal_fin_mask is not None):
        output['dorsal_fin_mask'] = dorsal_fin_mask
    caudal_fin_mask = import_segment_mask(os.path.join(root_dir, f'{data_idx}_caudal_fin_mask.png'))
    if (caudal_fin_mask is not None):
        output['caudal_fin_mask'] = caudal_fin_mask
    anal_fin_mask = import_segment_mask(os.path.join(root_dir, f'{data_idx}_anal_fin_mask.png'))
    if (anal_fin_mask is not None):
        output['anal_fin_mask'] = anal_fin_mask
    pelvic_fin_mask = import_segment_mask(os.path.join(root_dir, f'{data_idx}_pelvic_fin_mask.png'))
    if (pelvic_fin_mask is not None):
        output['pelvic_fin_mask'] = pelvic_fin_mask
    pectoral_fin_mask = import_segment_mask(os.path.join(root_dir, f'{data_idx}_pectoral_fin_mask.png'))
    if (pectoral_fin_mask is not None):
        output['pectoral_fin_mask'] = pectoral_fin_mask
    
    root_segmentation = import_root_segmentation(os.path.join(root_dir, 'marked_roots.json'), data_idx)
    if (root_segmentation is not None):
        output['root_segmentation'] = root_segmentation
    

    with open(os.path.join(root_dir, f'{data_idx}_metadata.json'), 'r') as f:
        fmetadata = json.load(f)
        output['metadata'] = {
            'cam_elev': fmetadata['cam_elev'],
            'cam_azim': fmetadata['cam_azim'],
            'cam_radius': fmetadata['cam_radius'],
            'cam_look_at_height': fmetadata['cam_look_at_height'],
            'cam_fovyangle': fmetadata['cam_fovyangle'],
        }

    print(f"loaded metadata = {output['metadata']}")
    return output

# ian: the format of this 'metadata' is different with the original file generated by Omniverse
def create_dataloader_with_single_view(from_path:str, batch_size = 1):

    num_views = len(glob.glob(os.path.join(from_path,'*_rgb.png')))
    train_data = []
    for i in range(num_views):
        data = load_rendered_png_and_camera_data(from_path, i)
        train_data.append(data)
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                            shuffle=True, pin_memory=True) 
    return train_data, dataloader

def convert_tensor_dict(d:dict):
    converted_dict = {}

    for k,v in d.items():        
        if isinstance(v, dict):
            converted_dict[k] = convert_tensor_dict(v)
        elif (torch.is_tensor(v)):            
            converted_dict[k] = v.detach().cpu().numpy().tolist()
        elif isinstance(v, np.ndarray):
            converted_dict[k] = v.tolist()
        else:
            print(f"UNKNOWN TYPE of {k}: {type(v)}")
            converted_dict[k] = v
    return converted_dict
    