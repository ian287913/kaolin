import os
import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
from pathlib import Path
from PIL import Image
import json

############################ Path ############################

root_path: Path = Path('./dibr_output/')
output_path: Path
def init_path():
    global root_path
    global output_path

    root_path = make_path(root_path)
    output_path = make_path(root_path / 'output_images')

def make_path(path: Path) -> Path:
    path.mkdir(exist_ok=True,parents=True)
    return path

############################ Save Image ############################

def tensor2numpy(tensor:torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().numpy()
    tensor = (tensor * 255).astype(np.uint8)
    return tensor

def save_image(tensor_image:torch.Tensor, save_path, save_name):
    numpy_image = tensor2numpy(tensor_image[0])
    Image.fromarray(numpy_image).save(save_path / f"{save_name}_rgb.png")

def import_png(path: Path):
    if os.path.exists(path):
        return torch.from_numpy(
            np.array(Image.open(path))
        )[:, :, :3].float() / 255.
    else:
        print(f"ERROR: file {path} does not exist.")
        return None

def load_rendered_png_and_camera_data(root_dir: Path, data_idx: int = 0):
    # resolution follows 'rgb'
    output = {}
    output['rgb'] = import_png(os.path.join(root_dir, f'{data_idx}_rgb.png'))

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
