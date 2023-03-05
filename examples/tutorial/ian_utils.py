import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
from pathlib import Path
from PIL import Image

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
