import numpy as np
import matplotlib.pyplot as plt
from mpl_point_clicker import clicker
from mpl_interactions import ioff, panhandler, zoom_factory
from matplotlib.widgets import Button
from matplotlib.widgets import Slider
import glob
import os
import json
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import ian_utils

# the axis: upper-left = (0,0) lower-right = (1,1)

rgb_name = "0_rgb.png"
mask_name = "*_mask.png"
loading_dir = "../resources/rendered_goldfish"

mask_dict = {}
current_image_idx = 0
current_image = None
overlay_image = None
fig:plt.Figure = None
ax:plt.Axes = None
klicker:clicker = None
opaque_slider:Slider = None
axesimg = None

marked_roots = {}

def load_image(path):
    loaded_image = plt.imread(path)
    if (loaded_image.shape[-1] > 3):
        loaded_image = loaded_image[:, :, 0:3]
    return loaded_image

def load_masks(dir, mask_name):
    mask_paths = glob.glob(os.path.join(dir,mask_name))
    mask_dict = {}
    for mask_path in mask_paths:
        base_name = os.path.basename(mask_path)
        loaded_mask = load_image(mask_path)
        mask_dict[base_name] = loaded_mask
    return mask_dict

def init_plot():
    global fig
    global ax
    global klicker
    global opaque_slider

    # set plot
    fig, ax = plt.subplots()
    ax.imshow(current_image)

    # set interactions
    zoom_factory(ax)
    ph = panhandler(fig, button=2)
    klicker = clicker(ax, ["start", "end"], markers=["o", "x"])

    # button
    axnext = fig.add_axes([0.85, 0.5, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(switch_mask)

    # slider
    plt.subplots_adjust(bottom=0.2, left=0.1, top=0.9)
    axslider = fig.add_axes([0.2, 0.05, 0.55, 0.05])

    last_opaque_val = 0.5
    if (opaque_slider is not None):
        last_opaque_val = opaque_slider.val
    opaque_slider = Slider(
        ax=axslider,
        label='Opacity',
        valmin=0.0,
        valmax=1.0,
        valinit=last_opaque_val,
    )
    opaque_slider.on_changed(update_image)

    update_image(None)

    plt.show() 

def switch_mask(event):
    global current_image_idx
    global marked_roots

    # validate selection
    points = klicker.get_positions()['start']
    if (len(points) < 2):
        print("failed to save with less than 2 points!")
        return
    
    # save and reset selected points
    marked_roots[list(mask_dict.keys())[current_image_idx]] = points
    klicker.set_positions({'start':[], 'end':[]})

    current_image_idx += 1
    if (set_current_image(current_image_idx) == True):
        update_image(None)
    else:
        print("end of masks.")
        plt.close()
        save_segmentation()

def save_segmentation():
    global marked_roots
    image_width = list(mask_dict.values())[0].shape[0]
    image_height = list(mask_dict.values())[0].shape[1]
    for roots in marked_roots.values():
        roots[:,0] = roots[:,0] / image_width
        roots[:,1] = 1 - (roots[:,1] / image_height)
    converted_export_dict = ian_utils.convert_tensor_dict(marked_roots)

    filepath = os.path.join(loading_dir,'marked_roots.json')
    with open(filepath, 'w') as fp:
        json.dump(converted_export_dict, fp, indent=4)
    print(f'file exported to {filepath}.')


def set_current_image(idx):
    global current_image
    global mask_dict
    if (idx >= len(mask_dict.keys())):
        return False
    else:
        current_image = mask_dict[list(mask_dict.keys())[idx]]
        return True

def update_image(event):
    global current_image
    global overlay_image
    global fig
    global axesimg

    ##ax.cla() # this will kill the klicker!
    combined_image = (current_image * (1 - opaque_slider.val)) + (overlay_image * opaque_slider.val)
    combined_image = combined_image.clip(max=1.0)

    if (axesimg is None):
        axesimg = ax.imshow(combined_image)
    else:
        axesimg.set_array(combined_image)
    plt.draw()

def main():
    global current_image
    global overlay_image
    global loading_dir
    global mask_dict
    global mask_name
    global fig
    global ax

    if (len(sys.argv) > 1):
        loading_dir = sys.argv[1]

    # get all masks
    mask_dict = load_masks(loading_dir, mask_name)
    overlay_image = load_image(os.path.join(loading_dir, rgb_name))
    for mask_name in mask_dict:
        print(f"maskname: {mask_name}")

    set_current_image(0)
    
    init_plot()

    print("end of main")


if __name__ == "__main__":
    main()