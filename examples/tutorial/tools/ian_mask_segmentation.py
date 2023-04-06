import numpy as np
import matplotlib.pyplot as plt
from mpl_point_clicker import clicker
from mpl_interactions import ioff, panhandler, zoom_factory
from matplotlib.widgets import Button
from matplotlib.widgets import Slider
import glob
import os

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
    axnext = fig.add_axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(switch_mask)

    # slider
    axslider = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    opaque_slider = Slider(
        ax=axslider,
        label='Opacity',
        valmin=0.0,
        valmax=1.0,
        valinit=0.5,
    )
    opaque_slider.on_changed(update_image)


    plt.show() 

def switch_mask(event):
    global current_image_idx
    current_image_idx += 1
    if (set_current_image(current_image_idx) == True):
        update_image(None)
    else:
        print("end of masks.")

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
    global axesimg

    ##ax.cla() # this will kill the klicker!
    combined_image = current_image + (overlay_image * opaque_slider.val)

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

    # get all masks
    mask_dict = load_masks(loading_dir, mask_name)
    overlay_image = load_image(os.path.join(loading_dir, rgb_name))
    for mask_name in mask_dict:
        print(f"maskname: {mask_name}")

    set_current_image(0)

    init_plot()



    print(klicker.get_positions())

    print("end of main")


if __name__ == "__main__":
    main()