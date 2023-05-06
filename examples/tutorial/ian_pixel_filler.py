import json
import os
import glob
import time
import sys

import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from pathlib import Path
import ian_utils

import kaolin as kal
import numpy as np
import matplotlib.pylab as pylab
import codecs, json

import ian_torch_cubic_spline_interp
import ian_cubic_spline_optimizer
import ian_utils

class PixelFiller:
    # pixel: [w, h, 3]   
    # mask:  [w, h, 3] , black color means invalid
    @staticmethod
    def fill_empty_pixels(pixels : torch.Tensor, mask : torch.Tensor, iter_quota = 10):
        print(f'pixels.shape = {pixels.shape}')
        print(f'mask.shape = {mask.shape}')
        
        assert (pixels.shape[1:2] == mask.shape[1:2]), 'the shape[1:2] of pixels and mask should be the same'
        
        updated_mask = torch.zeros_like(pixels)
        while (iter_quota > 0):
            print(f'remaining iter : {iter_quota}')
            update_flag = False
            updated_mask[:, :, :] = 0

            for x in range(0, pixels.shape[0]):
                for y in range(0, pixels.shape[1]):
                    if (mask[x, y, 0] == 0):
                        # fill the pixel by neighbor
                        if (PixelFiller.calculate_color_by_neighbor(pixels, mask, updated_mask, x, y) == True):
                            update_flag = True

            if (update_flag == True):
                mask = mask + updated_mask
            else:
                break

            iter_quota -= 1

        return pixels
    
    @staticmethod
    def calculate_color_by_neighbor(pixels : torch.Tensor, mask : torch.Tensor, updated_mask : torch.Tensor, x, y):
        valid_neighbors = []
        neighbor_directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        for idx, neighbor_direction in enumerate(neighbor_directions):
            neighbor_x = x + neighbor_direction[0]
            neighbor_y = y + neighbor_direction[1]
            if (neighbor_x >= pixels.shape[0] or 
                neighbor_x < 0 or
                neighbor_y >= pixels.shape[1] or 
                neighbor_y < 0):
                continue
            if (mask[neighbor_x, neighbor_y, 0] != 0):
                valid_neighbors.append(pixels[neighbor_x, neighbor_y, :])
        if (len(valid_neighbors) != 0):
            updated_mask[x, y, 0] = 1
            pixels[x, y, :] = sum(valid_neighbors)/float(len(valid_neighbors))
            return True
        else:
            return False
    
    @staticmethod
    def Fill_pixels(pixels_path:Path, mask_path:Path, name:str, iter_quota):
        pixels = ian_utils.import_rgb(pixels_path)
        mask = ian_utils.import_rgb(mask_path)
        filled_pixels = PixelFiller.fill_empty_pixels(pixels, mask, iter_quota)
        print(f'filled_pixels.shape = {filled_pixels.shape}')
        # 512, 512, 3
        # 1, 512, 512, 3
        # 1, 512, 3, 512
        shaped_filled_pixels = (torch.clamp(filled_pixels, 0., 1.).unsqueeze(0))##.permute(0, 2, 3, 1)
        print(f'shaped_filled_pixels.shape = {shaped_filled_pixels.shape}')
        saved_path = ian_utils.save_image(shaped_filled_pixels, Path('./'), name)
        print(f'image saved as {saved_path}')

if __name__ == '__main__':
    assert (len(sys.argv) == 5), 'usage: python ian_pixel_filler.py <PIXEL_PATH> <MASK_PATH> <OUTPUT_NAME> <ITER_QUOTA>'
    
    start_time = time.time()

    PixelFiller.Fill_pixels(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]))

    end_time = time.time()
    time_lapsed = end_time - start_time
    print(f'Total time elapsed: {time_lapsed}s')
