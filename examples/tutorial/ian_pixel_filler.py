import json
import os
import glob
import time
import sys
import math

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
        # print(f'pixels.shape = {pixels.shape}')
        # print(f'mask.shape = {mask.shape}')
        
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
                print(f'iteration breaked')
                break

            iter_quota -= 1

        return (pixels, mask)
    
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
    def Split_Image(image:torch.Tensor, grid_size):
        # the margin in optimizer is set to 10 pixels
        margin_pixel = 5
        width = image.shape[0] / grid_size
        height = image.shape[1] / grid_size
        x_rects = []
        x_images = []
        for x in range(grid_size):
            y_rects = []
            y_images = []
            for y in range(grid_size):
                rect = {}
                end_x = image.shape[0] - math.floor(width * x)
                start_y = math.floor(width * y)
                start_x = image.shape[0] - math.floor(height * (x+1) - margin_pixel)
                end_y = math.floor(height * (y+1) - margin_pixel)
                rect['start_x'] = start_x
                rect['start_y'] = start_y
                rect['end_x'] = end_x
                rect['end_y'] = end_y
                y_rects.append(rect)
                y_images.append(image[start_x:end_x, start_y:end_y, :])
            x_rects.append(y_rects)
            x_images.append(y_images)
            
        return (x_images, x_rects)
    
    @staticmethod
    def Merge_Image_2(images, background_image, rects):
        #x_offset = 0
        for x in range(len(images)):
            #print(f"x_offset = {x_offset}")
            #y_offset = 0
            for y in range(len(images[x])):
                # print(f"y_offset = {y_offset}")

                # print(f"shape of {x},{y} = {images[x][y].shape}")
                
                # x_end = x_offset + images[x][y].shape[0]
                # y_end = y_offset + images[x][y].shape[1]
                rect = rects[x][y]

                # print(f"overriting at [{x_offset}:{x_end},{y_offset}:{y_end}]")
                print(f"overriting at [{rect['start_x']}:{rect['end_x']},{rect['start_y']}:{rect['end_y']}]")

                #background_image[x_offset:x_end, y_offset:y_end, :] = images[x][y][:, :, :]
                background_image[rect['start_x']:rect['end_x'], rect['start_y']:rect['end_y'], :] = images[x][y][:, :, :]

                #y_offset += images[x][y].shape[1]

            #x_offset += images[x][0].shape[0]
        return background_image


    @staticmethod
    def Merge_Image(images:torch.Tensor):
        merged_ys = []
        for x in range(len(images)):
             merged_ys.append(torch.cat(images[x], dim = 1))
        merged_xs = torch.cat(merged_ys, dim = 0)

        return merged_xs

    @staticmethod
    def Fill_pixels(pixels_path:Path, mask_path:Path, iter_quota, grid_size = 3):
        pixels = ian_utils.import_rgb(pixels_path)
        mask = ian_utils.import_rgb(mask_path)

        splitted_images, image_rects = PixelFiller.Split_Image(pixels, grid_size)
        splitted_masks, mask_rects = PixelFiller.Split_Image(mask, grid_size)

        for x in range(len(splitted_images)):
            for y in range(len(splitted_images[0])):
                print(f'x={x}/{len(splitted_images)}, y={y}/{len(splitted_images[0])}')
                (filled_pixels, filled_mask) = PixelFiller.fill_empty_pixels(splitted_images[x][y], splitted_masks[x][y], iter_quota)
                splitted_images[x][y] = filled_pixels
                splitted_masks[x][y] = filled_mask
        
        
        background_image = pixels.clone()
        background_mask = mask.clone()

        # background_image = torch.ones([pixels.shape[0], pixels.shape[1], pixels.shape[2]], dtype=pixels.dtype, device=pixels.device)
        # background_mask = torch.ones([pixels.shape[0], pixels.shape[1], pixels.shape[2]], dtype=pixels.dtype, device=pixels.device)


        # merged_images = PixelFiller.Merge_Image(splitted_images)
        # merged_masks = PixelFiller.Merge_Image(splitted_masks)
        merged_images = PixelFiller.Merge_Image_2(splitted_images, background_image, image_rects)
        merged_masks = PixelFiller.Merge_Image_2(splitted_masks, background_mask, mask_rects)


        print(f'merged_images.shape = {merged_images.shape}')
        print(f'merged_masks.shape = {merged_masks.shape}')
        # 512, 512, 3
        # 1, 512, 512, 3
        # 1, 512, 3, 512
        shaped_filled_pixels = (torch.clamp(merged_images, 0., 1.).unsqueeze(0))##.permute(0, 2, 3, 1)
        shaped_filled_mask = (torch.clamp(merged_masks, 0., 1.).unsqueeze(0))
        print(f'shaped_filled_pixels.shape = {shaped_filled_pixels.shape}')

        return (shaped_filled_pixels, shaped_filled_mask)
        # saved_path = ian_utils.save_image(shaped_filled_pixels, Path('./'), name)
        # ian_utils.save_image(shaped_filled_mask, Path('./'), name + "_mask")
        # print(f'image saved as {saved_path}')


if __name__ == '__main__':
    # assert (len(sys.argv) == 5), 'usage: python ian_pixel_filler.py <PIXEL_PATH> <MASK_PATH> <OUTPUT_NAME> <ITER_QUOTA>'
    # PixelFiller.Fill_pixels(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]))
    
    working_dir = Path(input("working dir:"))
    iter_quota = int(input("iter quota:"))

    pixel_path = working_dir / Path('texture_rgb.png')
    mask_path = working_dir / Path('valid_pixels_rgb.png')

    # fill pixels
    start_time = time.time()
    (filled_pixels, filled_mask) = PixelFiller.Fill_pixels(pixel_path, mask_path, iter_quota)
    print(f'Total time elapsed: {time.time() - start_time}s')

    # save image
    saved_path = ian_utils.save_image(filled_pixels, working_dir, f'filled_texture_{iter_quota}')
    print(f'image saved at {saved_path}')
    saved_path = ian_utils.save_image(filled_mask, working_dir, f'filled_valid_pixels_{iter_quota}')
    print(f'image saved at {saved_path}')

