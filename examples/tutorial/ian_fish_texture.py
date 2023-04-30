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
import numpy as np
import matplotlib.pylab as pylab
import codecs, json

import ian_torch_cubic_spline_interp
import ian_cubic_spline_optimizer
import ian_utils
import ian_renderer

class FishTexture:
    # pixel: [w, h, 3]   mask: [w, h, 3]
    @staticmethod
    def fill_empty_pixels(pixels : torch.Tensor, mask : torch.Tensor):
        assert (pixels.shape[1:2] == mask.shape[1:2]), 'the shape[1:2] of pixels and mask should be the same'
        
        iter_quota = 10
        updated_mask = torch.zeros_like(pixels)
        while (iter_quota > 0):
            update_flag = False
            updated_mask[:, :, :] = 0

            for x in range(0, pixels.shape[0]):
                for y in range(0, pixels.shape[1]):
                    if (mask[x, y, 0] == 0):
                        # fill the pixel by neighbor
                        if (FishTexture.calculate_color_by_neighbor(pixels, mask, updated_mask, x, y) == True):
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

    def __init__(self, 
                 texture_res, texture_lr = 5e-1, scheduler_step_size = 20, scheduler_gamma = 0.5,
                 ):
        
        self.dirty = True

        # parameters
        self.texture = torch.ones((1, 3, texture_res, texture_res), dtype=torch.float, device='cuda',
                            requires_grad=True)
        # Hyperparameters
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        self.texture_lr = texture_lr
        # initialize optimizers and schedulers
        self.texture_optim  = torch.optim.Adam(params=[self.texture], lr = self.texture_lr)
        self.texture_scheduler = torch.optim.lr_scheduler.StepLR(
            self.texture_optim,
            step_size=self.scheduler_step_size,
            gamma=self.scheduler_gamma)
        
        self.optimizer_parameters = {}
        self.optimizer_parameters['texture_lr'] = self.texture_lr
        self.optimizer_parameters['scheduler_step_size'] = self.scheduler_step_size
        self.optimizer_parameters['scheduler_gamma'] = self.scheduler_gamma

    def zero_grad(self):
        self.texture_optim.zero_grad()

    def step(self, step_splines = True):
        self.dirty = True
        self.texture_optim.step()
        self.texture_scheduler.step()
    
    def export_texture(self, path, name):
        ian_utils.save_image(torch.clamp(self.texture, 0., 1.).permute(0, 2, 3, 1), path, name)
