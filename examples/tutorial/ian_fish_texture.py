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
