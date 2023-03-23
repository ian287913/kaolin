import json
import os
import glob
import time

import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pylab as pylab
from matplotlib.widgets import Slider, Button, TextBox
from pathlib import Path

import kaolin as kal

import ian_torch_cubic_spline_interp

# Hyperparameters
y_weight = 100
y_lr = 5
t_lr = 5
scheduler_step_size = 20
scheduler_gamma = 0.5

################################## Class Definition ##################################

class CubicSplineOptimizer:
    def __init__(self, key_size = 3, y_lr = 5e-1, t_lr = 5e-1, scheduler_step_size = 20, scheduler_gamma = 0.5, init_key_ys = 1.0):
        # Hyperparameters
        self.y_lr = y_lr
        self.t_lr = t_lr
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma

        # initialize keys
        self.key_xs = torch.linspace(0, 1, key_size, dtype=torch.float, device='cuda',
                                     requires_grad=False)
        
        self.key_ys = torch.ones(key_size, dtype=torch.float, device='cuda',
                                    requires_grad=False) * init_key_ys
        self.key_ys.requires_grad = True

        self.key_ts = torch.zeros(key_size, dtype=torch.float, device='cuda',
                                    requires_grad=False)
        self.key_ts.requires_grad = True
        
        self.parameters = {}
        self.parameters['key_xs'] = self.key_xs
        self.parameters['key_ys'] = self.key_ys
        self.parameters['key_ts'] = self.key_ts

        # initialize optimizers and schedulers
        self.key_ys_optim  = torch.optim.Adam(params=[self.key_ys], lr = self.y_lr)
        self.key_ts_optim  = torch.optim.Adam(params=[self.key_ts], lr = self.t_lr)
        self.key_ys_scheduler = torch.optim.lr_scheduler.StepLR(
            self.key_ys_optim,
            step_size=self.scheduler_step_size,
            gamma=self.scheduler_gamma)
        self.key_ts_scheduler = torch.optim.lr_scheduler.StepLR(
            self.key_ts_optim,
            step_size=self.scheduler_step_size,
            gamma=self.scheduler_gamma)
    '''
    optimization related funcitons
    should call by this order: 
    1. zero_grad()
    2. calculate_ys()
    3. loss.backward() outside the class w.r.t. the tensors returned by calculate_ys()
    4. step()
    '''
    def zero_grad(self):
        self.key_ys_optim.zero_grad()
        self.key_ts_optim.zero_grad()
    def calculate_ys_with_lod_x(self, lod_x):
        sample_xs = torch.linspace(0, 1, lod_x, dtype=torch.float, device='cuda', requires_grad=False)
        return self.calculate_ys(sample_xs)
    def calculate_ys(self, sample_xs):
        return ian_torch_cubic_spline_interp.interp_func_with_tangent(self.key_xs, self.key_ys, self.key_ts, sample_xs)
    
    def calculate_negative_ys_loss(self, lod_x):
        ys = self.calculate_ys_with_lod_x(lod_x)
        loss = torch.exp(-ys)
        return torch.mean(loss)

    def step(self):
        # # try to print grad
        # print(f"self.key_ys_optim.param_groups = {self.key_ys_optim.param_groups}")
        # print(f"type = {type(self.key_ys_optim.param_groups)}")
        # print(f"type(param_groups[0]['params']) = {type(self.key_ys_optim.param_groups[0]['params'])}")

        self.key_ys_optim.step()
        self.key_ts_optim.step()
        self.key_ys_scheduler.step()
        self.key_ts_scheduler.step()


################################## Utils ##################################

def calculate_ground_truth(sample_xs):
    gt_key_size = 3
    gt_key_xs = torch.as_tensor(np.array([0, 0.5, 1]), dtype=torch.float, device='cuda')
    gt_key_ys = torch.as_tensor(np.array([0, -1, 2]), dtype=torch.float, device='cuda')
    gt_key_ts = torch.as_tensor(np.array([10, -2, -1]), dtype=torch.float, device='cuda')
    return ian_torch_cubic_spline_interp.interp_func_with_tangent(gt_key_xs, gt_key_ys, gt_key_ts, sample_xs)

def train_spline():
    # parameters
    num_epoch = 100
    visualize_epoch_interval = 20
    sample_size = 20
    key_size = 5

    spline_optimizer = CubicSplineOptimizer(
        key_size, 
        y_lr=y_lr, 
        t_lr=t_lr, 
        scheduler_step_size=scheduler_step_size, 
        scheduler_gamma=scheduler_gamma)
    
    sample_xs = torch.linspace(0, 1, sample_size, dtype=torch.float, device='cuda')

    gt_ys = calculate_ground_truth(sample_xs)

    for epoch in range(num_epoch):
        if (epoch % visualize_epoch_interval == 0):
            visualize_results(spline_optimizer, sample_xs, epoch)

        spline_optimizer.zero_grad()
        ###ys = spline_optimizer.calculate_ys(sample_xs)
        ys = spline_optimizer.calculate_ys_with_lod_x(sample_size)

        ### Compute Losses ###
        y_loss = torch.mean(torch.abs(gt_ys - ys))
        loss = (y_loss * y_weight)

        # # make dot
        # if (epoch % visualize_epoch_interval == 0):
        #     from torchviz import make_dot
        #     g = make_dot(loss, dict(key_ys = spline_optimizer.key_ys, key_ts = spline_optimizer.key_ts, y_loss = y_loss, output = loss))
        #     g.view()

        ### Update the parameters ###
        loss.backward()
        spline_optimizer.step()
        print(f"Epoch {epoch} - loss: {float(loss)}")

    visualize_results(spline_optimizer, sample_xs, num_epoch)

def visualize_results(spline_optimizer:CubicSplineOptimizer, sample_xs, epoch = 0):
    with torch.no_grad():
        ys = spline_optimizer.calculate_ys(sample_xs)
    gt_ys = calculate_ground_truth(sample_xs)

    pylab.scatter(sample_xs.cpu(), gt_ys.cpu(), label='Ground Truth', color='blue')
    pylab.plot(sample_xs.cpu(), ys.cpu(), label='Interpolated Curve', color='red')
    ##pylab.scatter(key_xs.detach().cpu(), key_ys.detach().cpu(), color='red')
    pylab.legend()
    pylab.title(f'epoch: {epoch}')
    pylab.waitforbuttonpress(0)
    pylab.cla()
    
    ##pylab.close()

    ##pylab.savefig(f"./optimization record/{epoch}.png")
    ##pylab.close()

if __name__ == "__main__":
    train_spline()
