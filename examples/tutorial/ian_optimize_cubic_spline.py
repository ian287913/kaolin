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
num_epoch = 100
batch_size = 1
y_weight = 10

y_lr = 5e-1
t_lr = 5e-1
# vertice_lr = 5e-4
scheduler_step_size = 20
scheduler_gamma = 0.5


vertices_init = None
vertice_shift = None
nb_faces = None
face_size = None
vertices_laplacian_matrix = None
nb_vertices = None

################################## Prepare Parameters ##################################

gt_key_size = 3
gt_key_xs = torch.as_tensor(np.array([0, 0.5, 1]), dtype=torch.float, device='cuda')
gt_key_ys = torch.as_tensor(np.array([0, -1, 2]), dtype=torch.float, device='cuda')
gt_key_ts = torch.as_tensor(np.array([10, -2, -1]), dtype=torch.float, device='cuda')

key_size = 3
key_xs = None
key_ys = None
key_ts = None


def prepare_loss():
    global key_xs
    global key_ys
    global key_ts

    key_xs = torch.linspace(0, 1, key_size, dtype=torch.float, device='cuda')

    key_ys = torch.zeros((key_size), dtype=torch.float, device='cuda',
                                requires_grad=True)
    key_ts = torch.zeros((key_size), dtype=torch.float, device='cuda',
                                requires_grad=True)
    
key_ys_optim = None
key_ts_optim = None
key_ys_scheduler = None
key_ts_scheduler = None

def setup_optimizer():
    global key_ys_optim
    global key_ts_optim
    global key_ys_scheduler
    global key_ts_scheduler
    key_ys_optim  = torch.optim.Adam(params=[key_ys], lr = y_lr)
    key_ts_optim  = torch.optim.Adam(params=[key_ts], lr = t_lr)

    key_ys_scheduler = torch.optim.lr_scheduler.StepLR(
        key_ys_optim,
        step_size=scheduler_step_size,
        gamma=scheduler_gamma)
    key_ts_scheduler = torch.optim.lr_scheduler.StepLR(
        key_ts_optim,
        step_size=scheduler_step_size,
        gamma=scheduler_gamma)

gt_ys = None
def calculate_groun_truth(sample_xs):
    global gt_ys
    gt_ys = ian_torch_cubic_spline_interp.interp_func_with_tangent(gt_key_xs, gt_key_ys, gt_key_ts, sample_xs)

################################## Training ##################################

sample_size = 100
sample_xs = None

def train():
    global sample_xs
    sample_xs = torch.linspace(0, 1, sample_size, dtype=torch.float, device='cuda')

    calculate_groun_truth(sample_xs)

    for epoch in range(num_epoch):
        if (epoch % 10 == 0):
            visualize_results(sample_xs, epoch)

        loss = train_iter(epoch, sample_xs)
        
        # step scheduler
        key_ys_scheduler.step()
        key_ts_scheduler.step()
        print(f"Epoch {epoch} - loss: {float(loss)}")

    visualize_results(sample_xs, num_epoch)


def train_iter(epoch: int, sample_xs):
    global gt_ys

    # reset gradient
    key_ys_optim.zero_grad()
    key_ts_optim.zero_grad()

    # calculate sampled_Ys
    ys = ian_torch_cubic_spline_interp.interp_func_with_tangent(key_xs, key_ys, key_ts, sample_xs)

    ### Compute Losses ###
    y_loss = torch.mean(torch.abs(gt_ys - ys))
    loss = (y_loss * y_weight)

    ### Update the mesh ###
    loss.backward()
    key_ys_optim.step()
    key_ts_optim.step()

    return loss

def visualize_results(sample_xs, epoch = 0):
    with torch.no_grad():
        ys = ian_torch_cubic_spline_interp.interp_func_with_tangent(key_xs, key_ys, key_ts, sample_xs)

    print(f"key_xs = {key_xs}")
    print(f"key_ys = {key_ys}")
    print(f"key_ts = {key_ts}")

    pylab.scatter(sample_xs.cpu(), gt_ys.cpu(), label='Ground Truth', color='blue')
    pylab.plot(sample_xs.cpu(), ys.cpu(), label='Interpolated curve', color='red')
    pylab.scatter(key_xs.detach().cpu(), key_ys.detach().cpu(), label='Ground Truth', color='red')
    ##pylab.legend()
    ##pylab.show()
    pylab.savefig(f"./optimization record/{epoch}.png")
    pylab.close()


if __name__ == "__main__":
    prepare_loss()
    setup_optimizer()
    train()
    ##visualize_results(sample_xs)