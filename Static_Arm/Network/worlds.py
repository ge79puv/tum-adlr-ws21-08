from typing import Dict, Any
import copy

from Network.points_dataset import StartEndPointsDataset
from chompy.GridWorld import create_rectangle_image, create_perlin_image
from chompy.parameter import initialize_oc
import torch
from torch.utils.data import DataLoader

import numpy as np
from matplotlib import pyplot as plt
from chompy import plotting


class Worlds:

    def __init__(self, n_worlds, n_obs, min_max_obstacle_size_voxel, n_voxels, par):

        self.images = None
        self.pars = {}
        self.dataset = {}
        self.dataloader = {}

        self.n_worlds = n_worlds
        self.n_obs = n_obs
        self.par_origin = par
        self.min_max_obstacle_size_voxel = min_max_obstacle_size_voxel
        self.n_voxels = n_voxels

        self.create_worlds()

    def create_worlds(self):
        for i in range(self.n_worlds):
            n_obstacles = np.random.randint(self.n_obs[0], self.n_obs[1])
            img = create_rectangle_image(n=n_obstacles, size_limits=self.min_max_obstacle_size_voxel, n_voxels=self.n_voxels)
            par = copy.deepcopy(self.par_origin)
            initialize_oc(oc=par.oc, world=par.world, robot=par.robot, obstacle_img=img)
            self.pars[str(i)] = par

            img = torch.from_numpy(img[np.newaxis, np.newaxis, :]).float()  # torch.Size([1, 1, 64, 64])
            if i == 0:
                self.images = img
            if i > 0:
                self.images = torch.cat((self.images, img), 0)

    def points_loader(self, n_pairs, batch_size, shuffle=True):
        for i in range(self.n_worlds):
            par = self.pars[str(i)]
            self.dataset[str(i)] = StartEndPointsDataset(n_pairs, par)
            fig, ax = plotting.new_world_fig(limits=par.world.limits)
            plotting.plot_img_patch_w_outlines(img=par.oc.img, limits=par.world.limits, ax=ax)
            # plt.show()

            self.dataloader[str(i)] = DataLoader(self.dataset[str(i)], batch_size, shuffle)

