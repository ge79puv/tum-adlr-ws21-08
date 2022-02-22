from typing import Dict, Any
import copy

from GridWorld import obstacle_img2dist_img
from Network.dataset_points import StartEndPointsDataset, filter_non_collision
from Optimizer import feasibility_check
from chompy.GridWorld import create_rectangle_image, create_perlin_image
from chompy.parameter import initialize_oc
import torch
from torch.utils.data import DataLoader, Dataset

import numpy as np
from numpy.random import RandomState
from matplotlib import pyplot as plt
from chompy import plotting


class Worlds:

    def __init__(self, n_worlds, n_obs, min_max_obstacle_size_voxel, n_voxels, par):
        self.images = None
        self.dist_images = None
        self.rectangles = None
        self.pars = {}
        self.points_dataset = {}
        self.points_loader = {}

        self.n_worlds = n_worlds
        self.n_obs = n_obs
        self.par_origin = par
        self.min_max_obstacle_size_voxel = min_max_obstacle_size_voxel
        self.n_voxels = n_voxels

        self.create_worlds()

    def create_a_world(self):
        img, (rec_pos, rec_size) = create_rectangle_image(n=self.n_obs,
                                                          size_limits=self.min_max_obstacle_size_voxel,
                                                          n_voxels=self.n_voxels, return_rectangles=True)
        par = copy.deepcopy(self.par_origin)
        initialize_oc(oc=par.oc, world=par.world, robot=par.robot, obstacle_img=img)

        dist_img = torch.from_numpy(obstacle_img2dist_img(
            img=img, voxel_size=[1 / self.n_voxels[0], 1 / self.n_voxels[1]])[np.newaxis, np.newaxis, :]).float()
        img = torch.from_numpy(img[np.newaxis, np.newaxis, :]).float()  # torch.Size([1, 1, 64, 64])
        # rec = torch.cat((torch.from_numpy(rec_pos), torch.from_numpy(rec_size)), 1)
        rec_pos = torch.from_numpy(rec_pos)[np.newaxis, :]

        return par, img, dist_img, rec_pos

    def create_worlds(self):
        for i in range(self.n_worlds):
            par, img, dist_img, rec_pos = self.create_a_world()
            self.pars[str(i)] = par

            if i == 0:
                self.images = img
                self.dist_images = dist_img
                self.rectangles = rec_pos
            elif i > 0:
                self.images = torch.cat((self.images, img), 0)
                self.dist_images = torch.cat((self.dist_images, dist_img), 0)
                self.rectangles = torch.cat((self.rectangles, rec_pos), 0)

        self.images = ImagesDataset(self.n_worlds, self.images)
        self.dist_images = ImagesDataset(self.n_worlds, self.dist_images)
        self.rectangles = ImagesDataset(self.n_worlds, self.rectangles)

    def create_points_loader(self, n_pairs, batch_size, add_collision=None, shuffle=True):
        for i in range(self.n_worlds):
            par = self.pars[str(i)]
            SE = StartEndPointsDataset(n_pairs, par)

            if add_collision:
                SEc = StartEndPointsDataset(add_collision, par)
                waypoints = np.linspace(SEc.start_points, SEc.end_points, 10).swapaxes(0, 1)
                status = feasibility_check(waypoints[:, 1:-1, :], SEc.par)
                new_start_points, new_end_points = filter_non_collision(
                    SEc.start_points[status == 1], SEc.end_points[status == 1], SEc.par)
                SEc.start_points[status == 1] = new_start_points
                SEc.end_points[status == 1] = new_end_points
                SE.add_data(SEc)

            self.points_dataset[str(i)] = SE
            self.points_loader[str(i)] = DataLoader(self.points_dataset[str(i)], batch_size, shuffle)


class ImagesDataset(Dataset):
    def __init__(self, number, images):
        self.number = number
        self.images = images

    def __len__(self):
        return self.number

    def __getitem__(self, item):
        return self.images[item], item

    def __setitem__(self, key, value):
        self.images[key] = value


