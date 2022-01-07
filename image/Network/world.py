from chompy.GridWorld import create_rectangle_image, create_perlin_image
from chompy.parameter import initialize_oc
import torch
import numpy as np
from matplotlib import pyplot as plt
from chompy import plotting


def world(number, par, n_voxels):
    for i in range(number):
        n_obstacles = 5 + i
        min_max_obstacle_size_voxel = [5, 15]

        img = create_rectangle_image(n=n_obstacles, size_limits=min_max_obstacle_size_voxel, n_voxels=n_voxels)
        initialize_oc(oc=par.oc, world=par.world, robot=par.robot, obstacle_img=img)
        fig, ax = plotting.new_world_fig(limits=par.world.limits)
        plotting.plot_img_patch_w_outlines(img=par.oc.img, limits=par.world.limits, ax=ax)
        plt.show()

        img = img[np.newaxis, :]   # (1,64,64)
        img = torch.from_numpy(img[np.newaxis, :]).float()  # torch.Size([1, 1, 64, 64])
        if i == 0:
            temp = img
        if i > 0:
            temp = torch.cat((temp, img), 0)

    return temp   # torch.Size([number, 1, 64, 64])


