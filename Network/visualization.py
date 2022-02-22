import numpy as np
from matplotlib import pyplot as plt

import forward
from Optimizer import feasibility_check
from chompy import plotting
from obstacle_collision import oc_check, oc_dist


def plot_world(par, name, path=None):
    fig, ax = plotting.new_world_fig(limits=par.world.limits, title=name)
    plotting.plot_img_patch_w_outlines(img=par.oc.img, limits=par.world.limits, ax=ax)
    if path:
        plt.savefig(path + name)


def plot_paths(q_full, par, number, name, path=None, plot_collision=False):
    fig, ax = plotting.new_world_fig(limits=par.world.limits, title=name)
    plotting.plot_img_patch_w_outlines(img=par.oc.img, limits=par.world.limits, ax=ax)

    q_full = q_full.detach().numpy()
    if plot_collision:
        x_spheres = forward.get_x_spheres_substeps(q=q_full, n=par.oc.n_substeps_check, robot=par.robot)
        d = oc_dist(x_spheres=x_spheres, oc=par.oc)

        d_idx = d.argmin(axis=-2)
        d = d.min(axis=-2)  # min over all time steps
        d = d.min(axis=-1)

        feasible = np.array(d > par.oc.dist_threshold)
    n = 0
    for i in range(q_full.shape[0]):
        if n == number:
            break
        q = q_full[i]
        if plot_collision:
            if not feasible[i]:
                plotting.plot_x_path(x=q, r=par.robot.spheres_rad, ax=ax, marker='o', alpha=0.5)
                plotting.plot_circles(x=x_spheres[i][d_idx[i]], r=par.robot.spheres_rad, ax=ax, color='red', alpha=0.5)
                n += 1
        else:
            plotting.plot_x_path(x=q, r=par.robot.spheres_rad, ax=ax, marker='o', alpha=0.5)
            n += 1

    if plot_collision and n < number:
        for i in range(q_full.shape[0]):
            if n == number:
                break
            q = q_full[i]
            if plot_collision:
                if feasible[i]:
                    plotting.plot_x_path(x=q, r=par.robot.spheres_rad, ax=ax, marker='o', alpha=0.5)
                    n += 1

    if path:
        plt.savefig(path + name)
