from matplotlib import pyplot as plt
from chompy import plotting


def plot_spheres(q_full, par, number, name, robot, path=None, save=False):
    k = 0
    for q in q_full.detach().numpy():
        fig, ax = plotting.new_world_fig(limits=par.world.limits, title=name)
        plotting.plot_img_patch_w_outlines(img=par.oc.img, limits=par.world.limits, ax=ax)

        for i in range(q.shape[0]):
            if i == 0 or i == q.shape[0]-1:
                plotting.plot_spheres(q=q[i], robot=robot, ax=ax, color='red')
            plotting.plot_spheres(q=q[i], robot=robot, ax=ax)

        if save:
            plt.savefig(path + (name + "_path_" + str(k)) )
        k = k + 1
        if k == number:
            break

