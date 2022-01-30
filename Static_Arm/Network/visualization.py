from matplotlib import pyplot as plt

from chompy import plotting


def plot_paths(q_full, par, number, name, path=None, save=False):
    fig, ax = plotting.new_world_fig(limits=par.world.limits, title=name)
    plotting.plot_img_patch_w_outlines(img=par.oc.img, limits=par.world.limits, ax=ax)
    i = 0
    for q in q_full.detach().numpy():
        plotting.plot_x_path(x=q, r=par.robot.spheres_rad, ax=ax, marker='o', alpha=0.5)
        i = i + 1
        if i == number:
            break
    if save:
        plt.savefig(path + name)


def plot_spheres(q_full, par, number, name, robot, path=None, save=False):
    j = 0
    for q in q_full.detach().numpy():  # q_full: (number_path, waypoints, dof)  torch.Size([50, 8, 3])
        '''
        fig, ax = plotting.new_world_fig(limits=par.world.limits, title=name)
        plotting.plot_img_patch_w_outlines(img=par.oc.img, limits=par.world.limits, ax=ax)
        plotting.plot_spheres(q=q[0], robot=robot, ax=ax, color='r')  # start
        plotting.plot_spheres(q=q[q.shape[0] - 1], robot=robot, ax=ax)  # end
        plt.show()
        '''
        fig, ax = plotting.new_world_fig(limits=par.world.limits, title=name)
        plotting.plot_img_patch_w_outlines(img=par.oc.img, limits=par.world.limits, ax=ax)

        for i in range(q.shape[0]):
            if i == 0 or i == q.shape[0]-1:
                plotting.plot_spheres(q=q[i], robot=robot, ax=ax, color='red')
            plotting.plot_spheres(q=q[i], robot=robot, ax=ax)

        if save:
            plt.savefig(path + (name + "_path_" + str(j)) )
        j = j + 1
        if j == number:
            break

