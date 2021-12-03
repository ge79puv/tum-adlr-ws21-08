import plotting


def plot_paths(q_full, par, number):
    fig, ax = plotting.new_world_fig(limits=par.world.limits, title='Dummy')
    plotting.plot_img_patch_w_outlines(img=par.oc.img, limits=par.world.limits, ax=ax)
    i = 0
    for q in q_full.detach().numpy():
        plotting.plot_x_path(x=q, r=par.robot.spheres_rad, ax=ax, marker='o', alpha=0.5)
        i = i + 1
        if i == number:
            break

