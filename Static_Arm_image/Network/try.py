import numpy as np
import matplotlib.pyplot as plt
from chompy.Kinematic.Robots import SingleSphere02, StaticArm
from chompy.parameter import Parameter
from chompy.Optimizer import feasibility_check
from chompy.GridWorld import create_rectangle_image
from chompy.parameter import Parameter, initialize_oc
import chompy.plotting

np.random.seed(15)
radius = 0.3

robot = StaticArm(n_dof=3, limb_lengths=1.2, radius=0.1)
par = Parameter(robot=robot, obstacle_img='rectangle')


# par.world.limits = np.array([[-4, 4],
                         # [-4, 4]])

print(par.robot.limits)
print(par.world.limits)
# Sample random configurations
q = robot.sample_q((3, 2))

'''
# Forward Kinematic, get frames and position of the spheres
f = robot.get_frames(q)
x = robot.get_x_spheres(q)

# x[0][0] = [0, 0]
# x[0][15] = [0, 0]
print("x[0]", x[0])
q[0][0] = [0.5, 0.5, 0.5]
q[0][1] = [0.8, 0.8, 0.8]
x, dx_dq = robot.get_x_spheres_jac(q)
print("x[0][0]", x[0][0])
'''

n_obstacles = 2
min_max_obstacle_size_voxel = [15, 15]
n_voxels = (64, 64)
img = create_rectangle_image(n=n_obstacles, size_limits=min_max_obstacle_size_voxel, n_voxels=n_voxels)
initialize_oc(oc=par.oc, world=par.world, robot=par.robot, obstacle_img=img)

print(q.shape)
status = feasibility_check(q, par)
print(status)

fig, ax = chompy.plotting.new_world_fig(limits=par.world.limits,)
chompy.plotting.plot_img_patch_w_outlines(img=par.oc.img, limits=par.world.limits, ax=ax)
for i in range(q[0].shape[0]):
    chompy.plotting.plot_spheres(q=q[2][i], robot=robot, ax=ax)

plt.title('1')
plt.show()

