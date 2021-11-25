import numpy as np
import torch
from matplotlib import pyplot as plt

import plotting
from GridWorld import create_rectangle_image
from Kinematic.Robots import SingleSphere02
from Network.loss_function import chompy_partial_loss
from Network.network import Backbone2D, Dummy
from parameter import Parameter, initialize_oc

np.random.seed(2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# ======================== Initialization and Configuration ======================
radius = 0.3  # Size of the robot [m]
robot = SingleSphere02(radius=radius)

world_limits = np.array([[0, 10],  # x [m]
                         [0, 10]])  # y [m]
n_voxels = (64, 64)

par = Parameter(robot=robot, obstacle_img='rectangle')
par.robot.limits = world_limits
n_waypoints = 20
Dof = 2
start_end_number = 100

n_obstacles = 5
min_max_obstacle_size_voxel = [3, 15]

img = create_rectangle_image(n=n_obstacles, size_limits=min_max_obstacle_size_voxel, n_voxels=n_voxels)
initialize_oc(oc=par.oc, world=par.world, robot=par.robot, obstacle_img=img)


# plt.imshow(img)
# plt.show()

# ====================== Sample start and end points ========================
def sample_points(number, dof, image, limits):
    def sample(invalid):
        q_attempt = np.random.rand(invalid, dof)
        q_attempt_voxel = (q_attempt * image.shape).astype(int)
        mask = image[q_attempt_voxel[:, 0], q_attempt_voxel[:, 1]]
        if mask.sum() > 0:
            new = sample(mask.sum())
            q_attempt[mask] = new
        else:
            return q_attempt
        return q_attempt

    q_0 = sample(number)
    q_sampled = limits[:, 0] + q_0 * limits[:, 1]
    return torch.FloatTensor(q_sampled)


start_points = sample_points(start_end_number, Dof, img, world_limits)
end_points = sample_points(start_end_number, Dof, img, world_limits)
start_end_points = torch.flatten(torch.cat((start_points, end_points), 1))  # (start_end_number * 2 * dof)

# ========================== Neural Network ================================
model = Dummy(start_end_number * 2 * Dof, start_end_number * (n_waypoints - 2) * Dof)
# model = Backbone2D(start_end_number * 2 * Dof, start_end_number * (n_waypoints - 2) * Dof)
model.to(device)
# print("Model: ")
# print(model)
# for parameter in model.parameters():
#     print(parameter.shape)
#     print(parameter.data)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
q = model(start_end_points)

# =============================== Training ====================================
train_loss_history = []  # loss
for epoch in range(300):
    optimizer.zero_grad()
    q = model(start_end_points)

    q_reshaped = q.reshape(start_end_number, n_waypoints - 2, Dof)
    q_full = torch.cat((start_points[:, None, :], q_reshaped, end_points[:, None, :]), 1)

    length_cost, collision_cost, length_jac, collision_jac = chompy_partial_loss(q_full.detach().numpy(), par)
    temp = (q * torch.flatten(collision_jac) + q * torch.flatten(length_jac)).sum()
    temp.backward()

    train_loss_history.append(length_cost.sum() + collision_cost.sum())
    optimizer.step()

print(train_loss_history)
print('FINISH.')

# ============================ visualisation ============================
fig, ax = plotting.new_world_fig(limits=par.world.limits, title='Dummy')
plotting.plot_img_patch_w_outlines(img=par.oc.img, limits=par.world.limits, ax=ax)
q_full = torch.cat((start_points[:, None, :],
                    model(start_end_points).reshape(start_end_number, n_waypoints - 2, Dof),
                    end_points[:, None, :]),
                   1)
for q in q_full.detach().numpy():
    plotting.plot_x_path(x=q, r=par.robot.spheres_rad, ax=ax, marker='o', alpha=0.5)
plt.show()
