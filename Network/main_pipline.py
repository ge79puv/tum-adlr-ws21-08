import numpy as np
import torch
from matplotlib import pyplot as plt

import plotting
from GridWorld import create_rectangle_image
from Kinematic.Robots import SingleSphere02
from Network.loss_function import chompy_partial_loss
from Network.network import Backbone2D, Dummy
from Optimizer.obstacle_collision import oc_check2
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
start_end_number_train = 100
start_end_number_test = 100

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


start_points = sample_points(start_end_number_train, Dof, img, world_limits)
end_points = sample_points(start_end_number_train, Dof, img, world_limits)
start_end_points = torch.flatten(torch.cat((start_points, end_points), 1))  # (start_end_number * 2 * dof)

# ========================== Neural Network ================================
# model = Dummy(start_end_number_train * 2 * Dof, start_end_number_train * (n_waypoints - 2) * Dof)
model = Backbone2D(start_end_number_train * 2 * Dof, start_end_number_train * (n_waypoints - 2) * Dof)
model.to(device)
# print("Model: ")
# print(model)
# for parameter in model.parameters():
#     print(parameter.shape)
#     print(parameter.data)
# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)
# optimizer = torch.optim.Adam(model.parameters())
# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

# =============================== Training ====================================
train_loss_history = []  # loss
train_feasible_history = []
for epoch in range(101):

    q = model(start_end_points)

    q_reshaped = q.reshape(start_end_number_train, n_waypoints - 2, Dof)
    q_full = torch.cat((start_points[:, None, :], q_reshaped, end_points[:, None, :]), 1)

    length_cost, collision_cost, length_jac, collision_jac = chompy_partial_loss(q_full.detach().numpy(), par)

    train_loss_history.append(length_cost.sum() + collision_cost.sum())
    train_feasible_history.append(oc_check2(q_full.detach().numpy(), par.robot, par.oc).sum() / start_end_number_train)
    if epoch % 10 == 0:
        print(epoch, "train loss: ", train_loss_history[-1], "feasible rate: ", train_feasible_history[-1])

    temp = (50 * q * torch.flatten(collision_jac) + q * torch.flatten(length_jac)).sum()
    optimizer.zero_grad()
    temp.backward()
    optimizer.step()

print('FINISH.')
# torch.save(model.state_dict(), "./model")

# ============================ visualisation ============================
fig, ax = plotting.new_world_fig(limits=par.world.limits, title='Dummy')
plotting.plot_img_patch_w_outlines(img=par.oc.img, limits=par.world.limits, ax=ax)
q_full = torch.cat((start_points[:, None, :],
                    model(start_end_points).reshape(start_end_number_train, n_waypoints - 2, Dof),
                    end_points[:, None, :]),
                   1)
for q in q_full.detach().numpy():
    plotting.plot_x_path(x=q, r=par.robot.spheres_rad, ax=ax, marker='o', alpha=0.5)
plt.show()

# =========================== validation =================================
start_points = sample_points(start_end_number_test, Dof, img, world_limits)
end_points = sample_points(start_end_number_test, Dof, img, world_limits)
start_end_points = torch.flatten(torch.cat((start_points, end_points), 1))  # (start_end_number * 2 * dof)
q_full = torch.cat((start_points[:, None, :],
                    model(start_end_points).reshape(start_end_number_test, n_waypoints - 2, Dof),
                    end_points[:, None, :]),
                   1)
length_cost, collision_cost, length_jac, collision_jac = chompy_partial_loss(q_full.detach().numpy(), par)
print("test loss: ", length_cost.sum() + collision_cost.sum(),
      "feasible rate: ", oc_check2(q_full.detach().numpy(), par.robot, par.oc).sum() / start_end_number_test)

fig, ax = plotting.new_world_fig(limits=par.world.limits, title='Dummy')
plotting.plot_img_patch_w_outlines(img=par.oc.img, limits=par.world.limits, ax=ax)
for q in q_full.detach().numpy():
    plotting.plot_x_path(x=q, r=par.robot.spheres_rad, ax=ax, marker='o', alpha=0.5)
plt.show()
