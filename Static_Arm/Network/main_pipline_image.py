import os

import numpy as np
import torch
from matplotlib import pyplot as plt

from NURBS import NURBS
from worlds import Worlds
from chompy.Kinematic.Robots import SingleSphere02
from helper import Processing
from loss_function import chompy_partial_loss
from network import PlanFromImage
from visualization import plot_paths
from chompy.Optimizer.obstacle_collision import oc_check2
from chompy.parameter import Parameter

# ======================== Initialization and Configuration ======================
np.random.seed(10)
device = torch.device("cpu")
print(device)

radius = 0.3  # Size of the robot [m]
robot = SingleSphere02(radius=radius)
par = Parameter(robot=robot, obstacle_img='rectangle')

world_limits = np.array([[0, 10],  # x [m]
                         [0, 10]])  # y [m]
proc = Processing(world_limits)
par.robot.limits = world_limits
# par.oc.n_substeps = 5
# par.oc.n_substeps_check = 5

n_waypoints = 20
u = np.linspace(0.01, 0.99, n_waypoints)
n_control_points = 1  # ToDo
degree = 3
Dof = par.robot.n_dof

save_image = True
plot_path = './plot/images/test/'
os.makedirs(plot_path, exist_ok=True)

# ============================ Worlds and Points =============================

n_obstacles = [5, 10]
min_max_obstacle_size_voxel = [3, 15]
n_voxels = (64, 64)

n_worlds_train = 10
start_end_number_train = 500  # in every world
train_batch_size = 50
worlds_train = Worlds(n_worlds_train, n_obstacles, min_max_obstacle_size_voxel, n_voxels, par)
worlds_train.points_loader(start_end_number_train, train_batch_size, shuffle=True)

# n_worlds_test = 2
# start_end_number_test = 100
# test_batch_size = 50
# worlds_test = Worlds(n_worlds_test, n_obstacles, min_max_obstacle_size_voxel, n_voxels, par)
# worlds_test.points_loader(start_end_number_test, test_batch_size, shuffle=False)

# ========================== Neural Network ================================
model = PlanFromImage(n_voxels, 2 * Dof, n_control_points, Dof, ctrlpts=True)
model.to(device)
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# ================================ Optimizer ================================
# TODO: choose optimizer and corresponding hyperparameters
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # TODO: adaptive learning rate
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

# =============================== Training ====================================
weight = np.array([1, 5])
repeat = 0
min_test_loss = np.inf

early_stop = 20
train_loss_history = []
train_feasible_history = []
test_loss_history = []
test_feasible_history = []

print("Use both length jac and collision jac...")

for epoch in range(501):

    # training
    model.train()
    train_loss, train_feasible = 0, 0
    worlds = np.random.permutation(range(worlds_train.n_worlds))
    for i in worlds:
        for _, (start_points_train, end_points_train) in enumerate(worlds_train.dataloader[str(i)]):
            pairs_train = torch.cat((proc.preprocessing(start_points_train), proc.preprocessing(end_points_train)),
                                    1)  # (number, 2 * dof)
            # print(worlds_train.images.shape)  torch.Size([10, 1, 64, 64])
            control_points, control_point_weights = model(worlds_train.images, pairs_train)
            q_p = torch.cat((start_points_train[:, None, :],
                             proc.postprocessing(control_points),
                             end_points_train[:, None, :]), 1)

            nurbs = NURBS(p=q_p, degree=degree, w=control_point_weights, u=u)
            q = nurbs.evaluate()
            q_full = torch.cat((start_points_train[:, None, :], q, end_points_train[:, None, :]), 1)
            length_cost, collision_cost, length_jac, collision_jac = \
                chompy_partial_loss(q_full.detach().numpy(), worlds_train.pars[str(i)])
            # print("length_cost", len(length_cost))  [50]
            do_dq = weight[0] * length_cost.mean() * length_jac / np.sqrt(length_cost[:, None, None]) + weight[
                1] * collision_jac
            dq_dp = nurbs.evaluate_jac()
            do_dp = torch.matmul(torch.moveaxis(dq_dp, 1, 2), do_dq.float())

            train_loss += (weight[0] * length_cost + weight[1] * collision_cost).mean()
            train_feasible += oc_check2(q_full.detach().numpy(), worlds_train.pars[str(i)].robot,
                                        worlds_train.pars[str(i)].oc, verbose=0).mean()

            temp = (do_dp[:, 1:-1, :] * control_points).mean()
            optimizer.zero_grad()
            temp.backward()
            optimizer.step()

    train_loss_history.append(train_loss / len(worlds_train.dataloader[str(0)]) / worlds_train.n_worlds)
    train_feasible_history.append(train_feasible / len(worlds_train.dataloader[str(0)]) / worlds_train.n_worlds)

    # test
    model.eval()
    test_loss, test_feasible = 0, 0
    with torch.no_grad():
        if epoch % 10 == 0:
            check = np.random.randint(0, worlds_train.n_worlds)
            pairs_record = torch.cat((proc.preprocessing(worlds_train.dataset[str(check)].start_points),
                                      proc.preprocessing(worlds_train.dataset[str(check)].end_points)), 1)
            control_points, control_point_weights = model(worlds_train.images[check][None, ...], pairs_record)
            q_p = torch.cat((worlds_train.dataset[str(check)].start_points[:, None, :],
                             proc.postprocessing(control_points),
                             worlds_train.dataset[str(check)].end_points[:, None, :]), 1)

            nurbs = NURBS(p=q_p, degree=degree, w=control_point_weights, u=u)
            q = nurbs.evaluate()
            q_full = torch.cat((worlds_train.dataset[str(check)].start_points[:, None, :], q,
                                worlds_train.dataset[str(check)].end_points[:, None, :]), 1)

            name = 'record_train_epoch_' + str(epoch)
            plot_paths(q_full, worlds_train.pars[str(check)], 10, name, plot_path, save=save_image)
            # plt.show()

    print("epoch ", epoch)
    print("train loss: ", train_loss_history[-1], ",  feasible rate: ", train_feasible_history[-1])

plt.show()

plt.figure(1)
plt.plot(train_feasible_history, label='training')
plt.plot(test_feasible_history, label='test')
plt.title('feasible rate')
plt.legend()
plt.axis([0, None, 0, 1])
plt.savefig(plot_path + 'feasible')
plt.show()

plt.figure(2)
plt.plot(train_loss_history, label='training')
plt.plot(test_loss_history, label='test')
plt.title('loss')
plt.legend()
plt.savefig(plot_path + 'loss')
plt.show()
