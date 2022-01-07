import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from NURBS import NURBS
from chompy.GridWorld import create_rectangle_image
from chompy.Kinematic.Robots import SingleSphere02
from points_dataset import StartEndPointsDataset
from helper import Processing
from loss_function import chompy_partial_loss
from network import Backbone2D, Dummy
from visualization import plot_paths
from chompy.Optimizer.obstacle_collision import oc_check2
from chompy.parameter import Parameter, initialize_oc


# ======================== Initialization and Configuration ======================
np.random.seed(10)
device = torch.device("cpu")
print(device)

radius = 0.3  # Size of the robot [m]
robot = SingleSphere02(radius=radius)

world_limits = np.array([[0, 10],  # x [m]
                         [0, 10]])  # y [m]
n_voxels = (64, 64)

par = Parameter(robot=robot, obstacle_img='rectangle')
par.robot.limits = world_limits
par.oc.n_substeps = 5
par.oc.n_substeps_check = 5
n_waypoints = 15    # TODO
n_control_points = 5
degree = 3
Dof = par.robot.n_dof


n_obstacles = 5
min_max_obstacle_size_voxel = [3, 15]

img = create_rectangle_image(n=n_obstacles, size_limits=min_max_obstacle_size_voxel, n_voxels=n_voxels)
initialize_oc(oc=par.oc, world=par.world, robot=par.robot, obstacle_img=img)

save_image = True
plot_path = './plot/test/'
os.makedirs(plot_path, exist_ok=True)

# =============================== Dataset ===========================
proc = Processing(world_limits)
start_end_number_train = 2000
start_end_number_test = 100
start_end_number_record = 10

train_batch_size = 50
test_batch_size = 50

training_data = StartEndPointsDataset(start_end_number_train - 5, par, proc)
record_data_train = StartEndPointsDataset(5, par, proc)
training_data.add_data(record_data_train)
train_dataloader = DataLoader(training_data, batch_size=train_batch_size, shuffle=True)
test_data = StartEndPointsDataset(start_end_number_test, par, proc)
test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

# ========================== Neural Network ================================
# model = Dummy(2 * Dof, n_control_points * Dof)
model = Backbone2D(2 * Dof, n_control_points, Dof)
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
weight = np.array([1, 2])
repeat = 0
min_test_loss = np.inf

early_stop = 10
train_loss_history = []
train_feasible_history = []
test_loss_history = []
test_feasible_history = []

print("Use both length jac and collision jac...")
for epoch in range(101):

    # training
    model.train()
    train_loss, train_feasible = 0, 0
    for _, (start_points_train, end_points_train, pairs_train) in enumerate(train_dataloader):
        control_points, control_point_weights = model(pairs_train)

        nurbs = NURBS(p=control_points, degree=degree, w=control_point_weights)
        q = nurbs.evaluate()
        # torch.Size([50, 15, 2])
        q_pp = proc.postprocessing(q)
        q_full = torch.cat((start_points_train[:, None, :], q_pp, end_points_train[:, None, :]), 1)
        # torch.Size([50, 17, 2])

        length_cost, collision_cost, length_jac, collision_jac = chompy_partial_loss(q_full.detach().numpy(), par)
        do_dq = weight[0] * length_jac + weight[1] * collision_jac
        dq_dp = nurbs.evaluate_jac()
        do_dp = torch.matmul(torch.moveaxis(dq_dp, 1, 2), do_dq.float())

        train_loss += (weight[0] * length_cost + weight[1] * collision_cost).mean()
        train_feasible += oc_check2(q_full.detach().numpy(), par.robot, par.oc, verbose=0).mean()

        temp = (do_dp * control_points).mean()
        optimizer.zero_grad()
        temp.backward()
        optimizer.step()

    train_loss_history.append(train_loss / len(train_dataloader))
    train_feasible_history.append(train_feasible / len(train_dataloader))

    # test
    model.eval()
    test_loss, test_feasible = 0, 0
    with torch.no_grad():
        control_points, control_point_weights = model(record_data_train.pairs)
        nurbs = NURBS(p=control_points, degree=degree, w=control_point_weights)
        q = nurbs.evaluate()
        q_pp = proc.postprocessing(q)
        q_full = torch.cat((record_data_train.start_points[:, None, :], q_pp, record_data_train.end_points[:, None, :]), 1)
        if epoch % 5 == 0:
            name = 'record_train_epoch_' + str(epoch)
            plot_paths(q_full, par, 5, name, plot_path, save=save_image)
            plt.show()
    #     for _, (start_points_test, end_points_test, pairs_test) in enumerate(test_dataloader):
    #         q_test = model(pairs_test)
    #         q_reshaped_test = q_test.reshape(test_batch_size, n_waypoints - 2, Dof)
    #         q_pp_test = proc.postprocessing(q_reshaped_test)
    #         q_full_test = torch.cat((start_points_test[:, None, :], q_pp_test, end_points_test[:, None, :]), 1)
    #
    #         length_cost, collision_cost, length_jac, collision_jac = chompy_partial_loss(q_full_test.detach().numpy(), par)
    #         test_loss += (later_weight[0] * length_cost.sum() + later_weight[1] * collision_cost.sum()) / test_batch_size
    #         test_feasible += oc_check2(q_full_test.detach().numpy(), par.robot, par.oc, verbose=0).sum() / test_batch_size
    #
    #     if epoch % 5 == 0:
    #         name = 'test_epoch_' + str(epoch)
    #         plot_paths(q_full_test, par, 10, name, plot_path, save=save_image)
    #         plt.show()
    #
    #     if test_loss < min_test_loss and change is True:
    #         min_test_loss = test_loss
    #         repeat = 0
    #     elif change is True:
    #         repeat += 1
    #         if repeat >= early_stop:
    #             print("epoch: ", epoch, "early stop.")
    #             break
    #
    #     test_loss_history.append(test_loss / len(test_dataloader))
    #     test_feasible_history.append(test_feasible / len(test_dataloader))

    print("epoch ", epoch)
    print("train loss: ", train_loss_history[-1], ",  feasible rate: ", train_feasible_history[-1])
    # print("test loss: ", test_loss_history[-1], ",  feasible rate: ", test_feasible_history[-1])

#
# # =========================== Save the results ========================
# print('FINISH.')
# torch.save(model.state_dict(), "model")
# np.save("test_feasible_L1C5", test_feasible_history)
#
plt.show()

plt.figure(1)
plt.plot(train_feasible_history, label='training')
plt.plot(test_feasible_history, label='test')
plt.title('feasible rate')
plt.legend()
plt.axis([0, None, 0, 1])
plt.savefig(plot_path + 'feasible')
plt.show()
#
# plt.figure(2)
# plt.plot(train_loss_history, label='training')
# plt.plot(test_loss_history, label='test')
# plt.title('loss')
# plt.legend()
# plt.savefig(plot_path + 'loss')
# plt.show()