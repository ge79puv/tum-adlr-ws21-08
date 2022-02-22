import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from Network.nurbs import NURBS
from chompy.GridWorld import create_rectangle_image
from chompy.Kinematic.Robots import SingleSphere02
from dataset_points import StartEndPointsDataset
from helper import Processing
from loss_function import chompy_partial_loss
from network import PlanFromState
from visualization import plot_paths, plot_world
from chompy.Optimizer.obstacle_collision import oc_check2
from chompy.parameter import Parameter, initialize_oc

# ======================== Initialization and Configuration ======================
np.random.seed(10)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

radius = 0.3  # Size of the robot [m]
robot = SingleSphere02(radius=radius)
par = Parameter(robot=robot, obstacle_img='rectangle')

world_limits = np.array([[0, 10],  # x [m]
                         [0, 10]])  # y [m]
proc = Processing(world_limits)
par.robot.limits = world_limits
par.oc.n_substeps = 5
par.oc.n_substeps_check = 5

plot_path = './plot/path_repr/world0/test/'
os.makedirs(plot_path, exist_ok=True)

"""
Three kinds of path representation: "global", "relative", "nurbs"
"global": global waypoints coordinates in configuration space
"relative": relative coordinates to the connecting straight line
"nurbs": spline using control points and control weights
Different path representation require different parameters which can be chosen in the following.
"""
path_representation = "global"
Dof = par.robot.n_dof

# ============================ Worlds =============================
n_obstacles = 5
min_max_obstacle_size_voxel = [3, 15]
n_voxels = (64, 64)
img = create_rectangle_image(n=n_obstacles, size_limits=min_max_obstacle_size_voxel, n_voxels=n_voxels)
initialize_oc(oc=par.oc, world=par.world, robot=par.robot, obstacle_img=img)
plot_world(par, 'world', plot_path)

# =============================== Points Dataset ===========================
train_start_end_number = 2000
test_start_end_number = 500
record_start_end_number = 100

train_batch_size = 100
test_batch_size = 100

training_data = StartEndPointsDataset(train_start_end_number, par)
# training_data.set_collision_rate(0.5)
training_data.get_collision_rate()
train_dataloader = DataLoader(training_data, batch_size=train_batch_size, shuffle=True)
record_idx = np.random.randint(0, train_start_end_number, record_start_end_number)
record_start_points = training_data.start_points[record_idx]
record_end_points = training_data.end_points[record_idx]
test_data = StartEndPointsDataset(test_start_end_number, par)
# test_data.set_collision_rate(0.5)
test_data.get_collision_rate()
test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

# ========================== Neural Network ================================
if path_representation == "global" or path_representation == "relative":
    n_waypoints = 10
    model = PlanFromState(2 * Dof, n_waypoints, Dof, ctrlpts=False)
elif path_representation == "nurbs":
    n_waypoints = 10
    u = np.linspace(0.1, 0.9, n_waypoints)
    # u = np.array([0.1, 0.25, 0.35, 0.4, 0.45, 0.5,
    #               0.55, 0.6, 0.65, 0.75, 0.9])
    n_control_points = 1
    degree = 3
    model = PlanFromState(2 * Dof, n_control_points, Dof, ctrlpts=True)

model.to(device)
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# ================================ Optimizer ================================
# TODO: choose optimizer and corresponding hyperparameters
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # TODO: adaptive learning rate
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

# =============================== Training ====================================
weight = np.array([1, 10])

min_test_loss = np.inf
repeat = 0
early_stop = 20
max_epoch = 500
min_epoch = 100
use_length_only_epoch = -1  # used to initialize a straight connecting line for "global"
random_walk_epoch = -1      # used to initialize with some noise for "relative"

train_loss_history = []
train_length_loss_history = []
train_feasible_history = []
test_loss_history = []
test_length_loss_history = []
test_feasible_history = []

for epoch in range(max_epoch):

    # training
    model.train()
    train_loss, train_length_loss, train_feasible = 0, 0, 0
    for b, (start_points_train, end_points_train) in enumerate(train_dataloader):
        pairs_train = torch.cat((proc.preprocessing(start_points_train), proc.preprocessing(end_points_train)),
                                1)  # (number, 2 * dof)

        if path_representation == "global" or path_representation == "relative":
            q = model(pairs_train.to(device)).reshape(train_batch_size, n_waypoints, Dof).to("cpu")

            if path_representation == "global":
                # x_spheres = forward.get_x_spheres_substeps(q=q, n=par.oc.n_substeps, robot=par.robot)
                q_full = torch.cat((start_points_train[:, None, :],
                                    proc.postprocessing(q),
                                    end_points_train[:, None, :]), 1)
            else:  # path_representation == "relative"
                straight_line_points = torch.moveaxis(
                    (torch.from_numpy(np.linspace(proc.preprocessing(start_points_train),
                                                  proc.preprocessing(end_points_train), n_waypoints + 2))),
                    0, 1)[:, 1:-1, :]
                # random walk initialization
                if epoch <= random_walk_epoch:
                    walk = (torch.rand(q.shape) - 0.5) * 2
                    q = q + walk
                q_full = torch.cat((start_points_train[:, None, :],
                                    proc.postprocessing(straight_line_points + q),
                                    end_points_train[:, None, :]), 1)

            length_cost, collision_cost, length_jac, collision_jac = chompy_partial_loss(q_full.detach().numpy(), par)
            # first use length jac
            if path_representation == "global" and epoch < use_length_only_epoch:
                temp = ((weight[0] * torch.flatten(length_jac)) * torch.flatten(q)).mean()
            else:
                temp = ((weight[0] * torch.flatten(length_jac)
                         + weight[1] * torch.flatten(collision_jac)) * torch.flatten(q)).mean()
                # temp = ((weight[0] * torch.flatten(length_jac * np.sqrt(length_cost).mean() / np.sqrt(length_cost[
                # :, None, None])) + weight[1] * torch.flatten(collision_jac)) * torch.flatten(q)).mean()

        elif path_representation == "nurbs":
            control_points, control_point_weights = model(pairs_train.to(device))
            control_points.to("cpu")
            control_point_weights.to("cpu")
            q_p = torch.cat((start_points_train[:, None, :],
                             proc.postprocessing(control_points),  # + start_points_train[:, None, :],
                             end_points_train[:, None, :]), 1)

            nurbs = NURBS(p=q_p, degree=degree, w=control_point_weights, u=u)
            q = nurbs.evaluate()
            q_full = torch.cat((start_points_train[:, None, :], q, end_points_train[:, None, :]), 1)
            length_cost, collision_cost, length_jac, collision_jac = chompy_partial_loss(q_full.detach().numpy(), par)

            do_dq = weight[0] * length_cost.mean() * length_jac / np.sqrt(length_cost[:, None, None]) + weight[
                1] * collision_jac
            dq_dp = nurbs.evaluate_jac()
            do_dp = torch.matmul(torch.moveaxis(dq_dp, 1, 2), do_dq.float())

            temp = (do_dp[:, 1:-1, :] * control_points).mean()

        optimizer.zero_grad()
        temp.backward()
        optimizer.step()

        train_loss += (weight[0] * length_cost + weight[1] * collision_cost).sum()
        train_length_loss += weight[0] * length_cost.sum()
        feasible = oc_check2(q_full.detach().numpy(), par.robot, par.oc, verbose=0)
        train_feasible += feasible.sum()

    train_loss_history.append(train_loss / train_start_end_number)
    train_length_loss_history.append(train_length_loss / train_start_end_number)
    train_feasible_history.append(train_feasible / train_start_end_number)

    model.eval()
    test_loss, test_length_loss, test_feasible = 0, 0, 0
    with torch.no_grad():
        # record train process
        if epoch % 10 == 0:
            pairs_record = torch.cat((proc.preprocessing(record_start_points),
                                      proc.preprocessing(record_end_points)), 1)
            if path_representation == "global" or path_representation == "relative":
                q = model(pairs_record.to(device)).reshape(record_start_end_number, n_waypoints, Dof).to("cpu")
                if path_representation == "global":
                    q_full = torch.cat((record_start_points[:, None, :],
                                        proc.postprocessing(q),
                                        record_end_points[:, None, :]), 1)
                else:  # path_representation == "relative"
                    straight_line_points = torch.moveaxis(
                        (torch.from_numpy(np.linspace(proc.preprocessing(record_start_points),
                                                      proc.preprocessing(record_end_points), n_waypoints + 2))),
                        0, 1)[:, 1:-1, :]
                    q_full = torch.cat((record_start_points[:, None, :],
                                        proc.postprocessing(straight_line_points + q),
                                        record_end_points[:, None, :]), 1)
            elif path_representation == "nurbs":
                control_points, control_point_weights = model(pairs_record.to(device))
                control_points.to("cpu")
                control_point_weights.to("cpu")
                q_p = torch.cat((record_start_points[:, None, :],
                                 proc.postprocessing(control_points),
                                 record_end_points[:, None, :]), 1)

                nurbs = NURBS(p=q_p, degree=degree, w=control_point_weights, u=u)
                q = nurbs.evaluate()
                q_full = torch.cat(
                    (record_start_points[:, None, :], q, record_end_points[:, None, :]), 1)

            name = 'train_epoch_' + str(epoch)
            plot_paths(q_full, par, 10, name, plot_path)
            plt.show()

        # evaluation
        for _, (start_points_test, end_points_test) in enumerate(test_dataloader):
            pairs_test = torch.cat((proc.preprocessing(start_points_test), proc.preprocessing(end_points_test)),
                                   1)  # (number, 2 * dof)
            if path_representation == "global" or path_representation == "relative":
                q = model(pairs_test.to(device)).reshape(test_batch_size, n_waypoints, Dof).to("cpu")
                if path_representation == "global":
                    q_full = torch.cat((start_points_test[:, None, :],
                                        proc.postprocessing(q),
                                        end_points_test[:, None, :]), 1)
                else:  # path_representation == "relative"
                    straight_line_points = torch.moveaxis(
                        (torch.from_numpy(np.linspace(proc.preprocessing(start_points_test),
                                                      proc.preprocessing(end_points_test), n_waypoints + 2))),
                        0, 1)[:, 1:-1, :]
                    q_full = torch.cat((start_points_test[:, None, :],
                                        proc.postprocessing(straight_line_points + q),
                                        end_points_test[:, None, :]), 1)

            elif path_representation == "nurbs":
                control_points, control_point_weights = model(pairs_test.to(device))
                control_points.to("cpu")
                control_point_weights.to("cpu")
                q_p = torch.cat((start_points_test[:, None, :],
                                 proc.postprocessing(control_points),
                                 end_points_test[:, None, :]), 1)

                nurbs = NURBS(p=q_p, degree=degree, w=control_point_weights, u=u)
                q = nurbs.evaluate()
                q_full = torch.cat((start_points_test[:, None, :], q, end_points_test[:, None, :]), 1)

            length_cost, collision_cost, length_jac, collision_jac = chompy_partial_loss(q_full.detach().numpy(), par)
            test_loss += (weight[0] * length_cost + weight[1] * collision_cost).sum()
            test_length_loss += weight[0] * length_cost.sum()
            test_feasible += oc_check2(q_full.detach().numpy(), par.robot, par.oc, verbose=0).sum()

        if epoch % 10 == 0:
            name = 'test_epoch_' + str(epoch)
            plot_paths(q_full, par, 10, name, plot_path)
            plt.show()

        if test_loss < min_test_loss:
            min_test_loss = test_loss
            repeat = 0
        elif epoch > min_epoch:
            repeat += 1
            if repeat >= early_stop:
                name = 'test_epoch_' + str(epoch) + '_collided'
                plot_paths(q_full, par, 7, name, plot_path, plot_collision=True)
                plt.show()
                print("epoch: ", epoch, "early stop.")
                break

        test_loss_history.append(test_loss / test_start_end_number)
        test_length_loss_history.append(test_length_loss / test_start_end_number)
        test_feasible_history.append(test_feasible / test_start_end_number)

    print("epoch ", epoch)
    print("train loss: ", train_loss_history[-1], ", length loss: ", train_length_loss_history[-1],
          ", feasible rate: ", train_feasible_history[-1])
    print("test loss: ", test_loss_history[-1], ", length loss: ", test_length_loss_history[-1],
          ",  feasible rate: ", test_feasible_history[-1])
    plt.close('all')

# =========================== Save the results ========================
print('FINISH.')
if plot_path:
    torch.save(model.state_dict(), plot_path + "model")
    np.savez(plot_path + "loss_history",
             train_loss_history=train_loss_history,
             train_length_loss_history=train_length_loss_history,
             train_feasible_history=train_feasible_history,
             test_loss_history=test_loss_history,
             test_length_loss_history=test_length_loss_history,
             test_feasible_history=test_feasible_history)

plt.figure(1)
plt.plot(train_loss_history[2:], label='training total')
plt.plot(test_loss_history[2:], label='test total')
plt.plot(train_length_loss_history[2:], label='training length')
plt.plot(test_length_loss_history[2:], label='test length')
plt.title('loss')
plt.legend()
plt.savefig(plot_path + 'loss')
plt.show()

plt.figure(2)
plt.plot(train_feasible_history, label='training')
plt.plot(test_feasible_history, label='test')
plt.title('feasible rate')
plt.legend()
plt.axis([0, None, 0, 1])
plt.savefig(plot_path + 'feasible')
plt.show()
