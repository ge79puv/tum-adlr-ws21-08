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
from network import PlanFromState, Dummy
from visualization import plot_paths
from chompy.Optimizer.obstacle_collision import oc_check2
from chompy.parameter import Parameter, initialize_oc

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

n_waypoints = 50
u = np.linspace(0.01, 0.99, n_waypoints)
n_control_points = 1
degree = 3
Dof = par.robot.n_dof

save_image = True
plot_path = './plot/nurbs/N1D3_C1/'
os.makedirs(plot_path, exist_ok=True)

# ============================ Worlds =============================
n_obstacles = 5
min_max_obstacle_size_voxel = [3, 15]
n_voxels = (64, 64)
img = create_rectangle_image(n=n_obstacles, size_limits=min_max_obstacle_size_voxel, n_voxels=n_voxels)
initialize_oc(oc=par.oc, world=par.world, robot=par.robot, obstacle_img=img)

# =============================== Points Dataset ===========================
start_end_number_train = 2000
start_end_number_test = 100
start_end_number_record = 10

train_batch_size = 50
test_batch_size = 50

training_data = StartEndPointsDataset(start_end_number_train - start_end_number_record, par)
record_data_train = StartEndPointsDataset(start_end_number_record, par)
training_data.add_data(record_data_train)
train_dataloader = DataLoader(training_data, batch_size=train_batch_size, shuffle=True)
test_data = StartEndPointsDataset(start_end_number_test, par)
test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

# ========================== Neural Network ================================
# Nurbs representation
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
weight = np.array([1, 1])
repeat = 0
min_test_loss = np.inf

early_stop = 20
train_loss_history = []
train_feasible_history = []
test_loss_history = []
test_feasible_history = []

print("Use both length jac and collision jac...")
for epoch in range(101):

    # training
    model.train()
    train_loss, train_feasible = 0, 0
    for _, (start_points_train, end_points_train) in enumerate(train_dataloader):
        pairs_train = torch.cat((proc.preprocessing(start_points_train), proc.preprocessing(end_points_train)),
                                1)  # (number, 2 * dof)
        control_points, control_point_weights = model(pairs_train)
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

        train_loss += (weight[0] * length_cost + weight[1] * collision_cost).mean()
        train_feasible += oc_check2(q_full.detach().numpy(), par.robot, par.oc, verbose=0).mean()

        temp = (do_dp[:, 1:-1, :] * control_points).mean()
        optimizer.zero_grad()
        temp.backward()
        optimizer.step()

    train_loss_history.append(train_loss / len(train_dataloader))
    train_feasible_history.append(train_feasible / len(train_dataloader))

    # test
    model.eval()
    test_loss, test_feasible = 0, 0
    with torch.no_grad():

        if epoch % 5 == 0:
            pairs_record = torch.cat((proc.preprocessing(record_data_train.start_points),
                                      proc.preprocessing(record_data_train.end_points)), 1)
            control_points, control_point_weights = model(pairs_record)
            q_p = torch.cat((record_data_train.start_points[:, None, :],
                             proc.postprocessing(control_points),  # + record_data_train.start_points[:, None, :],
                             record_data_train.end_points[:, None, :]), 1)

            nurbs = NURBS(p=q_p, degree=degree, w=control_point_weights, u=u)
            q = nurbs.evaluate()
            q_full = torch.cat(
                (record_data_train.start_points[:, None, :], q, record_data_train.end_points[:, None, :]), 1)

            name = 'record_train_epoch_' + str(epoch)
            plot_paths(q_full, par, 10, name, plot_path, save=save_image)
            plt.show()

        for _, (start_points_test, end_points_test) in enumerate(test_dataloader):
            pairs_test = torch.cat((proc.preprocessing(start_points_test), proc.preprocessing(end_points_test)),
                                   1)  # (number, 2 * dof)
            control_points, control_point_weights = model(pairs_test)
            q_p = torch.cat((start_points_test[:, None, :],
                             proc.postprocessing(control_points),  # + start_points_test[:, None, :],
                             end_points_test[:, None, :]), 1)

            nurbs = NURBS(p=q_p, degree=degree, w=control_point_weights, u=u)
            q = nurbs.evaluate()
            q_full = torch.cat((start_points_test[:, None, :], q, end_points_test[:, None, :]), 1)
            length_cost, collision_cost, length_jac, collision_jac = chompy_partial_loss(q_full.detach().numpy(), par)

            test_loss += (weight[0] * length_cost + weight[1] * collision_cost).mean()
            test_feasible += oc_check2(q_full.detach().numpy(), par.robot, par.oc, verbose=0).mean()

        if epoch % 5 == 0:
            name = 'test_epoch_' + str(epoch)
            plot_paths(q_full, par, 10, name, plot_path, save=save_image)
            plt.show()

        if test_loss < min_test_loss:
            min_test_loss = test_loss
            repeat = 0
        else:
            repeat += 1
            if repeat >= early_stop:
                print("epoch: ", epoch, "early stop.")
                break

        test_loss_history.append(test_loss / len(test_dataloader))
        test_feasible_history.append(test_feasible / len(test_dataloader))

    print("epoch ", epoch)
    print("train loss: ", train_loss_history[-1], ",  feasible rate: ", train_feasible_history[-1])
    print("test loss: ", test_loss_history[-1], ",  feasible rate: ", test_feasible_history[-1])

# =========================== Save the results ========================
print('FINISH.')
torch.save(model.state_dict(), "model")
np.save("test_feasible_nurbs_N1D3_C1.npy", test_feasible_history)

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
