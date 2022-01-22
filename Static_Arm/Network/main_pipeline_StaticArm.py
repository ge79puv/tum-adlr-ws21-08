import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from chompy.GridWorld import create_rectangle_image
from chompy.Kinematic.Robots import SingleSphere02, StaticArm
from points_dataset import StartEndPointsDataset, BasePointsDataset
from helper import Processing
from loss_function import chompy_partial_loss
from network import PlanFromState, PlanFromArmState
from visualization import plot_paths, plot_spheres
from chompy.Optimizer.obstacle_collision import oc_check2
from chompy.parameter import Parameter, initialize_oc
from chompy import plotting

# ======================== Initialization and Configuration ======================
np.random.seed(18)
device = torch.device("cpu")
print(device)

radius = 0.3  # Size of the robot [m]
robot = StaticArm(n_dof=3, limb_lengths=1.2, radius=0.1)  # n_dof should be 3 or 4
par = Parameter(robot=robot, obstacle_img='rectangle')    # perlin

par.world.limits = np.array([[-5, 5],
                         [-5, 5]])
par.oc.n_substeps = 5
par.oc.n_substeps_check = 5

n_waypoints = 6
Dof = par.robot.n_dof   # 3

save_image = True
plot_path = './plot/arm/test/'
os.makedirs(plot_path, exist_ok=True)

# ============================ Worlds =============================
n_obstacles = 5
min_max_obstacle_size_voxel = [3, 14]
n_voxels = (64, 64)
img = create_rectangle_image(n=n_obstacles, size_limits=min_max_obstacle_size_voxel, n_voxels=n_voxels)
initialize_oc(oc=par.oc, world=par.world, robot=par.robot, obstacle_img=img)

# =============================== Points Dataset ===========================
start_end_number_train = 1200
start_end_number_test = 400
start_end_number_record = 10

train_batch_size = 50
test_batch_size = 50

training_data = BasePointsDataset(start_end_number_train - start_end_number_record, par)
record_data_train = BasePointsDataset(start_end_number_record, par)
training_data.add_data(record_data_train)
train_dataloader = DataLoader(training_data, batch_size=train_batch_size, shuffle=True)
test_data = BasePointsDataset(start_end_number_test, par)
test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

# ========================== Neural Network ================================
# waypoint representation
model = PlanFromArmState(2 * Dof, n_waypoints, Dof)
model.to(device)

# ================================ Optimizer ================================
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# =============================== Training ====================================
weight = np.array([1, 3])
repeat = 0
min_test_loss = np.inf

early_stop = 20
train_loss_history = []
train_feasible_history = []
test_loss_history = []
test_feasible_history = []

img = torch.from_numpy(img[np.newaxis, np.newaxis, :]).float()

for epoch in range(201):
    # if epoch == 50:
        # break
    # training
    model.train()
    train_loss, train_feasible = 0, 0

    for _, (start_points_train, end_points_train) in enumerate(train_dataloader):
        pairs_train = torch.cat((start_points_train, end_points_train), 1)  # (number, 2 * dof) torch.Size([50, 6])

        q = model(img, pairs_train).reshape(train_batch_size, n_waypoints, Dof)  # torch.Size([50, 6, 3])
        q_full = torch.cat((start_points_train[:, None, :], q, end_points_train[:, None, :]), 1)  # ([50, 8, 3])

        length_cost, collision_cost, length_jac, collision_jac = chompy_partial_loss(q_full.detach().numpy(), par)
        # collision_cost (50,)  collision_jac torch.Size([50, 6, 3]) torch.flatten(q) torch.Size([900])
        # Todo: change: weight, limb_length, map, network, image_merge, feasible_check

        train_loss += (weight[0] * length_cost + weight[1] * collision_cost).mean()
        train_feasible += oc_check2(q_full.detach().numpy(), par.robot, par.oc, verbose=0).mean()

        temp = ((weight[0] * torch.flatten(length_cost.mean() * length_jac / np.sqrt(length_cost[:, None, None]))
                 + weight[1] * torch.flatten(collision_jac)) * torch.flatten(q)).mean()
        optimizer.zero_grad()
        temp.backward()
        optimizer.step()

    train_loss_history.append(train_loss / len(train_dataloader))
    train_feasible_history.append(train_feasible / len(train_dataloader))

    # if epoch % 5 == 0:
        # name = 'record_train_epoch_' + str(epoch)
        # plot_spheres(q_full, par, 5, name, robot, plot_path, save=save_image)
        # plt.show()

    # test
    model.eval()
    test_loss, test_feasible = 0, 0
    with torch.no_grad():
        pairs_record = torch.cat((record_data_train.start_points, record_data_train.end_points), 1)
        q = model(img, pairs_record).reshape(start_end_number_record, n_waypoints, Dof)
        q_full = torch.cat((record_data_train.start_points[:, None, :],
                            q, record_data_train.end_points[:, None, :]), 1)

        if epoch % 5 == 0:
            name = 'record_train_epoch_' + str(epoch)
            plot_spheres(q_full, par, 2, name, robot, plot_path, save=save_image)
            # plt.show()
            # print(q)

        for _, (start_points_test, end_points_test) in enumerate(test_dataloader):
            pairs_test = torch.cat((start_points_test, end_points_test), 1)  # (number, 2 * dof)
            q = model(img, pairs_test).reshape(train_batch_size, n_waypoints, Dof)
            q_full = torch.cat((start_points_test[:, None, :], q, end_points_test[:, None, :]), 1)

            length_cost, collision_cost, length_jac, collision_jac = chompy_partial_loss(q_full.detach().numpy(),
                                                                                         par)

            test_loss += (weight[0] * length_cost + weight[1] * collision_cost).mean()
            test_feasible += oc_check2(q_full.detach().numpy(), par.robot, par.oc, verbose=0).mean()
        '''
        if epoch % 5 == 0: 
            name = 'test_epoch_' + str(epoch)
            plot_spheres(q_full, par, 5, name, robot, plot_path, save=save_image)
            # plt.show()
        '''

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

