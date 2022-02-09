import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from chompy.GridWorld import create_rectangle_image
from chompy.Kinematic.Robots import SingleSphere02, StaticArm
from points_dataset import BasePointsDataset
from helper import joints_preprocessing, joints_postprocessing
from loss_function import chompy_partial_loss
from network import PlanFromState, PlanFromArmStateImage
from visualization import plot_paths, plot_spheres
from chompy.Optimizer.obstacle_collision import oc_check2
from chompy.parameter import Parameter, initialize_oc
from chompy import plotting
from chompy.GridWorld.obstacle_distance import obstacle_img2dist_img
from Network.worlds import Worlds

# ======================== Initialization and Configuration ======================
np.random.seed(4)  # para!
device = torch.device("cpu")  # "cuda:0" if torch.cuda.is_available() else "cpu"

radius = 0.3  # Size of the robot [m]
robot = StaticArm(n_dof=3, limb_lengths=1.2, radius=0.1)  # n_dof should be 3 or 4
par = Parameter(robot=robot, obstacle_img='rectangle')

# par.world.limits = np.array([[-5, 5],
                         # [-5, 5]])
par.oc.n_substeps = 5
par.oc.n_substeps_check = 5

n_waypoints = 9
Dof = par.robot.n_dof   # 3

save_image = True
plot_path = './plot/arm_image/test/'
os.makedirs(plot_path, exist_ok=True)

# ============================ Worlds =============================
n_obstacles = 5
min_max_obstacle_size_voxel = [3, 14]  # 4,5,[3,14],[20,1000,4,50],[2,400,1,20]
n_voxels = (64, 64)                    # 4,5,[3,14],[20,1000,4,50],[2,400,1,20]
# =============================== Points Dataset ===========================

n_worlds_train = 20  # 100
start_end_number_train = 1000   # in every world
worlds_batch_train = 4  # 10
points_batch_train = 50
worlds_train = Worlds(n_worlds_train, n_obstacles, min_max_obstacle_size_voxel, n_voxels, par)
worlds_train.create_points_loader(start_end_number_train, points_batch_train, shuffle=True)  # don't need collision_rate
worlds_loader_train = DataLoader(worlds_train.dist_images, batch_size=worlds_batch_train, shuffle=True)

n_worlds_test = 2  # 2
start_end_number_test = 400
worlds_batch_test = 1  # 5
points_batch_test = 20
worlds_test = Worlds(n_worlds_test, n_obstacles, min_max_obstacle_size_voxel, n_voxels, par)
worlds_test.create_points_loader(start_end_number_test, points_batch_test, shuffle=False)
worlds_loader_test = DataLoader(worlds_test.dist_images, batch_size=worlds_batch_test, shuffle=False)

# ========================== Neural Network ================================
# waypoint representation
model = PlanFromArmStateImage(2 * Dof, n_waypoints, Dof)
model.to(device)
# ================================ Optimizer ================================
optimizer = torch.optim.SGD(model.parameters(), lr=0.004, momentum=0.9)
# =============================== Training ====================================
weight = np.array([1, 1])
repeat = 0
min_test_loss = np.inf

early_stop = 20
train_loss_history = []
train_feasible_history = []
test_loss_history = []
test_feasible_history = []

for epoch in range(501):

    # training
    model.train()
    train_loss, train_feasible = 0, 0

    for _ in range(int(start_end_number_train / points_batch_train)):
        for _, (imgs, idx) in enumerate(worlds_loader_train):

            idx = np.array(idx)
            start_points_train = torch.tensor([])
            end_points_train = torch.tensor([])
            pairs_train = torch.tensor([])
            for id in idx:
                start_points, end_points = next(iter(worlds_train.points_loader[str(id)]))
                # print("start_points", start_points.shape)

                start_points_train = torch.cat((start_points_train, start_points), 0)
                end_points_train = torch.cat((end_points_train, end_points), 0)
                pairs = torch.cat((start_points, end_points), 1)
                pairs_train = torch.cat((pairs_train, pairs), 0)

            # print("start_points_train", start_points_train.shape)
            # print("imgs", imgs.shape)
            # imgs, pairs_train = imgs.cuda(), pairs_train.cuda()

            q = model(imgs, pairs_train).reshape(points_batch_train * worlds_batch_train, n_waypoints, Dof)
            # Path representation 1 : global coordinates in configuration space
            q_full = torch.cat((joints_postprocessing(start_points_train[:, None, :]),
                                joints_postprocessing(q),  # predict waypoints
                                joints_postprocessing(end_points_train[:, None, :])), 1)

            length_cost, collision_cost = torch.tensor([]), torch.tensor([])
            length_jac, collision_jac = torch.tensor([]), torch.tensor([])
            for i in range(worlds_batch_train):
                lc, cc, lj, cj = chompy_partial_loss(
                    q_full[i * points_batch_train:(i + 1) * points_batch_train].detach().numpy(),
                    worlds_train.pars[str(idx[i])])
                length_cost = np.concatenate((length_cost, lc), 0)
                collision_cost = np.concatenate((collision_cost, cc), 0)
                length_jac = torch.cat((length_jac, lj), 0)
                collision_jac = torch.cat((collision_jac, cj), 0)

                train_feasible += oc_check2(q_full[i * points_batch_train:(i + 1) * points_batch_train].detach().numpy(),
                                            worlds_train.pars[str(idx[i])].robot,
                                            worlds_train.pars[str(idx[i])].oc, verbose=0).mean()

            train_loss += (weight[0] * length_cost + weight[1] * collision_cost).mean()

            temp = ((weight[0] * torch.flatten(length_cost.mean() * length_jac / np.sqrt(length_cost[:, None, None]))
                     + weight[1] * torch.flatten(collision_jac)) * torch.flatten(q)).mean()
            optimizer.zero_grad()
            temp.backward()
            optimizer.step()

    train_loss_history.append(train_loss / (start_end_number_train / points_batch_train)
                              / (n_worlds_train / worlds_batch_train))
    train_feasible_history.append(train_feasible / int(start_end_number_train / points_batch_train) / n_worlds_train)

    # test
    model.eval()
    test_loss, test_feasible = 0, 0
    with torch.no_grad():

        if epoch % 5 == 0:
            check = 1   # np.random.randint(0, worlds_train.n_worlds)
            pairs_record = torch.cat((worlds_train.points_dataset[str(check)].start_points,
                                      worlds_train.points_dataset[str(check)].end_points), 1)
            q = model(worlds_train.dist_images[check][0][None, ...], pairs_record).reshape(start_end_number_train, n_waypoints, Dof)
            q_full = torch.cat((joints_postprocessing(worlds_train.points_dataset[str(check)].start_points[:, None, :]),
                                joints_postprocessing(q),
                                joints_postprocessing(worlds_train.points_dataset[str(check)].end_points[:, None, :])), 1)

            name = 'record_train_epoch_' + str(epoch)
            plot_spheres(q_full, worlds_train.pars[str(check)], 10, name, robot, plot_path, save=save_image)
            # plt.show()

        for _ in range(int(start_end_number_test / points_batch_test)):
            for _, (imgs, idx) in enumerate(worlds_loader_test):
                idx = np.array(idx)
                start_points_test = torch.tensor([])
                end_points_test = torch.tensor([])
                pairs_test = torch.tensor([])
                for i in idx:
                    start_points, end_points = next(iter(worlds_test.points_loader[str(i)]))
                    start_points_test = torch.cat((start_points_test, start_points), 0)
                    end_points_test = torch.cat((end_points_test, end_points), 0)
                    pairs = torch.cat((start_points, end_points), 1)
                    pairs_test = torch.cat((pairs_test, pairs), 0)

                q = model(imgs, pairs_test).reshape(points_batch_test * worlds_batch_test, n_waypoints, Dof)
                q_full = torch.cat((joints_postprocessing(start_points_test[:, None, :]),
                                   joints_postprocessing(q),
                                   joints_postprocessing(end_points_test[:, None, :])), 1)

                length_cost, collision_cost = torch.tensor([]), torch.tensor([])
                length_jac, collision_jac = torch.tensor([]), torch.tensor([])
                for i in range(worlds_batch_test):
                    lc, cc, lj, cj = chompy_partial_loss(
                        q_full[i * points_batch_test:(i + 1) * points_batch_test].detach().numpy(),
                        worlds_test.pars[str(idx[i])])
                    length_cost = np.concatenate((length_cost, lc), 0)
                    collision_cost = np.concatenate((collision_cost, cc), 0)
                    length_jac = torch.cat((length_jac, lj), 0)
                    collision_jac = torch.cat((collision_jac, cj), 0)

                    test_feasible += oc_check2(q_full[i * points_batch_test:(i + 1) * points_batch_test].detach().numpy(),
                                               worlds_test.pars[str(idx[i])].robot,
                                               worlds_test.pars[str(idx[i])].oc, verbose=0).mean()

                test_loss += (weight[0] * length_cost + weight[1] * collision_cost).mean()

        if epoch % 5 == 0:
            check = 0  # 1
            name = 'test_epoch_' + str(epoch)
            plot_spheres(q_full[check * points_batch_test:(check + 1) * points_batch_test],
                         worlds_test.pars[str(idx[check])], 10, name, robot, plot_path, save=save_image)
            # plt.show()

        if test_loss < min_test_loss:
            min_test_loss = test_loss
            repeat = 0
        elif epoch > 200:
            repeat += 1
            if repeat >= early_stop:
                print("epoch: ", epoch, "early stop.")
                break

        test_loss_history.append(test_loss / int(start_end_number_test / points_batch_test)
                                 / (n_worlds_test / worlds_batch_test))
        test_feasible_history.append(test_feasible / int(start_end_number_test / points_batch_test) / n_worlds_test)

    print("epoch ", epoch)
    print("train loss: ", train_loss_history[-1], ",  feasible rate: ", train_feasible_history[-1])
    print("test loss: ", test_loss_history[-1], ",  feasible rate: ", test_feasible_history[-1])

# =========================== Save the results ========================
print('FINISH.')
# torch.save(model.state_dict(), "model_")

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

