import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from Network.worlds import Worlds
from chompy.Kinematic.Robots import SingleSphere02
from helper import Processing
from loss_function import chompy_partial_loss
from network import PlanFromImage
from visualization import plot_paths
from chompy.Optimizer.obstacle_collision import oc_check2
from chompy.parameter import Parameter

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

n_waypoints = 10
Dof = par.robot.n_dof

save_image = True
plot_path = './plot/images/test/'
os.makedirs(plot_path, exist_ok=True)

# ============================ Worlds and Points =============================
n_obstacles = 5
min_max_obstacle_size_voxel = [3, 15]
n_voxels = (64, 64)

n_worlds_train = 200
start_end_number_train = 2000   # in every world
worlds_batch_train = 10
points_batch_train = 20
worlds_train = Worlds(n_worlds_train, n_obstacles, min_max_obstacle_size_voxel, n_voxels, par)
worlds_train.create_points_loader(start_end_number_train, points_batch_train, shuffle=True)
worlds_loader_train = DataLoader(worlds_train.dist_images, batch_size=worlds_batch_train, shuffle=True)

n_worlds_test = 10
start_end_number_test = 100
worlds_batch_test = 10
points_batch_test = 20
worlds_test = Worlds(n_worlds_test, n_obstacles, min_max_obstacle_size_voxel, n_voxels, par)
worlds_test.create_points_loader(start_end_number_test, points_batch_test, shuffle=False)
worlds_loader_test = DataLoader(worlds_test.dist_images, batch_size=worlds_batch_test, shuffle=False)

# ========================== Neural Network ================================
# encoder = torch.load("encoder")
model = PlanFromImage(n_voxels, 2 * Dof, n_waypoints, Dof)
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
weight = np.array([1, 5])
repeat = 0
min_test_loss = np.inf

early_stop = 10
train_loss_history = []
train_feasible_history = []
test_loss_history = []
test_feasible_history = []

print("Use both length jac and collision jac...")
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
                start_points_train = torch.cat((start_points_train, start_points), 0)
                end_points_train = torch.cat((end_points_train, end_points), 0)
                pairs = torch.cat((proc.preprocessing(start_points), proc.preprocessing(end_points)), 1)
                pairs_train = torch.cat((pairs_train, pairs), 0)

            q = model(imgs, pairs_train).reshape(points_batch_train * worlds_batch_train, n_waypoints, Dof)
            # Path representation 1 : global coordinates in configuration space
            q_full = torch.cat((start_points_train[:, None, :],
                                proc.postprocessing(q),  # predict waypoints
                                end_points_train[:, None, :]), 1)

            # Path representation 2: relative coordinates to the connecting straight line
            # straight_line_points = torch.moveaxis(
            #     (torch.from_numpy(np.linspace(proc.preprocessing(start_points_train),
            #                                   proc.preprocessing(end_points_train), n_waypoints + 2))), 0, 1)[:, 1:-1, :]
            # # random walk
            # if epoch <= 10:
            #     walk = (torch.rand(q.shape) - 0.5) * 2
            #     q = q + walk
            # q_full = torch.cat((start_points_train[:, None, :],
            #                     proc.postprocessing(straight_line_points + q),
            #                     end_points_train[:, None, :]), 1)

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

        if epoch % 10 == 0:
            check = 0   # np.random.randint(0, worlds_train.n_worlds)
            pairs_record = torch.cat((proc.preprocessing(worlds_train.points_dataset[str(check)].start_points),
                                      proc.preprocessing(worlds_train.points_dataset[str(check)].end_points)), 1)
            q = model(worlds_train.dist_images[check][0][None, ...], pairs_record).reshape(start_end_number_train, n_waypoints, Dof)
            q_full = torch.cat((worlds_train.points_dataset[str(check)].start_points[:, None, :],
                                proc.postprocessing(q),
                                worlds_train.points_dataset[str(check)].end_points[:, None, :]), 1)
            # straight_line_points = torch.moveaxis(
            #     (torch.from_numpy(np.linspace(proc.preprocessing(worlds_train.dataset[str(check)].start_points),
            #                                   proc.preprocessing(worlds_train.dataset[str(check)].end_points), n_waypoints + 2))), 0, 1)[:, 1:-1, :]
            # q_full = torch.cat((worlds_train.dataset[str(check)].start_points[:, None, :],
            #                     proc.postprocessing(straight_line_points + q),
            #                     worlds_train.dataset[str(check)].end_points[:, None, :]), 1)

            name = 'record_train_epoch_' + str(epoch)
            plot_paths(q_full, worlds_train.pars[str(check)], 10, name, plot_path, save=save_image)
            plt.show()

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
                    pairs = torch.cat((proc.preprocessing(start_points), proc.preprocessing(end_points)), 1)
                    pairs_test = torch.cat((pairs_test, pairs), 0)

                q = model(imgs, pairs_test).reshape(points_batch_test * worlds_batch_test, n_waypoints, Dof)
                q_full = torch.cat((start_points_test[:, None, :],
                                 proc.postprocessing(q),
                                 end_points_test[:, None, :]), 1)
                # straight_line_points = torch.moveaxis(
                #     (torch.from_numpy(np.linspace(proc.preprocessing(start_points_test),
                #                                   proc.preprocessing(end_points_test), n_waypoints + 2))), 0, 1)[:, 1:-1, :]
                # q_full = torch.cat((start_points_test[:, None, :],
                #                     proc.postprocessing(straight_line_points + q),
                #                     end_points_test[:, None, :]), 1)

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

        if epoch % 20 == 0:
            check = 0
            name = 'test_epoch_' + str(epoch)
            plot_paths(q_full[check * points_batch_test:(check + 1) * points_batch_test],
                       worlds_test.pars[str(idx[check])], 10, name, plot_path, save=save_image)
            plt.show()

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
