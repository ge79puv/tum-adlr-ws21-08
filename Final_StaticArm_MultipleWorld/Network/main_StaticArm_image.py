import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from chompy.Kinematic.Robots import StaticArm
from helper import joints_postprocessing
from loss_function import chompy_partial_loss
from network import PlanFromArmStateImage
from visualization import plot_spheres
from chompy.Optimizer.obstacle_collision import oc_check2
from chompy.parameter import Parameter
from Network.worlds import Worlds

# ======================== Initialization and Configuration ======================
np.random.seed(4)
device = torch.device("cpu")

robot = StaticArm(n_dof=3, limb_lengths=1.2, radius=0.1)
par = Parameter(robot=robot, obstacle_img='rectangle')

par.oc.n_substeps = 5
par.oc.n_substeps_check = 5

n_waypoints = 6
Dof = par.robot.n_dof

save_image = True
plot_path = './plot/arm_image/test_global/'
os.makedirs(plot_path, exist_ok=True)

n_obstacles = 5
min_max_obstacle_size_voxel = [3, 14]
n_voxels = (64, 64)

# =============================== Points Dataset ===========================

n_worlds_train = 40
start_end_number_train = 3000
worlds_batch_train = 4
points_batch_train = 50
worlds_train = Worlds(n_worlds_train, n_obstacles, min_max_obstacle_size_voxel, n_voxels, par)
worlds_train.create_points_loader(start_end_number_train, points_batch_train, shuffle=True)
worlds_loader_train = DataLoader(worlds_train.dist_images, batch_size=worlds_batch_train, shuffle=True)

n_worlds_test = 8
start_end_number_test = 1000
worlds_batch_test = 2
points_batch_test = 50
worlds_test = Worlds(n_worlds_test, n_obstacles, min_max_obstacle_size_voxel, n_voxels, par)
worlds_test.create_points_loader(start_end_number_test, points_batch_test, shuffle=False)
worlds_loader_test = DataLoader(worlds_test.dist_images, batch_size=worlds_batch_test, shuffle=False)

# ========================== Model ================================
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

            idx = np.array(idx)  # world indexes in this world batch
            start_points_train = torch.tensor([])
            end_points_train = torch.tensor([])
            pairs_train = torch.tensor([])
            for id in idx:
                start_points, end_points = next(iter(worlds_train.points_loader[str(id)]))

                start_points_train = torch.cat((start_points_train, start_points), 0)
                end_points_train = torch.cat((end_points_train, end_points), 0)
                pairs = torch.cat((start_points, end_points), 1)
                pairs_train = torch.cat((pairs_train, pairs), 0)

            # imgs: torch.Size([4, 1, 64, 64])  pairs_train: torch.Size([200, 6])
            q = model(imgs, pairs_train).reshape(points_batch_train * worlds_batch_train, n_waypoints, Dof)

            q_full = torch.cat((joints_postprocessing(start_points_train[:, None, :]),
                                joints_postprocessing(q),
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

        if epoch % 10 == 0:
            check = 0
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
    plt.close('all')

# =========================== Save the results ========================
print('FINISH.')
torch.save(model.state_dict(), "model_arm_image")
np.savez("arm_loss", train_loss_history=train_loss_history, train_feasible_history=train_feasible_history,
         test_loss_history=test_loss_history, test_feasible_history=test_feasible_history)

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

