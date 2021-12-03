import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from GridWorld import create_rectangle_image
from Kinematic.Robots import SingleSphere02
from Network.points_dataset import StartEndPointsDataset
from Network.helper import Processing
from Network.loss_function import chompy_partial_loss
from Network.network import Backbone2D, Dummy
from Network.visualization import plot_paths
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
par.oc.n_substeps = 5
par.oc.n_substeps_check = 5
n_waypoints = 15
Dof = par.robot.n_dof

n_obstacles = 0
min_max_obstacle_size_voxel = [3, 15]

img = create_rectangle_image(n=n_obstacles, size_limits=min_max_obstacle_size_voxel, n_voxels=n_voxels)
initialize_oc(oc=par.oc, world=par.world, robot=par.robot, obstacle_img=img)

# =============================== Dataset ===========================
proc = Processing(world_limits)
start_end_number_train = 500
start_end_number_test = 100
train_batch_size = 50
test_batch_size = 50
training_data = StartEndPointsDataset(start_end_number_train, Dof, img, world_limits, proc)
train_dataloader = DataLoader(training_data, batch_size=train_batch_size, shuffle=True)
test_data = StartEndPointsDataset(start_end_number_train, Dof, img, world_limits, proc)
test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=True)

# ========================== Neural Network ================================
# model = Dummy(2 * Dof, (n_waypoints - 2) * Dof)
model = Backbone2D(2 * Dof, (n_waypoints - 2) * Dof)
model.to(device)
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# ================================ Optimizer ================================
# TODO: choose optimizer and corresponding hyperparameters
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)  # TODO: adaptive learning rate
# optimizer = torch.optim.Adam(model.parameters())
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

# =============================== Training ====================================
# first use only length jac, after change_epoch use both length and collision jac with later_weight
weight = 0
change_epoch = 101
later_weight = 2
change, repeat = False, 0
min_test_loss = np.inf

early_change, early_stop = 5, 100
train_loss_history = []
train_feasible_history = []
test_loss_history = []
test_feasible_history = []

print("Use only length jac")
for epoch in range(5001):

    if epoch % 100 == 0:
        print("epoch: ", epoch)
    if epoch == change_epoch:
        print("epoch: ", epoch, "use both length and collision jac...")
        weight = later_weight

    # training
    model.train()
    train_loss, train_feasible = 0, 0
    for batch_idx, (start_points, end_points, pairs) in enumerate(train_dataloader):
        q = model(pairs)
        q_reshaped = q.reshape(train_batch_size, n_waypoints - 2, Dof)
        q_pp = proc.postprocessing(q_reshaped)
        q_full = torch.cat((start_points[:, None, :], q_pp, end_points[:, None, :]), 1)

        length_cost, collision_cost, length_jac, collision_jac = chompy_partial_loss(q_full.detach().numpy(), par)
        temp = (weight * torch.flatten(q) * torch.flatten(collision_jac)
                + torch.flatten(q) * torch.flatten(length_jac)).sum() / train_batch_size
        optimizer.zero_grad()
        temp.backward()
        optimizer.step()

        loss = (length_cost.sum() + later_weight * collision_cost.sum()) / train_batch_size
        train_loss += loss
        feasible = oc_check2(q_full.detach().numpy(), par.robot, par.oc).sum() / train_batch_size
        train_feasible += feasible

        if epoch % 100 == 0:
            print(f"loss: {loss:>7f}  [{(batch_idx + 1) * train_batch_size:>5d}/{start_end_number_train:>5d}]")

    if epoch % 100 == 0:
        plot_paths(q_full, par, number=50)
        plt.show()

    train_loss_history.append(train_loss / len(train_dataloader))
    train_feasible_history.append(train_feasible / len(train_dataloader))

    # test
    # model.eval() # TODO
    # test_loss, test_feasible = 0, 0
    # with torch.no_grad():
    #     for _, (start_points, end_points, pairs) in enumerate(train_dataloader):
    #         q = model(pairs)
    #         q_reshaped = q.reshape(train_batch_size, n_waypoints - 2, Dof)
    #         q_pp = proc.postprocessing(q_reshaped)
    #         q_full = torch.cat((start_points[:, None, :], q_pp, end_points[:, None, :]), 1)
    #
    #         length_cost, collision_cost, length_jac, collision_jac = chompy_partial_loss(q_full.detach().numpy(), par)
    #         test_loss += (length_cost.sum() + later_weight * collision_cost.sum()) / test_batch_size
    #         test_feasible += oc_check2(q_full.detach().numpy(), par.robot, par.oc).sum() / test_batch_size
    #
    #     if epoch % 20 == 0:
    #         plot_paths(q_full, par, number=10)
    #         plt.show()
    #
    #     test_loss /= len(test_dataloader)
    #     test_feasible /= len(test_dataloader)
    #
    #     if test_loss < min_test_loss:
    #         min_test_loss = test_loss
    #         repeat = 0
    #     else:
    #         repeat += 1
    #         if repeat >= early_change and epoch >= change_epoch and change is False:
    #             print("epoch: ", epoch, "use both length and collision jac...")
    #             weight = later_weight
    #             min_test_loss = np.inf
    #             repeat = 0
    #             change = True
    #         elif repeat >= early_stop and change is True:
    #             print("epoch: ", epoch, "early stop.")
    #             break
    #
    # test_loss_history.append(test_loss)
    # test_feasible_history.append(test_feasible)
    #
    if epoch % 100 == 0:
        print("train loss mean: ", train_loss_history[-1], ",  feasible rate: ", train_feasible_history[-1])
    #     print("test loss: ", test_loss_history[-1], ",  feasible rate: ", test_feasible_history[-1])


print('FINISH.')
torch.save(model.state_dict(), "./model")

# =========================== validation =================================
#
# model = Backbone2D(2 * Dof, (n_waypoints - 2) * Dof)
# model.load_state_dict(torch.load("./model"))
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
