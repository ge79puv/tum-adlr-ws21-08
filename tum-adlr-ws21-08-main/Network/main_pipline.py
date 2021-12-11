import os
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from chompy.GridWorld import create_rectangle_image
from chompy.Kinematic.Robots import SingleSphere02
from points_dataset import StartEndPointsDataset
from helper import Processing
from loss_function import chompy_partial_loss
from network import Backbone2D, Dummy
from visualization import plot_paths
from chompy.Optimizer.obstacle_collision import oc_check2
from chompy.parameter import Parameter, initialize_oc
from Network.points_dataset import sample_points


np.random.seed(10)
device = torch.device("cpu")            # "cuda:0" if torch.cuda.is_available() else
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

n_obstacles = 10
min_max_obstacle_size_voxel = [3, 15]

img = create_rectangle_image(n=n_obstacles, size_limits=min_max_obstacle_size_voxel, n_voxels=n_voxels)
initialize_oc(oc=par.oc, world=par.world, robot=par.robot, obstacle_img=img)

save_image = True
plot_path = './plot/o10/BothJacAfter30C5/'
os.makedirs(plot_path, exist_ok=True)

# =============================== Dataset ===========================
proc = Processing(world_limits)
start_end_number_train = 6000
start_end_number_val = 2000
start_end_number_test = 2000
start_end_number_record = 5

train_batch_size = 50
val_batch_size = 50
test_batch_size = 50

training_data = StartEndPointsDataset(start_end_number_train - 5, img, par, proc)
record_data_train = StartEndPointsDataset(5, img, par, proc)
training_data.add_data(record_data_train)
train_dataloader = DataLoader(training_data, batch_size=train_batch_size, shuffle=True)
val_data = StartEndPointsDataset(start_end_number_val, img, par, proc)        # correct
val_dataloader = DataLoader(val_data, batch_size=val_batch_size, shuffle=False)
test_data = StartEndPointsDataset(start_end_number_test, img, par, proc)        # correct
test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)
record_data = StartEndPointsDataset(start_end_number_record, img, par, proc)

# ========================== Neural Network ================================
# model = Dummy(2 * Dof, (n_waypoints - 2) * Dof)
model = Backbone2D(2 * Dof, (n_waypoints - 2) * Dof)
model.to(device)
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# ================================ Optimizer ================================
# TODO: choose optimizer and corresponding hyperparameters
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # TODO: adaptive learning rate
# optimizer = torch.optim.Adam(model.parameters())
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

# =============================== Training ====================================
# first use only length jac, after change_epoch use both length and collision jac with later_weight
weight = np.array([1, 0])
change_epoch = 30
later_weight = np.array([1, 5])
change, repeat = False, 0
min_val_loss = np.inf

early_stop = 10
train_loss_history = []
train_feasible_history = []
val_loss_history = []
val_feasible_history = []

print("Use only length jac...")

for epoch in range(301):    # 301

    if epoch == change_epoch:
        weight = later_weight
        change = True
        print("Use both length jac and collision jac...")
    # training
    model.train()
    train_loss, train_feasible = 0, 0
    for batch_idx, (start_points, end_points, pairs) in enumerate(train_dataloader):
        q = model(pairs)
        q_reshaped = q.reshape(train_batch_size, n_waypoints - 2, Dof)
        q_pp = proc.postprocessing(q_reshaped)
        q_full = torch.cat((start_points[:, None, :], q_pp, end_points[:, None, :]), 1)

        length_cost, collision_cost, length_jac, collision_jac = chompy_partial_loss(q_full.detach().numpy(), par)
        train_loss += (later_weight[0] * length_cost.sum() + later_weight[1] * collision_cost.sum()) / train_batch_size
        train_feasible += oc_check2(q_full.detach().numpy(), par.robot, par.oc, verbose=0).sum() / train_batch_size

        temp = (weight[0] * torch.flatten(q) * torch.flatten(length_jac)
                + weight[1] * torch.flatten(q) * torch.flatten(collision_jac)).sum() / train_batch_size
        optimizer.zero_grad()
        temp.backward()
        optimizer.step()

        # loss = (weight[0] * length_cost.sum() + weight[1] * collision_cost.sum()) / train_batch_size
        # train_loss += loss
        # feasible = oc_check2(q_full.detach().numpy(), par.robot, par.oc, verbose=0).sum() / train_batch_size
        # train_feasible += feasible
    '''
    if epoch % 10 == 0:
        name = 'train_epoch_' + str(epoch)
        plot_paths(q_full, par, 10, name, plot_path, save=save_image)
        plt.show()
    '''
    train_loss /= len(train_dataloader)
    train_feasible /= len(train_dataloader)
    train_loss_history.append(train_loss)
    train_feasible_history.append(train_feasible)

    # validation
    model.eval()
    val_loss, val_feasible = 0, 0
    with torch.no_grad():
        q = model(record_data_train.pairs)
        q_reshaped = q.reshape(-1, n_waypoints - 2, Dof)
        q_pp = proc.postprocessing(q_reshaped)
        q_full = torch.cat((record_data_train.start_points[:, None, :], q_pp, record_data_train.end_points[:, None, :]), 1)
        if epoch % 5 == 0:
            name = 'record_train_epoch_' + str(epoch)
            plot_paths(q_full, par, 5, name, plot_path, save=save_image)

        q = model(record_data.pairs)
        q_reshaped = q.reshape(-1, n_waypoints - 2, Dof)
        q_pp = proc.postprocessing(q_reshaped)
        q_full = torch.cat((record_data.start_points[:, None, :], q_pp, record_data.end_points[:, None, :]), 1)
        if epoch % 5 == 0:
            name = 'record_epoch_' + str(epoch)
            plot_paths(q_full, par, start_end_number_record, name, plot_path, save=save_image)

        for _, (start_points, end_points, pairs) in enumerate(val_dataloader):     # correct
            q = model(pairs)
            q_reshaped = q.reshape(val_batch_size, n_waypoints - 2, Dof)
            q_pp = proc.postprocessing(q_reshaped)
            q_full = torch.cat((start_points[:, None, :], q_pp, end_points[:, None, :]), 1)

            length_cost, collision_cost, length_jac, collision_jac = chompy_partial_loss(q_full.detach().numpy(), par)
            val_loss += (later_weight[0] * length_cost.sum() + later_weight[1] * collision_cost.sum()) / val_batch_size
            val_feasible += oc_check2(q_full.detach().numpy(), par.robot, par.oc, verbose=0).sum() / val_batch_size
        '''
        if epoch % 10 == 0:
            name = 'val_epoch_' + str(epoch)
            plot_paths(q_full, par, 10, name, plot_path, save=save_image)
            plt.show()
        '''
        val_loss /= len(val_dataloader)
        val_feasible /= len(val_dataloader)
        val_loss_history.append(val_loss)
        val_feasible_history.append(val_feasible)

        if val_loss < min_val_loss and change is True:
            min_val_loss = val_loss
            repeat = 0
            best_model_weights = copy.deepcopy(model.state_dict())
        else:
            repeat += 1
            if repeat >= early_stop and change is True:
                print("epoch: ", epoch, "early stop.")
                break

    if epoch % 1 == 0:
        print("epoch ", epoch)
        print("train loss mean: ", train_loss_history[-1], ",  feasible rate: ", train_feasible_history[-1])
        print("val loss mean: ", val_loss_history[-1], ",  feasible rate: ", val_feasible_history[-1])

print('FINISH.')
torch.save(best_model_weights, "model")

model.load_state_dict(torch.load("model"))
model.eval()
test_loss, test_feasible = 0, 0
with torch.no_grad():

    for _, (start_points, end_points, pairs) in enumerate(test_dataloader):
        q = model(pairs)
        q_reshaped = q.reshape(train_batch_size, n_waypoints - 2, Dof)
        q_pp = proc.postprocessing(q_reshaped)
        q_full = torch.cat((start_points[:, None, :], q_pp, end_points[:, None, :]), 1)

        length_cost, collision_cost, length_jac, collision_jac = chompy_partial_loss(q_full.detach().numpy(), par)
        test_loss += (later_weight[0] * length_cost.sum() + later_weight[1] * collision_cost.sum()) / test_batch_size
        test_feasible += oc_check2(q_full.detach().numpy(), par.robot, par.oc, verbose=0).sum() / test_batch_size
    '''
    if epoch % 10 == 0:
        name = 'val_epoch_' + str(epoch)
        plot_paths(q_full, par, 10, name, plot_path, save=save_image)
        plt.show()
    '''
    test_loss /= len(test_dataloader)
    test_feasible /= len(test_dataloader)

print("test loss mean: ", test_loss, ",  test feasible rate: ", test_feasible)

'''
plt.show()
plt.figure(1)
plt.plot(train_loss_history, label='training')
plt.plot(val_loss_history, label='val')
plt.title('loss')
plt.legend()
plt.savefig(plot_path + 'loss')
plt.show()

plt.figure(2)
plt.plot(train_feasible_history, label='training')
plt.plot(val_feasible_history, label='val')
plt.title('feasible rate')
plt.legend()
plt.savefig(plot_path + 'feasible')
plt.show()
'''



