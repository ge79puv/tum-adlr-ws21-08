import os
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from chompy.GridWorld import create_rectangle_image
from chompy.Kinematic.Robots import StaticArm
from points_dataset import BasePointsDataset
from helper import joints_postprocessing
from loss_function import chompy_partial_loss
from network import PlanFromArmState
from visualization import plot_spheres
from chompy.Optimizer.obstacle_collision import oc_check2
from chompy.parameter import Parameter, initialize_oc

# ======================== Initialization and Configuration ======================
np.random.seed(18)
device = torch.device("cpu")

robot = StaticArm(n_dof=3, limb_lengths=1.2, radius=0.1)
par = Parameter(robot=robot, obstacle_img='rectangle')

par.oc.n_substeps = 5
par.oc.n_substeps_check = 5

n_waypoints = 6
Dof = par.robot.n_dof

save_image = True
plot_path = './plot/arm/test_relative/'
os.makedirs(plot_path, exist_ok=True)

# ============================ Worlds =============================
n_obstacles = 5
min_max_obstacle_size_voxel = [3, 14]
n_voxels = (64, 64)
img = create_rectangle_image(n=n_obstacles, size_limits=min_max_obstacle_size_voxel, n_voxels=n_voxels)
initialize_oc(oc=par.oc, world=par.world, robot=par.robot, obstacle_img=img)

# =============================== Points Dataset ===========================
start_end_number_train = 3000
start_end_number_test = 1000
start_end_number_record = 10

train_batch_size = 50
test_batch_size = 50

training_data = BasePointsDataset(start_end_number_train - start_end_number_record, par)
record_data_train = BasePointsDataset(start_end_number_record, par)
training_data.add_data(record_data_train)
train_dataloader = DataLoader(training_data, batch_size=train_batch_size, shuffle=True)
test_data = BasePointsDataset(start_end_number_test, par)
test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

# ========================== Model ================================
model = PlanFromArmState(2 * Dof, n_waypoints, Dof)
model.to(device)

# ================================ Optimizer ================================
optimizer = torch.optim.SGD(model.parameters(), lr=0.004, momentum=0.9)

# =============================== Training ====================================
weight = np.array([1, 1])
repeat = 0
min_test_loss = np.inf

early_stop = 20
train_loss_history = []
train_length_loss_history = []
train_feasible_history = []
test_loss_history = []
test_length_loss_history = []
test_feasible_history = []

for epoch in range(501):

    # training
    model.train()
    train_loss, train_length_loss, train_feasible = 0, 0, 0

    for _, (start_points_train, end_points_train) in enumerate(train_dataloader):

        uniform_waypoints = None
        delta = (end_points_train - start_points_train)/n_waypoints

        for i in range(n_waypoints):

            if i == 0:
                uniform_waypoints = start_points_train + delta*1
                uniform_waypoints = torch.unsqueeze(uniform_waypoints, 1)
            if i > 0:
                shift = torch.unsqueeze(start_points_train + delta*(i+1), 1)
                uniform_waypoints = torch.cat((uniform_waypoints, shift), 1)

        pairs_train = torch.cat((start_points_train, end_points_train), 1)
        q = model(pairs_train).reshape(train_batch_size, n_waypoints, Dof)

        q_full = torch.cat((joints_postprocessing(start_points_train[:, None, :]),
                            joints_postprocessing(uniform_waypoints + q),
                            joints_postprocessing(end_points_train[:, None, :])), 1)

        length_cost, collision_cost, length_jac, collision_jac = chompy_partial_loss(q_full.detach().numpy(), par)

        train_loss += (weight[0] * length_cost + weight[1] * collision_cost).mean()
        train_length_loss += (weight[0] * length_cost).mean()
        train_feasible += oc_check2(q_full.detach().numpy(), par.robot, par.oc, verbose=0).mean()

        temp = ((weight[0] * torch.flatten(length_cost.mean() * length_jac / np.sqrt(length_cost[:, None, None]))
                 + weight[1] * torch.flatten(collision_jac)) * torch.flatten(q)).mean()

        optimizer.zero_grad()
        temp.backward()
        optimizer.step()

    train_loss_history.append(train_loss / len(train_dataloader))
    train_length_loss_history.append(train_length_loss / len(train_dataloader))
    train_feasible_history.append(train_feasible / len(train_dataloader))

    # test
    model.eval()
    test_loss, test_length_loss, test_feasible = 0, 0, 0
    with torch.no_grad():

        pairs_record = torch.cat((record_data_train.start_points, record_data_train.end_points), 1)
        q = model(pairs_record).reshape(start_end_number_record, n_waypoints, Dof)
        q_full = torch.cat((joints_postprocessing(record_data_train.start_points[:, None, :]),
                            joints_postprocessing(q),
                            joints_postprocessing(record_data_train.end_points[:, None, :])), 1)

        if epoch % 10 == 0:
            name = 'record_train_epoch_' + str(epoch)
            plot_spheres(q_full, par, 10, name, robot, plot_path, save=save_image)
            # plt.show()

        for _, (start_points_test, end_points_test) in enumerate(test_dataloader):
            uniform_waypoints = None
            delta = (end_points_test - start_points_test) / n_waypoints

            for i in range(n_waypoints):

                if i == 0:
                    uniform_waypoints = start_points_test + delta * 1
                    uniform_waypoints = torch.unsqueeze(uniform_waypoints, 1)
                if i > 0:
                    shift = torch.unsqueeze(start_points_test + delta * (i + 1), 1)
                    uniform_waypoints = torch.cat((uniform_waypoints, shift), 1)

            pairs_test = torch.cat((start_points_test, end_points_test), 1)
            q = model(pairs_test).reshape(train_batch_size, n_waypoints, Dof)

            q_full = torch.cat((joints_postprocessing(start_points_test[:, None, :]),
                                joints_postprocessing(uniform_waypoints + q),
                                joints_postprocessing(end_points_test[:, None, :])), 1)

            length_cost, collision_cost, length_jac, collision_jac = chompy_partial_loss(q_full.detach().numpy(),
                                                                                         par)
            test_loss += (weight[0] * length_cost + weight[1] * collision_cost).mean()
            test_length_loss += (weight[0] * length_cost).mean()
            test_feasible += oc_check2(q_full.detach().numpy(), par.robot, par.oc, verbose=0).mean()

        if epoch % 10 == 0:
            name = 'test_epoch_' + str(epoch)
            plot_spheres(q_full, par, 10, name, robot, plot_path, save=save_image)
            # plt.show()

        if test_loss < min_test_loss:
            min_test_loss = test_loss
            repeat = 0
            best_model_weights = copy.deepcopy(model.state_dict())

        elif epoch > 200:
            repeat += 1
            if repeat >= early_stop:
                print("epoch: ", epoch, "early stop.")
                break

        test_loss_history.append(test_loss / len(test_dataloader))
        test_length_loss_history.append(test_length_loss / len(test_dataloader))
        test_feasible_history.append(test_feasible / len(test_dataloader))

    print("epoch ", epoch)
    print("train loss: ", train_loss_history[-1], "train length loss: ", train_length_loss_history[-1],
          ",  feasible rate: ", train_feasible_history[-1])
    print("test loss: ", test_loss_history[-1], "test length loss: ", test_length_loss_history[-1],
          ",  feasible rate: ", test_feasible_history[-1])
    plt.close('all')

# =========================== Save the results ========================
print('FINISH.')
torch.save(best_model_weights, "model_arm_relative")
np.savez("arm_loss_relative.npz", train_loss_history=train_loss_history, train_length_loss_history=train_length_loss_history,
         train_feasible_history=train_feasible_history, test_loss_history=test_loss_history,
         test_length_loss_history=test_length_loss_history, test_feasible_history=test_feasible_history)

plt.figure(1)
plt.plot(train_feasible_history, label='training feasibility')
plt.plot(test_feasible_history, label='test feasibility')
plt.title('feasible rate')
plt.axis([0, None, 0, 1])
plt.legend()
plt.savefig(plot_path + 'feasible')
plt.show()

plt.figure(2)
plt.plot(train_loss_history, label='training loss')
plt.plot(test_loss_history, label='test loss')
plt.title('loss')
plt.legend()
plt.savefig(plot_path + 'loss')
plt.show()

