import os
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from NURBS import NURBS
from chompy.Kinematic.Robots import SingleSphere02
from points_dataset import StartEndPointsDataset
from helper import Processing
from loss_function import chompy_partial_loss
from image_network import Backbone2D
from visualization import plot_paths
from chompy.Optimizer.obstacle_collision import oc_check2
from chompy.parameter import Parameter, initialize_oc
from world import world

# ======================== Initialization and Configuration ======================
np.random.seed(30)
device = torch.device("cpu")

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
n_control_points = 5  # Todo 3
degree = 3
Dof = par.robot.n_dof

world_number = 7
test_number = 2
images = world(number=world_number, par=par, n_voxels=n_voxels)

save_image = True
plot_path1 = './plot/train1/'
os.makedirs(plot_path1, exist_ok=True)
plot_path2 = './plot/test1/'
os.makedirs(plot_path2, exist_ok=True)

# =============================== Dataset ===========================
proc = Processing(world_limits)
start_end_number_train = 6000
start_end_number_val = 2000
start_end_number_test = 2000
start_end_number_record = 5

train_batch_size = 50
val_batch_size = 50
test_batch_size = 50

# ========================== Neural Network ================================
model = Backbone2D(2 * Dof, n_control_points, Dof)
model.to(device)

# ================================ Optimizer ================================
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# =============================== Training ====================================
weight = np.array([1, 2])
early_stop = 10


training_data = StartEndPointsDataset(start_end_number_train - 5, par, proc)
record_data_train = StartEndPointsDataset(5, par, proc)
training_data.add_data(record_data_train)
train_dataloader = DataLoader(training_data, batch_size=train_batch_size, shuffle=True)
val_data = StartEndPointsDataset(start_end_number_val, par, proc)
val_dataloader = DataLoader(val_data, batch_size=val_batch_size, shuffle=False)

train_loss_history = []
train_feasible_history = []
val_loss_history = []
val_feasible_history = []

repeat = 0
min_val_loss = np.inf

for epoch in range(21):  # Todo  101
    # training
    model.train()
    train_loss, train_feasible = 0, 0
    for _, (start_points_train, end_points_train, pairs_train) in enumerate(train_dataloader):
        # torch.Size([10, 1, 64, 64])
        control_points, control_point_weights = model(pairs_train, images.index_select(0, torch.tensor(range(world_number - test_number))))
        nurbs = NURBS(p=control_points, degree=degree, w=control_point_weights)
        q = nurbs.evaluate()   # torch.Size([50, 15, 2])
        q_pp = proc.postprocessing(q)
        q_full = torch.cat((start_points_train[:, None, :], q_pp, end_points_train[:, None, :]), 1)  # [50, 17, 2]

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

    # validation
    model.eval()
    val_loss, val_feasible = 0, 0
    with torch.no_grad():
        for _, (start_points_val, end_points_val, pairs_val) in enumerate(val_dataloader):
            control_points, control_point_weights = model(pairs_val, images.index_select(0, torch.tensor(range(world_number - test_number))))
            nurbs = NURBS(p=control_points, degree=degree, w=control_point_weights)
            q = nurbs.evaluate()
            q_pp = proc.postprocessing(q)
            q_full = torch.cat((start_points_val[:, None, :], q_pp, end_points_val[:, None, :]), 1)

            length_cost, collision_cost, length_jac, collision_jac = chompy_partial_loss(q_full.detach().numpy(), par)
            val_loss += (weight[0] * length_cost.sum() + weight[1] * collision_cost).mean()
            val_feasible += oc_check2(q_full.detach().numpy(), par.robot, par.oc, verbose=0).mean()

        val_loss /= len(val_dataloader)
        val_feasible /= len(val_dataloader)
        val_loss_history.append(val_loss)
        val_feasible_history.append(val_feasible)

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            repeat = 0
            best_model_weights = copy.deepcopy(model.state_dict())
        else:
            repeat += 1
            print("repeat", repeat)
            if repeat >= early_stop:
                print("epoch: ", epoch, "early stop.")
                break

    print("epoch ", epoch)
    print("train loss: ", train_loss_history[-1], ",  feasible rate: ", train_feasible_history[-1])
    print("val loss: ", val_loss_history[-1], ",  feasible rate: ", val_feasible_history[-1])

print('FINISH.')
torch.save(best_model_weights, "model")
np.save("train_feasible", train_feasible_history)
np.save("val_feasible", val_feasible_history)

model.load_state_dict(torch.load("model"))
model.eval()
test_loss, test_feasible = 0, 0

for i in range(test_number):
    initialize_oc(oc=par.oc, world=par.world, robot=par.robot, obstacle_img=np.squeeze(images[world_number - test_number + i].numpy()))

    test_data = StartEndPointsDataset(start_end_number_test, par, proc)
    test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

    test_loss_history = []
    test_feasible_history = []

    print("test_world ", i)

    with torch.no_grad():
        for _, (start_points_test, end_points_test, pairs_test) in enumerate(test_dataloader):
            control_points, control_point_weights = model(pairs_test, images.index_select(0, torch.tensor([world_number - test_number + i])))
            nurbs = NURBS(p=control_points, degree=degree, w=control_point_weights)
            q = nurbs.evaluate()
            q_pp = proc.postprocessing(q)
            q_full = torch.cat((start_points_test[:, None, :], q_pp, end_points_test[:, None, :]), 1)

            length_cost, collision_cost, length_jac, collision_jac = chompy_partial_loss(q_full.detach().numpy(), par)
            test_loss += (weight[0] * length_cost + weight[1] * collision_cost).mean()
            test_feasible += oc_check2(q_full.detach().numpy(), par.robot, par.oc, verbose=0).mean()

        test_loss /= len(test_dataloader)
        test_feasible /= len(test_dataloader)
        test_loss_history.append(test_loss)
        test_feasible_history.append(test_feasible)

    print("test loss: ", test_loss_history[-1], ",  feasible rate: ", test_feasible_history[-1])

    name = 'test' + str(i)
    plot_paths(q_full, par, 5, name, plot_path2, save=save_image)
    plt.show()


