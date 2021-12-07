import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from GridWorld import create_rectangle_image
from Kinematic.Robots import SingleSphere02
from Network.points_dataset import StartEndPointsDataset
from Network.points_dataset import sample_points
from Network.helper import Processing
from Network.loss_function import chompy_partial_loss
from Network.network import Backbone2D, Dummy
from Network.visualization import plot_paths
from Optimizer.obstacle_collision import oc_check2
from parameter import Parameter, initialize_oc

np.random.seed(2)
device = torch.device("cpu")  # "cuda:0" if torch.cuda.is_available() else
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

n_obstacles = 5
min_max_obstacle_size_voxel = [3, 15]

img = create_rectangle_image(n=n_obstacles, size_limits=min_max_obstacle_size_voxel, n_voxels=n_voxels)
initialize_oc(oc=par.oc, world=par.world, robot=par.robot, obstacle_img=img)

# =============================== Dataset ===========================
proc = Processing(world_limits)
start_end_number_train = 5000  # 500 5000
start_end_number_test = 1000  # 100 1000
train_batch_size = 100  # 50 100
test_batch_size = 100  # 50 100
training_data = StartEndPointsDataset(start_end_number_train, Dof, img, world_limits, proc)
train_dataloader = DataLoader(training_data, batch_size=train_batch_size, shuffle=True)
test_data = StartEndPointsDataset(start_end_number_train, Dof, img, world_limits, proc)
test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

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
weight = 0  # 0
change_epoch = 10  # 100 10
later_weight = 2  # 2
change, repeat = False, 0
min_test_loss = np.inf

early_change, early_stop = 5, 100
train_loss_history = []
train_feasible_history = []
test_loss_history = []
test_feasible_history = []

print("Use only length jac")
for epoch in range(100):  # 5000 100

    if (epoch + 1) % 10 == 0:
        print("epoch: ", epoch + 1)
    if epoch == change_epoch:
        print("epoch: ", epoch, "use both length and collision jac...")
        weight = later_weight

    # training
    model.train()
    train_loss, train_feasible = 0, 0
    for batch_idx, (start_points, end_points, q_se, q_es, pairs) in enumerate(train_dataloader):
        # pairs = pairs.cuda()
        q1 = model(q_se)
        q_reshaped1 = q1.reshape(train_batch_size, n_waypoints - 2, Dof)
        q_pp1 = proc.postprocessing(q_reshaped1)
        q_full1 = torch.cat((start_points[:, None, :], q_pp1, end_points[:, None, :]), 1)
        # .detach().numpy()
        length_cost1, collision_cost1, length_jac1, collision_jac1 = chompy_partial_loss(q_full1.detach().numpy(), par)

        temp1 = (weight * torch.flatten(q1) * torch.flatten(collision_jac1)
                 + torch.flatten(q1) * torch.flatten(length_jac1)).sum() / train_batch_size
        optimizer.zero_grad()
        temp1.backward()
        optimizer.step()

        q2 = model(q_es)
        q_reshaped2 = q2.reshape(train_batch_size, n_waypoints - 2, Dof)
        q_pp2 = proc.postprocessing(q_reshaped2)
        q_full2 = torch.cat((end_points[:, None, :], q_pp2, start_points[:, None, :]), 1)
        # .detach().numpy()
        length_cost2, collision_cost2, length_jac2, collision_jac2 = chompy_partial_loss(q_full2.detach().numpy(), par)
        '''
        temp = ( ((weight * torch.flatten(q1) * torch.flatten(collision_jac1)
                + torch.flatten(q1) * torch.flatten(length_jac1)).sum() / train_batch_size) +
                ((weight * torch.flatten(q2) * torch.flatten(collision_jac2)
                    + torch.flatten(q2) * torch.flatten(length_jac2)).sum() / train_batch_size) )
        '''
        temp2 = (weight * torch.flatten(q2) * torch.flatten(collision_jac2)
                 + torch.flatten(q2) * torch.flatten(length_jac2)).sum() / train_batch_size
        optimizer.zero_grad()
        temp2.backward()
        optimizer.step()

        loss = (((length_cost1.sum() + later_weight * collision_cost1.sum()) / train_batch_size) +
                ((length_cost2.sum() + later_weight * collision_cost2.sum()) / train_batch_size)) / 2
        train_loss += loss
        feasible = ((oc_check2(q_full1.detach().numpy(), par.robot, par.oc).sum() / train_batch_size) +
                    (oc_check2(q_full2.detach().numpy(), par.robot, par.oc).sum() / train_batch_size)) / 2
        train_feasible += feasible

        if (epoch + 1) % 10 == 0:
            print(f"loss: {loss:>7f}  [{(batch_idx + 1) * train_batch_size:>5d}/{start_end_number_train:>5d}]")

    # q_full = torch.cat((q_full1, q_full2), 1)
    #if (epoch + 1) % 10 == 0:
        #plot_paths(q_full2, par, number=10)
        #plt.show()

    train_loss_history.append(train_loss / len(train_dataloader))
    train_feasible_history.append(train_feasible / len(train_dataloader))
    if (epoch + 1) % 10 == 0:
        print("train loss mean: ", train_loss_history[-1], ",  feasible rate: ", train_feasible_history[-1])
        # print("test loss: ", test_loss_history[-1], ",  feasible rate: ", test_feasible_history[-1])

print('Train Finish.')
torch.save(model.state_dict(), "./model")

# =========================== test =================================
model.load_state_dict(torch.load("./model"))

start_points_test = sample_points(start_end_number_test, Dof, img, world_limits)
end_points_test = sample_points(start_end_number_test, Dof, img, world_limits)
start_end_points_test = torch.cat((start_points_test, end_points_test), 1)
q_full_test = torch.cat((start_points_test[:, None, :],
                         proc.postprocessing(model(start_end_points_test).reshape(start_end_number_test, n_waypoints - 2, Dof)),
                         end_points_test[:, None, :]),
                        1)
length_cost, collision_cost, length_jac, collision_jac = chompy_partial_loss(q_full_test.detach().numpy(), par)
print("test loss: ", length_cost.sum() + collision_cost.sum(),
      "feasible rate: ", oc_check2(q_full_test.detach().numpy(), par.robot, par.oc).sum() / start_end_number_test)

plot_paths(q_full_test, par, number=10)
plt.show()


'''
# =========================== test =================================
model.load_state_dict(torch.load("./model"))
model.eval()
test_loss, test_feasible = 0, 0

with torch.no_grad():
    for batch_idx, (start_points, end_points, q_se, q_es, pairs) in enumerate(test_dataloader):
        q = model(pairs)
        q_reshaped = q.reshape(train_batch_size, n_waypoints - 2, Dof)
        q_pp = proc.postprocessing(q_reshaped)
        q_full = torch.cat((start_points[:, None, :], q_pp, end_points[:, None, :]), 1)

        length_cost, collision_cost, length_jac, collision_jac = chompy_partial_loss(q_full.detach().numpy(), par)
        test_loss += (length_cost.sum() + later_weight * collision_cost.sum()) / test_batch_size
        test_feasible += oc_check2(q_full.detach().numpy(), par.robot, par.oc).sum() / test_batch_size

    plot_paths(q_full, par, number=10)
    plt.show()

    test_loss /= len(test_dataloader)
    test_feasible /= len(test_dataloader)

    print("test loss mean is {0}, test feasible mean is {1}".format(test_loss, test_feasible))

'''

'''
# test
model.eval()  # TODO
test_loss, test_feasible = 0, 0
with torch.no_grad():
    for batch_idx, (start_points, end_points, q_se, q_es, pairs) in enumerate(test_dataloader):
        q = model(pairs)
        q_reshaped = q.reshape(train_batch_size, n_waypoints - 2, Dof)
        q_pp = proc.postprocessing(q_reshaped)
        q_full = torch.cat((start_points[:, None, :], q_pp, end_points[:, None, :]), 1)

        length_cost, collision_cost, length_jac, collision_jac = chompy_partial_loss(q_full.detach().numpy(), par)
        test_loss += (length_cost.sum() + later_weight * collision_cost.sum()) / test_batch_size
        test_feasible += oc_check2(q_full.detach().numpy(), par.robot, par.oc).sum() / test_batch_size
        # if (epoch + 1) % 10 == 0:
            # print(f"loss: {loss:>7f}  [{(batch_idx + 1) * train_batch_size:>5d}/{start_end_number_train:>5d}]")

    plot_paths(q_full, par, number=10)
    plt.show()

    test_loss /= len(test_dataloader)
    test_feasible /= len(test_dataloader)

    # test_loss_history.append(test_loss)
    # test_feasible_history.append(test_feasible)

    if test_loss < min_test_loss:
        min_test_loss = test_loss
        repeat = 0
    else:
        repeat += 1
        if repeat >= early_change and epoch >= change_epoch and change is False:
            print("epoch: ", epoch, "use both length and collision jac...")
            weight = later_weight
            min_test_loss = np.inf
            repeat = 0
            change = True
        elif repeat >= early_stop and change is True:
            print("epoch: ", epoch, "early stop.")
            break

'''