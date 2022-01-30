import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

import plotting
from GridWorld.random_obstacles import create_rectangle_image, create_perlin_image
from Kinematic.Robots import StaticArm, SingleSphere02
from parameter import Parameter, initialize_oc
from Optimizer import length as o_len
from Optimizer import obstacle_collision as o_oc
from Kinematic import forward


robot = SingleSphere02(radius=0.3)
par = Parameter(robot=robot, obstacle_img='perlin')

# Sample random configurations
n_paths = 100
q = robot.sample_q((n_paths, 20))  # 3 random paths with 20 waypoints each (n_paths, 20, 2)

# Block World
n_voxels = (64, 64)
n_obstacles = 7
min_max_obstacle_size_voxel = [5, 20]
img = create_rectangle_image(n=n_obstacles, size_limits=min_max_obstacle_size_voxel,
                             n_voxels=n_voxels)

'''
res = 4
threshold = 0.5
img = create_perlin_image(n_voxels=n_voxels, res=res, threshold=threshold)
initialize_oc(oc=par.oc, world=par.world, robot=par.robot, obstacle_img=img)
# plt.imshow(img)
# plt.show()
'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True
# torch.cuda.empty_cache()

'''
batch_size = 64

train_set = TensorDataset(x_train, y_train)
val_set = TensorDataset(x_val, y_val)

dataloaders = {'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True),
               'val': DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True)}

# dataloaders['test'] = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, pin_memory=True)
'''


def perform_weighting(weighting,
                        oc=0, length=0):
    return (weighting.collision * oc +
            weighting.length * length)


class UPRLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_path, par, return_separate=False):

        # par = torch.from_numpy(par).float()
        # par = Variable(par, requires_grad=True)
        q_path = q_path.cpu()

        len_cost = o_len.q_cost(q=q_path, infinity_joints=par.robot.infinity_joints,
                                joint_weighting=par.weighting.joint_motion)
        # len_cost = torch.from_numpy(len_cost).float()

        x_spheres = forward.get_x_spheres_substeps(q=q_path, n=par.oc.n_substeps, robot=par.robot)
        # x_spheres = x_spheres.cpu().numpy()
        oc_cost = o_oc.oc_cost_w_length(x_spheres=x_spheres, oc=par.oc)
        # oc_cost = torch.from_numpy(oc_cost).float()

        q_path = torch.from_numpy(q).float()     # Todo
        q_path = q_path.cuda()

        if return_separate:
            return len_cost, oc_cost
        else:
            return perform_weighting(weighting=par.weighting, oc=oc_cost, length=len_cost)


class HighwayMLP(nn.Module):

    def __init__(self,
                 input_size,
                 gate_bias=-2,
                 activation_function=F.relu,
                 gate_activation=F.softmax):

        super().__init__()

        self.activation_function = activation_function
        self.gate_activation = gate_activation

        self.normal_layer = nn.Linear(input_size, input_size)

        self.gate_layer = nn.Linear(input_size, input_size)
        self.gate_layer.bias.data.fill_(gate_bias)

    def forward(self, x):

        normal_layer_result = self.activation_function(self.normal_layer(x))
        gate_layer_result = self.gate_activation(self.gate_layer(x))

        multiplyed_gate_and_normal = torch.mul(normal_layer_result, gate_layer_result)
        multiplyed_gate_and_input = torch.mul((1 - gate_layer_result), x)

        return torch.add(multiplyed_gate_and_normal,
                         multiplyed_gate_and_input)


class UPR(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(64, 128),    # Todo
            nn.Linear(128, 128))
        self.highway_layers = nn.ModuleList([HighwayMLP(128, activation_function=F.relu)
                                             for _ in range(10)])
        self.fcn = nn.Sequential(
            nn.Linear(128, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256))
        self.fcn64 = nn.Linear(256, 64)    # Todo

    def forward(self, x):
        out = self.input(x)
        for current_layer in self.highway_layers:
            out = current_layer(out)
        out = self.fcn(out)
        # out = out.reshape(out.size(0), -1)    # Todo
        out = self.fcn64(out)
        return out


def initialize_weight(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
# torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')


learning_rate = 0.5
num_epochs = 2
# lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

model = UPR()
model.to(device)

criterion = UPRLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)


def train(q, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        i = 0
        losses = []
        q = torch.from_numpy(q).float()

        for q[i] in q:

            q_path = q[i].unsqueeze(0)
            q_path = q_path.cuda()
            q_path = Variable(q_path, requires_grad=True)

            # pred = model(q_path)
            loss = criterion(q_path, par)
            losses.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print('Epoch: [{}/{}], Step: [{}/{}], Loss: {}'.format(epoch + 1, num_epochs, i + 1, n_paths, loss.item()))


model.apply(initialize_weight)
train(q, num_epochs)

'''
# Plot multi starts and optimal solution
fig, ax = plotting.new_world_fig(limits=par.world.limits, title='Multi-Starts')
plotting.plot_img_patch_w_outlines(img=par.oc.img, limits=par.world.limits, ax=ax)
for q in q_ms:
    plotting.plot_x_path(x=q, r=par.robot.spheres_rad, ax=ax, marker='o', alpha=0.5)

plotting.plot_x_path(x=q_opt, r=par.robot.spheres_rad, ax=ax, marker='o', color='k')

plt.show()
'''



