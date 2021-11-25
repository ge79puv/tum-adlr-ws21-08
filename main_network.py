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
from Network.network import Backbone2D


robot = SingleSphere02(radius=0.3)
par = Parameter(robot=robot, obstacle_img=None)

# Sample random configurations
n_paths = 100
q_start = sample

# Block World
n_voxels = (64, 64)
n_obstacles = 7
min_max_obstacle_size_voxel = [5, 20]
img = create_rectangle_image(n=n_obstacles, size_limits=min_max_obstacle_size_voxel,
                             n_voxels=n_voxels)
plt.imshow(img)
plt.show()
'''
res = 4
threshold = 0.5
img = create_perlin_image(n_voxels=n_voxels, res=res, threshold=threshold)
initialize_oc(oc=par.oc, world=par.world, robot=par.robot, obstacle_img=img)
# plt.imshow(img)
# plt.show()
'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
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

model = Backbone2D()
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



